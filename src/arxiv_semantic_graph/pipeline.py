#!/usr/bin/env python3
# ================================================================
# ArXiv Pipeline: Embed (USE) -> HNSW Index -> Histogram -> Graph -> KMeans
# GPU-safe, streaming, headless plotting, clean logs.
# ================================================================
import os, json, csv, glob, random, argparse, warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Silence non-critical noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # TF: show only errors
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
import tensorflow_hub as hub
import hnswlib

from sklearn.cluster import MiniBatchKMeans


# -------------------- GPU / HEADLESS CONFIG ---------------------
def setup_environment(use_gpu: bool, vram_mib: int):
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    if use_gpu:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    gpus = tf.config.list_physical_devices("GPU")
    if use_gpu and gpus:
        try:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
            tf.config.set_logical_device_configuration(
                gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=vram_mib)]
            )
            print(f"[GPU] Using logical GPU capped at ~{vram_mib} MiB")
        except Exception as e:
            print(f"[GPU] Config failed, falling back to CPU: {e}")
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        print("[CPU] Running without GPU")


# -------------------- IO & STREAM HELPERS -----------------------
def iter_papers(path: Path, min_year: int):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            try:
                y = int(p.get("update_date", "0-0-0").split("-")[0])
            except Exception:
                continue
            if y >= min_year:
                t = (p.get("title") or "").strip()
                a = (p.get("abstract") or "").strip()
                if a:
                    yield t, y, a


def embed_batch(texts, use_layer, batch: int):
    out, cur = [], []
    for t in texts:
        cur.append(t)
        if len(cur) == batch:
            out.append(use_layer(cur).numpy()); cur = []
    if cur:
        out.append(use_layer(cur).numpy())
    return np.vstack(out)


def shard_stream(meta_json: Path, out_dir: Path, min_year: int, batch: int, shard: int, use_layer):
    titles, years, absbuf = [], [], []
    shard_id, total = 0, 0
    for title, year, abstract in iter_papers(meta_json, min_year):
        titles.append(title); years.append(year); absbuf.append(abstract)
        if len(absbuf) >= shard:
            emb = embed_batch(absbuf, use_layer, batch).astype(np.float16)
            np.save(out_dir / f"emb_{shard_id:05d}.npy", emb)
            with open(out_dir / f"meta_{shard_id:05d}.tsv", "w", encoding="utf-8") as g:
                for t, y in zip(titles, years):
                    g.write(f"{y}\t{t}\n")
            total += len(absbuf)
            print(f"[Shard {shard_id}] saved {len(absbuf)} (total {total})")
            shard_id += 1; titles.clear(); years.clear(); absbuf.clear()
    if absbuf:
        emb = embed_batch(absbuf, use_layer, batch).astype(np.float16)
        np.save(out_dir / f"emb_{shard_id:05d}.npy", emb)
        with open(out_dir / f"meta_{shard_id:05d}.tsv", "w", encoding="utf-8") as g:
            for t, y in zip(titles, years):
                g.write(f"{y}\t{t}\n")
        total += len(absbuf)
        print(f"[Shard {shard_id}] saved {len(absbuf)} (total {total})")
    return total


def load_shapes(emb_paths):
    shapes = [np.load(p, mmap_mode="r").shape for p in emb_paths]
    N = sum(s[0] for s in shapes)
    D = shapes[0][1]
    return N, D


def iter_vectors(emb_paths, chunk=20000):
    acc = 0
    for p in emb_paths:
        arr = np.load(p, mmap_mode="r")
        n = arr.shape[0]
        for i in range(0, n, chunk):
            j = min(i + chunk, n)
            yield acc + i, acc + j, np.asarray(arr[i:j], dtype="float32")
        acc += n


# -------------------- META LOOKUP -------------------------------
def id_to_meta(global_id: int, emb_paths, out_dir: Path):
    """Return (year, title) for a global id, robust to bad lines."""
    acc = 0
    for shard_id, f in enumerate(emb_paths):
        n = np.load(f, mmap_mode="r").shape[0]
        if global_id < acc + n:
            offset = global_id - acc
            meta = out_dir / f"meta_{shard_id:05d}.tsv"
            with open(meta, "r", encoding="utf-8") as g:
                for i, line in enumerate(g):
                    if i == offset:
                        line = line.strip()
                        if not line:
                            return None, f"<empty line in {meta.name}>"
                        parts = line.split("\t", 1)
                        if len(parts) == 2:
                            y, t = parts
                            try:
                                return int(y), t
                            except ValueError:
                                return None, t
                        else:
                            return None, line
            break
        acc += n
    return None, f"<unknown {global_id}>"


# -------------------- HNSW INDEX -------------------------------
def build_or_load_index(emb_paths, dim: int, out_dir: Path, efc=200, M=16, threads=8):
    index_path = out_dir / "hnsw.index"
    index = hnswlib.Index(space="cosine", dim=dim)
    if not index_path.exists():
        print("[HNSW] Building index…")
        N = sum(np.load(p, mmap_mode="r").shape[0] for p in emb_paths)
        index.init_index(max_elements=N, ef_construction=efc, M=M)
        index.set_num_threads(threads)
        start = 0
        for p in emb_paths:
            arr = np.load(p, mmap_mode="r").astype("float32")
            n = arr.shape[0]
            index.add_items(arr, ids=np.arange(start, start + n))
            start += n
            print(f"  Added {start}/{N}")
        index.save_index(str(index_path))
    else:
        index.load_index(str(index_path))
        index.set_num_threads(threads)
        print("[HNSW] Loaded existing index.")
    index.set_ef(100)
    return index


# -------------------- HISTOGRAMS & THRESHOLDS -------------------
def compute_distance_histograms(index, emb_paths, out_dir: Path, k=6):
    PLOTS = out_dir / "plots"; PLOTS.mkdir(exist_ok=True)
    bins = np.linspace(0.0, 1.2, 121)
    centers = 0.5 * (bins[1:] + bins[:-1])
    global_hist = np.zeros(len(bins)-1, int)
    rank_hists  = np.zeros((5, len(bins)-1), int)
    chunk_stats = []

    print("[Stats] Building distance histograms…")
    total = 0
    for a, b, q in iter_vectors(emb_paths, chunk=20000):
        labels, dist = index.knn_query(q, k=k)
        nd = dist[:, 1:]  # skip self
        flat = nd.ravel()
        h, _ = np.histogram(flat, bins=bins); global_hist += h
        for r in range(5):
            hr, _ = np.histogram(nd[:, r], bins=bins)
            rank_hists[r] += hr
        chunk_stats.append({
            "start":a, "end":b,
            "mean":float(np.mean(flat)),
            "median":float(np.median(flat)),
            "std":float(np.std(flat)),
            "min":float(np.min(flat)),
            "max":float(np.max(flat))
        })
        total += (b - a)
        if total % 100000 == 0:
            print(f"  processed {total} vectors")

    means=np.array([s["mean"] for s in chunk_stats])
    summary={
      "mean":float(means.mean()),
      "median":float(np.median(means)),
      "std":float(np.mean([s["std"] for s in chunk_stats])),
      "min":float(min(s["min"] for s in chunk_stats)),
      "max":float(max(s["max"] for s in chunk_stats))
    }
    with open(PLOTS/"distance_stats.json","w") as f: json.dump(summary,f,indent=2)
    print("[Stats]", json.dumps(summary, indent=2))

    # plots
    plt.figure(figsize=(12,6))
    plt.bar(centers, global_hist, width=(bins[1]-bins[0]), edgecolor="black", alpha=0.75)
    plt.xlabel("Cosine distance to nearest neighbors"); plt.ylabel("Frequency")
    plt.title("Distribution of distances to 5 nearest neighbors (all papers)")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(PLOTS/"dist_hist_all.png", dpi=150)

    for r in range(5):
        plt.figure(figsize=(10,5))
        plt.bar(centers, rank_hists[r], width=(bins[1]-bins[0]), edgecolor="black", alpha=0.75)
        plt.xlabel(f"Cosine distance to rank-{r+1} neighbor"); plt.ylabel("Frequency")
        plt.title(f"Distribution to {r+1}-th nearest neighbor")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(PLOTS/f"dist_rank_{r+1}.png", dpi=150)

    return bins, global_hist


def choose_tau_from_percentile(bins, global_hist, pkeep: float):
    cdf = np.cumsum(global_hist) / max(global_hist.sum(), 1)
    idx = np.searchsorted(cdf, pkeep)
    tau = float(bins[min(idx+1, len(bins)-1)])
    print(f"[Threshold] tau ≈ {tau:.4f} (≈{int(pkeep*100)}th percentile)")
    return tau


# -------------------- BUILD EDGE LIST ---------------------------
def build_edges_threshold(index, emb_paths, out_dir: Path, tau: float, k_for_search=50, sample_edges=2000):
    EDGE_OUT = out_dir / f"edges_tau{tau:.3f}.tsv"
    with open(EDGE_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["src", "dst", "dist", "weight"])
        print(f"[Graph] Writing edges to {EDGE_OUT} (K={k_for_search}, tau={tau:.4f})")
        for a, b, q in iter_vectors(emb_paths, chunk=20000):
            labels, dists = index.knn_query(q, k=k_for_search)
            for i in range(labels.shape[0]):
                src = a + i
                for j in range(1, labels.shape[1]):  # skip self
                    d = float(dists[i, j])
                    if d <= tau:
                        dst = int(labels[i, j])
                        if src < dst:
                            w.writerow([src, dst, d, 1.0 - d])
            print(f"  edges for nodes {a}-{b} processed.")
    print("[Graph] Edge list done.")
    # quick stats
    N = sum(np.load(p, mmap_mode="r").shape[0] for p in emb_paths)
    degree = {}
    edge_count = 0
    with open(EDGE_OUT, "r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            s = int(row["src"]); d = int(row["dst"])
            degree[s] = degree.get(s, 0) + 1
            degree[d] = degree.get(d, 0) + 1
            edge_count += 1
    avg_deg = (2.0*edge_count)/max(N, 1)
    med_deg = float(np.median(list(degree.values()))) if degree else 0.0
    iso = int(N - len(degree))
    stats_path = out_dir / "arxiv_graph_stats.txt"
    with open(stats_path, "w") as f:
        f.write("Graph Statistics\n")
        f.write("="*80 + "\n\n")
        f.write(f"Nodes: {N}\n")
        f.write(f"Edges: {edge_count}\n")
        f.write(f"Average Degree: {avg_deg:.2f}\n")
        f.write(f"Isolated Nodes: {iso}\n")
        f.write(f"Median Degree: {med_deg:.2f}\n")
    print(f"[Graph] Stats written -> {stats_path}")

    # sample a human-readable title edgelist
    title_edge_list_path = out_dir / "arxiv_edge_list_sample.txt"
    with open(EDGE_OUT, "r", encoding="utf-8") as f_in, \
         open(title_edge_list_path, "w", encoding="utf-8") as f_out:
        r = csv.DictReader(f_in, delimiter="\t")
        f_out.write("title1\ttitle2\tdist\tweight\n")
        for i, row in enumerate(r):
            if i >= sample_edges: break
            t1 = (id_to_meta(int(row["src"]), emb_paths, out_dir)[1] or "").replace("\t"," ").replace("\n"," ").strip()
            t2 = (id_to_meta(int(row["dst"]), emb_paths, out_dir)[1] or "").replace("\t"," ").replace("\n"," ").strip()
            f_out.write(f"{t1}\t{t2}\t{row['dist']}\t{row['weight']}\n")
    print(f"[Graph] Sample title edges -> {title_edge_list_path}")
    return EDGE_OUT


# -------------------- KMEANS (MINIBATCH) ------------------------
def minibatch_kmeans_on_embeddings(emb_paths, out_dir: Path, clusters: int, batch_size=2048, passes=2, random_state=42):
    """Stream over shards and partial_fit a MiniBatchKMeans."""
    if clusters <= 0:
        print("[KMeans] Skipped (clusters <= 0)")
        return None
    print(f"[KMeans] MiniBatchKMeans: k={clusters}, batch={batch_size}, passes={passes}")
    kmeans = MiniBatchKMeans(n_clusters=clusters, batch_size=batch_size, random_state=random_state, n_init="auto")
    for p in emb_paths:
        arr = np.load(p, mmap_mode="r").astype("float32")
        n = arr.shape[0]
        for _ in range(passes):
            for i in range(0, n, batch_size):
                j = min(i + batch_size, n)
                kmeans.partial_fit(arr[i:j])
    # Assign labels streaming and save
    labels_out = out_dir / "kmeans_labels.npy"
    with open(labels_out, "wb") as _:
        pass  # just ensure file exists
    all_labels = []
    for p in emb_paths:
        arr = np.load(p, mmap_mode="r").astype("float32")
        preds = kmeans.predict(arr)
        all_labels.append(preds)
    labels = np.concatenate(all_labels)
    np.save(out_dir / "kmeans_labels.npy", labels)
    print(f"[KMeans] Saved labels -> {out_dir / 'kmeans_labels.npy'}")
    return labels


# -------------------- MAIN -------------------------------------
def main():
    ap = argparse.ArgumentParser(description="ArXiv USE→HNSW→Graph→KMeans pipeline")
    ap.add_argument("--file", type=str, required=True, help="Path to arxiv-metadata-oai-snapshot.json")
    ap.add_argument("--out", type=str, default="emb_stream", help="Output directory")
    ap.add_argument("--min-year", type=int, default=2020)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--shard", type=int, default=25000)
    ap.add_argument("--use-gpu", action="store_true")
    ap.add_argument("--vram-mib", type=int, default=3500)
    ap.add_argument("--k-for-search", type=int, default=50, help="Neighbors per query when building edges")
    ap.add_argument("--tau", type=float, default=None, help="Cosine distance threshold (if set, overrides pkeep)")
    ap.add_argument("--pkeep", type=float, default=0.20, help="Percentile of neighbor distances to keep (0..1)")
    ap.add_argument("--clusters", type=int, default=0, help="MiniBatchKMeans clusters (0 to skip)")
    ap.add_argument("--sample-edges", type=int, default=2000)
    args = ap.parse_args()

    meta_json = Path(args.file)
    out_dir = Path(args.out); out_dir.mkdir(exist_ok=True)

    setup_environment(args.use_gpu, args.vram_mib)

    print("[USE] Loading Universal Sentence Encoder…")
    use_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                               input_shape=[], dtype=tf.string, trainable=False)

    # Step 1: Embed in shards (or reuse)
    if not list(out_dir.glob("emb_*.npy")):
        print("[Embed] Streaming & embedding abstracts…")
        total = shard_stream(meta_json, out_dir, args.min_year, args.batch, args.shard, use_layer)
        print(f"[Embed] Done. Total embedded: {total}")
    else:
        print("[Embed] Shards exist. Skipping embedding.")

    # Step 2: Load shapes & build/load index
    emb_paths = sorted(glob.glob(str(out_dir / "emb_*.npy")))
    if not emb_paths:
        raise RuntimeError("No embedding shards found.")
    N, D = load_shapes(emb_paths)
    print(f"[Info] Total vectors: {N:,}  dim: {D}")

    index = build_or_load_index(emb_paths, D, out_dir)

    # Step 3: Histograms & pick threshold
    bins, global_hist = compute_distance_histograms(index, emb_paths, out_dir)
    if args.tau is not None:
        tau = float(args.tau)
        print(f"[Threshold] Using provided tau = {tau:.4f}")
    else:
        tau = choose_tau_from_percentile(bins, global_hist, args.pkeep)

    # Step 4: Build edge list at threshold
    edge_path = build_edges_threshold(index, emb_paths, out_dir, tau, k_for_search=args.k_for_search, sample_edges=args.sample_edges)

    # Step 5: Optional MiniBatchKMeans on embeddings
    if args.clusters and args.clusters > 0:
        _ = minibatch_kmeans_on_embeddings(emb_paths, out_dir, clusters=args.clusters)

    # Sample recommendations (sanity check)
    if N > 10:
        qid = random.randint(0, N-1)
        labels, dists = index.knn_query(np.load(emb_paths[0], mmap_mode="r")[:1].astype("float32"), k=1)  # dummy warmup
        print("\n" + "="*80)
        y, t = id_to_meta(qid, emb_paths, out_dir)
        print(f"Query [{qid}] ({y}) {t}\n")
        q_acc = 0
        # fetch vector for qid
        vec = None
        for p in emb_paths:
            arr = np.load(p, mmap_mode="r")
            n = arr.shape[0]
            if qid < q_acc + n:
                vec = np.asarray(arr[qid - q_acc: qid - q_acc + 1], dtype="float32")
                break
            q_acc += n
        labels, dists = index.knn_query(vec, k=6)
        ids = labels[0].astype(int).tolist(); ds = dists[0].tolist()
        if ids and ids[0] == qid:
            ids, ds = ids[1:], ds[1:]
        for i, (rid, d) in enumerate(zip(ids[:5], ds[:5]), 1):
            ry, rt = id_to_meta(int(rid), emb_paths, out_dir)
            print(f"  {i}) [{rid}] (d={d:.4f}) ({ry}) {rt}")
        print("="*80 + "\n")

    print("[Done] Outputs in:", out_dir)


if __name__ == "__main__":
    main()
