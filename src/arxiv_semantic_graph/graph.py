#!/usr/bin/env python3
"""
Graph construction and distance histogram utilities for ArXiv embeddings.

Usage:
    python graph.py --emb-dir ./emb_stream --out-dir ./graph_output --tau 0.35 --k-for-search 50
    
    Or to compute histograms and auto-select tau:
    python graph.py --emb-dir ./emb_stream --out-dir ./graph_output --pkeep 0.20 --k-for-search 50

    # With custom parameters
    python graph.py --emb-dir ./emb_stream --out-dir ./graph_output \ --tau 0.35 --k-for-search 100 --sample-edges 5000

Arguments:
    --emb-dir: Directory containing emb_*.npy and meta_*.tsv shard files
    --out-dir: Output directory for graphs and statistics
    --tau: Cosine distance threshold for edge creation (overrides --pkeep)
    --pkeep: Percentile of distances to keep (e.g., 0.20 = 20th percentile)
    --k-for-search: Number of nearest neighbors to query per node (default: 50)
    --k-hist: Number of neighbors for histogram analysis (default: 6)
    --sample-edges: Number of sample edges with titles to output (default: 2000)
"""
import csv
import glob
import json
from pathlib import Path
import argparse
from typing import Sequence, Dict, Any, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless plotting
import matplotlib.pyplot as plt
import hnswlib


# -------------------- IO HELPERS -----------------------
def load_shapes(emb_paths: Sequence[str]) -> Tuple[int, int]:
    """Get total number of vectors and dimensionality."""
    shapes = [np.load(p, mmap_mode="r").shape for p in emb_paths]
    N = sum(s[0] for s in shapes)
    D = shapes[0][1]
    return N, D


def iter_vectors(emb_paths: Sequence[str], chunk: int = 20000):
    """Stream embeddings from shards in chunks."""
    acc = 0
    for p in emb_paths:
        arr = np.load(p, mmap_mode="r")
        n = arr.shape[0]
        for i in range(0, n, chunk):
            j = min(i + chunk, n)
            yield acc + i, acc + j, np.asarray(arr[i:j], dtype="float32")
        acc += n


def id_to_meta(global_id: int, emb_paths: Sequence[str], emb_dir: Path) -> Tuple[int | None, str]:
    """Return (year, title) for a global id."""
    acc = 0
    for shard_id, f in enumerate(emb_paths):
        n = np.load(f, mmap_mode="r").shape[0]
        if global_id < acc + n:
            offset = global_id - acc
            meta = emb_dir / f"meta_{shard_id:05d}.tsv"
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


# -------------------- HNSW INDEX -----------------------
def build_or_load_index(emb_paths: Sequence[str], dim: int, out_dir: Path, 
                        efc: int = 200, M: int = 16, threads: int = 8):
    """Build or load HNSW index from embedding shards."""
    out_dir.mkdir(parents=True, exist_ok=True)
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


# -------------------- HISTOGRAMS & THRESHOLDS -----------------------
def compute_knn_distance_histogram(
    emb_dir: str,
    index,
    k: int = 6,
    out_dir: str | None = None,
) -> Dict[str, Any]:
    """
    Compute histogram of distances to k nearest neighbors.
    Returns bins, histogram counts, and summary statistics.
    """
    emb_path = Path(emb_dir)
    emb_paths = sorted(glob.glob(str(emb_path / "emb_*.npy")))
    
    if out_dir is not None:
        plots_dir = Path(out_dir) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
    else:
        plots_dir = None
    
    bins = np.linspace(0.0, 1.2, 121)
    centers = 0.5 * (bins[1:] + bins[:-1])
    global_hist = np.zeros(len(bins) - 1, int)
    rank_hists = np.zeros((5, len(bins) - 1), int)
    chunk_stats = []
    
    print(f"[Histogram] Computing distance distributions with k={k}…")
    total = 0
    for a, b, q in iter_vectors(emb_paths, chunk=20000):
        labels, dist = index.knn_query(q, k=k)
        nd = dist[:, 1:]  # skip self
        flat = nd.ravel()
        h, _ = np.histogram(flat, bins=bins)
        global_hist += h
        
        for r in range(min(5, nd.shape[1])):
            hr, _ = np.histogram(nd[:, r], bins=bins)
            rank_hists[r] += hr
        
        chunk_stats.append({
            "start": a, "end": b,
            "mean": float(np.mean(flat)),
            "median": float(np.median(flat)),
            "std": float(np.std(flat)),
            "min": float(np.min(flat)),
            "max": float(np.max(flat))
        })
        total += (b - a)
        if total % 100000 == 0:
            print(f"  Processed {total} vectors")
    
    means = np.array([s["mean"] for s in chunk_stats])
    summary = {
        "mean": float(means.mean()),
        "median": float(np.median(means)),
        "std": float(np.mean([s["std"] for s in chunk_stats])),
        "min": float(min(s["min"] for s in chunk_stats)),
        "max": float(max(s["max"] for s in chunk_stats))
    }
    
    if plots_dir:
        with open(plots_dir / "distance_stats.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Plot overall histogram
        plt.figure(figsize=(12, 6))
        plt.bar(centers, global_hist, width=(bins[1] - bins[0]), 
                edgecolor="black", alpha=0.75)
        plt.xlabel("Cosine distance to nearest neighbors")
        plt.ylabel("Frequency")
        plt.title("Distribution of distances to 5 nearest neighbors (all papers)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "dist_hist_all.png", dpi=150)
        plt.close()
        
        # Plot per-rank histograms
        for r in range(5):
            plt.figure(figsize=(10, 5))
            plt.bar(centers, rank_hists[r], width=(bins[1] - bins[0]), 
                    edgecolor="black", alpha=0.75)
            plt.xlabel(f"Cosine distance to rank-{r+1} neighbor")
            plt.ylabel("Frequency")
            plt.title(f"Distribution to {r+1}-th nearest neighbor")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_dir / f"dist_rank_{r+1}.png", dpi=150)
            plt.close()
    
    print("[Histogram] Statistics:", json.dumps(summary, indent=2))
    
    return {
        "bins": bins,
        "global_hist": global_hist,
        "rank_hists": rank_hists,
        "summary": summary
    }


def choose_tau_from_percentile(bins: np.ndarray, global_hist: np.ndarray, 
                               pkeep: float) -> float:
    """Choose distance threshold tau based on percentile of histogram."""
    cdf = np.cumsum(global_hist) / max(global_hist.sum(), 1)
    idx = np.searchsorted(cdf, pkeep)
    tau = float(bins[min(idx + 1, len(bins) - 1)])
    print(f"[Threshold] tau ≈ {tau:.4f} (≈{int(pkeep*100)}th percentile)")
    return tau


# -------------------- GRAPH CONSTRUCTION -----------------------
def build_graph_for_tau(
    emb_dir: str,
    index,
    tau: float,
    k_for_search: int,
    out_dir: str,
    sample_edges: int = 2000,
) -> str:
    """
    Build edge list using the given distance threshold tau.
    Only creates edges between nodes with distance <= tau.
    Returns path to the edge list TSV file.
    """
    emb_path = Path(emb_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    emb_paths = sorted(glob.glob(str(emb_path / "emb_*.npy")))
    edge_path = out_path / f"edges_tau{tau:.3f}.tsv"
    
    print(f"[Graph] Building edge list at tau={tau:.4f}, k={k_for_search}")
    with open(edge_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["src", "dst", "dist", "weight"])
        
        for a, b, q in iter_vectors(emb_paths, chunk=20000):
            labels, dists = index.knn_query(q, k=k_for_search)
            for i in range(labels.shape[0]):
                src = a + i
                for j in range(1, labels.shape[1]):  # skip self
                    d = float(dists[i, j])
                    if d <= tau:
                        dst = int(labels[i, j])
                        if src < dst:  # avoid duplicate edges
                            w.writerow([src, dst, d, 1.0 - d])
            print(f"  Processed nodes {a}-{b}")
    
    print(f"[Graph] Edge list written to {edge_path}")
    
    # Create sample with paper titles
    title_edge_path = out_path / "arxiv_edge_list_sample.txt"
    with open(edge_path, "r", encoding="utf-8") as f_in, \
         open(title_edge_path, "w", encoding="utf-8") as f_out:
        r = csv.DictReader(f_in, delimiter="\t")
        f_out.write("title1\ttitle2\tdist\tweight\n")
        for i, row in enumerate(r):
            if i >= sample_edges:
                break
            t1 = (id_to_meta(int(row["src"]), emb_paths, emb_path)[1] or "")
            t1 = t1.replace("\t", " ").replace("\n", " ").strip()
            t2 = (id_to_meta(int(row["dst"]), emb_paths, emb_path)[1] or "")
            t2 = t2.replace("\t", " ").replace("\n", " ").strip()
            f_out.write(f"{t1}\t{t2}\t{row['dist']}\t{row['weight']}\n")
    
    print(f"[Graph] Sample edges with titles -> {title_edge_path}")
    return str(edge_path)


def compute_graph_stats(edge_path: str, num_nodes: int) -> Dict[str, Any]:
    """
    Compute basic graph statistics from an edge list.
    Returns dict with node count, edge count, avg degree, isolated nodes, etc.
    """
    print(f"[Stats] Computing graph statistics for {edge_path}")
    
    degree = {}
    edge_count = 0
    
    with open(edge_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            s = int(row["src"])
            d = int(row["dst"])
            degree[s] = degree.get(s, 0) + 1
            degree[d] = degree.get(d, 0) + 1
            edge_count += 1
    
    avg_deg = (2.0 * edge_count) / max(num_nodes, 1)
    med_deg = float(np.median(list(degree.values()))) if degree else 0.0
    isolated = int(num_nodes - len(degree))
    
    stats = {
        "nodes": num_nodes,
        "edges": edge_count,
        "avg_degree": avg_deg,
        "median_degree": med_deg,
        "isolated_nodes": isolated,
    }
    
    # Write to file
    stats_path = Path(edge_path).parent / "arxiv_graph_stats.txt"
    with open(stats_path, "w") as f:
        f.write("Graph Statistics\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Nodes: {stats['nodes']}\n")
        f.write(f"Edges: {stats['edges']}\n")
        f.write(f"Average Degree: {stats['avg_degree']:.2f}\n")
        f.write(f"Median Degree: {stats['median_degree']:.2f}\n")
        f.write(f"Isolated Nodes: {stats['isolated_nodes']}\n")
    
    print(f"[Stats] Written to {stats_path}")
    print(f"[Stats] {json.dumps(stats, indent=2)}")
    
    return stats


# -------------------- MAIN -----------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Graph construction utilities for ArXiv semantic embeddings"
    )
    ap.add_argument("--emb-dir", type=str, required=True,
                    help="Directory with emb_*.npy and meta_*.tsv shards")
    ap.add_argument("--out-dir", type=str, required=True,
                    help="Output directory for graphs and plots")
    ap.add_argument("--tau", type=float, default=None,
                    help="Distance threshold for edges (overrides --pkeep)")
    ap.add_argument("--pkeep", type=float, default=0.20,
                    help="Percentile of distances to keep (0..1, default: 0.20)")
    ap.add_argument("--k-for-search", type=int, default=50,
                    help="Number of neighbors to query per node (default: 50)")
    ap.add_argument("--k-hist", type=int, default=6,
                    help="Number of neighbors for histogram (default: 6)")
    ap.add_argument("--sample-edges", type=int, default=2000,
                    help="Number of sample edges with titles (default: 2000)")
    args = ap.parse_args()
    
    emb_dir = Path(args.emb_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load embedding paths
    emb_paths = sorted(glob.glob(str(emb_dir / "emb_*.npy")))
    if not emb_paths:
        raise RuntimeError(f"No embedding shards found in {emb_dir}")
    
    N, D = load_shapes(emb_paths)
    print(f"[Info] Found {len(emb_paths)} shards: {N:,} vectors, dim={D}")
    
    # Build or load HNSW index
    index = build_or_load_index(emb_paths, D, out_dir)
    
    # Compute distance histograms
    hist_result = compute_knn_distance_histogram(
        str(emb_dir), index, k=args.k_hist, out_dir=str(out_dir)
    )
    
    # Determine tau threshold
    if args.tau is not None:
        tau = float(args.tau)
        print(f"[Threshold] Using provided tau = {tau:.4f}")
    else:
        tau = choose_tau_from_percentile(
            hist_result["bins"], hist_result["global_hist"], args.pkeep
        )
    
    # Build graph
    edge_path = build_graph_for_tau(
        str(emb_dir), index, tau, args.k_for_search, 
        str(out_dir), args.sample_edges
    )
    
    # Compute statistics
    stats = compute_graph_stats(edge_path, N)
    
    print("\n" + "=" * 80)
    print("[Done] Graph construction complete!")
    print(f"  Edge list: {edge_path}")
    print(f"  Statistics: {out_dir / 'arxiv_graph_stats.txt'}")
    print(f"  Plots: {out_dir / 'plots'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
