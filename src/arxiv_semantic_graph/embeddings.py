#!/usr/bin/env python3
"""
Embeddings module: compute USE embeddings in shards.

Usage:
    # Basic usage with CPU
    python embeddings.py --file arxiv-metadata-oai-snapshot.json --out emb_output
    
    # With GPU and custom parameters
    python embeddings.py --file arxiv-metadata-oai-snapshot.json --out emb_output \
        --min-year 2020 --min-words 20 --batch-size 128 --shard-size 50000 \
        --use-gpu --vram-mib 4096
    
    # Force recompute even if embeddings exist
    python embeddings.py --file arxiv-metadata-oai-snapshot.json --out emb_output \
        --force-recompute
    
Parameters:
    --file          : Path to arxiv-metadata-oai-snapshot.json
    --out           : Output directory for embeddings and metadata
    --min-year      : Minimum year to filter papers (default: 2015)
    --min-words     : Minimum word count in abstract (default: 10)
    --batch-size    : Number of abstracts to embed at once (default: 64)
    --shard-size    : Number of papers per shard file (default: 25000)
    --use-gpu       : Enable GPU acceleration
    --vram-mib      : GPU memory limit in MiB (default: 3500)
    --force-recompute : Recompute embeddings even if they exist
"""

from pathlib import Path
import argparse
import os, json, glob, warnings

import numpy as np

# Silence non-critical noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # TF: show only errors
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
import tensorflow_hub as hub


# -------------------- GPU / HEADLESS CONFIG ---------------------
def setup_environment(use_gpu: bool, vram_mib: int):
    """Configure GPU/CPU and headless matplotlib."""
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
def iter_papers(path: Path, min_year: int, min_words: int):
    """
    Stream papers from Arxiv JSON, filtering by year and word count.
    
    Yields
    ------
    tuple of (title: str, year: int, abstract: str)
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                p = json.loads(line)
                # Extract year
                y = int(p.get("update_date", "0-0-0").split("-")[0])
                if y < min_year:
                    continue
                
                # Extract title and abstract
                t = (p.get("title") or "").strip()
                a = (p.get("abstract") or "").strip()
                
                if not a:
                    continue
                
                # Check word count
                word_count = len(a.split())
                if word_count < min_words:
                    continue
                
                yield t, y, a
            except Exception:
                continue


def embed_batch(texts, use_layer, batch: int):
    """
    Embed a list of texts using USE in batches.
    
    Parameters
    ----------
    texts : list of str
        Texts to embed
    use_layer : hub.KerasLayer
        Universal Sentence Encoder layer
    batch : int
        Batch size for embedding
    
    Returns
    -------
    np.ndarray
        Embeddings of shape (len(texts), 512)
    """
    out, cur = [], []
    for t in texts:
        cur.append(t)
        if len(cur) == batch:
            out.append(use_layer(cur).numpy())
            cur = []
    if cur:
        out.append(use_layer(cur).numpy())
    return np.vstack(out)


def shard_stream(
    meta_json: Path,
    out_dir: Path,
    min_year: int,
    min_words: int,
    batch: int,
    shard: int,
    use_layer,
    force_recompute: bool
):
    """
    Stream through papers, embed abstracts, and save in shards.
    
    Parameters
    ----------
    meta_json : Path
        Path to arxiv-metadata-oai-snapshot.json
    out_dir : Path
        Output directory for embeddings
    min_year : int
        Minimum year filter
    min_words : int
        Minimum word count filter
    batch : int
        Batch size for embedding
    shard : int
        Number of papers per shard
    use_layer : hub.KerasLayer
        Universal Sentence Encoder layer
    force_recompute : bool
        If True, recompute even if shards exist
    
    Returns
    -------
    int
        Total number of papers embedded
    """
    titles, years, absbuf = [], [], []
    shard_id, total = 0, 0
    
    for title, year, abstract in iter_papers(meta_json, min_year, min_words):
        titles.append(title)
        years.append(year)
        absbuf.append(abstract)
        
        if len(absbuf) >= shard:
            # Check if this shard already exists
            emb_path = out_dir / f"emb_{shard_id:05d}.npy"
            meta_path = out_dir / f"meta_{shard_id:05d}.tsv"
            
            if not force_recompute and emb_path.exists() and meta_path.exists():
                print(f"[Shard {shard_id}] already exists, skipping (total {total + len(absbuf)})")
                shard_id += 1
                total += len(absbuf)
                titles.clear()
                years.clear()
                absbuf.clear()
                continue
            
            # Compute embeddings
            emb = embed_batch(absbuf, use_layer, batch).astype(np.float16)
            np.save(emb_path, emb)
            
            # Save metadata
            with open(meta_path, "w", encoding="utf-8") as g:
                for t, y in zip(titles, years):
                    g.write(f"{y}\t{t}\n")
            
            total += len(absbuf)
            print(f"[Shard {shard_id}] saved {len(absbuf)} papers (total {total})")
            shard_id += 1
            titles.clear()
            years.clear()
            absbuf.clear()
    
    # Handle remaining papers
    if absbuf:
        emb_path = out_dir / f"emb_{shard_id:05d}.npy"
        meta_path = out_dir / f"meta_{shard_id:05d}.tsv"
        
        if not force_recompute and emb_path.exists() and meta_path.exists():
            print(f"[Shard {shard_id}] already exists, skipping (total {total + len(absbuf)})")
            total += len(absbuf)
        else:
            emb = embed_batch(absbuf, use_layer, batch).astype(np.float16)
            np.save(emb_path, emb)
            
            with open(meta_path, "w", encoding="utf-8") as g:
                for t, y in zip(titles, years):
                    g.write(f"{y}\t{t}\n")
            
            total += len(absbuf)
            print(f"[Shard {shard_id}] saved {len(absbuf)} papers (total {total})")
    
    return total


def load_shapes(emb_paths):
    """Load shapes of embedding shards without loading full arrays."""
    shapes = [np.load(p, mmap_mode="r").shape for p in emb_paths]
    N = sum(s[0] for s in shapes)
    D = shapes[0][1] if shapes else 0
    return N, D


def iter_vectors(emb_paths, chunk=20000):
    """Iterate over embedding vectors in chunks."""
    acc = 0
    for p in emb_paths:
        arr = np.load(p, mmap_mode="r")
        n = arr.shape[0]
        for i in range(0, n, chunk):
            j = min(i + chunk, n)
            yield acc + i, acc + j, np.asarray(arr[i:j], dtype="float32")
        acc += n


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


def run_embeddings(
    file_path: str,
    out_dir: str,
    min_year: int = 2015,
    min_words: int = 10,
    batch_size: int = 64,
    shard_size: int = 25000,
    use_gpu: bool = False,
    vram_mib: int = 3500,
    force_recompute: bool = False,
) -> int:
    """
    Stream the Arxiv file, filter papers, and compute USE embeddings.

    Parameters
    ----------
    file_path : str
        Path to arxiv-metadata-oai-snapshot.json
    out_dir : str
        Output directory for embeddings and metadata
    min_year : int
        Minimum year to filter papers
    min_words : int
        Minimum word count in abstract
    batch_size : int
        Number of abstracts to embed at once
    shard_size : int
        Number of papers per shard file
    use_gpu : bool
        Enable GPU acceleration
    vram_mib : int
        GPU memory limit in MiB
    force_recompute : bool
        Recompute embeddings even if they exist

    Returns
    -------
    int
        Total number of embedded papers.
    """
    meta_json = Path(file_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # Setup environment
    setup_environment(use_gpu, vram_mib)
    
    # Check if embeddings already exist
    existing_shards = sorted(glob.glob(str(out / "emb_*.npy")))
    if existing_shards and not force_recompute:
        print(f"[Embeddings] Found {len(existing_shards)} existing shards in {out_dir}")
        N, D = load_shapes(existing_shards)
        print(f"[Embeddings] Total vectors: {N:,}, dim: {D}")
        print("[Embeddings] Use --force-recompute to recompute")
        return N
    
    print(f"[Embeddings] Reading from: {file_path}")
    print(f"[Embeddings] Writing shards to: {out_dir}")
    print(f"[Embeddings] Filters: min_year={min_year}, min_words={min_words}")
    print(f"[Embeddings] Batch size: {batch_size}, Shard size: {shard_size}")
    
    # Load Universal Sentence Encoder
    print("[USE] Loading Universal Sentence Encoder...")
    MODEL_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"
    use_layer = hub.KerasLayer(
        MODEL_URL,
        input_shape=[],
        dtype=tf.string,
        trainable=False,
        name="use"
    )
    print("[USE] Model loaded successfully")
    
    # Stream and embed
    print("[Embeddings] Starting streaming and embedding...")
    total = shard_stream(
        meta_json=meta_json,
        out_dir=out,
        min_year=min_year,
        min_words=min_words,
        batch=batch_size,
        shard=shard_size,
        use_layer=use_layer,
        force_recompute=force_recompute
    )
    
    print(f"[Embeddings] Complete! Total papers embedded: {total:,}")
    
    # Verify output
    emb_paths = sorted(glob.glob(str(out / "emb_*.npy")))
    if emb_paths:
        N, D = load_shapes(emb_paths)
        print(f"[Embeddings] Verification: {len(emb_paths)} shards, {N:,} vectors, dim={D}")
    
    return total


def main() -> None:
    """Main entry point for embeddings computation."""
    ap = argparse.ArgumentParser(
        description="Compute USE embeddings for Arxiv abstracts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    ap.add_argument("--file", type=str, required=True,
                    help="Path to arxiv-metadata-oai-snapshot.json")
    ap.add_argument("--out", type=str, required=True,
                    help="Output directory for embeddings")
    ap.add_argument("--min-year", type=int, default=2015,
                    help="Minimum year to filter papers (default: 2015)")
    ap.add_argument("--min-words", type=int, default=10,
                    help="Minimum word count in abstract (default: 10)")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="Batch size for embedding (default: 64)")
    ap.add_argument("--shard-size", type=int, default=25000,
                    help="Papers per shard file (default: 25000)")
    ap.add_argument("--use-gpu", action="store_true",
                    help="Enable GPU acceleration")
    ap.add_argument("--vram-mib", type=int, default=3500,
                    help="GPU memory limit in MiB (default: 3500)")
    ap.add_argument("--force-recompute", action="store_true",
                    help="Recompute embeddings even if they exist")
    args = ap.parse_args()

    total = run_embeddings(
        file_path=args.file,
        out_dir=args.out,
        min_year=args.min_year,
        min_words=args.min_words,
        batch_size=args.batch_size,
        shard_size=args.shard_size,
        use_gpu=args.use_gpu,
        vram_mib=args.vram_mib,
        force_recompute=args.force_recompute,
    )
    
    print(f"\n{'='*80}")
    print(f"Embedding computation complete!")
    print(f"Total papers processed: {total:,}")
    print(f"Output directory: {args.out}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
