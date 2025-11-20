#!/usr/bin/env python3
"""
HNSW index construction and loading.
"""

from pathlib import Path
import argparse


def build_or_load_hnsw(
    emb_dir: str,
    space: str = "cosine",
    ef_construction: int = 200,
    M: int = 16,
    ef_search: int = 100,
    num_threads: int = 8,
):
    """
    Build or load an HNSW index for the embeddings stored in emb_*.npy.

    Returns
    -------
    index : hnswlib.Index (when implemented)
    N : int
        Total number of vectors
    D : int
        Embedding dimension
    """
    print(f"[HNSW] (skeleton) would scan embeddings in: {emb_dir}")
    # TODO: implementar lógica real con hnswlib.
    index = None
    N, D = 0, 0
    return index, N, D


def main() -> None:
    ap = argparse.ArgumentParser(description="Build or load HNSW index for Arxiv embeddings")
    ap.add_argument("--emb-dir", type=str, required=True, help="Directory with emb_*.npy")
    args = ap.parse_args()

    _index, N, D = build_or_load_hnsw(args.emb_dir)
    print(f"[HNSW] (skeleton) index built/loaded with N={N}, D={D}")


if __name__ == "__main__":
    main()
