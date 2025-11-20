#!/usr/bin/env python3
"""
Embeddings module: compute USE embeddings in shards.
"""

from pathlib import Path
import argparse


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

    Returns
    -------
    int
        Total number of embedded papers.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[Embeddings] (skeleton) would read from: {file_path}")
    print(f"[Embeddings] (skeleton) would write shards to: {out_dir}")
    print(f"[Embeddings] (skeleton) min_year={min_year}, min_words={min_words}, use_gpu={use_gpu}")
    # TODO: implementar lógica real con USE + sharding.
    total = 0
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute USE embeddings for Arxiv abstracts")
    ap.add_argument("--file", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--min-year", type=int, default=2015)
    ap.add_argument("--min-words", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--shard-size", type=int, default=25000)
    ap.add_argument("--use-gpu", action="store_true")
    ap.add_argument("--vram-mib", type=int, default=3500)
    ap.add_argument("--force-recompute", action="store_true")
    args = ap.parse_args()

    run_embeddings(
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


if __name__ == "__main__":
    main()
