#!/usr/bin/env python3
"""
Simple recommendation utilities based on the HNSW index.
"""

from pathlib import Path
import argparse
import random
from typing import List, Dict, Any


def load_metadata(emb_dir: str) -> List[Dict[str, Any]]:
    """
    Load metadata (year, title) from meta_*.tsv files.

    Skeleton: just prints what it would do.
    """
    print(f"[Recommend] (skeleton) would read meta_*.tsv from {emb_dir}")
    return []


def recommend_for_id(
    emb_dir: str,
    index,
    paper_id: int,
    k: int = 5,
    tau: float | None = None,
) -> List[Dict[str, Any]]:
    """
    Recommend k nearest neighbours for the given paper id.

    Skeleton version: just describes the call.
    """
    print(f"[Recommend] (skeleton) would query HNSW for id={paper_id}, k={k}, tau={tau}")
    return []


def recommend_random(
    emb_dir: str,
    index,
    k: int = 5,
    tau: float | None = None,
) -> List[Dict[str, Any]]:
    """
    Pick a random paper id and recommend neighbours.
    """
    paper_id = random.randint(0, 10)  # dummy
    print(f"[Recommend] (skeleton) picked random id={paper_id}")
    return recommend_for_id(emb_dir, index, paper_id, k=k, tau=tau)


def main() -> None:
    ap = argparse.ArgumentParser(description="Recommendation demo based on HNSW index")
    ap.add_argument("--emb-dir", type=str, required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tau", type=float, default=None)
    args = ap.parse_args()

    print("[Recommend] (skeleton) main – here we would:")
    print("  1) load HNSW index")
    print("  2) pick a random paper")
    print("  3) print k nearest neighbours")


if __name__ == "__main__":
    main()
