#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) for the arXiv metadata snapshot.

This script:
- Streams the JSONL file line by line (no need to load everything in memory).
- Extracts:
    * publication year (from 'update_date')
    * abstract length in words
- Generates:
    * histogram of years
    * histogram of abstract lengths
    * a small text file with summary statistics
    * short-abstract quality indicators (<5 and <10 words)

It can be used either as a standalone script (CLI) or imported from
run_all.py via the run_eda(...) function.
"""

import os
os.environ["MPLBACKEND"] = "Agg"  # headless backend to avoid Qt/Wayland issues

import json
import argparse
from pathlib import Path
from collections import Counter
from typing import Optional, Iterator, Tuple, List

import numpy as np
import matplotlib.pyplot as plt


def iter_papers(
    path: Path,
    max_papers: Optional[int] = None,
    min_year: int = 0,
) -> Iterator[Tuple[int, str]]:
    """
    Stream papers from the JSONL file, yielding (year, abstract_text).

    Parameters
    ----------
    path : Path
        Path to the arxiv-metadata-oai-snapshot.json file (JSONL).
    max_papers : int or None, optional
        Optional limit for speed/testing. If None, read all valid papers.
    min_year : int, optional
        Minimum year to include. Papers with update_date year < min_year are skipped.
    """
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                p = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Year from update_date: "YYYY-MM-DD"
            y_str = (p.get("update_date") or "0-0-0").split("-")[0]
            try:
                year = int(y_str)
            except ValueError:
                continue

            if year < min_year:
                continue

            abstract = (p.get("abstract") or "").strip()
            if not abstract:
                continue

            yield year, abstract

            count += 1
            if max_papers is not None and count >= max_papers:
                break


def run_eda(
    file_path: str,
    out_dir: str,
    max_papers: Optional[int] = None,
    min_year: int = 0,
) -> None:
    """
    Run EDA on the given arXiv metadata file.

    Parameters
    ----------
    file_path : str
        Path to arxiv-metadata-oai-snapshot.json (JSONL).
    out_dir : str
        Output directory where plots and stats will be stored.
    max_papers : int or None, optional
        Maximum number of papers to read (None = read all).
    min_year : int, optional
        Minimum year to include (filter by update_date year).
    """
    data_path = Path(file_path)
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    years: List[int] = []
    lengths: List[int] = []

    print(f"[EDA] Reading from {data_path}")
    print(f"[EDA] max_papers={max_papers}, min_year={min_year}")

    for year, abstract in iter_papers(data_path, max_papers=max_papers, min_year=min_year):
        years.append(year)
        lengths.append(len(abstract.split()))

    if not years:
        print("[EDA] No valid papers found. Check the input file or filters.")
        return

    years_arr = np.array(years)
    lengths_arr = np.array(lengths)

    print(f"[EDA] Total valid papers: {len(years_arr)}")
    print(f"[EDA] Year range: {years_arr.min()} - {years_arr.max()}")
    print(
        "[EDA] Abstract length (words): "
        f"mean={lengths_arr.mean():.1f}, "
        f"median={np.median(lengths_arr):.1f}, "
        f"min={lengths_arr.min()}, "
        f"max={lengths_arr.max()}"
    )

    # ---------- Short abstract statistics ----------
    short_5 = int(np.sum(lengths_arr < 5))
    short_10 = int(np.sum(lengths_arr < 10))

    pct_5 = 100.0 * short_5 / len(lengths_arr)
    pct_10 = 100.0 * short_10 / len(lengths_arr)

    print(f"[EDA] Abstracts < 5 words : {short_5} ({pct_5:.2f}%)")
    print(f"[EDA] Abstracts < 10 words: {short_10} ({pct_10:.2f}%)")

    # ---------- Plot 1: Year distribution ----------
    plt.figure()
    year_counts = Counter(years_arr)
    xs = sorted(year_counts.keys())
    ys = [year_counts[x] for x in xs]
    plt.bar(xs, ys)
    plt.xlabel("Year")
    plt.ylabel("Number of papers")
    plt.title("Distribution of papers by year")
    plt.tight_layout()
    year_plot_path = out_dir_path / "year_distribution.png"
    plt.savefig(year_plot_path, dpi=150)
    plt.close()
    print(f"[EDA] Saved year distribution -> {year_plot_path}")

    # ---------- Plot 2: Abstract length distribution ----------
    plt.figure()
    plt.hist(lengths_arr, bins=50)
    plt.xlabel("Abstract length (words)")
    plt.ylabel("Frequency")
    plt.title("Distribution of abstract lengths")
    plt.tight_layout()
    length_plot_path = out_dir_path / "abstract_length_distribution.png"
    plt.savefig(length_plot_path, dpi=150)
    plt.close()
    print(f"[EDA] Saved abstract length distribution -> {length_plot_path}")

    # ---------- Save stats ----------
    stats_path = out_dir_path / "eda_stats.txt"
    with stats_path.open("w", encoding="utf-8") as f:
        f.write("EDA Statistics\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total valid papers: {len(years_arr)}\n")
        f.write(f"Year range: {years_arr.min()} - {years_arr.max()}\n\n")
        f.write("Abstract length (words):\n")
        f.write(f"  mean   = {lengths_arr.mean():.2f}\n")
        f.write(f"  median = {np.median(lengths_arr):.2f}\n")
        f.write(f"  min    = {lengths_arr.min()}\n")
        f.write(f"  max    = {lengths_arr.max()}\n")

        f.write("\nShort abstract quality indicators:\n")
        f.write(f"  abstracts < 5 words   = {short_5} ({pct_5:.2f}%)\n")
        f.write(f"  abstracts < 10 words  = {short_10} ({pct_10:.2f}%)\n")

    print(f"[EDA] Stats written -> {stats_path}")
    print("[EDA] Done.")


def main():
    ap = argparse.ArgumentParser(description="EDA for arXiv metadata snapshot")
    ap.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to arxiv-metadata-oai-snapshot.json (JSONL file)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="outputs/eda",
        help="Output directory for plots and stats",
    )
    ap.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Maximum number of papers to read (None = read all dataset)",
    )
    ap.add_argument(
        "--min-year",
        type=int,
        default=0,
        help="Minimum year to include (filter on update_date year)",
    )
    args = ap.parse_args()

    run_eda(
        file_path=args.file,
        out_dir=args.out,
        max_papers=args.max_papers,
        min_year=args.min_year,
    )


if __name__ == "__main__":
    main()
