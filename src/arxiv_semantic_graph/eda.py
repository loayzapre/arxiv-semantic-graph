"""
EDA utilities for the arXiv metadata snapshot.

This module is meant to be used from a Jupyter notebook.
It provides:

- iter_papers(...)          -> stream (year, abstract)
- load_eda_data(...)        -> return years, lengths
- compute_eda_stats(...)    -> summary statistics as a dict
- plot_year_distribution(...) and plot_length_distribution(...)
"""

import json
from pathlib import Path
from typing import Iterator, Tuple, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


# ---------------------------------------------------------
# Core iterator over papers
# ---------------------------------------------------------
def iter_papers(
    path: Path,
    max_papers: Optional[int] = None,
    min_year: int = 0,
) -> Iterator[Tuple[int, str]]:
    """
    Stream (year, abstract) from the JSONL file.

    Parameters
    ----------
    path : Path
        Path to arxiv-metadata-oai-snapshot.json
    max_papers : int or None
        Optional limit to speed up experiments.
    min_year : int
        Minimum update_date year to include.
    """
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                p = json.loads(line)
            except json.JSONDecodeError:
                continue

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


# ---------------------------------------------------------
# 1. Load years + abstract lengths
# ---------------------------------------------------------
def load_eda_data(
    file_path: str,
    max_papers: Optional[int] = None,
    min_year: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load years and abstract lengths as numpy arrays.

    Returns
    -------
    years : np.ndarray of shape (N,)
    lengths : np.ndarray of shape (N,)
    """
    path = Path(file_path)
    years = []
    lengths = []

    for year, abstract in iter_papers(path, max_papers=max_papers, min_year=min_year):
        years.append(year)
        lengths.append(len(abstract.split()))

    if not years:
        raise RuntimeError("No valid papers found. Check file_path/min_year filters.")

    return np.array(years, dtype=int), np.array(lengths, dtype=int)


# ---------------------------------------------------------
# 2. Compute summary statistics
# ---------------------------------------------------------
def compute_eda_stats(
    years: np.ndarray,
    lengths: np.ndarray,
) -> Dict[str, float]:
    """
    Compute basic statistics for years and abstract lengths.
    """
    short_5 = int(np.sum(lengths < 5))
    short_10 = int(np.sum(lengths < 10))
    total = len(lengths)

    return {
        "total_papers": int(total),
        "year_min": int(years.min()),
        "year_max": int(years.max()),
        "len_mean": float(lengths.mean()),
        "len_median": float(np.median(lengths)),
        "len_min": int(lengths.min()),
        "len_max": int(lengths.max()),
        "short_5": short_5,
        "short_5_pct": 100.0 * short_5 / total,
        "short_10": short_10,
        "short_10_pct": 100.0 * short_10 / total,
    }


# ---------------------------------------------------------
# 3. Plotting helpers
# ---------------------------------------------------------
def plot_year_distribution(
    years: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot the distribution of papers by year.
    """
    counts = Counter(years.tolist())
    xs = sorted(counts.keys())
    ys = [counts[x] for x in xs]

    plt.figure(figsize=(7, 4))
    plt.bar(xs, ys)
    plt.xlabel("Year")
    plt.ylabel("Number of papers")
    plt.title("Distribution of papers by year")
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)

    if show:
        plt.show()
    plt.close()

def plot_length_distribution(
    lengths: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot the distribution of abstract lengths (words).
    """
    plt.figure(figsize=(7, 4))
    plt.hist(lengths, bins=50)
    plt.xlabel("Abstract length (words)")
    plt.ylabel("Frequency")
    plt.title("Distribution of abstract lengths")
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)

    if show:
        plt.show()
    plt.close()
