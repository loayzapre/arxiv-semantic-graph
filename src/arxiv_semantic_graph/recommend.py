#!/usr/bin/env python3
"""
Recommendation utilities based on the HNSW index + Louvain communities.

- Given an embedding directory (with emb_*.npy + meta_*.tsv),
- and a trained HNSW index,
we can:

1) Pick a random paper.
2) Retrieve its k nearest neighbours.
3) Mark neighbours as "significant" if they lie in the same
   Louvain community as the query paper, based on the best graph.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import random
from typing import List, Dict, Any, Optional

from arxiv_semantic_graph import embeddings as emb_mod


# -------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------


def _load_emb_paths(emb_dir: Path) -> List[Path]:
    """Return sorted list of embedding shard paths (emb_*.npy)."""
    emb_paths = sorted(emb_dir.glob("emb_*.npy"))
    if not emb_paths:
        raise FileNotFoundError(f"No emb_*.npy shards found in {emb_dir}")
    return emb_paths


def _get_paper_meta(global_id: int, emb_paths: List[Path], emb_dir: Path) -> Dict[str, Any]:
    """
    Get metadata (year, title) for a global paper id, using the helper
    from embeddings.py.
    """
    year, title = emb_mod.id_to_meta(global_id, [str(p) for p in emb_paths], emb_dir)
    return {"id": global_id, "year": year, "title": title}


def _load_best_graph_partition(outputs_dir: Path) -> Dict[int, int]:
    """
    Load Louvain partition for the best graph, using best_graph_config.json.

    Returns
    -------
    dict
        Mapping: node_id -> community_id.
        If config or partition are missing, returns an empty dict.
    """
    cfg_path = outputs_dir / "best_graph_config.json"
    if not cfg_path.exists():
        print(f"[Recommend] best_graph_config.json not found in {outputs_dir}, "
              "Louvain significance labels will be disabled.")
        return {}

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    louvain_dir = Path(cfg["louvain_dir"])
    partition_path = louvain_dir / "louvain_partition.tsv"

    if not partition_path.exists():
        print(f"[Recommend] Louvain partition not found at {partition_path}, "
              "Louvain significance labels will be disabled.")
        return {}

    node_to_comm: Dict[int, int] = {}
    with partition_path.open("r", encoding="utf-8") as f:
        header = f.readline()  # skip header: node\tcommunity
        for line in f:
            line = line.strip()
            if not line:
                continue
            node_str, comm_str = line.split("\t")
            try:
                node = int(node_str)
                comm = int(comm_str)
                node_to_comm[node] = comm
            except ValueError:
                continue

    print(f"[Recommend] Loaded Louvain partition for {len(node_to_comm):,} nodes.")
    return node_to_comm


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------


def load_metadata(emb_dir: str) -> List[Dict[str, Any]]:
    """
    Load all metadata (year, title) from meta_*.tsv files and assign
    global ids (0..N-1), consistent with the embedding shards.

    NOTE: For very large corpora this may be heavy in memory, so
    in practice it is often better to query metadata on demand via
    `_get_paper_meta`. This function is mainly for debugging / EDA.
    """
    emb_dir_path = Path(emb_dir)
    emb_paths = _load_emb_paths(emb_dir_path)

    meta_records: List[Dict[str, Any]] = []
    global_id = 0

    for shard_id, emb_path in enumerate(emb_paths):
        # meta file with the same shard id
        meta_path = emb_dir_path / f"meta_{shard_id:05d}.tsv"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {meta_path}")

        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    continue
                year_str, title = parts
                try:
                    year = int(year_str)
                except ValueError:
                    year = None
                meta_records.append(
                    {"id": global_id, "year": year, "title": title}
                )
                global_id += 1

    print(f"[Recommend] Loaded metadata for {len(meta_records):,} papers.")
    return meta_records


def recommend_for_id(
    emb_dir: str,
    index,
    paper_id: int,
    k: int = 5,
    tau: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Recommend k nearest neighbours for the given paper id.

    The function:
      1) Uses the HNSW index to query k nearest neighbours.
      2) Fetches metadata (year, title) for each neighbour.
      3) Uses the best Louvain partition (if available) to determine
         whether each neighbour lies in the same community as the query
         paper.

    Parameters
    ----------
    emb_dir : str
        Directory containing emb_*.npy and meta_*.tsv.

    index : hnswlib.Index
        Pre-loaded HNSW index with embeddings.

    paper_id : int
        Global paper id (0-based).

    k : int, default=5
        Number of neighbours to return.

    tau : float or None, optional
        Currently unused; kept for compatibility with earlier skeleton.

    Returns
    -------
    list of dict
        Each dict has keys:
          - id
          - year
          - title
          - distance
          - in_partition (bool)
          - same_community (bool)
          - significance ("significant" / "non-significant")
    """
    emb_dir_path = Path(emb_dir)
    emb_paths = _load_emb_paths(emb_dir_path)
    outputs_dir = emb_dir_path.parent

    # Load Louvain partition for best graph (if any)
    node_to_comm = _load_best_graph_partition(outputs_dir)

    # Basic sanity check on paper_id range
    max_id = index.get_current_count()
    if paper_id < 0 or paper_id >= max_id:
        raise ValueError(f"paper_id {paper_id} out of range [0, {max_id})")

    # Query embedding via the index
    query_vec = index.get_items([paper_id])  # shape: (1, dim)

    # k+1 because the first neighbour is usually the query itself
    labels, distances = index.knn_query(query_vec, k=k + 1)
    labels = labels[0]
    distances = distances[0]

    # Remove self if present
    neighbours = [
        (int(lbl), float(dist))
        for lbl, dist in zip(labels, distances)
        if int(lbl) != paper_id
    ][:k]

    # Louvain community of the query paper (if present)
    query_comm = node_to_comm.get(paper_id, None)

    results: List[Dict[str, Any]] = []
    for nid, dist in neighbours:
        meta = _get_paper_meta(nid, emb_paths, emb_dir_path)

        in_part = nid in node_to_comm
        same_comm = in_part and (query_comm is not None) and (node_to_comm[nid] == query_comm)
        significance = "significant" if same_comm else "non-significant"

        results.append(
            {
                "id": nid,
                "year": meta["year"],
                "title": meta["title"],
                "distance": dist,
                "in_partition": in_part,
                "same_community": same_comm,
                "significance": significance,
            }
        )

    return results


def recommend_random(
    emb_dir: str,
    index,
    k: int = 5,
    tau: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Pick a random paper id and recommend neighbours.

    Prints the query paper's metadata and returns the list of neighbours
    as described in `recommend_for_id`.
    """
    max_id = index.get_current_count()
    paper_id = random.randint(0, max_id - 1)

    emb_dir_path = Path(emb_dir)
    emb_paths = _load_emb_paths(emb_dir_path)

    query_meta = _get_paper_meta(paper_id, emb_paths, emb_dir_path)
    print("\n============================")
    print(" Random query paper")
    print("============================")
    print(f"ID    : {paper_id}")
    print(f"Year  : {query_meta['year']}")
    print(f"Title : {query_meta['title']}")
    print("============================\n")

    neighbours = recommend_for_id(
        emb_dir=emb_dir,
        index=index,
        paper_id=paper_id,
        k=k,
        tau=tau,
    )

    print("[Recommendations]")
    for rec in neighbours:
        tag = "✓ significant" if rec["significance"] == "significant" else "· non-significant"
        print(f"- (id={rec['id']}, year={rec['year']}) {tag}")
        print(f"    {rec['title']}")
        print(f"    distance={rec['distance']:.4f}\n")

    return neighbours


# -------------------------------------------------------------------
# CLI entry point (simple demo)
# -------------------------------------------------------------------


def main() -> None:
    """
    Simple CLI demo.

    NOTE: The CLI here is intentionally minimal. In practice you'll
    usually import this module in a notebook, load your HNSW index
    there, and then call `recommend_random` or `recommend_for_id`.
    """
    ap = argparse.ArgumentParser(description="Recommendation demo based on HNSW index")
    ap.add_argument("--emb-dir", type=str, required=True, help="Directory with emb_*.npy + meta_*.tsv")
    ap.add_argument("--index-path", type=str, required=True, help="Path to HNSW index (.bin)")
    ap.add_argument("--index-dim", type=int, required=True, help="Embedding dimension used in HNSW")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    import hnswlib  # local import to keep dependency optional at import time

    emb_dir = args.emb_dir
    index_path = args.index_path
    dim = args.index_dim

    # For CLI we assume max_elements is large enough; current_count will
    # reflect actual loaded items.
    index = hnswlib.Index(space="cosine", dim=dim)
    index.load_index(index_path)

    recommend_random(emb_dir=emb_dir, index=index, k=args.k)


if __name__ == "__main__":
    main()
