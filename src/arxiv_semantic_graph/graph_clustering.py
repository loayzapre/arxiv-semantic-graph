#!/usr/bin/env python3
"""
Graph clustering (e.g. Louvain) and modularity computation.
"""

from pathlib import Path
import argparse
import csv
import json
from typing import Dict, Any

import networkx as nx
from community import community_louvain


def run_louvain(
    edge_path: str,
    num_nodes: int,
    out_dir: str,
) -> Dict[str, Any]:
    """
    Run Louvain on the given edge list and compute modularity.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    print(f"[Louvain] Loading edge list from {edge_path} (N={num_nodes})")

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))  # ensure isolated nodes are present

    with open(edge_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                src = int(row["src"])
                dst = int(row["dst"])
                weight = float(row.get("weight", 1.0))
            except Exception:
                continue
            G.add_edge(src, dst, weight=weight)

    if G.number_of_edges() == 0:
        print("[Louvain] No edges found; skipping clustering.")
        result = {"modularity": None, "num_communities": 0}
    else:
        partition = community_louvain.best_partition(G, weight="weight")
        modularity = community_louvain.modularity(partition, G, weight="weight")
        num_communities = len(set(partition.values()))

        print(f"[Louvain] Modularity = {modularity:.4f}")
        print(f"[Louvain] Communities found = {num_communities}")

        # Save full partition mapping (node -> community) to disk
        partition_path = Path(out_dir) / "louvain_partition.tsv"
        with partition_path.open("w", encoding="utf-8") as f:
            f.write("node\tcommunity\n")
            for node, comm in partition.items():
                f.write(f"{node}\t{comm}\n")
        print(f"[Louvain] Partition saved to {partition_path}")

        # Also write a short human-readable summary
        summary = {
            "modularity": modularity,
            "num_communities": num_communities,
            "edges": G.number_of_edges(),
            "nodes": G.number_of_nodes(),
        }
        summary_path = Path(out_dir) / "louvain_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[Louvain] Summary saved to {summary_path}")

        result = {
            "modularity": modularity,
            "num_communities": num_communities,
        }

    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Graph clustering (Louvain) for Arxiv graph")
    ap.add_argument("--edges", type=str, required=True, help="Path to edges_tau*.tsv")
    ap.add_argument("--num-nodes", type=int, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    args = ap.parse_args()

    run_louvain(
        edge_path=args.edges,
        num_nodes=args.num_nodes,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
