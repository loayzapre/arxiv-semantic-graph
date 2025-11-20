"""
Arxiv Semantic Graph package.

This package contains the logic for:
- EDA on the Arxiv metadata
- Embedding abstracts with USE
- Building and querying an HNSW index
- Constructing and analysing a semantic graph
- Graph clustering (Louvain)
- Simple paper recommendation
"""

from . import eda, embeddings, hnsw_index, graph, graph_clustering, recommend

__all__ = ["eda", "embeddings", "hnsw_index", "graph", "graph_clustering", "recommend"]
