ArXiv Semantic Graph
====================

This project builds a semantic similarity graph of arXiv papers using text
embeddings, approximate nearest-neighbour search, a global distance threshold,
and graph clustering. The goal is to discover topic communities and recommend
semantically similar papers.

Pipeline Overview
-----------------

1. Exploratory Data Analysis (EDA)
   - Stream the arXiv metadata JSONL file.
   - Analyse publication years and abstract length (in words).
   - Report the proportion of very short abstracts (< 5 and < 10 words).
   - Justify basic filtering decisions (e.g. minimum year, minimum length).

2. Text Embeddings (Universal Sentence Encoder)
   - Convert each abstract into a 512-dimensional embedding using USE.
   - Process the data in streaming and store embeddings in sharded .npy files.
   - Save metadata (year, title) in parallel .tsv shard files.

3. HNSW Index (Similar Items / kNN)
   - Build a Hierarchical Navigable Small World (HNSW) index on all embeddings.
   - This provides fast approximate nearest-neighbour search in the embedding space.
   - The index is reused for distance analysis, graph construction, and recommendation.

4. Distance Histogram and Global Threshold τ
   - For each paper, query its k nearest neighbours (e.g. k=5) using HNSW.
   - Collect all neighbour distances and build a global histogram.
   - Select one or more candidate distance thresholds τ (e.g. based on percentiles).
   - τ defines when two papers are considered semantically similar.

5. Semantic Graph Construction
   - For each candidate τ:
     - For every paper, query k neighbours (e.g. k=50) from HNSW.
     - Add an edge between two papers if their distance ≤ τ.
   - This yields a semantic similarity graph:
     - nodes = papers,
     - edges = strong semantic similarity.
   - Compute simple graph statistics (edges, average degree, isolated nodes).

6. Graph Clustering with Louvain
   - Run the Louvain algorithm on the similarity graph.
   - Obtain communities (clusters of papers) and the modularity score.
   - Communities are interpreted as topic groups.
   - Optionally, summarise each community by top TF-IDF terms or a representative paper.

7. Semantic Coherence and Recommendation
   - For each paper, use the HNSW index to find its nearest neighbours.
   - A relation is considered “significant” when neighbours fall in the same
     Louvain community under the chosen τ.
   - As a demo, select a paper and recommend its nearest neighbours, reporting
     titles, distances, and community assignments.

Project Layout
--------------

src/arxiv_semantic_graph/
  eda.py             – EDA utilities (loading, statistics, plots)
  embeddings.py      – USE embeddings in shards
  graph.py           – distance histogram, graph construction, graph stats
  graph_clustering.py – Louvain clustering and modularity
  recommend.py       – simple recommendation based on HNSW + Louvain
  pipeline.py        - old code

notebooks/
  project_demo.ipynb – main notebook that calls the functions above and
                       visualises the results

data/
  arxiv-metadata-oai-snapshot.json  (not included; must be downloaded separately) from https://www.kaggle.com/datasets/Cornell-University/arxiv/data

outputs/
  eda/
  embeddings/
  hnsw/
  graphs/
  louvain/

Environment
-----------

A conda/micromamba environment is provided in:
  environment.yml