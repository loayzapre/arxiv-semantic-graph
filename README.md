ArXiv Semantic Graph
====================

This project builds a semantic similarity graph of arXiv papers using text
embeddings. The goal is to explore relationships between papers, detect
communities, and recommend similar papers.

Pipeline (Simple Overview)
--------------------------

1. EDA (Exploratory Data Analysis)
   Basic statistics of the dataset (years, abstract length, data quality).

2. Embeddings (Universal Sentence Encoder)
   Abstracts → 512-dimensional vectors. Stored in shards.

3. HNSW Index (Approximate kNN)
   A fast nearest-neighbour search structure built on top of the embeddings.

4. Tau Selection (Global Distance Threshold)
   A histogram of kNN distances is used to choose a global threshold τ that
   defines when two papers are considered semantically similar.

5. Graph Construction
   Using τ, a graph is created where nodes = papers and edges = semantic
   similarities.

6. Graph Clustering (Louvain)
   Communities reveal thematic structures in the corpus.

7. Recommendation Demo
   Select a random paper and find its closest semantic neighbours.

Project Structure
-----------------

src/
  arxiv_semantic_graph/
    eda.py
    embeddings.py
    hnsw_index.py
    tau_selection.py
    graph.py
    graph_clustering.py
    recommend.py

notebooks/
  project_demo.ipynb

data/
outputs/

How to Run Each Step
--------------------

EDA:
    python -m src.arxiv_semantic_graph.eda --file data/arxiv.json --out outputs/eda

Embeddings:
    python -m src.arxiv_semantic_graph.embeddings --file data/arxiv.json --out outputs/embeddings

HNSW:
    python -m src.arxiv_semantic_graph.hnsw_index --emb-dir outputs/embeddings

Graph (choose τ):
    python -m src.arxiv_semantic_graph.graph --emb-dir outputs/embeddings --out-dir outputs/graphs --tau 0.32

Notebook
--------

The notebook in notebooks/project_demo.ipynb loads the results, shows plots,
computes communities and shows recommendations.

Requirements
-----------

The conda environment file is included as:
    environment.yml

License
-------

MIT
