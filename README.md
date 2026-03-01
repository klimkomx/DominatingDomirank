## DominatingDomirank

This repository accompanies the article **"Dominating Domirank on Dominating Set (and k-median problem)"** and contains the reference code and experiments used around Domirank for dominating set–style tasks (and related k-median experiments).

### Repository structure

- **`article_experiments.ipynb`**: Main notebook with experiments/demos used for the article.
- **`src/`**: Python sources (Domirank implementation and supporting utilities/experiments).
  - **`domirank.py`**: Core Domirank implementation.
  - **`domirank_cg.py`**: Alternative/extended Domirank variant (CG-related implementation).
  - **`reweights.py`**: Reweighting/utility routines used by the algorithms.
  - **`supplementary.py`**: Supplementary helpers used in experiments.
  - **`tester.py`**, **`tester_igraph.py`**: Experiment/test runners (including an `igraph`-based runner).
  - **`k_medians_test.py`**: k-medians-related experiments/tests.
  - **`vcover_test.py`**: Vertex cover–related experiments/tests.
- **`data/`**: Example input graphs in `.gml` format used by the experiments.
  - **`instruction_for_large_graphs.md`**: Notes/instructions for working with large graphs.