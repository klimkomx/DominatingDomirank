import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
import tqdm
from time import time


def _compute_vertex_cover_curve(order, graph: ig.Graph):
    """
    Given a ranking (higher is better) and an igraph graph, compute the
    vertex-cover-style curve:

    - Covering set S_k consists of top-k nodes by ranking.
    - A node v is considered covered if v is in S_k or it has at least one
      neighbor in S_k.
    - For each k from 0 to n, we record the number of uncovered nodes.

    Returns
    -------
    uncovered_counts : np.ndarray of shape (n + 1,)
        uncovered_counts[k] = number of uncovered nodes when |S_k| = k.
    """
    n = graph.vcount()
    if n == 0:
        return np.array([0], dtype=int)

    order = np.asarray(order, dtype=float)
    if order.shape[0] != n:
        raise ValueError(f"Ranking length {order.shape[0]} != number of nodes {n}")

    # Sort vertices by descending score (higher = better)
    # For stability, break ties by vertex index.
    indices = np.arange(n)
    # Negate order so argsort gives descending ranking
    sorted_indices = indices[np.argsort(-order, kind="mergesort")]

    # Build adjacency list once
    neighbors = [graph.neighbors(v) for v in range(n)]

    covered = np.zeros(n, dtype=bool)
    uncovered_counts = np.empty(n + 1, dtype=int)

    # k = 0 (empty covering set)
    uncovered_counts[0] = n

    for k, v in enumerate(sorted_indices, start=1):
        new_covered = 0
        if not covered[v]:
            covered[v] = True
            new_covered += 1
        for u in neighbors[v]:
            if not covered[u]:
                covered[u] = True
                new_covered += 1
        # uncovered_counts[k] = int(n - covered.sum())
        uncovered_counts[k] = uncovered_counts[k-1] - new_covered

    return uncovered_counts


def batch_vertex_cover_analysis(functions, graphs, graph_names=None, verbose=True):
    """
    Analyze vertex-cover-like behavior of multiple centrality functions on
    multiple igraph graphs.

    Parameters
    ----------
    functions : list of tuples
        List of tuples (name, function) or (name, function, p), where:
        - name is the function name (for legend/printing)
        - function is a callable taking an igraph.Graph and returning a ranking
          (list/array of scores, higher = more central)
        - skip_please (optinal) is a check function. Returns True if graph should be skipped for this centrlity 
          (e.g. graph is large and the centrality is betweeness)
    graphs : list of igraph.Graph
        Graphs to analyze.
    graph_names : list of str, optional
        Names used for printing and plot titles. If None, indices are used.

    Returns
    -------
    dict
        results[graph_name] = list of (func_name, uncovered_counts, mean_uncovered) tuples,
        where:
        - uncovered_counts is a numpy array of length n + 1
        - mean_uncovered is the mean number of uncovered nodes over all k (discrete area).
    """
    if graph_names is None:
        graph_names = [f"Graph_{i}" for i in range(len(graphs))]

    results = {}

    for graph, graph_name in zip(graphs, graph_names):
        if verbose:
            print(f"\n=== Vertex-cover analysis: {graph_name} ===")
            print(f"Number of nodes: {graph.vcount()}")
            print(f"Number of edges: {graph.ecount()}")

        n = graph.vcount()
        funcs_results = []
        for func_tuple in tqdm.tqdm(functions, disable=verbose):
            # Handle both (name, func) and (name, func, p)
            if len(func_tuple) == 2:
                func_name, func = func_tuple
                skip_please = None
            elif len(func_tuple) == 3:
                func_name, func, skip_please = func_tuple
            else:
                print(f"✗ Invalid function tuple format: {func_tuple}")
                continue

            # Skip bad/large graphs if requested
            if skip_please is not None and skip_please(graph, func_name):
                    continue

            try:
                if verbose:
                    print(f"Computing {func_name}...")
                start = time()
                ranking = func(graph)
                end = time()
                ranking = np.asarray(ranking, dtype=float)
                if ranking.shape[0] != n:
                    raise ValueError(
                        f"{func_name}: ranking length {ranking.shape[0]} "
                        f"!= number of nodes {n}"
                    )

                uncovered_counts = _compute_vertex_cover_curve(ranking, graph)
                mean_uncovered = float(uncovered_counts.mean())
                funcs_results.append(dict(func_name=func_name, uncovered_counts=uncovered_counts, 
                                   metric=mean_uncovered, runtime=end - start))
                if verbose:
                    print(
                        f"✓ {func_name} successfully computed "
                        f"(mean uncovered nodes: {mean_uncovered:.4f})"
                    )
            except Exception as e:
                print(f"✗ Error computing {func_name}: {e}")
                continue
        if not verbose:
            results[graph_name] = funcs_results 
            continue
        if curves:
            # Build plot: x-axis = fraction of nodes in covering set (0..1),
            # y-axis = number of uncovered nodes.
            fig = plt.figure(figsize=(8, 6))
            x = np.linspace(0.0, 1.0, n + 1)

            for func_name, uncovered_counts, mean_uncovered in curves:
                plt.plot(x, uncovered_counts, label=f"{func_name} (mean={mean_uncovered:.2f})")

            plt.xlabel("Fraction of nodes in covering set")
            plt.ylabel("Number of uncovered nodes")
            plt.title(f"Vertex-cover curves for {graph_name}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

            results[graph_name] = curves
        else:
            print(f"Failed to compute any curves for {graph_name}")
            results[graph_name] = []

    return results


