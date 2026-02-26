import numpy as np
import igraph as ig
import tqdm
from time import time

def select_top_k(ranking, k):
    """
    Return indices of the top-k nodes according to the provided ranking.
    Higher scores are considered better.
    """
    ranking = np.asarray(ranking, dtype=float)
    n = ranking.shape[0]
    if k <= 0:
        raise ValueError("k must be positive")
    if k > n:
        k = n

    indices = np.arange(n)
    sorted_idx = indices[np.argsort(-ranking, kind="mergesort")]
    return sorted_idx[:k]


def _sum_distances_to_set(graph: ig.Graph, selected_nodes):
    """
    Compute, for each node in the graph, the distance to the closest node
    in selected_nodes. Return the sum of these distances and the number of
    unreachable nodes.
    """
    if len(selected_nodes) == 0:
        return np.inf, graph.vcount()

    # shortest_paths returns matrix (n x len(selected_nodes))
    distances = np.array(
        graph.distances(source=selected_nodes, mode="all"), dtype=float
    )
    min_dist = distances.min(axis=0)


    total_distance = float(min_dist.sum())

    return total_distance, 0


def k_medians_test(functions, graph: ig.Graph, k, verbose=True):
    """
    For each centrality function, pick the top-k nodes and compute the sum
    of distances from every node to the closest selected node.

    Parameters
    ----------
    functions : list
        List of tuples describing the centrality to evaluate. Supported forms:
        - (name, func) or (name, func, skip_check_func): `func` is a callable that
          takes (igraph.Graph, k) and returns an iterable of node indices of
          length >= k to be used directly as the selected set. When a skip_check_func 
          is provided, the graph is skipped if skip_check_func returns True
    graph : igraph.Graph
        Graph on which to evaluate the centralities.
    k : int
        Number of top nodes to select as the covering/median set.

    Returns
    -------
    list of dict
        Each dict contains:
        {
            "name": centrality name,
            "top_nodes": array of selected node indices,
            "total_distance": total distance to nearest selected node,
            "mean_distance": total_distance / n (np.inf if unreachable nodes),
            "unreachable_nodes": count of nodes without a path to the set,
        }
    """
    if not isinstance(graph, ig.Graph):
        raise TypeError("graph must be an igraph.Graph instance")

    n = graph.vcount()
    if n == 0:
        raise ValueError("Graph has no nodes")

    results = []

    for func_tuple in tqdm.tqdm(functions, disable=verbose):
        if len(func_tuple) == 2:
            func_name, func_or_nodes = func_tuple
            skip_please = None
        elif len(func_tuple) == 3:
            func_name, func_or_nodes, skip_please = func_tuple
        else:
            raise ValueError(f"Invalid function tuple: {func_tuple}")

        try:
            if not callable(func_or_nodes):
                raise ValueError(
                    f"{func_name}: expected a callable (graph, k) -> nodes"
                )

            # Skip large/unsuitable graphs if requested
            if skip_please is not None and skip_please(graph, func_name):
                    continue

            if verbose:
                print(f"Computing {func_name}...")

            start = time()
            nodes = np.asarray(func_or_nodes(graph, k), dtype=int).ravel()
            end = time()
            if nodes.size < k:
                raise ValueError(
                    f"{func_name}: expected at least {k} nodes, got {nodes.size}"
                )
            if np.any(nodes < 0) or np.any(nodes >= n):
                raise ValueError(
                    f"{func_name}: node indices must be within [0, {n-1}]"
                )

            # Deduplicate while preserving order, then trim to k.
            _, unique_idx = np.unique(nodes, return_index=True)
            nodes_unique = nodes[np.sort(unique_idx)]
            top_nodes = np.asarray(nodes_unique[:k], dtype=int)

            total_distance, unreachable = _sum_distances_to_set(graph, top_nodes)
            if np.isfinite(total_distance):
                mean_distance = total_distance / n
            else:
                mean_distance = np.inf

            result = {
                "func_name": func_name,
                "top_nodes": top_nodes,
                "total_distance": total_distance,
                "metric": mean_distance,
                "unreachable_nodes": unreachable,
                "runtime": end - start, 
            }
            results.append(result)

            if unreachable:
                if verbose:
                    print(
                        f"✓ {func_name}: total distance is ∞ "
                        f"(unreachable nodes: {unreachable})"
                    )
            else:
                if verbose:
                    print(
                        f"✓ {func_name}: total distance = {total_distance:.4f}, "
                        f"mean distance = {mean_distance:.4f}"
                    )
        except Exception as exc:
            print(f"✗ Error computing {func_name}: {exc}")

    return results


def batch_k_medians_test(functions, graphs, k, graph_names=None, verbose=True):
    """
    Run k-medians test on multiple graphs with multiple centrality functions.
    
    Parameters
    ----------
    functions : list of tuples
        List of tuples (name, func) or (name, func, skip_check_func). The callable
        must take an igraph.Graph and return a ranking array/list.
    graphs : list of igraph.Graph
        List of graphs to test.
    k : int
        Number of top nodes to select as the covering/median set.
    graph_names : list of str, optional
        Names for each graph. If None, indices are used.
        
    Returns
    -------
    dict
        Dictionary where keys are graph names and values are lists of result
        dictionaries (same format as k_medians_test returns).
    """
    if graph_names is None:
        graph_names = [f"Graph_{i}" for i in range(len(graphs))]
    
    # if len(graphs) != len(graph_names):
    #     raise ValueError(
    #         f"Number of graphs ({len(graphs)}) != number of names ({len(graph_names)})"
    #     )
    
    results = {}
    
    for graph, graph_name in zip(graphs, graph_names):
        if verbose:
            print(f"\n=== K-medians test: {graph_name} ===")
            print(f"Number of nodes: {graph.vcount()}")
            print(f"Number of edges: {graph.ecount()}")
            print(f"k = {k}")
        
        graph_results = k_medians_test(functions, graph, k, verbose=verbose)
        results[graph_name] = graph_results
        
        if graph_results:
            if verbose:
                print(f"\nSummary for {graph_name}:")
            for res in graph_results:
                if res["unreachable_nodes"] > 0:
                    if verbose:
                        print(
                            f"  {res['name']}: total distance = ∞ "
                            f"(unreachable: {res['unreachable_nodes']})"
                        )
                else:
                    if verbose:
                        print(
                            f"  {res['name']}: total = {res['total_distance']:.4f}, "
                            f"mean = {res['mean_distance']:.4f}"
                        )
        else:
            print(f"Failed to compute any results for {graph_name}")
    
    return results


