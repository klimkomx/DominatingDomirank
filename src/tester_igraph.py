import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
from . import domirank
import os
import networkx as nx
import tqdm
from time import time
        


def _destruction_igraph(order, _graph):
    """
    Destruction analysis using igraph.
    
    Parameters:
    ----------
    order : list or array
        Ranking values (higher is better)
    _graph : igraph.Graph
        Graph to analyze
        
    Returns:
    ----------
    list : Maximum component sizes after each node removal
    """
    _graph_copy = _graph.copy()

    order = -np.array(order)
    indices = np.arange(len(order))

    sorted_indices = indices[np.argsort(order)]
    sorted_order = order[sorted_indices]

    # Find unique values and where they start in the sorted array
    unique_values, unique_indices = np.unique(sorted_order, return_index=True)

    # Prepare an array to hold shuffled indices
    shuffled_sorted_indices = sorted_indices.copy()

    # Shuffle indices for each unique value by finding begin and end positions
    for i, start_idx in enumerate(unique_indices):
        # Find end index: start of next unique value, or end of array
        end_idx = unique_indices[i + 1] if i + 1 < len(unique_indices) else len(sorted_indices)
        
        # Shuffle the range directly
        shuffled_sorted_indices[start_idx:end_idx] = np.random.permutation(shuffled_sorted_indices[start_idx:end_idx])
    
    nodes_sorted = shuffled_sorted_indices
    max_comp_size = []

    for vertex_id in nodes_sorted:
        # Get connected components using igraph
        components = _graph_copy.components()
        cc_len = components.sizes()  # Get sizes of all components

        max_comp_size.append(max(cc_len) if cc_len else 0)

        # Delete vertex by name (names don't change when vertices are deleted)
        vertex_name = str(vertex_id)
        try:
            vertex = _graph_copy.vs.find(name=vertex_name)
            _graph_copy.delete_vertices(vertex)
        except ValueError:
            # Vertex already deleted (shouldn't happen, but handle gracefully)
            continue

    return max_comp_size


class UnionFind:
    """
    Union-Find (Disjoint Set Union) data structure for tracking connected components.
    """
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n
        self.max_size = 1
    
    def find(self, x):
        """Find the root of x with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        """Union two sets and update max_size."""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return
        
        # Union by size
        if self.size[root_x] < self.size[root_y]:
            root_x, root_y = root_y, root_x
        
        self.parent[root_y] = root_x
        self.size[root_x] += self.size[root_y]
        self.max_size = max(self.max_size, self.size[root_x])


def destruction_igraph_bottom(order, _graph):
    """
    Destruction analysis from bottom: builds graph incrementally by adding vertices
    in order of lowest priority, tracking largest component size.
    
    Parameters:
    ----------
    order : list or array
        Ranking values (higher is better, so lower values = lower priority)
    _graph : igraph.Graph
        Graph to analyze
        
    Returns:
    ----------
    list : Maximum component sizes after each vertex addition (from bottom up)
    """
    n = _graph.vcount()
    
    # Sort vertices by priority (lowest first)
    # Lower order value = lower priority = process first
    order = np.array(order)
    indices = np.arange(n)
    
    sorted_indices = indices[np.argsort(order)]
    sorted_order = order[sorted_indices]
    
    # Find unique values and where they start in the sorted array
    unique_values, unique_indices = np.unique(sorted_order, return_index=True)
    
    # Prepare an array to hold shuffled indices
    shuffled_sorted_indices = sorted_indices.copy()
    
    # Shuffle indices for each unique value by finding begin and end positions
    for i, start_idx in enumerate(unique_indices):
        # Find end index: start of next unique value, or end of array
        end_idx = unique_indices[i + 1] if i + 1 < len(unique_indices) else len(sorted_indices)
        
        # Shuffle the range directly
        shuffled_sorted_indices[start_idx:end_idx] = np.random.permutation(shuffled_sorted_indices[start_idx:end_idx])
    
    # Process vertices in order of lowest priority
    vertices_to_process = shuffled_sorted_indices
    # vertices_to_process = sorted_indices
    
    # Initialize Union-Find: each vertex starts as its own component (size 1)
    uf = UnionFind(n)
    max_comp_size = [1]  # Start with all components of size 1
    
    # Track which vertices have been processed
    processed = set()
    
    # Get adjacency list for efficient edge lookup
    adj_list = {}
    for v in range(n):
        neighbors = _graph.neighbors(v)
        adj_list[v] = neighbors
    
    # Process vertices one by one, starting with lowest priority
    for vertex_id in vertices_to_process:
        processed.add(vertex_id)
        
        # Connect this vertex to its neighbors that have already been processed
        for neighbor in adj_list[vertex_id]:
            if neighbor in processed:
                # Both vertices are processed, so connect their components
                uf.union(vertex_id, neighbor)
        
        # Record the current largest component size
        max_comp_size.append(uf.max_size)
    
    # Reverse the result to match destruction plot format (from large to small)
    return max_comp_size[::-1]


def _destruction_plot_igraph(rankings, _graph):
    """
    Plot destruction analysis using igraph.
    
    Parameters:
    ----------
    rankings : list of tuples
        List of (name, ranking) tuples
    _graph : igraph.Graph
        Graph to analyze
    """

    _graph.vs["name"] = [str(i) for i in range(_graph.vcount())]

    for name, rank in rankings:
        # comp_size = _destruction_igraph(rank, _graph)
        comp_size = destruction_igraph_bottom(rank, _graph)
        print(np.mean(comp_size))
        fig = plt.figure(10)
        ourRangeNew = np.linspace(0, 1, len(comp_size))
        plt.plot(ourRangeNew, comp_size, label=name)
        plt.legend()
        plt.xlabel('fraction of nodes removed')
        plt.ylabel('largest connected component')
    plt.show()


def batch_destruction_analysis(functions, graphs, graph_names=None, verbose=True):
    """
    Function for testing various ranking algorithms on multiple graphs using igraph.
    
    Parameters:
    ----------
    functions : list of tuples
        List of tuples (name, function) or (name, function, p), where:
        - name is the function name
        - function is a function that takes an igraph Graph and returns a ranking
        - skip_please (optional) is a check function. Returns True if graph should be skipped for this centrlity 
          (e.g. graph is large and the centrality is betweeness)
    graphs : list of igraph.Graph
        List of igraph graphs for testing
    graph_names : list of str, optional
        Graph names for display. If None, indices are used
        
    Returns:
    ----------
    dict : Dictionary with results for each graph
    """
    if graph_names is None:
        graph_names = [f"Graph_{i}" for i in range(len(graphs))]
    
    results = {}
    
    for graph, graph_name in zip(graphs, graph_names):
        if verbose:
            print(f"\n=== Graph Analysis: {graph_name} ===")
            print(f"Number of nodes: {graph.vcount()}")
            print(f"Number of edges: {graph.ecount()}")
        
        funcs_results = []
        rankings = []
        for func_tuple in tqdm.tqdm(functions, disable=verbose):
            # Handle both (name, func) and (name, func, p) formats
            if len(func_tuple) == 2:
                func_name, func = func_tuple
                skip_please = None
            elif len(func_tuple) == 3:
                func_name, func, skip_please = func_tuple
            else:
                print(f"✗ Invalid function tuple format: {func_tuple}")
                continue
            
            # Skip large/unsuitable graphs if requested
            if skip_please is not None and skip_please(graph, func_name):
                    continue
            
            try:
                if verbose:
                    print(f"Computing {func_name}...")

                start = time()
                ranking = func(graph)
                end = time()
                ranking = np.asarray(ranking, dtype=float)
                mean_destruction = np.mean(destruction_igraph_bottom(ranking, graph))
                funcs_results.append(dict(func_name=func_name, ranking=ranking,
                                   metric=mean_destruction, runtime=end - start))
                rankings.append((func_name, ranking))
                if verbose:
                    print(f"✓ {func_name} successfully computed")
            except Exception as e:
                print(f"✗ Error computing {func_name}: {e}")
                continue

        results[graph_name] = funcs_results
        if verbose and rankings:
            print(f"Building destruction plot for {graph_name}...")
            _destruction_plot_igraph(rankings, graph)
            results[graph_name] = rankings
        # else:
        #     print(f"Failed to compute any rankings for {graph_name}")
        #     results[graph_name] = []
    
    return results


# Example ranking functions for testing using igraph
def degree_centrality_ranking(g):
    """Ranking by degree centrality using igraph"""
    centrality = g.degree()
    # Normalize by (n-1) for undirected graphs, (n-1) for directed
    # n = g.vcount()
    # if n > 1:
        # centrality = [c / (n - 1) if not g.is_directed() else c / (n - 1) for c in centrality]
    return centrality


def betweenness_centrality_ranking(g, cutoff=None):
    """Ranking by betweenness centrality using igraph"""
    if cutoff is None:
        cutoff = min(500, g.vcount())
    centrality = g.betweenness(cutoff=cutoff)
    # Normalize
    n = g.vcount()
    if n > 2:
        if g.is_directed():
            denom = (n - 1) * (n - 2)
        else:
            denom = (n - 1) * (n - 2) / 2
        if denom > 0:
            centrality = [c / denom for c in centrality]
    return centrality


def closeness_centrality_ranking(g):
    """Ranking by closeness centrality using igraph"""
    centrality = g.closeness()
    return centrality


def eigenvector_centrality_ranking(g, max_iter=1000):
    """Ranking by eigenvector centrality using igraph"""
    try:
        centrality = g.eigenvector_centrality(directed=g.is_directed(), scale=True)
        return centrality
    except:
        # Fallback: compute manually using power iteration
        adjacency = np.array(g.get_adjacency().data)
        n = adjacency.shape[0]
        if n == 0:
            return []
        x = np.ones(n) / np.sqrt(n)
        for _ in range(max_iter):
            x_new = adjacency @ x
            norm = np.linalg.norm(x_new)
            if norm == 0:
                break
            x_new = x_new / norm
            if np.linalg.norm(x_new - x) < 1e-6:
                break
            x = x_new
        return list(x)


def pagerank_ranking(g, damping=0.85):
    """Ranking by PageRank using igraph"""
    centrality = g.pagerank(damping=damping)
    return centrality


def louvain_ranking(g):
    """Ranking based on Louvain algorithm using igraph"""
    try:
        # Use igraph's community detection
        communities = g.community_multilevel()
        
        # Get membership for each vertex
        membership = communities.membership
        
        # Count community sizes
        community_sizes = {}
        for vertex_id, comm_id in enumerate(membership):
            if comm_id not in community_sizes:
                community_sizes[comm_id] = 0
            community_sizes[comm_id] += 1
        
        # Create ranking based on community size
        ranking = [community_sizes[membership[i]] for i in range(g.vcount())]
        
        return ranking
    except Exception as e:
        print(f"Error in louvain_ranking: {e}")
        # Fallback: return uniform ranking
        return [1.0] * g.vcount()


def domirank_ranking(g, param=0.1):
    """Ranking by DomiRank - converts to networkx, computes, returns ranking"""
    # Note: calculateDomirank requires networkx, so we still need conversion here
    G_nx = nx.Graph() if not g.is_directed() else nx.DiGraph()
    G_nx.add_nodes_from(range(g.vcount()))
    G_nx.add_edges_from([(e.source, e.target) for e in g.es])
    # print("ccc")
    # print(len(G_nx))
    return domirank.calculateDomirank(G_nx, param)


def domirank_ranking_igraph(g, param = 0.01):
    GAdj = g.get_adjacency_sparse()
    # print(GAdj)
    GAdj = GAdj.astype(float)
    # print(GAdjModified)
    # lambN = domirank.find_eigenvalue(GAdj, maxIter = 500, dt = 0.1, checkStep = 25)
    lambN = domirank.find_eigenvalue_efficient(GAdj)

    _, order = domirank.domirank(GAdj, analytical = False, sigma = -param / lambN)

    return order

def load_graphs_from_directory(directory_path, file_extensions=['.graphml', '.gml', '.edgelist']):
    """
    Loads all graphs from the specified directory using igraph.
    
    Parameters:
    ----------
    directory_path : str
        Path to directory with graph files
    file_extensions : list of str
        File extensions to load
        
    Returns:
    ----------
    tuple : (graphs, graph_names) - lists of igraph graphs and their names
    """
    graphs = []
    graph_names = []
    
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} not found")
        return graphs, graph_names
    
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        # Check file extension
        if any(filename.lower().endswith(ext) for ext in file_extensions):
            try:
                if filename.lower().endswith('.graphml'):
                    g = ig.Graph.Read_GraphML(file_path)
                elif filename.lower().endswith('.gml'):
                    g = ig.Graph.Read_GML(file_path)
                elif filename.lower().endswith('.edgelist'):
                    g = ig.Graph.Read_Edgelist(file_path, directed=False)
                else:
                    continue
                
                # Convert to undir  ected graph if needed
                if g.is_directed():
                    g = g.as_undirected()
                g.es["weight"] = None
                g = g.simplify()

                comp = g.components()
                
                largest_index = comp.sizes().index(max(comp.sizes()))

                largest_vertices = comp[largest_index]
                g = g.subgraph(largest_vertices)



                # Remove isolated nodes (vertices with degree 0)
                isolated = [v.index for v in g.vs if v.degree() == 0]
                if isolated:
                    g.delete_vertices(isolated)
                
                # Renumber vertices to be consecutive integers starting from 0
                # (igraph already does this, but ensure it)
                g.name = filename
                graphs.append(g)
                graph_names.append(filename)
                print(f"Loaded graph: {filename} ({g.vcount()} nodes, {g.ecount()} edges)", flush=True)
                
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
    
    return graphs, graph_names


def simple_destruction_test(functions, graphs, graph_names=None, save_plots=False, output_dir="plots"):
    """
    Simplified version of the function for quick testing using igraph.
    
    Parameters:
    ----------
    functions : list of tuples
        List of tuples (name, function) or (name, function, p), where:
        - name is the function name
        - function is a function that takes an igraph Graph and returns a ranking
        - p (optional) is the maximum graph size threshold - if graph has more than p nodes,
          this centrality will be skipped and not included in the destruction plot
    graphs : list of igraph.Graph
        List of igraph graphs
    graph_names : list of str, optional
        Graph names
    save_plots : bool
        Whether to save plots to files
    output_dir : str
        Directory for saving plots
    """
    if graph_names is None:
        graph_names = [f"Graph_{i}" for i in range(len(graphs))]
    
    if save_plots and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for graph, graph_name in zip(graphs, graph_names):
        print(f"\n=== {graph_name} ===")
        
        rankings = []
        for func_tuple in functions:
            # Handle both (name, func) and (name, func, p) formats
            if len(func_tuple) == 2:
                func_name, func = func_tuple
                p = None
            elif len(func_tuple) == 3:
                func_name, func, p = func_tuple
            else:
                print(f"✗ Invalid function tuple format: {func_tuple}")
                continue
            
            # Check graph size threshold - skip if graph is larger than p
            if p is not None and graph.vcount() > p:
                print(f"⊘ Skipping {func_name} (graph size {graph.vcount()} > threshold {p})")
                continue
            
            try:
                ranking = func(graph)
                rankings.append((func_name, ranking))
                print(f"✓ {func_name}")
            except Exception as e:
                print(f"✗ {func_name}: {e}")
        
        if rankings:
            # Use igraph-based destruction plot
            _destruction_plot_igraph(rankings, graph)
            
            if save_plots:
                plt.savefig(os.path.join(output_dir, f"{graph_name.replace(' ', '_')}.png"), 
                           dpi=300, bbox_inches='tight')
                print(f"Plot saved: {output_dir}/{graph_name.replace(' ', '_')}.png")
            plt.show()


# Example usage
if __name__ == "__main__":
    print("=== Example 1: Synthetic graphs ===")
    
    # Create test graphs using igraph
    graphs = []
    graph_names = []
    
    # Erdős-Rényi graph
    g1 = ig.Graph.Erdos_Renyi(n=100, m=int(100 * 99 * 0.1 / 2))
    graphs.append(g1)
    graph_names.append("Erdős-Rényi (n=100, p=0.1)")
    
    # Barabási-Albert graph
    g2 = ig.Graph.Barabasi(n=100, m=5, directed=False)
    graphs.append(g2)
    graph_names.append("Barabási-Albert (n=100, m=5)")
    
    # Watts-Strogatz graph
    g3 = ig.Graph.Watts_Strogatz(dim=1, size=100, nei=3, p=0.3)
    graphs.append(g3)
    graph_names.append("Watts-Strogatz (n=100, k=6, p=0.3)")
    
    # Define functions for testing
    functions = [
        ("Degree Centrality", degree_centrality_ranking),
        ("Betweenness Centrality", betweenness_centrality_ranking),
        ("Closeness Centrality", closeness_centrality_ranking),
        ("Eigenvector Centrality", eigenvector_centrality_ranking),
        ("PageRank", pagerank_ranking),
        ("DomiRank", domirank_ranking),
        ("Louvain", louvain_ranking),
    ]
    
    # Run analysis
    results = batch_destruction_analysis(functions, graphs, graph_names)
    
    print("\n=== Analysis completed ===")
    for graph_name, rankings in results.items():
        print(f"{graph_name}: {len(rankings)} successful rankings")
    
    print("\n=== Example 2: Loading graphs from files ===")
    
    # Try to load graphs from data directory
    data_graphs, data_names = load_graphs_from_directory("data")
    
    if data_graphs:
        print(f"Loaded {len(data_graphs)} graphs from data directory")
        
        # Use only basic functions for real graphs (they may be large)
        basic_functions = [
            ("Degree Centrality", degree_centrality_ranking),
            ("Betweenness Centrality", betweenness_centrality_ranking),
            ("PageRank", pagerank_ranking),
            ("DomiRank", domirank_ranking),
        ]
        
        # Run simplified analysis
        simple_destruction_test(basic_functions, data_graphs, data_names, 
                               save_plots=True, output_dir="destruction_plots")
    else:
        print("No graphs found in data directory")
    
    print("\n=== Example 3: Quick test with saving ===")
    
    # Create a small graph for quick test
    # igraph doesn't have karate club built-in, create a simple test graph
    small_graph = ig.Graph.Famous("Zachary")
    if small_graph is None:
        # Fallback: create a small random graph
        small_graph = ig.Graph.Erdos_Renyi(n=34, m=78)
    
    small_graphs = [small_graph]
    small_names = ["Zachary Karate Club"]
    
    # Only fast functions
    quick_functions = [
        ("Degree Centrality", degree_centrality_ranking),
        ("PageRank", pagerank_ranking),
        ("DomiRank", domirank_ranking),
    ]
    
    simple_destruction_test(quick_functions, small_graphs, small_names, 
                           save_plots=True, output_dir="quick_test_plots")

