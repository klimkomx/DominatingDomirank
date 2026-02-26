import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from .destruction import destruction_plot
from .domirank import calculateDomirank
import itertools
import os
import tqdm

def batch_destruction_analysis(functions, graphs, graph_names=None, verbose=True):
    """
    Function for testing various ranking algorithms on multiple graphs.
    
    Parameters:
    ----------
    functions : list of tuples
        List of tuples (name, function) or (name, function, p), where:
        - name is the function name
        - function is a function that takes a networkx graph and returns a ranking
        - p (optional) is the maximum graph size threshold - if graph has more than p nodes,
          this centrality will be skipped and not included in the destruction plot
    graphs : list of networkx.Graph
        List of graphs for testing
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
            print(f"Number of nodes: {graph.number_of_nodes()}")
            print(f"Number of edges: {graph.number_of_edges()}")
        
        # Prepare list of rankings for this graph
        rankings = []
        
        for func_tuple in tqdm.tqdm(functions, disable=verbose):
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
            if p is not None and graph.number_of_nodes() > p:
                print(f"⊘ Skipping {func_name} (graph size {graph.number_of_nodes()} > threshold {p})")
                continue
            
            try:
                if verbose:
                    print(f"Computing {func_name}...")
                ranking = func(graph)
                rankings.append((func_name, ranking))
                if verbose:
                    print(f"✓ {func_name} successfully computed")
            except Exception as e:
                print(f"✗ Error computing {func_name}: {e}")
                continue
        if not verbose:
            results[graph_name] = rankings
            continue
        if rankings:
            # Call destruction_plot for this graph
            print(f"Building destruction plot for {graph_name}...")
            destruction_plot(rankings, graph)
            results[graph_name] = rankings
        else:
            print(f"Failed to compute any rankings for {graph_name}")
            results[graph_name] = []
    
    return results


# Example ranking functions for testing
def degree_centrality_ranking(G):
    """Ranking by degree centrality"""
    centrality = nx.degree_centrality(G)
    return list(centrality.values())


def betweenness_centrality_ranking(G):
    """Ranking by betweenness centrality"""
    centrality = nx.betweenness_centrality(G, 500)
    return list(centrality.values())


def closeness_centrality_ranking(G):
    """Ranking by closeness centrality"""
    centrality = nx.closeness_centrality(G)
    return list(centrality.values())


def eigenvector_centrality_ranking(G):
    """Ranking by eigenvector centrality"""
    centrality = nx.eigenvector_centrality(G, max_iter=1000)
    return list(centrality.values())


def pagerank_ranking(G):
    """Ranking by PageRank"""
    centrality = nx.pagerank(G)
    return list(centrality.values())


def load_graphs_from_directory(directory_path, file_extensions=['.graphml', '.gml', '.edgelist']):
    """
    Loads all graphs from the specified directory.
    
    Parameters:
    ----------
    directory_path : str
        Path to directory with graph files
    file_extensions : list of str
        File extensions to load
        
    Returns:
    ----------
    tuple : (graphs, graph_names) - lists of graphs and their names
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
                    G = nx.read_graphml(file_path)
                elif filename.lower().endswith('.gml'):
                    G = nx.read_gml(file_path)
                elif filename.lower().endswith('.edgelist'):
                    G = nx.read_edgelist(file_path)
                else:
                    continue
                
                # Convert to undirected graph if needed
                if G.is_directed():
                    G = G.to_undirected()
                G = nx.convert_node_labels_to_integers(G)
                # Remove isolated nodes
                G.remove_nodes_from(list(nx.isolates(G)))
                
                G.name = filename
                graphs.append(G)
                graph_names.append(filename)
                print(f"Loaded graph: {filename} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
                
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
    
    return graphs, graph_names


def simple_destruction_test(functions, graphs, graph_names=None, save_plots=False, output_dir="plots"):
    """
    Simplified version of the function for quick testing.
    
    Parameters:
    ----------
    functions : list of tuples
        List of tuples (name, function) or (name, function, p), where:
        - name is the function name
        - function is a function that takes a networkx graph and returns a ranking
        - p (optional) is the maximum graph size threshold - if graph has more than p nodes,
          this centrality will be skipped and not included in the destruction plot
    graphs : list of networkx.Graph
        List of graphs
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
            if p is not None and graph.number_of_nodes() > p:
                print(f"⊘ Skipping {func_name} (graph size {graph.number_of_nodes()} > threshold {p})")
                continue
            
            try:
                ranking = func(graph)
                rankings.append((func_name, ranking))
                print(f"✓ {func_name}")
            except Exception as e:
                print(f"✗ {func_name}: {e}")
        
        if rankings:
            destruction_plot(rankings, graph)
            
            if save_plots:
                plt.savefig(os.path.join(output_dir, f"{graph_name.replace(' ', '_')}.png"), 
                           dpi=300, bbox_inches='tight')
                print(f"Plot saved: {output_dir}/{graph_name.replace(' ', '_')}.png")
            plt.show()