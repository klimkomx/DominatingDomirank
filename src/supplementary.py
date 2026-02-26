import networkx as nx
import numpy as np
import scipy.sparse as sp

def relabel_nodes(G, yield_map=False):
    '''
    Relabels the nodes to be from 0 to len(G).
    Yield_map returns an extra output as a dict in case you want to save the hash-map to retrieve node-id.
    '''
    if yield_map:
        nodes = dict(zip(range(len(G)), G.nodes()))
        G = nx.relabel_nodes(G, dict(zip(G.nodes(), range(len(G)))))
        return G, nodes
    else:
        G = nx.relabel_nodes(G, dict(zip(G.nodes(), range(len(G)))))
        return G


def connect(G, graph):
    # get number of nodes and edges
    graph_parameters = graph.readline().split()
    vertices_number = int(graph_parameters[0],base=10)
    edges_number = int(graph_parameters[1],base=10)
    print(vertices_number)
    print(edges_number)
    # add nodes to networkx graph
    for i in range(vertices_number):
        G.add_node(i+1, contracted = False,imp=0,level=0,contr_neighbours=0)
    # add edges to networkx graph
    edge = graph.readline()
    while edge:
        edge_parameters = edge.split()
        source_node = int(edge_parameters[0], base=10)
        target_node = int(edge_parameters[1], base=10)
        edge_weight = 1
        found_exist = False
        for i in G[source_node]:
            # already store the edge with different weight, choose the min weight
            if i == target_node:
                G[source_node][target_node]['weight'] = 1
                found_exist = True
                break
        if not found_exist:
            G.add_edge(source_node,target_node,weight=edge_weight)
        edge = graph.readline()

def get_largest_component(G, strong=False):
    '''
    Get the largest component of a graph, either from scipy.sparse or from networkx.Graph datatype.
    The argument changes whether or not you want to find the strong or weak connected components of the graph.
    '''
    if isinstance(G, nx.Graph):  # check if it is a networkx Graph
        if nx.is_directed(G):
            if strong:
                GMask = max(nx.strongly_connected_components(G), key=len)
            else:
                GMask = max(nx.weakly_connected_components(G), key=len)
        else:
            GMask = max(nx.connected_components(G), key=len)
        G = G.subgraph(GMask)
    else:
        raise TypeError('You must input a networkx.Graph Data-Type')
    return G

def get_component_size(G, strong=False):
    '''
    Get the largest component size of a graph, either from scipy.sparse or from networkx.Graph datatype.
    The argument changes whether or not you want to find the strong or weak connected components of the graph.
    '''
    if isinstance(G, nx.Graph):  # check if it is a networkx Graph
        if nx.is_directed(G):
            if strong:
                GMask = max(nx.strongly_connected_components(G), key=len)
            else:
                GMask = max(nx.weakly_connected_components(G), key=len)
        else:
            GMask = max(nx.connected_components(G), key=len)
        return len(GMask)
    elif isinstance(G, (sp.sparse.csr_matrix, sp.sparse.csr_array)):
        if strong:
            connection_type = 'strong'
        else:
            connection_type = 'weak'
        _, labels = sp.sparse.csgraph.connected_components(G, directed=True, connection=connection_type)
        return np.bincount(labels).max()
    else:
        raise TypeError('You must input a networkx.Graph Data-Type or scipy.sparse.csr array')

def get_link_size(G):
    '''
    Get the number of links in the graph.
    '''
    if isinstance(G, nx.Graph):  # check if it is a networkx Graph
        return len(G.edges())
    elif isinstance(G, (sp.sparse.csr_matrix, sp.sparse.csr_array)):
        return G.sum()
    else:
        raise TypeError('You must input a networkx.Graph Data-Type or scipy.sparse.csr array')

def remove_node(G, removedNode):
    '''
    Removes the node from the graph by removing it from a networkx.Graph type, or zeroing the edges in array form.
    '''
    if isinstance(G, nx.Graph):  # check if it is a networkx Graph
        if isinstance(removedNode, int):
            G.remove_node(removedNode)
        else:
            for node in removedNode:
                G.remove_node(node)  # remove node in graph form
        return G
    elif isinstance(G, (sp.sparse.csr_matrix, sp.sparse.csr_array)):
        diag = sp.sparse.eye(G.shape[0], format='csr')
        diag[removedNode, removedNode] = 0  # set the rows and columns that are equal to zero in the sparse array
        G = diag @ G
        return G @ diag
