import networkx as nx
import numpy as np
import scipy.sparse as sp
import itertools
import graphblas as gb
# from numba import njit
# import chain

def compute_overlap_matrix_fast(A, divide_by_deg=True, pow=1.0):
    # Convert to GraphBLAS
    A_g = gb.io.from_scipy_sparse(A)

    # ---- Degree vector (deg[i] = degree(i) + 1) ----
    deg_g = A_g.reduce_rowwise().new()
    deg_g = (deg_g + 1).new()    # add 1 for self-inclusion

    # ---- Common neighbors only where edges exist ----
    # GraphBLAS masked multiplication:
    # A2_masked[i,j] = sum_k A[i,k] * A[k,j]  only for (i,j) in A
    A2_g = (A_g @ A_g).new(mask=A_g.S)

    # Add the +2 self-inclusion: (you had A2 + 2*A)
    A2_g = (A2_g + (A_g * 2)).new()

    # ---- Convert to COO for degree normalization ----
    cn = gb.io.to_scipy_sparse(A2_g).tocoo()
    row, col = cn.row, cn.col

    if divide_by_deg:
        # degree normalization: min(deg[i], deg[j])
        deg = deg_g.to_dense()
        edge_min = np.minimum(deg[row], deg[col])

        # ratio = cn / min_degree
        ratio_data = cn.data / edge_min
        # replace common_neighbors == 0 → ratio = 0
        ratio_data[cn.data == 0] = 0
    else:
        ratio_data = cn.data

    ratio_data = ratio_data ** pow

    # return final sparse matrix
    R = sp.coo_matrix((ratio_data, (row, col)), shape=A.shape)
    return R


def compute_overlap_matrix_sq(A, divide_by_deg=True, use_adjacency_mask=True):
    deg = A.sum(axis=1).A.ravel() + 1  # vertex is included in its own neighbours set 

    A2 = A @ A + 2 * A  # both vertices are included in neighbours intersection

    cn = A2.tocoo()
    if use_adjacency_mask:
        cn = (A2.multiply(A)).tocoo()   # common neighbors only where edges exist

    row, col = cn.row, cn.col
    if divide_by_deg:
        # compute min degree for each edge
        edge_min = np.minimum(deg[row], deg[col])

        ratio_data = cn.data / edge_min
        # replace nan with 0 (common neighbors == 0)
        ratio_data[cn.data == 0] = 0
    else:
        ratio_data = cn.data

    R = sp.coo_matrix((ratio_data, (row, col)), shape=A.shape)
    return R

def compute_overlap_matrix(adj_matrix, inter_param=None):
    n = adj_matrix.shape[0]

    data = []
    row_indices = []
    col_indices = []

    for u in range(n):
        N_u = set(adj_matrix[[u]].nonzero()[1]).union({u})
        for v in N_u:
            if v <= u:
                continue
            N_v = set(adj_matrix[[v]].nonzero()[1]).union({v})
            intersection_size = len(N_u & N_v)
            union_size = min(len(N_u),  len(N_v)) # change to jaccard 


            coefficient = (intersection_size / union_size)
            if inter_param is not None:
                coefficient = coefficient ** inter_param

            data.append(coefficient)
            row_indices.append(u)
            col_indices.append(v)

            data.append(coefficient)
            row_indices.append(v)
            col_indices.append(u)

    overlap_matrix = sp.coo_matrix((data, (row_indices, col_indices)), shape=(n, n))
    return overlap_matrix


def compute_overlap_overlap_matrix(adj_matrix, inter_param=None):
    n = adj_matrix.shape[0]
    adj_csr = adj_matrix.tocsr()
    a_data = adj_csr.data
    a_indices = adj_csr.indices
    a_indptr = adj_csr.indptr
    a_indptr  = a_indptr.astype(np.int64)
    a_indices = a_indices.astype(np.int64)
    a_data = a_data.astype(np.float32)

    edge_set = set(zip(*np.nonzero(adj_matrix)))

    data = []
    row_indices = []
    col_indices = []

    for u in range(n):
        N_u = set(adj_matrix[[u]].nonzero()[1]).union({u})
        for v in N_u:
            if v <= u:
                continue
            N_v = set(adj_matrix[[v]].nonzero()[1]).union({v})
            N_both = N_u & N_v
            intersection_size = len(N_both)
            tranzitive_pairs_num = 1

            # for x, y in N_both:
            nodes = np.array(list(N_both), dtype=np.int64)



            
            tranzitive_pairs_num += count_edges(a_indptr, a_indices, nodes)
            # union_size = min(len(N_u),  len(N_v)) # change to jaccard 


            coefficient = tranzitive_pairs_num

            if inter_param is not None:
                coefficient = coefficient ** inter_param

            data.append(coefficient)
            row_indices.append(u)
            col_indices.append(v)

            data.append(coefficient)
            row_indices.append(v)
            col_indices.append(u)

    overlap_matrix = sp.coo_matrix((data, (row_indices, col_indices)), shape=(n, n))
    return overlap_matrix
