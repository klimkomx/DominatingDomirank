import numpy as np
from tqdm.auto import tqdm
import scipy

def get_lmax(A, tol=1e-9):
    """Finds maximum eigval using power method"""
    assert A.shape[0] == A.shape[1]
    n = A.shape[0]

    np.random.seed(228)
    x = np.random.randn(n).astype(np.float32)
    x /= np.linalg.norm(x)

    lmax = lmax_prev = -np.inf

    # By Jensen's ineq, \|Ax^{k+1}\| >= \|Ax^{k}\|, \|x^i\| = 1
    while np.isinf(lmax) or lmax - lmax_prev > tol:
        lmax_prev = lmax
        x = A @ x
        if len(x.shape) > 1:
            x = x.A[0]  # if A is scipy sparse, then x is matrix
        lmax = np.linalg.norm(x)
        x /= lmax

    return lmax

def get_lmax_lmin(A, tol=1e-6):
    """Finds maximum and minimum eigvals using power method"""
    assert A.shape[0] == A.shape[1]
    n = A.shape[0]

    lmax = get_lmax(A, tol)
    L = lmax * (1 + 0.1 * np.sign(lmax))  # L > lmax
    I = scipy.sparse.identity(n)
    lmin = L - get_lmax(L * I - A, tol)  

    return lmax, lmin


def sqn(x):
    return (x * x).sum()

def calc_domirank(A, centrality, sigma=None, sigma_normed=None, g0=None, max_iters=1000, tol=1e-5, use_cg=True, use_orig_crit=True, callback=None):
    """
    - sigma: absolute value of sigma (will be prioritezid if sigma_normed is also provided)
    - sigma_normed: relative value of sigma: sigma_normed \in [0,1) corresponds to sigma \in [0, sigma_max). 
    - use_orig_crit: use same stopiing criteria and same tolerance scale as original domirank
    Same as `param` in domirank_ranking_igraph
    """

    assert sigma is not None or sigma_normed is not None, "provide sigma or sigma_normed"

    if sigma is None:
        lmax, lmin = get_lmax_lmin(A)
        sigma_max = - 1 / lmin
        sigma = sigma_normed * sigma_max
        beta = 1 / (1 + lmax * sigma)  # beta = stepsize in gd, Lipschitz constant of grad <= 1 + lambda_max * sigma 
    else:
        beta = 1

    centrality = centrality.astype(np.float32)
    alpha = sigma * beta

    if g0 is None:
        g0 = np.zeros(A.shape[1], dtype=np.float32)

    g_prev = g = g0
    
    M = alpha * A + beta * scipy.sparse.identity(A.shape[0], dtype=np.float32)
    b = - alpha * centrality

    grad_curr = np.zeros(g0.size, dtype=np.float32)

    if use_orig_crit:
        tol *= A.shape[0] 
   
    # for i in tqdm(range(max_iters)): 
    for i in range(max_iters): 
        if use_cg:
            grad_prev = grad_curr
            grad_curr = M @ g + b
            m = sqn(grad_curr) / sqn(grad_prev) if i > 0 else 0 
            p = grad_curr - m * (g - g_prev)
            h = sqn(grad_curr) / (p @ M @ p) 
            g_prev = g
            g = g - h * p
            diff = grad_curr / beta
        else: 
            diff = g - sigma * (centrality - A @ g)
            g = g - beta * diff

        conv_metric = np.linalg.norm(diff, ord=1) if use_orig_crit else np.linalg.norm(diff)
        if callback is not None:
            callback(conv_metric)
        if conv_metric < tol:
            break

    if i == max_iters - 1:
        print(f"Warning: reached {max_iters=}, {conv_metric=}, {tol=}") 
    
    return g
