import numpy as np
import networkx as nx
import scipy.sparse as sps


def domirank(G, analytical=True, sigma=-1, dt=0.1, epsilon=1e-5, maxIter=1000, checkStep=10):
    '''
    G is the input graph as a (preferably) sparse array.
    This solves the dynamical equation presented in the Paper: "DomiRank Centrality: revealing structural fragility of
    complex networks via node dominance" and yields the following output: bool, DomiRankCentrality
    Here, sigma needs to be chosen a priori.
    dt determines the step size, usually, 0.1 is sufficiently fine for most networks.
    maxIter is the depth that you are searching with in case you don't converge or diverge before that.
    checkStep is the amount of steps that you go before checking if you have converged or diverged.
    This algorithm scales with O(m) where m is the links in your sparse array.
    '''
    if isinstance(G, nx.Graph):  # check if it is a networkx Graph
        G = nx.to_scipy_sparse_matrix(G)  # convert to scipy sparse if it is a graph
    else:
        G = G.copy()

    if not analytical:
        if sigma == -1:
            sigma, _ = optimal_sigma(G, analytical=False, dt=dt, epsilon=epsilon, maxIter=maxIter, checkStep=checkStep)
        pGAdj = sigma * G.astype(np.float32)
        Psi = np.ones(pGAdj.shape[0]).astype(np.float32) / pGAdj.shape[0]
        maxVals = np.zeros(int(maxIter / checkStep)).astype(np.float32)
        dt = np.float32(dt)
        j = 0
        boundary = epsilon * pGAdj.shape[0] * dt
        for i in range(maxIter):
            tempVal = ((pGAdj @ (1 - Psi)) - Psi) * dt
            Psi += tempVal.real
            if i % checkStep == 0:
                if np.abs(tempVal).sum() < boundary:
                    break
                maxVals[j] = tempVal.max()
                if i == 0:
                    initialChange = maxVals[j]
                if j > 0:
                    if maxVals[j] > maxVals[j - 1] and maxVals[j - 1] > maxVals[j - 2]:
                        return False, Psi
                j += 1
        return True, Psi
    else:
        if sigma == -1:
            sigma = optimal_sigma(G, analytical=True, dt=dt, epsilon=epsilon, maxIter=maxIter, checkStep=checkStep)
        Psi = sps.linalg.spsolve(sigma * G + sps.identity(G.shape[0]), sigma * G.sum(axis=-1))
        return True, Psi

def find_eigenvalue(G, minVal=0, maxVal=1, maxDepth=100, dt=0.1, epsilon=1e-5, maxIter=100, checkStep=10):
    '''
    G: is the input graph as a sparse array.
    Finds the largest negative eigenvalue of an adjacency matrix using the DomiRank algorithm.
    Increase maxDepth for increased accuracy.
    Increase maxIter if DomiRank doesn't start diverging within 100 iterations.
    Decrease checkStep for increased error-finding for the values of sigma that are too large.
    '''
    x = (minVal + maxVal) / G.sum(axis=-1).max()
    minValStored = 0
    for i in range(maxDepth):
        if maxVal - minVal < epsilon:
            break
        if domirank(G, False, x, dt, epsilon, maxIter, checkStep)[0]:
            minVal = x
            x = (minVal + maxVal) / 2
            minValStored = minVal
        else:
            maxVal = (x + maxVal) / 2
            x = (minVal + maxVal) / 2
        # if minVal == 0:
            # print(f'Current Interval : [-inf, -{1 / maxVal}]')
        # else:
            # print(f'Current Interval : [-{1 / minVal}, -{1 / maxVal}]')
    finalVal = (maxVal + minVal) / 2
    return -1 / finalVal

# def process_iteration(q, i, analytical, sigma, spArray, maxIter, checkStep, dt, epsilon, sampling):
#     tf, domiDist = domirank(spArray, analytical=analytical, sigma=sigma, dt=dt, epsilon=epsilon, maxIter=maxIter, checkStep=checkStep)
#     domiAttack = generate_attack(domiDist)
#     ourTempAttack, __ = network_attack_sampled(spArray, domiAttack, sampling=sampling)
#     finalErrors = ourTempAttack.sum()
#     q.put(finalErrors)

# def optimal_sigma(spArray, analytical=True, endVal=0, startval=0.000001, iterationNo=100, dt=0.1, epsilon=1e-5, maxIter=100, checkStep=10, maxDepth=100, sampling=0):
#     '''
#     This part finds the optimal sigma by searching the space, here are the novel parameters:
#     spArray: is the input sparse array/matrix for the network.
#     startVal: is the starting value of the space that you want to search.
#     endVal: is the ending value of the space that you want to search (normally it should be the eigenvalue)
#     iterationNo: the number of partitions of the space between lambN that you set

#     return : the function returns the value of sigma - the numerator of the fraction of (\sigma)/(-1*lambN)
#     '''
#     if endVal == 0:
#         endVal = find_eigenvalue(spArray, maxDepth=maxDepth, dt=dt, epsilon=epsilon, maxIter=maxIter, checkStep=checkStep)

#     import multiprocessing as mp
#     endval = -0.9999 / endVal
#     tempRange = np.arange(startval, endval + (endval - startval) / iterationNo, (endval - startval) / iterationNo)
#     processes = []
#     q = mp.Queue()
#     for i, sigma in enumerate(tempRange):
#         p = mp.Process(target=process_iteration, args=(q, i, analytical, sigma, spArray, maxIter, checkStep, dt, epsilon, sampling))
#         p.start()
#         processes.append(p)

#     results = []
#     for p in processes:
#         p.join()
#         result = q.get()
#         results.append(result)
#     finalErrors = np.array(results)
#     minEig = np.where(finalErrors == finalErrors.min())[0][-1]
#     minEig = tempRange[minEig]
#     return minEig, finalErrors


def unweightedAdj(GAdj):
    return sps.csr_matrix((GAdj.data / GAdj.data, GAdj.indices, GAdj.indptr))

def calculateDomirank(G, param = 0.01, change=unweightedAdj):
    GAdj = nx.to_scipy_sparse_array(G)
    # print(GAdj)
    GAdj = GAdj.astype(float)
    GAdjModified = change(GAdj)
    # print(GAdjModified)
    lambN = find_eigenvalue(GAdjModified, maxIter = 500, dt = 0.1, checkStep = 25)

    _, order = domirank(GAdjModified, analytical = False, sigma = -param * lambN)

    return order


# def calculateRandomWalkDomirank(G, rw_length=4, rw_num=1000, param = 0.01):
#     coefficient_matrix = compute_rw_matrix(G, rw_length, rw_num)
#     GAdj = nx.to_scipy_sparse_array(G)
#     GAdj = GAdj.astype(float)
#     GAdjModified = GAdj.multiply(coefficient_matrix)
#     print(GAdjModified)
#     lambN = find_eigenvalue(GAdjModified, maxIter = 500, dt = 0.1, checkStep = 25)
#     _, order = domirank(GAdjModified, analytical = False, sigma = -param * lambN)
#     return order

def generalized_domirank(A, a, R, r, two_stage=False, epsilon=1e-5, maxIter=1000, checkStep=10):
    n = a.shape[0]
    x = np.ones(n).astype(np.float32)
    maxVals = np.zeros(int(maxIter / checkStep)).astype(np.float32)
    boundary = epsilon * n
    j = 0
    for i in range(maxIter):
        if two_stage:
            delta1 = R@x + r
            x1 = x + delta1
            delta2 = A@x1 + a
            delta = delta1 + delta2
            if np.abs(delta).sum() < boundary:
                break
            x = x1 + delta2
        else:
            delta = (A+R)@x + a + r
            if np.abs(delta).sum() < boundary:
                break
            x += delta
        if i % checkStep == 0:
                if np.abs(delta).sum() < boundary:
                    break
                maxVals[j] = delta.max()
                if i == 0:
                    initialChange = maxVals[j]
                if j > 0:
                    if maxVals[j] > maxVals[j - 1] and maxVals[j - 1] > maxVals[j - 2]:
                        return False, x
                j += 1
    return True, x

def find_eigenvalue_efficient(G):
    vals, vecs = sps.linalg.eigsh(G, k=1, which='SA')
    return vals[0]
