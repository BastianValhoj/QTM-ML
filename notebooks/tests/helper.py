import numpy as np
import scipy.linalg as lin
import scipy.sparse as spa
import scicpy.sparse.linalg as spla 
from tqdm.auto import tqdm

# @njit
def _infer_solver(M):
    if spa.isspmatrix(M):
        lu = spla.splu(M)
        solver = lambda e: lu.solve(e)
    else:
        lu, piv = lin.lu_factor(M)
        solver = lambda e: lin.lu_solve((lu, piv), e)
    return solver

def diag_inv(M):
    K = len(M)
    diag = np.zeros(K, dtype=complex)
    
    solver = _infer_solver(M)
    for i, idx in enumerate(tqdm(np.arange(K), desc="Sparse diagonal solves", leave=False)):
        ei = np.zeros(K)
        ei[idx] = 1.0
        xi = solver(ei)
        diag[i] = xi[idx]
    return diag