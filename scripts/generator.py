import numpy as np
import sisl
from sisl import Hamiltonian
from tqdm.auto import tqdm

from scipy import linalg
import scipy.sparse as spa
import scipy.sparse.linalg as spla

#%% helper functions

def dysonEQ(z, S_mat, H_mat, rse, alist):
    
    invG = z*S_mat - H_mat
    RSE = rse.self_energy(z)
    RSE_reordered = RSE[np.ix_(alist, alist)]
    
    invG -= RSE_reordered
    return invG, RSE_reordered

def diag_inv_solver(invG):
    if spa.isspmatrix(invG):
        lu = spla.splu(invG)
        solver = lambda b: lu.solver(b)
    else:
        lu_and_piv = linalg.lu_factor(invG)
        solver = lambda b: linalg.lu_solve(lu_and_piv, b)
    return solver

def invertGF(invG, full_mat=False):
    if full_mat: return linalg.inv(invG)
    
    K, _ = invG.shape
    diagG = np.zeros(K, dtype=complex)
    solver = diag_inv_solver(invG)
    
    for idx in tqdm(range(K), desc="diagonal of inverse"):
        ei = np.zeros(K)
        ei[idx] = 1.0
        xi = solver(ei)
        diagG[idx] = xi[idx]
    return diagG

def calc_ldos(G):
    if G.ndim == 1: diagG = G
    elif G.ndim == 2: diagG = np.diag(G)
    else: raise ValueError(f"Don't know what went wrong, but n dims of Greens is neither 1D or 2D,\n {G.ndim = };\t{G.shape = }")
    return -(1.0/np.pi) * np.imag(diagG)


#%% generate system and LDOS
def systemInit(bond=1.43, t=-2.7):
    graphene = sisl.geom.graphene(bond)
    Ham0 = Hamiltonian(graphene)
    r = (0.1*bond, bond+1e-2)
    t = (0.0, t)
    Ham0.construct([r, t])
    return Ham0

def setup_ham_rse(Ham, tile=4, nk1=100, eta=1e-3j):
    if isinstance(tile, (int, float, np.int16, np.int32, np.int64)): 
        Na = Nb = int(tile)
    elif isinstance(tile, (tuple, list)):
        Na, Nb = tile
    else:
        raise ValueError(f"invalid tile input: {tile} of type : {type(tile)}")
    
    rse = sisl.RealSpaceSE(Ham, 0, 1, (Na, Nb, 1))
    rse.setup(eta=eta, bz=sisl.MonkhorstPack(Ham, [1, nk1, 1]))
    H = Ham.tile(Na, 0).tile(Nb, 1)
    H.set_nsc([1,1,1])
    _, elec_indices = rse.real_space_coupling(ret_indices=True)
    
    all_atoms = np.arange(0, H.na)
    inside_atoms = np.delete(all_atoms, elec_indices)
    alist = np.concatenate([elec_indices, inside_atoms])
    
    H_final = H.sub(alist)
    H_final.reduce()
    
    return H_final, rse, alist, elec_indices

def calculate_spectral_GF(energies, eta, Ham_sub, rse, alist, elist):
    if not isinstance(eta, complex):
        eta = eta*1j # convert real values energy perturbation to complex
    num_elec_atoms = len(elist) # number of atoms in electrode
    num_atoms = len(Ham_sub) # number of atoms in system (device + electrode)
    num_E = len(energies) # number of energy values
    H_mat = Ham_sub.Hk(format="array", dtype=complex) # get Hamiltonian at Gamma point
    S_mat = Ham_sub.Sk(format="array", dtype=complex) # get hopping matrix at Gamma point
    
    out = {"E": energies,
           "eta": eta,
           "H_re": Ham_sub,
           "electrode_idx": elist,
           "atoms_idx": alist}
    GF = np.empty(shape=(num_E, num_atoms, num_atoms), dtype=complex)
    RSEs = np.empty_like(GF, dtype=complex)
    for i, E in enumerate(tqdm(energies, desc="Looping energies")):
        z = E + eta
        invG, RSE_reordered = dysonEQ(z=z, S_mat=S_mat, H_mat=H_mat, rse=rse, alist=alist)
        GF[i, :, :] = invertGF(invG, full_mat=True)
        RSEs[i, :, :] = RSE_reordered
    out["GF"] = GF
    out["RSE"] = RSEs
    return out
        
    
    
    
    

def calculate_spectral_density(energies, eta, H_dev, rse, alist, elist):
    assert isinstance(eta, complex), "Energy perturbation must be complex and should be purely imag."
    nC = len(elist) # number atoms in electrode 
    nS = len(H_dev) # number of sites/atoms
    nE = len(energies) # number of energies
    H_mat = H_dev.Hk(format="array")
    S_mat = H_dev.Sk(format="array")
    
    LDOS = np.zeros(shape=(nS, nE), dtype=float)
    for i, E in enumerate(tqdm(energies, desc="LDOS calc")):
        z = E + eta
        invG, _ = dysonEQ(z, S_mat, H_mat, rse, alist)
        G = invertGF(invG, full_mat=True)
        LDOS[:, i] = calc_ldos(G)
    
    return LDOS
    