import matplotlib
import numpy as np
import sisl 
from tqdm import tqdm   


def calc_RSE(E,eta,rse,H_sub,S_sub,alist=None):
    z = E +  eta
    RSE = rse.self_energy(z)
    if alist is not None:
        RSE = RSE[np.ix_(alist, alist)]
    return RSE

def calc_GF(E,eta,rse,H_sub,S_sub,alist=None):
    z = E +  eta
    A = z*S_sub - H_sub 
    RSE = rse.self_energy(z)
    if alist is not None:
        RSE = RSE[np.ix_(alist, alist)]
        A[0:len(alist),0:len(alist)] -= RSE  
    else: 
        A -= RSE
    G = np.linalg.inv(A)   
    dos = - (1.0 / np.pi) * np.imag(np.trace(G@S_sub)) 
    ldos = - (1.0 / np.pi) * np.imag(np.diag(G@S_sub))
    return dos, ldos



def loop_GF(energies,eta,rse,H_sub,S_sub,alist=None):
    # Observables
    N = H_sub.shape[0]
    DOS = np.zeros_like(energies, dtype=float)
    LDOS = np.zeros((N, len(energies)), dtype=float)  

    # Loop
    for iE, E in enumerate(tqdm(energies, desc='DOS')):
        dos,ldos =calc_GF(E,eta,rse,H_sub,S_sub,alist=alist)
        DOS[iE] = dos
        LDOS[:,iE] = ldos
    return DOS, LDOS
