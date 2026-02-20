# %%
import sisl
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from pathlib import Path
from generator import calculate_spectral_density, systemInit, setup_ham_rse
from itertools import product

outdir = Path("ldos_combos")
outdir.mkdir(exist_ok=True, parents=False)

# %%
Ns = np.array([(i*2)+1 for i in range(3,21,3)])
k_samples = np.arange(50, 300, 50)
etas = np.array([1e-1, 5e-2, 2e-2, 1e-2, 8e-3, 5e-3, 3e-3, 2e-3, 1e-3])

dE = 0.1
Emax = 3.0 # eV
Emin = - Emax
energies = np.array([i*dE for i in range(-3, 4)]); print(energies)
# energies = np.arange(Emin, Emax +  dE, dE)

params = set(product(Ns, k_samples, etas))

# %%
Ham = systemInit(bond=1.43, t=-2.7)
nk1 = 250
eta = 1e-3j
out = {"nk1": nk1,
       "eta": eta,
       "E": energies,
       }

for Na in tqdm(Ns, desc="Looping over size"):
    H_final, rse, alist, elist = setup_ham_rse(Ham, tile=int(Na), nk1=nk1, eta=eta)
    LDOS = calculate_spectral_density(energies, eta, H_final, rse, alist, elist)
    nC = len(elist)
    LDOS_dev = LDOS[nC:, :]
    out[f"{Na}"] = LDOS
    out[f"device_{Na}"] = LDOS_dev
    out[f"alist_{Na}"] = alist
    out[f"elist_{Na}"] = elist
    # save the self-energies 
    # maybe save the spectral density

# %%
print(out.keys())

# %%
filename = outdir / f"ldos-conv-NN.npz"
np.savez(filename, **out)


