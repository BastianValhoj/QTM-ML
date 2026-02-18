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
# Ns = np.array([(i*2)+1 for i in range(3,21,3)])
# k_samples = np.arange(50, 300, 50)
etas = np.array([1e-1j, 5e-2j, 2e-2j, 1e-2j, 8e-3j, 5e-3j, 3e-3j, 2e-3j, 1e-3j])

dE = 0.1
Emax = 3.0 # eV
Emin = - Emax
energies = np.array([i*dE for i in range(-3, 4)]); print(energies)
# energies = np.arange(Emin, Emax +  dE, dE)

# params = set(product(Ns, k_samples, etas))

# %%
Ham = systemInit(bond=1.43, t=-2.7)
nk1 = 250
Na = 14
# eta = 1e-3j
out = {"nk1": nk1,
       "Na": Na,
       "E": energies
       }

for eta in tqdm(etas, desc="Looping over size"):
    H_final, rse, alist, elist = setup_ham_rse(Ham, tile=int(Na), nk1=nk1, eta=eta)
    LDOS = calculate_spectral_density(energies, eta, H_final, rse, alist, elist)
    nC = len(elist)
    LDOS_dev = LDOS[nC:, :]
    out[f"{eta}"] = LDOS_dev
    
out["alist"] = alist
out["elist"] = elist


# %%
print(out.keys())

# %%
filename = outdir / f"ldos-conv-eta.npz"
np.savez(filename, **out)


