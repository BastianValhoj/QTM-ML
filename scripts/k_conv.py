# %%
import sisl
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from pathlib import Path
from generator import calculate_spectral_density, systemInit, setup_ham_rse
from itertools import product

script_dir = Path(__file__).parent
outdir = script_dir / "ldos_combos"
outdir.mkdir(exist_ok=True, parents=False)

# %%
# Ns = np.array([(i*2)+1 for i in range(3,21,3)])
k_samples = np.arange(50, 300, 50, dtype=int)
# etas = np.array([1e-1, 5e-2, 2e-2, 1e-2, 8e-3, 5e-3, 3e-3, 2e-3, 1e-3])

dE = 0.1
Emax = 3.0 # eV
Emin = - Emax
energies = np.array([i*dE for i in range(-3, 4)]); print(energies)
# energies = np.arange(Emin, Emax +  dE, dE)

# params = set(product(Ns, k_samples, etas))

# %%
Ham = systemInit(bond=1.43, t=-2.7)
Na = 14
eta = 1e-3j
out = {"Na": Na,
       "eta": eta,
       "E": energies,
       }

for nk1 in tqdm(k_samples, desc="Looping over size"):
    H_final, rse, alist, elist = setup_ham_rse(Ham, tile=int(Na), nk1=nk1, eta=eta)
    LDOS = calculate_spectral_density(energies, eta, H_final, rse, alist, elist)
    nC = len(elist)
    LDOS_dev = LDOS[nC:, :]
    out[f"{nk1}"] = LDOS_dev

out["alist"] = alist
out["elist"] = elist


# %%
print(out.keys())

# %%
filename = outdir / f"{Path(__file__).stem}-ldos.npz"
print(f"saving results to file {filename}")
np.savez(filename, **out)


