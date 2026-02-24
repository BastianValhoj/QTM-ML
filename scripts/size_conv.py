# %%
import sisl
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from pathlib import Path
from generator import calculate_spectral_GF, systemInit, setup_ham_rse
from itertools import product

# %%

OUTDIR = Path(__file__).parent / "ldos_combos"
OUTDIR.mkdir(exist_ok=True, parents=False)
# print(OUTDIR)
# raise EOFError("testing, debugging")

# %%
LIST_OF_N = np.array([(i*4)+1 for i in range(2,11)])
NK1 = 300
ETA = 1e-3j
DELTA_E = 0.1
EMAX = 3.0 # eV
EMIN = -EMAX
ENERGIES = np.array([0])
# ENERGIES = np.array([i*DELTA_E for i in range(-1, 2)])
# energies = np.arange(Emin, Emax +  dE, dE)

# params = set(product(Ns, k_samples, etas))

def main():
    Ham0 = systemInit(bond=1.43, t=-2.7)
    out = {"nk1": NK1,
        "eta": ETA,
        "E": ENERGIES,
        }
    OUTPUTS= {}
    diffs = np.empty(shape=(len(LIST_OF_N), ), dtype=complex)
    for i, N in enumerate(tqdm(LIST_OF_N, desc="Looping over size")):
        n0 = 0
        n_halfway_side = (N//2)+1
        E0_idx = ENERGIES[np.nonzero(ENERGIES==0)]
        H_final, rse, atoms_idxs, electrode_idxs = setup_ham_rse(Ham0, tile=int(N), nk1=NK1, eta=ETA)
        out = calculate_spectral_GF(energies=ENERGIES, eta=ETA, Ham_sub=H_final, rse=rse, alist=atoms_idxs, elist=electrode_idxs)
        OUTPUTS[f"{N}"] = out
        
        rse_re = out["RSE"]
        rse_diag = np.diagonal(rse_re, axis1=1, axis2=2).copy()
        rse_diff, = rse_diag[E0_idx, n0] - rse_diag[E0_idx, n_halfway_side]
        diffs[i] = rse_diff
    OUTPUTS["rse_diff"] = diffs
    
    return OUTPUTS
        
        # LDOS = calculate_spectral_density(ENERGIES, ETA, H_final, rse, alist, elist)
        # nC = len(elist)
        # LDOS_dev = LDOS[nC:, :]
        # out[f"{Na}"] = LDOS
        # out[f"device_{Na}"] = LDOS_dev
        # out[f"alist_{Na}"] = alist
        # out[f"elist_{Na}"] = elist
    
if __name__ == "__main__":
    results = main()
    print("Calculations done! output has keys:\n {}".format(results.keys()))
    filename = OUTDIR / f"ldos-conv-NN.npz"
    np.savez(filename, **results)
# # %%
# print(out.keys())

# # %%
# filename = OUTDIR / f"ldos-conv-NN.npz"
# np.savez(filename, **out)


