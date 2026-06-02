import sisl
import numpy as np
from pathlib import Path
import h5py
from tqdm.auto import tqdm


SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "conv_data"

BOND = 1.42
Vpppi = -2.7

ENERGIES = np.arange(-0.4, 0.4+0.1, 0.1)[::-1]
ETAS = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])

N_zig = 17
N_arm = 9

def make_hamiltonian(geom):
    
    H = sisl.Hamiltonian(geom)
    r = (0.1*BOND, BOND+1e-2)
    t = (0.0, Vpppi)
    H.construct([r, t])
    return H



gr_zig = sisl.geom.graphene(bond=BOND)
with h5py.File(DATA_DIR / "calc_dos_vs_eta.h5", "w") as file:
    file.attrs["E"] = ENERGIES
    file.attrs["ETA"] = ETAS
    
    for kind in ["zigzag", "armchair"]:
        group_kind = file.create_group(kind)
        if kind == "zigzag":
            base_geom = sisl.geom.graphene(bond=BOND)
            N = N_zig
        
        else: # kind == "armchair"
            from mytools.construct import all_armchair
            base_geom = all_armchair(bond=BOND)
            N = N_arm
        
        Ham0 = make_hamiltonian(base_geom)
        rsse = sisl.RealSpaceSE(Ham0, 0, 1, (N, N, 1))
        
        nk1 = np.ceil(1200/N)
        group_kind.attrs["N"] = N
        group_kind.attrs["nk1"] = nk1
        
        for eta in tqdm(ETAS, desc="looping etas"):
            rsse.setup(eta=eta, bz=sisl.MonkhorstPack(Ham0, [1, nk1, 1]))
            
            Ham_elec, elec_idx = rsse.real_space_coupling(True)
            Ham_NN = rsse.real_space_parent()
            
            # Hk = Ham_NN.Hk(dtype=np.complex128)
            # Sk = Ham_NN.Sk(dtype=np.complex128)
            all_bulk_rsse = []
            for E in tqdm(ENERGIES, desc="Looping energies", leave=False):
                invG = rsse.self_energy(E=E, bulk=True)
                all_bulk_rsse.append(invG)
            GFs = np.linalg.inv(all_bulk_rsse)
            LDOS = (-1/np.pi)*np.diagonal(np.imag(GFs), axis1=1, axis2=2)
            DOS = LDOS.sum(axis=-1)
            group_kind.create_dataset(f"eta_{eta:.1e}", data=DOS)
            
print("Done!")
    
    
