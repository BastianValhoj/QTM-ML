import sisl
import numpy as np
from pathlib import Path
import h5py
from tqdm.auto import tqdm


SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "conv_data"

BOND = 1.42
Vpppi = -2.7

ENERGIES = [0]

NK1S_ENUM = np.arange(800, 1300+1, step=100)
ETA = 1e-3

N_zig = [2*i+1 for i in range(1,4)][::-1]
N_arm = [4*i+1 for i in range(1,4)][::-1]

def make_hamiltonian(geom):
    
    H = sisl.Hamiltonian(geom)
    r = (0.1*BOND, BOND+1e-2)
    t = (0.0, Vpppi)
    H.construct([r, t])
    return H



gr_zig = sisl.geom.graphene(bond=BOND)
with h5py.File(DATA_DIR / "calc_dos_vs_nk1.h5", "w") as file:
    file.attrs["E"] = ENERGIES
    file.attrs["ETA"] = ETA
    file.attrs["NK1_enum"] = NK1S_ENUM
    
    for kind in ["zigzag", "armchair"]:
        tqdm.write("### kind")
        group_kind = file.create_group(kind)
        if kind == "zigzag":
            base_geom = sisl.geom.graphene(bond=BOND)
            Nlist = N_zig
        
        else: # kind == "armchair"
            from mytools.construct import all_armchair
            base_geom = all_armchair(bond=BOND)
            Nlist = N_arm
        
        Ham0 = make_hamiltonian(base_geom)
        group_kind.attrs["N"] = Nlist
        
        for N in tqdm(Nlist, desc="Looping over N"):
            
            
            rsse = sisl.RealSpaceSE(Ham0, 0, 1, (N, N, 1))
            
            DOSE0_vs_nk1 = []
            for nk1 in tqdm(NK1S_ENUM, desc="looping nk1", leave=False):
                rsse.setup(eta=ETA, bz=sisl.MonkhorstPack(Ham0, [1, np.ceil(nk1/N), 1]))
                
                Ham_elec, elec_idx = rsse.real_space_coupling(True)
                Ham_NN = rsse.real_space_parent()
                
                # Hk = Ham_NN.Hk(dtype=np.complex128)
                # Sk = Ham_NN.Sk(dtype=np.complex128)
                
                invG = rsse.self_energy(E=0.0, bulk=True)
                GFs = np.linalg.inv(invG)
                LDOS = (-1/np.pi)*np.diagonal(np.imag(GFs), axis1=0, axis2=1)
                DOS = LDOS.sum(axis=-1)
                DOSE0_vs_nk1.append(DOS)
            DOSE0_vs_nk1 = np.array(DOSE0_vs_nk1)
            group_kind.create_dataset(f"N{N}", data=DOSE0_vs_nk1)
                
print("Done!")
        
        
