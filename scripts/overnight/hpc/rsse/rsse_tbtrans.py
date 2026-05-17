import sisl
import numpy as np
from tqdm.auto import tqdm

from mytools.construct import all_armchair, make_edge
from mytools.scalingv2 import extrapolate, rsse_mapping

from pathlib import Path
import os

#import h5py



def dag(arr):
    return np.conjugate(np.swapaxes(arr, axis1=-2, axis2=-1))



### parameters
#bond_length = 1.42
N0 = 11
script_dir = Path(__file__).parent
rsse_out = script_dir / f"TBT-Nw{N0}"



## basic geometry
rsse_out.mkdir(parents=False, exist_ok=True)
#hopping_dist = (0.1, bond_length+1e-2)
#hopping_term = (0.0, -2.7)

eta = 1e-3
nkpts = 1200
nk1 = int(np.ceil(nkpts/N0))

## energies
emax = 1.0
emin = -emax
estep = 0.1
energies = np.arange(emin, emax, estep) + eta*1j

Ham0 = sisl.get_sile(script_dir / "Ham0.nc").read_hamiltonian()
graphene_cell = Ham0.geometry

BZ = sisl.MonkhorstPack(Ham0, [1, nk1, 1])
rsse = sisl.RealSpaceSE(Ham0, 0, 1, (N0, N0, 1))
rsse.setup(eta=eta, bz=BZ)

## Expanded structure
Ham_elec, elec_idx = rsse.real_space_coupling(ret_indices=True)
Ham_NN = rsse.real_space_parent()

na = Ham_NN.na
all_idx = np.arange(na)
device_idx = np.delete(all_idx, elec_idx)
sub_idx = np.concat([elec_idx, device_idx])

Ham_reorder = Ham_NN.sub(sub_idx)
Ham_reorder.write(rsse_out / "Device.nc")
Ham_reorder.geometry.write(rsse_out / "device.fdf")

np.savetxt(rsse_out / "elec_indices.txt", elec_idx, fmt="%d")
np.savetxt(rsse_out / "device_idx.txt", device_idx, fmt="%d")
np.savez(rsse_out / "energies", E=energies)


geom_edge = make_edge(graphene_cell, N0, N0)
geom_edge.write(rsse_out / "geom_edge.fdf")


####################################################################################
local_results = []
for iE, E in enumerate(tqdm(energies, desc=f"Generating self-energies")):
    #z = E + eta*1j
    RSSE = rsse.self_energy(E)
    local_results.append([iE, RSSE])

####################################################################################

rsse_dict = {iE: RSSE for iE, RSSE in local_results}

with sisl.io.tbtgfSileTBtrans(rsse_out / "graphene.TBTGF") as f:
    f.write_header(BZ, energies)
    for ispin, new_k, k, E, in tqdm(f, desc="Writing TBTGF"):
        if new_k:
            H = Ham_elec.Hk(format="array", dtype=np.complex64)
            S = Ham_elec.Sk(format="array", dtype=np.complex64)
            f.write_hamiltonian(H, S)
        # find the corresponding energy index
        iE = np.argmin(np.abs(energies-E))
        RSSE = rsse_dict[iE]
        f.write_self_energy(RSSE)
with open(rsse_out, "w") as f:
    f.write(
"""
SystemName        graphene
SystemLabel       graphene

# ---------- Hamiltonian ----------
TBT.HS Device.nc

#---------- Electrode ----------
%block TS.Elecs
    Elec
%endblock TS.Elecs

%block TS.Elec.Elec
    HS Elec.nc
    semi-inf-direction  abc 
    electrode-position 1
    Out-of-core true
    tbt.Gf      graphene.TBTGF
%endblock 

# ---------- k-point sampling ----------
TBT.k [1 1 1]

# ---------- Contour line ----------
%block TBT.Contour.line
  part line
     from -3 eV to 3 eV
       file contour.E 
%endblock TBT.Contour.line

# ---------- Output ----------
TBT.DOS.A
TBT.DOS.Gf
TBT.Current.Orb
""")
    f.close()
    print("Job finished")
