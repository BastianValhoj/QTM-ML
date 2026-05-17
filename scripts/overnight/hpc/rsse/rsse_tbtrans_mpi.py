import sisl
import numpy as np
from tqdm.auto import tqdm

from mytools.construct import all_armchair, make_edge
from mytools.scalingv2 import extrapolate, rsse_mapping

from pathlib import Path
import os

#import h5py
from mpi4py import MPI
import socket

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f"[Rank {rank}/{size}] PID={os.getpid()} Host={'socket.hostname()'} [not implemented]", flush=True)

def dag(arr):
    return np.conjugate(np.swapaxes(arr, axis1=-2, axis2=-1))


script_dir = Path(__file__).parent
rsse_out = script_dir / "TBT"



## basic geometry
if rank == 0:
    rsse_out.mkdir(parents=False, exist_ok=True)
    
    ### parameters
    #bond_length = 1.42
    N0 = N1 = 11
    #hopping_dist = (0.1, bond_length+1e-2)
    #hopping_term = (0.0, -2.7)
    
    eta = 1e-3
    nkpts = 1200
    nk1 = int(np.ceil(nkpts/N1))

    ## energies
    emax = 1.0
    emin = -emax
    estep = 0.1
    energies = np.arange(emin, emax, estep)

    #graphene_cell = all_armchair(bond=bond_length)
    #print("Ham0.nsc = {}".format(Ham0.nsc))
    
    #BZ = sisl.MonkhorstPack(Ham0, [1, nk1, 1])
else:
    N0 = None
    N1 = None
    eta = None
    nk1 = None
    energies = None
    #Ham0 = None
    #graphene_cell = None
    #rsse = None
    #BZ = None

# broadcast
N0 = comm.bcast(N0, root=0)
N1 = comm.bcast(N1, root=0)
eta = comm.bcast(eta, root=0)
nk1 = comm.bcast(nk1, root=0) 
#Ham0 = comm.bcast(Ham0, root=0)
energies = comm.bcast(energies, root=0)

BZ = comm.bcast(BZ, root=0)
Ham0 = sisl.get_sile(script_dir / "Ham0.nc").read_hamiltonian()
graphene_cell = Ham0.geometry
rsse = sisl.RealSpaceSE(Ham0, 0, 1, (N0, N1, 1))
rsse.setup(eta=eta, bz=BZ)

## Expanded structure
if rank == 0:
        
    Ham_elec, elec_idx = rsse.real_space_coupling(ret_indices=True)
    Ham_NN = rsse.real_space_parent()
    
    na = Ham_NN.na
    all_idx = np.arange(na)
    device_idx = np.delete(all_idx, elec_idx)
    sub_idx = np.concat(elec_idx, device_idx)

    Ham_reorder = Ham_NN.sub(sub_idx)
    Ham_reorder.write(rsse_out / "Device.nc")
    Ham_reorder.geometry.write(rsse_out / "device.fdf")

    np.savetxt(rsse_out / "elec_indices.txt", elec_idx, fmt="%d")
    np.savetxt(rsse_out / "device_idx.txt", device_idx, fmt="%d")
    np.savez(rsse_out / "energies", E=energies)
    
    
    geom_edge = make_edge(graphene_cell, N0, N0)
    geom_edge.write(rsse_out / "geom_edge.fdf")

else:
    Ham_elec = None
    sub_idx = None
    elec_idx = None
    Ham_reorder = None
    geom_edge = None

sub_idx = comm.bcast(sub_idx, root=0)
elec_idx = comm.bcast(elec_idx, root=0)
Ham_elec = comm.bcast(Ham_elec, root=0)
Ham_reorder = comm.bcast(Ham_reorder, root=0)
geom_edge = comm.bcast(geom_edge, root=0)

comm.Barrier() # wait for all ranks to catch up 
####################################################################################
# distribute energies over ranks
local_idx = np.array_split(np.arange(len(energies)), size)[rank]
print(f"Rank {rank} handles {len(local_idx)} energies", flush=True)

local_results = []
for iE in tqdm(local_idx, desc=f"Generating self-energies for rank {rank}"):
    E = energies[iE] + eta*1j
    print("E: ", np.round(E.real, 3), "eta: ", np.round(E.imag, 3))
    RSSE = rsse.self_energy(E)
    local_results.append([iE, RSSE])

all_results = comm.gather(local_results, root=0)

comm.Barrier() # wait for all ranks to catch up


####################################################################################

# use rank 0 to write 
if rank == 0:
    flat = [item for sublist in all_results for item in sublist] # concatenate list to a 'list of lists'
    flat.sort(key=lambda x: x[0]) # sort list by first their first elements (the energy index)
    rsse_dict = {iE: RSSE for iE, RSSE in flat}
    
    with sisl.io.tbtgfSileTBtrans(rsse_out / "graphene.TBTGF") as f:
        f.write_header(BZ, energies)
        for ispin, new_k, k, E, in tqdm(f, desc="Writing TBTGF"):
            H = Ham_reorder.Hk(format="array", dtype=np.complex128)
            S = Ham_reorder.Sk(format="array", dtype=np.complex128)
            f.write_hamiltonian(H, S)
        # find the corresponding energy index
        iE = np.argmin(np.abs(energies-e))
        RSE = rse_dict[iE]
        f.write_self_energy(RSE)
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
