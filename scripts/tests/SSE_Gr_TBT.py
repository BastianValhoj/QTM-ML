######################################################################
# python script for generating RSSE of Graphene for TBTrans using MPI
######################################################################
import matplotlib
from matplotlib import pyplot as plt
import ase
from ase.lattice import HEX2D
import numpy as np
import scipy as sp
import os.path as osp
import sisl
import sisl.viz
import re
import sys
import pandas as pd
import time
from tqdm import tqdm   
from multiprocessing import Pool
from mpi4py import MPI
import os
import socket
#from SE_utils import read_contour 
from pathlib import Path

#import sys

# outdir = sys.argv

mm=np.dot
def dagger(M):
    return np.conjugate(np.transpose(M))
def abs2(z):
    return (z*np.conjugate(z)).real

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(f"[Rank {rank}/{size}] PID={os.getpid()} Host={socket.gethostname()}", flush=True)
########################################################################################################################################

HOME_DIR= Path(__file__).parent
Dir=HOME_DIR / "test_siesta_output"
Dir.mkdir(parents=False, exist_ok=True)

if rank == 0:
    # Electrode
    Ham0=sisl.get_sile('./Elec.nc').read_hamiltonian()
    #Ham0=Ham0.tile(3,0).tile(3,1) 
    print("nsc: ",Ham0.nsc)
    N0 = 20
    N1 = 20
    eta = 0.001* 1j
    nkpts = 1200
    nk1 = int(np.ceil(nkpts/N1))
else: 
    Ham0=None
    N0=None
    N1=None
    eta=None  
    nk1=None

# broadcast 
Ham0 = comm.bcast(Ham0, root=0)
N0 = comm.bcast(N0, root=0)
N1 = comm.bcast(N1, root=0)
eta = comm.bcast(eta, root=0)
nk1 = comm.bcast(nk1, root=0)

# build RealSpaceSE on all ranks !
rse = sisl.RealSpaceSE(Ham0, 0, 1, (N0, N1, 1))
rse.setup(eta= np.real(eta), bz=sisl.MonkhorstPack(Ham0, [1, nk1, 1]))

########################################################################################################################################
if rank == 0:
    # Electrode
    H_elec, elec_indices=rse.real_space_coupling(ret_indices=True)
    H_elec.write(Dir / 'Elec.nc')

    # Device Hamiltonian
    # equivalent to H = rse.real_space_parent()
    HamNN=Ham0.tile(N0,0).tile(N1,1)
    HamNN.set_nsc([1,1,1])
    geomNN = HamNN.geometry
    
    # Get indices inside atoms and elec atoms
    all_atoms=np.arange(0,HamNN.na)
    inside_atoms = np.delete(all_atoms, elec_indices)    
    # reorder indixes
    alist = np.concatenate([elec_indices, inside_atoms])
    HamNN_reorder=HamNN.sub(alist)
    HamNN_reorder.set_nsc([1,1,1])
    HamNN_reorder.write(Dir/'Device.nc')
    HamNN_reorder.geometry.write(Dir/"device.fdf")

    np.savetxt(Dir / "elec_indices.txt", elec_indices, fmt="%d")
    np.savetxt(Dir / "inside_atoms.txt", inside_atoms, fmt="%d")

    dE=0.1
    Emin, Emax = -3.0, 3.0
    E = np.arange(Emin, Emax + dE / 2, dE)
    sisl.io.tableSile(Dir/"contour.E", 'w').write_data(E, np.zeros(E.size) + dE)
    print("Finish writting Electrode and Device")  
else: 
    H_elec=None
    HamNN_reorder=None
    E=None

########################################################################################################################################
H_elec = comm.bcast(H_elec, root=0)
HamNN_reorder = comm.bcast(HamNN_reorder, root=0)
E = comm.bcast(E, root=0)
comm.Barrier()

bz = sisl.BrillouinZone(H_elec)
energies=E+eta

# distribute energies over ranks
local_idx = np.array_split(np.arange(len(energies)), size)[rank]
print(f"Rank {rank} handles {len(local_idx)} energies", flush=True)
local_results = []
for iE in tqdm(local_idx, desc="Generating self-energies"):
    e = energies[iE]
    print("E: ",np.round(e.real,3),"eta: ", np.round(e.imag,3))
    RSE = rse.self_energy(e, bulk=True, coupling=True, dtype=np.complex64)
    local_results.append((iE, RSE))
all_results = comm.gather(local_results, root=0)
comm.Barrier()
########################################################################################################################################
# only rank 0 writes 
if rank == 0: 
    flat = [item for sublist in all_results for item in sublist]
    flat.sort(key=lambda x: x[0])
    rse_dict = {iE: RSE for iE, RSE in flat}
    with sisl.io.tbtgfSileTBtrans(Dir/ "graphene.TBTGF") as f:
        f.write_header(bz, energies)
        for ispin, new_k, k, e in tqdm(f, desc="Writing TBTGF"):
            print("spin: ",ispin)
            if new_k:
                H = H_elec.Hk(format="array", dtype=np.complex64)
                S = H_elec.Sk(format="array", dtype=np.complex64)
                f.write_hamiltonian(H, S)
            # find the corresponding energy index
            iE = np.argmin(np.abs(energies - e))
            RSE = rse_dict[iE]
            f.write_self_energy(RSE)
    f = open(Dir/"tbt.fdf", 'w')
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
