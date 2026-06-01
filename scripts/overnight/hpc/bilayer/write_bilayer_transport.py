import sisl
import numpy as np
from pathlib import Path
import argparse
from tqdm.auto import tqdm

from mpi4py import MPI
import os
import socket
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f"[Rank {rank}/{size}] PID={os.getpid()}  HOST={socket.gethostname()}", flush=True)
###############################

if rank == 0:
    parser = argparse.ArgumentParser(description="Write Ham.nc, electrode.TBTGF, and shell for given bilayer device")
    parser.add_argument("--nc", type=int, default=0,  help="NC parameter    (default: 0)")
    parser.add_argument("--ns", type=int, default=6,  help="N_SMALL parameter    (default: 6)")
    parser.add_argument("--nt", type=int, default=30, help="N_BIG/target parameter    (default: 30)")
    parser.add_argument("--stack", type=str, default="AA", help="Whether to use A-A or A-B stacking (latter translated by bondlength along y)")
    parser.add_argument("--angle", type=str, default=0, help="Which angle the top layer has been rotated by")
    # parser.add_argument("--shift", type=int, default=0, help="How far (in Angstrom) from the edges to remove inter-layer coupling")
    args = parser.parse_args()


    NC = args.nc
    N = args.ns
    NT = args.nt # or 50 (or 100)

    STACK = args.stack
    ANGLE = args.angle

    WORK_DIR = Path.home() / "w3"
    RSSE_DIR = WORK_DIR / "rsse_data" / f"TBT-NC{NC}_{N}_to_{NT}"

    BILA_DIR = WORK_DIR / "bilayer_data"
    HAMS_DIR = BILA_DIR / "ham" / f"{STACK.upper()}_stack"
    TILE_DIR = HAMS_DIR / f"NC{NC}-N{N}_to_{NT}"
    ANGLE_DIR = TILE_DIR / f"angle-{ANGLE}"
    # OUT_DIR = ANGLE_DIR / f"shift-{SHIFT}"

    # mpi4py has trouble loading .npz files, so we convert to python dictionary
    _tmp = np.load(TILE_DIR / "params.npz")
    
    PARAMS = {key: _tmp[key] for key in _tmp.files}
    MU_TOP = PARAMS["mu_top"]
    MU_BOTTOM = PARAMS["mu_bottom"]
    BIAS = MU_TOP - MU_BOTTOM
    WHICH = PARAMS["which"]
    
    SHIFTS = np.array([0, 10, 20]) # Å
else:
    NC = None
    N = None
    NT = None

    STACK = None
    ANGLE = None

    WORK_DIR = None
    RSSE_DIR = None

    BILA_DIR = None
    HAMS_DIR = None
    
    TILE_DIR = None
    ANGLE_DIR = None
    # OUT_DIR = None

    PARAMS = None
    MU_TOP = None
    MU_BOTTOM = None
    BIAS = None
    
    SHIFTS = None

comm.Barrier()

NC = comm.bcast(NC, root=0)
N = comm.bcast(N, root=0)
NT = comm.bcast(NT, root=0)

STACK = comm.bcast(STACK, root=0)
ANGLE = comm.bcast(ANGLE, root=0)

WORK_DIR = comm.bcast(WORK_DIR, root=0)
RSSE_DIR = comm.bcast(RSSE_DIR, root=0)

BILA_DIR = comm.bcast(BILA_DIR, root=0)
HAMS_DIR = comm.bcast(HAMS_DIR, root=0)

TILE_DIR = comm.bcast(TILE_DIR, root=0)
ANGLE_DIR = comm.bcast(ANGLE_DIR, root=0)
# OUT_DIR = comm.bcast(OUT_DIR, root=0)

PARAMS = comm.bcast(PARAMS, root=0)
MU_TOP = comm.bcast(MU_TOP, root=0)
MU_BOTTOM = comm.bcast(MU_BOTTOM, root=0)
BIAS = comm.bcast(BIAS, root=0)

SHIFTS = comm.bcast(SHIFTS, root=0)

comm.Barrier()

local_shifts = np.array_split(SHIFTS, size)[rank]

def remove_edge_coupling(H, idx_top, idx_bot, shift):
    
    Ham = H.copy()
    # get center and stacked device height
    center = Ham.center()
    height = np.max(Ham.xyz[:,2]) - np.min(Ham.xyz[:, 2])
    
    # get all radii of Top (Bottom) atoms
    radii_top = np.linalg.norm(Ham.xyz[idx_top, :2] - center[:2], axis=1)
    radii_bot = np.linalg.norm(Ham.xyz[idx_bot, :2], axis=1)
    
    # find the greatest radius of Top (Bottom) to find the radius of the whole layer
    max_possible_rad_top = np.max(radii_top)
    max_possible_rad_bot = np.max(radii_bot)
    
    # Use the radius of the layer with the smallest radius as basis for the 'shrinked' device region.
    max_electrode_radii = np.min([max_possible_rad_top, max_possible_rad_bot])
    
    # offset the max region by some distance (in Angstrom)
    Rmax = max_electrode_radii - shift
    
    # Define a geoemtric cylinder and find all atoms indices within this clyinder
    cylinder = sisl.shape.EllipticalCylinder(Rmax, h=height+10, center=center)
    within_Rmax_mask = cylinder.within(Ham.xyz)
    
    # use numpy 'NOT' operator "~" on the array
    in_boundary_region = ~within_Rmax_mask
    
    # make csr
    H_csr = Ham.tocsr()
    
    # find i, j indices of nonzero elements
    rows, cols = H_csr.nonzero()
    
    # initialize array 
    is_top = np.zeros(shape=Ham.no, dtype=bool)
    is_bot = np.zeros(shape=Ham.no, dtype=bool)
    
    # set indices of Top (Bottom) belogining to Top (Bottom) to True
    is_top[idx_top] = True
    is_bot[idx_bot] = True
    
    # identify interlayer pairs. check   top->bot      OR     bot->top   
    # (rows being 'from' and cols being 'to')
    is_interlayer = (is_top[rows] & is_bot[cols]) | (is_top[cols] & is_bot[rows])
    
    # find coupling elements where: coupling is inter-layer AND either the atom coupling *TO* or *being* coupled to is in boundary region.
    to_zero = is_interlayer & (in_boundary_region[rows] | in_boundary_region[cols])
    print(f"# [Rank {rank}] : Number of elements set to 0 : {to_zero.sum()}")
    
    # loop over indices that 'should' be zero and set the hamiltonian to zero.
    for i, j in zip(rows[to_zero], cols[to_zero]):
        Ham[i,j] = 0.0
    
    Ham.eliminate_zeros()
    return Ham


def write_shell(time="00:30", cores=8, memory=8):
    shellinput=f"""
#!/bin/bash
# LSBATCH: User input
### Job name (shown in queue and in email notificaitons)
#BSUB -J trans-{NC}-{N}-{NT}-{STACK}
### Specify which queue (here, just the HPC)
#BSUB -q hpc
### Request flag: reserve 16GB available for the duration of the job
#BSUB -R "rusage[mem={memory}GB]"
### Notify when job begins
#BSUB -B
### Notify when job ends
#BSUB -N
### Wall-clock time (HH:MM)
#BSUB -W {time}
### Number of processors
#BSUB -n {cores}
### Request ressrouce: Use single host for ressources
#BSUB -R "span[hosts=1]"
### mail notification
#BSUB -u s192943@student.dtu.dk
### -- Specify the output and error file. %J is the job-id --
#BSUB -o out.bi-trans-NC{NC}-{N}-{NT}-{STACK}.%J.log
#BSUB -e err.bi-trans-NC{NC}-{N}-{NT}-{STACK}.%J.log

source /dtu/sw/dcc/dcc-sw.bash
module load siesta

BASE={TILE_DIR}
for angle_dir in $BASE/angle-*; do
    for shift_dir in $angle_dir/shift-*; do
        echo "### Running tbtrans in $shift_dir"
        cd $shift_dir && mpirun -n {cores} tbtrans RUNTBT.fdf > out.tbt.log
    done
done
echo "## All calculations finished"
"""
    with open(TILE_DIR / "submit_runtbt.sh", "w") as file:
        file.write(shellinput)
    

def write_fdf(output_path, vol, eta, Emin, Emax, bottom_elec_start):
    fdfinput = f"""
SystemName siesta
SystemLabel trans
TBT.HS ./Ham_bi_shift.nc

TBT.T.Bulk True
TBT.Current.Orb True
TBT.DOS.Elecs True
TBT.DOS.A.All True
TBT.DOS.GF True

TBT.Voltage {vol} eV

%block TS.Elecs
    top
    bottom
%endblock TS.Elecs

TBT.Elecs.Eta  {eta} eV
TBT.Contours.Eta {eta} eV ## Actually matters..

%block TBT.contour.line
    from {Emin}. eV to {Emax}. eV # write whatever contour, since we will overwrite it with the one we computed
    file ../../contour.IN
%endblock TBT.contour.line

%block TBT.ChemPots
    top
    bottom
%endblock TBT.ChemPots

%block TBT.ChemPot.top
    mu V/2
%endblock TBT.ChemPot.top

%block TBT.ChemPot.bottom
    mu -V/2
%endblock TBT.ChemPot.bottom

%block Ts.Elec.top
    HS ./Ham_elec_top.nc
    semi-inf-direction abc
    Electrode-position {1}
    Out-of-core True
    bulk True
    chem-pot top
    tbt.gf ./RSSE-TOP.TBTGF
%endblock Ts.Elec.top

%block Ts.Elec.bottom
    HS ./Ham_elec_bottom.nc
    semi-inf-direction abc
    Electrode-position {bottom_elec_start}
    Out-of-core True
    bulk True
    chem-pot bottom
    tbt.gf ./RSSE-BOTTOM.TBTGF
%endblock Ts.Elec.bottom
"""
    with open(output_path / f"RUNTBT.fdf", "w") as file:
        file.write(fdfinput)

def main():
    # define parameters at rank 0
    if rank == 0:
        print(f"### Compiling files for transport for NC={NC}, and N={N} to {NT}, for {STACK} stacking, with angle={ANGLE} deg", flush=True)
        # read contour of original RSSE calcualtions
        contour = sisl.io.table.tableSile(RSSE_DIR / "contour.IN")
        energies, eta  = contour.read_data()
        eta = eta[0]
        # save contour to the output dir
        if not (TILE_DIR / "contour.IN").is_file():
            print(f"copying contour file from {RSSE_DIR / 'contour.IN'}")
            sisl.io.table.tableSile(TILE_DIR / "contour.IN", "w").write_data(energies, eta + np.zeros_like(energies))
        
        # Read precomputed RSSE for top/bottom
        se_top = sisl.get_sile(RSSE_DIR / f"tbt-{WHICH[0]}.TBT.SE.nc")
        print(f"Using 'tbt-{WHICH[0]}.TBT.SE.nc' as top layer")
        se_bottom = sisl.get_sile(RSSE_DIR / f"tbt-{WHICH[1]}.TBT.SE.nc")
        print(f"Using 'tbt-{WHICH[1]}.TBT.SE.nc' as bottom layer")
        
        # loop over angles and write the respective TBTGF for the top
        Ham_bilayer_sile = sisl.get_sile(ANGLE_DIR / f"Ham_bilayer.nc")
        Ham_bi = Ham_bilayer_sile.read_hamiltonian()
        
        elec_idx_top = PARAMS["elec_top"]
        device_idx_top = PARAMS["device_top"]
        idx_top = list(elec_idx_top) + list(device_idx_top)
        N_elec_top = len(elec_idx_top)
        
        elec_idx_bottom = PARAMS["elec_bottom"]
        device_idx_bottom = PARAMS["device_bottom"]
        idx_bottom = list(elec_idx_bottom) + list(device_idx_bottom)
    else:
        # contour = None
        energies = None
        eta = None
        
        se_top = None
        se_bottom = None
        
        Ham_bilayer_sile = None
        Ham_bi = None
        elec_idx_top = None
        device_idx_top = None
        idx_top = None
        
        elec_idx_bottom = None
        device_idx_bottom = None
        idx_bottom = None
        N_elec_top = None
    
    # broadcast paramters to other ranks
    # contour = comm.bcast(contour, root=0)
    energies = comm.bcast(energies, root=0)
    eta = comm.bcast(eta, root=0)
    
    # se_top = comm.bcast(se_top, root=0)
    # se_bottom = comm.bcast(se_bottom, root=0)
    
    # Ham_bilayer_sile = comm.bcast(Ham_bilayer_sile, root=0)
    Ham_bi = comm.bcast(Ham_bi, root=0)
    elec_idx_top = comm.bcast(elec_idx_top, root=0)
    device_idx_top = comm.bcast(device_idx_top, root=0)
    idx_top = comm.bcast(idx_top, root=0)
    
    elec_idx_bottom = comm.bcast(elec_idx_bottom, root=0)
    device_idx_bottom = comm.bcast(device_idx_bottom, root=0)
    idx_bottom = comm.bcast(idx_bottom, root=0)
    N_elec_top = comm.bcast(N_elec_top, root=0)
    comm.Barrier()
    
    ################## Calcualte Ham for different shifts on all ranks
    local_results = []
    for shift in tqdm(local_shifts, desc=f"Rank {rank}/{size} calculating ham with shift"):
        Ham_bi_shift = remove_edge_coupling(
            H=Ham_bi, 
            idx_top=idx_top, 
            idx_bot=idx_bottom, 
            shift=shift)
        Ham_elec_top = Ham_bi_shift.sub(elec_idx_top)
        Ham_elec_bottom = Ham_bi_shift.sub(elec_idx_bottom)
        
        local_results.append((shift, Ham_bi_shift, Ham_elec_top, Ham_elec_bottom))
    
    all_results = comm.gather(local_results, root=0)
    comm.barrier()
    
    ############ Write results on rank 0
    if rank == 0:
        for results in tqdm(all_results, desc="Writing on rank 0 for each shift"):
            for shift, Ham_bi_shift, Ham_elec_top, Ham_elec_bottom in results:
                output_dir = ANGLE_DIR / f"shift-{shift}"
                output_dir.mkdir(parents=True, exist_ok=True)
                if not output_dir.exists():
                    raise ValueError(f"Directory `{output_dir}` does not exist")
                
                Ham_elec_top.write(output_dir / f"Ham_elec_top.nc")
                Ham_elec_bottom.write(output_dir / "Ham_elec_bottom.nc")
                Ham_bi_shift.write(output_dir / f"Ham_bi_shift.nc")
        
                with sisl.io.tbtgfSileTBtrans(output_dir / f"RSSE-TOP.TBTGF") as file:
                    bz = sisl.BrillouinZone(Ham_elec_top)
                    file.write_header(bz, energies + eta*1j, mu=MU_TOP)
                    for ispin, new_k, k, E in tqdm(file, desc=f"Writing top TBTGF"):
                        if new_k:
                            Sk = Ham_elec_top.Sk(format="array", dtype=np.complex128)
                            Hk = Ham_elec_top.Hk(format="array", dtype=np.complex128)
                            file.write_hamiltonian(H=Hk, S=Sk)
                        se = se_top.self_energy(elec=0, E=E, k=k, sort=True)
                        se_bulk = (np.real(E) + eta*1j)*Sk - Hk - se
                        file.write_self_energy(se_bulk)
                
                with sisl.io.tbtgfSileTBtrans(output_dir / f"RSSE-BOTTOM.TBTGF") as file:
                    bz = sisl.BrillouinZone(Ham_elec_bottom)
                    file.write_header(bz, energies + eta*1j, mu=MU_BOTTOM)
                    for ispin, new_k, k, E in tqdm(file, desc="Writing bottom RSSE (angle independent)"):
                        if new_k:
                            Sk = Ham_elec_bottom.Hk(format="array", dtype=np.complex128)
                            Hk = Ham_elec_bottom.Hk(format="array", dtype=np.complex128)
                            file.write_hamiltonian(H=Hk, S=Sk)
                        se = se_bottom.self_energy(elec=0, E=E, k=k, sort=True)
                        se_bulk = (np.real(E) + eta*1j)*Sk - Hk - se
                        file.write_self_energy(se_bulk)
        
                write_fdf(
                    output_path=output_dir,
                    vol=BIAS, 
                    eta=eta, 
                    Emin=np.min(energies), 
                    Emax=np.max(energies), 
                    bottom_elec_start=N_elec_top+1
                )
        write_shell(
            time="01:00",
            cores=8,
            memory=8,
        )
        print("#### Rank 0 done writing ############", flush=True)
    print(f"## Rank {rank}/{size} done...")

if __name__ == "__main__":
    main()