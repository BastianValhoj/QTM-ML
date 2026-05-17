import sisl 
import numpy as np
from pathlib import Path
from mytools.tbbi import tbbi_opt

from tqdm.auto import tqdm

import argparse 

# %% parser of inputs
parser = argparse.ArgumentParser(description="Write Ham.nc, electrode.TBTGF, and shell for given bilayer device")
parser.add_argument("--nc", type=int, default=0,  help="NC parameter    (default: 0)")
parser.add_argument("--ns", type=int, default=6,  help="N_SMALL parameter    (default: 6)")
parser.add_argument("--nb", type=int, default=30, help="N_BIG/target parameter    (default: 30)")

args = parser.parse_args()


NC = args.nc
N = args.ns
N_target = args.nb # or 50 (or 100)


# if N > N_target:
#     raise ValueError("N cannot be greater than N_target")
# if N == N_target:
#     print(f"WARNING: N == N_target == {N}, running in self-consistency check mode.")
if NC > (N-3)//2:
    raise ValueError(f"""##############
                 ### NC can be as large as `N = 1 + 2*(NC + 1)` meaning `NC < (N-3)/2)`
                 ### for the given input {NC=} and {N=} this meanns NC < {(N-3)/2}
                     """)

# %% define paths for data
SCRIPT_DIR = Path(__file__).parent

WORK_DIR = Path.home() / "w3"
RSSE_DIR = WORK_DIR / "rsse_data"

# index 1 from the list ['TBT-NC{NC}', 'N', 'to', '{N_target}']
N_bases = [int(d.name.split("_")[1]) 
           for d in RSSE_DIR.glob(f"TBT-NC{NC}_*_to_{N_target}")]
def get_sigma(se, E=0):
    tmp = se.self_energy(elec='Border', E=E, k=[0,0,0], sort=True).data
    pvt = se.pivot(elec='Border', in_device=True, sort=True)

    sigma  = np.zeros(shape=(se.no_d, se.no_d), dtype=np.complex128)
    sigma[np.ix_(pvt,pvt)] += tmp
    return sigma


# def device_down(se):
#     geom = se.geometry
#     geom = geom.sub(se.a_dev)
#     return geom

# def stack_device(se_top, se_bottom, d=3.35):
#     geom_top = se_top.geometry
#     geom_bottom = se_bottom.geometry
#     device_top = geom_top.sub(se_top.a_dev)
#     device_bottom = geom_bottom.sub(se_bottom.a_dev)
    
#     device_top = device_top.translate([0,0,d])

#     device = device_top.add(device_bottom) # device_top atom indices are first, then device_bottom
#     a_dev = np.arange(device.na)
#     a_top = a_dev[:se_top.na_dev]
#     a_bottom = a_dev[se_top.na_dev:device.na]
#     idx_dict = {'top': a_top, 'bottom':a_bottom}
#     return device, idx_dict

def stack_device(se_top, se_bottom, d=3.35):
    
    # get geometries
    geom_top = se_top.geometry.copy()
    geom_top = geom_top.translate([0,0,d])
    geom_bottom = se_bottom.geometry.copy()

    # get 'electrode' indices from pivot 
    elec_idx_top = se_top.pivot(elec=0, in_device=False, sort=True)
    elec_idx_bottom = se_bottom.pivot(elec=0, in_device=False, sort=True)

    # get device indices
    down_idx_top = se_top.a_dev
    device_idx_top = np.setdiff1d(down_idx_top, elec_idx_top)

    down_idx_bottom = se_bottom.a_dev
    device_idx_bottom = np.setdiff1d(down_idx_bottom, elec_idx_bottom)

    # save elec indices of non-downfolded, and bottom is offset by whole top layer (non-downfolded)
    elec_idx = np.concat([elec_idx_top, elec_idx_bottom+se_top.na])

    # save device indices of non-downfolded, and bottom is offset by whole top layer (non-downfolded)
    device_idx = np.concat([device_idx_top, device_idx_bottom+se_top.na])

    # get final reordering indices for subbing the full geometries of top and bottom
    reorder_sub_idx = np.concat([elec_idx, device_idx])

    # removed names indices (needed for the 'add' method to work)
    geom_top.names.clear()
    geom_bottom.names.clear()

    # stack the layers and sub according to reordering
    geom = geom_top.add(geom_bottom)
    geom_bilayer = geom.sub(reorder_sub_idx)


    # get length of different relevant regions (electrode/device of top/bottom)
    N_elec_top = len(elec_idx_top)
    N_elec_bottom = len(elec_idx_bottom)
    N_elec_total = N_elec_top + N_elec_bottom

    N_device_top = len(device_idx_top)
    N_device_bottom = len(device_idx_bottom)
    N_device_total = N_device_top + N_device_bottom

    # save indices of subbed geometry to dict
    idx_dict = {
        'elec_top': range(N_elec_top),
        'elec_bottom': range(N_elec_top, N_elec_total),
        'device_top': range(N_elec_total, N_elec_total + N_device_top),
        'device_bottom': range(N_elec_total + N_device_top, N_elec_total + N_device_total)
        }
    return geom_bilayer, idx_dict


def main(d=3.35, v=0):
    # choose directory of data and ensure it exists
    data_dir = RSSE_DIR / f"TBT-NC{NC}_{N}_to_{N_target}"
    if not data_dir.exists(): raise FileNotFoundError(f"Cannot find directory: '{data_dir.name}' at '{RSSE_DIR}'")
    
    # create output data for angle 'v'
    out_dir = WORK_DIR / "bilayer_data" / f"TBT-NC{NC}_{N}_to_{N_target}-v{v}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # get siles for both electrodes
    se_top = sisl.get_sile(data_dir / "tbt-top.TBT.SE.nc")
    se_bottom = sisl.get_sile(data_dir / "tbt-bottom.TBT.SE.nc")
    
    # get contour from 'data dir' and save to 'out dir'
    contour = sisl.io.table.tableSile(data_dir / "contour.IN")
    energies, eta = contour.read_data()
    eta = eta[0]
    sisl.io.table.tableSile(out_dir/"contour.IN", "w").write_data(energies, eta + np.zeros_like(energies))
    
    # check the energies used are the same, or use intersecting values
    if not np.allclose(se_top.E, se_bottom.E):
        print("The same energies has not been used between top and bottom.\n Will use inersect of the two:", flush=True)
        decimals = 8  # adjust tolerance
        energies = np.intersect1d(
            np.round(se_top.E, decimals),
            np.round(se_bottom.E, decimals)
        )
        print(energies, flush=True)
    else:
        energies = np.asarray(se_top.E)
    
    # check the top electrode is smaller than bottom
    if se_top.na_dev >= se_bottom.na_dev:
        raise ValueError("top has equal or more atoms than bottom: {:} > {:}".format(se_top.na_dev, se_bottom.na_dev))
    geom_bilayer, idx_dict = stack_device(se_top, se_bottom, d)
    
    # ensure the z-coordinates of each layer (top has z=d, bottom has z=0)
    if not np.all(geom_bilayer.xyz[idx_dict['elec_top'], 2] == d):
        raise LookupError(f"Top elec do not have z-position d={d} Å.\n Check stacking method...")
    if not np.all(geom_bilayer.xyz[idx_dict['device_top'], 2] == d):
        raise LookupError(f"Top device do not have z-position d={d} Å.\n Check stacking method...")
    if not np.all(geom_bilayer.xyz[idx_dict['elec_bottom'], 2] == 0):
        raise LookupError(f"Bottom elec do not have z-position d={0} Å.\n Check stacking method...")
    if not np.all(geom_bilayer.xyz[idx_dict['device_bottom'], 2] == 0):
        raise LookupError(f"Bottom device do not have z-position d={0} Å.\n Check stacking method...")
    
    # create Slater-Koster Hamiltonian
    Ham_bilayer = tbbi_opt(
        geom=geom_bilayer,  # bilayer geometry
        os_0=0.0,           # on-site for bottom (or top)
        os_1=None,          # on-site for top (or bottom) if None: use os_0
        Vpppi=-2.7,         # intra-layer coupling from pi-bonds
        Vpps=0.48,          # inter-layer coupling from sigma-bonds
        d0_00=1.42,         # intra-layer distance
        d0_01=3.35,         # inter-layer distance
        q0_00=2.0,          # decay rate for pi-bonds hopping integral
        q0_01=None,         # decay rate for sigma-bonds hopping integral (if None use q0_00)
        dangling=0.0,       # shift in on-site energy for edge atoms
        finite=True,        # nsc=(1,1,1), no periodicity
    )
    
    # get Hamiltonian "electrode" (border) region of both layers
    Ham_elec_top = Ham_bilayer.sub(idx_dict['elec_top'])
    Ham_elec_bottom = Ham_bilayer.sub(idx_dict['elec_bottom'])
    N_elec_top = len(idx_dict['elec_top'])
    
    Ham_elec_top.write(out_dir / "Ham_elec_top.nc")
    Ham_elec_bottom.write(out_dir / "Ham_elec_bottom.nc")
    Ham_bilayer.write(out_dir / "Ham_bilayer.nc")
    
    
    # write TBTGF for top
    with sisl.io.tbtgfSileTBtrans(out_dir / "RSSE-TOP.TBTGF") as f:
        bz = sisl.BrillouinZone(Ham_elec_top)
        f.write_header(bz, energies + eta*1j)
        for ispin, new_k, k, E in tqdm(f, desc="Writing top TBTGF"):
            if new_k:
                Sk = Ham_elec_top.Sk(format="array", dtype=np.complex128)
                Hk = Ham_elec_top.Hk(format="array", dtype=np.complex128)
                f.write_hamiltonian(H=Hk, S=Sk)
            se = se_top.self_energy(elec='Border', E=E, k=k, sort=True)
            f.write_self_energy(se)
    
    # write TBTGF for bottom
    with sisl.io.tbtgfSileTBtrans(out_dir / "RSSE-BOTTOM.TBTGF") as f:
        bz = sisl.BrillouinZone(Ham_elec_bottom)
        f.write_header(bz, energies + eta*1j)
        for ispin, new_k, k, E in tqdm(f, desc="Writing bottom TBTGF"):
            if new_k:
                Sk = Ham_elec_bottom.Sk(format="array", dtype=np.complex128)
                Hk = Ham_elec_bottom.Hk(format="array", dtype=np.complex128)
                f.write_hamiltonian(H=Hk, S=Sk)
            se = se_bottom.self_energy(elec='Border', E=E, k=k, sort=True)
            f.write_self_energy(se)
    
    
    # write fdf file for tbtrans calculations
    fdfinput = f"""SystemName siesta
SystemLabel trans
TBT.HS ./Ham_bilayer.nc

TBT.T.Bulk True
TBT.DOS.Elecs True
TBT.DOS.A.All True
TBT.DOS.GF True

%block TS.Elecs
    top
    bottom
%endblock TS.Elecs

TBT.Elecs.Eta  {eta} eV
TBT.Contours.Eta {eta} eV ## Actually matters..

%block TBT.contour.line
    from {np.min(energies)}. eV to {np.max(energies)}. eV # write whatever contour, since we will overwrite it with the one we computed
    file ./contour.IN
%endblock TBT.contour.line

%block Ts.Elec.top
    HS ./Ham_elec_top.nc
    semi-inf-direction abc
    Electrode-position {1}
    Out-of-core True
    bulk True
    tbt.gf RSSE-TOP.TBTGF
%endblock Ts.Elec.top

%block Ts.Elec.bottom
    HS ./Ham_elec_bottom.nc
    semi-inf-direction abc
    Electrode-position {N_elec_top+1}
    Out-of-core True
    bulk True
    tbt.gf RSSE-BOTTOM.TBTGF
%endblock Ts.Elec.bottom
"""
    with open(out_dir / "RUNTBT.fdf", "w") as f:
        f.write(fdfinput)
    
    # Write shell script for submitting to LSF queue
    shell_script=f"""
# LSBATCH: User input
#!/bin/bash
### Job name (shown in queue and in email notificaitons)
#BSUB -J TBT.T-{NC}-{N}-{N_target}
### Specify which queue (here, just the HPC)
#BSUB -q hpc
### Request flag: reserve 16GB available for the duration of the job
#BSUB -R "rusage[mem=16GB]"
### Notify when job begins
#BSUB -B
### Notify when job ends
#BSUB -N
### Wall-clock time (HH:MM)
#BSUB -W 0:30
### Number of processors
#BSUB -n 8
### Request ressrouce: Use single host for ressources
#BSUB -R "span[hosts=1]"
### mail notification
#BSUB -u s192943@student.dtu.dk
### -- Specify the output and error file. %J is the job-id -- 
#BSUB -o out.transport-NC{NC}-{N}-{N_target}.%J.log
#BSUB -e err.transport-NC{NC}-{N}-{N_target}.%J.log

source /dtu/sw/dcc/dcc-sw.bash
module load siesta

mpirun -n 8 tbtrans RUNTBT.fdf > out.tbt.log
"""
    submit_path = out_dir / "submit_runtbt.sh"
    with open(submit_path, "w") as f:
        f.write(shell_script)
    print(f"Submit script written to {submit_path}")
    
    
if __name__ == "__main__":
    print("##################################", flush=True)
    print(f"### Writing for {N} to {N_target}", flush=True)
    print("##################################", flush=True)
    main(d=3.35, v=0)