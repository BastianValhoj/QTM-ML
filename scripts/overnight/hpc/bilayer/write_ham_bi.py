import sisl
import numpy as np
import argparse

from pathlib import Path

from tqdm.auto import tqdm

from mytools.tbbi import tbbi_opt

import json

parser = argparse.ArgumentParser(description="Write Ham.nc, electrode.TBTGF, and shell for given bilayer device")
parser.add_argument("--nc", type=int, default=0,  help="NC parameter    (default: 0)")
parser.add_argument("--ns", type=int, default=6,  help="N_SMALL parameter    (default: 6)")
parser.add_argument("--nt", type=int, default=30, help="N_BIG/target parameter    (default: 30)")
# parser.add_argument("--which", type=tuple, nargs=2, default=("bottom", "bottom"), help="Tuple of which device to use for given NC, N, and target (Default 'bottom', 'bottom')")
# parser.add_argument("--a", type=float, default=0, help="How much to rotate (in deg) the top layer relative to bottom.")
args = parser.parse_args()

NC = args.nc
N = args.ns
NT = args.nt

# WHICH = args.which
WHICH = ("bottom", "bottom")
ANGLES = np.arange(0, 60, step=5)

MU_TOP = 0.5 # eV
MU_BOTTOM = -0.5 # eV
INTER_LAYER_DIST = 3.35 # Å


if NC > (N-3)//2:
    raise ValueError(f"""##############
                 ### NC can be as large as `N = 1 + 2*(NC + 1)` meaning `NC < (N-3)/2)`
                 ### for the given input {NC=} and {N=} this meanns NC < {(N-3)/2}
                     """)

SCRIPT_DIR = Path(__file__).parent

WORK_DIR = Path.home() / "w3"
RSSE_DIR = WORK_DIR / "rsse_data"
DATA_DIR = RSSE_DIR / f"TBT-NC{NC}_{N}_to_{NT}"

OUT_DIR = WORK_DIR / "bilayer_data" / "ham" / f"NC{NC}-N{N}_to_{NT}"
OUT_DIR.mkdir(exist_ok=True, parents=True)


def stack_device(data_path, which=("bottom", "bottom"), d=3.35, angle=None):
    assert (which[0] == "top" or which[0] == "bottom") and (which[1] == "top" or which[1] == "bottom"), "`which` has to be iterable of either strings `top` or `bottom`"

    assert isinstance(data_path, Path), "data_path has to be of type pathlib.Path`"
    se_top = sisl.get_sile(data_path / f"tbt-{which[0]}.TBT.SE.nc")
    se_bottom = sisl.get_sile(data_path / f"tbt-{which[1]}.TBT.SE.nc")
    
    # # get geometries
    geom_top = se_top.geometry.copy()
    geom_top = geom_top.translate([0, 0, d])
    
    if not (angle is None):
        geom_top = geom_top.rotate(angle, origin=geom_top.center(), v=[0,0,1], what="xyz")
    # geom_top.xyz[:, 2] = d
    geom_bottom = se_bottom.geometry.copy()
    # geom_bottom.xyz[:, 2] = 0
    
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
        'elec_top': np.arange(N_elec_top),
        'elec_bottom': np.arange(N_elec_top, N_elec_total),
        'device_top': np.arange(N_elec_total, N_elec_total + N_device_top),
        'device_bottom': np.arange(N_elec_total + N_device_top, N_elec_total + N_device_total)
        }
    return geom_bilayer, idx_dict



def main():
    geom_bilayer_no_rot, idx_dict = stack_device(DATA_DIR, which=WHICH, d=INTER_LAYER_DIST, angle=None)
    idx_top = list(idx_dict["elec_top"]) + list(idx_dict["device_top"])
    for angle in tqdm(ANGLES, desc=f"Angles"):
        # geom_bilayer, idx_dict = stack_device(DATA_DIR, which=WHICH, d=3.35, angle=a)
        geom_bilayer = geom_bilayer_no_rot.rotate(
            [angle, [0,0,1]],       # rotate by `angle` around axis [0,0,1] (z)
            rad=False,              # Use degrees
            what="xyz",             # rotate only atom coords -- not unit cell
            atoms=idx_top,          # atom indices which should be rotated
            origin=geom_bilayer_no_rot.center(), # pivot point
            )
        
        
        Ham_bilayer, Htb = tbbi_opt(
            geom=geom_bilayer,      # bilayer geometry
            os_0=MU_BOTTOM,         # on-site for bottom
            os_1=MU_TOP,            # on-site for top if None: use os_0
            Vpppi=-2.7,             # intra-layer coupling from pi-bonds
            Vpps=0.48,              # inter-layer coupling from sigma-bonds
            d0_00=1.42,             # intra-layer distance (same layer)
            d0_01=INTER_LAYER_DIST, # inter-layer distance (different layer)
            q0_00=2.0,              # decay rate for pi-bonds hopping integral
            q0_01=None,             # decay rate for sigma-bonds hopping integral (if None use q0_00)
            dangling=0.0,           # shift in on-site energy for edge atoms
            finite=True,            # nsc=(1,1,1), and do not create supercell from unpit geometry
        )
        if angle == ANGLES[0]:
            Htb.write(OUT_DIR / "Ham_bi_no_coupling.nc")
            print("### for no angle: save hamiltonian of no ineter-layer coupling: to `Ham_bi_no_coupling.nc")
        Ham_bilayer.write(OUT_DIR / f"Ham_bilayer_angle{angle}.nc")
        print(f"### Saved Hamiltonian for bilayer and angle {angle}")
    
    idx_dict["mu_top"] = MU_TOP
    idx_dict["mu_bottom"] = MU_BOTTOM
    idx_dict["which"] = WHICH
    idx_dict["d"] = INTER_LAYER_DIST
    np.savez(OUT_DIR / "params", **idx_dict)
    print(f"### Saved idx to npz : `{'params.npz'}`")
        
if __name__ == "__main__":
    print(f"### Angle resolved Hams for NC={NC}, N={N}, NT={NT}", flush=True)
    main()