import sisl 
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from pathlib import Path

from mytools.construct import all_armchair, make_edge
from mytools.scalingv2 import rsse_mapping

from tqdm.auto import tqdm

# %% Parameters
EMAX = 4.0
EMIN = -4.0
ESTEP = 0.1
ENERGIES = np.arange(EMIN, EMAX+ESTEP, ESTEP).round(3)
NUM_ENERGIES = len(ENERGIES)

N_SMALL = 10
N_BIG = 20
NC = 1
ETA = 1e-3
NK1 = lambda N: int(np.ceil(1200/N))


# %% Paths

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / 'rsse_data' / f"TBT-test_{N_SMALL}_to_{N_BIG}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True) 

# %% Functions
def make_base() -> sisl.Hamiltonian:
    bond = 1.42
    graphene_cell = all_armchair(bond)
    Ham0 = sisl.Hamiltonian(graphene_cell)
    R = (0.1, bond+1e-2)
    T = (0.0, -2.7)
    Ham0.construct([R, T])
    return Ham0


def build_edges(geom : sisl.Hamiltonian | sisl.Geometry, 
                n_small : int, 
                n_big : int
                ) -> tuple[sisl.Geometry, sisl.Geometry]:
    geom_edge_small = make_edge(geom, n_small, n_small)
    geom_edge_big = make_edge(geom, n_big, n_big)
    return geom_edge_small, geom_edge_big

def resub_ham(rsse) -> dict:
    Ham_elec, elec_idx = rsse.real_space_coupling(ret_indices=True)
    Ham_NN = rsse.real_space_parent()
    all_idx = np.arange(Ham_NN.na)
    device_idx = np.delete(all_idx, elec_idx)
    sub_idx = np.concatenate([elec_idx, device_idx])
    Ham_re = Ham_NN.sub(sub_idx)
    
    resub_dict = {
        'Ham_elec': Ham_elec,
        'Ham_re': Ham_re,
        'elec_idx': elec_idx,
        'device_idx': device_idx,
        'num_elec': len(elec_idx),
        'sub_idx': sub_idx
    }
    return resub_dict

def write_fdf_device_indices_grouped(indices, output_file : Path | str = "TBT_Atoms_Device.fdf"):
    # Group consecutive indices into sequences
    # Input: indices in sisl (0-based)
    # Writes: indices in siesta/fdf (1-based!)
    sequences = []
    current_sequence = [indices[0]]

    for i in range(1, len(indices)):
        if indices[i] == indices[i - 1] + 1:  # Check if the current index is consecutive
            current_sequence.append(indices[i])
        else:
            sequences.append(current_sequence)
            current_sequence = [indices[i]]

    # Append the last sequence
    sequences.append(current_sequence)
    # Convert sequences into FDF-compatible ranges
    fdf_lines = []
    for seq in sequences:
        if len(seq) > 1:
            fdf_lines.append(f"  atom [{seq[0]+1} -- {seq[-1]+1}]")
        else:
            fdf_lines.append(f"  atom {seq[0]+1}")

    # Combine into the FDF block
    fdf_block = "%block TBT.Atoms.Device\n" + "\n".join(fdf_lines) + "\n%endblock"

    # Print the FDF block
    print(fdf_block)
    # Specify the output file name
    #output_file = "TBT_Atoms_Device.fdf"

    # Write the FDF block to the file
    with open(output_file, "w") as f:
        f.write(fdf_block)
    try:
        print(f"FDF block written to '{output_file.relative_to(SCRIPT_DIR)}'")
    except Exception:
                print(f"FDF block written to '{output_file}'")



def build_rsse(Ham : sisl.Hamiltonian,
               N : int
               ) -> sisl.RealSpaceSE:
    rsse = sisl.RealSpaceSE(Ham, 0, 1, (N, N, 1))
    BZ = sisl.MonkhorstPack(Ham, [1, NK1(N), 1])
    rsse.setup(eta=ETA, bz=BZ)
    return rsse

def compute_rsse(Ham_re : sisl.Hamiltonian, 
                 rsse : sisl.RealSpaceSE, 
                 energies : np.ndarray) -> np.ndarray:
    rsse_collection = np.zeros(shape=(NUM_ENERGIES, Ham_re.na, Ham_re.na), dtype=np.complex128)
    for iE, E in enumerate(tqdm(energies, desc='Computing RSSE')):
        rsse_collection[iE] = rsse.self_energy(E, bulk=False, coupling=True)
    return rsse_collection

# %% main script

def main():
    print("Building Hamiltonians and geometries...")
    Ham0 = make_base()
    geom_edge_small, geom_edge_big = build_edges(Ham0, N_SMALL, N_BIG)
    
    
    print("Building RSSE objects...")
    rsse_small = build_rsse(Ham0, N_SMALL)
    rsse_big = build_rsse(Ham0, N_BIG)
    
    print("Resubstituting Hamiltonians...")
    resub_small = resub_ham(rsse_small)
    resub_big = resub_ham(rsse_big)
    
    save_dict = {
        'energies': ENERGIES,
        'resub_small': resub_small,
        'resub_big': resub_big
    }
    
    Ham_elec_small : sisl.Hamiltonian = resub_small['Ham_elec']
    Ham_elec_big : sisl.Hamiltonian = resub_big['Ham_elec']
    
    Ham_re_small : sisl.Hamiltonian = resub_small['Ham_re']
    Ham_re_big : sisl.Hamiltonian = resub_big['Ham_re']
    
    num_elec_small = resub_small['num_elec']
    num_elec_big = resub_big['num_elec']
    
    xyz_small = Ham_elec_small.geometry.xyz
    xyz_big = Ham_elec_big.geometry.xyz
    tree_small = cKDTree(xyz_small)
    tree_big = cKDTree(xyz_big)
    
    def match_coupling_atoms(i_big, j_big, *, tol=0.1):
        pairA = None
        pairB = None


        i_small = big_to_small_idx_mapping[i_big] # Parent atom i in `small`
        dR_ji = xyz_big[j_big] - xyz_big[i_big] # distance between atoms i and j in `big`
        r_j = xyz_small[i_small] + dR_ji
        dr_ji, j_small = tree_small.query(r_j, k=1) # find the nearest atom in `small` to position r_j
        # if the distance is within the tolerence, use the indicies as a pair
        if dr_ji < tol:
            pairA = (i_small, j_small)

        # Do the same but reversed
        j_small = big_to_small_idx_mapping[j_big]
        dR_ij = xyz_big[i_big] - xyz_big[j_big]
        r_i = xyz_small[j_small] + dR_ij
        dr_ij, i_small = tree_small.query(r_i, k=1)
        if dr_ij < tol:
            pairB = (i_small, j_small)
        return pairA, pairB
    
    def scale_coupling(small, big):
        MaxRange = np.max(np.linalg.norm(small.cell, axis=1))
        rows = []
        cols = []
        data = []
        for i_big in tqdm(range(big.na), desc="Looping over indices in big"):
            dist, idx = tree_big.query(big.xyz[i_big], distance_upper_bound=MaxRange+0.1, k=int(small.na))
            # print(idx[dist < MaxRange])
            for j_big in idx[dist < MaxRange]:
                row = i_big * num_elec_big + j_big
                pairs = []
                pairA, pairB = match_coupling_atoms(i_big, j_big)
                if not (pairA is None):
                    pairs.append(pairA)
                if not (pairB is None):
                    pairs.append(pairB)
                if (pairA is None) and (pairB is None):
                    continue
                weight = 1. / len(pairs) # how to average between coupling (a_i', a_j') and (a_j', a_i')
                for i_small, j_small in pairs:
                    col = i_small * num_elec_small + j_small 
                    rows.append(row)
                    cols.append(col)
                    data.append(weight)
        print(len(rows), len(cols), len(data))
        return coo_matrix((data, (rows, cols)), shape=(num_elec_big**2, num_elec_small**2)).tocsr()
    
    
    big_to_small_idx_mapping = rsse_mapping(Ham_elec_small, Ham_elec_big, geom_edge_small, geom_edge_big, N_SMALL, N_BIG, Ham0.na, NC, ret_parts=False)
    mapped_indices = list(big_to_small_idx_mapping.values())
    
    rsse_collection_small = compute_rsse(Ham_elec_small, rsse_small, ENERGIES)
    # rsse_collection_big = compute_rsse(Ham_elec_big, rsse_big, ENERGIES)
    
    scaling = scale_coupling(Ham_elec_small, Ham_elec_big)
    
    # rsse_collection_extrapolated = np.array([
    #     scaling @ rsse_collection_small[iE].ravel() 
    #     for iE in range(NUM_ENERGIES)
    # ]).reshape((NUM_ENERGIES, num_elec_big, num_elec_big))
    small_flat = rsse_collection_small.reshape(NUM_ENERGIES, -1)
    rsse_collection_extrapolated = (small_flat @ scaling.T).reshape(NUM_ENERGIES, num_elec_big, num_elec_big)
    
    # write contour 
    sisl.io.table.tableSile(OUTPUT_DIR / "contour.IN", "w").write_data(ENERGIES, np.zeros(ENERGIES.shape), np.zeros(ENERGIES.shape) + ETA)
    
    Ham0.write(OUTPUT_DIR / "Ham0.nc")
    Ham_elec_big.write(OUTPUT_DIR / "Ham_elec_big.nc")
    Ham_elec_small.write(OUTPUT_DIR / "Ham_elec_small.nc")
    Ham_re_big.write(OUTPUT_DIR / "Ham_re_big.nc") # device Hamiltonian for `big`
    Ham_re_small.write(OUTPUT_DIR / "Ham_re_small.nc") # device Hamiltonian for `small`
    with sisl.io.tbtgfSileTBtrans(OUTPUT_DIR / "RSSE.TBTGF") as f:
        bz = sisl.MonkhorstPack(Ham0, [1, NK1(N_BIG), 1])
        f.write_header(bz, ENERGIES + 1j*ETA)
        for ispin, new_k, k, E in tqdm(f, unit="RSSE"):
            # tqdm.write(f"ispin={ispin}, new_k={new_k}, k={k}, E={E}")
            if new_k:
                Sk = Ham_elec_big.Sk(format="array")
                Hk = Ham_elec_big.Hk(format="array")
                f.write_hamiltonian(Hk, Sk)
            E_idx = np.where(np.isclose(ENERGIES, E.real, atol=1e-5))[0][0]
            f.write_self_energy(rsse_collection_extrapolated[E_idx])
            
    L = np.linalg.norm(Ham_re_big.cell[0]) - 2*np.linalg.norm(Ham0.cell[0])
    print(f"Device length: {L:.2f} Angstrom")
    Rmax = np.sin(np.pi/3.) * L
    Radius = np.round(0.5*Rmax, 1)
    print(f"Radius of device = {Radius:.1f} Angstrom")
    cylinder = sisl.shape.EllipticalCylinder(v=Radius, h=50., center=Ham_re_big.center())
    proj_atoms_idx = cylinder.within(Ham_re_big.xyz).nonzero()[0]
    print(f"Number of atoms in the cylinder projection: {len(proj_atoms_idx)}")
    write_fdf_device_indices_grouped(proj_atoms_idx, output_file=OUTPUT_DIR / "TBT_Atoms_Device.fdf")
    
    
    
    fdfinput = f"""SystemName siesta
SystemLabel tbt
TBT.HS ./Ham_re_big.nc

%include ./TBT_Atoms_Device.fdf

TBT.k [1, {NK1(N_BIG)}, 1]

TBT.CDF.SelfEnergy.Save
TBT.T.Bulk True
TBT.DOS.Elecs True
TBT.DOS.A.All True
TBT.DOS.GF True

%block TS.Elecs
    Bottom
%endblock TS.Elecs

TBT.Elecs.Eta  {ETA} eV
TBT.Contours.Eta 0.00001 eV ## Actually matters..

%block TBT.contour.line
    from -2000. eV to 2000. eV # write whatever contour, since we will overwrite it with the one we computed
    file contour.IN
%endblock TBT.contour.line

%block Ts.Elec.Bottom
    HS ./Ham_elec_big.nc
    semi-inf-direction abc
    Electrode-position {1}
    Out-of-core True
    tbt.gf RSSE.TBTGF
%endblock Ts.Elec.Bottom
"""
    with open(OUTPUT_DIR / "RUNTBT.fdf", "w") as f:
        f.write(fdfinput)
    print(f"FDF input written to '{OUTPUT_DIR.relative_to(SCRIPT_DIR) / 'TBT_input.fdf'}'")
    
# %%
if __name__ == "__main__":
    main()
