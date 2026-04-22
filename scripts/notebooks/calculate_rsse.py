import marimo

__generated_with = "0.23.2"
app = marimo.App()


@app.cell
def _():
    import sisl 
    import numpy as np
    from tqdm.auto import tqdm
    from pathlib import Path

    from mytools.construct import all_armchair, make_edge
    from mytools.scalingv2 import extrapolate, rsse_mapping

    import matplotlib.pyplot as plt
    import marimo as mo

    from typing import cast, Tuple, Dict, List, Any

    return (
        Any,
        Dict,
        Path,
        Tuple,
        all_armchair,
        cast,
        make_edge,
        mo,
        np,
        plt,
        rsse_mapping,
        sisl,
        tqdm,
    )


@app.cell
def _(cast, np, sisl):
    def as_geom(obj) -> sisl.Geometry:
        return cast(sisl.Geometry, obj)

    def as_ham(obj) -> sisl.Hamiltonian:
        return cast(sisl.Hamiltonian, obj)

    def as_rsse(obj) -> sisl.RealSpaceSE:
        return cast(sisl.RealSpaceSE, obj)

    def as_ndarray(obj) -> np.ndarray:
        return cast(np.ndarray, obj)


    return as_geom, as_ham, as_ndarray


@app.cell
def _(np):
    bond_length: float= 1.42
    bond_angle: float = np.pi/3
    inner_radius: float = bond_length*np.cos(bond_angle/2)

    N0 = 4
    N_small = 6
    N_big = 11
    NC = 1
    hopping_dist: tuple[float, float] = (0.1, bond_length+1e-2)
    hopping_term = (0.0, -2.7)

    eta = 1e-3
    num_k = 400
    return (
        NC,
        N_big,
        N_small,
        bond_length,
        eta,
        hopping_dist,
        hopping_term,
        num_k,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## helper functions
    """)
    return


@app.cell
def _(Any, Dict, Tuple, np, sisl):
    # will sub the hamiltonian to order the electrode indices first
    def resub_ham(
        rsse: sisl.RealSpaceSE
        ) -> Tuple[sisl.Hamiltonian, Dict[str, Any]]:
        Ham_rs, elec_idx = rsse.real_space_coupling(ret_indices=True)
        Ham_NN = rsse.real_space_parent()
        all_idx = np.arange(Ham_NN.na)
        device_idx = np.delete(all_idx, elec_idx)
        sub_idx = np.concat([elec_idx, device_idx])
        Ham_NN_re = Ham_NN.sub(sub_idx)

        out = {
            "H_elec": Ham_rs,
            "elec_idx": elec_idx,
            "sub_idx": sub_idx,
            "total_atoms": Ham_NN_re.na,
        }
        return Ham_NN_re, out

    return


@app.cell
def _(
    all_armchair,
    bond_length: float,
    hopping_dist: tuple[float, float],
    hopping_term,
    sisl,
):
    graphene_cell: sisl.Geometry = all_armchair(bond=bond_length)

    Ham0 = sisl.Hamiltonian(geometry=graphene_cell)
    Ham0.construct(func=[hopping_dist, hopping_term])
    return Ham0, graphene_cell


@app.cell
def _(Ham0, N_big, N_small, eta, num_k, sisl):
    rsse_small = sisl.RealSpaceSE(Ham0, 0, 1, (N_small, N_small, 1))
    rsse_small.setup(eta=eta, bz=sisl.MonkhorstPack(Ham0, [1, num_k, 1]))

    rsse_big = sisl.RealSpaceSE(Ham0, 0, 1, (N_big, N_big, 1))
    rsse_big.setup(eta=eta, bz=sisl.MonkhorstPack(Ham0, [1, num_k, 1]))
    return rsse_big, rsse_small


@app.cell
def _(as_ham, as_ndarray, rsse_big, rsse_small):
    _out = rsse_small.real_space_coupling(True)
    Ham_elec_small = as_ham(_out[0])
    elec_idx_small = as_ndarray(_out[1])

    _out = rsse_big.real_space_coupling(True)
    Ham_elec_big = as_ham(_out[0])
    elec_idx_big = as_ndarray(_out[1])
    return Ham_elec_big, Ham_elec_small, elec_idx_big, elec_idx_small


@app.cell
def _(as_ham, rsse_big, rsse_small, sisl):
    Ham_NN_small: sisl.Hamiltonian = rsse_small.real_space_parent()
    Ham_NN_big: sisl.Hamiltonian = rsse_big.real_space_parent()
    Ham_NN_small = as_ham(Ham_NN_small)
    Ham_NN_big = as_ham(Ham_NN_big)
    return Ham_NN_big, Ham_NN_small


@app.cell
def _(
    Ham_NN_big: "sisl.Hamiltonian",
    Ham_NN_small: "sisl.Hamiltonian",
    as_ndarray,
    np,
):
    all_idx_small = as_ndarray(np.arange(Ham_NN_small.na))
    all_idx_big = as_ndarray(np.arange(Ham_NN_big.na))
    return all_idx_big, all_idx_small


@app.cell
def _(all_idx_big, all_idx_small, elec_idx_big, elec_idx_small, np):
    device_idx_small = np.delete(all_idx_small, elec_idx_small)
    device_idx_big = np.delete(all_idx_big, elec_idx_big)
    return device_idx_big, device_idx_small


@app.cell
def _(device_idx_big, device_idx_small, elec_idx_big, elec_idx_small, np):
    sub_idx_small = np.concat([elec_idx_small, device_idx_small])
    sub_idx_big = np.concat([elec_idx_big, device_idx_big])
    return sub_idx_big, sub_idx_small


@app.cell
def _(
    Ham_NN_big: "sisl.Hamiltonian",
    Ham_NN_small: "sisl.Hamiltonian",
    as_ham,
    sub_idx_small,
):
    Ham_reorder_small = as_ham(Ham_NN_small.sub(sub_idx_small))
    Ham_reorder_big = as_ham(Ham_NN_big.sub(sub_idx_small))
    return


@app.cell
def _(N_big, N_small, as_geom, graphene_cell: "sisl.Geometry", make_edge):
    geom_edge_small = as_geom(make_edge(graphene_cell, N_small, N_small))
    geom_edge_big = as_geom(make_edge(graphene_cell, N_big, N_big))
    return geom_edge_big, geom_edge_small


@app.cell
def _(Ham_NN_big: "sisl.Hamiltonian", Ham_NN_small: "sisl.Hamiltonian", cast):
    total_atoms_small: int = cast(int, Ham_NN_small.na)
    total_atoms_big: int = cast(int, Ham_NN_big.na)
    return total_atoms_big, total_atoms_small


@app.cell
def _(np):
    emax: float = 0.5
    emin: float = -emax
    estep = 0.1
    energies = np.arange(emin, emax, estep)
    return (energies,)


@app.cell
def _(Path, mo):
    rsse_out_dir = Path("./rsse")
    recompute_rsse_collection = mo.ui.checkbox(value=False, label="Recompute RSSEs")
    recompute_rsse_collection
    return recompute_rsse_collection, rsse_out_dir


@app.cell
def _(
    N_big,
    as_ndarray,
    energies,
    np,
    recompute_rsse_collection,
    rsse_big,
    rsse_out_dir,
    sub_idx_big,
    total_atoms_big: int,
    tqdm,
):
    _filename = f"calculate_rsse_N{N_big}"
    if recompute_rsse_collection.value is True:
        rsse_collection_big = np.zeros((energies.shape[0], total_atoms_big, total_atoms_big), dtype=complex)
        for _i, _E in enumerate(tqdm(energies, desc="Geometry: Big")):
            # print(i)
            _rsse = rsse_big.self_energy(_E)
            rsse_collection_big[_i] += _rsse[np.ix_(sub_idx_big, sub_idx_big)]
        np.savez(rsse_out_dir / f"calculate_rsse_N{N_big}", arr=rsse_collection_big)
    else:
        print(f"loaded {_filename}.npz")
        rsse_collection_big = as_ndarray(np.load(rsse_out_dir / f"{_filename}.npz")["arr"])
    return (rsse_collection_big,)


@app.cell
def _(
    N_small,
    as_ndarray,
    energies,
    np,
    recompute_rsse_collection,
    rsse_out_dir,
    rsse_small,
    sub_idx_small,
    total_atoms_small: int,
    tqdm,
):
    _filename =  f"calculate_rsse_N{N_small}"
    if recompute_rsse_collection.value is True:
        rsse_collection_small = np.zeros((energies.shape[0], total_atoms_small, total_atoms_small), dtype=complex)
        for _i, _E in enumerate(tqdm(energies, desc="Geometry: Small")):
            _rsse = rsse_small.self_energy(_E)
            rsse_collection_small[_i] += _rsse[np.ix_(sub_idx_small, sub_idx_small)]
        np.savez(rsse_out_dir /_filename, arr=rsse_collection_small)
    else:
        print(f"loaded {_filename}.npz")
        rsse_collection_small = as_ndarray(np.load(rsse_out_dir / f"{_filename}.npz")["arr"])
    return (rsse_collection_small,)


@app.cell
def _(
    Ham0,
    Ham_elec_big,
    Ham_elec_small,
    NC,
    N_big,
    N_small,
    cast,
    geom_edge_big,
    geom_edge_small,
    rsse_mapping,
):
    big_to_small_idx = rsse_mapping(Ham_elec_small, Ham_elec_big, geom_edge_small, geom_edge_big, N_small, N_big, Ham0.na, NC)
    big_to_small_idx = cast(dict[int, int], big_to_small_idx)
    mapped_indices = list(big_to_small_idx.values())
    return big_to_small_idx, mapped_indices


@app.cell
def _(
    elec_idx_big,
    mapped_indices,
    np,
    rsse_collection_big,
    rsse_collection_small,
):
    rsse_collection_extrapolation = np.zeros_like(rsse_collection_big)
    rsse_collection_extrapolation[:, :len(elec_idx_big), :len(elec_idx_big)] = rsse_collection_small[:, mapped_indices, :][:, :, mapped_indices]
    return (rsse_collection_extrapolation,)


@app.cell
def _(
    elec_idx_big,
    elec_idx_small,
    energies,
    plt,
    rsse_collection_big,
    rsse_collection_extrapolation,
    rsse_collection_small,
):
    _eidx = len(energies) // 2

    fig, axes = plt.subplots(1,3)
    for i, lab in enumerate(["Small", "Extrapol", "Big"]):
        axes[i].set(title=lab)

    axes[0].imshow(rsse_collection_small[_eidx, :len(elec_idx_small), :len(elec_idx_small)].imag)
    axes[1].imshow(rsse_collection_extrapolation[_eidx, :len(elec_idx_big), :len(elec_idx_big)].imag)
    axes[2].imshow(rsse_collection_big[_eidx, :len(elec_idx_big), :len(elec_idx_big)].imag)

    return


@app.cell
def _(big_to_small_idx, energies, np, rsse_collection_small):
    _eidx = len(energies) // 2
    rsse_collection_small[np.ix_([_eidx], list(big_to_small_idx.values()), list(big_to_small_idx.values()))].squeeze()
    return


if __name__ == "__main__":
    app.run()
