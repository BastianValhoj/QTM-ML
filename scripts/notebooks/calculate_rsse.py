# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.2",
# ]
# ///

import marimo

__generated_with = "0.23.2"
app = marimo.App()


@app.cell
def _():
    import sisl 
    import numpy as np
    from tqdm.auto import tqdm

    from mytools.construct import all_armchair, make_edge
    from mytools.scalingv2 import extrapolate, rsse_mapping

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import marimo as mo

    from pathlib import Path
    import os

    from typing import cast, Tuple, Dict, List, Any

    return (
        Any,
        Dict,
        Path,
        Tuple,
        all_armchair,
        cast,
        make_axes_locatable,
        make_edge,
        mo,
        np,
        os,
        plt,
        rsse_mapping,
        sisl,
        tqdm,
    )


@app.cell
def _(Path):
    script_dir = Path(__file__).parent.absolute()
    rsse_out_dir = script_dir / "rsse"
    rsse_out_dir.mkdir(parents=True, exist_ok=True)
    return (rsse_out_dir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Make checkbox (✅) for recomputing RSSE's
    """)
    return


@app.cell
def _(mo):
    recompute_rsse_collection = mo.ui.checkbox(value=False, label="Recompute RSSEs")
    return (recompute_rsse_collection,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Allow for using recompute when running as a script
    """)
    return


@app.cell
def _(os, recompute_rsse_collection):
    force_recompute_env: bool = os.getenv("RECOMPUTE", "false").lower() == "true"

    should_recompute: bool = recompute_rsse_collection.value or force_recompute_env
    return (should_recompute,)


@app.cell
def _(recompute_rsse_collection):
    recompute_rsse_collection
    return


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
    sub_idx_big,
    sub_idx_small,
):
    Ham_reorder_small = as_ham(Ham_NN_small.sub(sub_idx_small))
    Ham_reorder_big = as_ham(Ham_NN_big.sub(sub_idx_big))
    return Ham_reorder_big, Ham_reorder_small


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
    emax: float = 1.
    emin: float = -emax
    estep = 0.1
    energies = np.arange(emin, emax, estep)
    return (energies,)


@app.cell
def _(recompute_rsse_collection):
    recompute_rsse_collection
    return


@app.cell
def _(
    N_big,
    as_ndarray,
    energies,
    np,
    rsse_big,
    rsse_out_dir,
    should_recompute: bool,
    sub_idx_big,
    total_atoms_big: int,
    tqdm,
):
    _filename = f"calculate_rsse_N{N_big}"
    if should_recompute is True:
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
    rsse_out_dir,
    rsse_small,
    should_recompute: bool,
    sub_idx_small,
    total_atoms_small: int,
    tqdm,
):
    _filename =  f"calculate_rsse_N{N_small}"
    if should_recompute is True:
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
    return (mapped_indices,)


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
    make_axes_locatable,
    np,
    plt,
    rsse_collection_big,
    rsse_collection_extrapolation,
    rsse_collection_small,
):
    _eidx = len(energies) // 2

    _fig, _axes = plt.subplots(1,3)
    for _i, _lab in enumerate(["Small", "Extrapol", "Big"]):
        _axes[_i].set(title=_lab)

    _cmap = "RdBu"
    _vmax: float = np.max([np.abs(rsse_collection_small[_eidx, :len(elec_idx_small), :len(elec_idx_small)].imag).max(), np.abs(rsse_collection_big[_eidx, :len(elec_idx_big), :len(elec_idx_big)].imag).max(), np.abs(rsse_collection_extrapolation[_eidx, :len(elec_idx_big), :len(elec_idx_big)].imag).max()])
    _vmin: float = -_vmax

    _axes[0].imshow(rsse_collection_small[_eidx, :len(elec_idx_small), :len(elec_idx_small)].imag, cmap=_cmap, vmin=_vmin, vmax=_vmax)
    _axes[1].imshow(rsse_collection_extrapolation[_eidx, :len(elec_idx_big), :len(elec_idx_big)].imag, cmap=_cmap, vmin=_vmin, vmax=_vmax)
    _sc = _axes[2].imshow(rsse_collection_big[_eidx, :len(elec_idx_big), :len(elec_idx_big)].imag, cmap=_cmap, vmin=_vmin, vmax=_vmax)

    _divider = make_axes_locatable(_axes[2])
    _cax = _divider.append_axes("right", size="5%", pad=0.1)
    _fig.colorbar(_sc, _cax)
    plt.show()
    return


@app.cell
def _(Ham_reorder_big, Ham_reorder_small):
    Hk_small = Ham_reorder_small.Hk(format="array")
    Sk_small = Ham_reorder_small.Sk(format="array")

    Hk_big = Ham_reorder_big.Hk(format="array")
    Sk_big = Ham_reorder_big.Sk(format="array")
    return Hk_big, Hk_small, Sk_big, Sk_small


@app.cell
def _(Hk_small, Sk_small, energies, eta, np, rsse_collection_small):
    invGF_small = (energies[:, None, None] + eta*1j)*Sk_small[None, :, :] - Hk_small[None, :, :] - rsse_collection_small
    GF_small = np.linalg.inv(invGF_small)

    ldos_small = (-1/np.pi)*np.imag(np.diagonal(GF_small, axis1=1, axis2=2))
    dos_small = ldos_small.sum(axis=-1)
    return (dos_small,)


@app.cell
def _(Ham_reorder_big, Ham_reorder_small):
    print(Ham_reorder_big.na, Ham_reorder_small.na)
    return


@app.cell
def _(Hk_big, Sk_big, energies, eta, np, rsse_collection_big):
    invGF_big = (energies[:, None, None] + eta*1j)*Sk_big[None, :, :] - Hk_big[None, :, :] - rsse_collection_big
    GF_big = np.linalg.inv(invGF_big)

    ldos_big = (-1/np.pi)*np.imag(np.diagonal(GF_big, axis1=1, axis2=2))
    dos_big = ldos_big.sum(axis=-1)
    return (dos_big,)


@app.cell
def _(Hk_big, Sk_big, energies, eta, np, rsse_collection_extrapolation):
    invGF_extrapolation = (energies[:, None, None] + eta*1j)*Sk_big[None, :, :] - Hk_big[None, :, :] - rsse_collection_extrapolation
    GF_extrapolation = np.linalg.inv(invGF_extrapolation)

    ldos_extrapolation= (-1/np.pi)*np.imag(np.diagonal(GF_extrapolation, axis1=1, axis2=2))
    dos_extrapolation = ldos_extrapolation.sum(axis=-1)
    return (dos_extrapolation,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Purely extrapolation solution
    """)
    return


@app.cell
def _(dos_big, dos_extrapolation, dos_small, energies, plt):
    _eidx = len(energies) // 2

    _fig, _axes = plt.subplots(1, 2, sharey=True)

    for _ax, _lab in zip(_axes, ["Small", "Extra + Big"]):
        _ax.grid()
        _ax.set(
            title=_lab,
            # ylabel="$E$ [eV]",
            xlabel="DOS")
    _axes[0].set_ylabel("$E$ [eV]")
    _axes[0].plot(dos_small, energies, color="royalblue")
    _axes[1].plot(dos_extrapolation, energies, linestyle="--", color="crimson", label="Extrapolation")
    _axes[1].plot(dos_big, energies, linestyle="-", color="royalblue", label="Big")
    _axes[1].legend(loc="center right")
    _fig.get_constrained_layout()
    plt.show()
    return


@app.cell
def _(Ham_reorder_big, Ham_reorder_small, elec_idx_big, elec_idx_small):
    diff_small = Ham_reorder_small.xyz[:len(elec_idx_small), None, :] - Ham_reorder_small.xyz[None, :len(elec_idx_small), :]
    diff_big = Ham_reorder_big.xyz[:len(elec_idx_big), None, :] - Ham_reorder_big.xyz[None, :len(elec_idx_big), :]
    return diff_big, diff_small


@app.cell
def _(diff_big, diff_small, np):
    dist_small = np.linalg.norm(diff_small, axis=-1)  # (N_small, N_small)
    dist_big   = np.linalg.norm(diff_big,   axis=-1)  # (N_big,   N_big)
    return dist_big, dist_small


@app.cell
def _(dist_small):
    max_dist_small = dist_small.max(axis=-1)
    # max_dist_big = dist_big.max(axis=-1)
    return (max_dist_small,)


@app.cell
def _(dist_big, mapped_indices, max_dist_small):
    dist_mask = dist_big > max_dist_small[mapped_indices]
    return (dist_mask,)


@app.cell
def _(dist_mask, elec_idx_big, np, rsse_collection_extrapolation):
    elec_block = rsse_collection_extrapolation[:, :len(elec_idx_big), :len(elec_idx_big)]
    rsse_collection_extrapolation_corrected = rsse_collection_extrapolation.copy()
    rsse_collection_extrapolation_corrected[:, :len(elec_idx_big), :len(elec_idx_big)] = np.where(dist_mask, 0.0, elec_block)
    return (rsse_collection_extrapolation_corrected,)


@app.cell
def _(
    Hk_big,
    Sk_big,
    energies,
    eta,
    np,
    rsse_collection_extrapolation_corrected,
):
    invGF_corrected = (energies[:, None, None] + eta*1j)*Sk_big[None, :, :] - Hk_big[None, :, :] - rsse_collection_extrapolation_corrected
    GF_corrected = np.linalg.inv(invGF_corrected)

    ldos_corrected = (-1/np.pi)*np.imag(np.diagonal(GF_corrected, axis1=1, axis2=2))
    dos_corrected = ldos_corrected.sum(axis=-1)
    return (dos_corrected,)


@app.cell
def _(dos_big, dos_corrected, dos_extrapolation, dos_small, energies, plt):
    _eidx = len(energies) // 2

    _fig, _axes = plt.subplots(1, 2, sharey=True)

    for _ax, _lab in zip(_axes, ["Small", "Extra + Big"]):
        _ax.grid()
        _ax.set(
            title=_lab,
            # ylabel="$E$ [eV]",
            xlabel="DOS")
    _axes[0].set_ylabel("$E$ [eV]")
    _axes[0].plot(dos_small, energies, color="royalblue")
    _axes[1].plot(dos_extrapolation, energies, linestyle="--", color="crimson", label="Extrapolation")
    _axes[1].plot(dos_corrected, energies, linestyle="-.", color="darkgreen", label="Corrected")
    _axes[1].plot(dos_big, energies, linestyle="-", color="royalblue", label="Big")
    _axes[1].legend(loc="center right")
    _fig.get_constrained_layout()

    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
