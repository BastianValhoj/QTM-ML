import marimo

__generated_with = "0.23.4"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import sisl
    import numpy as np
    from mytools.construct import all_armchair, make_edge
    from mytools.scalingv2 import rsse_mapping
    from scipy.spatial import cKDTree

    from tqdm.auto import tqdm
    import matplotlib.pyplot as plt

    from pathlib import Path


@app.cell
def _():
    script_dir = Path(__file__).parent
    return (script_dir,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Create a single function that extrapolates the rsse of a *small* structure to a structure of tiles `N0*N1`
    """)
    return


@app.cell
def _(script_dir):
    Ham0 = sisl.get_sile(script_dir / "Ham0.nc").read_hamiltonian()
    return (Ham0,)


@app.cell
def _():
    N_small = 6
    N_big = 11
    NC = 1

    eta = 1e-3
    nk1 = lambda N: int(np.ceil(1200/N))

    emax = 1.0
    emin = -emax
    estep = 0.1
    energies = np.arange(emin, emax+estep, estep).round(3)
    return NC, N_big, N_small, energies, eta, nk1


@app.cell
def _(Ham0, N_big, N_small):
    geom_edge_small = make_edge(geom=Ham0.geometry, N0=N_small, N1=N_small)
    geom_edge_big = make_edge(geom=Ham0.geometry, N0=N_big, N1=N_big)
    return geom_edge_big, geom_edge_small


@app.function
def resub_ham(rsse):
    Ham_elec, elec_idx = rsse.real_space_coupling(ret_indices=True)
    Ham_NN = rsse.real_space_parent()

    all_idx = np.arange(Ham_NN.na)
    device_idx = np.delete(all_idx, elec_idx)
    sub_idx = np.concat([elec_idx, device_idx])
    Ham_reorder = Ham_NN.sub(sub_idx)

    resub_dict = {
        "Ham_elec": Ham_elec,
        "Ham_re": Ham_reorder,
        "sub_idx": sub_idx,
        "elec_idx": elec_idx,
    }
    return resub_dict


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Create small: rsse obj, dict, electrode, sub_idx, Ham_re
    """)
    return


@app.cell
def _(Ham0, N_small, eta, nk1):
    rsse_small = sisl.RealSpaceSE(Ham0, 0, 1, (N_small, N_small, 1))
    rsse_small.setup(eta=eta, bz=sisl.MonkhorstPack(Ham0, [1, nk1(N_small), 1]))
    return (rsse_small,)


@app.cell
def _(rsse_small):
    resub_dict_small = resub_ham(rsse_small)
    Ham_elec_small = resub_dict_small["Ham_elec"]
    Ham_re_small = resub_dict_small["Ham_re"]
    elec_idx_small = resub_dict_small["elec_idx"]
    sub_idx_small = resub_dict_small["sub_idx"]
    na_small = Ham_re_small.na
    nelec_small = len(elec_idx_small)

    print(Ham_re_small.nsc)
    return Ham_elec_small, Ham_re_small, na_small, nelec_small, sub_idx_small


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Create small: rsse obj, dict, electrode, sub_idx, Ham_re
    """)
    return


@app.cell
def _(Ham0, N_big, eta, nk1):
    rsse_big = sisl.RealSpaceSE(Ham0, 0, 1, (N_big, N_big, 1))
    rsse_big.setup(eta=eta, bz=sisl.MonkhorstPack(Ham0, [1, nk1(N_big), 1]))
    return (rsse_big,)


@app.cell
def _(Ham_re_small, rsse_big):
    resub_dict_big = resub_ham(rsse_big)
    Ham_elec_big = resub_dict_big["Ham_elec"]
    Ham_re_big = resub_dict_big["Ham_re"]
    elec_idx_big = resub_dict_big["elec_idx"]
    sub_idx_big = resub_dict_big["sub_idx"]
    na_big = Ham_re_big.na
    nelec_big = len(elec_idx_big)

    print(Ham_re_small.nsc)
    return Ham_elec_big, Ham_re_big, na_big, nelec_big


@app.cell
def _(
    Ham0,
    Ham_elec_big,
    Ham_elec_small,
    NC,
    N_big,
    N_small,
    geom_edge_big,
    geom_edge_small,
):
    big_to_small_idx  = rsse_mapping(Ham_elec_small, Ham_elec_big, geom_edge_small, geom_edge_big, N_small, N_big, Ham0.na, NC)
    mapped_indices = list(big_to_small_idx.values())
    return (mapped_indices,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Compute $\Sigma(E)$ for the `small`
    """)
    return


@app.cell
def _(
    Ham_re_small,
    energies,
    eta,
    na_small,
    nelec_small,
    rsse_small,
    sub_idx_small,
):
    invG_small = np.zeros(shape=(len(energies), na_small, na_small), dtype=np.complex128)
    rsse_collection_small = np.zeros(shape=(len(energies), *(nelec_small,)*2), dtype=np.complex128)
    for _iE, _E in enumerate(tqdm(energies, desc="Small")):
        # standard parameters that is needed no matter what
        _z = _E + eta*1j
        _Hk = Ham_re_small.Hk()
        _Sk = Ham_re_small.Sk()
        _invG = (_z*_Sk - _Hk)
        invG_small[_iE, :, :] += _invG 

        if rsse_collection_small.shape[1] == na_small: # if we use the dense/full memory version
            _RSSE = rsse_small.self_energy(_z, bulk=False, coupling=False)[np.ix_(sub_idx_small, sub_idx_small)]
            invG_small[_iE, :, :] += - _RSSE

        elif rsse_collection_small.shape[1] == nelec_small: # If we use the sparse/reduced memory version (only the coupling terms are included in RSSE matrix)
            _RSSE = rsse_small.self_energy(_z, bulk=False, coupling=True)
            invG_small[_iE, :nelec_small, :nelec_small] += - _RSSE

        rsse_collection_small[_iE, :, :] += _RSSE
    return invG_small, rsse_collection_small


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Compute $\Sigma(E)$ for the `big`
    """)
    return


@app.cell
def _(energies, na_big, rsse_big):
    invG_big = np.zeros(shape=(len(energies), na_big, na_big), dtype=np.complex128)
    for _iE, _E in enumerate(tqdm(energies, desc="test")):
        _invG = rsse_big.self_energy(_E, bulk=True, coupling=False)
        invG_big[_iE, :, :] = _invG
    return (invG_big,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Extrapolate the `small` $\Sigma(E)$
    """)
    return


@app.cell
def _(energies, mapped_indices, nelec_big, rsse_collection_small):
    rsse_collection_extra = np.zeros(shape=(len(energies), *(nelec_big,)*2), dtype=np.complex128)
    rsse_collection_extra[:, :nelec_big, :nelec_big]  = rsse_collection_small[:, :, mapped_indices][:, mapped_indices, :]
    return (rsse_collection_extra,)


@app.cell
def _(rsse_collection_extra, rsse_collection_small):
    _fig, _ax = plt.subplots(1,2)
    _eidx = 6
    _ax[0].imshow(rsse_collection_small[_eidx].imag)
    _ax[1].imshow(rsse_collection_extra[_eidx].imag)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Show DOS for `big` and `extra`
    """)
    return


@app.cell
def _(Ham_re_big, energies, eta, nelec_big, rsse_collection_extra):
    invG_extra = (energies[:, None, None] + eta*1j)*Ham_re_big.Sk(format="array")[None, :, :] - Ham_re_big.Hk(format="array")[None, :, :]
    invG_extra[:, :nelec_big, :nelec_big] += - rsse_collection_extra
    return (invG_extra,)


@app.cell
def _():
    num_atoms_picker = mo.ui.dropdown(value=str(1), options={str(key+1):int(val) for key, val in enumerate(np.arange(1,31))}, label="Number of atoms for LDOS")
    num_atoms_picker
    return (num_atoms_picker,)


@app.cell
def _(num_atoms_picker):
    num_center_atoms = num_atoms_picker.value
    return (num_center_atoms,)


@app.cell
def _(Ham_re_small, invG_small, num_center_atoms):
    _tree = cKDTree(Ham_re_small.xyz)
    _dd, _ii = _tree.query(Ham_re_small.center(), k=num_center_atoms)
    _ii = np.atleast_1d(_ii)
    dos_small = (-1/(num_center_atoms*np.pi)) * np.trace(np.linalg.inv(invG_small).imag[:, _ii, :][:,:,_ii], axis1=1, axis2=2)
    return (dos_small,)


@app.cell
def _(Ham_re_big, invG_big, num_center_atoms):
    _tree = cKDTree(Ham_re_big.xyz)
    _dd, _ii = _tree.query(Ham_re_big.center(), k=num_center_atoms)
    _ii = np.atleast_1d(_ii)
    dos_big   = (-1/(num_center_atoms*np.pi))   * np.trace(np.linalg.inv(invG_big)[:, _ii, :][:, :, _ii].imag,   axis1=1, axis2=2)
    return (dos_big,)


@app.cell
def _(Ham_re_big, invG_extra, num_center_atoms):
    _tree = cKDTree(Ham_re_big.xyz)
    _dd, _ii = _tree.query(Ham_re_big.center(), k=num_center_atoms)
    _ii = np.atleast_1d(_ii)
    dos_extra = (-1/(num_center_atoms*np.pi))   * np.trace(np.linalg.inv(invG_extra).imag[:, _ii, :][:, :, _ii], axis1=1, axis2=2)
    return (dos_extra,)


@app.cell(hide_code=True)
def _(dos_big, dos_extra, dos_small, energies, num_atoms_picker):
    _fig, _axes = plt.subplots()
    _axes.plot(energies, dos_small, label="small")
    _axes.plot(energies, dos_big, label="big")
    _axes.plot(energies, dos_extra, label="extra")
    _axes.legend()
    _axes.grid()
    _fig, num_atoms_picker
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Distance informed coupling
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
