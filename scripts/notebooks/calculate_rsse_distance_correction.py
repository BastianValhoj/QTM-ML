import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import sisl
    import numpy as np
    from mytools.construct import all_armchair, make_edge
    from mytools.scalingv2 import rsse_mapping
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix

    from tqdm.auto import tqdm
    import matplotlib.pyplot as plt

    from pathlib import Path


@app.cell
def _():
    script_dir = Path(__file__).parent
    return (script_dir,)


@app.cell
def _(Ham0, N_small):
    make_edge(Ham0.geometry, N_small, N_small)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Create a distance informed extrapolation of rsse from a "`small`" structure to  "`big`" structure
    """)
    return


@app.cell
def _(script_dir):
    Ham0 = sisl.get_sile(script_dir / "Ham0.nc").read_hamiltonian()
    return (Ham0,)


@app.cell
def _():
    N_small = 6
    N_big = 13
    NC = 1

    eta = 1e-3
    nk1 = lambda N: int(np.ceil(1200/N))

    emax = 2.0
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
    return Ham_elec_big, Ham_re_big, elec_idx_big, nelec_big


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
    return big_to_small_idx, mapped_indices


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
            raise ValueError("This method is deprecated ")
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
def _():
    # invG_big = np.zeros(shape=(len(energies), na_big, na_big), dtype=np.complex128)
    # for _iE, _E in enumerate(tqdm(energies, desc="Big")):
    #     _invG = rsse_big.self_energy(_E, bulk=True, coupling=False)
    #     invG_big[_iE, :, :] = _invG
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Extrapolate the `small` $\Sigma(E)$
    """)
    return


@app.cell
def _(energies, mapped_indices, nelec_big, rsse_collection_small):
    rsse_collection_extra = np.zeros(shape=(len(energies), *(nelec_big,)*2), dtype=np.complex128)
    rsse_collection_extra[:, :nelec_big, :nelec_big] = rsse_collection_small[:, :, mapped_indices][:, mapped_indices, :]
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
    num_atoms_picker = mo.ui.dropdown(
        value=str(1), 
        options={str(key+1):int(val) for key, val in enumerate(np.arange(1,31))}, 
        label="Number of atoms for DOS"
    )
    num_atoms_picker
    return (num_atoms_picker,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Calculate DOS per (center) atoms
    """)
    return


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
def _():
    # _tree = cKDTree(Ham_re_big.xyz)
    # _dd, _ii = _tree.query(Ham_re_big.center(), k=num_center_atoms)
    # _ii = np.atleast_1d(_ii)
    # dos_big   = (-1/(num_center_atoms*np.pi))   * np.trace(np.linalg.inv(invG_big)[:, _ii, :][:, :, _ii].imag,   axis1=1, axis2=2)
    return


@app.cell
def _(Ham_re_big, invG_extra, num_center_atoms):
    _tree = cKDTree(Ham_re_big.xyz)
    _dd, _ii = _tree.query(Ham_re_big.center(), k=num_center_atoms)
    _ii = np.atleast_1d(_ii)
    dos_extra = (-1/(num_center_atoms*np.pi))   * np.trace(np.linalg.inv(invG_extra).imag[:, _ii, :][:, :, _ii], axis1=1, axis2=2)
    return (dos_extra,)


@app.cell(hide_code=True)
def _():
    # _fig, _axes = plt.subplots()
    # _axes.plot(energies, dos_small, label="small")
    # # _axes.plot(energies, dos_big, label="big")
    # _axes.plot(energies, dos_extra, label="extra")
    # _axes.legend()
    # _axes.grid()
    # mo.output.append([mo.md("### Plot DOS per atom"),_fig, num_atoms_picker])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Distance informed coupling
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Let $A_i$ be the $i$'th atom of `large` and $a_{i'}$ the $i$'th atom of `small`.
    It is not physical to use the same coupling between $\Sigma(A_{i}, A_j)$ as that of $\Sigma(a_{i'}, a_{j'})$, due to the size difference. Since the coupling decays with distance, if we naïvely used this coupling, then we would have a stronger coupling between two atoms that are much farther away in `larger` than what the same distance would result in for the `small`.

    We therefore adopt a correction using the following:
    1. For atoms $A_i$ and $A_j$ we calculate the distance: $\Delta R_{ij} = R_{i} - R_j$.
    2. We need to find the equivalent coupling pair in `small`: $a_{i'}$ and $a_{j'}$ (with $a_{i'}$ being the on-site _"parent"_ of $A_i$).
    3. We determine the parent of $A_j$ as the atom $a_{j'}$ that has the position $r_{j'} = r_{i'} + \Delta{R}_{ij}$
    """)
    return


@app.cell
def _(Ham_elec_big, Ham_elec_small):
    xyz_small = Ham_elec_small.xyz
    xyz_big = Ham_elec_big.xyz
    return xyz_big, xyz_small


@app.cell
def _(big_to_small_idx, xyz_big, xyz_small):
    tree_small = cKDTree(xyz_small) 
    def match_coupling_atoms(i_big, j_big, *, tol=0.1):
        pairA = None
        pairB = None


        i_small = big_to_small_idx[i_big] # Parent atom i in `small`
        dR_ji = xyz_big[j_big] - xyz_big[i_big] # distance between atoms i and j in `big`
        r_j = xyz_small[i_small] + dR_ji
        dr_ji, j_small = tree_small.query(r_j, k=1) # find the nearest atom in `small` to position r_j
        # if the distance is within the tolerence, use the indicies as a pair
        if dr_ji < tol:
            pairA = (i_small, j_small)

        # Do the same but reversed
        j_small = big_to_small_idx[j_big]
        dR_ij = xyz_big[i_big] - xyz_big[j_big]
        r_i = xyz_small[j_small] + dR_ij
        dr_ij, i_small = tree_small.query(r_i, k=1)
        if dr_ij < tol:
            pairB = (i_small, j_small)
        return pairA, pairB

    return (match_coupling_atoms,)


@app.cell
def _(match_coupling_atoms, nelec_big, nelec_small, xyz_big):
    tree_big = cKDTree(xyz_big)
    def scale_coupling(small, big):
        MaxRange = np.max(np.linalg.norm(small.cell, axis=1))
        rows = []
        cols = []
        data = []
        for i_big in tqdm(range(big.na), desc="Looping over indices in big"):
            dist, idx = tree_big.query(big.xyz[i_big], distance_upper_bound=MaxRange+0.1, k=int(small.na))
            # print(idx[dist < MaxRange])
            for j_big in idx[dist < MaxRange]:
                row = i_big * nelec_big + j_big
                pairs = []
                pairA, pairB = match_coupling_atoms(i_big, j_big)
                if not (pairA is None):
                    pairs.append(pairA)
                if not (pairB is None):
                    pairs.append(pairB)
                if (pairA is None) and (pairB is None):
                    continue
                weight = 1. / len(pairs) # how to everage between coupling (a_i', a_j') and (a_j', a_i')
                for i_small, j_small in pairs:
                    col = i_small * nelec_small + j_small 
                    rows.append(row)
                    cols.append(col)
                    data.append(weight)
        print(len(rows), len(cols), len(data))
        return coo_matrix((data, (rows, cols)), shape=(nelec_big**2, nelec_small**2)).tocsr() 

    return (scale_coupling,)


@app.cell
def _():
    return


@app.cell
def _(Ham_elec_big, Ham_elec_small, scale_coupling):
    scaling = scale_coupling(Ham_elec_small, Ham_elec_big)
    return (scaling,)


@app.cell
def _(energies, nelec_big, rsse_collection_small, scaling):
    # rsse_collection_dist = np.array([
    #     distance_correction @ rsse_collection_small[_eidx].ravel()
    #     for _eidx in range(len(energies))
    # ]).reshape(-1, nelec_big, nelec_big)
    small_flat = rsse_collection_small.reshape(len(energies), -1)
    print(scaling.shape, small_flat.shape)
    rsse_collection_dist = (small_flat @ scaling.T).reshape(len(energies), nelec_big, nelec_big)
    return (rsse_collection_dist,)


@app.cell
def _(Ham_re_big, energies, eta, nelec_big, rsse_collection_dist):
    invG_dist = (energies[:, None, None] + eta*1j)*Ham_re_big.Sk(format="array")[None, :, :] - Ham_re_big.Hk(format="array")[None, :, :]
    invG_dist[:, :nelec_big, :nelec_big] += - rsse_collection_dist
    return (invG_dist,)


@app.cell
def _(Ham_re_big, invG_dist, num_center_atoms):
    _tree = cKDTree(Ham_re_big.xyz)
    _dd, _ii = _tree.query(Ham_re_big.center(), k=num_center_atoms)
    _ii = np.atleast_1d(_ii)
    dos_dist = (-1./(num_center_atoms*np.pi)) * np.trace(np.linalg.inv(invG_dist).imag[:, _ii, :][:, :, _ii], axis1=1, axis2=2)
    return (dos_dist,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Plot DOS per atom
    """)
    return


@app.cell(hide_code=True)
def _(
    N_big,
    N_small,
    dos_dist,
    dos_extra,
    dos_small,
    energies,
    num_atoms_picker,
):
    from mytools.plots import thesis_fig
    _fig, _axes = plt.subplots(1, 2, sharey=True, figsize=(8,4))
    # _fig, _axes = thesis_fig(subplots=(1,2), sharey=True)
    # _axes[0].plot(energies, dos_small, label="small")
    _axes[0].plot(energies, dos_small, label="small", color="k")
    _axes[0].plot(energies, dos_extra, label="extra")
    _axes[0].plot(energies, dos_dist, label="dist", color="red")
    _axes[0].set(title=r"(L)DOS per atom")

    _axes[1].plot(energies, dos_small - dos_extra, label="extra")
    _axes[1].plot(energies, dos_small - dos_dist, label="dist", color="red")
    _axes[1].set(title=r"difference with `$small$` (per atom)")
    _fig.suptitle(fr"N : {N_small} $\to$ {N_big}")
    for _ax in _axes:
        _ax.legend()
        _ax.grid()
    mo.output.append([_fig, num_atoms_picker])
    return


@app.cell
def _(rsse_collection_dist, rsse_collection_small):
    _fig, _ax = plt.subplots()

    print("Original (small)")
    print("max Im(Sigma):", rsse_collection_small.imag.max())
    print("min Im(Sigma):", rsse_collection_small.imag.min())

    print("Extrapolation")
    print("max Im(Sigma): ", rsse_collection_dist.imag.max())
    print("min Im(Sigma): ", rsse_collection_dist.imag.min())

    _ax.imshow(rsse_collection_dist.imag[0])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Compute RMSE for both methods
    """)
    return


@app.cell
def _(dos_dist, dos_extra, dos_small):
    def rmse(arr1, arr2):
        return np.sqrt(
            np.mean( (arr1 - arr2)**2)
        )
    print("RMSE extra : {:.3e}".format(rmse(dos_small, dos_extra)))
    print("RMSE dist  : {:.3e}".format(rmse(dos_small, dos_dist)))
    return


@app.cell
def _(Ham_re_big, elec_idx_big):
    from ase.visualize import view

    atoms = np.concatenate([
        [343, 347, 405],
        np.arange(408, 425),
        [427],
        np.arange(475, 500),
        np.arange(544, 573),
        np.arange(617, 642),
        [690],
        np.arange(692, 709),
        [712, 770, 774]
    ]) - 1
    # view(Ham_re_big.geometry.to.ase())

    Ham_re_big.geometry.plot(axes="xy", atoms_style=[
        dict(atoms=atoms, color="red"),
        dict(atoms=range(len(elec_idx_big)), color="blue")
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Easy to use method for calculating the RSSE extrapolated (with distance correction)
    """)
    return


@app.cell
def _(N_big, N_small):
    SCRIPT_DIR = Path(__file__).parent
    OUTPUT_DIR = SCRIPT_DIR / 'rsse_data' / f"TBT-test_{N_small}_to_{N_big}"
    OUTPUT_DIR.relative_to(SCRIPT_DIR)
    return


@app.cell
def _(Ham0, Ham_elec_big, Ham_re_big):
    Ham_elec_big.nsc, Ham_re_big.nsc, Ham0.nsc
    return


@app.cell
def _(N_big, nk1):
    nk1(N_big)
    return


@app.cell
def _(Ham0, N_big, nk1):
    sisl.MonkhorstPack(Ham0, [1,nk1(N_big), 1]).k
    return


@app.cell
def _(Ham_elec_big):
    sisl.BrillouinZone(Ham_elec_big).k
    return


if __name__ == "__main__":
    app.run()
