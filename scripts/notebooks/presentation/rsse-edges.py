import marimo

__generated_with = "0.23.9"
app = marimo.App()

with app.setup:
    import sisl
    import numpy as np
    import matplotlib.pyplot as plt
    import h5py
    import matplotlib.colors as mcolor
    import matplotlib.animation as animation
    import marimo as mo

    from tqdm.auto import tqdm

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.ticker import LogFormatterSciNotation, ScalarFormatter
    import matplotlib.ticker as mticker

    from pathlib import Path

    from mytools.construct import all_armchair


@app.cell
def _():
    NOTEBOOK = Path(__file__)
    NOTEBOOK_DIR = NOTEBOOK.parent
    return NOTEBOOK, NOTEBOOK_DIR


@app.cell
def _():
    zigzag = sisl.geom.graphene()
    armchair = all_armchair()
    return armchair, zigzag


@app.cell
def _(zigzag):
    zigzag.plot(axes="xy", backend="matplotlib")
    return


@app.cell
def _(armchair):
    armchair.plot(axes="xy", backend="matplotlib")
    return


@app.cell
def _():
    eta = 1e-3
    nk1 = int(np.ceil(2400/12))
    energies = np.arange(0, 0.15, 0.1)

    Rs = (0.1, 1.44)
    Ts = (0, -2.7)
    return Rs, Ts, energies, eta, nk1


@app.cell
def _(Rs, Ts, armchair, zigzag):
    Ham0_z = sisl.Hamiltonian(zigzag)
    Ham0_a = sisl.Hamiltonian(armchair)

    N_z = 12
    N_a = 7

    for _H in [Ham0_a, Ham0_z]:
        _H.construct([Rs, Ts])
    return Ham0_a, Ham0_z, N_a, N_z


@app.cell
def _(Ham0_a, Ham0_z, N_a, N_z, eta, nk1):
    rsse_z = sisl.RealSpaceSE(Ham0_z, 0, 1, (N_z, N_z, 1))
    rsse_z.setup(eta=eta, bz=sisl.MonkhorstPack(Ham0_z, [1,nk1,1]))
    rsse_a = sisl.RealSpaceSE(Ham0_a, 0, 1, (N_a, N_a, 1))
    rsse_a.setup(eta=eta, bz = sisl.MonkhorstPack(Ham0_a, [1,nk1,1]))
    return rsse_a, rsse_z


@app.cell
def _(rsse_a, rsse_z):
    rsse_a.real_space_parent().na, rsse_z.real_space_parent().na
    return


@app.function
def reorder_ham(rsse, ret_idx=False):
    ham_unfold = rsse.real_space_parent()
    _, idx_elec = rsse.real_space_coupling(True)
    all_idx = np.arange(ham_unfold.no)
    dev_idx = np.setdiff1d(all_idx, idx_elec)
    idx = np.concat([idx_elec, dev_idx])
    if ret_idx:
        return idx
    else:
        return ham_unfold.sub(idx)


@app.cell
def _(rsse_a, rsse_z):
    Ham_re_z = reorder_ham(rsse_z)
    Ham_re_a = reorder_ham(rsse_a)
    return Ham_re_a, Ham_re_z


@app.cell
def _(NOTEBOOK_DIR, energies, eta, nk1, rsse_a, rsse_z):
    with h5py.File(NOTEBOOK_DIR / "rsse-edges_data.h5", "w") as f:
        f.attrs["eta"] = eta
        f.attrs["nk1"] = nk1

        f.create_dataset("energies", data=energies )
        _num_E = len(energies)

        for _edge, _rsse in [("armchair", rsse_a), ("zigzag", rsse_z)]:
            _no = _rsse.real_space_parent().no
            _ds = f.create_dataset(_edge, shape=(_num_E, _no, _no), dtype=np.complex128)
            _idx = reorder_ham(_rsse, ret_idx=True)
            for _i, _E in enumerate(tqdm(energies, desc=f"looping {_edge}")):
                _ds[_i] = _rsse.self_energy(_E, [0,0,0], bulk=False)[np.ix_(_idx, _idx)]
    return


@app.cell
def _(N_a, N_z):
    site_z = mo.ui. slider(0, N_z, label="site Zigzag", value=0, debounce=True, show_value=True, include_input=True)
    site_a = mo.ui.slider(0, N_a*2+1, label="site Armchair", value=0, debounce=True, show_value=True, include_input=True)

    params = mo.hstack([site_z, site_a], justify="start")
    return params, site_a, site_z


@app.cell
def _(
    Ham_re_a,
    Ham_re_z,
    NOTEBOOK,
    NOTEBOOK_DIR,
    energies,
    params,
    site_a,
    site_z,
):
    _cmap = "RdBu"
    _fig, _ax = plt.subplots(2, len(energies), figsize=(10,5))

    with h5py.File(NOTEBOOK_DIR / f"{NOTEBOOK.stem}_data.h5", "r") as _f:
        _energies = np.asarray(_f["energies"])
        _SE_z = np.asarray(_f["zigzag"]).imag    # (num_E, no, no)
        _SE_a = np.asarray(_f["armchair"]).imag

    _vmax_z = np.max(np.abs(_SE_z))
    _vmin_z = -_vmax_z 
    _vmax_a = np.max(_SE_a)
    _vmin_a = -_vmax_a



    _xyz_z = Ham_re_z.xyz
    _xyz_a = Ham_re_a.xyz

    _site_z = site_z.value
    _site_a = site_a.value

    _size = 10

    for _i, _E in enumerate(tqdm(_energies)):
        _SE_site_z = _SE_z[_i, _site_z, :]
        _SE_site_a = _SE_a[_i, _site_a, :]
        _vmax_z = np.max(np.abs(_SE_site_z))
        _vmin_z = -_vmax_z 
        _vmax_a = np.max(np.abs(_SE_site_a))
        _vmin_a = -_vmax_a
        sc_z = _ax[0, _i].scatter(*_xyz_z[:, :2].T, s=_size, c=_SE_site_z, cmap=_cmap, 
            vmin=_vmin_z, vmax=_vmax_z,
        )
        sc_a = _ax[1, _i].scatter(*_xyz_a[:, :2].T, s=_size, c=_SE_site_a, cmap=_cmap, 
            vmin=_vmin_a, vmax=_vmax_a,
        )
        _ax[0, _i].set_title(f"E={_E:.1f} eV")

        _ax[0,_i].annotate(_site_z, xy=_xyz_z[_site_z, :2], xytext=(15, 15), xycoords="data", textcoords="offset points",
            arrowprops=dict(arrowstyle="->")
        )
        _ax[1,_i].annotate(_site_a, xy=_xyz_a[_site_a, :2], xytext=(15, 15), xycoords="data", textcoords="offset points",
            arrowprops=dict(arrowstyle="->")
        )
        for _sc, _a in [(sc_z, _ax[0, _i]), (sc_a, _ax[1, _i])]:
            _cax = inset_axes(_a, width="2%", height="50%", loc="lower left")
            _cb = _fig.colorbar(_sc, cax=_cax)
            _cb.ax.set_title(r"Im($\Sigma_{ij}$)", pad=4)
            _fmt = mticker.ScalarFormatter(useMathText=True)
            _fmt.set_powerlimits((0, 0))  # force sci notation for all values
            _cb.formatter = _fmt
            _cb.ax.yaxis.get_offset_text().set_va("center")
            _cb.ax.yaxis.get_offset_text().set_ha("left")
            _cb.update_ticks()

    for _a in _ax.flatten():
        _a.axis("equal")
        _a.set(xticks=[], yticks=[])
        _a.axis("off")


    # fig.set_constrained_layout(True)
    _fig.tight_layout()
    mo.vstack([params, _fig])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Make it a GIF
    """)
    return


@app.cell
def _(Ham_re_a, Ham_re_z, NOTEBOOK, NOTEBOOK_DIR):


    def make_frame(fig, ax, _site_z, _site_a):
        # remove all inset axes (anything that's not in the main ax grid)
        main_axes = set(ax.flatten())
        for _a in fig.axes[:]:
            if _a not in main_axes:
                _a.remove()

        for _a in ax.flatten():
            _a.cla()

        for _i, _E in enumerate(_energies):
            _SE_site_z = _SE_z[_i, _site_z, :]
            _SE_site_a = _SE_a[_i, _site_a, :]
            _vmax_z = np.max(np.abs(_SE_site_z))
            _vmax_a = np.max(np.abs(_SE_site_a))

            sc_z = ax[0, _i].scatter(*_xyz_z[:, :2].T, s=_size, c=_SE_site_z, cmap=_cmap,
                vmin=-_vmax_z, vmax=_vmax_z)
            sc_a = ax[1, _i].scatter(*_xyz_a[:, :2].T, s=_size, c=_SE_site_a, cmap=_cmap,
                vmin=-_vmax_a, vmax=_vmax_a)

            ax[0, _i].set_title(f"E={_E:.1f} eV")
            ax[0, _i].annotate(_site_z, xy=_xyz_z[_site_z, :2], xytext=(15, 15),
                xycoords="data", textcoords="offset points", arrowprops=dict(arrowstyle="->"))
            ax[1, _i].annotate(_site_a, xy=_xyz_a[_site_a, :2], xytext=(15, 15),
                xycoords="data", textcoords="offset points", arrowprops=dict(arrowstyle="->"))

            for _sc, _ax in [(sc_z, ax[0, _i]), (sc_a, ax[1, _i])]:
                _cax = inset_axes(_ax, width="2%", height="50%", loc="lower left")
                _cb = fig.colorbar(_sc, cax=_cax)
                _cb.ax.set_title(r"Im($\Sigma_{ij}$)", pad=4)
                _fmt = mticker.ScalarFormatter(useMathText=True)
                _fmt.set_powerlimits((0, 0))
                _cb.formatter = _fmt
                _cb.update_ticks()

        for _a in ax.flatten():
            _a.axis("equal")
            _a.set(xticks=[], yticks=[])
            _a.axis("off")

    # Load data once outside loop
    with h5py.File(NOTEBOOK_DIR / f"{NOTEBOOK.stem}_data.h5", "r") as _f:
        _energies = np.asarray(_f["energies"])
        _SE_z = np.asarray(_f["zigzag"]).imag
        _SE_a = np.asarray(_f["armchair"]).imag

    _xyz_z = Ham_re_z.xyz
    _xyz_a = Ham_re_a.xyz
    _size = 10
    _cmap = "RdBu"

    # Define frame indices — change these as needed
    indices_z = [0, 1, 2, 3, 4, 5, 6,  7,  8,  9, 11]
    indices_a = [0, 1, 2, 3, 5, 7, 9, 10, 11, 12, 13]
    frames = list(zip(indices_z, indices_a))

    _fig, _ax = plt.subplots(2, len(_energies), figsize=(10, 5))

    def _animate(frame):
        _site_z, _site_a = frame
        make_frame(_fig, _ax, _site_z, _site_a)

    _ani = animation.FuncAnimation(_fig, _animate, frames=frames)
    _ani.save(NOTEBOOK_DIR / "rsse-edges.gif", writer="pillow", fps=1)
    plt.close(_fig)
    print(f"Saved {len(frames)} frames")
    return


@app.cell
def _(Ham_re_a, Ham_re_z, NOTEBOOK, NOTEBOOK_DIR, N_a, N_z, eta):
    # Load data
    with h5py.File(NOTEBOOK_DIR / f"{NOTEBOOK.stem}_data.h5", "r") as _f:
        _energies = np.asarray(_f["energies"])
        _SE_z = np.asarray(_f["zigzag"]).imag
        _SE_a = np.asarray(_f["armchair"]).imag

    _xyz_z = Ham_re_z.xyz
    _xyz_a = Ham_re_a.xyz
    _size = 10
    _cmap = "RdBu"

    def make_frame_single(fig, ax, xyz, SE, site):
        main_axes = set(ax.flatten())
        for _a in fig.axes[:]:
            if _a not in main_axes:
                _a.remove()
        for _a in ax.flatten():
            _a.cla()

        for _i, _E in enumerate(_energies):
            _SE_site = SE[_i, site, :]
            _vmax = np.max(np.abs(SE[_i, ...]))
            sc = ax[_i].scatter(*xyz[:, :2].T, s=_size, c=_SE_site, cmap=_cmap,
                vmin=-_vmax, vmax=_vmax)
            ax[_i].set_title(fr"$E={_E:.1f}$ eV,  $\eta = 10^{{{np.log10(eta):.0f}}}$ eV")
            ax[_i].annotate(site, xy=xyz[site, :2], xytext=(15, 15),
                xycoords="data", textcoords="offset points", arrowprops=dict(arrowstyle="->"))
            _cax = inset_axes(ax[_i], width="2%", height="30%", loc="lower left")
            _cb = fig.colorbar(sc, cax=_cax, label=r"Im($\Sigma_{ij}$)")
            # _cb.ax.set_title(r"Im($\Sigma_{ij}$)", pad=4)
            _fmt = mticker.ScalarFormatter(useMathText=True)
            _fmt.set_powerlimits((0, 0))
            _cb.formatter = _fmt
            _cb.update_ticks()
            _cb.ax.yaxis.get_offset_text().set_va("bottom")
            _cb.ax.yaxis.get_offset_text().set_ha("left")

        for _a in ax.flatten():
            _a.axis("equal")
            _a.set(xticks=[], yticks=[])
            # _a.axis("off")
    
        fig.tight_layout()

    for edge_type, xyz, SE, indices in [
        ("zigzag",   _xyz_z, _SE_z, range(N_z)),
        ("armchair", _xyz_a, _SE_a, range(N_a * 2 + 1)),
    ]:
        _fig, _ax = plt.subplots(1, len(_energies), figsize=(10, 3))
        _fig.suptitle(f"{edge_type.capitalize()}")
        def animate(frame, fig=_fig, ax=_ax, xyz=xyz, SE=SE):
            make_frame_single(fig, ax, xyz, SE, frame)

        ani = animation.FuncAnimation(_fig, animate, frames=list(indices))
        _gif_name = f"rsse-edges_{edge_type}.gif"
        ani.save(NOTEBOOK_DIR / _gif_name, writer="pillow", fps=len(indices)/10)
        plt.close(_fig)
        print(f"Saved {_gif_name} ({len(list(indices))} frames)")
    return


if __name__ == "__main__":
    app.run()
