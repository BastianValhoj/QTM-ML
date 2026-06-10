import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")

with app.setup:
    import sisl
    import numpy as np
    from pathlib import Path
    from matplotlib.ticker import ScalarFormatter
    from mytools.plots import thesis_fig, label_subplots
    from mytools.construct import all_armchair

    import h5py
    import marimo as mo

    from tqdm.auto import tqdm


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Show geometry
    """)
    return


@app.cell
def _():
    N = 9
    return (N,)


@app.cell
def _(N):
    base = all_armchair()
    geom = base.tile(N,1).tile(N,0)
    return (geom,)


@app.cell
def _(geom):
    geom.plot(axes="xy", backend="matplotlib")
    _fig, ax_geom = thesis_fig(1,1)
    ax_geom.scatter(*geom.xyz[:, :2].T, )
    # draw vector along edge and state number of tiles
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Read conv data
    """)
    return


@app.cell
def _():
    DATA_DIR = Path(__file__).parent.parent / "conv_data"
    FIG_DIR = Path(__file__).parent.parent / "figures"
    (DATA_DIR / "calc_dos_vs_eta.h5").exists()
    return DATA_DIR, FIG_DIR


@app.cell
def _(DATA_DIR):
    DATA = h5py.File(DATA_DIR / "calc_dos_vs_eta.h5", "r")
    print("data keys:", DATA.keys())
    print("data attributes:",DATA.attrs.keys())
    print("keys of first dataset:", DATA[list(DATA.keys())[0]].keys())
    return (DATA,)


@app.cell
def _(DATA):
    ENERGIES = DATA.attrs["E"][:]
    ETAS = DATA.attrs["ETA"][:]
    return ENERGIES, ETAS


@app.cell
def _(ENERGIES):
    print(ENERGIES.shape, len(ENERGIES))
    print()
    return


@app.function
def eta_formatter(ETA):
    exponent = int(np.log10(ETA))
    mantissa = ETA / 10**exponent
    if mantissa == 1.0:
        return f"10^{{{exponent}}}"
    else:
        return f"{mantissa:.1f}\\times 10^{{[{exponent}]}}"


@app.cell
def _(DATA, ENERGIES, ETAS):
    _fig, _axes = thesis_fig(2,2, sharex="row", sharey="col")
    _marker_list = ["o", "^", "s", "p", "d"]
    _marker_dict = {f"{_eta:.1e}":_marker_list[_i] for _i, _eta in enumerate(ETAS)}
    for _i, _kind in enumerate(DATA.keys()):
        # print(kind)
        _current_data = DATA[_kind]
        _DOSE0_vs_ETA = []
        for _eta in ETAS:
            # print(ENERGIES.shape, _current_data[f"eta_{eta:.1e}"][:].shape)
            _DOS = _current_data[f"eta_{_eta:.1e}"][:]
            _axes[0, _i].semilogy(ENERGIES, _DOS, label=f"$\\eta = {eta_formatter(_eta)}$", linestyle="-", marker=_marker_dict[f"{_eta:.1e}"])
            _E0_idx = np.argwhere(np.isclose(ENERGIES, 0, atol=1e-12))[0,0]
            _DOSE0_vs_ETA.append(_DOS[_E0_idx])

        _axes[0,_i].set(title=_kind.capitalize())
        _axes[1,_i].loglog(ETAS, _DOSE0_vs_ETA, marker="o", color="k")

    _axes[0,0].set(
        ylabel=r"DOS ($\mathrm{eV}^{-1}$)",
        xlabel=r"$E$ (eV)"
        )
    _axes[0,1].set(xlabel=r"$E$ (eV)")
    _axes[1,0].set(
        ylabel=r"DOS(E=0) ($\mathrm{eV}^{-1}$)",
        xlabel=r"$\eta$ (eV)"
        )
    _axes[1,1].set(xlabel=r"$\eta$ (eV)")
    _handles, _labels = _axes[0,0].get_legend_handles_labels()
    _fig.legend(_handles, _labels, loc="upper center", bbox_to_anchor=(0.5, 0.9))

    for _ax in _axes.flatten():
        _ax.grid()
    _fig.set_constrained_layout_pads(wspace=0.1)
    _fig.set_constrained_layout(True)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Armchair only convergence
    """)
    return


@app.cell
def _(DATA, ETAS):
    DATA.attrs["ETA"]
    print(ETAS)
    return


@app.cell
def _(DATA, ENERGIES, ETAS, FIG_DIR):
    _E0_idx = np.argwhere(np.isclose(ENERGIES, 0, atol=1e-12))[0,0]
    _marker_list = ["o", "^", "s", "p", "d"]
    _marker_dict = {f"{_eta:.1e}":_marker_list[_i] for _i, _eta in enumerate(ETAS)}
    fig, axes = thesis_fig(1, 2, aspect=0.4, sharey=True)
    data_arm = DATA["armchair"]
    DOS_vs_eta = []
    for _eta in ETAS[:]:
        DOS_vs_eta.append(data_arm[f"eta_{_eta:.1e}"][_E0_idx])
        axes[0].semilogy(ENERGIES, data_arm[f"eta_{_eta:.1e}"][:], label=f"$\\eta = {eta_formatter(_eta)}$", linestyle="-", marker=_marker_dict[f"{_eta:.1e}"])
    axes[1].loglog(ETAS[:], DOS_vs_eta)

    axes[0].legend(loc=(0.56, 0.01))

    for _ax in axes:
        _ax.grid()


    axes[0].set(
        xlabel=r"$E$ (eV)",
        ylabel=r"DOS $(\mathrm{eV}^{-1}$)",
        title="DOS (normalized)"
    )
    axes[1].set(
        xlabel=r"$\eta$ (eV)",
        title="$\mathrm{DOS}(E=0)$ (normalized)"
    )
    label_subplots(axes, pos=(0.06, 1.13))
    fig.set_constrained_layout(True)
    fig.savefig(FIG_DIR / "eta_conv_armchair")
    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
