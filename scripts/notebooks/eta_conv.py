import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")

with app.setup:
    import sisl
    import numpy as np
    from pathlib import Path
    from matplotlib.ticker import ScalarFormatter
    from mytools.plots import thesis_fig, label_subplots

    import h5py
    import marimo as mo


@app.cell
def _():
    DATA_DIR = Path(__file__).parent.parent / "conv_data"
    FIG_DIR = Path(__file__).parent.parent / "figures"
    return (DATA_DIR,)


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
    _fig, _axes = thesis_fig(subplots=(2,2), sharex="row", sharey=True)
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

    

        _axes[1,_i].loglog(ETAS, _DOSE0_vs_ETA, marker="o")
    _handles, _labels = _axes[0,0].get_legend_handles_labels()
    _fig.legend(_handles, _labels, loc="upper center", bbox_to_anchor=(0.5, 0.9))

    _fig.set_constrained_layout_pads(wspace=0.1)
    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
