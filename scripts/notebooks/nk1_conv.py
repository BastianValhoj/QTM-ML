import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import numpy as np

    import matplotlib.pyplot as plt
    import h5py
    from pathlib import Path

    from mytools.plots import thesis_fig, label_subplots


@app.cell
def _():
    DATA_DIR = Path(__file__).parent.parent / "conv_data"
    FIG_DIR = Path(__file__).parent.parent / "figures"
    (DATA_DIR / "calc_dos_vs_nk1.h5").exists()
    return (DATA_DIR,)


@app.cell
def _(DATA_DIR):
    DATA = h5py.File(DATA_DIR / "calc_dos_vs_nk1.h5", "r")
    print("data groups:", DATA.keys())
    print("attributes:",DATA.attrs.keys())
    print("attributes of armchair group:", DATA["armchair"].attrs.keys())
    print("keys of armchair dataset:", DATA["armchair"].keys())
    print("keys of zigzag dataset:", DATA["zigzag"].keys())
    # print("keys of armchair dataset:", DATA["armchair"]["N_3"].keys())
    return (DATA,)


@app.cell
def _(DATA):
    ENERGIES = DATA.attrs["E"]
    # E0_IDX = np.argwhere(np.isclose(ENERGIES, 0, atol=1e-12))[0,0]
    # print(E0_IDX)
    ETA = DATA.attrs["ETA"]
    NK1_ENUM = DATA.attrs["NK1_enum"]
    print(ENERGIES)
    print(ETA)
    print(NK1_ENUM)

    return ENERGIES, NK1_ENUM


@app.cell
def _(DATA, NK1_ENUM):
    _marker_list = ["^", "*", "s", "o", "p", 'h',]
    _marker_dict = {_enum:_marker_list[_i] for _i,_enum in enumerate(NK1_ENUM)}
    _fig, _axes = thesis_fig(subplots=(1,2), sharex="col", sharey=True)
    for _i, _kind in enumerate(DATA.keys()):
        # if _kind == "zigzag": break
        _axes[_i].set(title=_kind.capitalize())

        _current_edge = DATA[_kind]
        # _Nlist = _current_edge.attrs["N"]
        # print(_Nlist)

        for _enum in NK1_ENUM:
            _data = _current_edge[f"nk1_enum{_enum}"][:]
            _Ns, _DOS = _data.T
            print(_data)
            _axes[_i].semilogy(_Ns, _DOS, marker=_marker_dict[_enum], label=str(_enum))
        _axes[_i].legend()

        _axes[_i].set(ylabel=r"DOS(E=0) ($\mathrm{eV}^{-1}$)", xlabel="$N$")

    _fig.set_constrained_layout(True)
    _fig
    return


@app.cell
def _():
    return


@app.cell
def _():
    # _marker_list = ["^", "*", "s", "o", "p"]
    # _marker_dict = {_enum:_marker_list[_i] for _i,_enum in enumerate(NK1_ENUM)}
    # _fig, _axes = thesis_fig(subplots=(1,2), sharex="col", sharey=True)
    # for _i, _kind in enumerate(DATA.keys()):
    #     _axes[_i].set(title=_kind.capitalize())

    #     _current_edge = DATA[_kind]
    #     _Nlist = _current_edge.attrs["N"]
    #     _N_vs_nk1 = []
    #     for _N in _Nlist:
    #         _current_nk1 = _current_edge[f"N_{_N}"]
    #         _nk1_at_E0 = []
    #         for _j, _nk1 in enumerate(NK1_ENUM):
    #             _data = _current_nk1[f"nk1_enum_{_nk1}"]
    #             _nk1_at_E0.append(_data[E0_IDX])
    #         _N_vs_nk1
    
    #     _axes[_i].semilogy(_N, _data[E0_IDX], marker=_marker_dict[_nk1])
    #     _axes[_i].set(xlabel="N")

    #         # for _j, _nk1 in enumerate(NK1_ENUM):
    #             # _data = _current_nk1[f"nk1_enum_{_nk1}"]
    #             # _axes[_j, 0].semilogy(, _data[E0_IDX], label=fr"$\left\lceil \frac{{{_nk1}}}{{{_N}}}\right\rceil$", marker=_marker_dict[_nk1])
    #         # _axes[_j,_i].legend()
    # _fig.set_constrained_layout(True)
    # _fig
    return


@app.cell
def _(DATA, ENERGIES, ETAS, eta_formatter):
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



        _axes[1,_i].loglog(ETAS, _DOSE0_vs_ETA, marker="o", color="k")
    _handles, _labels = _axes[0,0].get_legend_handles_labels()
    _fig.legend(_handles, _labels, loc="upper center", bbox_to_anchor=(0.5, 0.9))

    _fig.set_constrained_layout_pads(wspace=0.1)
    _fig
    return


if __name__ == "__main__":
    app.run()
