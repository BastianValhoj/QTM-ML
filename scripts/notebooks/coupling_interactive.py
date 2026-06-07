import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")

with app.setup:
    import sisl
    from mytools.construct import all_armchair
    from pathlib import Path
    import marimo as mo
    import h5py
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import numpy as np

    from mytools.plots import thesis_fig, label_subplots
    from matplotlib.ticker import ScalarFormatter


@app.cell
def _():
    SCRIPT_PATH = Path(__file__)
    print(f"Script path: {SCRIPT_PATH}")
    FIG_DIR = SCRIPT_PATH.parent.parent / "figures"
    print(f"Figure directory: {FIG_DIR}")

    return FIG_DIR, SCRIPT_PATH


@app.cell
def _():
    # Edge type choice
    kind_choice = mo.ui.dropdown(["armchair", "zigzag"], value="armchair", label="Edges")
    return (kind_choice,)


@app.cell
def _(SCRIPT_PATH, kind_choice):
    # Choose dataset based on edge type
    KIND = kind_choice.value
    DATA_DIR = SCRIPT_PATH.parent.parent / "conv_data"

    DATA = h5py.File(DATA_DIR / f"RSE_data-{KIND}.h5", 'r')
    print("Groups in DATA: {}".format(DATA.keys()))
    print("Attributes: {}".format(DATA.attrs.keys()))
    print("keys of first group: {}".format(DATA[list(DATA.keys())[0]].keys()))
    return DATA, DATA_DIR, KIND


@app.cell
def _(DATA):
    # Choice of tiling N based on available groups in the dataset
    nlist =  sorted([int(key.split('_')[1]) for key in list(DATA.keys())])
    tile_choice = mo.ui.slider(steps=nlist, debounce=True, show_value=True, label="Tiling, N")
    return (tile_choice,)


@app.cell
def _(DATA):
    # Choice of energy E based on available energy values in the dataset
    energies = DATA.attrs["E"].astype(float)
    # print("Available energy values: {}".format(energies))
    energy_choice = mo.ui.slider(steps=energies, debounce=True, include_input=True, value=0.0, label="Energy, E (eV)")
    return (energy_choice,)


@app.cell
def _(DATA):
    # Choice of eta based on available eta values in the dataset
    etalist = DATA.attrs["ETA"].astype(float).tolist()
    # print("Available eta values: {}".format(etalist))
    eta_choice = mo.ui.dropdown(etalist, value=etalist[1], label="$\eta$")
    return (eta_choice,)


@app.cell
def _(energy_choice, eta_choice, tile_choice):
    # global parameters chosen by the user
    N = tile_choice.value
    E = energy_choice.value
    ETA = eta_choice.value
    print("N : {}, E : {}, eta : {}".format(N, E, ETA))
    return E, ETA, N


@app.cell
def _(DATA, N):
    # structure specific data
    current_group = DATA[f"N_{N}"]
    xyz = current_group["xyz"][:]
    N_elec = len(current_group["elec_idx"][:])
    atoms_idx = current_group["atoms_idx"][:]
    elec_idx = atoms_idx[:N_elec]
    print("Number of atoms: {}, Number of electrons: {}".format(len(atoms_idx), N_elec))
    return N_elec, current_group, xyz


@app.cell
def _(N_elec):
    site_choice = mo.ui.slider(step=1, start=0, stop=N_elec-1, debounce=True, include_input=True, label="Site index")
    return (site_choice,)


@app.cell
def _(energy_choice, eta_choice, kind_choice, site_choice, tile_choice):
    # collective parameter widget
    params = mo.vstack([kind_choice, tile_choice, site_choice, mo.hstack([energy_choice, mo.md("+"), eta_choice], justify="start")], justify="start", align="stretch")
    params
    return (params,)


@app.cell
def _(ETA, current_group):
    # which RSSE dataset based on eta
    RSSE = current_group[f"eta_{ETA:.1e}"][:]
    return (RSSE,)


@app.cell
def _(DATA, E):
    # Which energy point to plot
    energy_idx = np.argmin(np.abs(DATA.attrs["E"].astype(float) - E))
    print("Energy index: {}, Energy value: {}".format(energy_idx, DATA.attrs["E"].astype(float)[energy_idx]))
    return (energy_idx,)


@app.cell
def _(site_choice):
    site = site_choice.value
    print("Chosen site index: {}".format(site))
    return (site,)


@app.cell
def _(KIND, N):
    number_atoms_first_edge = N if KIND == "zigzag" else 4+3+2*(N-2)
    print("Number of atoms in the first edge: {}".format(number_atoms_first_edge))
    return (number_atoms_first_edge,)


@app.cell
def _(N_elec, RSSE, energy_idx, number_atoms_first_edge, params, site, xyz):
    _fig, _axes = thesis_fig(subplots=(2,2), fraction=1)
    _axes = _axes.flatten()

    coupling = RSSE[energy_idx, site, :].imag
    onsite = np.diagonal(RSSE[energy_idx, ...]).imag
    _alpha = 0.6

    # Plot Coupling vs. distance to site
    distances = np.linalg.norm(xyz - xyz[site], axis=1)
    _axes[0].scatter(distances[:number_atoms_first_edge], coupling[:number_atoms_first_edge], marker='x', s=20, label="edges", color="red", alpha=_alpha)
    _axes[0].scatter(distances[number_atoms_first_edge:N_elec], coupling[number_atoms_first_edge:N_elec], marker='x', s=20, label="edges", color="blue", alpha=_alpha)
    _axes[0].scatter(distances[N_elec:], coupling[N_elec:], marker='.', s=10, label="bulk", color="k", alpha=_alpha)
    # _axes[0].legend()
    _axes[0].set_xlabel("Distance to site (Angstrom)")
    _axes[0].set_ylabel("Coupling (eV)")


    # Plot scatter of atoms colored by coupling
    ## set colorbar to be symmetric around zero
    _vmax = np.max(np.abs(coupling))
    _vmin = -_vmax
    sc = _axes[1].scatter(xyz[:,0], xyz[:,1], c=coupling, cmap="RdBu", marker=".", s=20, vmin=_vmin, vmax=_vmax)
    cax = make_axes_locatable(_axes[1]).append_axes("right", size="5%", pad=0.5)
    _fig.colorbar(sc, cax=cax, label="Coupling (eV)")
    ## annotate the site
    _axes[1].annotate(site, xy=xyz[site][:2], xycoords="data", xytext=(7,3), textcoords="offset points", arrowprops=dict(arrowstyle="->"))
    _handles, _labels = _axes[0].get_legend_handles_labels()

    _axes[-2].remove()

    _fig.legend(_handles, _labels, loc="upper left", bbox_to_anchor=(0.0, 0.5))



    # Plot onsites in scatter plot
    _vmax = np.max(np.abs(onsite))
    _vmin = -_vmax
    sc = _axes[-1].scatter(xyz[:,0], xyz[:,1], c=onsite, cmap="RdBu", marker=".", s=20, vmin=_vmin, vmax=_vmax)
    cax = make_axes_locatable(_axes[-1]).append_axes("right", size="5%", pad=0.05)
    _fig.colorbar(sc, cax=cax, label="Onsite (eV)")
    _fig.set_constrained_layout(True)
    mo.hstack([_fig, params], justify="start", align="start")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Plot armchair and zig-zag together
    """)
    return


@app.cell
def _(DATA_DIR):
    DATA_arm = h5py.File(DATA_DIR / f"RSE_data-armchair.h5", 'r')
    DATA_zig = h5py.File(DATA_DIR / f"RSE_data-zigzag.h5", 'r')
    return DATA_arm, DATA_zig


@app.cell
def _(DATA_arm, DATA_zig):
    nlist_arm = sorted([int(key.split('_')[1]) for key in list(DATA_arm.keys())])
    nlist_zig = sorted([int(key.split('_')[1]) for key in list(DATA_zig.keys())])
    return nlist_arm, nlist_zig


@app.cell
def _(nlist_arm, nlist_zig):
    tile_choice_arm = mo.ui.slider(steps=nlist_arm, value=nlist_arm[-2], debounce=True, show_value=True, label="Tiling armchair, N")
    tile_choice_zig = mo.ui.slider(steps=nlist_zig, value=nlist_zig[-2], debounce=True, show_value=True, label="Tiling zigzag, N")
    return tile_choice_arm, tile_choice_zig


@app.cell
def _(DATA_arm, DATA_zig, tile_choice_arm, tile_choice_zig):
    current_group_arm = DATA_arm[f"N_{tile_choice_arm.value}"]
    current_group_zig = DATA_zig[f"N_{tile_choice_zig.value}"]
    return current_group_arm, current_group_zig


@app.cell
def _(current_group_arm, current_group_zig, tile_choice_arm, tile_choice_zig):
    N_elec_arm = len(current_group_arm["elec_idx"][:])
    N_elec_zig = len(current_group_zig["elec_idx"][:])
    elec_idx_arm = current_group_arm["atoms_idx"][:N_elec_arm]
    elec_idx_zig = current_group_zig["atoms_idx"][:N_elec_zig]
    N_first_edge_arm = 4+3+2*(tile_choice_arm.value-2)
    N_first_edge_zig = tile_choice_zig.value
    return N_elec_arm, N_elec_zig, N_first_edge_arm, N_first_edge_zig


@app.cell
def _(N_first_edge_arm, N_first_edge_zig):
    site_choice_arm = mo.ui.slider(step=1, start=0, stop=N_first_edge_arm, value=N_first_edge_arm//2, debounce=True, include_input=True, label="Site index armchair")
    site_choice_zig = mo.ui.slider(step=1, start=0, stop=N_first_edge_zig, value=N_first_edge_zig//2, debounce=True, include_input=True, label="Site index zigzag")
    return site_choice_arm, site_choice_zig


@app.cell
def _(site_choice_arm, site_choice_zig, tile_choice_arm, tile_choice_zig):
    zig_params = mo.vstack([mo.md("##Zigzag params"), tile_choice_zig, site_choice_zig], justify="start")
    arm_params = mo.vstack([mo.md("##Armchair params"), tile_choice_arm, site_choice_arm], justify="start")
    return arm_params, zig_params


@app.cell
def _(energy_choice, eta_choice):
    energy_params = mo.hstack([energy_choice, mo.md("+"), eta_choice], justify="start", align="start")
    return (energy_params,)


@app.cell
def _():
    cbarformatter = ScalarFormatter(useMathText=True)
    cbarformatter.set_scientific(True)
    cbarformatter.set_powerlimits((0,0))
    return (cbarformatter,)


@app.function
def eta_formatter(ETA):
    exponent = int(np.log10(ETA))
    mantissa = ETA / 10**exponent
    if mantissa == 1.0:
        return f"10^{{{exponent}}}"
    else:
        return f"{mantissa:.1f}\\times 10^{{{exponent}}}"


@app.cell
def _(
    E,
    ETA,
    FIG_DIR,
    N_elec_arm,
    N_elec_zig,
    N_first_edge_arm,
    N_first_edge_zig,
    SCRIPT_PATH,
    arm_params,
    cbarformatter,
    current_group_arm,
    current_group_zig,
    energy_idx,
    energy_params,
    site_choice_arm,
    site_choice_zig,
    zig_params,
):
    _fig, _axes = thesis_fig(subplots=(3,2))
    _alpha = 0.6
    for _i, (_kind, _group, _N_elec, _N_first_edge) in enumerate(zip(["armchair", "zigzag"], [current_group_arm, current_group_zig], [N_elec_arm, N_elec_zig], [N_first_edge_arm, N_first_edge_zig])):
        _rsse = _group[f"eta_{ETA:.1e}"][:]
        _site = site_choice_arm.value if _kind == "armchair" else site_choice_zig.value
        _coupling = _rsse[energy_idx, _site, :].imag
        _onsite = np.diagonal(_rsse[energy_idx, ...]).imag    
        _distances = np.linalg.norm(_group["xyz"][:] - _group["xyz"][_site], axis=1)
        _first_edge_not_site = np.arange(_N_first_edge)[np.arange(_N_first_edge) != _site]


        _cbar_side = "right" if _i == 0 else "right"
        _axes[0,_i].scatter(_distances[_site], _coupling[_site], marker='^', s=50, label=f"site", color="red", alpha=_alpha)
        _axes[0, _i].scatter(_distances[_first_edge_not_site], _coupling[_first_edge_not_site], marker='x', s=20, label="1st edge", color="red", alpha=_alpha)
        _axes[0, _i].scatter(_distances[_N_first_edge:_N_elec], _coupling[_N_first_edge:_N_elec], marker='x', s=20, label="other edges", color="blue", alpha=_alpha)
        _axes[0, _i].scatter(_distances[_N_elec:], _coupling[_N_elec:], marker='.', s=10, label="bulk", color="k", alpha=_alpha)
        _axes[0, _i].set(xlabel="Distance to site ($\mathrm{\\AA}$)", ylabel="Coupling (eV)", title=f"{_kind.capitalize()}")


        _vmax = np.max(np.abs(_coupling))
        _vmin = -_vmax
        _sc = _axes[1, _i].scatter(_group["xyz"][:,0], _group["xyz"][:,1], c=_coupling, cmap="RdBu", marker=".", s=20, vmin=_vmin, vmax=_vmax)
        _cax = make_axes_locatable(_axes[1, _i]).append_axes(_cbar_side, size="5%", pad=0.05)
        cbar = _fig.colorbar(_sc, cax=_cax, label="Coupling (eV)")
        cbar.formatter = cbarformatter
        cbar.update_ticks()
        _axes[1, _i].annotate(_site, xy=_group["xyz"][_site][:2], xycoords="data", xytext=(15,15), textcoords="offset points", arrowprops=dict(arrowstyle="->"))


        _vmax = np.max(np.abs(_onsite))
        _vmin = -_vmax
        _axes[2, _i].scatter(_group["xyz"][:,0], _group["xyz"][:,1], c=_onsite, cmap="RdBu", marker=".", s=20, vmin=_vmin, vmax=_vmax)
        _cax = make_axes_locatable(_axes[2, _i]).append_axes(_cbar_side, size="5%", pad=0.05)
        cbar = _fig.colorbar(_sc, cax=_cax, label="Onsite (eV)")
        cbar.formatter = cbarformatter
        cbar.update_ticks()

    for _i, _ax in enumerate(_axes[1:].flatten()):
        _ax.set_aspect("equal")
        _ax.set(xlabel="x ($\mathrm{\\AA}$)", ylabel="y ($\mathrm{\\AA}$)")
        # _ax.set_yticks([])

    _handles, _labels = _axes[0,0].get_legend_handles_labels()
    _fig.legend(_handles, _labels, loc="upper left", bbox_to_anchor=(0.0, 1.09), ncol=2)
    # _fig.tight_layout()
    _fig.set_constrained_layout(True)
    _fig.set_constrained_layout_pads(wspace=0.2,)
    label_subplots(_axes.flatten(),)
    _eta_power = int(np.log10(ETA))

    _fig.suptitle(f"$z = {E:.1f} + i{eta_formatter(ETA)}$", y=1.05, x=0.55)
    _fig.savefig(FIG_DIR / f"{SCRIPT_PATH.stem}_E{E:.2f}_eta{ETA:.1e}.pdf", bbox_inches="tight")
    mo.hstack([_fig, mo.vstack([energy_params,zig_params, arm_params], justify="start", align="center")], justify="start")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
