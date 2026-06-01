import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")

with app.setup:
    import sisl
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from ase.lattice import HEX2D
    from ase.lattice import HEX

    import marimo as mo


@app.cell
def _():
    divisions_choice = mo.ui.slider(start=30, stop=400, step=10, value=50, label="Divisions", debounce=True, include_input=True)
    stack_choice = mo.ui.dropdown(["AA", "AB"], value="AA", label="Stacking")
    tile_choice = mo.ui.dropdown(["1x1", "2x2", "3x3", "4x4", None], value="2x2", label="Tile")
    mo.hstack([stack_choice, tile_choice, divisions_choice], justify="start")
    return divisions_choice, stack_choice, tile_choice


@app.cell
def _(divisions_choice, stack_choice):
    STACK = stack_choice.value
    divisions = divisions_choice.value
    return STACK, divisions


@app.cell
def _(STACK, tile_choice):
    WORK_DIR = Path.home() / "w3"
    BILA_DIR = WORK_DIR / "bilayer_data"
    DFT_DIR = BILA_DIR / "DFT"
    if tile_choice.value is None:
        STACK_DIR = DFT_DIR / f"{STACK}_stack"
    else:
        tile_val = tile_choice.value
        STACK_DIR = DFT_DIR / f"{STACK}_stack_{tile_val}"
    STACK_DIR.is_dir()
    return (STACK_DIR,)


@app.cell
def _(STACK_DIR):
    sile = sisl.get_sile(STACK_DIR / f"RUN.fdf")
    # Ham = sile.read_hamiltonian()
    ham = sile.read_hamiltonian()
    geom = ham.geometry
    print(geom.nsc)
    print(ham.nsc)
    return geom, ham, sile


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Show geometry
    """)
    return


@app.cell
def _(geom):
    print("d=", geom.xyz[:, 2].max() - geom.xyz[:, 2].min())
    idx_top = np.where(geom.xyz[:, 2] == geom.xyz[:, 2].max())[0]
    idx_bot = np.where(geom.xyz[:, 2] == geom.xyz[:, 2].min())[0]
    print("Top layer atoms:", idx_top)
    print("Bottom layer atoms:", idx_bot)   
    geom.plot(axes=["x", [0,1,1/20]], backend="matplotlib", 
        atoms_style=[
            dict(atoms=idx_top, color="red",  size=10), 
            dict(atoms=idx_bot, color="blue", size=15),
        ]
    )
    return (idx_top,)


@app.cell
def _(ham):
    bz = sisl.BrillouinZone(ham)
    for _att in dir(bz.apply):
        if not _att.startswith("_"):
            print(_att)
    return


@app.cell
def _(geom):
    for _att in dir(geom.lattice):
        if not _att.startswith("_"):
            print(_att)
    return


@app.cell
def _(sile):
    for _att in dir(sile):
        if not _att.startswith("_") and "read" in _att:
            print(_att)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Show DOS
    """)
    return


@app.cell
def _(STACK_DIR, sile):
    # (STACK_DIR / "graphene_bilayer.DOS")
    _fig, _ax = plt.subplots()
    _energies, DOS = np.loadtxt(STACK_DIR / "bilayer_calc.DOS").T
    _ax.plot(_energies - sile.read_fermi_level(), DOS)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Make bandpath
    """)
    return


@app.cell
def _(divisions):
    hex = HEX2D(a=1.42)
    # hex = HEX(a=1.42, c=3.35)

    hexpath = hex.bandpath(path="MGKM", npoints=divisions, special_points={"G": [0, 0, 0], "M": [0.5, 0, 0], "K": [ 1 / 3, - 2 / 3, 0]})
    for _att in dir(hexpath):
        if not _att.startswith("_"):
            print(_att)

    hexpath.plot()
    return (hexpath,)


@app.cell
def _(STACK_DIR, divisions_choice, stack_choice, tile_choice):
    # SYMEMTRY_POINTS, LABELS, lineark,  _ = sisl.get_sile(STACK_DIR / "graphene_bilayer.bands").read_data()
    _fig = sisl.get_sile(STACK_DIR / "bilayer_calc.bands").plot(backend="matplotlib")

    mo.hstack([_fig, mo.vstack([stack_choice, tile_choice, divisions_choice])], justify="start")
    return


@app.cell
def _():
    mo.md(r"""
    # Plot bands from sisl Hamiltonian + PDOS
    """)
    return


@app.cell
def _(STACK_DIR, divisions, ham, hexpath):
    bands = sisl.BandStructure(ham, points=list(hexpath.special_points.values()), divisions=divisions)
    # lk, lklab, labels= bands.lineark(True)
    eigs = bands.apply.array.eigh()

    _PDOS = sisl.io.pdosSileSiesta(STACK_DIR / "bilayer_calc.PDOS")
    SIESTA_PDOS_DATA = _PDOS.read_data()
    SIESTA_PDOS_DATA[2]
    return SIESTA_PDOS_DATA, eigs


@app.cell
def _(ham):
    es = ham.eigenstate()
    return (es,)


@app.cell
def _(SIESTA_PDOS_DATA, idx_top):
    def get_pdos(geom, data):
        nE = data.shape[2]
        pdos_data = data.squeeze()
        no = geom.no
        na = geom.na
        # for iorb in range(geom.atoms[0].no):
        #     print(iorb, geom.atoms[0].orbitals[iorb].name())
        pdos = {
            name:np.zeros(nE) for name in ["top s", "top px", "top py", "top pz", "bottom s", "bottom px", "bottom py", "bottom pz"]
            }
        for atom_idx in range(na):
            which = "top" if atom_idx in idx_top else "bottom"
            first_orb = geom.a2o(atom_idx)
            for local_iorb in range(geom.atoms[atom_idx].no):
                iorb = first_orb + local_iorb
                orb_name = geom.atoms[atom_idx].orbitals[local_iorb].name()
                if "s" in orb_name:
                    pdos[f"{which} s"] += pdos_data[iorb, :]
                elif "px" in orb_name:
                    pdos[f"{which} px"] += pdos_data[iorb, :]
                elif "py" in orb_name:
                    pdos[f"{which} py"] += pdos_data[iorb, :]
                elif "pz" in orb_name:
                    pdos[f"{which} pz"] += pdos_data[iorb, :]


        return pdos

    get_pdos(SIESTA_PDOS_DATA[0], SIESTA_PDOS_DATA[2])
    return (get_pdos,)


@app.cell
def _(hexpath):
    lk, kl, labels = hexpath.get_linear_kpoint_axis()
    return kl, labels, lk


@app.cell
def _(es):
    # _orb = "s"
    # plt.plot(energies , get_pdos(geom, es.PDOS(energies))[f"top {_orb}"])
    # plt.plot(energies , get_pdos(geom, es.PDOS(energies))[f"bottom {_orb}"])
    _energies = np.linspace(-10, 10, 1000)
    plt.plot(_energies , es.PDOS(_energies).squeeze().T)
    return


@app.cell
def _(
    SIESTA_PDOS_DATA,
    STACK,
    divisions_choice,
    eigs,
    geom,
    get_pdos,
    kl,
    labels,
    lk,
    stack_choice,
    tile_choice,
):
    _emax = 4
    _emin = -_emax

    _fig, _axes = plt.subplots(1, 2)
    _axes[0].plot(lk, eigs, color="k")
    _ylim = (_emin, _emax)
    _axes[0].set(ylim=_ylim, xlim=(min(kl), max(kl)), xticks=kl, xticklabels=labels, ylabel="Energy (eV)", xlabel=r"$\boldsymbol{k}$-path")
    pdos = get_pdos(geom, SIESTA_PDOS_DATA[2])
    energies = SIESTA_PDOS_DATA[1]
    # energies = np.linspace(_emin, _emax, 1000)
    # pdos = get_pdos(geom, es.PDOS(energies) - sile.read_fermi_level())
    # _orbitals = ["s", "px", "py", "pz"]
    _orbitals = ["pz"]
    for _orb in _orbitals:
        _axes[1].plot(pdos[f"top {_orb}"], energies, label=f"Top {_orb}", marker="o", markersize=2)
        _axes[1].plot(pdos[f"bottom {_orb}"], energies, label=f"Bottom {_orb}", marker="D", markersize=2)
    _axes[1].legend()
    _axes[1].set(ylim=_ylim, xlabel="PDOS", ylabel="Energy (eV)")

    for _ax in _axes:
        _emin1 = np.argwhere(pdos["top pz"] == np.min(pdos["top pz"]))[0][0]
        _emin2 = np.argwhere(pdos["bottom pz"] == np.min(pdos["bottom pz"]))[0][0]
        _ax.axhline(energies[_emin1], color="grey", ls="--", lw=1)
        _ax.axhline(energies[_emin2], color="grey", ls="--", lw=1)
        # _ax.axhline(0.5, color="grey", ls="--", lw=1)
    _fig.suptitle(f"{STACK} stacking")
    _fig.set_constrained_layout(True)
    mo.hstack([_fig, mo.vstack([stack_choice, tile_choice, divisions_choice])], justify="start", align="start")
    return


if __name__ == "__main__":
    app.run()
