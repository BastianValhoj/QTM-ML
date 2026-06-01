import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")

with app.setup:
    import sisl
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm.auto import tqdm
    from pathlib import Path

    import marimo as mo


@app.cell
def _():
    nc_choice = mo.ui.dropdown([0, 2], value=0, label="NC")
    return (nc_choice,)


@app.cell
def _():
    stack_choice = mo.ui.dropdown(["AA", "AB"], value="AA", label="Stacking", searchable=True)
    return (stack_choice,)


@app.cell
def _():
    angle_choice = mo.ui.dropdown([0, 20, 40], value=0, label="Angle (${}^\circ$deg)", searchable=True)
    return (angle_choice,)


@app.cell
def _():
    shift_choice = mo.ui.dropdown([0, 10, 20], value=0, label="Shift ($\mathrm{\AA}$)", searchable=True)
    return (shift_choice,)


@app.cell
def _(nc_choice):
    NC = nc_choice.value
    _nc_start = 1 + 2*(1+NC)
    n_choice = mo.ui.slider(start=_nc_start, stop=12,step=1,value=7, label="N", show_value=False, debounce=True, include_input=True)
    return NC, n_choice


@app.cell
def _(angle_choice, n_choice, nc_choice, shift_choice, stack_choice):
    params = mo.vstack([nc_choice, stack_choice, n_choice, angle_choice, shift_choice], align="start", gap=1)
    params
    return (params,)


@app.cell
def _(angle_choice, n_choice, shift_choice, stack_choice):
    N = n_choice.value
    NT = 30
    STACK = stack_choice.value
    ANGLE = angle_choice.value
    SHIFT = shift_choice.value
    return ANGLE, N, NT, SHIFT, STACK


@app.cell
def _(ANGLE, N, NC, NT, SHIFT, STACK):
    SCRIPT_DIR = Path(__file__).parent
    WORK_DIR = Path.home() / "w3"
    BI_DATA = WORK_DIR / "bilayer_data"
    STACK_DIR = BI_DATA / "ham" / f"{STACK}_stack"
    TILE_DIR = STACK_DIR / f"NC{NC}-N{N}_to_{NT}"
    ANGLE_DIR = TILE_DIR / f"angle-{ANGLE}"
    SHIFT_DIR = ANGLE_DIR / f"shift-{SHIFT}"
    print(SHIFT_DIR.relative_to(STACK_DIR))
    print(SHIFT_DIR.exists())
    return ANGLE_DIR, SHIFT_DIR, TILE_DIR


@app.cell
def _(TILE_DIR):
    PARAMS = np.load(TILE_DIR / "params.npz")
    elec_top = PARAMS["elec_top"]
    elec_bot = PARAMS["elec_bottom"]
    dev_top = PARAMS["device_top"]
    dev_bot = PARAMS["device_bottom"]
    idx_top = list(elec_top) + list(dev_top)
    idx_bot = list(elec_bot) + list(dev_bot)
    return idx_bot, idx_top


@app.cell
def _(SHIFT_DIR):
    trans_sile = sisl.get_sile(SHIFT_DIR / "trans.TBT.nc")
    for _att in dir(trans_sile):
        if "trans" in _att:
            print(_att)
    return (trans_sile,)


@app.cell
def _(trans_sile):
    trans_sile.transmission_bulk
    return


@app.cell
def _(ANGLE, SHIFT, STACK, idx_bot, idx_top, params, trans_sile):
    # circle = lambda R: R*np.vstack([np.cos(np.linspace(0,2*np.pi,300)), np.sin(np.linspace(0, 2*np.pi,300))])

    _fig, _axes = plt.subplots(1,3, figsize=(10,4))
    markers={"top": "d", "bottom": "^"}
    _adev = trans_sile.a_dev
    _nadev = trans_sile.na_dev
    _E = trans_sile.E
    _alpha = 0.5
    _inax = _axes[0].inset_axes(bounds=(0.3, 0.7, 0.4, 0.2), xlim=(-1,1), ylim=(0, 0.01))
    for _elec in ["top", "bottom"]:
        _ADOS = trans_sile.ADOS(elec=_elec, atoms=_adev) / _nadev
        _axes[0].plot(_E, _ADOS, label=_elec, linestyle="-", marker=markers[_elec], markersize=4, alpha=_alpha)
        _inax.plot(_E, _ADOS, linestyle="-", marker=markers[_elec], markersize=4, alpha=_alpha+0.3)


    _axes[0].indicate_inset_zoom(_inax, edgecolor="black", alpha=1)

    _trans = trans_sile.transmission(elec_from=0, elec_to=1)
    _axes[1].plot(_E, _trans, label=r"top $\to$ bottom")

    for _ax in _axes[:2]:
        _ax.grid()
        _ax.axvline(-0.5, color="k", linestyle="--", alpha=0.6, label="Applied bias ($\pm 0.5$)")
        _ax.axvline(0.5, color="k", linestyle="--", alpha=0.6)
    _inax.grid()
    _inax.axvline(-0.5, color="k", linestyle="--", alpha=0.6)
    _inax.axvline(0.5, color="k", linestyle="--", alpha=0.6)


    _axes[0].set(
        xlabel="E (eV)",
        ylabel="DOS (1/eV)",
        title="DOS (normalized)",
    )
    _axes[1].set(
        xlabel="E (eV)",
        ylabel="Transmission",
        title=r"Transmission, (top $\to$ bottom)"
    )

    _geom = trans_sile.read_geometry()
    _atominset = _axes[2].inset_axes(bounds=(0.5, 0.1, 0.5, 0.4), xlim=(100,115), ylim=(20,35))

    for _ax in [_axes[2], _atominset]:
        if _ax == _axes[2]:
            _size = 1, 1
        else:
            _size = 8, 12

        _ax.scatter(*_geom.xyz[idx_top, :2].T, marker='.', color="red", s=_size[0], label="top atoms", zorder=2)
        _ax.scatter(*_geom.xyz[idx_bot, :2].T, marker='x', color="black", s=_size[1], label="bottom atoms", zorder=1)

    _axes[2].indicate_inset_zoom(_atominset, edgecolor="black", alpha=1)
    _xstart = np.min(_geom.xyz[:,0])
    _axes[2].plot([_xstart,_xstart+SHIFT], [0,0], color="k", linewidth=3, zorder=3)
    if SHIFT != 0:
        _axes[2].annotate(xy=(_xstart+SHIFT, 0), text=f"{SHIFT} Å", ha="center", va="center", xytext=(_xstart+SHIFT/2, 20), 
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.75),
            arrowprops=dict(edgecolor="black",facecolor="grey", headwidth=8, width=1, headlength=10)
            )
        _axes[2].annotate(xy=(_xstart, 0), text="", ha="center", va="center", xytext=(_xstart+SHIFT/2, 20), 
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.75),
            arrowprops=dict(edgecolor="black", facecolor="grey", headwidth=8, width=1, headlength=10)
            )

    # _axes[2].plot(circle(_geom.center()))

    # _axes[2].set(ylim=(-20, 20), xlim=(55, 85))
    _axes[2].axis("equal")
    _axes[2].set(xlabel="x (Å)", ylabel="y (Å)", title="Atoms (x,y), with ruler showing shift")

    # for _att in dir(_axes[0]):
    #     if "legend" in _att:
    #         print(_att)
    _handles, _labels = _axes[0].get_legend_handles_labels()
    _fig.legend(_handles, _labels, bbox_to_anchor=(0., 1.1), loc="upper left")
    _handles, _labels = _atominset.get_legend_handles_labels()
    _fig.legend(_handles, _labels, bbox_to_anchor=(0.8, 1.1), loc="upper left")
    _fig.suptitle(fr"{STACK} stacking, rot=${ANGLE}^\circ$, Shift={SHIFT}Å")
    _fig.set_constrained_layout(True)
    mo.hstack([_fig, params], align="start", gap=1)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $\ $
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $\ $
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $\ $
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $\ $
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $\ $
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $\ $
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $\ $
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $\ $
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $\ $
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $\ $
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # No inter-layer coupling
    """)
    return


@app.cell
def _(TILE_DIR, angle_choice, shift_choice, stack_choice):
    _fig, _axes = plt.subplots()
    _file = sisl.get_sile(TILE_DIR / "Ham_bi_no_coupling.nc")
    _adev = _file.a_dev
    _nadev = _file.na_dev
    _E = _file.E
    _ADOS = _file.ADOS(elec=0, atoms=_adev) / _nadev
    _axes.plot(_E, _ADOS)
    mo.hstack([_fig, mo.vstack([stack_choice, angle_choice, shift_choice], align="start", gap=1)], align="start", gap=1)
    return


@app.cell
def _():
    SHIFTS = [0,10,20]
    size = 2
    rank = 0
    np.array_split(SHIFTS, size)[rank]
    return


@app.cell
def _(ANGLE_DIR, SHIFT_DIR):
    test1 = sisl.get_sile(SHIFT_DIR / "Ham_bi_shift.nc").read_hamiltonian()
    test2 = sisl.get_sile(ANGLE_DIR / "Ham_bilayer.nc").read_hamiltonian()
    rows1, cols1 = test1.nonzero()
    rows2, cols2 = test2.nonzero()
    len(rows1), len(rows2)
    return


if __name__ == "__main__":
    app.run()
