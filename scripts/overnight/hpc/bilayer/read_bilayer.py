import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")

with app.setup:
    import sisl
    import matplotlib.pyplot as plt
    import numpy as np

    from pathlib import Path
    import matplotlib.colors as mcolors


@app.cell
def _():
    SCRIPT_DIR = Path(__file__).parent
    WORK_DIR = Path.home() / "w3"
    return (WORK_DIR,)


@app.cell
def _(WORK_DIR):
    NC = 0
    N_target = 30
    V = 0

    for _d in (WORK_DIR / "bilayer_data").glob(f"TBT-NC{NC}_*_to_{N_target}-v{V}"):
        print(_d.name)
    return NC, N_target, V


@app.cell
def _(NC):
    N_start = 1 + 2*(1+NC)
    N_bases = range(N_start, 13)
    return (N_bases,)


@app.cell
def _(NC, N_target, V, WORK_DIR):
    DATA_DIR = lambda N : WORK_DIR / "bilayer_data" / f"TBT-NC{NC}_{N}_to_{N_target}-v{V}"
    return (DATA_DIR,)


@app.cell
def _(DATA_DIR):
    tbtout = sisl.get_sile(DATA_DIR(3) / "trans.TBT.nc")
    for _att in dir(tbtout):
        if "tra" in _att and not _att.startswith("_"):
            print(_att)
    print(tbtout.elecs)
    print(tbtout.transmission)
    return (tbtout,)


@app.cell
def _():
    # tbtout.
    return


@app.cell
def _(N_bases):
    cmap = plt.get_cmap("tab10")  # or any other colormap
    norm = mcolors.Normalize(vmin=3, vmax=max(N_bases)) # 
    return cmap, norm


@app.cell
def _(DATA_DIR, NC, N_bases, N_target, cmap, norm):
    _fig, _ax = plt.subplots()
    _cases = N_bases
    # _cases = [3,4,7]
    for _N in _cases:
        _tbtout = sisl.get_sile(DATA_DIR(_N) / "trans.TBT.nc")
        _ax.plot(_tbtout.E, _tbtout.transmission('top', 'bottom'), 
            label=f"N={_N}", color=cmap(norm(_N)),
            marker="", linestyle="-")

    _ax.set(xlabel="E [eV]", ylabel="T",
        # xlim=(-2,2)
        )
    # _ax.axvline(0, color="k", linestyle="--")
    _ax.axhline(0, color="k", linestyle="--", label=r"$E_f$")
    _ax.legend()
    _fig.suptitle(f"Transport from Top to Bottom, NC={NC} & target={N_target}")
    return


@app.cell
def _(tbtout):
    tbtout.ADOS
    return


@app.cell
def _(tbtout):
    _fig, _ax = plt.subplots()

    _ax.plot(tbtout.E, tbtout.ADOS(elec='top'))
    return


if __name__ == "__main__":
    app.run()
