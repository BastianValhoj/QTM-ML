import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")

with app.setup:
    import sisl
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    import os
    from typing import cast

    import marimo as mo

    # import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from scipy.signal import find_peaks


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Extract device atoms of downfolded region:
    """)
    return


@app.function
def read_fdf_device_atoms(filepath):
    """
    Reads a .fdf file and extracts atom indices from a TBT.Atoms.Device block.
    Returns 0-based indices (converts from 1-based fdf format).
    """
    import re
    indices = []
    in_block = False

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if re.match(r'%block\s+TBT\.Atoms\.Device', line, re.IGNORECASE):
                in_block = True
                continue
            if re.match(r'%endblock', line, re.IGNORECASE):
                if in_block:
                    break
            if not in_block:
                continue

            # Match range: atom [X -- Y]
            range_match = re.match(r'atom\s+\[(\d+)\s*--\s*(\d+)\]', line)
            if range_match:
                start, end = int(range_match.group(1)), int(range_match.group(2))
                indices.extend(range(start, end + 1))
                continue

            # Match single: atom X
            single_match = re.match(r'atom\s+(\d+)', line)
            if single_match:
                indices.append(int(single_match.group(1)))

    return np.array(indices) - 1


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # define the extrapolation case we want to investigate
    """)
    return


@app.cell
def _():
    NC = 1
    Nstart = 1 + 2*(NC+1) # min number of tiles
    N_bases = range(Nstart, 13)
    N_target = 50
    print(N_bases)
    return NC, N_bases, N_target


@app.cell
def _(NC, N_target):
    from ase.io.cube import DATA
    SCRIPT_DIR = Path(__file__).parent
    WORK_DIR = Path.home() / "w3"
    DATA_DIR = WORK_DIR / "rsse_data"
    for _d in DATA_DIR.glob(f"TBT-NC{NC}*_to_{N_target}"):
        print(_d.name)
    return (DATA_DIR,)


@app.cell
def _(DATA_DIR, NC, N_target):
    OUT_DIR = lambda N: DATA_DIR / f"TBT-NC{NC}_{N}_to_{N_target}"
    # OUT_DIR = lambda N: DATA_DIR / f"TBT-test_{N}_to_{N_target}"
    return (OUT_DIR,)


@app.cell
def _(OUT_DIR):
    _N = 12
    print(OUT_DIR(_N))
    print([_file for _file in os.listdir(OUT_DIR(_N)) if ("TBT" in _file) and (".nc" in _file)])
    return


@app.cell
def _(N_bases, OUT_DIR):
    Ham_elec_big = sisl.get_sile(OUT_DIR(N_bases[0]) / "Ham_elec_big.nc").read_hamiltonian()
    Ham_elec_small = sisl.get_sile(OUT_DIR(N_bases[0]) / "Ham_elec_small.nc").read_hamiltonian()
    print(Ham_elec_small.na, Ham_elec_big.na)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Read RSSE from TBT
    """)
    return


@app.cell
def _(N_bases, OUT_DIR):
    tbtse_top = sisl.get_sile(OUT_DIR(N_bases[0])  / "tbt-top.TBT.SE.nc")
    tbtse_bottom = sisl.get_sile(OUT_DIR(N_bases[0])  / "tbt-bottom.TBT.SE.nc")
    pivot = tbtse_top.pivot(0, in_device=True, sort=True)
    print(pivot.T.shape)
    print(tbtse_top.a_down(elec=0))
    return (tbtse_top,)


@app.cell
def _(tbtse_top):
    eta = tbtse_top.eta()
    energies = tbtse_top.E
    num_energies = tbtse_top.nE
    print(tbtse_top.elecs)
    return


@app.cell
def _(N_bases, OUT_DIR):
    tbtout_top = sisl.get_sile(OUT_DIR(N_bases[0]) / "tbt-top.TBT.nc")
    tbtout_bot = sisl.get_sile(OUT_DIR(N_bases[0]) / "tbt-bottom.TBT.nc")
    geom_top = tbtout_top.read_geometry()
    return tbtout_bot, tbtout_top


@app.cell
def _(NC, N_bases, N_target, OUT_DIR):
    cmap = plt.get_cmap("tab10")  # or any other colormap
    norm = mcolors.Normalize(vmin=3, vmax=max(N_bases))

    # _fig, _axes = plt.subplots(1,2, sharey=True)
    from mytools.plots import thesis_fig
    _fig, _axes = thesis_fig(subplots=(1,2), sharey=True)

    for _N in N_bases:
        _tbtout_top = sisl.get_sile(OUT_DIR(_N) / "tbt-top.TBT.nc")
        _tbtout_bottom = sisl.get_sile(OUT_DIR(_N) / "tbt-bottom.TBT.nc")

        _color = cmap(norm(_N))
        _axes[0].plot(_tbtout_top.E, _tbtout_top.ADOS(atoms=_tbtout_top.a_dev)/_tbtout_top.na_d, '.', alpha=1., color=_color)
        _axes[1].plot(_tbtout_bottom.E, _tbtout_bottom.ADOS(atoms=_tbtout_bottom.a_dev)/_tbtout_bottom.na_d, '.', alpha=1., color=_color)

    _temp = sisl.get_sile(OUT_DIR(N_bases[-1]) / "tbt-og.TBT.nc")
    _ados = _temp.ADOS(atoms=_temp.a_dev)/_temp.na_d
    _peaks_idx, _  = find_peaks(_ados, prominence=0.) # tune prominence to filter noise

    with np.printoptions(formatter={"float": lambda x: f"{x:05.2f} |", "int": lambda x: f"{x:>4.0f}  |", }, linewidth=120):
        print(f"                       N : {np.array(N_bases)}")
        print("-"*100)
        print(f"         N_big / N_small : {N_target / np.array(N_bases)}")
        print(f"   (N_big - 2) / N_small : {(N_target-2) / np.array(N_bases)}")
        print(f"  (N_big - NC) / N_small : {(N_target-NC) / np.array(N_bases)}")
        print(f"(N_big - 2*NC) / N_small : {(N_target-2*NC) / np.array(N_bases)}")

    for _ax in _axes:
        _ax.plot(_temp.E, _ados, color="k", alpha=0.5)
        _ax.set(ylabel="DOS", xlabel="E")
        _ax.grid()
        for _idx in _peaks_idx:
            _ax.axvline(_temp.E[_idx], color="k", linestyle="--", alpha=0.5, linewidth=0.8)



    _ncols = np.ceil( (len(N_bases)+1) / 3)
    _fig.legend([f"N={_n}" for _n in N_bases] + [f"Exact (N={_N})"], ncols=_ncols, bbox_to_anchor=(0.5, -0.1), loc="upper center")
    _fig.suptitle(f"Extrapolation to target N={N_target}, NC={NC}", y=1.2)
    _axes[0].set(title="Top electrode")
    _axes[1].set(title="Bottom electrode")
    _fig.savefig(f"figures/read_rsse_NC{NC}_{N_bases[0]}-{N_bases[-1]}_to_{N_target}")
    _fig
    return cmap, norm


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Possible explanation for deviation in DOS:

    - 6, 10 are divisors of 30, but not 8.
      - Not divisors

    > check whether other divisors are well behaved

    > check whether other non-divisors are problems
    """)
    return


@app.cell
def _():
    print(f"{2:<2} test")
    print(f"{2   } test")
    return


@app.cell
def _(OUT_DIR, cmap, norm):
    _fig, _axes = plt.subplots(1,2, sharey=True)

    _N = 6
    _tbtout_top = sisl.get_sile(OUT_DIR(_N) / "tbt-top.TBT.nc")
    _tbtout_bottom = sisl.get_sile(OUT_DIR(_N) / "tbt-bottom.TBT.nc")
    _axes[0].plot(_tbtout_top.E, _tbtout_top.ADOS(atoms=_tbtout_top.a_dev)/_tbtout_top.na_d, '.', label=f"N={8}", color=cmap(norm(_N)))
    _axes[1].plot(_tbtout_bottom.E, _tbtout_bottom.ADOS(atoms=_tbtout_bottom.a_dev)/_tbtout_bottom.na_d, '.', color=cmap(norm(_N)))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Plot geometry with highlighted top and bottom
    """)
    return


@app.cell
def _(N_target, tbtout_bot, tbtout_top):
    mo.stop(N_target > 30, output="### File too large -- are you sure you wish to plot?")

    geom_bot = tbtout_bot.read_geometry()
    _fig = geom_bot.plot(axes="xy", backend="matplotlib", 
        atoms_style=[
            dict(atoms=tbtout_bot.a_dev, color="red", border_width=0.5),
            dict(atoms=tbtout_top.a_dev, color="blue", border_width=0.5),
            dict(atoms=np.delete(range(tbtout_top.na), np.unique(np.concat([tbtout_bot.a_dev, tbtout_top.a_dev]))), color="grey", border_width=0.5)
            ], 
        show_cell=False,
    )
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Checking RSSE
    """)
    return


@app.cell
def _(tbtout_bot):
    from six import b
    Hs = []
    Ss = []
    RSSE = []
    Es = []
    with sisl.io.tbtgfSileTBtrans(tbtout_bot.file.parent / "RSSE-OG.TBTGF") as _file:
        _nspin, _no, _ks, _Es = _file.read_header()
        for _ispin, _new_k, _k, _E in _file:

            if _new_k:
                _Hk, _Sk = _file.read_hamiltonian()
                Hs.append(_Hk)
                Ss.append(_Sk)
            RSSE.append(_file.self_energy(E=_E, k=_k))
            Es.append(_E)

    RSSE = np.array(RSSE).squeeze()
    H = np.array(Hs).squeeze()
    S = np.array(Ss).squeeze()
    energy = np.array(np.real(Es))
    return


if __name__ == "__main__":
    app.run()
