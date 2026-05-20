import marimo

__generated_with = "0.23.6"
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
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    from mytools.plots import thesis_fig, label_subplots


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
    # Define the extrapolation case we want to investigate
    """)
    return


@app.cell
def _():
    NC = 2
    Nstart = 1 + 2*(NC+1) # min number of tiles
    N_bases = range(Nstart, 13)
    N_target = 100
    print(N_bases)
    return NC, N_bases, N_target


@app.cell
def _(NC, N_target):
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
    _N = 10
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
    return tbtse_bottom, tbtse_top


@app.cell
def _(tbtse_bottom, tbtse_top):
    print("Atoms in down-folded regions:")
    print(f"Top : {tbtse_top.na_dev}")
    print(f"Top : {tbtse_bottom.na_dev}")
    return


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
def _():
    ~ np.array([True, False])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Plot using bottom row for values below threshold
    """)
    return


@app.cell
def _(N_bases):
    cmap = plt.get_cmap("tab10")  # or any other colormap
    norm = mcolors.Normalize(vmin=3, vmax=max(N_bases)) # 
    return cmap, norm


@app.cell
def _(NC, N_bases):
    ################# _i :  0    1    2    3    4    5    6    7    8    9
    ################# _N :  3    4    5    6    7    8    9    10   11   12
    markers = [ 
        ('o', 'none'), # make the maker facecolor 'none' / hollow
        ('s', None), # None defaults to marker color
        ('D', None), 
        ('^', None), 
        ('v', None), 
        ('x', None), 
        ('+', None), 
        ('*', None), 
        ('p', None), 
        ('h', None)]  # 10 markers for N=3..12
    markersize=4
    _stop = 14
    _start = 1+2*(1+NC)
    print([(2*NC+_i)%len(markers) for _i, _N in enumerate(range(_start, _stop))])
    marker_map = {_N: markers[(2*NC+_i)%len(markers)] for _i, _N in enumerate(N_bases)}
    print(marker_map)
    return marker_map, markersize


@app.cell
def _(NC, N_bases, N_target, OUT_DIR, cmap, marker_map, markersize, norm):

    _fig, _axes = thesis_fig(subplots=(2,2), sharey="row", sharex=False, gridspec_kw={"height_ratios": [1, 1/2]})

    # read data for plotting Ground Truth (no extrapolation)
    _temp = sisl.get_sile(OUT_DIR(N_bases[-1]) / "tbt-og.TBT.nc")
    _ados = _temp.ADOS(atoms=_temp.a_dev)/_temp.na_d
    _peaks_idx, _  = find_peaks(_ados, prominence=0.) # tune prominence to filter noise

    # every row share y axis independently
    for _i in range(2):
        _axes[_i,0].sharey(_axes[_i, 1])

    # Determine split for when values in one should be on second plot
    _ysplit = 0
    _alphas = 0.75 # transparency of scatter plot
    _needs_bottom_row = False

    _Ns_to_show = N_bases
    # loop over different tilings
    for _N in _Ns_to_show:
        _tbtout_top = sisl.get_sile(OUT_DIR(_N) / "tbt-top.TBT.nc")
        _tbtout_bottom = sisl.get_sile(OUT_DIR(_N) / "tbt-bottom.TBT.nc")
        _ados_top = _tbtout_top.ADOS(atoms=_tbtout_top.a_dev) / _tbtout_top.na_d
        _ados_bottom = _tbtout_bottom.ADOS(atoms=_tbtout_bottom.a_dev) / _tbtout_bottom.na_d

        # mask for positive valued indices
        _pos_mask_top = _ados_top > _ysplit
        _pos_mask_bottom = _ados_bottom > _ysplit

        ## use the standardized colors (each value of N, e.g. N=12, has the same color even when if the total number of scatter plots change)
        _color = cmap(norm(_N))
        _marker, _mfc = marker_map[_N]

        # plot positive valued ADOS in first row
        _axes[0,0].plot(_tbtout_top.E[_pos_mask_top], _ados_top[_pos_mask_top], alpha=_alphas, color=_color, marker=_marker, markerfacecolor=_mfc, markersize=markersize, linestyle="none",label=f"N={_N}")
        _axes[0,1].plot(_tbtout_bottom.E[_pos_mask_bottom], _ados_bottom[_pos_mask_bottom], alpha=_alphas, color=_color, marker=_marker, markerfacecolor=_mfc, markersize=markersize, linestyle="none")

        # if not all values where positive, use second row to show negative data points
        if not (_pos_mask_top.all() and _pos_mask_bottom.all()):
            _needs_bottom_row = True
            _axes[1,0].plot(_tbtout_top.E[~ _pos_mask_top], _ados_top[~ _pos_mask_top], alpha=_alphas, color=_color, marker=_marker, markerfacecolor=_mfc, markersize=markersize, linestyle="none")
            _axes[1,1].plot(_tbtout_bottom.E[~ _pos_mask_bottom], _ados_bottom[~ _pos_mask_bottom], alpha=_alphas, color=_color, marker=_marker, markerfacecolor=_mfc, markersize=markersize, linestyle="none")

    # if no negative ADOS values, remove second row, otherwise make columns of subplots share x axis
    if not _needs_bottom_row:
        _fig.delaxes(_axes[1,0])
        _fig.delaxes(_axes[1,1])
    else:
        _axes[0,0].sharex(_axes[1,0])
        _axes[0,1].sharex(_axes[1,1])

    for _ax in _axes[0,:]:
        _ax.plot(_temp.E, _ados, color="k", alpha=0.5, label=f"Exact (N={N_bases[-1]})")


    ## Make inset in top row subplots
    for _col, _ax in enumerate([_axes[0,0], _axes[0,1]]):
        _inset = inset_axes(_ax, width="100%", height="100%", loc="upper center", 
        bbox_to_anchor=(0.37, 0.6, 0.4, 0.4), bbox_transform=_ax.transAxes
        )

        # re-plot the same data on the inset
        for _N in _Ns_to_show:
            _inset_tbt = sisl.get_sile(OUT_DIR(_N) / f"tbt-{'top' if _col == 0 else 'bottom'}.TBT.nc")
            _inset_ados = _inset_tbt.ADOS(atoms=_inset_tbt.a_dev) / _inset_tbt.na_d
            _mask = (_inset_tbt.E >= -1) & (_inset_tbt.E <= +1)

            _color = cmap(norm(_N))
            _marker, _mfc = marker_map[_N]
            _inset.plot(_inset_tbt.E[_mask], _inset_ados[_mask], alpha=_alphas, color=_color, marker=_marker, markerfacecolor=_mfc, markersize=markersize, linestyle="none")

        _mask = (_temp.E >= -1) & (_temp.E <= +1)
        _inset.plot(_temp.E[_mask], _ados[_mask], color="k", alpha=0.5)
        _inset.set(xlim=(-1,1))
        _inset.grid(True, alpha=0.4)
        _inset.tick_params(labelsize=9) 

        mark_inset(_ax, _inset, 3, 4, linestyle="--", lw=1, edgecolor="k", alpha=0.8, zorder=0)


    with np.printoptions(formatter={"float": lambda x: f"{x:05.2f} |", "int": lambda x: f"{x:>4.0f}  |", }, linewidth=120):
        print(f"                           N : {np.array(N_bases)}")
        print("-"*100)
        print(f"             N_big / N_small : {N_target / np.array(N_bases)}")
        print(f"       (N_big - 2) / N_small : {(N_target-2) / np.array(N_bases)}")
        print(f"      (N_big - NC) / N_small : {(N_target-NC) / np.array(N_bases)}")
        print(f"    (N_big - 2*NC) / N_small : {(N_target-2*NC) / np.array(N_bases)}")
        print(f"(N_big - 2*NC - 2) / N_small : {(N_target-2*NC-2) / np.array(N_bases)}")


    # Set limits, labels, and peak/pole lines
    for _i, _ax in enumerate(_axes.flatten()):
        _ax.grid()
        _ylow, _yhigh = _ax.get_ylim()
        if _i in [0,1]:
            _ax.set(ylim=(min(0, _ysplit), _yhigh))
        if _i in [2,3]:
            _ax.set(ylim=(_ylow+_ysplit-1e-2, max(0, _ysplit)))

        # if _i in [0,2]:
        #     _ax.set(ylabel=r"DOS $\left[\mathrm{eV}^{-1}\right]$")
        if _i in [2,3]:
            _ax.set(xlabel="E [eV]")
        if not _needs_bottom_row:
            _ax.set(xlabel="E [eV]")
        for _idx in _peaks_idx:
            _ax.axvline(_temp.E[_idx], color="k", linestyle="--", alpha=0.75, linewidth=0.8, zorder=0)

    label_subplots(_axes)

    # Create legend (placement depends on if there is a second row or not)
    _ncols = np.ceil( (len(N_bases)+1) / 3)
    _handles, _labels = _axes[0,0].get_legend_handles_labels()
    if _needs_bottom_row:
        _fig.legend(_handles, _labels, ncols=_ncols, bbox_to_anchor=(0.5, -0.01), loc="upper center")
    else:
        _fig.legend(_handles, _labels, ncols=_ncols, bbox_to_anchor=(0.5, 0.29), loc="upper center")
    _fig.suptitle(f"Extrapolation to target N={N_target}, NC={NC}", y=1.01)
    _axes[0,0].set(title="Top layer")
    _axes[0,1].set(title="Bottom layer")
    _fig.supylabel(r"DOS $\left[\mathrm{eV}^{-1}\right]$")
    _fig.savefig(f"figures/read_rsse_NC{NC}_{N_bases[0]}-{N_bases[-1]}_to_{N_target}")
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Plot for using subplots below and above for extremes
    """)
    return


@app.cell
def _(
    NC,
    N_bases,
    N_target,
    OUT_DIR,
    cmap,
    marker_map,
    markersize,
    norm,
    tbtse_bottom,
):
    _fig, _axes = thesis_fig(subplots=(3,2), sharey=False, sharex=False, gridspec_kw={"height_ratios": [1/2, 1, 1/2]})

    # read data for plotting Ground Truth (no extrapolation)
    _temp = sisl.get_sile(OUT_DIR(N_bases[-1]) / "tbt-og.TBT.nc")
    _ados = _temp.ADOS(atoms=_temp.a_dev)/_temp.na_d
    _peaks_idx, _  = find_peaks(_ados, prominence=0.) # tune prominence to filter noise

    # every row share y axis independently
    for _i in range(2):
        _axes[_i,0].sharey(_axes[_i, 1])

    # Determine split for when values in one should be on second plot
    _ylow_split = -0.01
    _needs_bottom_row = False

    _yhigh_split = np.max(_ados) + 1e-1  # adjust threshold to taste
    _needs_top_row = False

    _alphas = 0.75 # transparency of scatter plot

    _Ns_to_show = N_bases
    # _Ns_to_show = [3,4,5,6,12]
    # loop over different tilings
    scatter_plots = []
    for _N in _Ns_to_show:
        _tbtout_top = sisl.get_sile(OUT_DIR(_N) / "tbt-top.TBT.nc")
        _tbtout_bottom = sisl.get_sile(OUT_DIR(_N) / "tbt-bottom.TBT.nc")
        _ados_top = _tbtout_top.ADOS(atoms=_tbtout_top.a_dev) / _tbtout_top.na_d
        _ados_bottom = _tbtout_bottom.ADOS(atoms=_tbtout_bottom.a_dev) / _tbtout_bottom.na_d

        if np.max(np.abs(tbtse_bottom.E)) > 4:
            print(f"Bottom electrode for N = {_N} is > |4|")
        # mask for positive valued indices and below threshold
        _mask_top = (_ados_top > _ylow_split) & (_ados_top < _yhigh_split)
        _mask_bottom = (_ados_bottom > _ylow_split) & (_ados_bottom < _yhigh_split)

        ## use the standardized colors (each value of N, e.g. N=12, has the same color even when if the total number of scatter plots change)
        _color = cmap(norm(_N))
        _marker, _mfc = marker_map[_N]

        # plot positive valued ADOS in first row
        plots, = _axes[1,0].plot(_tbtout_top.E[_mask_top], _ados_top[_mask_top], alpha=_alphas, color=_color, marker=_marker, markerfacecolor=_mfc, markersize=markersize, linestyle="none")
        _axes[1,1].plot(_tbtout_bottom.E[_mask_bottom], _ados_bottom[_mask_bottom], alpha=_alphas, color=_color, marker=_marker, markerfacecolor=_mfc, markersize=markersize, linestyle="none")
        scatter_plots.append(plots)

        # if not all values where positive, use second row to show negative data points
        if not (_mask_top.all() and _mask_bottom.all()):
            # _needs_bottom_row = True
            # _needs_top_row = True
            _mask_top_low = (_ados_top < _ylow_split)
            _mask_bottom_low = (_ados_bottom < _ylow_split)
            if (_mask_top_low.any()) or (_mask_bottom_low.any()):
                print(f"Values below {_ylow_split}")
                # print(_ados_top[_mask_top_low])
                # print(_ados_bottom[_mask_bottom_low])
                _needs_bottom_row = True
                _axes[2,0].plot(_tbtout_top.E[_mask_top_low], _ados_top[_mask_top_low], alpha=_alphas, color=_color, marker=_marker, markerfacecolor=_mfc, markersize=markersize, linestyle="none")
                _axes[2,1].plot(_tbtout_bottom.E[_mask_bottom_low], _ados_bottom[_mask_bottom_low], alpha=_alphas, color=_color, marker=_marker, markerfacecolor=_mfc, markersize=markersize, linestyle="none")

            _mask_top_high = (_ados_top > _yhigh_split)
            _mask_bottom_high = (_ados_bottom > _yhigh_split)
            if (_mask_top_high.any()) or (_mask_bottom_high.any()):
                print(f"Values above {_yhigh_split}")
                # print(_ados_top[_mask_top_high])
                # print(_ados_bottom[_mask_bottom_high])
                _needs_top_row = True
                _axes[0,0].plot(_tbtout_top.E[_mask_top_high], _ados_top[_mask_top_high], alpha=_alphas, color=_color, marker=_marker, markerfacecolor=_mfc, markersize=markersize, linestyle="none")
                _axes[0,1].plot(_tbtout_bottom.E[_mask_bottom_high], _ados_bottom[_mask_bottom_high], alpha=_alphas, color=_color, marker=_marker, markerfacecolor=_mfc, markersize=markersize, linestyle="none")


    # if no negative ADOS values, remove second row, otherwise make columns of subplots share x axis
    _axes[0,0].sharex(_axes[1,0])
    _axes[0,1].sharex(_axes[1,1])

    _axes[2,0].sharex(_axes[1,0])
    _axes[2,1].sharex(_axes[1,1])

    _axes[0,0].sharey(_axes[0,1])
    _axes[1,0].sharey(_axes[1,1])
    _axes[2,0].sharey(_axes[2,1])


    if not _needs_top_row:
        print("Does not need top row")
        if not _needs_bottom_row:
            print("Also does not need bottom row")
            _fig.delaxes(_axes[2,0])
            _fig.delaxes(_axes[2,1])
        _fig.delaxes(_axes[0,0])
        _fig.delaxes(_axes[0,1])
    elif _needs_top_row and (not _needs_bottom_row):
        print("Needs top row, do not need bottom row")
        _fig.delaxes(_axes[2,0])
        _fig.delaxes(_axes[2,1])

    for _i, _ax in enumerate(_axes[1,:]):
        if _i == 0:
            plots, = _ax.plot(_temp.E, _ados, color="k", alpha=0.5)
            scatter_plots.append(plots)
        else:
            _ax.plot(_temp.E, _ados, color="k", alpha=0.5)


    ## Make inset in top row subplots
    for _col, _ax in enumerate([_axes[1,0], _axes[1,1]]):
        _inset = inset_axes(_ax, width="100%", height="100%", loc="upper center", 
        bbox_to_anchor=(0.37, 0.6, 0.4, 0.4), bbox_transform=_ax.transAxes
        )

        # re-plot the same data on the inset
        for _N in _Ns_to_show:
            _inset_tbt = sisl.get_sile(OUT_DIR(_N) / f"tbt-{'top' if _col == 1 else 'bottom'}.TBT.nc")
            _inset_ados = _inset_tbt.ADOS(atoms=_inset_tbt.a_dev) / _inset_tbt.na_d
            _mask = (_inset_tbt.E >= -1) & (_inset_tbt.E <= +1)

            _color = cmap(norm(_N))
            _marker, _mfc =marker_map[_N]
            _inset.plot(_inset_tbt.E[_mask], _inset_ados[_mask], alpha=_alphas, color=_color, marker=_marker, markerfacecolor=_mfc, markersize=markersize, linestyle="none")

        _mask = (_temp.E >= -1) & (_temp.E <= +1)
        _inset.plot(_temp.E[_mask], _ados[_mask], color="k", alpha=0.5)
        _inset.set(xlim=(-1,1))
        _inset.grid(True, alpha=0.4)
        _inset.tick_params(labelsize=9) 

        mark_inset(_ax, _inset, 3, 4, linestyle="--", lw=1, edgecolor="k", alpha=0.8, zorder=0)


    with np.printoptions(formatter={"float": lambda x: f"{x:05.2f} |", "int": lambda x: f"{x:>4.0f}  |", }, linewidth=120):
        print(f"                           N : {np.array(N_bases)}")
        print("-"*100)
        print(f"             N_big / N_small : {N_target / np.array(N_bases)}")
        print(f"       (N_big - 2) / N_small : {(N_target-2) / np.array(N_bases)}")
        print(f"      (N_big - NC) / N_small : {(N_target-NC) / np.array(N_bases)}")
        print(f"    (N_big - 2*NC) / N_small : {(N_target-2*NC) / np.array(N_bases)}")
        print(f"(N_big - 2*NC - 2) / N_small : {(N_target-2*NC-2) / np.array(N_bases)}")


    # Set limits, labels, and peak/pole lines
    for _i, _ax in enumerate(_axes.flatten()):
        _ax.grid()
        _ylow, _yhigh = _ax.get_ylim()
        if _needs_top_row and (_i in [0,1]):
            _ax.set(ylim=(_yhigh_split, _yhigh+_yhigh_split))
        if _i in [2,3]:
            _ax.set(ylim=(min(0,_ylow_split), _yhigh_split))
        if _i in [4,5]:
            _ax.set(ylim=(_ylow+_ylow_split, max(_ylow_split, 0)))

        if (_i % 2 == 0):
            _ax.set(ylabel=r"DOS $\left[\mathrm{eV}^{-1}\right]$")

        if _needs_bottom_row and (_i in [4,5]):
            _ax.set(xlabel="E [eV]")
        elif not _needs_bottom_row and (_i in [2,3]):
            _ax.set(xlabel="E [eV]")
        for _idx in _peaks_idx:
            _ax.axvline(_temp.E[_idx], color="k", linestyle="--", alpha=0.5, linewidth=0.8, zorder=0)



    # Create legend (placement depends on if there is a second row or not)
    _ncols = np.ceil( (len(N_bases)+1) / 3)
    if _needs_bottom_row:
        _fig.legend(scatter_plots, [f"N={_n}" for _n in _Ns_to_show] + [f"Exact (N={_N})"], ncols=_ncols, bbox_to_anchor=(0.5, -0.01), loc="upper center")
    else:
        _fig.legend(scatter_plots, [f"N={_n}" for _n in _Ns_to_show] + [f"Exact (N={_N})"], ncols=_ncols, bbox_to_anchor=(0.5, 0.2), loc="upper center")

    if _needs_top_row:
        _vspace = 1.
    else:
        _vspace = 0.76

    _fig.suptitle(f"Extrapolation to target N={N_target}, NC={NC}", y=_vspace)
    if _needs_top_row:
        _axes[0,0].set(title="Top layer")
        _axes[0,1].set(title="Bottom layer")
    else:
        _axes[1,0].set(title="Top layer")
        _axes[1,1].set(title="Bottom layer")

    if not _needs_top_row:
        _axes = _axes[[1,2,0]]
    label_subplots(_axes)
    _fig.tight_layout()
    _fig.savefig(f"figures/read_rsse_NC{NC}_{N_bases[0]}-{N_bases[-1]}_to_{N_target}")
    print(_axes.shape)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Possible explanation for deviation in DOS:

    - 6, 10 are divisors of 30, but not 8.
      - Not divisors

    > check whether other divisors are well behaved

    > check whether other non-divisors are problems
    """)
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
    # Plot geometry with highlighted top and bottom
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
    # Checking RSSE
    """)
    return


@app.cell
def _(tbtout_bot):
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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
