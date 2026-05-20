import marimo

__generated_with = "0.23.6"
app = marimo.App()

with app.setup:
    import sisl
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    from itertools import product
    import pandas as pd

    from tqdm.auto import tqdm


@app.cell
def _():
    NC = 0
    Nstart = 1 + 2*(NC+1) # min number of tiles
    N_bases = range(Nstart, 13)
    N_target = 30
    print(N_bases)
    return


@app.cell
def _():
    SCRIPT_DIR = Path(__file__).parent
    WORK_DIR = Path.home() / "w3"
    DATA_DIR = WORK_DIR / "rsse_data"
    FIG_DIR = SCRIPT_DIR / "figures"
    return DATA_DIR, FIG_DIR


@app.cell
def _(DATA_DIR):
    OUT_DIR = lambda NC, N, target: DATA_DIR / f"TBT-NC{NC}_{N}_to_{target}"
    return (OUT_DIR,)


@app.function
def valid(NC, N):
    if NC == 1 and N < 5: return False
    if NC == 2 and N < 7: return False
    return True


@app.cell
def _():
    targets = [30, 50, 100]
    NCs = [0,1,2]
    Ns = range(3,13)
    NC_target_combo = [(_nc, _n, _nt) for _nc, _n, _nt in product(NCs, Ns, targets) if valid(_nc, _n)]
    print(NC_target_combo)
    return NC_target_combo, NCs, targets


@app.cell
def _(NC_target_combo, OUT_DIR):
    tbtse_og = sisl.get_sile(OUT_DIR(0, 3, 30) / "tbt-og.TBT.nc")
    ados_og = tbtse_og.ADOS(atoms=tbtse_og.a_dev) / tbtse_og.na_dev

    use_global = True

    rms_top = []
    rms_bottom = []
    for i, (_nc, _n, _nt) in enumerate(tqdm(NC_target_combo)):
        _tbtse_top = sisl.get_sile(OUT_DIR(_nc, _n, _nt) / "tbt-top.TBT.nc")
        _tbtse_bottom = sisl.get_sile(OUT_DIR(_nc, _n, _nt) / "tbt-bottom.TBT.nc")


        _ados_top = _tbtse_top.ADOS(atoms=_tbtse_top.a_dev) / _tbtse_top.na_dev    
        _ados_bottom = _tbtse_bottom.ADOS(atoms=_tbtse_bottom.a_dev) / _tbtse_bottom.na_dev
        if not use_global:
            _tbtse_og = sisl.get_sile(OUT_DIR(_nc, _n, _nt) / "tbt-og.TBT.nc")
            _ados_og =  _tbtse_og.ADOS(atoms=_tbtse_og.a_dev) / _tbtse_og.na_dev
        else:
            _ados_og = ados_og

        _rms_top = np.sqrt(np.mean( (_ados_top - _ados_og)**2))
        _rms_bottom = np.sqrt(np.mean( (_ados_bottom - _ados_og)**2))
        rms_top.append(_rms_top)
        rms_bottom.append(_rms_bottom)
    return rms_bottom, rms_top


@app.cell
def _(NC_target_combo, rms_bottom, rms_top):
    df = pd.DataFrame(NC_target_combo, columns=["NC", "N", "Target"])
    df["Top"] = rms_top
    df["Bottom"] = rms_bottom
    df
    return (df,)


@app.cell
def _(FIG_DIR, NCs, df, targets):
    # _fig, _axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True, sharey=True)
    from mytools.plots import thesis_fig
    _fig, _axes = thesis_fig(subplots=(3,3), sharey=True, sharex="col")

    for _row_idx, _target in enumerate(targets):
        for _col_idx, _nc in enumerate(NCs):
            _ax = _axes[_row_idx, _col_idx]
            _df_sub = df[(df["Target"] == _target) & (df["NC"] == _nc)]

            _ax.semilogy(_df_sub["N"], _df_sub["Top"],    marker="d", label="Top")
            _ax.semilogy(_df_sub["N"], _df_sub["Bottom"], marker="o", label="Bottom")

            if _row_idx == 0:
                _ax.set_title(f"NC={_nc}")
            if _col_idx == 0:
                _ax.set_ylabel(f"Target={_target}")
            _ax.grid()
            _ax.set_xticks(np.unique(_df_sub.N))

    _handles, _labels = _axes[0, 0].get_legend_handles_labels()
    _fig.legend(_handles, _labels, loc="upper right")
    _fig.supxlabel("Tiling, N")
    _fig.supylabel("RMSE")
    _fig.suptitle("RMSE — Top & Bottom", y=1.01)
    _fig.savefig(FIG_DIR / str(Path(__file__).stem))
    _fig
    return


@app.cell
def _():
    str(Path(__file__).stem)
    return


if __name__ == "__main__":
    app.run()
