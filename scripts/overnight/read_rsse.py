import marimo

__generated_with = "0.23.4"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import sisl
    from tqdm.auto import tqdm
    from pathlib import Path


@app.cell
def _():
    script_dir = Path(__file__).parent
    output_dir = script_dir / "rsse_data"

    return (output_dir,)


@app.cell
def _():
    N1 = 6
    N2 = 12
    return N1, N2


@app.cell
def _(N1, N2, output_dir):
    dir = output_dir / f"TBT-test_{N1}_to_{N2}"
    dir.exists()
    return (dir,)


@app.cell
def _(dir):
    rsse_sile = sisl.io.tbtgfSileTBtrans(dir / "tbt.TBT.SE.nc")
    energies = []
    RSSE = []
    Hks = []
    Sks = []
    with sisl.io.tbtgfSileTBtrans(dir / "tbt.TBT.SE.nc", "r") as sile:
        num_spin, num_orb, kpts, energies = sile.read_header()
        for ispin, new_k, k, E in sile:
            if new_k:
                H, S = sile.read_hamiltonian()
    print(energies)
    return


@app.cell
def _(dir):
    import os
    os.chdir(dir)
    os.getcwd(), os.listdir()
    return (os,)


@app.cell
def _(os):
    os.getcwd()
    return


if __name__ == "__main__":
    app.run()
