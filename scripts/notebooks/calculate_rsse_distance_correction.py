import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import sisl
    import numpy as np
    from mytools.construct import all_armchair, make_edge
    from mytools.scalingv2 import rsse_mapping

    from pathlib import Path


@app.cell
def _():
    script_dir = Path(__file__).parent
    return (script_dir,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Create a single function that extrapolates the rsse of a *small* structure to a structure of tiles `N0*N1`
    """)
    return


@app.cell
def _(script_dir):
    Ham0 = sisl.get_sile(script_dir / "Ham0.nc").read_hamiltonian()
    return (Ham0,)


@app.cell
def _():
    N_small = 7
    N_big = 13
    NC = 1

    eta = 1e-3
    nk1 = lambda N: int(np.ceil(1200/N))

    emax = 1.0
    emin = -emax
    estep = 0.1
    energies = np.arange(emin, emax+estep, estep).round(3)
    return NC, N_big, N_small, eta, nk1


@app.cell
def _(Ham0, N_big, N_small, eta, nk1):
    rsse_small = sisl.RealSpaceSE(Ham0, 0, 1, (N_small, N_small, 1))
    rsse_small.setup(eta=eta, bz=sisl.MonkhorstPack(Ham0, [1, nk1(N_small), 1]))

    rsse_big = sisl.RealSpaceSE(Ham0, 0, 1, (N_big, N_big, 1))
    rsse_big.setup(eta=eta, bz=sisl.MonkhorstPack(Ham0, [1, nk1(N_big), 1]))
    return rsse_big, rsse_small


@app.cell
def _(rsse_big, rsse_small):
    Ham_elec_small, elec_idx_small = rsse_small.real_space_coupling(ret_indices=True)
    Ham_elec_big, elec_idx_small = rsse_big.real_space_coupling(ret_indices=True)
    return Ham_elec_big, Ham_elec_small


@app.cell
def _(Ham0, N_big, N_small):
    geom_edge_small = make_edge(geom=Ham0.geometry, N0=N_small, N1=N_small)
    geom_edge_big = make_edge(geom=Ham0.geometry, N0=N_big, N1=N_big)

    # Ham_elec_small.geometry.plot(axes="xy")
    return geom_edge_big, geom_edge_small


@app.cell
def _(
    Ham0,
    Ham_elec_big,
    Ham_elec_small,
    NC,
    N_big,
    N_small,
    geom_edge_big,
    geom_edge_small,
):
    big_to_small_idx  = rsse_mapping(Ham_elec_small, Ham_elec_big, geom_edge_small, geom_edge_big, N_small, N_big, Ham0.na, NC)
    mapped_indices = list(big_to_small_idx.values())
    return


if __name__ == "__main__":
    app.run()
