import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")

with app.setup:
    import sisl
    import numpy as np
    import matplotlib.pyplot as plt

    from ase.lattice import HEX2D, HEX

    from mytools.construct import all_armchair
    from mytools.tbbi import tbbi_opt
    from mytools.plots import thesis_fig, label_subplots

    from pathlib import Path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # System paramters
    """)
    return


@app.cell
def _():
    bond = 1.42
    Vpppi = -2.7
    Vpps = 0.48

    divs = 200

    Rs = (bond*0.1, bond+1e-2)
    Ts = (0.0, Vpppi)
    return Vpppi, Vpps, bond, divs


@app.cell
def _(bond):
    gr_base = all_armchair(bond)
    # gr_base = gr_base.add_vacuum(10, 2)
    return (gr_base,)


@app.cell
def _(gr_base):
    geom_bottom = gr_base.copy()
    return (geom_bottom,)


@app.cell
def _(bond, gr_base):
    geom_top_AA = gr_base.translate((0, 0, 3.55))
    geom_top_AB = gr_base.translate((0, bond, 3.35))
    return geom_top_AA, geom_top_AB


@app.cell
def _(geom_bottom, geom_top_AA):
    geom_AA = geom_top_AA.add(geom_bottom)
    geom_AA.plot(axes=["y", [1,0,1]], backend="matplotlib")
    return (geom_AA,)


@app.cell
def _(geom_bottom, geom_top_AB):
    geom_AB = geom_top_AB.add(geom_bottom)
    geom_AB.plot(axes=["y", [1,0,1]], backend="matplotlib")
    return (geom_AB,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Band path visualized
    """)
    return


@app.cell
def _(bond):
    band_path_AA = HEX(bond, c=3.55).bandpath(path="GMKG")
    band_path_AA.plot()
    return (band_path_AA,)


@app.cell
def _(bond):
    band_path_AB = HEX(bond, c=3.35).bandpath(path="GMKG")
    return (band_path_AB,)


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Creating Hamiltonain
    """)
    return


@app.cell
def _():
    mu = 0.5
    return (mu,)


@app.cell
def _(Vpppi, Vpps, geom_AA, mu):
    Ham_AA = tbbi_opt(geom_AA, -mu, mu, Vpppi=Vpppi, Vpps=Vpps, dangling=0.0)
    return (Ham_AA,)


@app.cell
def _(Vpppi, Vpps, geom_AB, mu):
    Ham_AB = tbbi_opt(geom_AB, -mu, mu, Vpppi=Vpppi, Vpps=Vpps, dangling=0.0)
    return (Ham_AB,)


@app.cell
def _(Ham_AA):
    Ham_AA.lattice
    return


@app.cell
def _(Ham_AA, band_path_AA, divs):
    _dict = {key:val for key, val in band_path_AA.special_points.items() if key in band_path_AA.path}
    band_AA = sisl.BandStructure(
        Ham_AA,
        points=[_dict["G"], _dict["M"], _dict["K"]],
        divisions=divs,
        names=[r"$\Gamma$", "M", "K"]
    )
    band_AA.plot(backend="matplotlib")
    return


@app.cell
def _(Ham_AB, band_path_AB, divs):
    _dict = {key:val for key, val in band_path_AB.special_points.items() if key in band_path_AB.path}
    band_AB = sisl.BandStructure(
        Ham_AB,
        points=[_dict["G"], _dict["M"], _dict["K"]],
        divisions=divs,
        names=[r"$\Gamma$", "M", "K"]
    )
    band_AB.plot(backend="matplotlib")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
