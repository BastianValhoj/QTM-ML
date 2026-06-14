import marimo

__generated_with = "0.23.9"
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

    Rs = (bond*0.1, bond+1e-2)
    Ts = (0.0, Vpppi)
    return Vpppi, Vpps, bond


@app.cell
def _():
    divs = 250
    return (divs,)


@app.cell
def _(bond):
    gr_base = sisl.geom.graphene(bond=bond)
    # gr_base = all_armchair(bond)
    # gr_base = gr_base.add_vacuum(10, 2)
    return (gr_base,)


@app.cell
def _(gr_base):
    geom_bottom = gr_base.copy()
    return (geom_bottom,)


@app.cell
def _(bond, gr_base):
    geom_top_AA = gr_base.translate((0, 0, 3.55))
    geom_top_AB = gr_base.translate((bond, 0, 3.35))
    return geom_top_AA, geom_top_AB


@app.cell
def _(geom_bottom, geom_top_AA):
    geom_AA = geom_top_AA.add(geom_bottom)
    print(geom_AA.xyz)
    geom_AA.plot(axes=["x", "y"], backend="matplotlib")

    return (geom_AA,)


@app.cell
def _(geom_bottom, geom_top_AB):
    geom_AB = geom_top_AB.add(geom_bottom)
    geom_AB.plot(axes=["x", "y"], backend="matplotlib")
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
    band_path_AB = HEX(bond, c=3.35).bandpath(path="MGKM")
    band_path_AB.plot()
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
        points=[_dict["M"], _dict["G"], _dict["K"], _dict["M"]],
        divisions=divs,
        names=["M", r"$\Gamma$", "K", "M"]
    )
    band_AA.plot(backend="matplotlib", Erange=[-10,10])
    return


@app.cell
def _():
    # BZ_AA = sisl.BrillouinZone(Ham_AA)


    # eigs = np.array([BZ_AA.apply.array.eig(k) for k in band_AA.k])
    # BZ_AA.apply.array.
    return


@app.cell
def _(Ham_AB, band_path_AB, divs):
    _dict = {key:val for key, val in band_path_AB.special_points.items() if key in band_path_AB.path}
    band_AB = sisl.BandStructure(
        Ham_AB,
        points=[_dict["M"], _dict["G"], _dict["K"], _dict["M"],],
        divisions=divs,
        names=["M", r"$\Gamma$", "K", "M",]
    )
    band_AB.plot(backend="matplotlib", Erange=[-10,10])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fatbands
    """)
    return


@app.cell(hide_code=True)
def _(bond, divs):
    hex = HEX2D(a=bond)
    hexpath = hex.bandpath("MGKM", npoints=divs)
    print(hexpath.special_points)
    hexpath.plot()
    return (hexpath,)


@app.cell
def _():
    return


@app.cell
def _(hexpath):
    hex_points = hexpath.special_points
    special_points = np.array([
        hex_points["M"],
        hex_points["G"],
        hex_points["K"],
        hex_points["M"],
    ])

    names = np.array([
        r"$\mathrm{M}$",
        r"$\Gamma$",
        r"$\mathrm{K}$",
        r"$\mathrm{M}$",

    ])

    return names, special_points


@app.cell
def _(Ham_AA, Ham_AB, divs, hexpath, names, special_points):
    bands_AA = sisl.BandStructure(Ham_AA, points=special_points, divisions=divs, names=names)
    bands_AB = sisl.BandStructure(Ham_AB, points=special_points, divisions=divs, names=names)
    lineark, kticks, klabels = hexpath.get_linear_kpoint_axis()
    # kpts_AA = bands_AA.k
    energies = np.linspace(-4,4,100)
    return bands_AA, bands_AB, klabels, kticks, lineark


app._unparsable_cell(
    r"""
    geom_center, np.max(geom_AA.xyz[])
    """,
    name="_"
)


@app.cell
def _(geom_AA):
    geom_center = geom_AA.center()
    idx_top = np.where(geom_AA.xyz[:, 2] > geom_center[2])[0]
    idx_bottom = np.where(geom_AA.xyz[:, 2] < geom_center[2])[0]
    idx_top, idx_bottom
    return idx_bottom, idx_top


@app.cell
def _(Ham_AA, Ham_AB, bands_AA, bands_AB, idx_bottom, idx_top):
    Hks_AA = np.array([Ham_AA.Hk(k=k, format="array") for k in bands_AA.k])
    eigs_AA, states_AA = np.linalg.eigh(Hks_AA)
    porps_AA = np.abs(states_AA)**2

    Hks_AB = np.array([Ham_AB.Hk(k=k, format="array") for k in bands_AB.k])
    eigs_AB, states_AB = np.linalg.eigh(Hks_AB)
    porps_AB = np.abs(states_AB)**2

    top_w = np.sum(porps_AA[:, idx_top, :], axis=1)
    bot_w = np.sum(porps_AA[:, idx_bottom, :], axis=1)
    return Hks_AA, bot_w, eigs_AA, eigs_AB, top_w


@app.cell
def _(Hks_AA):
    print(Hks_AA.shape)
    return


@app.cell
def _(bot_w, eigs_AA, top_w):
    print(eigs_AA[:].shape, bot_w.shape, top_w.shape)
    return


@app.cell
def _(eigs_AA, eigs_AB, klabels, kticks, lineark):
    _fig, _ax = thesis_fig()
    scale = 0.
    # for atom in range(Ham_AA.na):
    #     lower = eigs[:, atom] - (top_w[:, atom]*scale)
    #     upper = eigs[:, atom] + (bot_w[:, atom]*scale)
    #     _ax.fill_between(lineark, lower, upper, alpha=0.4, edgecolor="r", color="r")

    # _eigs = eigs

    _ax.plot(lineark, eigs_AA, color='red', linestyle=":", alpha=0.4, label="AA")
    _ax.plot(lineark, eigs_AB, color='blue', linestyle="--", alpha=0.4, label="AB")
    _ax.set(
        xticks=kticks, 
        xticklabels=klabels,
    
        xlabel="$\mathbf{k}$",
        ylabel=r"$E$",

        xlim=(np.min(lineark), np.max(lineark)),
        ylim=(-4,4),
    )
    _ax.grid()
    _fig
    return


if __name__ == "__main__":
    app.run()
