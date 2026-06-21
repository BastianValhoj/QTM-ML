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
    divs = 250


    dAA = 3.55
    # dAA = 3.35
    dAB = 3.35

    path = ["G", "M", "K", "G"]

    Emax = 7
    Emin = -Emax
    Erange = (Emin, Emax)

    energies = np.linspace(Emin, Emax, 100)
    # Rs = (bond*0.1, bond+1e-2)
    # Ts = (0.0, Vpppi)
    return Erange, Vpppi, Vpps, bond, dAA, dAB, divs, energies, path


@app.cell
def _(bond):
    gr_base = sisl.geom.graphene(bond=bond, vacuum=15)
    return (gr_base,)


@app.cell
def _(gr_base):
    geom_bottom = gr_base.copy()
    return (geom_bottom,)


@app.cell
def _(bond, dAA, dAB, gr_base):
    geom_top_AA = gr_base.translate((0, 0, dAA))
    geom_top_AB = gr_base.translate((bond, 0, dAB))
    return geom_top_AA, geom_top_AB


@app.cell
def _(geom_AA, gr_base):
    print(gr_base.cell[2])       # should be [0, 0, 20]
    print(geom_AA.cell[2])       # same — top atoms at z=3.55 inside a 20Å cell
    return


@app.cell
def _(dAA, geom_bottom, geom_top_AA):
    geom_AA = geom_top_AA.add(geom_bottom)
    geom_AA = geom_AA.translate((0,0,geom_AA.cell[2,2]/2-dAA/2))
    print(geom_AA.xyz)
    geom_AA.plot(axes=["x", "y"], backend="matplotlib")
    return (geom_AA,)


@app.cell
def _(dAB, geom_bottom, geom_top_AB):
    geom_AB = geom_top_AB.add(geom_bottom)
    geom_AB = geom_AB.translate((0,0,geom_AB.cell[2,2]/2 - dAB/2))
    geom_AB.plot(axes=["x", "y"], backend="matplotlib")
    return (geom_AB,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Band path visualized
    """)
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
def _(Vpppi, Vpps, bond, dAA, geom_AA, mu):
    Ham_AA = tbbi_opt(
        geom_AA, 
        -mu,
        mu, 
        Vpppi=Vpppi, 
        Vpps=Vpps,
        d0_00=bond, 
        d0_01=dAA, 
        dangling=0.0)
    return (Ham_AA,)


@app.cell
def _(Vpppi, Vpps, bond, dAB, geom_AB, mu):
    Ham_AB = tbbi_opt(
        geom_AB, 
        -mu, 
        mu, 
        Vpppi=Vpppi, 
        Vpps=Vpps,
        d0_00=bond, 
        d0_01=dAB,
        dangling=0.0)
    return (Ham_AB,)


@app.cell
def _(path):
    special_points = {"G":[0.,0.,0.], "K":[2/3,1/3,0], "M":[1/2, 1/2, 0]}

    kpoints = []
    knames = []
    for _lab in path:
        knames.append(_lab)
        kpoints.append(special_points[_lab])

    for _kpt, _n in zip(kpoints,knames):
        print(_n, _kpt)
    return knames, kpoints


@app.cell
def _(Ham_AA, Ham_AB, divs, knames, kpoints):
    bands_AA = sisl.BandStructure(Ham_AA, points=kpoints, divisions=divs, names=knames)
    bands_AB = sisl.BandStructure(Ham_AB, points=kpoints, divisions=divs, names=knames)
    lineark, kticks, klabels = bands_AA.lineark(True)
    return klabels, kticks, lineark


@app.function
def get_idx(geom):
    center = geom.center()
    idx_top = np.where(geom.xyz[:, 2] > center[2])[0]
    idx_bottom = np.where(geom.xyz[:, 2] < center[2])[0]
    return idx_top, idx_bottom


@app.cell
def _(geom_AA):
    AA_top, AA_bottom = get_idx(geom_AA)
    AA_top, AA_bottom
    return


@app.cell
def _(geom_AB):
    AB_top, AB_bottom = get_idx(geom_AB)
    return


@app.cell
def _(divs, energies):
    def get_pdos(ham):
        bz = sisl.MonkhorstPack(ham, nkpt=[divs, divs, 1])
        bz_avg = bz.apply.average
        pdos = bz_avg.eigenstate(wrap=lambda es: es.PDOS(energies)).squeeze() / ham.na
        return pdos

    return (get_pdos,)


@app.cell
def _(Ham_AA, Ham_AB, get_pdos):
    PDOS_AA = get_pdos(Ham_AA)
    PDOS_AB = get_pdos(Ham_AB)
    print(PDOS_AA.shape)
    print(PDOS_AB.shape)
    return


@app.cell
def _(band_AA, band_AB):
    eigs_AA = band_AA.apply.array.eigh()
    eigs_AB = band_AB.apply.array.eigh()
    return eigs_AA, eigs_AB


@app.cell
def _(Ham_AA, divs, knames, kpoints):
    band_AA = sisl.BandStructure(
        Ham_AA,
        points=kpoints,
        divisions=divs,
        names=knames,
    )
    band_AA.plot(backend="matplotlib", Erange=[-10,10])
    return (band_AA,)


@app.cell
def _(Ham_AB, divs, knames, kpoints):
    band_AB = sisl.BandStructure(
        Ham_AB,
        points=kpoints,
        divisions=divs,
        names=knames,
    )
    band_AB.plot(backend="matplotlib", Erange=[-10,10])
    return (band_AB,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fatbands
    """)
    return


@app.cell
def _(Erange, eigs_AA, eigs_AB, klabels, kticks, lineark):
    _fig, _ax = thesis_fig()
    scale = 0.
    # for atom in range(Ham_AA.na):
    #     lower = eigs[:, atom] - (top_w[:, atom]*scale)
    #     upper = eigs[:, atom] + (bot_w[:, atom]*scale)
    #     _ax.fill_between(lineark, lower, upper, alpha=0.4, edgecolor="r", color="r")

    # _eigs = eigs
    _ax.plot(lineark, eigs_AA[:,0], color='red', alpha=0.9, label="AA", 
        linestyle="-", marker="o", markerfacecolor='None', markevery=12)
    _ax.plot(lineark, eigs_AA[:,1:], color='red', alpha=0.9, 
        linestyle="-", marker="o", markerfacecolor='None', markevery=12)


    _ax.plot(lineark, eigs_AB[:,0], color='k', linestyle="-", alpha=0.9, lw=2, label="AB")
    _ax.plot(lineark, eigs_AB[:,1:], color='k', linestyle="-", alpha=0.9, lw=2)


    _ax.set(
        xticks=kticks, 
        xticklabels=klabels,

        xlabel="$\mathbf{k}$  ($\mathrm{\AA}^{-1}$)",
        ylabel=r"$E - E_F$  (eV)",

        xlim=(np.min(lineark), np.max(lineark)),
        ylim=Erange,
    )
    _ax.legend()
    _ax.grid()
    _fig
    return


if __name__ == "__main__":
    app.run()
