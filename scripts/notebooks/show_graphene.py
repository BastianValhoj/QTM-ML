import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")

with app.setup:
    import sisl
    import numpy as np
    import matplotlib.pyplot as plt

    from scipy.signal import find_peaks

    from ase.visualize import view

    from pathlib import Path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Directories
    """)
    return


@app.cell
def _():
    NOTEBOOK_DIR = Path(__file__).parent
    FIG_DIR = NOTEBOOK_DIR.parent / "figures"
    FIG_DIR.exists()
    return (FIG_DIR,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Show cutting graphene to armchair or zigzag edges
    """)
    return


@app.cell
def _():
    gr = sisl.geom.graphene(orthogonal=True).tile(10,0).tile(10,1)
    return (gr,)


@app.cell
def _(gr):
    view(gr.to.ase())
    return


@app.cell
def _(gr):
    xy = gr.xyz[:, :2]
    arm_edge_idx = np.arange(284,298)
    zig_edge_idx = np.array([284,245,246,247,248,209,210,211,212,173,174])
    all_idx = np.arange(len(xy))
    non_edge_idx = np.delete(all_idx, np.concat([arm_edge_idx,zig_edge_idx]))
    return arm_edge_idx, xy, zig_edge_idx


@app.cell
def _(xy):
    X, Y = np.array([
        (xy[324] + xy[285])/2.,
        (xy[243] + xy[284])/2.,
    ]).T

    U, V = np.array([
        (xy[297] + xy[336])/2.,
        (xy[173] + xy[172])/2.,
    ]).T

    print(U[1], V[1])
    return U, V, X, Y


@app.cell
def _(gr):
    out = gr.plot(axes="xy", backend="matplotlib")

    # for _att in dir(out.axes):
    #     if not _att.startswith("_"):
    #         print(_att)
    print([_att for _att in dir(out.axes) if not _att.startswith("_")][:9])
    print([_att for _att in dir(out.axes) if not _att.startswith("_")][9:18])
    print([_att for _att in dir(out.axes) if not _att.startswith("_")][18:27])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot: graphene and cuts
    """)
    return


@app.cell
def _(FIG_DIR, U, V, X, Y, arm_edge_idx, xy, zig_edge_idx):
    from mytools.plots import thesis_fig
    fig, ax = thesis_fig()

    ax.scatter(*xy.T, color="grey", alpha=0.6)
    ax.scatter(*xy[arm_edge_idx].T, color="red", alpha=0.4)
    ax.scatter(*xy[zig_edge_idx].T, color="blue", alpha=0.4)
    ax.quiver(X, Y, U-X, V-Y, scale=1, units="xy", angles="xy")

    # ax.annotate

    ax.axis("off")
    fig.savefig(FIG_DIR / f"{Path(__file__).stem}")

    fig
    return (thesis_fig,)


@app.cell
def _(mo):
    mo.md(r"""
    # Show band structure and DOS
    """)
    return


@app.cell
def _():
    bond = 1.42
    Vpppi = -2.7

    divisions = 100

    Emax = 6
    Emin = -Emax
    energies = np.linspace(Emin, Emax, 100)
    Erange = (Emin, Emax)
    return Emax, Emin, Erange, Vpppi, bond, divisions, energies


@app.cell
def _(bond):
    geom = sisl.geom.graphene(bond)
    return (geom,)


@app.cell
def _(Vpppi, bond, geom):
    Ham = sisl.Hamiltonian(geom)
    _R = (0.1, bond+1e-2)
    _T = (0.0, Vpppi)
    Ham.construct([_R, _T])
    return (Ham,)


@app.cell
def _(FIG_DIR, bond):
    from ase.lattice import HEX2D
    path = ["G", "M", "K"]
    hexbandpath = HEX2D(a=bond).bandpath(path=path)
    special_points = hexbandpath.special_points

    kpoints = []
    knames = []
    for _lab in path:
    # for _lab in ["M", "G", "K", "M"]:
        kpoints.append(special_points[_lab])
        knames.append(_lab)

    _axes = hexbandpath.plot()

    _fig = _axes.get_figure()
    _fig.savefig(FIG_DIR / "show_graphene_brillouin_zone")
    _fig
    return knames, kpoints, special_points


@app.cell
def _(special_points):
    print(special_points)
    return


@app.cell
def _(Ham, divisions, knames, kpoints):
    bs = sisl.BandStructure(Ham, points=kpoints, divisions=divisions, names=knames)
    lineark, kticks, klabels = bs.lineark(True)
    return bs, klabels, kticks, lineark


@app.cell
def _(Ham, divisions, energies):
    bz = sisl.MonkhorstPack(Ham, nkpt=[divisions, divisions, 1])
    bz_avg = bz.apply.average
    DOS = bz_avg.eigenstate(wrap=lambda es:es.DOS(energies)).squeeze() / Ham.no
    print(DOS.shape)
    return (DOS,)


@app.cell
def _(bs):
    eigs = bs.apply.array.eigh()
    return (eigs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot: band and DOS
    """)
    return


@app.cell
def _(bond):
    def plot_graphene_unitcell(ax_inset, geom):
        """
        Draw the 2-atom graphene unit cell with bonds and lattice vectors.
        geom must have nsc >= [3,3,1] so periodic NN bonds are visible.
        """
        a1, a2 = geom.cell[0, :2], geom.cell[1, :2]
        A = geom.xyz[0, :2]
        B = geom.xyz[1, :2]

        # NN bonds (including periodic images)
        geom.set_nsc([3, 3, 1])
        for ia in range(geom.na):
            idx, xyz_n = geom.close(ia, R=[0.1, bond + 0.01], ret_xyz=True)
            for Rj in xyz_n[1]:
                ri, rj = geom.xyz[ia, :2], Rj[:2]
                ax_inset.plot([ri[0], rj[0]], [ri[1], rj[1]], 'k-', lw=1., zorder=1)

        # Atoms
        ax_inset.scatter(*A, s=80, color='blue', zorder=3)
        ax_inset.scatter(*B, s=80, color='red', zorder=3)
        ax_inset.annotate('A', A, xytext=(-10, 2), textcoords='offset points', ha="center", va="center",
                          color='k')
        ax_inset.annotate('B', B, xytext=(10, 2), textcoords='offset points', ha="center", va="center",
                          color='k')

        # Lattice vectors
        for vec, label in [(a1, r'$\mathbf{a}_1$'), (a2, r'$\mathbf{a}_2$')]:
            ax_inset.annotate('', xy=A + vec, xytext=A, zorder=0,
                              arrowprops=dict(arrowstyle='->', color='grey', lw=1.5))
            if vec is a1:
                va = "top"
            elif vec is a2:
                va = "bottom"
            ax_inset.text(*(A + vec * 0.55), label,
                          color='k', ha='right', va=va)

        ax_inset.set_aspect('equal')
        ax_inset.axis('off')

    return (plot_graphene_unitcell,)


@app.cell
def _(
    DOS,
    Emax,
    Emin,
    Erange,
    FIG_DIR,
    eigs,
    energies,
    geom,
    klabels,
    kticks,
    lineark,
    plot_graphene_unitcell,
    thesis_fig,
):
    _fig, _ax = thesis_fig(1,2, sharey=True, aspect=0.5)

    _ax[0].plot(lineark, eigs, color="k")
    _ymin, _ymax = _ax[0].get_ylim()
    _ax[0].set(
        xticks=kticks,
        xticklabels=klabels,
        xlim=(lineark[0], lineark[-1]),
        ylim=Erange,
        yticks=np.arange(Emin, Emax+0.1, 3),

        ylabel=r"$E-E_F$   (eV)",
        xlabel=r"$\mathbf{k}$   ($\mathrm{\AA}^{-1}$)",
        title="Band Structure"
    )
    _ax[0].grid()


    _ax[1].plot(DOS, energies, color="k")
    _ax[1].set(
        xlabel=r"DOS   ($\mathrm{eV}^{-1}$)",
        title=r"DOS (normalized)"
    )
    _ax[1].grid()

    for _peak in find_peaks(DOS)[0]:
        # print(_peak)
        _ax[0].axhline(energies[_peak], linestyle=":", color="k")
        _ax[1].axhline(energies[_peak], linestyle=":", color="k")



    _ax_inset = _ax[0].inset_axes([0.02, 0.28, 0.38, 0.42])

    plot_graphene_unitcell(_ax_inset, geom)


    _fig.set_constrained_layout(True)
    _fig.suptitle("Graphene")

    _fig.savefig(FIG_DIR / "show_graphene_band_DOS")
    _fig
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
