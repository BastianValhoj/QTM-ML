import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")

with app.setup:
    import sisl
    import numpy as np
    import matplotlib.pyplot as plt

    from mytools.construct import all_armchair
    from mytools.plots import thesis_fig, label_subplots

    from pathlib import Path
    from matplotlib.patches import Annulus


@app.cell
def _():
    NOTEBOOK_DIR = Path(__file__).parent
    FIG_DIR = NOTEBOOK_DIR.parent / "figures"
    FIG_DIR.exists()
    return (FIG_DIR,)


@app.cell
def _():
    bond = 1.42
    N = 13
    return N, bond


@app.cell
def _(bond):
    gr = all_armchair(bond)
    ham0 = sisl.Hamiltonian(gr)
    r = (0.1, bond+1e-2, 2*bond, 3*bond)
    t = (0, -2.7, 0.48, 0.33)
    ham0.construct([r,t])
    return gr, ham0


@app.cell
def _(N, ham0):
    rsse = sisl.RealSpaceSE(ham0, 0, 1, (N, N, 1))
    # rsse.setup(eta=1e-3, bz=sisl.MonkhorstPack(ham0, [1, 1200, 1])) 
    return (rsse,)


@app.cell
def _(rsse):
    _, elec_idx = rsse.real_space_coupling(True)
    geom = rsse.real_space_parent().geometry
    geom.plot(axes="xy", backend="matplotlib", atoms_style=dict(atoms=elec_idx, color="red"), show_cell=False)
    return elec_idx, geom


@app.cell
def _(geom):
    xyz = geom.xyz
    return (xyz,)


@app.cell
def _(geom, gr):
    L = np.linalg.norm(geom.cell[0]) - 2*np.linalg.norm(gr.cell[0])
    radius = np.sin(np.pi/3)*L / 2
    cirlce = sisl.shape.EllipticalCylinder(v=radius, h=50., center=geom.center())
    down_fold_idx = cirlce.within(geom.xyz).nonzero()[0]
    down_fold_idx
    return down_fold_idx, radius


@app.cell
def _(down_fold_idx, elec_idx, geom, xyz):
    atom_idx = np.arange(geom.na)
    elec_xyz = xyz[elec_idx]
    non_elec_idx = np.delete(atom_idx, elec_idx)
    non_elec_or_device_idx = np.delete(atom_idx, np.concat([down_fold_idx, elec_idx]))
    return (non_elec_or_device_idx,)


@app.cell
def _(geom, radius):
    theta= np.linspace(0, 2*np.pi, 100)
    circ = radius* np.column_stack([np.cos(theta), np.sin(theta)])
    inner_circ = 1/2*circ
    circ += geom.center()[:2]
    inner_circ += geom.center()[:2]
    return circ, inner_circ


@app.cell
def _(geom, radius):
    X, Y = geom.center()[:2]
    U, V = np.array([radius, 0])
    return U, V, X, Y


@app.cell
def _(
    FIG_DIR,
    U,
    V,
    X,
    Y,
    circ,
    down_fold_idx,
    elec_idx,
    inner_circ,
    non_elec_or_device_idx,
    xyz,
):
    _fig, _ax = thesis_fig(fraction=0.8)
    _ax.scatter(*xyz[elec_idx,:2].T, color="red", label="Electrode region", s=10)
    _ax.scatter(*xyz[non_elec_or_device_idx, :2].T, color="grey", s=10)
    _ax.scatter(*xyz[down_fold_idx, :2].T, color="lightskyblue", label="Down-fold region", s=10)
    _ax.legend()
    # ax.quiver(X, Y, U, V, scale_units="xy", scale=1, headwidth=6)
    _ax.annotate(r"$R$", xy=((X + V), (Y + U)), xytext=(0, 0), textcoords="offset points", fontsize=15, va="top", ha="center",
        # bbox=dict(facecolor="grey", alpha=0.3)
    )
    _ax.annotate(r"$R/2$", xy=((X + V/2), (Y + U/2)), xytext=(0, 0), textcoords="offset points", fontsize=15, va="top", ha="center",
        # bbox=dict(facecolor="grey", alpha=0.3)
    )
    _ax.set_aspect("equal")
    _ax.set(xlabel=r"$x$ ($\mathrm{\AA}$)",
    ylabel=r"$y$ ($\mathrm{\AA}$)")
    _ax.plot(*circ.T, color="k", linestyle="--")
    _ax.plot(*inner_circ.T, color="k", linestyle="--")
    # _fig.suptitle("Down-folding target structure")
    _fig.savefig(FIG_DIR / "show_down-fold_with_R")
    _ax.axis("off")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Show decoupling region
    """)
    return


@app.cell
def _():
    W = 4
    return (W,)


@app.cell
def _(FIG_DIR, W, down_fold_idx, geom, radius, xyz):

    _fig, _ax = thesis_fig(fraction=0.8)


    _ax.scatter(*xyz[down_fold_idx,:2].T, color="lightskyblue", s=15, label="Device atoms")
    _annulus = Annulus(xy=geom.center()[:2], r=radius, width=W, alpha=0.6, color="grey", label="Decoupling region")
    _ax.add_patch(_annulus)
    _xstart = geom.center()[0] - radius
    _ax.plot([_xstart, _xstart+W], [0,0], lw=3, color="k", label="$W$")
    _ax.legend(loc="center")
    _fig.set_constrained_layout(True)
    _ax.set_aspect("equal")
    _ax.axis("off")
    _fig.savefig(FIG_DIR / "show_down-fold_decoupling_region")
    _fig
    return


@app.cell
def _(
    FIG_DIR,
    U,
    V,
    W,
    X,
    Y,
    circ,
    down_fold_idx,
    elec_idx,
    geom,
    inner_circ,
    non_elec_or_device_idx,
    radius,
    xyz,
):
    fig, ax = thesis_fig(1, 2)
    ax[0].scatter(*xyz[elec_idx,:2].T, color="red", label="Electrode region", s=10)
    ax[0].scatter(*xyz[non_elec_or_device_idx, :2].T, color="grey", s=10)
    ax[0].scatter(*xyz[down_fold_idx, :2].T, color="lightskyblue", label="Down-fold region", s=10)
    # ax[0].legend()

    # ax.quiver(X, Y, U, V, scale_units="xy", scale=1, headwidth=6)
    ax[0].annotate(r"$R$", xy=((X + V), (Y + U)), xytext=(0, 0), textcoords="offset points", fontsize=15, va="top", ha="center",
        # bbox=dict(facecolor="grey", alpha=0.3)
    )
    ax[0].annotate(r"$R/2$", xy=((X + V/2), (Y + U/2)), xytext=(0, 0), textcoords="offset points", fontsize=15, va="top", ha="center",
        # bbox=dict(facecolor="grey", alpha=0.3)
    )
    ax[0].set_aspect("equal")
    ax[0].set(xlabel=r"$x$ ($\mathrm{\AA}$)",
        ylabel=r"$y$ ($\mathrm{\AA}$)")
    ax[0].plot(*circ.T, color="k", linestyle="--")
    ax[0].plot(*inner_circ.T, color="k", linestyle="--")


    ax[0].axis("off")

    ax[1].scatter(*xyz[down_fold_idx,:2].T, color="lightskyblue", s=15)
    _annulus = Annulus(xy=geom.center()[:2], r=radius, width=W, alpha=0.6, color="grey", label="Decoupling region")
    ax[1].add_patch(_annulus)
    _xstart = geom.center()[0] - radius
    ax[1].plot([_xstart, _xstart+W], [0,0], lw=3, color="k", label="$W$")


    ax[1].set_aspect("equal")
    ax[1].axis("off")
    _handles0, _labels0 = ax[0].get_legend_handles_labels()
    _handles1, _labels1 = ax[1].get_legend_handles_labels()

    # fig.legend(np.concat([_handles0, _handles1]), np.concat([_labels0, _labels1]), loc=(0.45, 0.0))
    ax[0].legend(loc="lower left")
    ax[1].legend(loc="lower right")
    label_subplots(ax)
    fig.set_constrained_layout(True)
    fig.savefig(FIG_DIR / "show_down-fold_combined")
    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
