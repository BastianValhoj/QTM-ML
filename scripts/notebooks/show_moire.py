import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import sisl

    import matplotlib.pyplot as plt


    from mytools.plots import thesis_fig, label_subplots
    from pathlib import Path

    return Path, label_subplots, np, sisl, thesis_fig


@app.cell
def _(Path):
    NOTEBOOK_DIR = Path(__file__).parent
    FIG_DIR = NOTEBOOK_DIR.parent / "figures"
    FIG_DIR.exists()
    return (FIG_DIR,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Edge plot parameters
    """)
    return


@app.cell
def _(sisl):
    gr = sisl.geom.graphene(orthogonal=True).tile(10,0).tile(10,1)
    return (gr,)


@app.cell
def _(gr, np):
    xy = gr.xyz[:, :2]
    arm_edge_idx = np.arange(284,298)
    zig_edge_idx = np.array([284,245,246,247,248,209,210,211,212,173,174])
    all_idx = np.arange(len(xy))
    non_edge_idx = np.delete(all_idx, np.concat([arm_edge_idx,zig_edge_idx]))
    return arm_edge_idx, xy, zig_edge_idx


@app.cell
def _(np, xy):
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Moire plot parameters
    """)
    return


@app.cell
def _(sisl):
    flake1 = sisl.geom.graphene_flake(15)
    flake1.plot(axes="xy", backend="matplotlib")
    return (flake1,)


@app.cell
def _(flake1):
    flake2 = flake1.rotate([10, [0,0,1]], origin=flake1.center(), what="xyz").translate((0,0,4))
    flake2.plot(axes="xy", backend="matplotlib")
    return (flake2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Combined plot
    """)
    return


@app.cell
def _(
    FIG_DIR,
    Path,
    U,
    V,
    X,
    Y,
    arm_edge_idx,
    flake1,
    flake2,
    label_subplots,
    thesis_fig,
    xy,
    zig_edge_idx,
):
    fig, ax = thesis_fig(1,2)
    xy1 = flake1.xyz[:, :2]
    xy2 = flake2.xyz[:, :2]

    ax[0].scatter(*xy1.T, s=0.5, color="k")
    ax[0].scatter(*xy2.T, s=0.5, color="k")


    ax[0].set_aspect("equal")
    # ax.set(
    #     xlabel=r"$x\ \ (\mathrm{\AA})$",
    #     ylabel=r""
    #     )

    ax[1].scatter(*xy.T, color="grey", alpha=0.6)
    ax[1].scatter(*xy[arm_edge_idx].T, color="red", alpha=0.4)
    ax[1].scatter(*xy[zig_edge_idx].T, color="blue", alpha=0.4)
    ax[1].quiver(X, Y, U-X, V-Y, scale=1, units="xy", angles="xy")

    # ax.annotate

    ax[0].axis("off")
    ax[1].axis("off")

    label_subplots(ax, pos=(-0.02, 0.95))

    fig.set_constrained_layout(True)
    fig.savefig(FIG_DIR / f"{Path(__file__).stem}_graphene_edges_cut")
    fig
    return xy1, xy2


@app.cell
def _(thesis_fig, xy1, xy2):
    _fig, _ax = thesis_fig()

    _ax.scatter(*xy1.T, s=2, color="k")
    _ax.scatter(*xy2.T, s=2, color="k")



    _ax.axis("equal")
    _ax.axis("off")
    _fig
    return


@app.cell
def _(sisl, thesis_fig):
    _fig, _ax = thesis_fig(1,1, figsize=(12,8), sharey=True)

    gr1 = sisl.geom.graphene().tile(8,0).tile(8,1)
    gr1 = gr1.translate(-gr1.center())
    gr2 = sisl.geom.graphene().tile(20,0).tile(20,1)
    gr2 = gr2.translate(-gr2.center()).translate((0, -50, 0))

    _ax.scatter(*gr1.xyz[:,:2].T, color="k", s=10)
    _ax.scatter(*gr2.xyz[:,:2].T, color="k", s=10)


    # for _a in _ax:
    #     # _a.axis("off")
    #     _a.axis("scaled")

    _ax.axis("off")
    _ax.axis("equal")

    _fig
    return


if __name__ == "__main__":
    app.run()
