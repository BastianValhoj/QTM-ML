import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")

with app.setup:
    import sisl
    import numpy as np
    import matplotlib.pyplot as plt

    from ase.visualize import view

    from pathlib import Path


@app.cell
def _():
    NOTEBOOK_DIR = Path(__file__).parent
    FIG_DIR = NOTEBOOK_DIR.parent / "figures"
    FIG_DIR.exists()
    return (FIG_DIR,)


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


@app.cell
def _(FIG_DIR, U, V, X, Y, arm_edge_idx, xy, zig_edge_idx):
    from mytools.plots import thesis_fig
    fig, ax = thesis_fig()

    ax.scatter(*xy.T, color="grey", alpha=0.6)
    ax.scatter(*xy[arm_edge_idx].T, color="red", alpha=0.4)
    ax.scatter(*xy[zig_edge_idx].T, color="blue", alpha=0.4)
    ax.quiver(X, Y, U-X, V-Y, scale=1, units="xy", angles="xy")

    ax.annotate

    ax.axis("off")
    fig.savefig(FIG_DIR / f"{Path(__file__).stem}")

    fig

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
