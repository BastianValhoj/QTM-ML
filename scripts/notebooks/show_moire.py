import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import sisl

    import matplotlib.pyplot as plt


    from mytools.plots import thesis_fig
    from pathlib import Path

    return Path, sisl, thesis_fig


@app.cell
def _(Path):
    NOTEBOOK_DIR = Path(__file__).parent
    FIG_DIR = NOTEBOOK_DIR.parent / "figures"
    FIG_DIR.exists()
    return (FIG_DIR,)


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


@app.cell
def _(FIG_DIR, Path, flake1, flake2, thesis_fig):
    fig, ax = thesis_fig(1,1)
    xy1 = flake1.xyz[:, :2]
    xy2 = flake2.xyz[:, :2]

    ax.scatter(*xy1.T, s=2, color="k")
    ax.scatter(*xy2.T, s=2, color="k")


    ax.set_aspect("equal")
    ax.set(
        xlabel=r"$x\ \ (\mathrm{\AA})$",
        ylabel=r""
        )

    ax.axis("off")
    fig.set_constrained_layout(True)
    fig.savefig(FIG_DIR / str(Path(__file__).stem))
    fig
    return


if __name__ == "__main__":
    app.run()
