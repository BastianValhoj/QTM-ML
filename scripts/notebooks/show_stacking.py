import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")

with app.setup:
    import sisl
    import matplotlib.pyplot as plt
    import numpy as np

    from mytools.construct import all_armchair
    from mytools.plots import thesis_fig, label_subplots


    from pathlib import Path


@app.cell
def _():
    NOTEBOOK = Path(__file__)
    NOTEBOOK_DIR = NOTEBOOK.parent
    SCRIPTS_DIR = NOTEBOOK_DIR.parent
    FIG_DIR = SCRIPTS_DIR / "figures"
    FIG_DIR.exists()
    return FIG_DIR, NOTEBOOK


@app.cell
def _():
    bond = 1.42
    return (bond,)


@app.cell
def _(FIG_DIR, NOTEBOOK, bond):
    fig, ax = thesis_fig(1,1, subplot_kw=dict(projection="3d"), fraction=1)

    # AB

    flake1 = all_armchair(bond)
    flake2 = all_armchair(bond).translate((0, bond, 2))
    xyz1 = flake1.xyz
    xyz2 = flake2.xyz



    ax.scatter(*xyz1[[0, 3, 4]].T, color="k", facecolor="k", zorder=3, depthshade=False )
    ax.scatter(*xyz1[[1,2,5]].T, color="k", facecolor="white", zorder=3, depthshade=False )

    ax.scatter(*xyz2[[0, 3, 4]].T, color="k", facecolor="k", zorder=3, depthshade=False )
    ax.scatter(*xyz2[[1, 2, 5]].T, color="k", facecolor="white", zorder=3, depthshade=False )

    bonds1 = np.vstack((xyz1[[0,2,4,5,3,1]], xyz1[0]))
    bonds2 = np.vstack((xyz2[[0,2,4,5,3,1]], xyz2[0]))
    ax.plot(*bonds1.T, color="k", lw=0.5)
    ax.plot(*bonds2.T, color="k", lw=0.5)

    ax.plot(*np.column_stack((xyz1[3], xyz2[5])), color="k", linestyle=":", zorder=0, lw=1)
    ax.plot(*np.column_stack((xyz1[0], xyz2[2])), color="k", linestyle=":", zorder=0, lw=1)
    ax.plot(*np.column_stack((xyz1[1], xyz2.mean(axis=0))), color="k", linestyle=":", zorder=0, lw=1)

    ax.text(*np.column_stack((xyz1[1], xyz2.mean(axis=0))).mean(axis=1), s=f"3.35 Å", ha="left", va="top")
    ax.text(*xyz1[3], s="A", va="top", ha="left")
    ax.text(*xyz1[1], s="B", va="top", ha="left")



    #AA
    offset = (0, 5, 0)

    flake1 = all_armchair(bond).translate(offset)
    flake2 = all_armchair(bond).translate(offset).translate((0, 0, 2.5))
    xyz1 = flake1.xyz
    xyz2 = flake2.xyz

    ax.plot(*np.column_stack((xyz1[0], xyz2[0])), color="k", linestyle=":", zorder=0, lw=1)
    # ax.plot(*np.column_stack((xyz1[1], xyz2[1])), color="k", linestyle=":", zorder=0, lw=1)
    ax.plot(*np.column_stack((xyz1[3], xyz2[3])), color="k", linestyle=":", zorder=0, lw=1)

    ax.scatter(*xyz1[[0, 3, 4]].T, color="k", facecolor="k", zorder=3, depthshade=False )
    ax.scatter(*xyz1[[1, 2, 5]].T, color="k", facecolor="white", zorder=3, depthshade=False )
    ax.scatter(*xyz2[[0, 3, 4]].T, color="k", facecolor="k", zorder=3, depthshade=False )
    ax.scatter(*xyz2[[1, 2, 5]].T, color="k", facecolor="white", zorder=3, depthshade=False )


    bonds1 = np.vstack((xyz1[[0,2,4,5,3,1]], xyz1[0]))
    bonds2 = np.vstack((xyz2[[0,2,4,5,3,1]], xyz2[0]))
    ax.plot(*bonds1.T, color="k", lw=0.5)
    ax.plot(*bonds2.T, color="k", lw=0.5)


    # print("{x}, {y}, {z}".format(x=np.mean([xyz1[1,0], xyz2[1,0]]), y=np.mean([xyz1[1,1], xyz2[1,1]]), z=3.55/2))
    # print(np.column_stack((xyz1[0], xyz2[0])).mean(axis=1))
    ax.text(*np.column_stack((xyz1[1], xyz2[1])).mean(axis=1), s=f"3.55 Å", ha="left", va="center")

    ax.text(*xyz1[3], s="A", va="top", ha="left")
    ax.text(*xyz1[1], s="B", va="top", ha="left")

    # ax.axis("equal")

    xlim = (0, 5)
    ylim = (-1.48, 7)
    zlim = (-0.3, 2.8)

    ax.set_box_aspect((xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0])) 

    ax.set(
        xlim=xlim,
        ylim=ylim,
        zlim=zlim,
        # xticks=[],
        # yticks=[],
        # zticks=[]
    )

    ax.axis("off")
    ax.view_init(elev=15, azim=15)
    fig.set_constrained_layout(True)
    # fig.tight_layout()
    # ax.margins(0)
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)


    _path = FIG_DIR / f"{NOTEBOOK.stem}"
    fig.savefig(_path, bbox_inches="tight", pad_inches=0.01)

    # from PIL import Image
    # im = Image.open(_path)
    # im.getbbox()  # gives the bounding box of non-white content
    # im.crop(im.getbbox()).save(_path)

    fig
    return


if __name__ == "__main__":
    app.run()
