import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")

with app.setup:
    import sisl
    import matplotlib.pyplot as plt
    import numpy as np

    from mytools.scalingv2 import get_centers, get_corners, get_edges, get_fractional
    from mytools.scalingv2 import rsse_to_edge, rsse_mapping, map_edges, map_corners, extrapolate
    from mytools.construct import make_edge, all_armchair
    from mytools.plots import thesis_fig, label_subplots

    from scipy.spatial import cKDTree
    from matplotlib.patches import FancyArrow
    from matplotlib.lines import Line2D

    from pathlib import Path


@app.cell
def _():
    Nb = 5
    NT = 11
    NC = 0
    return NC, NT, Nb


@app.cell
def _():
    SCRIPT_DIR = Path(__file__).parent
    FIG_DIR = SCRIPT_DIR.parent / "figures"
    print(FIG_DIR, FIG_DIR.exists())
    return (FIG_DIR,)


@app.cell
def _():
    bond = 1.42
    return (bond,)


@app.cell
def _(bond):
    gr = all_armchair(bond)
    return (gr,)


@app.cell
def _(NT, gr):
    geom_big = make_edge(gr, NT, NT)
    geom_big = geom_big.translate(-geom_big.center())
    geom_big.plot(axes="xy", backend="matplotlib")
    return (geom_big,)


@app.cell
def _(Nb, gr):
    geom_small = make_edge(gr, Nb, Nb)
    geom_small = geom_small.translate(-geom_small.center())
    geom_small.plot(axes="xy", backend="matplotlib")
    return (geom_small,)


@app.cell
def _(NC, NT, Nb, gr):
    corners_small = get_corners(Nb, Nb, gr.na, NC=NC)
    corners_big = get_corners(NT, NT, gr.na, NC=NC)
    return corners_big, corners_small


@app.cell
def _(NC, NT, Nb, gr):
    geom_edge_small = get_edges(Nb, Nb, gr.na, NC=NC)
    geom_edge_big = get_edges(NT, NT, gr.na, NC=NC)
    return geom_edge_big, geom_edge_small


@app.cell
def _(geom_big, geom_edge_big, geom_edge_small, geom_small, gr):
    centroids_edge_small = get_centers(geom_small.sub(geom_edge_small), na=gr.na)[:, :2]
    centroids_edge_big = get_centers(geom_big.sub(geom_edge_big), na=gr.na)[:, :2]
    return centroids_edge_big, centroids_edge_small


@app.cell
def _(corners_big, corners_small, geom_big, geom_small, gr):
    centroids_corner_small = get_centers(geom_small.sub(corners_small), gr.na)[:, :2]
    centroids_corner_big = get_centers(geom_big.sub(corners_big), gr.na)[:, :2]
    return centroids_corner_big, centroids_corner_small


@app.cell
def _(centroids_edge_big, centroids_edge_small, geom_big, geom_small):
    edge_frac_small = get_fractional(geom_small, centroids_edge_small)
    edge_frac_big = get_fractional(geom_big, centroids_edge_big)
    tree = cKDTree(edge_frac_small)
    dd, ii = tree.query(edge_frac_big, k=1)
    return edge_frac_big, edge_frac_small, ii


@app.cell
def _():
    # print(centroids_edge_small, "\n\n", centroids_edge_big)
    return


@app.cell
def _(
    centroids_corner_big,
    centroids_corner_small,
    centroids_edge_big,
    centroids_edge_small,
    ii,
):
    _fig, _axes = thesis_fig()
    _axes.scatter(*centroids_edge_big.T, marker="x", color="lightsteelblue", s=0)
    _axes.scatter(*centroids_edge_small.T, marker="s", color="darkorange", s=0)

    for _idx, (_x,_y) in enumerate(centroids_edge_small[:, :2]):
        _axes.annotate(str(_idx), xy=(_x,_y), va="center", ha="center", bbox=dict(color="darkorange", pad=0.2))

    for (_x,_y) in centroids_corner_small:
        _axes.annotate("C", xy=(_x,_y), va="center", ha="center", bbox=dict(color="grey", pad=0.2, alpha=.4))

    for _idx, (_x,_y) in enumerate(centroids_edge_big[:, :2]):
        _axes.annotate(str(ii[_idx]), xy=(_x,_y), va="center", ha="center", bbox=dict(color="lightsteelblue", pad=0.2))

    for (_x,_y) in centroids_corner_big:
        # print(x,y)
        _axes.annotate("C", xy=(_x,_y), va="center", ha="center", bbox=dict(color="grey", pad=0.2, alpha=0.4))

    X, Y = np.array([
        centroids_edge_big[0], 
        centroids_edge_small[0]
        ]).T
    U, V = np.array([
        centroids_edge_big[9], 
        centroids_edge_small[3]
        ]).T
    _axes.set(xlim=(-42, 42), ylim=(-24, 24), xlabel=r"$x$ ($\mathrm{\AA}$)", ylabel=r" $y$ ($\mathrm{\AA}$)")
    _axes.quiver(X, Y,  U - X, V - Y, scale_units="xy", scale=1, zorder=4)


    # arrow_proxy = FancyArrow(0, 0, 1, 0, width=0.3, color="black", length_includes_head=True)
    _arrow_proxy = Line2D([0], [0], marker=r'$\rightarrow$', color='black', 
                         markersize=15, linestyle='None', label='Coupling arrow')
    _axes.legend(handles=[_arrow_proxy], labels=["Coupling arrow"])

    _fig.set_constrained_layout(True)
    _fig.suptitle("Electrode edge centroids")
    _fig
    return U, V, X, Y


@app.cell
def _(
    FIG_DIR,
    NT,
    Nb,
    U,
    V,
    X,
    Y,
    centroids_corner_big,
    centroids_corner_small,
    centroids_edge_big,
    centroids_edge_small,
    edge_frac_big,
    edge_frac_small,
    ii,
):
    _fig, _axes = thesis_fig(subplots=(1,2), aspect=0.5)



    _axes[0].scatter(*edge_frac_small[:, :2].T, color="darkorange", label="Base centroids", marker="s", s=60)
    _axes[0].scatter(*edge_frac_big[:, :2].T, color="lightsteelblue", label="Target centroids", marker="x")

    _offset = 12
    _tol = 1e-2
    for _idx, (_x, _y) in enumerate(edge_frac_small[:, :2]):
        if (_x < _tol): _dx = +_offset
        elif (_x > 1 - _tol): _dx = -_offset
        else: _dx = 0


        if (_y < _tol): _dy = +_offset
        elif (_y > 1 - _tol): _dy = -_offset
        else: _dy = 0

        _axes[0].annotate(str(_idx), xy=(_x, _y), xycoords="data", xytext=(_dx, _dy), textcoords="offset points", ha="center", va="center",
            bbox=dict(facecolor="darkorange", pad=0.2, edgecolor="none", alpha=0.5))


    for _idx, (_x, _y) in enumerate(edge_frac_big[:, :2]):
        if (_x < _tol): _dx = -_offset
        elif (_x > 1 - _tol): _dx = +_offset
        else: _dx = 0


        if (_y < _tol): _dy = -_offset
        elif (_y > 1 - _tol): _dy = +_offset
        else: _dy = 0

        _axes[0].annotate(str(ii[_idx]), xy=(_x, _y), xycoords="data", xytext=(_dx, _dy), textcoords="offset points", ha="center", va="center",
            bbox=dict(facecolor="lightsteelblue", pad=0.2, edgecolor="none", alpha=0.5))

    _limit_tol = 1e-1
    _xmin, _xmax = _axes[0].get_xlim()
    _ymin, _ymax = _axes[0].get_ylim()
    _axes[0].set(
        xlim=(_xmin-_limit_tol, _xmax+_limit_tol),
        ylim=(_ymin-_limit_tol, _ymax+_limit_tol),
        xlabel="$x$ (fractional)",
        ylabel="$y$ (fractional)"
    )
    


    _axes[1].scatter(*centroids_edge_big.T, marker="x", color="lightsteelblue", s=0)
    _axes[1].scatter(*centroids_edge_small.T, marker="s", color="darkorange", s=0)

    for _idx, (_x,_y) in enumerate(centroids_edge_small[:, :2]):
        _axes[1].annotate(str(_idx), xy=(_x,_y), va="center", ha="center", bbox=dict(color="darkorange", pad=0.2))

    for (_x,_y) in centroids_corner_small:
        _axes[1].annotate("C", xy=(_x,_y), va="center", ha="center", bbox=dict(color="grey", pad=0.2, alpha=.4))

    for _idx, (_x,_y) in enumerate(centroids_edge_big[:, :2]):
        _axes[1].annotate(str(ii[_idx]), xy=(_x,_y), va="center", ha="center", bbox=dict(color="lightsteelblue", pad=0.2))

    for (_x,_y) in centroids_corner_big:
        # print(x,y)
        _axes[1].annotate("C", xy=(_x,_y), va="center", ha="center", bbox=dict(color="grey", pad=0.2, alpha=0.4))

    _axes[1].set(xlim=(-42, 42), ylim=(-24, 24), xlabel=r"$x$ ($\mathrm{\AA}$)", ylabel=r" $y$ ($\mathrm{\AA}$)")
    _axes[1].quiver(X, Y,  U - X, V - Y, scale_units="xy", scale=1, zorder=4)


    # arrow_proxy = FancyArrow(0, 0, 1, 0, width=0.3, color="black", length_includes_head=True)
    _arrow_proxy = Line2D([0], [0], marker=r'$\rightarrow$', color='black', 
                         markersize=15, linestyle='None', label='Coupling arrow')
    _axes[1].legend(handles=[_arrow_proxy], labels=["Coupling arrow"])

    _fig.set_constrained_layout(True)
    _fig.suptitle("Electrode edge centroids")
    label_subplots(_axes, pos=(0.1, 0.98))
    _axes[0].legend()
    _fig.savefig(FIG_DIR / f"show_coupling_centriods_kNN_and_coupling_{Nb}_to_{NT}")
    _fig
    return


@app.cell
def _():
    return


@app.cell
def _(U, V, X, Y):
    vec_length = np.array([U-X, V-Y])
    print(vec_length)
    np.linalg.norm(vec_length, axis=0)
    return


if __name__ == "__main__":
    app.run()
