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


@app.cell
def _():
    Nb = 5
    NT = 11

    NC = 0
    return NC, NT, Nb


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
    geom_big.plot(axes="xy")
    return (geom_big,)


@app.cell
def _(Nb, gr):
    geom_small = make_edge(gr, Nb, Nb)
    geom_small = geom_small.translate(-geom_small.center())
    geom_small.plot(axes="xy")
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
    return (ii,)


@app.cell
def _(centroids_edge_big, centroids_edge_small):
    print(centroids_edge_small, "\n\n", centroids_edge_big)
    return


@app.cell
def _(
    centroids_corner_big,
    centroids_corner_small,
    centroids_edge_big,
    centroids_edge_small,
    ii,
):
    fig, axes = thesis_fig()
    axes.scatter(*centroids_edge_big.T, marker="x", color="lightsteelblue", s=0)
    axes.scatter(*centroids_edge_small.T, marker="s", color="darkorange", s=0)

    for idx, (x,y) in enumerate(centroids_edge_small[:, :2]):
        axes.annotate(str(idx), xy=(x,y), va="center", ha="center", bbox=dict(color="darkorange", pad=0.2))

    for (x,y) in centroids_corner_small:
        axes.annotate("C", xy=(x,y), va="center", ha="center", bbox=dict(color="grey", pad=0.2, alpha=.4))

    for idx, (x,y) in enumerate(centroids_edge_big[:, :2]):
        axes.annotate(str(ii[idx]), xy=(x,y), va="center", ha="center", bbox=dict(color="lightsteelblue", pad=0.2))

    for (x,y) in centroids_corner_big:
        print(x,y)
        axes.annotate("C", xy=(x,y), va="center", ha="center", bbox=dict(color="grey", pad=0.2, alpha=0.4))

    X, Y = np.array([
        centroids_edge_big[0], 
        centroids_edge_small[0]
        ]).T
    U, V = np.array([
        centroids_edge_big[9] - 0.8, 
        centroids_edge_small[3]
        ]).T
    axes.set(xlim=(-42, 42), ylim=(-24, 24), xlabel=r"$x$ ($\mathrm{\AA}$)", ylabel=r" $y$ ($\mathrm{\AA}$)")
    axes.quiver(X, Y,  U - X, V - Y, scale_units="xy", scale=1)


    # arrow_proxy = FancyArrow(0, 0, 1, 0, width=0.3, color="black", length_includes_head=True)
    arrow_proxy = Line2D([0], [0], marker=r'$\rightarrow$', color='black', 
                         markersize=15, linestyle='None', label='Coupling arrow')
    axes.legend(handles=[arrow_proxy], labels=["Coupling arrow"])
    fig
    return


if __name__ == "__main__":
    app.run()
