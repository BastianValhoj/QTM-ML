import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")

with app.setup:
    import sisl
    import numpy as np
    import matplotlib.pyplot as plt

    from mytools.construct import all_armchair
    from mytools.construct import make_edge

    from mytools.scalingv2 import get_edges, get_corners
    from mytools.scalingv2 import get_centers
    from mytools.scalingv2 import get_fractional
    from mytools.scalingv2 import map_edges

    from scipy.spatial import cKDTree


@app.cell
def _():
    BOND = 1.42
    N_small = 5
    N_big = 11

    ETA = 1e-3
    NK1 = 400

    NC = 0

    R = (0.1, BOND+1e-2)
    T = (0.0, -2.7)
    return BOND, NC, N_big, N_small, R, T


@app.cell
def _(BOND, R, T):
    graphene6 = all_armchair(BOND)
    Ham0 = sisl.Hamiltonian(graphene6)
    Ham0.construct([R,T])
    return (Ham0,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Show the edge geometry
    """)
    return


@app.cell
def _(Ham0, N_small):
    geom_edge_small = make_edge(Ham0.geometry, N_small, N_small)
    geom_edge_small.plot(axes="xy", backend="matplotlib")
    return (geom_edge_small,)


@app.cell
def _(Ham0, N_big):
    geom_edge_big = make_edge(Ham0.geometry, N_big, N_big)
    geom_edge_big.plot(axes="xy", backend="matplotlib")
    return (geom_edge_big,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Geometry highligted with corners and edges
    """)
    return


@app.cell
def _(Ham0, NC, N_small, geom_edge_small):
    edges_small = get_edges(N_small, N_small, Ham0.na, NC=NC)
    corners_small = get_corners(N_small, N_small, na=Ham0.na, NC=NC)
    print(edges_small)
    print(corners_small)
    _astyle = [dict(atoms=edges_small, color="blue"),
                dict(atoms=corners_small, color="red")
    ]
    geom_edge_small.plot(axes="xy", backend="matplotlib", 
    atoms_style=_astyle, show_cell=False, show_bonds=True)
    return (edges_small,)


@app.cell
def _(Ham0, NC, N_big, geom_edge_big):
    edges_big = get_edges(N_big, N_big, Ham0.na, NC=NC)
    corners_big = get_corners(N_big, N_big, na=Ham0.na, NC=NC)
    print(edges_big)
    print(corners_big)
    _astyle = [dict(atoms=edges_big, color="blue"),
                dict(atoms=corners_big, color="red")
    ]
    geom_edge_big.plot(axes="xy", backend="matplotlib", 
    atoms_style=_astyle, show_cell=False, show_bonds=True)
    return (edges_big,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Centroids of edges w/ edges
    """)
    return


@app.cell
def _(Ham0, edges_small, geom_edge_small):
    edge_xyz_small = geom_edge_small.xyz[edges_small]
    centroids_small = get_centers(edge_xyz_small, na=Ham0.na)
    _fig, _ax = plt.subplots()
    _ax.scatter(*centroids_small[:, :2].T, marker="x")
    _ax.scatter(*edge_xyz_small[:, :2].T)
    return (centroids_small,)


@app.cell
def _(Ham0, edges_big, geom_edge_big):
    edge_xyz_big = geom_edge_big.xyz[edges_big]
    centroids_big = get_centers(edge_xyz_big, na=Ham0.na)
    _fig, _ax = plt.subplots()
    _ax.scatter(*centroids_big[:, :2].T, marker="x")
    _ax.scatter(*edge_xyz_big[:, :2].T)
    return (centroids_big,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fractional of centroids
    """)
    return


@app.cell
def _():
    import enum
    def show_centroids(coords, direction="in", ax=None, *, tol=1e-1, shift=0.05, labels=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.scatter(*coords[:, :2].T)
        if labels is None:
            labels = np.arange(len(coords))
        elif not (labels is None) and (len(labels) != len(coords)):
            raise ValueError("If given labels, must be of same length as number of data points")
        else: 
            pass
        for idx, (x,y) in zip(labels, coords[:, :2]):
            if (x < tol): dx = +shift
            elif (x > 1-tol): dx = -shift
            else: dx = 0

            if (y < tol): dy = +shift
            elif (y > 1-tol): dy = -shift
            else: dy = 0

            if direction == "in":
                pass
            elif direction == "out":
                dx = -dx
                dy = -dy
            ax.annotate(text=str(idx), xy=(x,y), xytext=(x+dx, y+dy), ha="center", va="center")



    return (show_centroids,)


@app.function
def new_frac(cell, coords):
    coords = coords[:, :2]
    MAT = cell[:2, :2]
    Inv = np.linalg.inv(MAT)
    frac = coords @ Inv
    return frac / np.max(frac, axis=0, keepdims=True)


@app.cell
def _(centroids_small, geom_edge_small, show_centroids):
    edge_frac_small = get_fractional(geom_edge_small, centroids_small)
    edge_frac_small = new_frac(geom_edge_small.cell, coords=centroids_small)
    print(edge_frac_small)
    _fig, _ax = plt.subplots()

    show_centroids(edge_frac_small, "in", ax=_ax)

    _ymin, _ymax = _ax.get_ylim()
    _xmin, _xmax = _ax.get_xlim()
    _ax.set(ylim=(min(_ymin, 0), max(_ymax, 0)),
    xlim=(min(_xmin, 0), max(_xmax, 0)))
    _ax.grid()
    _fig
    return (edge_frac_small,)


@app.cell
def _(centroids_big, geom_edge_big, show_centroids):
    edge_frac_big = get_fractional(geom_edge_big, centroids_big)
    # print(edge_frac_big)
    _fig, _ax = plt.subplots()
    show_centroids(edge_frac_big, "out", _ax)
    _ymin, _ymax = _ax.get_ylim()
    _xmin, _xmax = _ax.get_xlim()
    _ax.set(ylim=(min(_ymin, 0)-0.1, max(_ymax, 0)+0.1),
    xlim=(min(_xmin, 0)-0.1, max(_xmax, 0)+0.1))
    _ax.grid()
    _fig
    return (edge_frac_big,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Improve mapping of centroids
    """)
    return


@app.cell
def map_centroids():
    # def map_centroids(geom_small, geom_big, n_small, n_big, na=6, NC=0):
    #     xyz_small = geom_small.xyz
    #     xyz_big = geom_big.xyz

    #     edges_small = get_edges(n_small, n_small, na=na, NC=NC)
    #     edges_big  = get_edges(n_big, n_big, na=na, NC=NC)
    
    #     centroids_small = get_centers(xyz_small[edges_small], na=na)
    #     centroids_big = get_centers(xyz_big[edges_big], na=na)

    #     centroids_frac_small = get_fractional(geom_small, centroids_small)
    #     centroids_frac_big = get_fractional(geom_big, centroids_big)

    #     atoms_per_side_small = len(edges_small) // 4 // na
    #     atoms_per_side_big = len(edges_big) // 4 // na

    #     edge_idx_small_to_big = np.zeros(shape=4*atoms_per_side_big, dtype=int)

    #     for side in range(4):
    #         print(f"##side: {side}")

    
    #         small = centroids_frac_small[side*atoms_per_side_small : (side+1)*atoms_per_side_small]
    #         big = centroids_frac_big[side*atoms_per_side_big : (side+1)*atoms_per_side_big]
    #         # print(small)
    #         tree = cKDTree(small)
    #         _, idx = tree.query(big, k=1)
    #         print(idx)
    #         print(idx+side*atoms_per_side_small)
    #         print(side*atoms_per_side_big)
    #         # edge_idx_small_to_big[side*atoms_per_side_big : (side+1)*atoms_per_side_big] = idx + side*atoms_per_side_small
    # map_centroids(geom_edge_small, geom_edge_big, N_small, N_big, Ham0.na, NC=NC)




    return


@app.cell
def _(
    Ham0,
    NC,
    N_big,
    N_small,
    edge_frac_big,
    edge_frac_small,
    geom_edge_big,
    geom_edge_small,
    show_centroids,
):
    # print(edge_frac_big)
    from mytools.plots import thesis_fig
    _fig, _ax = thesis_fig()
    _, small_to_big_idx = map_edges(geom_edge_small, geom_edge_big, N_small, N_big, Ham0.na, NC=NC)
    show_centroids(edge_frac_small, "in", _ax)
    show_centroids(edge_frac_big, "out", _ax, labels=small_to_big_idx)

    # show_centroids(edge_frac_big, "out", _ax)
    _ymin, _ymax = _ax.get_ylim()
    _xmin, _xmax = _ax.get_xlim()
    _ax.set(ylim=(min(_ymin, 0)-0.1, max(_ymax, 0)+0.1),
    xlim=(min(_xmin, 0)-0.1, max(_xmax, 0)+0.1))
    _ax.grid()
    _fig.suptitle("kNN matching centroids")
    _fig
    return


if __name__ == "__main__":
    app.run()
