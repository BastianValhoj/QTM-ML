import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")

with app.setup:
    import sisl
    import numpy as np
    import matplotlib.pyplot as plt

    from scipy.spatial import cKDTree

    from mytools.plots import thesis_fig
    from mytools.construct import all_armchair, make_edge
    from mytools.scalingv2 import get_fractional, get_centers, get_corners, get_edges, map_edges

    from pathlib import Path


@app.cell
def _():
    SCRIPT_DIR = Path(__file__).parent
    FIG_DIR = SCRIPT_DIR.parent / "figures"
    print(FIG_DIR)
    return (FIG_DIR,)


@app.cell
def _():
    BOND = 1.42
    phi = np.pi/3
    B = BOND*np.cos(phi/2)

    # N0 = 3
    N_small = 5
    N_big = 10

    R = (0.1, BOND+1e-2)
    T = (0.0, -2.7)

    ETA = 1e-3
    NUMK = 400

    NC = 0
    return BOND, ETA, NC, NUMK, N_big, N_small, R, T


@app.cell
def _(BOND, R, T):
    graphene6 = all_armchair(BOND)

    Ham0 = sisl.Hamiltonian(graphene6)
    Ham0.construct([R, T])
    return (Ham0,)


@app.cell
def _(ETA, NUMK):
    def setup_rsse(Ham, N):
        rsse = sisl.RealSpaceSE(Ham, 0, 1, (N, N, 1))
        rsse.setup(eta=ETA, 
            bz=sisl.MonkhorstPack(Ham, [1,NUMK, 1]))
        return rsse

    return (setup_rsse,)


@app.cell
def _(setup_rsse):
    def edge_for_plot(Ham, N, NC):
        uc_geom = Ham.geometry
        rsse = setup_rsse(Ham, N)
        geom = make_edge(uc_geom, N, N)
        edges = get_edges(N, N, na=Ham.na, NC=NC)
        corners = get_corners(N, N, na=Ham.na, NC=NC)
        return geom, edges, corners

    return (edge_for_plot,)


@app.function
def resub_ham(rsse):
    Ham_rs, elec_idx = rsse.real_space_coupling(ret_indices=True)
    Ham_NN = rsse.real_space_parent()
    all_idx = np.arange(Ham_NN.na)
    device_idx = np.delete(all_idx, elec_idx)
    sub_idx = np.concat([elec_idx, device_idx])
    Ham_NN_re = Ham_NN.sub(sub_idx)

    out = {
        "H_elec": Ham_rs,
        "elec_idx": elec_idx,
        "sub_idx": sub_idx,

    }
    return Ham_NN_re, out


@app.cell
def _():
    # def outward_offset(xyz_centered, scale=0.5):
    #     """Unit outward normal from center for each atom, scaled."""
    #     norms = np.linalg.norm(xyz_centered[:, :2], axis=1, keepdims=True)
    #     unit_outward = xyz_centered[:, :2] / norms          # (N, 2)
    #     return unit_outward * scale
    return


@app.cell
def _(Ham0, NC, N_big, N_small, edge_for_plot):
    geom_edge_small, edges_small, corners_small = edge_for_plot(Ham0, N_small, NC)
    geom_edge_big, edges_big, corners_big = edge_for_plot(Ham0, N_big, NC)
    return edges_big, edges_small, geom_edge_big, geom_edge_small


@app.cell
def _():
    # geom_edge_big.plot(axes="xy", backend="matplotlib")
    return


@app.cell
def _(Ham0, NC, N_big, N_small, geom_edge_big, geom_edge_small):
    _, edge_idx1_to_idx2 = map_edges(geom_edge_small, geom_edge_big, N_small,N_big, na=Ham0.na, NC=NC)

    edge_idx1_to_idx2
    return


@app.cell
def _(FIG_DIR, Ham0, edges_big, edges_small, geom_edge_big, geom_edge_small):
    from matplotlib.transforms import Bbox
    centroids_edge_small = get_centers(geom_edge_small.xyz[edges_small], Ham0.na)
    centroids_edge_big = get_centers(geom_edge_big.xyz[edges_big], Ham0.na)

    centroids_frac_small = get_fractional(geom_edge_small, centroids_edge_small)
    centroids_frac_big = get_fractional(geom_edge_big, centroids_edge_big)
    tree = cKDTree(centroids_frac_small)
    dd, ii = tree.query(centroids_frac_big, k=1)



    fig, ax = thesis_fig()

    ax.scatter(*centroids_frac_small[:, :2].T, color="darkorange")
    ax.scatter(*centroids_frac_big[:, :2].T, color="lightsteelblue")

    ax.set(
        xlabel="x (fractional)",
        ylabel="y (fractional)"
        # xticklabels=np.arange(0, 1.2, 0.2).round(3),
        # yticklabels=np.arange(0, 1.2, 0.2).round(3),
        )
    shift = 5e-2
    tol: float = 0.1
    for idx, (x,y) in enumerate(centroids_frac_small):
        if (x < tol): dx = +shift
        elif (x > 1 - tol): dx = -shift
        else: dx = 0
    
        if (y < tol): dy = +shift
        elif (y > 1-tol): dy = -shift
        else: dy = 0
        ax.annotate(text=str(idx), xy=(x,y), xytext=(x+dx,y+dy), ha="center", va="center",
        bbox=dict(facecolor="darkorange", pad=0.2, edgecolor="none", alpha=0.5))

    for idx, (x,y) in enumerate(centroids_frac_big):
        if (x < tol): dx = -shift
        elif (x > 1-tol): dx = +shift
        else: dx = 0

        if (y < tol): dy = -shift
        elif (x > 1-tol): dy = +shift
        else: dy = 0

        ax.annotate(text=str(ii[idx]), xy=(x,y), xytext=(x+dx,y+dy), ha="center", va="center",
        bbox=dict(facecolor="lightsteelblue", pad=0.2, edgecolor="none", alpha=0.5))



    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    ax.set(ylim=(min(ymin-shift, -shift), max(ymax+shift, shift)), xlim=(min(xmin-shift, -shift), max(ymax+shift, shift)) )
    fig.suptitle("Centroid-based equivalence")
    fig.savefig(FIG_DIR / "show_edges_centroids")
    fig
    return


if __name__ == "__main__":
    app.run()
