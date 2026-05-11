import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")

with app.setup:
    import sisl
    import matplotlib.pyplot as plt
    import numpy as np

    from mytools.scalingv2 import get_centers, get_corners, get_edges, get_fractional
    from mytools.scalingv2 import rsse_to_edge, rsse_mapping, map_edges, map_corners, extrapolate
    from mytools.construct import make_edge, all_armchair
    from mytools.plots import thesis_fig

    from typing import cast


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
    BOND = 1.42
    phi = np.pi/3
    B = BOND*np.cos(phi/2)

    # N0 = 3
    N_small = 5
    N_big = 11

    R = (0.1, BOND+1e-2)
    T = (0.0, -2.7)

    ETA = 1e-3
    NUMK = 400
    return BOND, ETA, NUMK, N_big, N_small, R, T


@app.cell
def _(BOND, R, T):
    graphene6 = all_armchair(BOND)

    Ham0 = sisl.Hamiltonian(graphene6)
    Ham0.construct([R, T])
    return (Ham0,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Make the edges and RSSE
    """)
    return


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


@app.cell
def _():
    NClist = [0, 1]
    return


@app.cell
def _(Ham0, N_big, N_small, edge_for_plot):
    fig, axes = thesis_fig(subplots=(2,1))
    # fig = cast(plt.Figure, fig)
    # axes = cast(list[plt.Axes, plt.Axes], axes)
    for idx, ax in enumerate(axes):
        ax = cast(plt.Axes, ax)
        geom_edge_small, edges_small, corners_small = edge_for_plot(Ham0, N_small, NC=idx)
        geom_edge_big, edges_big, corners_big = edge_for_plot(Ham0, N_big, NC=idx)
        geom_edge_small = geom_edge_small.translate(-geom_edge_small.center())
        geom_edge_big = geom_edge_big.translate(-geom_edge_big.center())
        ax.scatter(*geom_edge_big[edges_big, :2].T, color="darkorange")
        ax.scatter(*geom_edge_small[edges_small, :2].T, color="darkorange")

        ax.scatter(*geom_edge_big[corners_big, :2].T, color="grey")
        ax.scatter(*geom_edge_small[corners_small, :2].T, color="grey")
        ax.set(title=f"NC={idx}", xticklabels="", yticklabels="")
        ax.set_axis_off()
    fig.savefig(f"../figures/show_corner_fixed_{N_small}_to_{N_big}")
    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
