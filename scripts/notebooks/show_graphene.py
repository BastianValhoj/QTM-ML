import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")

with app.setup:
    import sisl
    import numpy as np
    import matplotlib.pyplot as plt

    from scipy.signal import find_peaks

    from ase.visualize import view
    from ase.lattice import HEX2D
    from pathlib import Path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Directories
    """)
    return


@app.cell
def _():
    NOTEBOOK_DIR = Path(__file__).parent
    FIG_DIR = NOTEBOOK_DIR.parent / "figures"
    FIG_DIR.exists()
    return (FIG_DIR,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Show cutting graphene to armchair or zigzag edges
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define structure for cutting
    """)
    return


@app.cell
def _():
    gr = sisl.geom.graphene(orthogonal=True).tile(10,0).tile(10,1)
    return (gr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Determine edge atoms for each cut
    """)
    return


@app.cell
def _():
    # view(gr.to.ase())
    return


@app.cell
def _(gr):
    xy = gr.xyz[:, :2]
    arm_edge_idx = np.arange(284,298)
    zig_edge_idx = np.array([284,245,246,247,248,209,210,211,212,173,174])
    all_idx = np.arange(len(xy))
    non_edge_idx = np.delete(all_idx, np.concat([arm_edge_idx,zig_edge_idx]))
    return arm_edge_idx, xy, zig_edge_idx


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Vector showing the edges
    """)
    return


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
    _out = gr.plot(axes="xy", backend="matplotlib")

    # for _att in dir(out.axes):
    #     if not _att.startswith("_"):
    #         print(_att)
    print([_att for _att in dir(_out.axes) if not _att.startswith("_")][:9])
    print([_att for _att in dir(_out.axes) if not _att.startswith("_")][9:18])
    print([_att for _att in dir(_out.axes) if not _att.startswith("_")][18:27])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot: graphene and cuts
    """)
    return


@app.cell
def _(FIG_DIR, U, V, X, Y, arm_edge_idx, xy, zig_edge_idx):
    from mytools.plots import thesis_fig
    fig, ax = thesis_fig()

    ax.scatter(*xy.T, color="grey", alpha=0.6)
    ax.scatter(*xy[arm_edge_idx].T, color="red", alpha=0.4)
    ax.scatter(*xy[zig_edge_idx].T, color="blue", alpha=0.4)
    ax.quiver(X, Y, U-X, V-Y, scale=1, units="xy", angles="xy")

    # ax.annotate

    ax.axis("off")
    fig.savefig(FIG_DIR / f"{Path(__file__).stem}")

    fig
    return (thesis_fig,)


@app.cell
def _(mo):
    mo.md(r"""
    # Show band structure and DOS
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Some parameters to use
    """)
    return


@app.cell
def _():
    bond = 1.42
    Vpppi = -2.7

    divisions = 300

    Emax = 6
    Emin = -Emax
    energies = np.linspace(Emin, Emax, 100)
    Erange = (Emin, Emax)
    return Emax, Emin, Erange, Vpppi, bond, divisions, energies


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Geometry for band and DOS
    """)
    return


@app.cell
def _(geom):
    geom.icell*np.pi*2, geom.rcell
    return


@app.cell
def _(bond):
    # geom = sisl.geom.graphene(bond).tile(3,0).tile(3,1)
    geom = sisl.geom.graphene(bond)
    print(geom)
    return (geom,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Make Hamiltonian
    """)
    return


@app.cell
def _(Vpppi, bond, geom):
    from mytools.tbbi import tbbi_opt
    Ham = sisl.Hamiltonian(geom)
    _R = (0.1, bond+1e-2)
    _T = (0.0, Vpppi)
    Ham.construct([_R, _T])


    # Ham = tbbi_opt(
    #     geom,
    #     0,
    #     0,
    #     dangling=0.
    # )
    return (Ham,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define band path
    """)
    return


@app.cell
def _():
    path = [
        "G", 
        "M", 
        "K", 
        "G", 
    ]
    return (path,)


@app.cell
def _():
    special_points_ase={"G":[0.,0.,0.], "M":[1/2,0,0],"K":[1/3, 1/3, 0]}
    print(special_points_ase)
    return (special_points_ase,)


@app.cell
def _():
    special_points_sisl={"G":[0.,0.,0.], "K":[2/3,1/3,0], "M":[1/2, 1/2, 0]}
    print(special_points_sisl)
    return (special_points_sisl,)


@app.cell
def _(path, special_points_ase, special_points_sisl):
    kpoints_sisl = []
    kpoints_ase = []
    knames = []
    for _lab in path:
    # for _lab in ["M", "G", "K", "M"]:
        kpoints_ase.append(special_points_ase[_lab])
        kpoints_sisl.append(special_points_sisl[_lab])
        knames.append(_lab)

    print(knames)
    return knames, kpoints_sisl


@app.cell
def _(Ham, divisions, knames, kpoints_sisl):
    bs = sisl.BandStructure(Ham, points=kpoints_sisl, divisions=divisions, names=knames)
    lineark, kticks, klabels = bs.lineark(True)
    return bs, klabels, kticks, lineark


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Show Brillouin Zone -- WRONG

    The lattice vectors from `HEX2D` are different from those of the BZ for the geometry!!!!!
    """)
    return


@app.cell
def _():
    # _a_lat = np.linalg.norm(geom.cell[0])  # = sqrt(3)*bond ≈ 2.46 Å

    # _hexbandpath = HEX2D(a=_a_lat).bandpath(path=path, special_points=special_points_ase)
    # _hexbandpath.cartesian_kpts()
    # bz_ax = _hexbandpath.plot()

    # bz_fig = bz_ax.get_figure()
    # bz_fig.savefig(FIG_DIR / "show_graphene_brillouin_zone")
    # bz_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## DOS from k-averaged BZ - normalized by no. orbitals
    (no==na for 1 NN TB)
    """)
    return


@app.cell
def _(Ham, energies):
    bz = sisl.MonkhorstPack(Ham, nkpt=[90, 90, 1])
    bz_avg = bz.apply.average
    DOS = bz_avg.eigenstate(wrap=lambda es:es.DOS(energies)).squeeze() / Ham.no
    print(DOS.shape)
    return (DOS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Get eigenstates from band structure
    """)
    return


@app.cell
def _(bs):
    eigs = bs.apply.array.eigh()
    return (eigs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot: band and DOS
    """)
    return


@app.cell
def _(bond):
    def plot_graphene_unitcell(ax_inset, geom):
        """
        Draw the 2-atom graphene unit cell with bonds and lattice vectors.
        geom must have nsc >= [3,3,1] so periodic NN bonds are visible.
        """
        a1, a2 = geom.cell[0, :2], geom.cell[1, :2]
        A = geom.xyz[0, :2]
        B = geom.xyz[1, :2]

        # NN bonds (including periodic images)
        geom.set_nsc([3, 3, 1])
        for ia in range(geom.na):
            idx, xyz_n = geom.close(ia, R=[0.1, bond + 0.01], ret_xyz=True)
            for Rj in xyz_n[1]:
                ri, rj = geom.xyz[ia, :2], Rj[:2]
                ax_inset.plot([ri[0], rj[0]], [ri[1], rj[1]], 'k-', lw=1., zorder=1)

        # Atoms
        ax_inset.scatter(*A, s=80, color='blue', zorder=3)
        ax_inset.scatter(*B, s=80, color='red', zorder=3)
        ax_inset.annotate('A', A, xytext=(-10, 2), textcoords='offset points', ha="center", va="center",
                          color='k')
        ax_inset.annotate('B', B, xytext=(10, 2), textcoords='offset points', ha="center", va="center",
                          color='k')

        # Lattice vectors
        for vec, label in [(a1, r'$\mathbf{a}_1$'), (a2, r'$\mathbf{a}_2$')]:
            ax_inset.annotate('', xy=A + vec, xytext=A, zorder=0,
                              arrowprops=dict(arrowstyle='->', color='grey', lw=1.5))
            if vec is a1:
                va = "top"
            elif vec is a2:
                va = "bottom"
            ax_inset.text(*(A + vec * 0.55), label,
                          color='k', ha='right', va=va)

        ax_inset.set_aspect('equal')
        ax_inset.axis('off')

    return (plot_graphene_unitcell,)


@app.cell
def _(
    DOS,
    Emax,
    Emin,
    Erange,
    FIG_DIR,
    eigs,
    energies,
    geom,
    klabels,
    kticks,
    lineark,
    plot_graphene_unitcell,
    thesis_fig,
):
    _fig, _ax = thesis_fig(1,2, sharey=True, aspect=0.5)

    _ax[0].plot(lineark, eigs, color="k")
    _ymin, _ymax = _ax[0].get_ylim()
    _ax[0].set(
        xticks=kticks,
        xticklabels=klabels,
        xlim=(lineark[0], lineark[-1]),
        ylim=Erange,
        yticks=np.arange(Emin, Emax+0.1, 2),

        ylabel=r"$E-E_F$   (eV)",
        xlabel=r"$\mathbf{k}$   ($\mathrm{\AA}^{-1}$)",
        title="Band Structure"
    )
    _ax[0].grid()


    _ax[1].plot(DOS, energies, color="k")
    _ax[1].set(
        xlabel=r"DOS   ($\mathrm{eV}^{-1}$)",
        title=r"DOS (normalized)"
    )
    _ax[1].grid()

    for _peak in find_peaks(DOS)[0]:
        # print(_peak)
        _ax[0].axhline(energies[_peak], linestyle=":", color="k")
        _ax[1].axhline(energies[_peak], linestyle=":", color="k")



    _ax_inset = _ax[0].inset_axes([0.05, 0.28, 0.38, 0.42])
    plot_graphene_unitcell(_ax_inset, geom)

    # _bz_inset = _ax[0].inset_axes([0.8, 0.28, 0.38, 0.42])
    # _bz_inset.add_artist(bz_ax)


    _fig.set_constrained_layout(True)
    _fig.suptitle("Graphene")

    _fig.savefig(FIG_DIR / "show_graphene_band_DOS")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Plot Brillouinzone (for axes insets)
    """)
    return


@app.function
def plot_hexagonal_bz(ax, rcell):
    b1 = rcell[0, :2]
    b2 = rcell[1, :2]

    angle = np.degrees(np.arccos(
        np.dot(b1, b2) / (np.linalg.norm(b1) * np.linalg.norm(b2))
    ))

    if angle > 90:   # sisl convention (120°)
        K_points = np.array([
            ( 2*b1 +   b2) / 3,
            (   b1 + 2*b2) / 3,
            (  -b1 +   b2) / 3,
            (-2*b1 -   b2) / 3,
            (  -b1 - 2*b2) / 3,
            (   b1 -   b2) / 3,
        ])
    else:            # ASE convention (60°)
        K_points = np.array([
            ( 2*b1 -   b2) / 3,
            (   b1 +   b2) / 3,
            (  -b1 + 2*b2) / 3,
            (-2*b1 +   b2) / 3,
            (  -b1 -   b2) / 3,
            (   b1 - 2*b2) / 3,
        ])

    M_points = np.array([
        (K_points[i] + K_points[(i+1) % 6]) / 2
        for i in range(6)
    ])

    hex_xy = np.vstack([K_points, K_points[0]])
    ax.plot(hex_xy[:, 0], hex_xy[:, 1], 'k-', lw=1.2)

    # ax.scatter(*K_points.T, s=40, color='#E84855', zorder=3)
    # ax.scatter(*M_points.T, s=40, color='#2E86AB', zorder=3)
    ax.scatter(0, 0, s=60, color='k', zorder=3)
    ax.annotate('Γ', (0, 0),      xytext=(-5, 5), textcoords='offset points')
    ax.annotate('K', K_points[0], xytext=(5, 3), textcoords='offset points', color='k')
    ax.annotate('M', M_points[0], xytext=(5, 3), textcoords='offset points', color='k')

    ax.set_aspect('equal')
    lim = max(np.linalg.norm(b1), np.linalg.norm(b2)) * 1.3
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    ax.axis("off")

    return K_points, M_points


@app.cell
def plot_hexagonal_bz_ase():
    # def plot_hexagonal_bz_ase(ax, rcell):
    #     """
    #     Plot the hexagonal BZ using the ASE reciprocal cell convention (60° between vectors).

    #     Parameters
    #     ----------
    #     ax    : matplotlib axes
    #     rcell : array — ASE reciprocal cell, rows are vectors
    #     """

    #     b1 = rcell[0, :2]
    #     b2 = rcell[1, :2]

    #     K_points = np.array([
    #         ( 2*b1 -   b2) / 3,
    #         (   b1 +   b2) / 3,
    #         (  -b1 + 2*b2) / 3,
    #         (-2*b1 +   b2) / 3,
    #         (  -b1 -   b2) / 3,
    #         (   b1 - 2*b2) / 3,
    #     ])

    #     M_points = np.array([
    #         (K_points[i] + K_points[(i+1) % 6]) / 2
    #         for i in range(6)
    #     ])

    #     # Hexagon boundary
    #     hex_xy = np.vstack([K_points, K_points[0]])
    #     ax.plot(hex_xy[:, 0], hex_xy[:, 1], 'k-', lw=1.2)

    #     # Special points
    #     ax.scatter(*K_points.T, s=40, color='#E84855', zorder=3)
    #     ax.scatter(*M_points.T, s=40, color='#2E86AB', zorder=3)
    #     ax.scatter(0, 0, s=60, color='k', zorder=3)
    #     ax.annotate('Γ', (0, 0),      xytext=(5, 5), textcoords='offset points', fontsize=10)
    #     ax.annotate('K', K_points[0], xytext=(5, 3), textcoords='offset points', fontsize=10, color='#E84855')
    #     ax.annotate('M', M_points[0], xytext=(5, 3), textcoords='offset points', fontsize=10, color='#2E86AB')

    #     # # Reciprocal lattice vectors
    #     # for vec, name in [(b1, r'$\mathbf{b}_1$'), (b2, r'$\mathbf{b}_2$')]:
    #     #     ax.annotate('', xy=vec, xytext=(0, 0),
    #     #                 arrowprops=dict(arrowstyle='->', color='#444', lw=1.5))
    #     #     ax.text(*(vec * 0.55), name, fontsize=9, color='#444', ha='center')

    #     ax.set_aspect('equal')
    #     # ax.axis('off')
    return


@app.function
def plot_bz_path(ax, rcell, kpoints_frac, knames=None):
    """
    Plot a k-path on an existing BZ axes.

    Parameters
    ----------
    ax            : matplotlib axes (with BZ already plotted)
    rcell         : array — reciprocal cell, rows are vectors
    kpoints_frac  : list of (3,) arrays — k-points in fractional coordinates
    knames        : list of str, optional — labels for each k-point
    """
    b1 = rcell[0, :2]
    b2 = rcell[1, :2]

    kpoints_cart = np.array([kf[0]*b1 + kf[1]*b2 for kf in kpoints_frac])

    ax.plot(kpoints_cart[:, 0], kpoints_cart[:, 1], 'r--', lw=1.2, zorder=2)
    ax.scatter(kpoints_cart[:, 0], kpoints_cart[:, 1], s=30, color='r', zorder=3)

    if knames is not None:
        for kc, name in zip(kpoints_cart, knames):
            ax.annotate(name, kc, xytext=(5, 5), textcoords='offset points', fontsize=9, color='r')


@app.cell
def _(FIG_DIR, Ham, kpoints_sisl, thesis_fig):
    _fig, _ax = thesis_fig()

    plot_hexagonal_bz(_ax, Ham.icell)


    plot_bz_path(_ax, Ham.icell, kpoints_sisl, knames=None)
    # Reciprocal lattice vectors (rows of rcell)
    b1 = Ham.icell[0, :2]
    b2 = Ham.icell[1, :2]

    for _vec, _label in [(b1, r'$\mathbf{b}_1$'), (b2, r'$\mathbf{b}_2$')]:
        _ax.annotate('', xytext=_vec, xy=(0, 0),
                    arrowprops=dict(arrowstyle='<-', color='k', lw=1.5))
        _ax.text(*(_vec*0.75 + np.array([-0.06, 0])), _label, color='k', ha='center')


    # _fig.suptitle("BZ from sisl")
    _fig.savefig(FIG_DIR / "show_graphene_brillouin_zone")
    _fig
    # _ax.quiver(0,0, A[0], A[1], units="xy", angles="xy", scale=1)
    return


@app.cell
def _(Ham):
    # sisl reciprocal vectors in cartesian
    print("sisl b1:", Ham.rcell[0, :2])
    print("sisl b2:", Ham.rcell[1, :2])
    print("angle sisl:", np.degrees(np.arccos(
        np.dot(Ham.rcell[0,:2], Ham.rcell[1,:2]) /
        (np.linalg.norm(Ham.rcell[0,:2]) * np.linalg.norm(Ham.rcell[1,:2]))
    )))

    # ASE reciprocal vectors
    ase_cell = HEX2D(a=np.linalg.norm(Ham.cell[0])).bandpath().cell.reciprocal()
    print("\nASE b1:", ase_cell[0, :2])
    print("ASE b2:", ase_cell[1, :2])
    print("angle ASE:", np.degrees(np.arccos(
        np.dot(ase_cell[0,:2], ase_cell[1,:2]) /
        (np.linalg.norm(ase_cell[0,:2]) * np.linalg.norm(ase_cell[1,:2]))
    )))

    # Then check where K lands in cartesian for both
    K_frac = np.array([1/3, 1/3, 0])
    print("\nsisl K cartesian:", K_frac[:2] @ Ham.rcell[:2, :2])
    print("ASE  K cartesian:", K_frac[:2] @ ase_cell[:2, :2])
    return


@app.cell
def _(Ham):
    # ASE K in cartesian (using ASE's rcell)
    ase_rcell = HEX2D(a=np.linalg.norm(Ham.cell[0])).bandpath().cell.reciprocal()

    for name, kf in HEX2D(a=np.linalg.norm(Ham.cell[0])).bandpath(path="GMK").special_points.items():
        k_cart = kf[:2] @ ase_rcell[:2, :2]
        k_sisl = np.linalg.solve(Ham.rcell[:2, :2].T, k_cart)
        print(f"{name}: ASE frac={kf[:2]}  cart={k_cart.round(4)}  sisl frac={k_sisl.round(4)}")
    return (ase_rcell,)


@app.cell
def _(ase_rcell):

    print(ase_rcell)
    _b1 = ase_rcell[0, :2]
    _b2 = ase_rcell[1, :2]

    _K_points = np.array([
        ( 2*_b1 - _b2) / 3,
        (   _b1 + _b2) / 3,
        (  -_b1 + 2*_b2) / 3,
        (-2*_b1 +   _b2) / 3,
        (  -_b1 -   _b2) / 3,
        (   _b1 - 2*_b2) / 3,
    ])

    print("K norms:", np.linalg.norm(_K_points, axis=1).round(4))
    print("K points:\n", _K_points.round(4))
    print("\nb1:", _b1.round(4))
    print("b2:", _b2.round(4))
    print("angle:", np.degrees(np.arccos(
        np.dot(_b1, _b2) / (np.linalg.norm(_b1) * np.linalg.norm(_b2))
    )).round(2), "°")
    return


@app.cell
def _(ase_rcell, knames, kpoints_sisl):
    _fig, _ax = plt.subplots(figsize=(4, 4))

    plot_hexagonal_bz(_ax, ase_rcell[:2, :2])

    # Reciprocal lattice vectors (rows of rcell)
    _b1 = ase_rcell[0, :2]
    _b2 = ase_rcell[1, :2]

    plot_bz_path(_ax, ase_rcell, kpoints_sisl, knames=knames)

    for _vec, _label in [(_b1, r'$\mathbf{b}_1$'), (_b2, r'$\mathbf{b}_2$')]:
        _ax.annotate(_label, xytext=_vec, xy=(0, 0), ha="center", va="center",
                    arrowprops=dict(arrowstyle='<-', color='k', lw=2), zorder=1
                    )
        # _ax.text(*(_vec + np.array([-0.05, 0])), _label, fontsize=9, color='k', ha='right', va="center")


    _fig.suptitle("BZ from ase")
    _fig
    return


if __name__ == "__main__":
    app.run()
