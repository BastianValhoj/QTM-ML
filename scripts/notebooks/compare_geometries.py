import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")

with app.setup:
    import sisl
    import matplotlib.pyplot as plt
    from mytools.construct import all_armchair
    import numpy as np

    from pathlib import Path
    from itertools import combinations


@app.cell
def _():
    FIG_DIR = Path(__file__).parent.parent / "figures"
    FIG_DIR, FIG_DIR.exists()
    return (FIG_DIR,)


@app.cell
def _():
    bond = 1.42
    return (bond,)


@app.cell
def _(bond):
    geom_zig = sisl.geom.graphene(bond)
    geom_arm = all_armchair(bond)
    return geom_arm, geom_zig


@app.cell
def _(geom_zig):
    cell_zig = geom_zig.cell
    A_zig, B_zig = cell_zig[:2, :2]
    UC_zig = np.array([
        [0,0],
        A_zig,
        A_zig+B_zig,
        B_zig,
        [0,0]

    ])
    return A_zig, B_zig, UC_zig


@app.cell
def _(geom_arm):
    cell_arm = geom_arm.cell
    A_arm, B_arm = cell_arm[:2, :2]
    UC_arm = np.array([
        [0,0],
        A_arm,
        A_arm+B_arm,
        B_arm,
        [0,0],

    ])
    return A_arm, B_arm, UC_arm


@app.cell
def _(A_arm, A_zig, B_arm, B_zig, bond):
    Rz = lambda theta: np.array([[np.cos(np.deg2rad(theta)), -np.sin(np.deg2rad(theta))],[np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta))]])
    def plot_bonds(coord, kind, theta):
        if kind == "zigzag":
            A, B = A_zig, B_zig
        else:
            A, B = A_arm, B_arm
        vec = bond*A/np.linalg.norm(A)

        return Rz(theta)@vec



    return


@app.cell
def _(bond, geom_arm):
    def atom_bonds(geom):

        atoms_idx = np.arange(geom.na)
        atoms_pairs = combinations(atoms_idx, 2)
        lines = []
        for pair in atoms_pairs:
            if np.linalg.norm(geom.Rij(pair[0], pair[1])) <= bond+0.01:
                lines.append([geom.xyz[pair[0], :2], geom.xyz[pair[1], :2]])
        return np.asarray(lines)

    atom_bonds(geom_arm)
    return (atom_bonds,)


@app.cell
def _(geom_arm, geom_zig):
    geom_arm_tile = geom_arm.tile(2,0).tile(2,1)
    geom_zig_tile = geom_zig.tile(2,0).tile(2,1)
    return geom_arm_tile, geom_zig_tile


@app.cell
def _(
    A_arm,
    A_zig,
    B_arm,
    B_zig,
    FIG_DIR,
    UC_arm,
    UC_zig,
    atom_bonds,
    geom_arm,
    geom_arm_tile,
    geom_zig_tile,
):
    from mytools.plots import thesis_fig
    _fig, _axes = thesis_fig(subplots=(1,2), fraction=1)
    _size = 120

    _alpha = 0.15
    _axes[0].scatter(*geom_arm_tile.xyz[:, :2].T, color="grey", marker="o", s=_size)
    _axes[0].plot(*UC_arm.T)
    _axes[0].plot(*(UC_arm + B_arm).T, color="royalblue")
    _axes[0].plot(*(UC_arm + A_arm).T, color="royalblue")
    _axes[0].plot(*(UC_arm + A_arm + B_arm).T, color="royalblue")

    _axes[0].fill(*(UC_arm + B_arm).T, color="royalblue", alpha=_alpha)
    _axes[0].fill(*(UC_arm + A_arm).T, color="royalblue", alpha=_alpha)
    _axes[0].fill(*(UC_arm + A_arm + B_arm).T, color="royalblue", alpha=_alpha)

    _axes[0].annotate(text="A", xy=geom_arm.xyz[2,:2], xytext=(-10,-10), textcoords="offset points", zorder=5)
    _axes[0].annotate(text="B", xy=geom_arm.xyz[4,:2], xytext=(-10,-10), textcoords="offset points", zorder=5)

    _bond_lines = atom_bonds(geom_arm_tile)
    for _bond in _bond_lines:
        _axes[0].plot(*_bond.T, color="k", zorder=0)

    _axes[0].set(title="Armchair")



    _axes[1].scatter(*geom_zig_tile.xyz[:, :2].T, color="grey", marker="o", s=_size)
    _axes[1].plot(*UC_zig.T, color="royalblue")
    _axes[1].plot(*(UC_zig + B_zig).T, color="royalblue")
    _axes[1].plot(*(UC_zig + A_zig).T, color="royalblue")
    _axes[1].plot(*(UC_zig + A_zig + B_zig).T, color="royalblue")

    _axes[1].fill(*(UC_zig + B_zig).T, color="royalblue", alpha=_alpha)
    _axes[1].fill(*(UC_zig + A_zig).T, color="royalblue", alpha=_alpha)
    _axes[1].fill(*(UC_zig + A_zig + B_zig).T, color="royalblue", alpha=_alpha)

    _bond_lines = atom_bonds(geom_zig_tile)
    for _bond in _bond_lines:
        _axes[1].plot(*_bond.T, color="k", zorder=0)

    _axes[1].set(title="Zig-zag")


    _axes[1].annotate("A", xy=geom_zig_tile.xyz[0, :2], ha="center", va="center", xytext=(-5, 10), textcoords="offset points", zorder=5)
    _axes[1].annotate("B", xy=geom_zig_tile.xyz[1, :2], ha="center", va="center", xytext=(-5, 10), textcoords="offset points", zorder=5)

    for _ax in _axes:
        _ax.axis("equal")
        _ax.axis("off")
    _fig.savefig(FIG_DIR / f"{Path(__file__).stem}")
    _fig
    return (thesis_fig,)


@app.cell
def _(B_arm, UC_arm):
    UC_arm + B_arm
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Show with self-energy region
    """)
    return


@app.cell
def _(geom_arm, geom_zig):
    geom_se_arm = geom_arm.tile(4,0).tile(4,1)
    geom_se_zig = geom_zig.tile(6,0).tile(6,1)
    geom_se_arm.na, geom_se_zig.na
    return geom_se_arm, geom_se_zig


@app.cell
def _(A_arm, geom_se_arm):
    A_arm_super, B_arm_super = geom_se_arm.cell[:2, :2]
    UC_L_elec_arm = np.array([[0.0, 0.0],B_arm_super, A_arm+B_arm_super, A_arm, [0.,0.]])
    UC_R_elec_arm = UC_L_elec_arm-A_arm + A_arm_super
    return A_arm_super, UC_L_elec_arm, UC_R_elec_arm


@app.cell
def _():
    return


@app.cell
def _(A_arm_super, A_zig, B_arm, geom_se_zig):
    A_zig_super, B_zig_super = geom_se_zig.cell[:2, :2]
    UC_L_elec_zig = np.array([[0.0, 0.0], B_zig_super, A_zig+B_zig_super, A_zig, [0.,0.]])
    UC_R_elec_zig = UC_L_elec_zig-A_zig + A_zig_super
    UC_T_elec_zig = np.array([[0.,0.], A_arm_super, A_arm_super+B_arm, B_arm, [0.,0.]])
    return UC_L_elec_zig, UC_R_elec_zig


@app.cell
def _(
    FIG_DIR,
    UC_L_elec_arm,
    UC_L_elec_zig,
    UC_R_elec_arm,
    UC_R_elec_zig,
    atom_bonds,
    geom_se_arm,
    geom_se_zig,
    thesis_fig,
):
    _fig, _axes = thesis_fig(subplots=(1,2))

    _size = 30
    _alpha = 0.4
    _fontsize = 14
    _rot = 30

    _fill_color = "lightcoral"

    _axes[0].scatter(*geom_se_arm.xyz[:, :2].T, color="grey", marker="o", s=_size)

    _axes[0].plot(*UC_L_elec_arm.T, color=_fill_color)
    _axes[0].plot(*UC_R_elec_arm.T, color=_fill_color)
    # _axes[0].plot(*(UC_arm + A_arm).T, color="royalblue")
    # _axes[0].plot(*(UC_arm + A_arm + B_arm).T, color="royalblue")

    _axes[0].fill(*UC_L_elec_arm.T, color=_fill_color, alpha=_alpha)
    _axes[0].fill(*UC_R_elec_arm.T, color=_fill_color, alpha=_alpha)


    _axes[0].annotate(r"$\boldsymbol{\Sigma}$", xy=UC_R_elec_arm.mean(axis=0), textcoords="offset points", xytext=(0,0), ha="center", va="center", rotation=_rot, fontsize=_fontsize)
    _axes[0].annotate(r"$\boldsymbol{\Sigma}$", xy=UC_L_elec_arm.mean(axis=0), textcoords="offset points", xytext=(0,0), ha="center", va="center", rotation=_rot, fontsize=_fontsize)


    _bond_lines = atom_bonds(geom_se_arm)
    for _bond in _bond_lines:
        _axes[0].plot(*_bond.T, color="k", zorder=0)

    _axes[0].set(title="Armchair")



    _axes[1].scatter(*geom_se_zig.xyz[:, :2].T, color="grey", marker="o", s=_size)
    _axes[1].plot(*UC_L_elec_zig.T, color=_fill_color)
    _axes[1].plot(*UC_R_elec_zig.T, color=_fill_color)

    _axes[1].fill(*UC_L_elec_zig.T, color=_fill_color, alpha=_alpha)
    _axes[1].fill(*UC_R_elec_zig.T, color=_fill_color, alpha=_alpha)


    _axes[1].annotate(r"$\boldsymbol{\Sigma}$", xy=UC_R_elec_zig.mean(axis=0), textcoords="offset points", xytext=(0,0), ha="center", va="center", rotation=_rot, fontsize=_fontsize)
    _axes[1].annotate(r"$\boldsymbol{\Sigma}$", xy=UC_L_elec_zig.mean(axis=0), textcoords="offset points", xytext=(0,0), ha="center", va="center", rotation=_rot, fontsize=_fontsize)

    _bond_lines = atom_bonds(geom_se_zig)
    for _bond in _bond_lines:
        _axes[1].plot(*_bond.T, color="k", zorder=0)

    _axes[1].set(title="Zig-zag")


    for _ax in _axes:
        _ax.axis("equal")
        _ax.axis("off")
    _fig.savefig(FIG_DIR / f"{Path(__file__).stem}")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Unit cell atom sorting
    """)
    return


@app.cell
def _(atom_bonds, geom_arm, thesis_fig):
    _geom = geom_arm.tile(3,0)

    _fig, _ax = thesis_fig()
    _offset = 8, 12
    for _i in range(_geom.na):
        _xyz = _geom.xyz[_i, :2]
        _ax.scatter(*_xyz, color="grey", s=100, marker="o")
        _ax.annotate(_i, xy=_xyz, xycoords="data", textcoords="offset points", xytext=_offset, ha="center", va="center")


    _bond_lines = atom_bonds(_geom)
    for _bond in _bond_lines:
        _ax.plot(*_bond.T, color="k", zorder=0)
    _ymin, _ymax = _ax.get_ylim()
    _xmin, _xmax = _ax.get_xlim()
    _ax.axis("scaled")
    _ax.set(ylim=(_ymin, _ymax), xlim=(_xmin, _xmax))
    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
