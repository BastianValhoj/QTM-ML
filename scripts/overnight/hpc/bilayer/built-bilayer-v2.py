import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")

with app.setup:
    import sisl
    import numpy as np
    import matplotlib.pyplot as plt

    from pathlib import Path

    from mytools.tbbi import tbbi_opt


@app.cell
def _():
    NC = 0
    N_target = 30
    _Nstart = 1 + 2*(1+NC)
    N_bases = range(_Nstart, N_target)
    return NC, N_target


@app.cell
def _(NC, N_target):
    WORK_DIR = Path.home() / "w3"
    INPUT_DIR = WORK_DIR / "rsse_data"
    for _dir in INPUT_DIR.glob(f"TBT-NC{NC}_*_to_{N_target}"):
        print(_dir.name)
    return (INPUT_DIR,)


@app.cell
def _(INPUT_DIR, NC, N_target):
    DATA_DIR = lambda N: INPUT_DIR / f"TBT-NC{NC}_{N}_to_{N_target}"
    return (DATA_DIR,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Changed stacking to allow using geometry `top` or `bottom` in any combination
    """)
    return


@app.function
def stack_device(data_path, which=("bottom", "bottom"), d=3.35):
    assert (which[0] == "top" or which[0] == "bottom") and (which[1] == "top" or which[1] == "bottom"), "`which` has to be iterable of either strings `top` or `bottom`"

    assert isinstance(data_path, Path), "data_path has to be of type pathlib.Path`"
    se_top = sisl.get_sile(data_path / f"tbt-{which[0]}.TBT.SE.nc")
    se_bottom = sisl.get_sile(data_path / f"tbt-{which[1]}.TBT.SE.nc")
    # print(se_top.na, se_top.na_dev, se_top.na_d)
    # print(se_top.elecs)
    # print(se_bottom.elecs)
    
    # # get geometries
    geom_top = se_top.geometry.copy()
    geom_top = geom_top.translate([0, 0, d])
    # geom_top.xyz[:, 2] = d
    geom_bottom = se_bottom.geometry.copy()
    # geom_bottom.xyz[:, 2] = 0

    # get 'electrode' indices from pivot 
    elec_idx_top = se_top.pivot(elec=0, in_device=False, sort=True)
    elec_idx_bottom = se_bottom.pivot(elec=0, in_device=False, sort=True)

    # get device indices
    down_idx_top = se_top.a_dev
    device_idx_top = np.setdiff1d(down_idx_top, elec_idx_top)

    down_idx_bottom = se_bottom.a_dev
    device_idx_bottom = np.setdiff1d(down_idx_bottom, elec_idx_bottom)

    # save elec indices of non-downfolded, and bottom is offset by whole top layer (non-downfolded)
    elec_idx = np.concat([elec_idx_top, elec_idx_bottom+se_top.na])

    # save device indices of non-downfolded, and bottom is offset by whole top layer (non-downfolded)
    device_idx = np.concat([device_idx_top, device_idx_bottom+se_top.na])

    # get final reordering indices for subbing the full geometries of top and bottom
    reorder_sub_idx = np.concat([elec_idx, device_idx])

    # removed names indices (needed for the 'add' method to work)
    geom_top.names.clear()
    geom_bottom.names.clear()

    # stack the layers and sub according to reordering
    geom = geom_top.add(geom_bottom)
    geom_bilayer = geom.sub(reorder_sub_idx)


    # get length of different relevant regions (electrode/device of top/bottom)
    N_elec_top = len(elec_idx_top)
    N_elec_bottom = len(elec_idx_bottom)
    N_elec_total = N_elec_top + N_elec_bottom

    N_device_top = len(device_idx_top)
    N_device_bottom = len(device_idx_bottom)
    N_device_total = N_device_top + N_device_bottom

    # save indices of subbed geometry to dict
    idx_dict = {
        'elec_top': range(N_elec_top),
        'elec_bottom': range(N_elec_top, N_elec_total),
        'device_top': range(N_elec_total, N_elec_total + N_device_top),
        'device_bottom': range(N_elec_total + N_device_top, N_elec_total + N_device_total)
        }
    return geom_bilayer, idx_dict


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Make Bilayer Hamiltonian -- before applying SK-coupling mask
    """)
    return


@app.cell
def _(DATA_DIR):
    _N = 10
    inter_layer_dist = 3.35 # Å
    geom_bilayer, idx_dict = stack_device(DATA_DIR(_N), which=["bottom", "bottom"], d=inter_layer_dist)
    print({k: (min(v), max(v), len(v)) for k, v in idx_dict.items()})
    print(f"Total atoms: {geom_bilayer.na}")
    return geom_bilayer, idx_dict, inter_layer_dist


@app.cell
def _():
    tbbi_opt
    return


@app.cell
def _(geom_bilayer):
    Ham_bilayer = tbbi_opt(
        geom_bilayer,
        os_0=0.0,
        os_1=0.0,
        Vpppi=-2.7,
        Vpps=0.48,
        finite=True,
        dangling=0.0
    )
    return (Ham_bilayer,)


@app.cell
def _(Ham_bilayer, idx_dict):
    _fig1 = Ham_bilayer.geometry.plot(axes=[[1,0,0], [0,0,1]], show_cell=False, show_bonds=False, backend="matplotlib",
        atoms_style=[
            dict(atoms=idx_dict['elec_top'], color="red"),
            dict(atoms=idx_dict["elec_bottom"], color="blue")
        ]
    )
    _fig2 = Ham_bilayer.geometry.plot(axes=[[1,0,0], [0,1,1]], show_cell=False, show_bonds=False, backend="matplotlib",
        atoms_style=[
            dict(atoms=idx_dict['elec_top'], color="red"),
            dict(atoms=idx_dict["elec_bottom"], color="blue")
        ]
    )

    _fig1, _fig2
    return


@app.cell
def _(geom_bilayer, idx_dict, inter_layer_dist):
    print("all  elec  atoms in the   `top`    layer have  z == d :", (geom_bilayer.xyz[idx_dict['elec_top'], 2] == inter_layer_dist).all())
    print("all device atoms in the   `top`    layer have  z == d :", (geom_bilayer.xyz[idx_dict['device_top'], 2] == inter_layer_dist).all())
    print("all  elec  atoms in the  `bottom`  layer have  z == 0 :", (geom_bilayer.xyz[idx_dict['elec_bottom'], 2] == 0).all())
    print("all device atoms in the  `bottom`  layer have  z == 0 :", (geom_bilayer.xyz[idx_dict['device_bottom'], 2] == 0).all())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Make hollow cylinder mask
    """)
    return


@app.cell
def _():
    for _att in dir(sisl.shape.EllipticalCylinder):
        if not _att.startswith("_"):
            print(_att)
    return


@app.cell
def _(Ham_bilayer, idx_dict):
    geom_center = Ham_bilayer.center()[:2]
    idx_top = list(idx_dict["elec_top"]) + list(idx_dict["device_top"])
    idx_bottom = list(idx_dict["elec_bottom"]) + list(idx_dict["device_bottom"])

    _radii_top = np.linalg.norm(Ham_bilayer.xyz[idx_top, :2] - geom_center, axis=1)
    _radii_bottom = np.linalg.norm(Ham_bilayer.xyz[idx_bottom, :2] - geom_center, axis=1)

    _max_radius_top = np.max(_radii_top)
    _max_radius_bottom = np.max(_radii_bottom)

    _max_electrode_radii = np.min([_max_radius_top, _max_radius_bottom])
    _height = Ham_bilayer.xyz[:,2].max() - Ham_bilayer.xyz[:,2].min()

    shift = 12
    Rmax = _max_electrode_radii - shift
    print(Rmax)

    hollow_cylinder = sisl.shape.EllipticalCylinder(Rmax, h=_height+10, center=Ham_bilayer.center())
    within_Rmax_mask = hollow_cylinder.within(Ham_bilayer.xyz)
    within_Rmax_mask.sum(), Ham_bilayer.na
    return Rmax, geom_center, idx_bottom, idx_top, within_Rmax_mask


@app.cell
def _(Ham_bilayer, within_Rmax_mask):
    Ham_bilayer.geometry.plot(axes=[[1,0,0], [0,1,1]], show_cell=False, show_bonds=False, backend="matplotlib",
        atoms_style=[
            dict(atoms=within_Rmax_mask, color="red"),
            dict(atoms=~within_Rmax_mask, color="k", opacity=0.1),
        ]
    )
    return


@app.cell
def _(Ham_bilayer, Rmax, geom_center, idx_bottom, idx_top, within_Rmax_mask):
    # convert to compressed sparse row (CSR)
    H_csr = Ham_bilayer.tocsr(0)

    # get row and column index of non-zero elements
    rows, cols = H_csr.nonzero()

    ## layer membership boolean arrays
    # Initiate bool mask for indentifying elements in top of bottom
    is_top = np.zeros(Ham_bilayer.na, dtype=bool)
    is_bot = np.zeros(Ham_bilayer.na, dtype=bool)

    # set mask elements to true for indices belogning to correct subset
    is_top[idx_top] = True
    is_bot[idx_bottom] = True

    # identify interlayer pairs. check   top->bot      OR     bot->top   
    # (rows being 'from' and cols being 'to')
    is_interlayer = (is_top[rows] & is_bot[cols]) | (is_bot[rows] & is_top[cols])

    in_boundary_region = ~within_Rmax_mask # '~' is the numpy 'not' operator

    # indices to be zero: if indices is interlayer AND (either 'from' atom or 'to' atom is outside Rmax)
    to_zero    = is_interlayer & (in_boundary_region[rows] | in_boundary_region[cols])

    print(f"Total nonzero elements:     {len(rows):,}")
    print(f"Interlayer elements:        {is_interlayer.sum():,}")
    print(f"Elements to zero:           {to_zero.sum():,}")
    print(f"Interlayer elements kept:   {(is_interlayer & ~to_zero).sum():,}")

    for i, j in zip(rows[to_zero], cols[to_zero]):
        Ham_bilayer[i, j] = 0.0

    Ham_bilayer.eliminate_zeros()

    # verify
    H_csr_after = Ham_bilayer.tocsr(0)
    rows_a, cols_a = H_csr_after.nonzero()
    interlayer_after = (is_top[rows_a] & is_bot[cols_a]) | (is_bot[rows_a] & is_top[cols_a])
    print(f"\nAfter masking:")
    print(f"Total nonzero elements:     {len(rows_a):,}")
    print(f"Interlayer elements:        {interlayer_after.sum():,}")

    # verify no interlayer coupling exists outside Rmax
    interlayer_outside = interlayer_after & (in_boundary_region[rows_a] | in_boundary_region[cols_a])
    print(f"Interlayer outside Rmax:    {interlayer_outside.sum():,}  (should be 0)")

    # lateral distance of each atom in interlayer pairs from center
    r_rows = np.linalg.norm(Ham_bilayer.xyz[rows[is_interlayer], :2] - geom_center, axis=1)
    r_cols = np.linalg.norm(Ham_bilayer.xyz[cols[is_interlayer], :2] - geom_center, axis=1)

    print(f"Rmax:                        {Rmax:.2f} Å")
    print(f"Max radius of rows in pairs: {r_rows.max():.2f} Å")
    print(f"Max radius of cols in pairs: {r_cols.max():.2f} Å")
    return


if __name__ == "__main__":
    app.run()
