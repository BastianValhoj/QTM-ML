import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")

with app.setup:
    import sisl
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    import os


@app.cell
def _():
    NC = 0
    _Nstart = 1 + 2*(1+NC)
    N_bases = range(_Nstart, 13)
    N_target = 30
    return NC, N_target


@app.cell
def _(NC, N_target):
    SCRIPT_DIR = Path(__file__).parent
    WORK_DIR = Path.home() / "w3"
    DATA_DIR = WORK_DIR / "rsse_data"
    for _d in DATA_DIR.glob(f"TBT-NC{NC}*_to_{N_target}"):
        print(_d.name)
    return (DATA_DIR,)


@app.cell
def _(DATA_DIR, NC, N_target):
    OUT_DIR = lambda N: DATA_DIR / f"TBT-NC{NC}_{N}_to_{N_target}"
    return (OUT_DIR,)


@app.cell
def _(OUT_DIR):
    _N = 10
    print(OUT_DIR(_N).relative_to(Path.home()))
    print([_file for _file in os.listdir(OUT_DIR(_N)) if ("TBT" in _file) and (".nc" in _file)])
    return


@app.cell
def _(OUT_DIR):
    N_test = 10
    se_top = sisl.get_sile(OUT_DIR(N_test) / "tbt-top.TBT.SE.nc")
    se_bottom = sisl.get_sile(OUT_DIR(N_test) / "tbt-bottom.TBT.SE.nc")
    Ham_base = sisl.get_sile(OUT_DIR(N_test) / "Ham_re_big.nc").read_hamiltonian()
    return Ham_base, N_test, se_bottom, se_top


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Show down-folded self energies for either `top` or `bottom`
    """)
    return


@app.cell
def _(se_bottom, se_top):
    idx_down_top = se_top.a_down('border'.capitalize())
    idx_down_bottom = se_bottom.a_down('border'.capitalize())
    return


@app.cell
def _(se_top):
    def get_sigma(E=0):
        tmp = se_top.self_energy(elec='Border', E=E, k=[0,0,0], sort=True).data
        pvt = se_top.pivot(elec='Border', in_device=True, sort=True)
        se_top.no_d

        sigma  = np.zeros(shape=(se_top.no_d, se_top.no_d), dtype=np.complex128)
        sigma[np.ix_(pvt,pvt)] += tmp
        return sigma

    return (get_sigma,)


@app.cell
def _(get_sigma):
    sigma = get_sigma(E=1)
    vmax = np.abs(sigma.imag).max()
    vmin = -vmax
    plt.imshow(sigma.imag[:,:], cmap="viridis", vmin=vmin, vmax=vmax)
    return (sigma,)


@app.cell
def _(se_top, sigma):
    _fig, _ax = plt.subplots()
    pvt_full = se_top.pivot(elec=0, in_device=False, sort=True)
    pvt      = se_top.pivot(elec=0, in_device=True, sort=True)
    size = np.abs(np.diag(sigma[np.ix_(pvt, pvt)].imag))
    size = size / size.max()
    _ax.scatter(*se_top.xyz[pvt_full, :2].T, s=0.1+size*1e1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Verify whether the saved self energy is the electrode (down-folded of course!)
    """)
    return


@app.cell
def _(Ham_base, get_sigma, se_top):
    from tqdm.auto import tqdm
    Ham_top = Ham_base.sub(se_top.a_dev)

    invG = []
    for E in tqdm(se_top.E):
        Sk = Ham_top.Sk(dtype=complex)
        Hk = Ham_top.Hk(dtype=complex)
        z = E + se_top.eta()*1j
        _invg = z*Sk - Hk - get_sigma(E=E)
        # 
        invG.append(_invg)
    invG = np.asarray(invG)
    return Ham_top, invG


@app.cell
def _(invG, se_top):
    G = np.linalg.inv(invG)
    _ldos = -1/np.pi * np.imag(np.diagonal(G, axis1=1, axis2=2))
    dos = _ldos.sum(axis=1) / se_top.na_dev
    _fig, _ax = plt.subplots()
    _ax.plot(se_top.E, dos)
    return


@app.cell
def _(Ham_base):
    print([att for att in dir(Ham_base)
         if not att.startswith("_")])
    return


@app.cell
def _(Ham_base):
    bilayer_displacement = 3.35 # Å
    geom_bottom = Ham_base.geometry.copy()
    geom_top    = Ham_base.geometry.copy()
    print(geom_top.xyz[0])
    geom_top = geom_top.translate([0,0,bilayer_displacement])
    print(geom_top.xyz[0])
    return bilayer_displacement, geom_bottom, geom_top


@app.cell
def _(se_bottom, se_top):
    a_dev_top = se_top.a_dev
    a_dev_bottom = se_bottom.a_dev + se_top.na
    print(f"   Top device atoms: {len(a_dev_top)}")
    print(f"Bottom device atoms: {len(a_dev_bottom)}")
    return a_dev_bottom, a_dev_top


@app.cell
def _(a_dev_bottom, a_dev_top, se_top):
    a_dev = np.concat([a_dev_top, se_top.na + a_dev_bottom])
    se_top.na, se_top.na + a_dev_bottom
    return


@app.cell
def _(bilayer_displacement, geom_bottom, geom_top):
    geom_bottom.cell[2, 2] += bilayer_displacement   # or set explicitly
    test_geom = geom_top.add(geom_bottom)
    return (test_geom,)


@app.cell
def _(a_dev_bottom, a_dev_top, test_geom):
    a_buf = np.delete(np.arange(test_geom.na), np.concat([a_dev_top, a_dev_bottom]))
    a_buf.shape, test_geom.na
    return (a_buf,)


@app.cell
def _(a_buf, a_dev_bottom, a_dev_top, test_geom):
    test_geom.plot(axes=[[1,0,0], [0,1,0]], backend="matplotlib", show_bonds=False, show_cell=False,
        atoms_style=[
            dict(atoms=a_dev_bottom, color="blue", border_width=0.), 
            dict(atoms=a_dev_top, color="red", border_width=0,),
            dict(atoms=a_buf, opacity=0.07)
            ]
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Method for loading and stacking device regions from `*.TBT.SE.nc` files
    """)
    return


@app.cell
def _(geom_top):
    print([_att for _att in dir(geom_top)
        if not _att.startswith("_")
            ])
    print(geom_top)
    print(geom_top.names['Device'])
    return


@app.cell
def _(se_top):
    se_top.pivot
    return


@app.function
def stack_device(data_path, d=3.35):
    if isinstance(data_path, str):
        se_top = sisl.get_sile(data_path +"/tbt-top.TBT.SE.nc")
        se_bottom = sisl.get_sile(data_path +"/tbt-bottom.TBT.SE.nc")

    elif isinstance(data_path, Path):
        se_top = sisl.get_sile(data_path / "tbt-top.TBT.SE.nc")
        se_bottom = sisl.get_sile(data_path / "tbt-bottom.TBT.SE.nc")

    else:
        raise ValueError("'data_path' must be either of type: ['str', 'pathlib.Path']")
    if se_top.na_dev >= se_bottom.na_dev:
        raise ValueError(f"Device region of top layer has atoms equal or more than bottom layer: {se_top.na_dev} >= {se_bottom.na_dev}")

    # get geometries
    geom_top = se_top.geometry.copy()
    geom_top = geom_top.translate([0,0,d])
    geom_bottom = se_bottom.geometry.copy()

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


@app.cell
def _(geom_top):
    geom_top.add
    return


@app.cell
def _():
    _test = np.arange(9); print(_test)
    np.delete(_test, [1,2,3])
    return


@app.cell
def _(N_test, OUT_DIR):
    geom_bilayer, idx_dict = stack_device(OUT_DIR(N_test))
    # print(idx_dict)
    geom_bilayer.plot(axes=["x", [0, 1, 1]], show_cell=False, backend="matplotlib",
        atoms_style=[
            dict(atoms=idx_dict['elec_top'], color="red", border_width=0),
            dict(atoms=idx_dict['elec_bottom'], color="blue", border_width=0)
        ]
    )
    return geom_bilayer, idx_dict


@app.cell
def _(geom_bilayer, idx_dict):
    print("all  elec  atoms in the   `top`    layer have   z > 0 :", (geom_bilayer.xyz[idx_dict['elec_top'], 2] > 0).all())
    print("all device atoms in the   `top`    layer have   z > 0 :", (geom_bilayer.xyz[idx_dict['device_top'], 2] > 0).all())
    print("all  elec  atoms in the  `bottom`  layer have  z == 0 :", (geom_bilayer.xyz[idx_dict['elec_bottom'], 2] == 0).all())
    print("all device atoms in the  `bottom`  layer have  z == 0 :", (geom_bilayer.xyz[idx_dict['device_bottom'], 2] == 0).all())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Built bilayer TB hamiltonian
    """)
    return


@app.cell
def _():
    from mytools.tbbi import tbbi_opt
    tbbi_opt
    return (tbbi_opt,)


@app.cell
def _(geom_bilayer, tbbi_opt):
    Ham_bilayer = tbbi_opt(
        geom_bilayer,
        0.0,
        None,
        -2.7,
        0.48,
        1.42,
        3.35,
        2.0,
        None,
        0.0, # shift in on-site energy for edge atoms
        finite=True, # nsc=(1,1,1), no periodicity
    )
    return (Ham_bilayer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Ensure the indexing of the edge of electrode has the correct self-energies
    """)
    return


@app.cell
def _(Ham_top, se_top):
    _pvt = se_top.pivot(elec=0, in_device=True, sort=True)
    _se = se_top.self_energy(elec=0, E=0., sort=True, k=[0,0,0])
    # print(_pvt, _pvt.shape)
    Ham_top.geometry.plot(axes="xy", backend="matplotlib", show_bonds=False, show_cell=True,
    atoms_style=[
        dict(atoms=np.delete(range(Ham_top.na), _pvt), opacity=0.05, color="blue"),
        dict(atoms=_pvt, color="red", size=np.diag(np.abs(_se.imag))/np.diag(np.abs(_se.imag)).max())
    ])
    return


@app.cell
def _(Ham_bilayer, idx_dict):
    Ham_bilayer.geometry.plot(axes=[[1,0,0],[0,1,2]], backend="matplotlib", show_cell=False, show_bonds=False,
        atoms_style=[
            dict(atoms=idx_dict["elec_top"], color="red"),
            dict(atoms=idx_dict["device_top"], color="red", opacity=0.1),
            dict(atoms=idx_dict["elec_bottom"], color="blue"),
            dict(atoms=idx_dict["device_bottom"], color="blue", opacity=0.1),
        ]
    )
    return


@app.cell
def _(se_top):
    se_top.pivot
    return


@app.cell
def _(Ham_bilayer, se_bottom, se_top):
    _which = "bottom"
    if _which == "bottom":
        _pvt = se_bottom.pivot(elec=0, in_device=True, sort=True)
        _pvt = _pvt + se_top.na_dev
        _se = se_bottom.self_energy(elec=0, E=0., k=[0,0,0], sort=True)
    elif _which == "top":
        _pvt = se_top.pivot(elec=0, in_device=True, sort=True)
        _se = se_top.self_energy(elec=0, E=0., k=[0,0,0], sort=True)
    print(_pvt)
    _ham = Ham_bilayer.sub(_pvt)
    print(_ham.na, _se.shape)
    _size = np.diag(np.abs(_se.imag))
    _size = _size / _size.max() *5 + 0.01
    _ham.geometry.plot(axes=["x", "y"], backend="matplotlib", show_cell=False, show_bonds=False,
        atoms_style=dict(atoms=range(len(_pvt)), color="red", size=_size)
        )
    return


@app.cell
def _(se_top):
    for _d in dir(se_top.self_energy(elec=0,E=0., k=0, sort=True)):
        if not _d.startswith("_"):
            print(_d)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ensure reordering of `Ham_bilayer`works
    """)
    return


@app.cell
def _(Ham_bilayer, idx_dict, se_bottom, se_top):
    _which = "top"

    if _which == "bottom":
        # _pvt = se_bottom.pivot(elec=0, in_device=True, sort=True)
        _pvt = idx_dict["elec_bottom"]
        _se = se_bottom.self_energy(elec=0, E=0., k=[0,0,0], sort=True)
    elif _which == "top":
        # _pvt = se_top.pivot(elec=0, in_device=True, sort=True)
        _pvt = idx_dict["elec_top"]
        _se = se_top.self_energy(elec=0, E=0., k=[0,0,0], sort=True)
    _size = np.diag(np.abs(_se.imag))
    _size = _size / _size.max() *5 + 0.01
    Ham_bilayer.geometry.plot(axes=["x", "y"], backend="matplotlib", show_cell=False, show_bonds=False,
        atoms_style=[dict(atoms=_pvt, color="red", size=_size)]
        )
    return


@app.cell
def _(idx_dict):
    idx_dict['elec_bottom']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell
def _(Ham_bilayer, idx_dict):
    # Ham_re_bi = Ham_bilayer.sub(reorder_idx)
    Ham_bilayer.geometry.plot(axes="xy", backend="matplotlib", show_bonds=False, show_cell=False,
        atoms_style=[
            dict(atoms=idx_dict['elec_top'], color="red"),
            dict(atoms=idx_dict['elec_bottom'], color="blue")
        ]
    )
    return


@app.cell
def _(idx_dict):
    print(len(idx_dict["device_top"]) + len(idx_dict["elec_top"]))
    print(len(idx_dict["device_bottom"]) + len(idx_dict["elec_bottom"]))
    return


@app.cell
def _(Ham_bilayer, idx_dict):
    _test = Ham_bilayer.sub(idx_dict['elec_bottom'])
    _test.geometry.plot(axes="xy", backend="matplotlib")
    return


if __name__ == "__main__":
    app.run()
