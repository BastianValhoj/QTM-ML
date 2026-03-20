
import numpy as np
import sisl


# rotation matrices
Rz = lambda theta: np.array([[np.cos(theta), -np.sin(theta), 0],
                              [np.sin(theta), np.cos(theta), 0],
                              [0,0,1]])
Rx = lambda theta: Rz(theta)[np.ix_([2,0,1], [2,0,1])]
Ry = lambda theta: Rz(theta)[np.ix_([1,2,0], [1,2,0])]



def sort_atoms(geom, order='xzy', atol=1e-2):
    xyz = geom.xyz.round(5)
    mask = np.isclose(xyz, 0, atol=atol)
    xyz = np.where(mask, 0, xyz)
    mapping = {'x': 0, 'y': 1, 'z': 2}
    axes = [mapping[ax] for ax in reversed(order)]
    # print(axes)
    idxs = np.lexsort([xyz[:, axes[0]], xyz[:, axes[1]], xyz[:, axes[2]]])
    return idxs
def make_device(bond=1.42, kind="armchair"):
    gr = sisl.geom.graphene(bond=bond, atoms="C", orthogonal=True) # returns *orthogonal* unitcell instead of primitive
    
    if kind == "armchair":
        idxs = sort_atoms(gr, order='xzy', atol=1e-2)
        gr = gr.sub(idxs)
        return gr
    
    elif kind == "zigzag":
        gr = gr.rotate([180, [0,1,0]]).rotate([-90, [0,0,1]])
        idxs = sort_atoms(gr, order='xzy', atol=1e-2)
        # print(idxs)
        cell = gr.cell[[1,0,2], :]
        gr.set_lattice(cell)
        gr.set_nsc([3,3,1])
        gr = gr.sub(idxs)
        return gr



def all_armchair(bond):
    theta = np.pi*(3/2) # angle in hexagon lattice
    phi = theta/2
    b = bond*np.sqrt(3)/2
    gr = sisl.geom.graphene_flake(0, bond=bond)
    gr = gr\
        .translate(-gr.center())\
            .rotate([phi/2, [0,0,1]], rad=True, what="xyz") \
                .translate([2*b, 0, 0])
                
    base = np.array([3*bond, 0, 0])
    armchair_cell = np.array([Rz(-phi/2)@base,
                            Rz(phi/2)@base,
                            [0,0,20]])
    gr.set_lattice(armchair_cell)
    sort_idx = np.lexsort([gr.xyz[:, 2], gr.xyz[:, 1], gr.xyz[:, 0]])
    gr = gr.sub(sort_idx)
    return gr