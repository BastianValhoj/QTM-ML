
import numpy as np
import sisl


# rotation matrices
Rz = lambda theta: np.array([[np.cos(theta), -np.sin(theta), 0],
                              [np.sin(theta), np.cos(theta), 0],
                              [0,0,1]])
Rx = lambda theta: Rz(theta)[np.ix_([2,0,1], [2,0,1])]
Ry = lambda theta: Rz(theta)[np.ix_([1,2,0], [1,2,0])]



def sort_atoms(geom, order='xzy', atol=1e-2):
    """Sort atoms according to `order`

    Parameters
    ----------
    geom : sisl.Geometry, or child classes
        The geometry to reorder
    order : str, optional
        the order to sort (buy most to least important), by default 'xzy'
    atol : float, optional
        the tolerance for rounding +/- float to 0, by default 1e-2.
        Used for creating sorting mask, and therefore the entries of `-0.000000x` converts to `0.0` 

    Returns
    -------
    idxs : np.ndarray
        The indices sorting the atoms according to the `order`
    """
    xyz = geom.xyz.round(5)
    mask = np.isclose(xyz, 0, atol=atol)
    xyz = np.where(mask, 0, xyz)
    mapping = {'x': 0, 'y': 1, 'z': 2}
    axes = [mapping[ax] for ax in reversed(order)]
    # print(axes)
    idxs = np.lexsort([xyz[:, axes[0]], xyz[:, axes[1]], xyz[:, axes[2]]])
    return idxs

def make_device(bond=1.42, kind="armchair"):
    """Create a-NGR or z-NGR unitcell for tiling
    """
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

def all_armchair(bond=1.42):
    """create a graphene 2D layer having armchair edges all around.
    The regular method is `sisl.geom.graphene(bond)` returns a graphene layer of all zig-zag edges

    Parameters
    ----------
    bond : float, optional
        The bond length in graphene, default is 1.42

    Returns
    -------
    gr : sisl.Geometry
        The unitcell (of 6 atoms) for a all armchair edge graphene layer, with nsc=[3,3,1]
    """
    theta = np.pi*(2/3) # angle in hexagon lattice
    phi = theta/2 # half angle
    b = bond*np.sqrt(3)/2 # inner-radii of hexagon
    vac = 20
    gr = sisl.geom.graphene_flake(0, bond=bond, vacuum=vac) # based on flake structure for conveniece
    gr = gr.translate(-gr.center()) # center structure to origo
    gr = gr.rotate([phi/2, [0,0,1]], rad=True, what="xyz") # rotate  atoms by 30 degrees (only atoms and not unit cell)
    gr = gr.translate([2*b, 0, 0]) # translate twice by inner radii to make edges follow unit cell vectors
    
    # define and set the new unit vectors
    base = np.array([3*bond, 0, 0]) 
    armchair_cell = np.array([Rz(-phi/2)@base, # rotate vector A by -30 degrees (clockwise)
                            Rz(phi/2)@base, # rotate vector B by 30 degrees (counter-clockwise)
                            [0,0,vac]]) # no change for C vector
    
    gr.set_lattice(armchair_cell) 
    
    
    
    # sort atom by keys -- right-most / last key (the x-coord) is most 'dominant' and uses others as tie breakers 
    # this method only sorts correctly if we use fractional coordiantes
    fracxyz = gr.xyz @ np.linalg.inv(gr.cell)
    
    exponent = 6
    tol = 10**(-exponent)
    fracxyz = np.where(np.isclose(fracxyz, 0, atol=tol), 0.0, fracxyz.round(exponent)) 
    sort_idx = np.lexsort([fracxyz[:, 2], fracxyz[:, 1], fracxyz[:, 0]])
    gr = gr.sub(sort_idx)
    gr.set_nsc([3,3,1])
    return gr
