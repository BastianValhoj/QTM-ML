import numpy as np
from scipy.spatial import cKDTree


def dict2array(d):
    lst = list(d.values()) # list of list of values
    flat = np.concatenate(lst) # concatenate list of list to a 1D array
    return flat

def get_corner_atoms(coords, corner, k=1):
    """Find corner atom indices belonging to specific corner

    Parameters
    ----------
    coords : np.ndarray
        xyz coordinates for each atom
    corner : np.ndarray
        The xyz position of the corner(s)
    k : int, optional
        k-NN: the number of atoms to include in each atom, by default 1

    Returns
    -------
    np.ndarray of shape (k,)
        atomic indices for the atoms belonging to the corner
        
    """
    
    tree = cKDTree(coords)
    dists, idxs = tree.query(corner, k=k)
    return np.sort(idxs)

def find_corners(geom):
    """Find all corners of the geometry

    Parameters
    ----------
    geom : sisl.Geometry, or child class
        The geometry to find the corners of
        
    Returns
    -------
    dict of arrays
        keys are the four cornors of geom: `[0, A, A+B, B]`, where `A` and `B` are the cell vectors, 
        and the values are atom indices belogning to each corner.
    """
    A, B, _ = geom.cell
    corners = np.array([A-A, A, A+B, B])
    ks = np.array([4, 3, 4, 3])
    idxs = {}
    labels = ['0', 'A', 'A+B', 'B']
    for c, k, lab in zip(corners, ks, labels):
        idx = get_corner_atoms(geom.xyz, c, k)
        idxs[lab] = idx
    return idxs


def find_pair_edges(geom, corners):
    """Find a pair of edge atoms (edge not including the corners)

    Parameters
    ----------
    geom : sisl.Geometry
        The geometry to find the edges of
    corners : dict
        dictionary with keys being the atomic indices of the corner atoms

    Returns
    -------
    pair_center : np.ndarray 
        the xyz coordinates of the center between edges atom pairs
    ii : np.ndarray of shape (2, N)
        the indices of the N pairs of edge atoms
    """
    edge_atoms = np.delete(np.arange(geom.na), dict2array(corners))
    tree = cKDTree(geom.xyz)
    dist, ii = tree.query(geom.xyz[edge_atoms], k=2)
    ii = ii[0::2] # skip every second entry because that is the other part of the edge pair
    pair_center = geom.xyz[ii].mean(axis=1, keepdims=True).squeeze()
    return pair_center, ii


def get_fractional(geom, coords=None):
    """Map Cartesion coordinates to fractional coordinates, 
    between 0 and 1 representing position relative to unitcell vertices."""
    Inv = geom.icell.T[:2, :2]
    if coords is not None:
        coords = coords[:, :2]
    else:
        coords = geom.xyz[:, :2]
    # elif (coords is not None) and (coords.shape[1] == 3):
    #     coords = geom.xyz[:, :2]
    frac = coords @ Inv
    return frac / np.max(frac, axis=0, keepdims=True)


def find_equiv_pair(geom1, geom2):
    """Find pairs of atoms in geom2 that should be equivalent to pairs in geom1

    Parameters
    ----------
    geom1 : sisl.Geometry
        The geometry to scale
    geom2 : sisl.Geometry
        The target geometry

    Returns
    -------
    corners1 : dict of arrays
        keys are the four cornors of geom1: `[0, A, A+B, B]`, where `A` and `B` are the cell vectors, 
        and the values are atom indices belogning to each corner.
    corners2 : dict of arrays
        keys are the four cornors of geom2.
    pairCenter2 : np.ndarray
        the xy coordinates of edge pair centers.
    edgeID1 : np.ndarray
        The edge atom indices of geom1
    pairIndice : np.ndarray
        Indices of edgeID1 that should belong to the i'th pair of edge atoms in geom2
    
    Notes
    -----
    The usecase is for the `map_index` method, which essentially does the following:
    ```python
    ElecGeometry_1 = .... # the smaller structure
    ElecGeometry_2 = .... # the larger structure
    
    xy1 = ElecGeometry_1.xyz[:, :2]
    xy2 = ElecGeometry_2.xyz[:, :2]
    
    _, _, _, edgeID1, pairIndices = find_equiv_pair(ElecGeometry_1, ElecGeometry_2)
    
    # edgeID1 : the index pairs, (e.g. [4,5], [6,7], ...) from the smaller geometry
    # pairIndices : list of length K for the edge atom pairs in `ElecGeometry_2`.
    # E.g. [0, 0, 1, 1, ...] if the first two pairs in geom2 should have the 0'th pair in edgeID1)
    
    g1_to_g2_indices = edgeID1[pairIndices].flatten()
    
    fig, ax = plt.subplots()
    
    # plot all atom positions in (x,y)
    ax.scatter(*xy1.T, s=100)
    ax.scatter(*xy2.T, s=100)
    
    # loop over all atoms of the smaller geometry and label them according to their index
    for i in range(ElecGeometry_1.na): 
        ax.text(*xy[i], s=i)
        
    for j, geom1_idx in enumerate(g1_to_g2_indices):
        ax.text(*xy2[j], s=geom1_idx)
    
    plt.show()
    
    ```
    
    """
    # if geom1.na > geom2.na:
    #     _tmp = geom1.copy()
    #     geom1 = geom2.copy()
    #     geom2 = _tmp.copy()
    #     del _tmp
    
    assert geom1.na < geom2.na, f"The first Geometry has to be smallest ({geom1.na} !< {geom2.na})"

    # xyz1 = geom1.xyz
    # xyz2 = geom2.xyz
    corners1 = find_corners(geom1)
    corners2 = find_corners(geom2)
    pairCenter1, edgeID1 = find_pair_edges(geom1, corners1)
    pairCenter2, edgeID2 = find_pair_edges(geom2, corners2)
    
    fracpairCenter1 = get_fractional(geom1, coords=pairCenter1)
    fracpairCenter2 = get_fractional(geom2, coords=pairCenter2)
    tree = cKDTree(fracpairCenter1)
    dists, pairIndice = tree.query(fracpairCenter2, k=1) # find the coords in fracpairCenter2 that is closest to coords in fracpairCenter1
    
    return corners1, corners2, pairCenter2, edgeID1, pairIndice



def map_index(geom1, geom2):
    """Extrapolate atom indices of geom1 to equivalent in geom2 using k-NN method

    Parameters
    ----------
    geom1 : sisl.Geometry, or child class
        The (electrode) structure to scale of N (electrode) atoms 
    geom2 : sisl.Geometry, or child class
        The target (electrode) structure of K (electrode) atoms.
        Transfer the equivalent indices from geom1 to geom2

    Returns
    -------
    np.ndarray of shape (2, K)
        first row is the indices of `geom2`, and the second row is the equivalent index in `geom1`
    """
    corners1, corners2, centers, edgeIDs, pairIndices = find_equiv_pair(geom1, geom2)
    g1_to_g2_edgeIdx = edgeIDs[pairIndices].flatten()
    
    
    all_atoms = np.arange(geom2.na)
    corner_atoms = dict2array(corners2)
    edge_atoms = np.delete(all_atoms, corner_atoms)
    
    # take difference between subsequent indices of the 'actual' indices of geom2
    diffs = np.diff(edge_atoms)

    # find indices where elements of diff increase more than 1,
    # since when the edge indices increase more than 1 it has moved beyond a corner to another edge
    split_indices = np.where(diffs > 1)[0] + 1
    # split edge indices to list of arrays for each edge
    parts = np.split(g1_to_g2_edgeIdx, split_indices)
    g1_to_g2 = np.concatenate([
        corners1['0'], # first corner
        parts[0], # first edge
        corners1['A'], # corner after first edge
        parts[1], # both transverse edges
        corners1['B'], # corner BEFORE last edge (opposing semi-infinite)
        parts[2], # opposing semi-infinite edge is the last
        corners1['A+B'], # the last corner -- the corner of the largest indices (the last atoms)
    ])
    return g1_to_g2