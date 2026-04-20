import numpy as np
from scipy.spatial import cKDTree

def get_fractional(geometry, coords=None):
    """Map Cartesion coordinates to fractional coordinates, 
    between 0 and 1 representing position relative to unitcell vertices."""
    geom = geometry.copy()
    Inv = geom.icell.T[:2, :2]
    if coords is not None:
        coords = coords[:, :2]
    else:
        coords = geom.xyz[:, :2]
    # elif (coords is not None) and (coords.shape[1] == 3):
    #     coords = geom.xyz[:, :2]
    frac = coords @ Inv
    return frac / np.max(frac, axis=0, keepdims=True)


def _tile_indices(n1: int, n2: int) -> tuple[int, int, int, int]:
    return 2*(n1+n2)-4, n1-1, n1+n2-2, 2*n1+n2-3

def get_corners(tile1, tile2=None, na=6, NC=1):
    if NC < 0:
        raise ValueError("## NC must be > 0")
    if tile2 is None:
        tile2 = tile1
    BL, BR, TR, TL = _tile_indices(tile1, tile2)
    corners_atom_match = [
        range(0, na*(NC + 1)), # bottom left hex + 1*NC
        range(na*(BR - NC), na*(BR + NC + 1)), # bottom right hex + 2*NC
        range(na*(TR - NC), na*(TR + NC + 1)), # top right hex + 2*NC
        range(na*(TL - NC), na*(TL + NC + 1)), # top left hex + 2*NC
        range(na*(BL - NC), na*BL) # bottom left 1*NC
    ]
    return np.concat(corners_atom_match)

def map_corners(tiles1, tiles2, na=6, NC=1):
    if not isinstance(tiles1, (tuple, list)):
        N1a, N1b = (tiles1, None)
    if not isinstance(tiles2, (tuple, list)):
        N2a, N2b = (tiles2, None)
    corners_match1 = get_corners(N1a, N1b, na, NC=NC)
    corners_match2 = get_corners(N2a, N2b, na, NC=NC)
    mapping = {}
    for i1, i2 in zip(corners_match1.flatten(), corners_match2.flatten()):
        mapping[int(i1)] = int(i2)
    return mapping

def get_edges(tile1, tile2, na=6, NC=1):
    if (NC < 0) or (2*NC > tile1-2) or (2*NC > tile2-2):
        raise ValueError("## NC must be > 0 and below (tile1-2)/2 and (tile2-2)/2")
    if tile2 is None:
        tile2 = tile1
    BL, BR, TR, TL = _tile_indices(tile1, tile2)
    corners_atom_match = [
        range(na*(1+NC), na*(BR-NC)),
        range(na*(BR+NC+1), na*(TR-NC)),
        range(na*(TR+NC+1), na*(TL-NC)),
        range(na*(TL+NC+1), na*(BL-NC)),
    ]
    return np.concat(corners_atom_match)

def get_centers(coords, na):
    rolling_mean = []
    for i in range(0, len(coords), na):
        roll = coords[i:i+na].mean(axis=0)
        rolling_mean.append(roll)
    return np.array(rolling_mean)



def map_edges(g1, g2, tiles1, tiles2, na=6, NC=1):
    if isinstance(tiles1, (tuple, list)):
        N1a, N1b = tiles1
    else:
        N1a = N1b = tiles1
    
    if isinstance(tiles2, (tuple, list)):
        N2a, N2b = tiles2
    else:
        N2a = N2b = tiles2
        
    
    # get centroids
    edges1 = get_edges(N1a, N1b, na, NC)
    edges2 = get_edges(N2a, N2b, na, NC)
    cent1 = get_centers(g1.xyz[edges1], na)
    cent2 = get_centers(g2.xyz[edges2], na)
    
    # get fractional centroids
    cent1_frac = get_fractional(g1, cent1)
    cent2_frac = get_fractional(g2, cent2)
    
    tree = cKDTree(cent1_frac)
    _, edge_idx1_to_idx2 = tree.query(cent2_frac, k=1)
    return edges1.reshape(-1, na), edge_idx1_to_idx2

def _splitter(arr, diff_tol):
    diff = np.diff(arr)
    split_idx = np.where(np.abs(diff) >= diff_tol)[0] + 1
    parts = np.split(arr, split_idx)
    return parts

def extrapolate(g1, g2, tiles1, tiles2, na=6, NC=1):
    if isinstance(tiles1, (tuple, list)):
        N1a, N1b = tiles1
    else:
        N1a = N1b = tiles1
        
    # get edges and conversion of idx1 to idx2
    edges1, edge_idx1_to_idx2 = map_edges(g1, g2, tiles1, tiles2, na, NC)
    
    # get corners
    corners1 = get_corners(N1a, N1b, na, NC)
    corner_parts = _splitter(corners1, na)
    
    edge1_to_edges2 = edges1[edge_idx1_to_idx2]
    edge_parts = _splitter(edge1_to_edges2.flatten(), na)
    
    
    # append to final list
    g1_to_g2_idx = np.concat([
            corner_parts[0],
            edge_parts[0],
            corner_parts[1],
            edge_parts[1],
            corner_parts[2],
            edge_parts[2],
            corner_parts[3],
            edge_parts[3],
            corner_parts[4],
        ])
    return g1_to_g2_idx


def rsse_to_edge(rsse, edge):
    if not hasattr(rsse, 'xyz'):
        raise AttributeError("`rsse` must have .xyz attribute")
    if not hasattr(edge, 'xyz'):
        raise AttributeError("`edge` must have .xyz attribute")
        
    # the coords available to target
    tree = cKDTree(edge.xyz)
    
    # what to match
    dd, ii = tree.query(rsse.xyz, k=1)
    
    return ii

def rsse_mapping(rsse1, rsse2, geom1, geom2, tiles1, tiles2, na=6, NC=1, *, ret_parts=False):
    edge1_to_rsse1 = {int(edgeidx):int(rsseidx) for rsseidx, edgeidx  in enumerate(rsse_to_edge(rsse1, geom1))}
    rsse2_to_edge2 = {int(rsseidx):int(edgeidx) for rsseidx, edgeidx in enumerate(rsse_to_edge(rsse2, geom2))}

    # convert the g2edge index to the g1edge index
    edge2_to_edge1 = {int(g2idx):int(g1idx) for g2idx, g1idx in enumerate(extrapolate(geom1, geom2, tiles1, tiles2, na=na, NC=NC))}
    if ret_parts:
        return rsse2_to_edge2, edge2_to_edge1, edge1_to_rsse1
    else:
        keys = rsse2_to_edge2.keys()
        values = [edge1_to_rsse1[edge2_to_edge1[rsse2_to_edge2[key]]] 
                  for key in keys
                  ]
        return {int(k):int(val) for k,val in zip(keys,values)}
