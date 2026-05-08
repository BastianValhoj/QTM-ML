import numpy as np
from scipy.spatial import cKDTree
import sisl

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
        raise ValueError("## NC must be >= 0")
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
    return np.concat(corners_atom_match).astype(int)

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
        raise ValueError("We will only allow tiles1 being integer corresponding to tiling the same amount along vectors A and B")
        N1a, N1b = tiles1
    else:
        N1a = N1b = tiles1
    
    if isinstance(tiles2, (tuple, list)):
        raise ValueError("We will only allow tiles2 being integer corresponding to tiling the same amount along vectors A and B")
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
    
    # print(f"cent1 frac : {cent1_frac}")
    # print(f"cent2 frac : {cent2_frac}")
    # print()
    # print(f"edges2 : {edges2}")
    
    tree = cKDTree(cent1_frac)
    _, edge_idx1_to_idx2 = tree.query(cent2_frac, k=1)
    # return edges1.reshape(-1, na), edge_idx1_to_idx2
    
    # get centroids
    edges1 = get_edges(N1a, N1b, na, NC)
    edges2 = get_edges(N2a, N2b, na, NC)
    
    # cent1 = get_centers(g1.xyz[edges1], na)
    # cent2 = get_centers(g2.xyz[edges2], na)
    # 
    # # get fractional centroids
    # cent1_frac = get_fractional(g1, cent1)
    # cent2_frac = get_fractional(g2, cent2)
    
    # CORRECTION we will only support cases where Na=Nb so each side is equal length
    n1_per_side = edges1.shape[0] // 4  # number of atoms per side in small
    n2_per_side = edges2.shape[0] // 4  # number of atoms per side in big
    # print("centroids per side:", n1_per_side)
    # print(n2_per_side)
    # print("atoms in small edge:", edges1)
    
    # match only using kNN to centroids ON THE SAME EDGE!! otherwise some combination of tiles wrongfully match indices
    edge_idx1_to_idx2 = np.empty(edges2.shape[0] // na, dtype=int)
    for side in range(4):
        e1_side  = edges1[side*n1_per_side : (side+1)*n1_per_side]
        e2_side  = edges2[side*n2_per_side: (side+1)*n2_per_side]
        print(f"small edge atoms {side}:", e1_side)
        print(f"big edges atoms  {side}:", e2_side)
        cent1_side = get_centers(g1.xyz[e1_side.flatten()], na)
        cent2_side = get_centers(g2.xyz[e2_side.flatten()], na)
        # print(f"centroids 2 in edge {side}", cent1_side)
        # print(f"centroids 2 in edge {side}", cent2_side)
        # print(cent2_side)
        cent1_frac = get_fractional(g1, cent1_side)
        cent2_frac = get_fractional(g2, cent2_side)
        print("centroid frac small:", cent1_frac)
        print("centroid frac big  :", cent2_frac)
        tree = cKDTree(cent1_frac)
        _, idx = tree.query(cent2_frac, k=1)
        # print("idx")
        # print(idx[0])
        edge_idx1_to_idx2[side*n2_per_side:(side+1)*n2_per_side] = idx[0] + side*n1_per_side
    
    # tree = cKDTree(cent1_frac)
    # _, edge_idx1_to_idx2 = tree.query(cent2_frac, k=1)
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
    # print(f"len(edge_idx1_to_idx2) : {len(edge_idx1_to_idx2)}")
    # print(f"edge_idx1_to_idx2 : {edge_idx1_to_idx2}")
    # print(f"corners1 : {corners1}")
    # print(f"Number of corner parts : {len(corner_parts)}")
    # for i, p in enumerate(corner_parts):
    #     print(f"  corner_parts[{i}] : {p}")
    
    edge1_to_edges2 = edges1[edge_idx1_to_idx2]
    edge_parts = _splitter(edge1_to_edges2.flatten(), na)
    
    # print(f"Number of edge parts: {len(edge_parts)}")
    # for i, p in enumerate(edge_parts):
    #     print(f"  edge_parts[{i}]: {p}")
    
    
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
            # corner_parts[4],
        ])
    if NC>0: # if NC=0 no 5'th element in the corner_parts, otherwise will be the last 'additional'/NC-corner for the starting postion.
        g1_to_g2_idx = np.concat([g1_to_g2_idx, corner_parts[4]])
    # else:
        # print("NC = 0, so no additional extra NC-corner")
    return g1_to_g2_idx


def rsse_to_edge(rsse: sisl.Hamiltonian,
                 edge: sisl.Geometry) -> np.ndarray:
    if not hasattr(rsse, 'xyz'):
        raise AttributeError("`rsse` must have .xyz attribute")
    if not hasattr(edge, 'xyz'):
        raise AttributeError("`edge` must have .xyz attribute")
    
    if edge.na < rsse.na:
        raise AssertionError("Make sure that the edge has at least the same amount of atoms as rsse")
        
    # the coords available to target
    tree = cKDTree(edge.xyz)
    
    # what to match
    dd, ii = tree.query(rsse.xyz, k=1)
    
    return ii

def rsse_mapping(rsse1 : sisl.Hamiltonian, 
                 rsse2 : sisl.Hamiltonian, 
                 geom1 : sisl.Geometry, 
                 geom2 : sisl.Geometry, 
                 tiles1 : int | tuple[int, int], 
                 tiles2 : int | tuple[int, int],
                 na=6, 
                 NC=1, 
                 *, 
                 ret_parts=False
                 ) -> dict[int, int] | tuple[dict[int, int], dict[int, int], dict[int, int]]:
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
