import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from ase.visualize import view
from ase.visualize.plot import plot_atoms
from scipy.spatial import cKDTree
from scipy.linalg import block_diag



def match_atoms_by_xyz(big, small, tol=1e-6, center = True, check_Z=False):
    """
    Match atoms in `small` to atoms in `big` by coordinates.

    Parameters
    ----------
    big : sisl.Geometry
        The large geometry.
    small : sisl.Geometry
        The geometry assumed to be contained in `big`.
    tol : float
        Matching tolerance in Cartesian coordinates.
    center : bool
        If True, center geometries before matching
    check_Z : bool
        If True, also require matching atomic species.

    Returns
    -------
    idx_big : np.ndarray
        Indices in `big` corresponding to atoms in `small`,
        in the same order as atoms in `small`.
    """

    xyz_big = np.asarray(big.xyz)
    xyz_small = np.asarray(small.xyz)

    if center:
        xyz_big = np.asarray(big.xyz) - big.center()
        xyz_small = np.asarray(small.xyz) - small.center()


    tree = cKDTree(xyz_big)
    dist, idx_big = tree.query(xyz_small, distance_upper_bound=tol)

    # cKDTree returns idx == len(big) when no match is found
    bad = (idx_big >= len(xyz_big)) | np.isinf(dist)
    if np.any(bad):
        raise ValueError(
            f"{bad.sum()} atoms in `small` could not be matched in `big` within tol={tol}"
        )

    # ensure one-to-one matching
    if len(np.unique(idx_big)) != len(idx_big):
        raise ValueError(
            "Non-unique match found: multiple atoms in `small` map to the same atom in `big`."
        )

    if check_Z:
        Z_big = np.array([a.Z for a in big.atoms])
        Z_small = np.array([a.Z for a in small.atoms])
        badZ = Z_big[idx_big] != Z_small
        if np.any(badZ):
            raise ValueError(
                f"{badZ.sum()} matched atoms have different atomic numbers."
            )

    return idx_big

def geometry_complement(big, small, tol=1e-6, check_Z=False):
    idx_remove = match_atoms_by_xyz(big, small, tol=tol, check_Z=check_Z)
    idx_keep = np.setdiff1d(np.arange(big.na), idx_remove)
    return big.sub(idx_keep), idx_remove, idx_keep


def plotxy(geom):
    fig, ax = plt.subplots()
    atoms_plot = plot_atoms(geom.to.ase(), ax=ax)
    circles = fig.findobj(match=patches.Circle)
    xminp,yminp = np.min(np.array([c.center for c in circles]),axis=0)
    rav = np.mean(np.array([c.radius for c in circles]),axis=0)
    x,y = geom.xyz.T[0:2]
    xmin = min(x)
    ymin = min(y)
    ax.set_aspect('equal')
    plt.show()

def plot_with_numbers(geom):
    fig, ax = plt.subplots()
    atoms_plot = plot_atoms(geom.to.ase(), ax=ax)
    circles = fig.findobj(match=patches.Circle)
    xminp,yminp = np.min(np.array([c.center for c in circles]),axis=0)
    rav = np.mean(np.array([c.radius for c in circles]),axis=0)
    x,y = geom.xyz.T[0:2]
    xmin = min(x)
    ymin = min(y)
    dx,dy = xminp - xmin, yminp - ymin   
    for i,xyz in enumerate(geom.xyz):
        ax.text(x[i]+dx-0.5*rav,y[i]+dy, str(i), fontsize=8,
            #ha='center',va='center',
            bbox=dict(facecolor='white',alpha=0.7,edgecolor='none', pad=0.1))
        #print("center (in data coords):", c.center)
        #print("radius:", c.get_radius())
    ax.set_aspect('equal')
    plt.show()

def radial_angle_sort(geom):
    c = geom.center()
    pos = np.asarray(geom.xyz)[:, :2] - np.asarray(c)[:2]
    x = pos[:, 0]
    y = pos[:, 1]
    r = np.sqrt(x * x + y * y)
    ang = np.arctan2(x, y)
    #ang = (ang + 2 * np.pi) % (2 * np.pi)
    order = np.lexsort((ang, r))
    return geom.sub(order)



def coord_sort(geom, mode='xy', shell_width_r2=None, return_geom=False):
    """Sort atoms starting from center, then by radius shells and coordinates.

    Parameters
    ----------
    geom : sisl.Geometry
        Must have xyz attribute and center() method
    mode : {'xy', 'angle'}
        'xy': within each shell, sort by (x, y)
        'angle': within each shell, sort by angle from x-axis starting from leftmost
    shell_width_r2 : float or None
        If provided, group atoms into shells by floor(r^2 / shell_width_r2).
        If None, group by unique r^2 values (previous behavior).
    return_geom : bool, optional
        If True, attempt to return reordered geometry

    Returns
    -------
    order_or_geom : np.ndarray or sisl.Geometry
        Sorted indices or reordered geometry if return_geom=True
    """
    # positions relative to center
    c = geom.center()
    pos = np.asarray(geom.xyz)[:, :2] - np.asarray(c)[:2]
    x = pos[:, 0]
    y = pos[:, 1]
    r2 = x * x + y * y  # squared radius

    # Determine shells
    if shell_width_r2 is None:
        # use unique radii (rounded to handle numerical precision)
        shell_keys = np.unique(np.round(r2, 8))
        # we'll iterate over these r2 values directly
        by_shell = [(val, np.abs(r2 - val) < 1e-8) for val in shell_keys]
    else:
        if shell_width_r2 <= 0:
            raise ValueError('shell_width_r2 must be positive')
        shell_id = np.floor(r2 / float(shell_width_r2)).astype(int)
        unique_ids = np.unique(shell_id)
        by_shell = [(sid, shell_id == sid) for sid in unique_ids]

    # Sort shells in ascending order (center first)
    order = []

    # Process each shell
    for key, mask in by_shell:
        idx_at_r = np.where(mask)[0]
        if idx_at_r.size == 0:
            continue

        if mode == 'xy':
            # Sort by x then y within this shell
            idx_at_r = idx_at_r[np.lexsort((y[idx_at_r], x[idx_at_r]))]
        elif mode == 'angle':
            # Compute angles for points at this shell
            ang = np.arctan2(y[idx_at_r], x[idx_at_r])
            # Find leftmost atom as reference (min x)
            leftmost_pos = np.argmin(x[idx_at_r])
            base_ang = ang[leftmost_pos]
            # Adjust angles to start from leftmost (zero)
            adj_ang = np.mod(ang - base_ang, 2 * np.pi)
            idx_at_r = idx_at_r[np.argsort(adj_ang)]
        else:
            raise ValueError("mode must be 'xy' or 'angle'")

        order.extend(idx_at_r.tolist())

    order = np.array(order, dtype=int)

    if return_geom:
        try:
            return geom.sub(order)
        except Exception as e:
            print('Warning: geom.sub(order) failed, returning indices. Exception:', e)
            return order
    return order

def write_fdf_device_indices_grouped(indices, output_file="TBT_Atoms_Device.fdf"):
    # Group consecutive indices into sequences
    # Input: indices in sisl (0-based)
    # Writes: indices in siesta/fdf (1-based!)
    sequences = []
    current_sequence = [indices[0]]

    for i in range(1, len(indices)):
        if indices[i] == indices[i - 1] + 1:  # Check if the current index is consecutive
            current_sequence.append(indices[i])
        else:
            sequences.append(current_sequence)
            current_sequence = [indices[i]]

    # Append the last sequence
    sequences.append(current_sequence)
    # Convert sequences into FDF-compatible ranges
    fdf_lines = []
    for seq in sequences:
        if len(seq) > 1:
            fdf_lines.append(f"  atom [{seq[0]+1} -- {seq[-1]+1}]")
        else:
            fdf_lines.append(f"  atom {seq[0]+1}")

    # Combine into the FDF block
    fdf_block = "%block TBT.Atoms.Device\n" + "\n".join(fdf_lines) + "\n%endblock"

    # Print the FDF block
    print(fdf_block)
    # Specify the output file name
    #output_file = "TBT_Atoms_Device.fdf"

    # Write the FDF block to the file
    with open(output_file, "w") as f:
        f.write(fdf_block)
    print(f"FDF block written to {output_file}")


def plot_sigma_imaginary(geom, sigma, size_scale=100, cmap='RdBu_r'):
    """Plot imaginary part of sigma matrix as circles at atomic positions.
    
    Parameters
    ----------
    geom : sisl.Geometry
        Geometry with xyz coordinates
    sigma : np.ndarray
        Complex sigma matrix (can be 2D or higher dimensional)
    size_scale : float, optional
        Scaling factor for circle sizes (default: 100)
    cmap : str, optional
        Colormap name (default: 'RdBu_r')
    
    Returns
    -------
    fig, ax : matplotlib figure and axes objects
    """
    # Extract imaginary part
    sigma_imag = np.imag(sigma)
    
    # Handle different sigma dimensionalities - flatten to 1D if needed
    if sigma_imag.ndim > 1:
        # For diagonal elements or mean across all elements
        if sigma_imag.shape[0] == sigma_imag.shape[1]:
            sigma_imag = np.diag(sigma_imag)
        else:
            sigma_imag = sigma_imag.flatten()
    
    # Ensure we have the right number of values
    if len(sigma_imag) != len(geom):
        raise ValueError(f"sigma size {len(sigma_imag)} doesn't match geom size {len(geom)}")
    
    # Extract x, y positions
    x = geom.xyz[:, 0]
    y = geom.xyz[:, 1]
    
    # Normalize for visualization
    sigma_abs = np.abs(sigma_imag)
    sigma_norm = sigma_abs / np.max(sigma_abs) if np.max(sigma_abs) > 0 else sigma_abs
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot circles
    scatter = ax.scatter(x, y, s=sigma_norm * size_scale, c=sigma_imag, 
                         cmap=cmap, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Im(σ)', rotation=270, labelpad=20)
    
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Imaginary part of σ matrix')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig, ax


def _atomic_numbers_from_geometry(geom):
    """Return per-atom atomic numbers for a sisl.Geometry."""
    atoms = getattr(geom, 'atoms', None)
    if atoms is not None:
        for attr in ('Z', 'numbers'):
            if hasattr(atoms, attr):
                z = np.asarray(getattr(atoms, attr))
                if z.ndim == 1 and len(z) == len(geom):
                    return z.astype(int)
    z = np.empty(len(geom), dtype=int)
    for i in range(len(geom)):
        z[i] = int(geom.atoms[i].Z)
    return z


def _norm_frac_positions(geom):
    """Atom positions in fractional cell coordinates, normalized to [0, 1].

    No periodic boundary conditions: raw Cartesian positions are projected
    onto the cell vectors and then scaled so that the structure spans [0, 1]
    along each axis. This makes corners map to (0,0), (1,0), (0,1), (1,1)
    regardless of the absolute structure size.
    """
    frac = np.asarray(geom.xyz) @ np.linalg.inv(geom.cell).T  # (N, 3)
    lo, hi = frac.min(axis=0), frac.max(axis=0)
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    return (frac - lo) / span


def _neighbor_dist_score(xyz1, i1, xyz2, i2, R):
    """Mismatch between sorted neighbor distances of atom i1 (in xyz1) and i2 (in xyz2).

    No periodic boundary conditions.
    """
    def nbr_dists(xyz, idx):
        d = np.linalg.norm(xyz - xyz[idx], axis=1)
        return np.sort(d[(d > 1e-12) & (d <= R)])

    a = nbr_dists(xyz1, i1)
    b = nbr_dists(xyz2, i2)
    n = min(len(a), len(b))
    if n == 0:
        return float(abs(len(a) - len(b)))
    paired = float(np.sum(np.abs(a[:n] - b[:n]) / np.maximum(0.05, (a[:n] + b[:n]) / 2)))
    return paired + float(abs(len(a) - len(b)))


def find_most_similar_atom_environment(s1, s2, i1, R=5.0, return_details=False):
    """Find atom i2 in s2 whose environment best matches atom i1 in s1.

    Matching uses two criteria, with position dominating:
    1. Distance in normalized fractional coordinates (no PBC): corner/edge atoms
       map correctly to corner/edge atoms regardless of structure size or tiling.
    2. Sorted neighbor-distance mismatch up to R as a tiebreaker.

    Parameters
    ----------
    s1, s2         : sisl.Geometry
    i1             : int, atom index in s1
    R              : float, neighbor cutoff in Å (default 5.0)
    return_details : bool

    Returns
    -------
    i2      : int
    details : dict (only when return_details=True)
    """
    z1 = _atomic_numbers_from_geometry(s1)
    z2 = _atomic_numbers_from_geometry(s2)

    nf1 = _norm_frac_positions(s1)   # (N1, 3)
    nf2 = _norm_frac_positions(s2)   # (N2, 3)
    xyz1 = np.asarray(s1.xyz)
    xyz2 = np.asarray(s2.xyz)

    cand = np.where(z2 == z1[i1])[0]
    if cand.size == 0:
        raise ValueError('No atoms of the same type in s2.')

    # Primary score: distance in normalized fractional xy-space
    pos_dist = np.linalg.norm(nf2[cand, :2] - nf1[i1, :2], axis=1)

    # Tiebreaker: neighbor-shell distance mismatch
    env_score = np.array([_neighbor_dist_score(xyz1, i1, xyz2, int(c), R) for c in cand])
    env_norm = env_score / max(env_score.max(), 1e-12)

    total = pos_dist + 0.1 * env_norm

    order = np.argsort(total)
    best_i2 = int(cand[order[0]])

    if not return_details:
        return best_i2

    top = [(int(cand[k]), float(total[k]), float(pos_dist[k]), float(env_score[k]))
           for k in order[:10]]
    return best_i2, {
        'top_candidates':   top,
        'best_total_cost':  float(total[order[0]]),
        'best_pos_dist':    float(pos_dist[order[0]]),
        'best_env_score':   float(env_score[order[0]]),
        'R': float(R),
    }