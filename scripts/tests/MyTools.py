import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from ase.visualize import view
from ase.visualize.plot import plot_atoms

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
    r = np.round(np.sqrt(x * x + y * y), decimals=3)
    unique_r = np.unique(r)
    ang = np.arctan2(y, x)
    order = np.lexsort((ang[:],r[:]),axis=-1)
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