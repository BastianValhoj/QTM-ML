import marimo

__generated_with = "0.23.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Extrapolate real-space surrounding self-energy
    From small $N_1\times N_1$ tiling to bigger $N_2\times N_2$ tiling
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %load_ext autoreload
    # '%autoreload 2' command supported automatically in marimo
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    #import matplotlib.tri as tri
    import sisl as si
    import os
    from tqdm.auto import tqdm
    import time
    from scipy.spatial import cKDTree
    from scipy.linalg import block_diag
    from pathlib import Path

    return cKDTree, np, plt, si, tqdm


@app.cell
def _():
    from MyTools import match_atoms_by_xyz

    return (match_atoms_by_xyz,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The simple cell: Make sure this has a size > interaction range!
    Setup Armchair graphene lattice
    """)
    return


@app.cell
def _(np, si):
    a_cc = 1.42
    sq3 = np.sqrt(3.0)
    # 6-atom graphene cell (one hexagon)
    xyz = [[a_cc*np.cos(t),a_cc*np.sin(t),20.] for t in 2.*np.pi*np.arange(0,6)/6]
    # Lattice vectors
    cell = np.array([
        [3.0 * a_cc,              0.0,           0.0],
        [1.5 * a_cc,   1.5 * sq3 * a_cc,         0.0],
        [0.0,                     0.0,          20.0],  # vacuum
    ])
    graphene6 = si.Geometry(
        xyz,
        atoms=si.Atom("C"),
        lattice=si.Lattice(cell)
    )
    graphene6.set_nsc([3,3,1])
    return (graphene6,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Initial Hamiltonian
    """)
    return


@app.cell
def _(graphene6, si):
    # set up electrode structure and TB hamiltonian
    g_elecL = graphene6.copy() ## Use block expansion 
    H_elecL = si.Hamiltonian(g_elecL)
    #r = (0.1, 1.44, 2.5)
    #t = (0.0, -2.7, 0.001)
    r = (0.1, 1.44)
    t = (0.0, -2.7)
    H_elecL.construct([r,t])
    gu = H_elecL.geometry
    nau = gu.na
    return H_elecL, gu, nau


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Make a edge "frame" of the simple cell and the RSSE structure for system 1,2
    """)
    return


@app.function
def make_edge(geom,N0,N1):
    ## Make N0xN1 edge tile
    grev = geom.copy()
    grev.cell[0] = -grev.cell[0]
    grev.cell[1] = -grev.cell[1]
    gBot = geom.tile(N0-1,0)
    gRig = geom.tile(N1-1,1).move((N0-1)*geom.cell[0])
    gTop = grev.tile(N0-1,0).move((N1-1)*geom.cell[1]+(N0-1)*geom.cell[0])
    gLeft = grev.tile(N1-1,1).move((N1-1)*geom.cell[1])
    gtot = gBot + gRig + gTop + gLeft
    gtot.cell[0] = geom.cell[0]*N0
    gtot.cell[1] = geom.cell[1]*N1
    return gtot


@app.cell
def _(H_elecL, plot_with_numbers):
    plot_with_numbers(H_elecL)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Here we put the $N_1$ and $N_2$ and define RSSE's
    """)
    return


@app.cell
def _(np):
    N1 = 9; N2 = 12;
    Emin = -3.;Emax=3; dE=0.1; eta = 0.01
    En = np.arange(Emin, Emax, dE)
    return En, N1, N2, eta


@app.cell
def _(H_elecL, N1, N2, eta, si):
    RSSE1 = si.RealSpaceSE(H_elecL, semi_axis=0, k_axes=1, unfold=(N1, N1, 1),eta = eta)
    RSSE2 = si.RealSpaceSE(H_elecL, 0, 1, (N2, N2, 1), eta = eta)
    return RSSE1, RSSE2


@app.cell
def _(RSSE1, RSSE2):
    H_rs_elec1, rs_elec_indices1 = RSSE1.real_space_coupling(ret_indices=True)
    H_rs_elec2, rs_elec_indices2 = RSSE2.real_space_coupling(ret_indices=True)
    return rs_elec_indices1, rs_elec_indices2


@app.cell
def _(H_elecL, N1, N2, gu, rs_elec_indices1, rs_elec_indices2):
    g1 = H_elecL.geometry.tile(N1,0).tile(N1,1)
    g1edge =  make_edge(gu,N1,N1)
    g2 = H_elecL.geometry.tile(N2,0).tile(N2,1)
    g2edge = make_edge(gu, N2,N2)
    g1rsse = g1.sub(rs_elec_indices1)
    g2rsse = g2.sub(rs_elec_indices2)
    return g1edge, g1rsse, g2edge, g2rsse


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###  RSSE "frame" to edge matching (using centered xyz coordinates)
    """)
    return


@app.cell
def _(g1edge, g1rsse, g2edge, g2rsse, match_atoms_by_xyz):
    ## Match atoms between tiled edge (bigger) and rsse edge (contained in edge)
    g1frames_match = match_atoms_by_xyz(g1edge, g1rsse, tol=1e-6, center = True, check_Z=False)
    g1rsse_to_g1edge = {int(i1): int(i2) for i1, i2 in enumerate(g1frames_match)}
    g1edge_to_g1rsse = {int(i2): int(i1) for i1, i2 in enumerate(g1frames_match)}

    g2frames_match = match_atoms_by_xyz(g2edge, g2rsse, tol=1e-6, center = True, check_Z=False)
    g2rsse_to_g2edge = {int(i1): int(i2) for i1, i2 in enumerate(g2frames_match)}
    g2edge_to_g2rsse = {int(i2): int(i1) for i1, i2 in enumerate(g2frames_match)}
    len(g1frames_match),g1edge.na,g1rsse.na
    return g1edge_to_g1rsse, g2rsse_to_g2edge


@app.cell
def _(g1rsse, g2rsse, plot_with_numbers):
    plot_with_numbers(g1rsse); plot_with_numbers(g2rsse)
    return


@app.cell
def _(g1edge, g2edge, plot_with_numbers):
    plot_with_numbers(g1edge);plot_with_numbers(g2edge)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Edge-edge matching
    Here we define a NC which is the length of the corners to match 1-1 (NC=0 is only matching corner cells, NC=1 incl. one additional cell around the corner, etc.)
    """)
    return


@app.cell
def _(N1, N2, g2edge, nau, np):
    ## We match first repeated primitive (gu) surrounding cells
    ## Corner (bottom, top, left, right) cells for structure 1:
    BL,BR,TR,TL = 0,(N1-1),2*(N1-1),3*(N1-1)
    ## Corner cells for structure 2:
    BL2,BR2,TR2,TL2 =0,(N2-1),2*(N2-1),3*(N2-1)
    ## Lookup:
    NC=1 ## number of cells to match around a corner

    corners = [[range(BL,BL+NC+1),range(BL2,BL2+NC+1)],
                   [range(BR-NC,BR),range(BR2-NC,BR2)],[range(BR,BR+NC+1),range(BR2,BR2+NC+1)],
                    [range(TR-NC,TR),range(TR2-NC,TR2)],[range(TR,TR+NC+1),range(TR2,TR2+NC+1)],
                     [range(TL-NC,TL),range(TL2-NC,TL2)],[range(TL,TL+NC+1),range(TL2,TL2+NC+1)],
                     [range(TL+N1-NC-1,TL+N1-1),range(TL2+N2-NC-1,TL2+N2-1)]
                   ]
    corners_atom_match1 = np.concatenate([np.arange(nau*cm[0].start,nau*cm[0].stop) for cm in corners])
    corners_atom_match2 = np.concatenate([np.arange(nau*cm[1].start,nau*cm[1].stop) for cm in corners])

    sides = [[range(BL+NC+1,BR-NC),range(BL2+NC+1,BR2-NC)],
                 [range(BR+NC+1,TR-NC),range(BR2+NC+1,TR2-NC)],[range(TR+NC+1,TL-NC),range(TR2+NC+1,TL2-NC)],[range(TL+NC+1,TL+N1-1-NC),range(TL2+NC+1,TL2+N2-1-NC)]]

    ## "Interpolate sides"
    sidematch = []
    for sm in sides:
        #print(sm)
        for i2 in sm[1]:
            ii2 = sm[0].start + (sm[0].stop-sm[0].start)*(i2-sm[1].start)/(sm[1].stop - sm[1].start) ##!!!
            #print(i2,int(ii2))
            sidematch.append([int(ii2),i2]) ## cell index small, big
    side_atom_match1 = np.concatenate([np.arange(nau*sm[0],nau*(sm[0]+1)) for sm in sidematch])
    side_atom_match2 = np.concatenate([np.arange(nau*sm[1],nau*(sm[1]+1)) for sm in sidematch])

    print("Check:,",len(corners_atom_match1)==(1+2*NC)*4*nau,len(side_atom_match1)==(N2-2*(NC+1))*4*nau)
    (1+2*NC)*4*nau + (N2-2*(NC+1))*4*nau, g2edge.na
    return (
        corners_atom_match1,
        corners_atom_match2,
        side_atom_match1,
        side_atom_match2,
    )


@app.cell
def _(
    corners_atom_match1,
    corners_atom_match2,
    plt,
    side_atom_match1,
    side_atom_match2,
):
    plt.plot(corners_atom_match1,corners_atom_match2,'o');plt.plot(side_atom_match1,side_atom_match2,'x')
    return


@app.cell
def _(
    N1,
    N2,
    corners_atom_match1,
    corners_atom_match2,
    g1edge,
    g2edge,
    nau,
    np,
    side_atom_match1,
    side_atom_match2,
):
    match_all = np.array([np.concatenate([corners_atom_match1,side_atom_match1]),np.concatenate([corners_atom_match2,side_atom_match2])]).T
    s2_to_s1 = {int(i2): int(i1) for i1, i2 in match_all}
    g2edge_to_g1edge = s2_to_s1
    print("Check: ", max(match_all.T[0])+1,g1edge.na, 4*(N1-1)*nau)
    print("Check: ", max(match_all.T[1])+1,g2edge.na, 4*(N2-1)*nau)
    return g2edge_to_g1edge, s2_to_s1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### By-eye check
    """)
    return


@app.cell
def _(g1edge, g2edge, np, plt, s2_to_s1):
    s1 = g1edge.copy()
    s2 = g2edge.copy()
    s1_xyz = np.asarray(s1.xyz)
    # Visual check: s2 atoms labeled by their matched s1 indices, with s1 overlaid in center
    s2_xyz = np.asarray(s2.xyz)
    s1_center = s1_xyz.mean(axis=0)[:2]
    s2_center = s2_xyz.mean(axis=0)[:2]
    # Find centers
    shift_to_center = s2_center - s1_center
    s1_xyz_centered = s1_xyz.copy()
    s1_xyz_centered[:, :2] = s1_xyz_centered[:, :2] + shift_to_center
    # Shift s1 to center of s2
    fig, ax = plt.subplots(figsize=(16, 14))
    ax.scatter(s2_xyz[:, 0], s2_xyz[:, 1], s=120, alpha=0.5, c='lightblue', edgecolors='navy', linewidth=0.5, label='s2 (large structure)', zorder=1)
    for i2_1 in range(len(s2_xyz)):
        i1_matched = s2_to_s1.get(i2_1, '?')
        ax.text(s2_xyz[i2_1, 0], s2_xyz[i2_1, 1], str(i1_matched), fontsize=8, ha='center', va='center', color='darkblue', weight='bold', bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.8, edgecolor='navy', linewidth=0.5), zorder=3)
    ax.scatter(s1_xyz_centered[:, 0], s1_xyz_centered[:, 1], s=200, alpha=0.7, c='red', edgecolors='darkred', linewidth=1.5, label='s1 (small structure, centered)', marker='s', zorder=2)
    # Plot s2 atoms as base (light background)
    for i1 in range(len(s1_xyz_centered)):
        ax.text(s1_xyz_centered[i1, 0], s1_xyz_centered[i1, 1], str(i1), fontsize=9, ha='center', va='center', color='darkred', weight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8, edgecolor='red', linewidth=1.0), zorder=4)
    ax.set_title('Visual Verification: s2 atoms labeled by matched s1 indices\n(red squares = s1 atoms centered; numbers = matching s1 index)', fontsize=14, weight='bold')
    # Label each s2 atom with its matched s1 index
    ax.set_xlabel('X (Å)', fontsize=12)
    ax.set_ylabel('Y (Å)', fontsize=12)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    # Overlay s1 in the center with distinctive markers
    # Label s1 atoms with their own indices
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Finally we match all the way from g2rsse to g1rsse:
    g2rsse --> g2egde --> g1edge --> g1rsse
    """)
    return


@app.cell
def _(g1edge_to_g1rsse, g2edge_to_g1edge, g2rsse_to_g2edge):
    def match_final(ir2):
        ie2 = g2rsse_to_g2edge.get(ir2)
        ie1 = g2edge_to_g1edge.get(ie2)
        ir1 = g1edge_to_g1rsse.get(ie1) 
        return ir1

    return (match_final,)


@app.cell
def _(g2rsse, match_final, np, plt):
    plt.plot(np.array([match_final(ir2) for ir2 in range(g2rsse.na)]))
    return


@app.cell
def _(g1rsse, g2rsse, match_final, np, plt):
    s1_1 = g1rsse.copy()
    s2_1 = g2rsse.copy()
    s1_xyz_1 = np.asarray(s1_1.xyz)
    # Visual check: s2 atoms labeled by their matched s1 indices, with s1 overlaid in center
    s2_xyz_1 = np.asarray(s2_1.xyz)
    s1_center_1 = s1_xyz_1.mean(axis=0)[:2]
    s2_center_1 = s2_xyz_1.mean(axis=0)[:2]
    # Find centers
    shift_to_center_1 = s2_center_1 - s1_center_1
    s1_xyz_centered_1 = s1_xyz_1.copy()
    s1_xyz_centered_1[:, :2] = s1_xyz_centered_1[:, :2] + shift_to_center_1
    # Shift s1 to center of s2
    fig_1, ax_1 = plt.subplots(figsize=(16, 14))
    ax_1.scatter(s2_xyz_1[:, 0], s2_xyz_1[:, 1], s=120, alpha=0.5, c='lightblue', edgecolors='navy', linewidth=0.5, label='s2 (large structure)', zorder=1)
    for i2_2 in range(len(s2_xyz_1)):
        i1_matched_1 = match_final(i2_2)
        ax_1.text(s2_xyz_1[i2_2, 0], s2_xyz_1[i2_2, 1], str(i2_2) + '>' + str(i1_matched_1), fontsize=8, ha='center', va='center', color='darkblue', weight='bold', bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.8, edgecolor='navy', linewidth=0.5), zorder=3)
    ax_1.scatter(s1_xyz_centered_1[:, 0], s1_xyz_centered_1[:, 1], s=200, alpha=0.7, c='red', edgecolors='darkred', linewidth=1.5, label='s1 (small structure, centered)', marker='s', zorder=2)
    # Plot s2 atoms as base (light background)
    for i1_1 in range(len(s1_xyz_centered_1)):
        ax_1.text(s1_xyz_centered_1[i1_1, 0], s1_xyz_centered_1[i1_1, 1], str(i1_1), fontsize=9, ha='center', va='center', color='darkred', weight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8, edgecolor='red', linewidth=1.0), zorder=4)
    ax_1.set_title('Visual Verification: s2 atoms labeled by matched s1 indices\n(red squares = s1 atoms centered; numbers = matching s1 index)', fontsize=14, weight='bold')
    # Label each s2 atom with its matched s1 index and its own index
    ax_1.set_xlabel('X (Å)', fontsize=12)
    ax_1.set_ylabel('Y (Å)', fontsize=12)
    ax_1.legend(fontsize=11, loc='upper left')  #i1_matched = 1
    ax_1.grid(True, alpha=0.3)
    ax_1.set_aspect('equal')
    plt.tight_layout()
    # Overlay s1 in the center with distinctive markers
    # Label s1 atoms with their own indices
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### RSSE matching:
    #### The idea: To obtain $\Sigma(A_1,A_2)$ between two atomic sites in the big structure where $\Delta R = R_1-R_2$ we first
    1) match $A_1$ with a "similar" atom in the small structure $a_1$.
    2) Then we find the atom $a_2$ in the small structure closest to $r_2 = r_1 + \Delta R$, and
    3) approximate $\Sigma(A_1,A_2)\approx \Sigma(a_1,a_2)$. (To symmetricize we also do the matching procedure starting with $A_2$ and average the two results for $\Sigma$).
    4) If $r_2$ does *not* belong to the small structure we put $\Sigma(A_1,A_2)=0$.
    """)
    return


@app.cell
def _(H_elecL, N1, RSSE1, eta, np, rs_elec_indices1):
    H1 = H_elecL.tile(N1,0).tile(N1,1)
    H1.set_nsc([1,1,1])
    def getGF1(e):
        SeHSE1 = RSSE1.self_energy(e, bulk=True, coupling=True)
        invGF = H1.Sk()*(e - 1j*eta) - H1.Hk()
        invGF = invGF.todense()
        invGF[np.ix_(rs_elec_indices1, rs_elec_indices1)] = SeHSE1
        GF = np.linalg.inv(invGF)
        return GF

    return H1, getGF1


@app.cell
def _(H_elecL, N2, RSSE2, eta, np, rs_elec_indices2):
    H2 = H_elecL.tile(N2,0).tile(N2,1)
    H2.set_nsc([1,1,1])
    def getGF2(e):
        SeHSE2 = RSSE2.self_energy(e, bulk=True, coupling=True)
        invGF = H2.Sk()*(e - 1j*eta) - H2.Hk()
        invGF = invGF.todense()
        invGF[np.ix_(rs_elec_indices2, rs_elec_indices2)] = SeHSE2
        GF = np.linalg.inv(invGF)
        return GF

    return H2, getGF2


@app.cell
def _(np):
    def get_center_atoms(g,R):  # select atoms within R of the center of the structure
        r0 = g.center()
        dR = np.linalg.norm(r0 - g.xyz, axis=1)
        sel = np.nonzero(dR<R)[0]
        return sel

    return (get_center_atoms,)


@app.cell
def _(H2, get_center_atoms):
    centeratoms = get_center_atoms(H2.geometry,5.) #
    return (centeratoms,)


@app.cell
def _(En, H1, getGF1, np, tqdm):
    dos1 = np.array([-np.trace(getGF1(e)).imag/(H1.na*np.pi) for e in tqdm(En)]) # dos per atom
    return


@app.cell
def _(En, centeratoms, getGF2, np, tqdm):
    dos2sel = np.array([-np.sum(np.diagonal(getGF2(e))[centeratoms]).imag/(len(centeratoms)*np.pi) for e in tqdm(En)]) # dos per atom
    return (dos2sel,)


@app.cell
def _():
    # dos2 = np.array([-np.trace(getGF2(e)).imag/(H2.na*np.pi) for e in tqdm(En)]) # dos per atom
    return


@app.cell
def _(En, dos2sel, plt):
    plt.plot(En,dos2sel,'o')
    # plt.plot(En,dos2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Match pairs and setup transformation  $\Sigma_1\rightarrow \Sigma_2$
    """)
    return


@app.cell
def _(cKDTree, g1rsse, g2rsse, match_final, np):
    tree = cKDTree(np.asarray(g1rsse.xyz))
    def MatchPair2to1(ir2,jr2):
        pairA = None
        pairB = None
        # first try match A1 -> A2
        ir1 = match_final(ir2) ## find jr1 from distance vector
        dR = g2rsse.xyz[jr2] - g2rsse.xyz[ir2]    
        Rjr1 = g1rsse.xyz[ir1] + dR
        dist, idx = tree.query(Rjr1)
        if dist < 0.1:
            jr1 = idx
            pairA = (ir1,jr1)
        # first try match A2 -> A1 (switch i,j)
        jr1 = match_final(jr2) ## find ir1 from distance vector
        dR = g2rsse.xyz[ir2] - g2rsse.xyz[jr2]
        Rir1 = g1rsse.xyz[jr1] + dR
        dist, idx = tree.query(Rir1)
        if dist < 0.1:
            ir1 = idx
            pairB = (ir1,jr1)
        return pairA,pairB

    return (MatchPair2to1,)


@app.cell
def _(g1rsse, g2rsse):
    g1rsse.na,g2rsse.na
    return


@app.cell
def _(g1rsse, g2rsse):
    n1 = g1rsse.na
    n2 = g2rsse.na
    return n1, n2


@app.cell
def _(n1, n2):
    n1*n1,n2*n2
    return


@app.cell
def _(MatchPair2to1, cKDTree, g1rsse, g2rsse, np, tqdm):
    from scipy.sparse import coo_matrix
    MaxRange = g1rsse.cell[0, 0]
    # Initialize a finder for neighbors that are within L1/2
    print('Max Range (Å) = ', MaxRange)
    #MaxRange = 10.
    tree_1 = cKDTree(np.asarray(g2rsse.xyz))
    xyz2 = g2rsse.xyz

    def build_operator_average(g1rsse, g2rsse):
        rows = []
        cols = []
        data = []
        n1 = g1rsse.na
        n2 = g2rsse.na
        xyz2 = g2rsse.xyz
        print('M1,M2,M1-M2-mapping: ', n1 * n1, n2 * n2, n1 * n1 * n2 * n2)
        for ir2 in tqdm(range(n2)):
            dist, idx = tree_1.query(xyz2[ir2], distance_upper_bound=MaxRange + 0.1, k=int(g1rsse.na))
            for jr2 in idx[dist < MaxRange]:
                k2 = ir2 * n2 + jr2
                pairA, pairB = MatchPair2to1(ir2, jr2)
                pairs = []
                if pairA is not None:
                    pairs.append(pairA)
                if pairB is not None:
                    pairs.append(pairB)
                if not pairs:
                    continue
                w = 1.0 / len(pairs)
                for i1, j1 in pairs:
                    k1 = i1 * n1 + j1
                    rows.append(k2)
                    cols.append(k1)
                    data.append(w)
        print(len(rows), len(cols), len(data))
        return (rows, cols, data)
        return coo_matrix((data, (rows, cols)), shape=(n2 * n2, n1 * n1)).tocsr()
    ro, co, da = build_operator_average(g1rsse, g2rsse)
    #rint("Sparse map   ping elements: ", Map12.nnz)
    Map12 = build_operator_average(g1rsse, g2rsse)
    return Map12, co, ro


@app.cell
def _(co, ro):
    max(co),max(ro)
    return


@app.cell
def _(co, plt):
    plt.plot(co)
    return


@app.cell
def _(Map12, RSSE1, no_rsse2, np):
    SeHSE1 = RSSE1.self_energy(0.1, bulk=True, coupling=True)
    SeHSE2 = (Map12 @ SeHSE1.ravel()).reshape(no_rsse2,no_rsse2)
    mx1 = np.max(np.abs(SeHSE1))
    mx2 = np.max(np.abs(SeHSE2))
    print(np.max(np.abs(SeHSE2 - SeHSE2.T)))
    mx1,mx2
    return SeHSE1, SeHSE2


@app.cell
def _(SeHSE1, SeHSE2, g1rsse, np, plt):
    def plot_M_dist(geom,M):
        xyz = geom.xyz
        d = xyz[np.newaxis, :, :] - xyz[:, np.newaxis, :]
        dR = np.linalg.norm(d, axis=2).flatten()
        plt.scatter(dR,M.imag.flatten())
        print(np.max(np.abs(M)))
    plot_M_dist(g1rsse,SeHSE1);plot_M_dist(g1rsse,SeHSE2);
    return


@app.cell
def _(H2, Map12, RSSE1, eta, g1rsse, g2rsse, np, rs_elec_indices2):
    no_rsse1=g1rsse.na 
    no_rsse2=g2rsse.na
    def getGF1to2(e):
        SeHSE1 = RSSE1.self_energy(e, bulk=True, coupling=True)
        ## Here comes the magic!
        SeHSE2 = (Map12 @ SeHSE1.ravel()).reshape(no_rsse2,no_rsse2)
        #SeHSE2 = 0.5*(SeHSE2 + SeHSE2.T)
        invGF = H2.Sk()*(e - 1j*eta) - H2.Hk()
        invGF = invGF.todense()
        invGF[np.ix_(rs_elec_indices2, rs_elec_indices2)] = SeHSE2
        GF = np.linalg.inv(invGF)
        return GF

    return getGF1to2, no_rsse2


@app.cell
def _(En, H2, getGF1to2, np, tqdm):
    dos1to2 = np.array([-np.trace(getGF1to2(e)).imag/(H2.na*np.pi) for e in tqdm(En)]) # dos per atom
    return (dos1to2,)


@app.cell
def _(En, dos1to2, plt):
    plt.plot(En,dos1to2,'o')
    # plt.plot(En,dos2)
    return


@app.cell
def _(H1):
    H1.geometry.center()
    return


@app.cell
def _(H1, plot_with_numbers):
    plot_with_numbers(H1.geometry)
    return


@app.cell
def _(H2, get_center_atoms):
    get_center_atoms(H2.geometry,5)
    return


@app.cell
def _(En, H_elecL, coord_sort, eta, np, si, tqdm):
    RSSE = si.RealSpaceSE(H_elecL, 0, 1, (12, 12, 1))
    H_rs_elec, rs_elec_indices = RSSE.real_space_coupling(ret_indices=True)
    H_rs_elec.write(dir+"/H-rs_elec.TSHS")
    H = RSSE.real_space_parent()
    # Create the true device by re-arranging the atoms
    indices = np.arange(len(H.geometry))
    indices_dev = np.delete(indices, rs_elec_indices) ## 
    g_rs_dev = H.geometry.sub(indices_dev)
    idx_sort = coord_sort(g_rs_dev, mode='angle', shell_width_r2=2.6, return_geom=False)
    indices = indices_dev[idx_sort]
    # first electrodes, then rest of device
    indices = np.concatenate([rs_elec_indices, indices])
    # Now re-arange matrix
    H = H.sub(indices)
    Htot = H.copy()

    with si.io.tbtgfSileTBtrans(dir+"/RSSE.TBTGF") as f:
        bz = si.BrillouinZone(H_rs_elec) 
        f.write_header(bz, En + 1j*eta) # + eta)
        for ispin, new_k, k, e in tqdm(f, unit="rsSE"):
            if new_k:
                Sk = (H_rs_elec.Sk()).todense()   ### Note here the full PBC-ham
                Hk = (H_rs_elec.Hk()).todense()
                f.write_hamiltonian(Hk,Sk)
            SeHSE = RSSE.self_energy(e , bulk=True, coupling=True)
            f.write_self_energy(SeHSE)
    return


if __name__ == "__main__":
    app.run()
