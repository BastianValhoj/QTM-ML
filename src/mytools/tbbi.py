import sisl
import numpy as np
from numba import prange,njit
from scipy import sparse as sp
from tqdm.auto import tqdm
from joblib import Parallel, delayed

IP_cutoff=1.6
OP_cutoff=6.

def tbbi_opt(geom,os_0=0., os_1=None, Vpppi=-2.7,Vpps=0.48,d0_00=1.42,d0_01=3.35,q0_00=2.,q0_01=None,
             dangling=0.2, field=0., eps=0.44,
             finite = False, 
             joblib_nprocs = 1,
             joblib_backend = 'loky'):

    """Tight-binding Hamiltonian for graphene nanostructure bilayers. The model uses an
       orthogonal basis set and the matrix elements between different orbitals are
       determined by Slater-Koster-type two-center $\pi$ and $\sigma$ hopping integrals
       between $p_z$ orbitals:

       Parameters
       ----------
       geom: str, sisl.Geometry
          Atomic structure of the system.
       os_0: float, optional
          On-site energy of the atoms in the bottom layer.
       os_1: float, optional
          On-site energy of the atoms in the top layer. I will be equal to os_0 by default.
       field: float, optional
          Electric field perpendicular to the layers. os_1 will be determined according to
          this and the interlayer distance.
       eps: float, optional
          Screening coefficient accounting for the electric field-induced charge redistribution.
       Vpppi: float, optional
          $pp\pi$ hopping integral.
       Vpps: float, optional
          $pp\sigma$ hopping integral.
       d0_00: float, optional
          Interatomic distance between atoms in the same layer.
       d0_01: float, optional
          Interlayer distance.
       q0_00: float, optional
          Decay rate for the $pp\pi$ hopping integral.
       q0_01: float, optional
          Decay rate for the $pp\sigma$ hopping integral. It will be equal
          to q0_00 by default, representing isotropic decay rates.
       IP_cutoff: float, optional
          Cutofff of the hopping between atoms in the same layer.
       OP_cutoff: float, optional
          Cutoff distance of the hopping between atoms in different layers.
       dangling: float, optional
          On-site energy shift applied to atoms in edges. These can be atoms in the edges
          of the 1D graphene nanostructures or in the edges of pores in nanoporous graphene
          nanostructures.
    """

    if isinstance(geom,sisl.Geometry):
        g = geom.copy() #type:ignore
    else:
        g = sisl.get_sile(geom).read_geometry()

    #g = g.remove([i for i in range(g.na) if g.atoms[i].Z == 1])

    gtmp = sisl.Geometry(g.xyz,atoms=sisl.Atom(6,R=1.44),sc=g.sc)  ### Test
    Ham = sisl.Hamiltonian(gtmp)                                   ### Test
    if finite == False:
        Ham.set_nsc((3,3,1))
    else:
        Ham.set_nsc((1,1,1))
    
    xyz  = Ham.xyz

    #d0_01 = g.xyz[:,2].max() - g.xyz[:,2].min()


    if os_1 is None:
        os_1 = os_0

    os_1 += field*eps*d0_01
    
    if q0_01 is None:
        q0_01 = q0_00

    def cart_to_sph(r):
        # Wiki spherical coordinates
        Res   = np.zeros(r.shape)
        R     = np.sqrt(np.sum(r**2,axis=1))
        theta = np.arctan2( np.sqrt(r[:,0]**2 + r[:,1]**2 ) , r[:,2] )
        phi   = np.arctan2( r[:,1]                          , r[:,0] )
        Res[:,0] = R
        Res[:,1] = theta
        Res[:,2] = phi
        return Res

    def SK_pz(xi,xj):
        rij    = xj - xi
        if len(rij.shape)==1:
            rij = np.array([rij])
        return _SK_pz(rij)

    def _SK_pz(rij, OP_tol = .5):
        V = np.zeros(len(rij))
        sph    = cart_to_sph(rij)
        l      = np.sin(sph[:,1])
        d      = sph[:,0]
        OP_hop = ( d < OP_cutoff )*( np.abs(rij[:,2])>OP_tol )
        idx_inter = OP_hop.nonzero()[0]

        l      = l[idx_inter]  ### Test
        d      = d[idx_inter]  ###
        V[idx_inter] = Vpppi*l**2 * np.exp(-(d - d0_00)*q0_00) + Vpps * (1-l**2) * np.exp(-(d-d0_01)*q0_01)  ###

        return V
    
    Ham.construct(([0.1,1.5],[0.,-2.7]))   ### Test
    Vals   = []
    if finite == False:
        Hops   = [(0,0), (0,1),(1,0),(1,1),(-1,0),(0,-1),(-1,1),(1,-1),(-1,-1)]
    elif finite==True:
        Hops   = [(0,0)]
    R1 = Ham.cell[0]; R2 = Ham.cell[1]
    for ij in Hops:
        T   = ij[0]*R1 + ij[1]*R2
        V   = sp.csr_matrix((len(xyz), len(xyz)))
        # making global varible always last resort
        # deleted right after paralle comp is done
        global global_func_islice
        def global_func_islice(i_slice):
            Iv     = []
            Inside = []
            vals_v = []
            for i in tqdm(i_slice):
                Txyz = xyz + T
                dij     = np.linalg.norm(xyz[i] - Txyz, axis = 1)
                inside  = np.where(dij < (OP_cutoff + 3.))[0]
                vals    = SK_pz(xyz[i], Txyz[inside])
                Iv     += [np.ones(len(vals),dtype=int)*i]
                Inside += [inside]
                vals_v += [vals]
            return Iv, Inside, vals_v
        print('USING JOBLIB')
        iSLICES = np.array_split(np.arange(len(xyz)), joblib_nprocs)
        res = Parallel(n_jobs  = joblib_nprocs,
                       backend = joblib_backend)(delayed(global_func_islice)(islice) 
                                                 for islice in iSLICES)
        del global_func_islice
        #global varible deleted
        Iv     = []
        Inside = []
        vals_v = []
        for resi in res:
            Iv     += resi[0]
            Inside += resi[1]
            vals_v += resi[2]
        
        V[np.hstack(Iv), np.hstack(Inside)] = np.hstack(vals_v)
        
        @njit(parallel=True,  fastmath = True)
        def count_neighbors(xyz,xyz33):
            na1 = len(xyz)
            count = np.zeros(na1, dtype=np.int32)
            for i in prange(na1):
                count[i] += (np.sum((xyz[i] - xyz33)**2,axis=1)**0.5 < 1.6).sum()
            return count
        
        if ij == (0,0):
            interz = np.average(xyz[:,2])
            idx_0 = np.where(xyz[:,2]<interz)[0]
            idx_1 = np.where(xyz[:,2]>interz)[0]
            os = np.zeros(Ham.no)
            os[idx_0]   = os_0
            os[idx_1]   = os_1
            xyz33 = g.tile(3,axis=1).tile(3,axis=0).translate(v=-g.cell[0]-g.cell[1]).xyz
            countNN = count_neighbors(xyz, xyz33)
            os[countNN<4] += dangling
            #for kk in tqdm(range(len(xyz))):
            #    if countNN[kk]<4:
            #        os[kk] += dangling
            
            idx         = np.arange(len(os))
            V[idx,idx]  = os
        Vals  += [V]

    for i in range(len(Hops)):
        ij   =  Hops[i]
        v_i    =  Vals[i]
        I,J,vals = sp.find(v_i)
        Ham[I, J, ij] = vals
    
    Ham.eliminate_zeros()
    Ham.finalize()

    return Ham