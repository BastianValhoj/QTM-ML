import matplotlib
import numpy as np
import sisl 

mm=np.dot

def dagger(M):
    return np.conjugate(np.transpose(M))

def abs2(z):
    return (z*np.conjugate(z)).real
    
from tqdm import tqdm   


def PDOS_ORB(file):
    PDOS=sisl.get_sile(file)
    pdos_data=PDOS.read_data()
    geom = pdos_data[0]
    E = pdos_data[1]  # energies (in eV)
    PDOS_orb_tot = pdos_data[2][0]  # PDOS per orbital TOTAL
    PDOS_orb_z = pdos_data[2][1]  # PDOS per orbital TOTAL
    n_atoms = geom.na
    n_orbs = geom.no
    n_E = E.size
    # Get orbital-to-atom map
    orb2atom = geom.o2a(np.arange(0,geom.no))  # list of orbital indices per atom
    
    PDOS_orb_up=0.5*(PDOS_orb_tot+PDOS_orb_z)
    PDOS_orb_dn=0.5*(PDOS_orb_tot-PDOS_orb_z)
    
    # Total DOS
    DOS_orb_tot=PDOS_orb_tot.sum(axis=0)
    DOS_orb_z=PDOS_orb_z.sum(axis=0)
    
    DOS_orb_up=PDOS_orb_up.sum(axis=0)
    DOS_orb_dn=PDOS_orb_dn.sum(axis=0)
    return PDOS_orb_tot,PDOS_orb_z,PDOS_orb_up,PDOS_orb_dn,E

def DOS_ORB(file):
    PDOS_orb_tot,PDOS_orb_z,PDOS_orb_up,PDOS_orb_dn,E= PDOS_ORB(file)
    # Total DOS
    DOS_orb_tot=PDOS_orb_tot.sum(axis=0)
    DOS_orb_z=PDOS_orb_z.sum(axis=0)
    
    DOS_orb_up=PDOS_orb_up.sum(axis=0)
    DOS_orb_dn=PDOS_orb_dn.sum(axis=0)
    return DOS_orb_tot,DOS_orb_z,DOS_orb_up,DOS_orb_dn,E