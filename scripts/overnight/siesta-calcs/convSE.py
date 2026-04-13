import sisl
import h5py
import numpy as np
from pathlib import Path


cdir = Path(__file__).parent
arm_dir = list(cdir.glob('TBT-Nw*-armchair'))
zig_dir = list(cdir.glob('TBT-Nw*-zigzag'))

all_dir = arm_dir + zig_dir


def GetSigma(file, E=0, 
        elec=0, 
        k=[0.0, 0.0, 0.0]):
    tmp = file.self_energy(elec=0, E=E, k=k, sort=True).data
    pvt = file.pivot(elec=elec, in_device=True, sort=True).reshape(-1, 1)
    no_d = file.na_d

    Sigma = np.zeros(no_d, dtype=np.complex128)
    Sigma[pvt, pvt.T] += tmp
    return Sigma

for aNGR in arm_dir:
    file = sisl.io.get_sile(aNGR / 'armchair.TBT.SE.nc')
    
    
