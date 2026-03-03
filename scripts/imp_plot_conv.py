import numpy as np
from pathlib import Path
from generator import systemInit, setup_ham_rse, calculate_spectral_density
from tqdm.auto import tqdm
import sisl
import h5py
import gc

SCRIPT_DIR = Path(__file__).resolve().parent
OUTDIR = SCRIPT_DIR / "conv_data"
if not OUTDIR.exists():
    print("Err: dir '{}' does NOT exist".format(OUTDIR))
    CWD = Path.cwd()
    print('Create \'{}\' at \'{}\' '.format(OUTDIR, CWD))
    OUTDIR.mkdir(parents=False, exist_ok=False)
    # print('####\'{}\' created'.format(OUT_DIR))
else:
    print("dir '{}' exists!".format(OUTDIR))


# init global data
a = np.array([1.])
b = np.array([1, 3, 4, 5])*(-1)
ETAS = np.multiply.outer(10.0**b, a, dtype=np.float128).ravel()
print('etas ({}) : {}'.format(len(ETAS), ETAS))

dE = 0.1
EMAX = 0.4
EMIN = -EMAX
ENERGIES = np.arange(EMIN, EMAX+dE/2, dE, dtype=np.float128).round(1)
print('E ({})    : {}'.format(len(ENERGIES), ENERGIES))

E0_IDX = np.argwhere(ENERGIES == 0).ravel() # returns tuple of arrays of len == ndim(ENERGIES)
E0_IDX = E0_IDX[0] # since ENERGIES is 1d array, the tuple has len 1
print('idx(E=0) : {}'.format(E0_IDX))

NLIST = np.array([4*i+1 for i in range(2, 6)])[::-1]
print(f"Ns : {NLIST}, shape={NLIST.shape}")

NK1 = int(np.ceil(3*900/12))
SEMI_AXIS = 0 # semi-infinite axis
K_AXES = 1 # k-sampling axis/axes

_vmin = 0
_vmax = 0
with h5py.File(OUTDIR / 'RSE_data.h5', 'w') as file:
    file.attrs['E'] = ENERGIES
    file.attrs['E0_idx'] = E0_IDX
    file.attrs['ETA'] = ETAS
    file.attrs['N'] = NLIST

    for i, N in enumerate(tqdm(NLIST, desc="Looping tiling", leave=True)):
        
        group_N = file.create_group(f"N_{N}")
        
        Ham0 = systemInit(1.43, -2.7)
        HamNN = Ham0.tile(N, 0).tile(N, 1)
        HamNN.set_nsc([1,1,1])
        
        rse = sisl.RealSpaceSE(Ham0, SEMI_AXIS, K_AXES, (N, N, 1))
        rse.setup(eta=ETAS[0], bz=sisl.MonkhorstPack(Ham0, [1, NK1, 1]))
        _, elec_idx = rse.real_space_coupling(ret_indices=True)
        all_atoms = np.arange(0, HamNN.na)
        device_atoms = np.delete(all_atoms, elec_idx)
        atoms_idx = np.concatenate([elec_idx, device_atoms])
        HamNN_re = HamNN.sub(atoms_idx)
        HamNN_re.reduce()
        
        num_atoms = len(HamNN_re)
        num_E = len(ENERGIES)
        RSEs_shape = (num_E, num_atoms, num_atoms)
        
        group_N.create_dataset("xyz", data=HamNN_re.geometry.xyz, dtype=np.float64)
        group_N.create_dataset("atoms_idx", data=atoms_idx, dtype=np.int16)
        group_N.create_dataset("elec_idx", data=elec_idx, dtype=np.int16)
        
        del HamNN, HamNN_re
        gc.collect()
        

        for j, eta in enumerate(tqdm(ETAS, desc="Looping etas", leave=False)):
            if (j > 0):
                rse = sisl.RealSpaceSE(Ham0, SEMI_AXIS, K_AXES, (N, N, 1))
                rse.setup(eta=eta,
                        bz=sisl.MonkhorstPack(Ham0, [1, NK1, 1]))
            RSEs_group_eta = group_N.create_dataset(f"eta_{eta:.1e}",shape=RSEs_shape,
                                          dtype=np.complex128, compression='gzip')            
        
            for i, E in enumerate(tqdm(ENERGIES, desc='Looping energies', leave=False)):
                z = E + eta*1j
                # RSE_re = rse.self_energy(z)[atoms_idx, :][:, atoms_idx]
                RSEs_group_eta[i, :, :] = rse.self_energy(z)[atoms_idx, :][:, atoms_idx]
            _vmin = np.min([_vmin,
                            np.min([np.real(RSEs_group_eta),
                                    np.imag(RSEs_group_eta)
                                    ])
                            ])
            _vmax = np.max([_vmax,
                            np.max([np.real(RSEs_group_eta),
                                    np.imag(RSEs_group_eta)])
                            ])
                
            del rse, RSEs_group_eta #, RSE_re
            gc.collect()
    file.attrs['vmin'] = _vmin
    file.attrs['vmax'] = _vmax
    del group_N, all_atoms, device_atoms, elec_idx, atoms_idx
    gc.collect()

