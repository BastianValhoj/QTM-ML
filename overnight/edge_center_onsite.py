
import sisl
import numpy as np
from tqdm.auto import tqdm
import json
from pathlib import Path
from datetime import datetime
from mytools.construct import all_armchair



kind = "armchair"
cwd = Path(__file__).parent
param_file = cwd / "geom_params.json"
out_file = cwd / f"{kind}-edge"

## Global params
if param_file.exists():
    with open(param_file, 'r') as file:
        config = json.load(file)
    
    # Extract values
    BOND = config["BOND"]
    R = config["R"]
    T = config["T"]
    ETA = config['ETA']
    NK1 = config['NK1']
else:
    raise FileNotFoundError(f"Could not find config file at {param_file}")



def main():
    print(f"kind = {kind}")

    energies = np.arange(0.0, 0.3, 0.1)
    num_e = len(energies)
    if kind == "zigzag":

        start = 13
        stop = 31
        step = 2
        gr_base = sisl.geom.graphene(BOND)
        kNN = 10
        #kNNslice = lambda N: slice(N//2 - kNN//2, N//2 + kNN//2, 1)
        
    elif kind == "armchair":
        start = 10 
        stop = 20
        step = 2
        gr_base = all_armchair(BOND)
        kNN = 20
        #kNNslice = lambda N: slice(N - kNN//2, N + kNN//2, 1)
        
    tiles = np.arange(start, stop, step)
    num_n = len(tiles)

    ham0 = sisl.Hamiltonian(gr_base)
    ham0.construct([R, T])
    onsite_edge_center = np.zeros(shape=(num_n, num_e))
    edge_coupling = np.zeros(shape=(num_n, num_e, kNN), dtype=np.complex128)
    for i, N in enumerate(tqdm(tiles, desc='looping tiling')):
        pbar_e = tqdm(energies, desc='looping energy', leave=(num_n-1 == i)) # only leave if it is the last

        rse = sisl.RealSpaceSE(ham0, 0, 1, [N, N, 1])
        rse.setup(eta=ETA, bz=sisl.MonkhorstPack(ham0, [1,NK1,1]))
        ham_nn = rse.real_space_parent()
        _, elec_idx = rse.real_space_coupling(ret_indices=True)
        all_idx = np.arange(ham_nn.na)
        device_idx = np.delete(all_idx, elec_idx)
        all_idx = np.concat([elec_idx, device_idx])
        ham_nn.sub(all_idx)
        ham_nn.reduce()
        for j, en in enumerate(pbar_e):
            pbar_e.set_postfix(N=N, Na=ham_nn.na, E=f"{en:.2f}")
            
            z = en + ETA*1j
            se = rse.self_energy(z)
            se = se[all_idx, :][:, all_idx]
            if kind == "zigzag":
                edge_coupling[i,j, :] = se[N//2, N//2-kNN//2:N//2+kNN//2]
            elif kind == "armchair":
                edge_coupling[i,j, :] = se[N, N-kNN//2:N+kNN//2]
                
            
            se = np.imag(se)
            se_diag = np.diag(se)
            onsite_edge_center[i, j] = se_diag[N//2] # sorted to that the first N are on the same edge -- N//2 is the atom center most on first edge
    
    print(f"\n### Done. Will save to {out_file}.npz")
    return onsite_edge_center, edge_coupling, tiles, energies

if __name__ == "__main__":
    print("### started (HH:MM:SS)", datetime.now().strftime("%H:%M:%S"))
    onsite, couple, tiles, energies = main()
    np.savez(out_file, onsite=onsite, couple=couple, tiles=tiles, energies=energies, eta=ETA, nk1=NK1, bond=BOND)
    print("### completed (HH:MM:SS)", datetime.now().strftime("%H:%M:%S"))
    

