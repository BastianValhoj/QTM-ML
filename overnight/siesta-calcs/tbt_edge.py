import sys
from pathlib import Path
import sisl
import json

DEFAULT_NW = 10
if len(sys.argv) > 1:
    try:
        Nw = int(sys.argv[1])
        print(f"Using Nw provided by argument: {Nw}")
    except ValueError:
        print(f"Warning: Argument '{sys.argv[1]}' is not a number. Using defualt {DEFAULT_NW}")
        Nw = DEFAULT_NW
else:
    print(f"No argument procided. Using default {DEFAULT_NW}")
    Nw = DEFAULT_NW

file_dir = Path(__file__).parent

# load geom parameters from JSON file
with open(file_dir.parent / 'geom_params.json', 'r') as file:
    params = json.load(file)

BOND = params['BOND']
ETA = params['ETA']
R = params['R']
T = params['T']

phi = 30 # degrees
labels = ['zigzag', 'armchair']
num_k = 100
#Nw = 10
Nl = 3

def make_fdf(label, path, ksamp, bloch, semi_axis,eta):
    fdf = f"""
SystemLabel {label}
TBT.Directory {path}

TBT.k [{ksamp[0]} {ksamp[1]} {ksamp[2]}]

TBT.Elecs.Eta     {eta} eV 
TBT.Contours.Eta  {eta} eV
%block TBT.Contour.line
  from 0.0 eV to 1. eV  
   delta 0.01 eV
    method mid-rule
%endblock

TBT.HS {path}/H_dev.nc

TBT.CDF.SelfEnergy.Save
TBT.CDF.SelfEnergy.Save.Mean  True
TBT.T.Bulk True
TBT.DOS.Elecs True
TBT.DOS.A.All True
TBT.DOS.Gf True

%block TS.Elecs
  ELeft
  ERight
%endblock TS.Elecs

%block TS.ChemPots
  x
%endblock TS.ChemPots

%block TS.ChemPot.x
  mu 0.0 eV
  contour.eq
    begin
      C-x
      T-x
    end
%endblock TS.ChemPot.x

%block TBT.Elec.ELeft
   HS {path}/H_elec.nc
   chemical-potential x
   semi-inf-direction -{semi_axis}
   electrode-position 1
   Bloch {bloch[0]} {bloch[1]} {bloch[2]}
   Bulk True
%endblock TS.Elec.ELeft

%block TBT.Elec.ERight
   HS {path}/H_elec.nc
   chemical-potential x
   semi-inf-direction +{semi_axis}
   electrode-position {4*Nw*2+1}
   Bloch {bloch[0]} {bloch[1]} {bloch[2]}
   Bulk True
%endblock TBT.Elec.ERight
"""
    return fdf

base_geom = sisl.geom.graphene(BOND, orthogonal=True)
Ham0 = sisl.Hamiltonian(base_geom)
Ham0.construct([R,T])


for lab in labels:
    out_path = file_dir / f'TBT-Nw{Nw}-{lab}'
    out_path.mkdir(parents=True, exist_ok=True)
    Ham0.write(out_path / 'H_elec.nc')

    if lab == 'armchair':
        HamNN = Ham0.tile(Nw, 1).tile(Nl, 0)
        ksamp = [1, num_k, 1]
        bloch = [1, Nw, 1]
        semi_axis = 'A1'
    elif lab == 'zigzag':
        HamNN = Ham0.tile(Nw, 0).tile(Nl, 1)
        ksamp = [num_k, 1, 1]
        bloch = [Nw, 1, 1]
        semi_axis = 'A2'
    else:
        raise ValueError('lab was neither zig-zag or armchair')

    HamNN.write(out_path / 'H_dev.nc')

    fdf = make_fdf(label=lab,
            path=out_path,
            ksamp=ksamp,
            bloch=bloch,
            semi_axis=semi_axis,
            eta=ETA,
    )

    with open(out_path / 'RUNTBT.fdf', 'w') as file:
        file.write(fdf)
