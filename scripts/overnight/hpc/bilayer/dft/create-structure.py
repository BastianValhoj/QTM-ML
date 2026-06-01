import sisl
from pathlib import Path
from mytools.tbbi import tbbi_opt


d = 3.35
bond = 1.42
tile = 4

STACK = "AA"

WORK_DIR = Path.home() / "w3"
BILA_DIR = WORK_DIR / "bilayer_data"
DFT_DIR = BILA_DIR / "DFT"
STACK_DIR = DFT_DIR / f"{STACK}_stack_{tile}x{tile}"
STACK_DIR.mkdir(exist_ok=True, parents=True)

gr = sisl.geom.graphene(bond, vacuum=20)
grlayer = gr.translate([0, 0, 10])
grlayer = grlayer.tile(tile, 1).tile(tile, 0)
grlayer_top = grlayer.translate([0, 0,  d/2])
grlayer_bot = grlayer.translate([0, 0, -d/2])
if STACK == "AB":
    grlayer_top = grlayer_top.translate([bond, 0, 0])

gr_bilayer = grlayer_top.add(grlayer_bot)

# ham_bilayer, _ = tbbi_opt(
#     geom=gr_bilayer,
#     os_0=mu_bot,
#     os_1=mu_top,
#     Vpppi=-2.7,
#     Vpps=0.48,
#     finite=False,
#     dangling=0.0,
# )


# ham_bilayer.write(STACK_DIR / f"ham_bilayer.nc")

gr_bilayer.write(STACK_DIR / "graphene_bilayer.fdf")
print("done!")