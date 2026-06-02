import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")

with app.setup:
    import sisl
    from mytools.construct import all_armchair
    from pathlib import Path
    import marimo as mo
    import h5py


@app.cell
def _():
    tile_choice = mo.ui.slider(start=3, stop=13, step=1, value=3, debounce=True, include_input=True, label="Tiling, N")
    energy_choice = mo.ui.slider(start=-0.4, stop=0.4, step=0.1, value=0.0, debounce=True, include_input=True, label="Energy, E (eV)")
    eta_choice = mo.ui.dropdown([1e-1, 1e-2, 1e-3, 1e-4,1e-5], value=1e-2, label="$\eta$")
    kind_choice = mo.ui.dropdown(["armchair", "zigzag"], value="zigzag", label="Edges")

    params = mo.vstack([tile_choice, mo.hstack([energy_choice, mo.md("+"), eta_choice], justify="start"), kind_choice], justify="start", align="stretch")
    return kind_choice, params


@app.cell
def _(params):
    params
    return


@app.cell
def _(kind_choice):
    KIND = kind_choice.value
    DATA_DIR = Path(__file__).parent.parent / "conv_data"

    DATA = h5py.File(DATA_DIR / f"RSE_data-{KIND}.h5", 'r')
    print(DATA.keys())
    print(DATA.attrs.keys())
    print(DATA[list(DATA.keys())[0]].keys())
    return


if __name__ == "__main__":
    app.run()
