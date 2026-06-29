import marimo

__generated_with = "0.23.9"
app = marimo.App()

with app.setup:
    import sisl
    import numpy 
    import matplotlib.pyplot as plt

    from mytools.construct import all_armchair

    from pathlib import Path


@app.cell
def _():
    NOTEBOOK = Path(__file__)
    NOTEBOOK_DIR = NOTEBOOK.parent
    return


@app.cell
def _():
    zigzag = sisl.geom.graphene()
    return


if __name__ == "__main__":
    app.run()
