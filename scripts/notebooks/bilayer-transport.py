import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")

with app.setup:
    import sisl
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path


@app.cell
def _():
    SCRIPT_DIR = Path(__file__).parent.parent
    OUT_DIR = SCRIPT_DIR / "overnight" / "rsse_data" / "TBT-test_6_to_13"
    print(OUT_DIR)
    print("##Exists:", OUT_DIR.exists())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
