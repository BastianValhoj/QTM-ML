import marimo 

app = marimo.App(width="full")


with app.setup:
    import sisl
    import numpy as np
    import matplotlib.pyplot as plt
    
    from mytools.construct import all_armchair
    from mytools.plots import thesis_fig, label_subplots
    
    from pathlib import Path
    