import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import ticker
import numpy as np
from typing import Any, overload, Literal


fontsize = 11
tex_fonts = {
        #"text.usetex": True,            # use LaTeX for all text (use False to avoid calling external dependecies)
        "font.family": "serif",          # Mathc LaTeX serif
        "font.serif": ["Computer Modern Roman", "Times New Roman", "serif"],
        "mathtext.fontset": "cm",        # Ensure math looks like LaTex
        "font.size": fontsize,          # Standard thesis font size
        "axes.labelsize": fontsize,     # size of X and Y labels 
        "axes.titlesize": fontsize+2,   # Subplot titles (ax.set_title, or ax.set(title=...) calls)
        "xtick.labelsize": fontsize-2,  # X tick label size 
        "ytick.labelsize": fontsize-2,  # Y tick label size 
        "legend.fontsize": fontsize-2,  # Size of the legend 
        "legend.title_fontsize": fontsize+1, # Size of the legend title (rarely used)
        "axes.linewidth": 0.8,          #  
        "grid.linewidth": 0.5,          # 
        "lines.linewidth":1.5,          # 
        "figure.titlesize": fontsize+4, # Overall figure title size
        "axes.labelpad": 0,            # Distance between label and axis
        "savefig.dpi": 300,             # High resoltion for any raster elements
        "savefig.format": "pdf",        # Default to pdf for LaTeX compatibility
        "savefig.bbox": "tight",        # Equivalent to bbox_inches='tight'
        "savefig.pad_inches": 0.05,     # Quantify the 'tight' padding
        "savefig.transparent": False,   # Usually better for thesis background, and if I want to use slide decks with colored backgrounds
}

@overload
def thesis_fig(
    nrows: Literal[1] = 1, 
    ncols: Literal[1] = 1,
    *,
    width_pt=426.79135, 
    fraction=1, 
    subplots=None, 
    use_tex=False,
    aspect=None,
    **kwargs
) -> tuple[plt.Figure, plt.Axes]: ...

@overload
def thesis_fig(
    nrows = 1, 
    ncols = 1,
    *,
    width_pt=426.79135, 
    fraction=1, 
    subplots=None, 
    use_tex=False,
    aspect=None,
    **kwargs
) -> tuple[plt.Figure, plt.Axes]: ...

def thesis_fig(nrows=1, ncols=1, 
    *, width_pt=426.79135, fraction=1, subplots=None, use_tex=False, aspect=None, 
    **kwargs
    ) -> tuple[plt.Figure, plt.Axes | np.ndarray]:
    r"""
    Initializes a matplotlib figure with dimensions scaled to a LaTeX document.

    Parameters
    ----
    width_pt : float, default 426.79135,
        the `\textwidth` of the LaTeX doc.
    fraction: float, default 1.0,
        how much of the page width the plot should take (0.5=half, 1=whole)
    subplots : tuple, default (1, 1)
        (rows, columns)
    use_tex : bool, default False
        Whether to render the matplotlib text using tex -- supposedly better but significantly slower
    kwargs : *optional*,
        passed to plt.sublots() (e.g. sharex=True)

    Returns
    ---
    fig, axes: initialized figure and axes
    """

    # 1.  calculate Dimensions
    fig_width_pt = width_pt * fraction
    inches_per_pt = 1/72.27
    fig_width_in = fig_width_pt * inches_per_pt
    
    if ((nrows != 1) or (ncols != 1)) and not subplots is None:
        print(f"{nrows=}, {ncols=}, {subplots=}")
        print("Warning both `nrows, ncols` and `subplots` were given. Will use `nrows, ncols`")
        return thesis_fig(nrows, ncols, 
                          width_pt=width_pt, 
                          fraction=fraction, 
                          use_tex=use_tex, 
                          aspect=aspect, 
                          **kwargs)
    if not subplots is None:
        print("Warning: argument `subplots` is being deprecated. In the future use `ncols, nrows` as for `matplotlib.pyplot.subplots`")
        nrows, ncols = subplots
    
    if aspect is None:
        golden_ratio = (5**0.5 - 1) / 2
        fig_height_in = golden_ratio * fig_width_in * (nrows / ncols)
    else:
        fig_height_in = fig_width_in * aspect

    tex_fonts["figure.figsize"] = (fig_width_in, fig_height_in)
    tex_fonts["text.usetex"] = use_tex


    # 2. configure LaTeX-style fonts globally
    mpl.rcParams.update(tex_fonts)

    # fig, ax = plt.subplots(subplots[0], subplots[1], **kwargs)
    
    # 3.  This handles both integers and floats
    # formatter = ticker.StrMethodFormatter('{x:,}')
    
    # # If you have subplots, we iterate; if single plot, we wrap in list
    # axes = ax if isinstance(ax, (list, mpl.axes.Axes, np.ndarray)) else [ax]
    
    # # Handle 2D arrays of axes from subplots
    # if hasattr(axes, 'flat'):
    #     axes = axes.flat

    # for a in axes:
    #     a.xaxis.set_major_formatter(formatter)
    #     a.yaxis.set_major_formatter(formatter)

    return plt.subplots(nrows, ncols, **kwargs)



def label_subplots(axes, 
                   labels=['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)'], 
                   pos=(0.09, 0.95)):
    flat_axes = axes.flatten()
    
    x, y = pos

    for i, ax in enumerate(flat_axes):
        # x=-0.1, y=1.1 positions the label slightly outside the top-left corner
        # transform=ax.transAxes ensures (0,0) is bottom-left and (1,1) is top-right of the plot
        ax.text(x, y, labels[i], transform=ax.transAxes,
                fontsize=15, fontweight='bold', va='top', ha='right')
