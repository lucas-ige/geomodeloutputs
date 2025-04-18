# Copyright (c) 2024-now, Institut des Géosciences de l'Environnement, France.
#
# License: BSD 3-clause "new" or "revised" license (BSD-3-Clause).

"""Module geomodeloutputs: utilities for plotting model outputs."""

from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import cartopy

preset_lims = {
    "greenland": (-750000, 920000, -3460000, -533000),
}

preset_crs = {
    "greenland": cartopy.crs.NorthPolarStereo(
        central_longitude=-45,
        true_scale_latitude=70,
    ),
}

_PrepareFigResult = namedtuple("_PrepareFig", "wfig, hfig nw nh w h left "
                               "right bottom top wsep hsep lims rescale crs "
                               "coastlines fig axes")


def prepare_fig(
    nw=1,
    nh=1,
    w=3,
    h=4 / (1 + 5**0.5) * 2,
    left=0.1,
    right=0.1,
    bottom=0.1,
    top=0.1,
    wsep=0.1,
    hsep=0.1,
    crs=None,
    lims=None,
    rescale=None,
    coastlines=None,
):
    """Create and return a figure and its grid of subplots.

    Parameters:
    -----------
    nw : int
        The number of subplots along the width.
    nh : int
        The number of subplots along the height.
    w : numeric
        The width of each subplot, in inches.
    h : numeric
        The height of each subplot, in inches.
    left : numeric
        The left margin, in inches.
    right : numeric
        The right margin, in inches.
    bottom : numeric
        The bottom margin, in inches.
    top : numeric
        The top margin, in inches.
    wsep : numeric
        The horizontal separation between subplots, in inches.
    hsep : numeric
        The vertical separation between subplots, in inches.
    crs : None | str | pyproj.CRS
        The coordinate system used in the subplots. Can be specified as a
        string literal corresponding to one of the presets (eg. `"greenland"`).
    lims : None | str | [numeric, numeric, numeric, numeric]
        The x and y limits of the subplot, as [xmin, xmax, ymin, ymax]. Can be
        specified as a string literal corresponding to one of the presets
        (eg. `"greenland"`).
    rescale : None | bool
        Whether to rescale the height of subplots so that a distance of 1 on
        the y-axis is equal to a distance of 1 on the x-axis. If both `crs` and
        `lims` are not `None` (ie. the plot is a map), then rescaling happens
        no matter the value of this parameter.
    coastlines: None | dict
        Keyword-value pairs of parameters to pass to the coastlines plotting
        function (use `None` for no coastlines).

    Returns:
    --------
    NamedTuple
        Contains all the (post-processed) inputs and 4 additional properties:
         - wfig (float): the total width of the figure, in inches.
         - hfig (float): the total height of the figure, in inches.
         - fig (matplotlib.Figure): the handle to the figure.
         - axes (2D numpy.array of matplotlib.Axes): the handles to the axes.
    """
    crs = preset_crs[crs] if isinstance(crs, str) else crs
    lims = preset_lims[lims] if isinstance(lims, str) else lims
    if rescale or rescale is None and crs is not None and lims is not None:
        if lims is None:
            raise ValueError("Need explicit limits to rescale.")
        h = w * (lims[3]-lims[2]) / (lims[1]-lims[0])
    wfig = left + w*nw + wsep*(nw-1) + right
    hfig = bottom + h*nh + hsep*(nh-1) + top
    fig = plt.figure(figsize=(wfig, hfig))
    axes = [[None for j in range(nw)] for i in range(nh)]
    position = [None, 1-(top+h)/hfig, w/wfig, h/hfig]
    for i in range(nh):
        position[0] = left / wfig
        for j in range(nw):
            axes[i][j] = fig.add_subplot(position=position, projection=crs)
            if coastlines is not None:
                axes[i][j].coastlines(**coastlines)
            if lims is not None:
                axes[i][j].set_xlim(lims[0], lims[1])
                axes[i][j].set_ylim(lims[2], lims[3])
            position[0] += (w + wsep) / wfig
        position[1] -= (h + hsep) / hfig
    plt.sca(axes[0][0])
    return _PrepareFigResult(wfig, hfig, nw, nh, w, h, left, right, bottom,
                             top, wsep, hsep, lims, rescale, crs, coastlines,
                             fig, np.array(axes))


def units_mpl(units):
    """Return given units, formatted for displaying on Matplotlib plots.

    Parameters:
    -----------
    units : str
        The units to format (eg. "km s-1").

    Returns:
    --------
    str
        The units formatted for Matplotlib (eg. "km s$^{-1}$").

    """
    split = units.split()
    for i, s in enumerate(split):
        n = len(s) - 1
        while (n >= 0 and s[n] in "-0123456789"):
            n -= 1
        if n < 0:
            raise ValueError("Could not process units.")
        if n != len(s):
            split[i] = "%s$^{%s}$" % (s[:n+1], s[n+1:])
    return " ".join(split)
