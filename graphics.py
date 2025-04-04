"""Module geomodeloutputs: easily use files that are geoscience model outputs.

Copyright (2025-now) Institut des Géosciences de l'Environnement (IGE), France.

This software is released under the terms of the BSD 3-clause license:

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    (1) Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

    (2) Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.

    (3) The name of the author may not be used to endorse or promote products
    derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
OF SUCH DAMAGE.

"""

from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import cartopy

_PrepareFig = namedtuple("_PrepareFig", "wfig, hfig nw nh w h left right "
                         "bottom top wsep hsep lims rescale crs coastlines "
                         "fig axes")

preset_lims = {
    "greenland": (-750000, 920000, -3460000, -533000),
}

preset_crs = {
    "greenland": cartopy.crs.NorthPolarStereo(
        central_longitude=-45,
        true_scale_latitude=70,
    ),
}

def prepare_fig(
        nw = 1,
        nh = 1,
        w = 3,
        h = 4 / (1 + 5**0.5) * 2,
        left = 0.1,
        right = 0.1,
        bottom = 0.1,
        top = 0.1,
        wsep = 0.1,
        hsep = 0.1,
        crs = None,
        lims = None,
        rescale = None,
        coastlines = None,
):
    """Create and return the figure and all the subplots."""
    proj = preset_crs[crs] if isinstance(crs, str) else crs
    if lims is not None:
        limits = preset_lims[lims] if isinstance(lims, str) else lims
    if rescale or rescale is None and crs is not None and lims is not None:
        if lims is None:
            raise ValueError("Need explicit limits to rescale.")
        h = w * (limits[3]-limits[2]) / (limits[1]-limits[0])
    wfig = left + w*nw + wsep*(nw-1) + right
    hfig = bottom + h*nh + hsep*(nh-1) + top
    fig = plt.figure(figsize=(wfig, hfig))
    axes = [[None for j in range(nw)] for i in range(nh)]
    position = [None, 1-(top+h)/hfig, w/wfig, h/hfig]
    for i in range(nh):
        position[0] = left / wfig
        for j in range(nw):
            axes[i][j] = fig.add_subplot(position=position, projection=proj)
            if coastlines is not None:
                axes[i][j].coastlines(**coastlines)
            if lims is not None:
                axes[i][j].set_xlim(limits[0], limits[1])
                axes[i][j].set_ylim(limits[2], limits[3])
            position[0] += (w + wsep) / wfig
        position[1] -= (h + hsep) / hfig
    plt.sca(axes[0][0])
    return _PrepareFig(wfig, hfig, nw, nh, w, h, left, right, bottom, top,
                       wsep, hsep, lims, rescale, crs, coastlines,
                       fig, np.array(axes))

def units_mpl(units: str) -> str:
    """Return given units, formatted for Matplotlib."""
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
