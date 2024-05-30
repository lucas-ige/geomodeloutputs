"""Module geomodeloutputs: easily use files that are geoscience model outputs.

Copyright (2024-now) Institut des Géosciences de l'Environnement (IGE), France.

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

from abc import ABC, abstractproperty
import numpy as np
import pandas as pd
import xarray as xr
import pyproj
from matplotlib.tri import Triangulation
import cartopy
from ._genutils import method_cacher

def transformer_from_crs_pyproj(crs_pyproj, reverse=False):
    """Return the transformer corresponding to given pyproj CRS.

    The transformer is a function that transforms lon,lat to x,y (or the other
    way around if reverse is True).

    """
    fr = crs_pyproj.geodetic_crs
    to = crs_pyproj
    if reverse:
        fr, to = to, fr
    return pyproj.Transformer.from_crs(fr, to).transform

class GenericDatasetAccessor(ABC):

    """Template for all other xarray dataset accessors defined below."""

    def __init__(self, dataset):
        self._dataset = dataset
        self._cache = dict()

    def units(self, varname):
        """Return units of given variable, in a predictible format."""
        units = self._dataset[varname].attrs["units"]
        replacements = {
            "kg/(s*m2)": "kg m-2 s-1",
            "kg/m2/s": "kg m-2 s-1",
        }
        try:
            units = replacements[units]
        except KeyError:
            pass
        return units

    def time_coord(self, varname):
        """Return the name of the time coordinate associated with variable."""
        dim = "time_counter"
        if dim not in self._dataset[varname].dims:
            raise ValueError("Cannot determine name of time coordinate.")
        coord = self._dataset[varname].attrs["coordinates"]
        if " " in coord:
            coord = coord.split()[self._dataset[varname].dims.index(dim)]
        if coord.startswith("_"):
            coord = "time" + coord
        return coord

    def times(self, varname, dtype="datetime"):
        """Return array of times corresponding to given variable.

        Parameter "dtype" indicates which format will be used for the date
        objects. Type "datetime" corresponds to Python's standard library
        datetime object, type "pandas" corresponds to the default type used by
        pandas (and similarly for types "numpy" and "xarray").

        """
        values = self._dataset[self.time_coord(varname)]
        if dtype == "datetime":
            f = "%Y-%m-%d %H-%M-%S %f"
            def convert(t):
                return datetime.strptime(pd.to_datetime(t).strftime(f), f)
            values = np.vectorize(convert)(values.values)
        elif dtype == "pandas":
            values = pd.to_datetime(values.values)
        elif dtype == "numpy":
            values = values.values
        elif dtype != "xarray":
            raise ValueError("Invalid dtype: %s." % dtype)
        return values

    @abstractproperty
    def crs_pyproj(self):
        """Return the CRS (pyproj) corresponding to dataset."""
        pass

    @abstractproperty
    def crs_cartopy(self):
        """Return the CRS (cartopy) corresponding to dataset."""
        pass

    def ll2xy(self, lon, lat):
        """Convert from (lon,lat) to (x,y)."""
        f = transformer_from_crs_pyproj(self.crs_pyproj)
        return f(lon, lat)

    def xy2ll(self, x, y):
        """Convert from (x,y) to (lon,lat)."""
        f = transformer_from_crs_pyproj(self.crs_pyproj, reverse=True)
        return f(x, y)

@xr.register_dataset_accessor("elmerice")
class ElmerIceDatasetAccessor(GenericDatasetAccessor):

    @property
    def epsg(self):
        """Return the EPSG code associated with file."""
        epsg = self._dataset.attrs["projection"].split(":")
        # Here we account for a typo (espg) in the XIOS configuration files
        # that were used in ISMIP6 simulations, and potentially others
        if len(epsg) != 2 or epsg[0] not in ("epsg", "espg"):
            raise ValueError("Invalid value for projection global attribute.")
        return int(epsg[1])

    @property
    def icesheet(self):
        """Return name of icesheet, inferred from global attributes."""
        if self.epsg == 3031:
            return "Antarctica"
        elif self.epsg == 3413:
            return "Greenland"
        else:
            raise RuntimeError("Could not infer name of icesheet.")

    @property
    def crs_pyproj(self):
        """Return the CRS (pyproj) corresponding to dataset."""
        return pyproj.CRS.from_epsg(self.epsg)

    @property
    def crs_cartopy(self):
        """Return the CRS (cartopy) corresponding to the file."""
        # TODO make cartopy work directly with EPSG code if possible
        if self.epsg == 3031:
            return cartopy.crs.SouthPolarStereo(
                central_longitude=0, true_scale_latitude=-71)
        elif self.epsg == 3413:
            return cartopy.crs.NorthPolarStereo(
                central_longitude=-45, true_scale_latitude=70)
        else:
            raise ValueError("Unsupported projection: %d." % self.epsg)

    @property
    @method_cacher
    def meshname(self):
        """Return the name of the mesh (or None if it cannot be guessed)."""
        dims = self._dataset.dims.keys()
        candidates = [d for d in dims if len(d) > 6 and
                      d.startswith("n") and d.endswith("_edge")]
        if len(candidates) != 1:
            return None
        meshname = candidates[0][1:-5]
        for which in ("face", "node", "vertex"):
            if "n%s_%s" % (meshname, which) not in dims:
                return None
        return meshname

    @property
    def dimname_edge(self):
        """Return the name of the dimension that holds the number of edges."""
        name = self.meshname
        if name is not None:
            name = "n%s_edge" % name
        return name

    @property
    def dimname_face(self):
        """Return the name of the dimension that holds the number of faces."""
        name = self.meshname
        if name is not None:
            name = "n%s_face" % name
        return name

    @property
    def dimname_node(self):
        """Return the name of the dimension that holds the number of nodes."""
        name = self.meshname
        if name is not None:
            name = "n%s_node" % name
        return name

    @property
    def dimname_vertex(self):
        """Return the name of the dim. that holds the number of vertices."""
        name = self.meshname
        if name is not None:
            name = "n%s_vertex" % name
        return name

    @property
    @method_cacher
    def triangulation(self):
        """Return Triangulation object corresponding to data."""
        return Triangulation(
            self._dataset["x"].values[0,:],
            self._dataset["y"].values[0,:],
            self._dataset[self.meshname + "_face_nodes"].values)
