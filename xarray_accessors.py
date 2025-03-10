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
import itertools
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
import pyproj
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.tri import Triangulation
import cartopy
from .genutils import method_cacher
from .dateutils import datetime_plus_nmonths, CF_CALENDARTYPE_DEFAULT, \
                       CF_CALENDARTYPE_360DAYS

def preprocess_dataset_mar(ds):
    """Preprocessing function to open MAR multiple-file datasets."""
    units = ds["time"].attrs["units"]
    if units.startswith("MONTHS since "):
        f = "%Y-%m-%d %H:%M:%S"
        if len(units) == 18 and units.endswith(":0"):
            units += "0"
        start = datetime.strptime(units[13:], f)
        try:
            calendar = ds["time"].attrs["calendar"]
        except KeyError:
            calendar = CF_CALENDARTYPE_DEFAULT
        if calendar in CF_CALENDARTYPE_360DAYS:
            raise ValueError('This function is meant to deal with "months '
                             'since" time data with calendars other than '
                             '360-day calendars.')
        convert = lambda t: datetime_plus_nmonths(start, t, calendar)
        convert_all = np.vectorize(convert)
        out = ds.assign_coords(time=convert_all(ds["time"].values))
        return out
    else:
        return ds

def open_dataset_mar(filepath, **kwargs):
    """Function to open MAR datasets.

    Keyword arguments, if any, are passed to xarray.open_dataset.

    """
    return preprocess_dataset_mar(xr.open_dataset(filepath, **kwargs))

def open_mfdataset_mar(filepath, **kwargs):
    """Function to open MAR multiple-file datasets.

    Keyword arguments, if any, are passed to xarray.open_mfdataset.

    """
    if "preprocess" in kwargs:
        msg = ('This wrapper around xarray.open_mfdataset does not accept '
               '"preprocess" as a keyword argument.')
        raise ValueError(msg)
    return xr.open_mfdataset(filepath, preprocess=preprocess_dataset_mar,
                             **kwargs)

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

def _unique_guess_in_iterable(guesses, iterable):
    """Return unique guess that is found in iterable, error otherwise."""
    found = [guess in iterable for guess in guesses]
    if sum(found) != 1:
        raise ValueError("Zero or more than one guess(es) is in iterable.")
    return guesses[found.index(True)]

class GenericDatasetAccessor(ABC):

    """Template for all other xarray dataset accessors defined below."""

    def __init__(self, dataset):
        self._dataset = dataset
        self._cache = dict()

    def units_nice(self, varname):
        """Return units of given variable, in a predictible format."""
        units = self._dataset[varname].attrs["units"]
        replacements = {
            "-": None,
            "1": None,
            "kg/(s*m2)": "kg m-2 s-1",
            "kg/(m2*s)": "kg m-2 s-1",
            "kg/m2/s": "kg m-2 s-1",
        }
        try:
            units = replacements[units]
        except KeyError:
            pass
        return units

    def check_units(self, varname, expected, nice=True):
        """Raise exception if units of variable are not as expected."""
        if nice:
            actual = self.units_nice(varname)
        else:
            actual = self._dataset[varname].attrs["units"]
        if actual != expected:
            raise ValueError('Bad units: expected "%s", got "%s"' %
                             (expected, actual))

    @property
    def time_dim(self):
        """Return the name of the time dimension of the file."""
        guesses = ("time_counter", "time")
        return _unique_guess_in_iterable(guesses, self._dataset.dims)

    def time_coord(self, varname):
        """Return the name of the time coordinate associated with variable."""
        dim = self.time_dim
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

    def _check_dimname_guesses(self, guesses):
        """Return name of only dimension in guesses that is found, or error."""
        return _unique_guess_in_iterable(guesses, self._dataset.dims)

    def _check_varname_guesses(self, guesses):
        """Return name of only variable in guesses that is found, or error."""
        return _unique_guess_in_iterable(guesses, self._dataset)

@xr.register_dataset_accessor("wizard")
class WizardDatasetAccessor(GenericDatasetAccessor):

    @property
    def whoami(self):
        """Guess and return the name of the model that created the output."""
        try:
            name = self._dataset.attrs["name"]
        except KeyError:
            try:
                model = self._dataset.attrs["model"]
            except KeyError:
                pass
            else:
                if model.startswith("regional climate model MARv"):
                    return "mar"
        else:
            if name == "histmth":
                return "lmdz"
            elif name.startswith("ismip6_"):
                return "elmerice"
        msg = "Existential crisis: I cannot guess which model created me."
        raise ValueError(msg)

    @property
    def myself(self):
        """Return reference to named accessor corresponding to self."""
        return getattr(self._dataset, self.whoami)

    @property
    def crs_pyproj(self):
        """Return the CRS (pyproj) corresponding to dataset."""
        return self.myself.crs_pyproj

    @property
    def crs_cartopy(self):
        """Return the CRS (cartopy) corresponding to dataset."""
        return self.myself.crs_cartopy

    def time_coord(self, varname):
        """Return the name of the time coordinate associated with variable."""
        return self.myself.time_coord(varname)

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
        if self.epsg == 3031:
            proj = ("+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 "
                    "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs")
        elif self.epsg == 3413:
            proj = ("+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 "
                    "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs")
        else:
            raise ValueError("Unknown or unsupported EPSG: %d." % self.epsg)
        return pyproj.CRS.from_proj4(proj)

    @property
    def crs_cartopy(self):
        """Return the CRS (cartopy) corresponding to the file."""
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
        dims = self._dataset.sizes.keys()
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

@xr.register_dataset_accessor("lmdz")
class LMDzDatasetAccessor(GenericDatasetAccessor):

    surface_types = ("ter", "lic", "oce", "sic")

    @property
    def crs_pyproj(self):
        """Return the CRS (pyproj) corresponding to the file."""
        raise NotImplementedError("Not implemented yet.")

    @property
    def crs_cartopy(self):
        """Return the CRS (cartopy) corresponding to the file."""
        return cartopy.crs.PlateCarree()

    @property
    @method_cacher
    def cell_area(self):
        """Return an array of the areas of the grid cells.

        Currently, outputs from LMDz-DYNAMICO have incorrect values for
        extensive quantities such as grid cell areas (these values are
        incorrect because the wrong interpolation method is used in the model).

        This function calculates the correct values for the grid cell
        areas. Note that this function does not currently calculate areas for
        grid cells at the edge of the domain (because I am not sure how to
        handle these cases). For these grid cells, a NAN value is used.

        """
        lon, lat = self._dataset["lon"].values, self._dataset["lat"].values
        nlon, nlat = lon.size, lat.size
        lonmid = (lon[1:] + lon[:-1]) / 2
        latmid = (lat[1:] + lat[:-1]) / 2
        geod = pyproj.Geod(ellps="WGS84")
        out = np.full([nlat, nlon], np.nan)
        for i, j in itertools.product(range(1, nlat-1), range(1, nlon-1)):
            lons = (lon[j-1], lon[j], lon[j], lon[j-1])
            lats = (lat[i-1], lat[i-1], lat[i], lat[i])
            out[i,j] = geod.polygon_area_perimeter(lons, lats)[0]
        return out

    @property
    @method_cacher
    def grid_type(self):
        """Return the type of grid: "reg" (lat/lon) or "ico" (dynamico)."""
        dims = ("lon", "lat", "cell", "nvertex")
        dims = dict((d, d in self._dataset.dims) for d in dims)
        if (dims["lon"] and dims["lat"] and
            not dims["cell"] and not dims["nvertex"]):
            return "reg"
        elif (not dims["lon"] and not dims["lat"] and
              dims["cell"] and dims["nvertex"]):
            return "ico"
        else:
            raise ValueError("Cannot guess LMDz grid type.")

    @property
    def ncells(self):
        """Return the number of cells in grid (error if not dynamico)."""
        if self.grid_type == "ico":
            return self._dataset.sizes["cell"]
        else:
            raise ValueError("Invalid grid type for this method.")

    def plot_gridded_colors_ico(self, colors, box=None,
                                ax=None, ec="k", lw=0.5):
        """Plot given colors on dynamico grid as colored polygons."""
        if isinstance(colors, str):
            colors = [colors] * self.ncells
        if ax is None:
            ax = plt.gca()
        try:
            lon_bnds = self._dataset["lon_bnds"].values
            lat_bnds = self._dataset["lat_bnds"].values
        except KeyError:
            lon_bnds = self._dataset["bounds_lon"].values
            lat_bnds = self._dataset["bounds_lat"].values
        if box is not None:
            lon, lat = self._dataset["lon"].values, self._dataset["lat"].values
            idx = (lon >= box[0]) * (lon <= box[1]) * \
                  (lat >= box[2]) * (lat <= box[3])
            lon_bnds, lat_bnds = lon_bnds[idx,:], lat_bnds[idx,:]
            idx = np.array(range(self.ncells))[idx]
        else:
            idx = range(self.ncells)
        transform = cartopy.crs.PlateCarree()
        for i, idxi in enumerate(idx):
            coords = np.array(list(zip(lon_bnds[i,:], lat_bnds[i,:])))
            if coords[:,0].min() < -100 and coords[:,0].max() > 100:
                # These cells are annoying to plot so we skip them (for now)
                # TODO: fix this
                continue
            ax.add_patch(Polygon(coords, transform=transform,
                                 fc=colors[idxi], ec=ec, lw=lw))

    def plot_gridded_values_ico(self, values, cmap=mpl.colormaps["viridis"],
                                vmin=None, vmax=None, box=None,
                                ax=None, ec="k", lw=0.5):
        """Plot given values on dynamico grid as colored polygons."""
        if vmin is None:
            vmin = values.min()
        if vmax is None:
            vmax = values.max()
        colors = cmap(np.interp(values, [vmin, vmax], [0, 1]))
        self.plot_gridded_colors_ico(colors, box=box, ax=ax, ec=ec, lw=lw)

@xr.register_dataset_accessor("mar")
class MARDatasetAccessor(GenericDatasetAccessor):

    def time_coord(self, varname):
        """Return the name of the time coordinate associated with variable."""
        return "time" if "time" in self._dataset[varname].dims else None

    @property
    def crs_pyproj(self):
        """Return the CRS (pyproj) corresponding to the file."""
        try:
            prms = self._dataset.attrs["mapping"]
        except KeyError:
            prms = self._dataset.attrs["projection"]
        else:
            prms = prms.replace(";", ",")
        prms = dict(s.split("=") for s in prms.split(","))
        if prms["grid_mapping_name"] == "polar_stereographic":
            return pyproj.CRS.from_dict(dict(
                proj="stere", ellps=prms["ellipsoid"],
                lat_0=float(prms["latitude_of_projection_origin"]),
                lon_0=float(prms["straight_vertical_longitude_from_pole"]),
                lat_ts=float(prms["standard_parallel"]),
                x_0=float(prms["false_easting"]),
                y_0=float(prms["false_northing"]),
            ))
        else:
            raise NotImplementedError("Unsupported projection.")

    @property
    def crs_cartopy(self):
        """Return the CRS (cartopy) corresponding to the file."""
        try:
            prms = self._dataset.attrs["mapping"]
        except KeyError:
            prms = self._dataset.attrs["projection"]
        else:
            prms = prms.replace(";", ",")
        prms = dict(s.split("=") for s in prms.split(","))
        if prms["grid_mapping_name"] == "polar_stereographic":
            return cartopy.crs.Stereographic(
                globe=cartopy.crs.Globe(ellipse=prms["ellipsoid"]),
                central_latitude=float(prms["latitude_of_projection_origin"]),
                central_longitude=float(
                    prms["straight_vertical_longitude_from_pole"]),
                true_scale_latitude=float(prms["standard_parallel"]),
                false_easting=float(prms["false_easting"]),
                false_northing=float(prms["false_northing"]),
            )
        else:
            raise NotImplementedError("Unsupported projection.")
