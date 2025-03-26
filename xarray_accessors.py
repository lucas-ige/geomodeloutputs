# Copyright (2024-now) Institut des Géosciences de l'Environnement, France.
#
# This software is released under the terms of the BSD 3-clause license:
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     (1) Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#
#     (2) Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#
#     (3) The name of the author may not be used to endorse or promote products
#     derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
# EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Module geomodeloutputs: accessors to add functionality to datasets."""

from abc import ABC
from typing import Iterable
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
from ._typing import NumType, ColorType
from ._genutils import method_cacher
from .dateutils import datetime_plus_nmonths, CF_CALENDARTYPE_DEFAULT, \
                       CF_CALENDARTYPE_360DAYS

def _preprocess_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Preprocessing function to open non CF-compliant datasets.

    :param ds: the dataset opened with vanilla xarray.open_dataset.

    :returns: the processed dataset.

    """
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

def open_dataset(filepath: str, **kwargs) -> xr.Dataset:
    """Open dataset.

    This function acts as xarray.open_dataset, except that it can handle files
    that have non CF-compliant time units, such as "months since ...".

    :param filepath: location of the file on disk.

    :param \\*\\*kwargs: additional keyword arguments, if any, are passed "as
        is" to xarray.open_dataset.

    :returns: the opened dataset.

    """
    return _preprocess_dataset(xr.open_dataset(filepath, **kwargs))

def open_mfdataset(filepath: str, **kwargs) -> xr.Dataset:
    """Open multiple-file dataset.

    This function acts as xarray.open_mfdataset, except that it can handle
    files that have non CF-compliant time units, such as "months since ...".

    :param filepath: location of the file(s) on disk. It can be any pattern
        accepted by xarray.open_mfdataset.

    :param \\*\\*kwargs: additional keyword arguments, if any, are passed "as
        is" to xarray.open_dataset, with one exception: named argument
        "preprocess" is not allowed here.

    :returns: the opened dataset.

    """
    if "preprocess" in kwargs:
        msg = ('This wrapper around xarray.open_mfdataset does not accept '
               '"preprocess" as a keyword argument.')
        raise ValueError(msg)
    return xr.open_mfdataset(filepath, preprocess=_preprocess_dataset,
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

    def __init__(self, dataset : xr.Dataset) -> None:
        self._dataset = dataset
        self._cache = dict()

    def units_nice(self, varname: str) -> str or None:
        """Return units of given variable, in a predictible format.

        Predictable format:

         - uses single spaces to separate the dimensions in the units

         - uses negative exponents instead of division symbols

         - always orders dimensions in this order: mass, length, time

         - never uses parentheses

        :param varname: the name of the variable in the NetCDF file.

        :returns: the formatted units (or None for dimensionless variables).

        """
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

    def check_units(self, varname: str, expected: str,
                    nice: bool = True) -> None:
        """Raise ValueError if units of variable are not as expected."""
        if nice:
            actual = self.units_nice(varname)
        else:
            actual = self._dataset[varname].attrs["units"]
        if actual != expected:
            raise ValueError('Bad units: expected "%s", got "%s"' %
                             (expected, actual))

    @property
    def time_dim(self) -> str:
        """Return the name of the time dimension of the file."""
        guesses = ("time_counter", "time")
        return _unique_guess_in_iterable(guesses, self._dataset.dims)

    def time_coord(self, varname: str) -> str:
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

    def times(self, varname: str, dtype: str = "datetime"):
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

    @property
    def crs_pyproj(self):
        """The CRS (pyproj) corresponding to dataset."""
        raise NotImplementedError("Not implemented for this case.")

    @property
    def crs_cartopy(self):
        """The CRS (cartopy) corresponding to dataset."""
        raise NotImplementedError("Not implemented for this case.")

    def ll2xy(self, lon, lat):
        """Convert from (lon,lat) to (x,y)."""
        f = transformer_from_crs_pyproj(self.crs_pyproj)
        return f(lon, lat)

    def xy2ll(self, x, y):
        """Convert from (x,y) to (lon,lat)."""
        f = transformer_from_crs_pyproj(self.crs_pyproj, reverse=True)
        return f(x, y)

    def _guess_dimname(self, guesses):
        """Return name of only dimension in guesses that is found, or error."""
        return _unique_guess_in_iterable(guesses, self._dataset.dims)

    def _guess_varname(self, guesses):
        """Return name of only variable in guesses that is found, or error."""
        return _unique_guess_in_iterable(guesses, self._dataset)

    @property
    def dimname_ncells(self) -> str:
        """The name of the dimensions for the number of grid cells.

        This property makes sense for unstructured grids only.

        """
        return self._guess_dimname(["cells", "cell_mesh"])

    @property
    def ncells(self) -> int:
        """The number of cells in the grid.

        This property makes sense for unstructured grids only.

        """
        return self._dataset.sizes[self.dimname_ncells]

    @property
    @method_cacher
    def varnames_lonlat(self) -> tuple[str, str]:
        """The names of the longitude and latitude variables."""
        guesses_lon = ["lon", "lon_mesh"]
        lon_name = self._guess_varname(guesses_lon)
        guesses_lat = [s.replace("lon", "lat") for s in guesses_lon]
        lat_name = self._guess_varname(guesses_lat)
        if guesses_lon.index(lon_name) != guesses_lat.index(lat_name):
            raise ValueError("Inconsistent lon/lat variable names.")
        return lon_name, lat_name

    @property
    @method_cacher
    def varnames_lonlat_bounds(self) -> tuple[str, str]:
        """The names of the lon/lat bound variables.

        This property only makes sense for unstructured grids. For these grids,
        the bound variables are arrays of shape (n_cells, n_vertices) that
        contain the coordinates of the vertices of each cell.

        """
        guesses_lon = ["lon_bnds", "bounds_lon", "lon_mesh_bnds"]
        lon_name = self._guess_varname(guesses_lon)
        guesses_lat = [s.replace("lon", "lat") for s in guesses_lon]
        lat_name = self._guess_varname(guesses_lat)
        if guesses_lon.index(lon_name) != guesses_lat.index(lat_name):
            raise ValueError("Inconsistent lon/lat bound variable names.")
        return lon_name, lat_name

    def plot_ugridded_colors(
            self,
            colors: Iterable[ColorType],
            box: tuple[NumType, NumType, NumType, NumType] | None = None,
            ax: mpl.axes.Axes | None = None,
            **kwargs
        ) -> None:
        """Plot given colors as colored polygons on unstructured grid.

        :param colors: the face colors of the polygons. There must be exactly
            as many colors as there are cells in the grid.

        :param box: the longitude and latitude limits of the interesting part
            of the data, in the format (lon_min, lon_max, lat_min, lat_max).
            Grid cells outside of this range will not be plotted.

        :param ax: the Matplotlib axis object onto which to draw the data.

        :param \\*\\*kwargs: these are passed to Matplotlib's Polygon.

        """
        if ax is None:
            ax = plt.gca()
        lon_bnds = self._dataset[self.varnames_lonlat_bounds[0]].values
        lat_bnds = self._dataset[self.varnames_lonlat_bounds[1]].values
        if box is not None:
            lon = self._dataset[self.varnames_lonlat[0]].values
            lat = self._dataset[self.varnames_lonlat[1]].values
            idx = (lon >= box[0]) * (lon <= box[1]) * \
                  (lat >= box[2]) * (lat <= box[3])
            idx = np.array(range(self.ncells))[idx]
        else:
            idx = range(self.ncells)
        transform = cartopy.crs.PlateCarree()
        for i in idx:
            coords = np.array(list(zip(lon_bnds[i,:], lat_bnds[i,:])))
            if coords[:,0].min() < -100 and coords[:,0].max() > 100:
                # These cells are annoying to plot so we skip them (for now)
                # TODO: fix this
                continue
            ax.add_patch(Polygon(coords, transform=transform,
                                 fc=colors[i], **kwargs))

    def plot_ugridded_values(
            self,
            values: Iterable[NumType],
            cmap=mpl.colormaps["viridis"], #TODO: needs a type hint
            vmin: NumType | None = None,
            vmax: NumType | None = None,
            **kwargs,
        ) -> None:
        """Plot given values as colored polygons on unstructured grid.

        :param values: the values to be plotted. There must be exactly as many
            values as there are grids in the cell.

        :param cmap: the colormap to use.

        :param vmin: the minimum value to show on the color scale.

        :param vmax: the maximum value to show on the color scale.

        :param \\*\\*kwargs: these are passed to self.plot_ugridded_colors.

        """
        if vmin is None:
            vmin = values.min()
        if vmax is None:
            vmax = values.max()
        colors = cmap(np.interp(values, [vmin, vmax], [0, 1]))
        self.plot_ugridded_colors(colors, **kwargs)

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

    @property
    @method_cacher
    def map_face_node(self) -> np.typing.NDArray:
        """An array giving, for each face, the indices of its vertices.

        The indices are given in Python convention (ie. starting at 0).

        """
        n_faces = self._dataset.sizes[self.dimname_face]
        n_vertices = self._dataset.sizes[self.dimname_vertex]
        varname = self.meshname + "_face_nodes"
        out = np.array(self._dataset[varname].values)
        if out.shape != (n_faces, n_vertices):
            raise ValueError("Map array has invalid shape.")
        start_index = int(self._dataset[varname].start_index)
        if start_index > 0:
            out -= start_index
        elif start_index != 0:
            raise ValueError("Negative start index.")
        return out

    def node2face(self, values : Iterable) -> np.typing.NDArray:
        """Convert given node values to face values.

        :param values: the array of node values to convert.

        :returns: the corresponding array of face values.

        """
        if len(values) != self._dataset.sizes[self.dimname_node]:
            raise ValueError("Bad length for input.")
        map_ = self.map_face_node
        out = np.zeros(map_.shape[0])
        for i in range(len(out)):
            nodevals = [values[v] for v in map_[i,:]]
            out[i] = sum(nodevals) / len(nodevals)
        return out

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
