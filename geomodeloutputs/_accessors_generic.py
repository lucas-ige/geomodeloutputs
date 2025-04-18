# Copyright (c) 2024-2025, Institut des Géosciences de l'Environnement, France.
#
# License: BSD 3-clause "new" or "revised" license (BSD-3-Clause).

"""Module geomodeloutputs: accessors to add functionality to datasets."""

from abc import ABC
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import cartopy
from ._genutils import (
    method_cacher,
    unique_guess_in_iterable,
    transformer_from_crs,
)
from .graphics import units_mpl


class GenericDatasetAccessor(ABC):
    """Template for all other xarray dataset accessors defined below."""

    def __init__(self, dataset):
        self._dataset = dataset
        self._cache = dict()

    def __getitem__(self, *args, **kwargs):
        return self._dataset.__getitem__(*args, **kwargs)

    @property
    def dims(self):
        return self._dataset.dims

    @property
    def sizes(self):
        return self._dataset.sizes

    @property
    def attrs(self):
        return self._dataset.attrs

    def close(self, *args, **kwargs):
        return self._dataset.close(*args, **kwargs)

    def units_nice(self, varname):
        """Return units of given variable, in a predictible format.

        Predictable format:

         - uses single spaces to separate the dimensions in the units

         - uses negative exponents instead of division symbols

         - always orders dimensions in this order: mass, length, time

         - never uses parentheses

        Parameters
        ----------
        varname : str
            The name of the variable in the NetCDF file.

        Returns
        -------
        str
            The formatted units (or None for dimensionless variables).

        """
        units = self[varname].attrs["units"]
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
        """Make sure that units of given variable are as expected.

        Parameters
        ----------
        varname : str
            The name of the variable to check.
        expected : str
            The expected units.
        nice : bool
            Whether expected units are given as "nice" units
            (cf. method units_nice)

        Raises
        ------
        ValueError
            If the units are not as expected.

        """
        if nice:
            actual = self.units_nice(varname)
        else:
            actual = self[varname].attrs["units"]
        if actual != expected:
            raise ValueError(
                'Bad units: expected "%s", got "%s"' % (expected, actual)
            )

    def units_mpl(self, varname):
        """Return the units of given variable, formatted for Matplotlib."""
        return units_mpl(self.units_nice(varname))

    def vardesc(self, varname):
        """Return a (hopefully) human readable description of the variable."""
        for attr in ("long_name", "standard_name"):
            try:
                return self[varname].attrs[attr]
            except KeyError:
                continue
        return varname

    @property
    def dimname_time(self):
        """The name of the time dimension of the file."""
        return self._guess_dimname(("time_counter", "time"))

    @property
    def ntimes(self):
        """The number of time steps in the file."""
        return self.sizes[self.dimname_time]

    def time_coord(self, varname):
        """Return the name of the time coordinate associated with variable."""
        # TODO: this method needs a better / safer implementation
        dim = self.dimname_time
        if dim not in self[varname].dims:
            raise ValueError("Cannot determine name of time coordinate.")
        try:
            coord = self[varname].attrs["coordinates"]
        except KeyError:
            coord = " ".join(list(self[varname].coords.keys()))
        if " " in coord:
            coord = coord.split()[self[varname].dims.index(dim)]
        if coord.startswith("_"):
            coord = "time%s" % coord
        return coord

    def times(self, varname, dtype="datetime"):
        """Return array of times corresponding to given variable.

        Parameters
        ----------
        varname : str
            The name of the variable of interest.
        dtype : {"datetime", "pandas", "numpy" ,"xarray"}
            The data type of dates in the output.

        Returns
        -------
        numpy.array
            A numpy array containging the dates.

        """
        values = self[self.time_coord(varname)]
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
        tr = transformer_from_crs(self.crs)
        return tr.transform(lon, lat)

    def xy2ll(self, x, y):
        """Convert from (x,y) to (lon,lat)."""
        tr = transformer_from_crs(self.crs, reverse=True)
        return tr.transform(x, y)

    def _guess_dimname(self, guesses):
        """Return name of only dimension in guesses that is found, or error."""
        return unique_guess_in_iterable(guesses, self.dims)

    def _guess_varname(self, guesses):
        """Return name of only variable in guesses that is found, or error."""
        return unique_guess_in_iterable(guesses, self._dataset)

    @property
    def dimname_ncells(self):
        """The name of the dimensions for the number of grid cells.

        This property makes sense for unstructured grids only.

        """
        return self._guess_dimname(["cell", "cells", "cell_mesh"])

    @property
    def ncells(self):
        """The number of cells in the grid.

        This property makes sense for unstructured grids only.

        """
        return self.sizes[self.dimname_ncells]

    @property
    @method_cacher
    def varnames_lonlat(self):
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
    def varnames_lonlat_bounds(self):
        """The names of the lon/lat bound variables.

        This property only makes sense for unstructured grids. For these grids,
        the bound variables are arrays of shape (n_cells, n_vertices) that
        contain the coordinates of the vertices of each cell.

        """
        guesses_lon = [
            "lon_bnds",
            "lon_mesh_bnds",
            "bounds_lon",
            "bounds_lon_mesh",
        ]
        lon_name = self._guess_varname(guesses_lon)
        guesses_lat = [s.replace("lon", "lat") for s in guesses_lon]
        lat_name = self._guess_varname(guesses_lat)
        if guesses_lon.index(lon_name) != guesses_lat.index(lat_name):
            raise ValueError("Inconsistent lon/lat bound variable names.")
        return lon_name, lat_name

    def plot_ugridded_colors(self, colors, box=None, ax=None, **kwargs):
        """Plot given colors as colored polygons on unstructured grid.

        Parameters
        ----------
        colors : sequence of colors
            The face colors of the polygons. There must be exactly as many
            colors as there are cells in the grid.
        box : sequence of four numbers
            The longitude and latitude limits of the interesting part of the
            data, in the format (lon_min, lon_max, lat_min, lat_max). Grid
            cells outse of this range will not be plotted.
        ax : Matplotlib axes object
            The Matplotlib axis object onto which to draw the data (default is
            current axis).
        **kwarg
            These are passed "as is" to Matplotlib's Polygon.

        """
        if ax is None:
            ax = plt.gca()
        lon_bnds = self[self.varnames_lonlat_bounds[0]].values
        lat_bnds = self[self.varnames_lonlat_bounds[1]].values
        if box is not None:
            lon = self[self.varnames_lonlat[0]].values
            lat = self[self.varnames_lonlat[1]].values
            idx = (
                (lon >= box[0])
                * (lon <= box[1])
                * (lat >= box[2])
                * (lat <= box[3])
            )
            idx = np.array(range(self.ncells))[idx]
        else:
            idx = range(self.ncells)
        transform = cartopy.crs.PlateCarree()
        for i in idx:
            coords = np.array(list(zip(lon_bnds[i, :], lat_bnds[i, :])))
            if coords[:, 0].min() < -100 and coords[:, 0].max() > 100:
                # These cells are annoying to plot so we skip them (for now)
                # TODO: fix this
                continue
            ax.add_patch(
                Polygon(coords, transform=transform, fc=colors[i], **kwargs)
            )

    def plot_ugridded_values(
        self, values, cmap="viridis", vmin=None, vmax=None, **kwargs
    ):
        """Plot given values as colored polygons on unstructured grid.

        Parameters
        ----------
        values : numpy.array
            The values to be plotted. There must be exactly as many values as
            there are grids in the cell.
        cmap : Matplotlib color map, or just its name
            The colormap to use.
        vmin : numeric
            The minimum value to show on the color scale.
        vmax : numeric
            The maximum value to show on the color scale.
        **kwargs
            These are passed "as is" to self.plot_ugridded_colors.

        """
        if vmin is None:
            vmin = values.min()
        if vmax is None:
            vmax = values.max()
        if isinstance(cmap, str):
            cmap = mpl.colormaps[cmap]
        colors = cmap(np.interp(values, np.array([vmin, vmax]), [0, 1]))
        self.plot_ugridded_colors(colors, **kwargs)


@xr.register_dataset_accessor("wizard")
class WizardDatasetAccessor(GenericDatasetAccessor):

    @property
    @method_cacher
    def whoami(self):
        """Guess and return the name of the model that created the output."""
        try:
            name = self.attrs["name"]
        except KeyError:
            try:
                model = self.attrs["model"]
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
    @method_cacher
    def myself(self):
        """The reference to named accessor corresponding to self."""
        return getattr(self._dataset, self.whoami)

    @property
    @method_cacher
    def crs_pyproj(self):
        """The CRS (pyproj) corresponding to the dataset."""
        return self.myself.crs_pyproj

    @property
    @method_cacher
    def crs_cartopy(self):
        """The CRS (cartopy) corresponding to the dataset."""
        return self.myself.crs_cartopy

    @property
    @method_cacher
    def crs(self):
        """The CRS corresponding to the dataset.

        We choose here to return the cartopy CRS rather than the pyproj CRS
        because the cartopy CRS is a subclass of the pyproj CRS, so it
        potentially has additional functionalily.

        """
        return self.myself.crs_cartopy

    def time_coord(self, varname):
        """Return the name of the time coordinate associated with variable."""
        return self.myself.time_coord(varname)
