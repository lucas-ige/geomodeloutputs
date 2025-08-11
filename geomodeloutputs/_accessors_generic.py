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

    def attrvalue_among_guesses(self, varname, attrnames):
        """Return the value of the first attribute found among given list."""
        var = self[varname]
        for attrname in attrnames:
            try:
                return var.attrs[attrname]
            except KeyError:
                pass
        raise ValueError("None of the given names exist as attribute.")

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
        units = self.attrvalue_among_guesses(varname, ("units", "unit"))
        replacements = {
            "-": None,
            "1": None,
            "m2/s2": "m2 s-2",
            "kg/m2": "kg m-2",
            "kg/(s*m2)": "kg m-2 s-1",
            "kg/(m2*s)": "kg m-2 s-1",
            "kg/m2/s": "kg m-2 s-1",
            "W m{-2}": "W m-2",
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

    @property
    @method_cacher
    def crs(self):
        """The CRS corresponding to the dataset.

        We choose here to return the cartopy CRS rather than the pyproj CRS
        because the cartopy CRS is a subclass of the pyproj CRS, so it
        potentially has additional functionalily.

        """
        return self.crs_cartopy

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
        return self._guess_dimname(["cell", "cells", "ncells", "cell_mesh"])

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

    def cell_bounds(self, idx=None):
        """Return the coordinates of the bounds of given grid cell.

        Parameters
        ----------
        idx : int | None
            Index (0-based) of the grid cell of interest. Python-style negative
            indices are accepted. If idx is None, then this function returns a
            list of the coordinates of the bounds of all cells in the grid.

        Returns
        -------
        list of (lon,lat) coords (or list of these)
            The coordinates of the bounds of given grid cell(s), where
            duplicate points have been removed.

        """
        lon, lat = self.varnames_lonlat_bounds
        if self[lon].values.shape != self[lat].values.shape:
            raise ValueError("Shapes of bounds arrays mismatch.")

        def coords(i):
            """Return coordinates of cell i, removing duplicate points."""
            x = list(self[lon].values[i, :])
            y = list(self[lat].values[i, :])
            j = 1
            while j < len(x):
                if x[j] == x[j - 1] and y[j] == y[j - 1]:
                    del x[j], y[j]
                else:
                    j += 1
            return list(zip(x, y))

        if idx is None:
            return [coords(i) for i in range(self.ncells)]
        else:
            return coords(idx)

    def plot_ugridded_colors_and_labels(
        self,
        colors=None,
        labels=None,
        box=None,
        ax=None,
        prm_poly=dict(),
        prm_text=dict(),
    ):
        """Plot given colors as colored polygons on unstructured grid.

        Parameters
        ----------
        colors : sequence of colors | None
            If not `None`, then the face colors of the polygons (there must be
            exactly as many colors as there are cells in the grid).
        labels : sequence of strings | None
            If not `None`, then the labels of the polygons (there must be
            exactly as many labels as there are cells in the grid).
        box : sequence of four numbers
            The longitude and latitude limits of the interesting part of the
            data, in the format (lon_min, lon_max, lat_min, lat_max). Grid
            cells out of this range will not be plotted.
        ax : Matplotlib axes object
            The Matplotlib axis object onto which to draw the data (default is
            current axis).
        **prm_poly
            These are passed "as is" to Matplotlib's Polygon.
        **prm_text
            These are passed "as is" to Matplotlib's text.

        """
        if ax is None:
            ax = plt.gca()
        lon_bnds = self[self.varnames_lonlat_bounds[0]].values
        lat_bnds = self[self.varnames_lonlat_bounds[1]].values
        if box is not None or labels is not None:
            lon = self[self.varnames_lonlat[0]].values
            lat = self[self.varnames_lonlat[1]].values
        if box is not None:
            idx = (
                (lon >= box[0])
                * (lon <= box[1])
                * (lat >= box[2])
                * (lat <= box[3])
            )
            idx = np.array(range(self.ncells))[idx]
        else:
            idx = range(self.ncells)
        transform = cartopy.crs.Geodetic()
        for i in idx:
            if colors[i] is None:
                continue
            coords = np.array(list(zip(lon_bnds[i, :], lat_bnds[i, :])))
            ax.add_patch(
                Polygon(coords, transform=transform, fc=colors[i], **prm_poly)
            )
            if labels is not None:
                ax.text(
                    lon[i],
                    lat[i],
                    labels[i],
                    ha="center",
                    va="center",
                    clip_on=True,  # Currently does not work (bug in matplotlib/cartopy)
                    transform=transform,
                    **prm_text,
                )

    def plot_ugridded(
        self,
        values,
        labels=False,
        cmap="viridis",
        vmin=None,
        vmax=None,
        prm_poly=dict(),
        prm_text=dict(),
        **kwargs,
    ):
        """Plot given values as colored polygons on unstructured grid.

        Parameters
        ----------
        values : numpy.array
            The values to be plotted. There must be exactly as many values as
            there are grids in the cell.
        labels : bool | str | sequence of str
            Whether to print labels or not. If not a boolean, it can be a str
            that specifies the format of the labels (eg. "%.2f") or a sequence
            of str that gives the labels themselves.
        cmap : None | Matplotlib color map or just its name
            The colormap to use. Set not None to prevent plotting colors.
        vmin : numeric
            The minimum value to show on the color scale.
        vmax : numeric
            The maximum value to show on the color scale.
        prm_poly : dict
            These parameters are passed "as is" to Matplotlib's Polygon.
        prm_text : dict
            These parameters are passed "as is" to Matplotlib's text.
        **kwargs
            These are passed "as is" to plot_ugridded_colors_and_labels.

        """
        if isinstance(labels, bool) and labels:
            labels = [str(v) for v in values]
        elif isinstance(labels, bool):
            labels = None
        elif isinstance(labels, str):
            labels = [labels % v for v in values]
        elif len(labels) != len(values):
            raise ValueError("Bad number of labels.")
        if vmin is None:
            vmin = np.nanmin(values)
        if vmax is None:
            vmax = np.nanmax(values)
        if cmap is None:
            colors = ["none"] * len(values)
        else:
            if isinstance(cmap, str):
                cmap = mpl.colormaps[cmap]
            colors = cmap(np.interp(values, np.array([vmin, vmax]), [0, 1]))
        # TODO is the following needed?
        colors = [
            None if values[i] is None or np.isnan(values[i]) else col
            for i, col in enumerate(colors)
        ]
        self.plot_ugridded_colors_and_labels(
            colors=colors,
            labels=labels,
            prm_poly=prm_poly,
            prm_text=prm_text,
            **kwargs,
        )
