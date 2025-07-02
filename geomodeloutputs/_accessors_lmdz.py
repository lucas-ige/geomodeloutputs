# Copyright (c) 2024-2025, Institut des Géosciences de l'Environnement, France.
#
# License: BSD 3-clause "new" or "revised" license (BSD-3-Clause).

"""Module geomodeloutputs: accessors to add functionality to datasets."""

import itertools
import numpy as np
import xarray as xr
import pyproj
import cartopy
from ._genutils import method_cacher
from ._accessors_generic import GenericDatasetAccessor


@xr.register_dataset_accessor("lmdz")
class LMDzDatasetAccessor(GenericDatasetAccessor):

    surface_types = ("ter", "lic", "oce", "sic")

    @property
    @method_cacher
    def crs_pyproj(self):
        """Return the CRS (pyproj) corresponding to the file."""
        raise NotImplementedError("Not implemented yet.")

    @property
    @method_cacher
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
        lon, lat = self["lon"].values, self["lat"].values
        nlon, nlat = lon.size, lat.size
        geod = pyproj.Geod(ellps="WGS84")
        out = np.full([nlat, nlon], np.nan)
        for i, j in itertools.product(range(1, nlat - 1), range(1, nlon - 1)):
            lons = (lon[j - 1], lon[j], lon[j], lon[j - 1])
            lats = (lat[i - 1], lat[i - 1], lat[i], lat[i])
            out[i, j] = geod.polygon_area_perimeter(lons, lats)[0]
        return out

    @property
    @method_cacher
    def grid_type(self):
        """Return the type of grid: "reg" (lat/lon) or "ico" (dynamico)."""
        dims = dict(
            (d, d in self.dims) for d in ("lon", "lat", "cell", "nvertex")
        )
        if (
            dims["lon"]
            and dims["lat"]
            and not dims["cell"]
            and not dims["nvertex"]
        ):
            return "reg"
        elif (
            not dims["lon"]
            and not dims["lat"]
            and dims["cell"]
            and dims["nvertex"]
        ):
            return "ico"
        else:
            raise ValueError("Cannot guess LMDz grid type.")

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
