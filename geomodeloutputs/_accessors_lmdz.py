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
from ._accessors_common import CommonDatasetAccessor


@xr.register_dataset_accessor("lmdz")
class LMDzDatasetAccessor(CommonDatasetAccessor):
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
        return cartopy.crs.Geodetic()

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
        geod = pyproj.Geod(ellps="WGS84")
        if self.grid_type == "reg":
            lon, lat = self.varnames_lonlat
            lon, lat = self[lon].values, self[lat].values
            nlon, nlat = lon.size, lat.size
            out = np.full([nlat, nlon], np.nan)
            for i, j in itertools.product(
                range(1, nlat - 1), range(1, nlon - 1)
            ):
                lons = (lon[j - 1], lon[j], lon[j], lon[j - 1])
                lats = (lat[i - 1], lat[i - 1], lat[i], lat[i])
                out[i, j] = geod.polygon_area_perimeter(lons, lats)[0]
        elif self.grid_type == "ico":
            lon, lat = self.varnames_lonlat_bounds
            lon, lat = self[lon].values, self[lat].values
            out = np.full(self.ncells, np.nan)
            for i in range(self.ncells):
                out[i] = geod.polygon_area_perimeter(lon[i, :], lat[i, :])[0]
        else:
            raise ValueError("Unknown grid type: %s." % self.grid_type)
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

    @property
    @method_cacher
    def nbp(self):
        """Return the value of NBP for given grid (Dynamico grid only)."""
        if self.grid_type != "ico":
            raise ValueError("This property is only valid for Dynamico grids.")
        return {16002: 40, 36002: 60}[self.ncells]
