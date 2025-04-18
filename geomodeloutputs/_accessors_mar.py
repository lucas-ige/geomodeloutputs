# Copyright (c) 2024-2025, Institut des Géosciences de l'Environnement, France.
#
# License: BSD 3-clause "new" or "revised" license (BSD-3-Clause).

"""Module geomodeloutputs: accessors to add functionality to datasets."""

import xarray as xr
import pyproj
import cartopy
from ._accessors_generic import GenericDatasetAccessor

@xr.register_dataset_accessor("mar")
class MARDatasetAccessor(GenericDatasetAccessor):

    def time_coord(self, varname):
        """Return the name of the time coordinate associated with variable."""
        return "time" if "time" in self[varname].dims else None

    @property
    def crs_pyproj(self):
        """Return the CRS (pyproj) corresponding to the file."""
        try:
            prms = self.attrs["mapping"]
        except KeyError:
            prms = self.attrs["projection"]
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
            prms = self.attrs["mapping"]
        except KeyError:
            prms = self.attrs["projection"]
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
