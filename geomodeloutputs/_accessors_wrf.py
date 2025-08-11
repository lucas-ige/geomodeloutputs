# Copyright (c) 2024-2025, Institut des Géosciences de l'Environnement, France.
#
# License: BSD 3-clause "new" or "revised" license (BSD-3-Clause).

"""Module geomodeloutputs: accessors to add functionality to datasets.

This file adds an accessor for WRF and WRF-Chem model outputs.

"""

import warnings
import xarray as xr
import pyproj
import cartopy
from ._accessors_generic import GenericDatasetAccessor


@xr.register_dataset_accessor("wrf")
class WRFDatasetAccessor(GenericDatasetAccessor):
    """Accessor for WRF and WRF-Chem outputs."""

    @property
    def crs_pyproj(self):
        """Return the CRS (pyproj) corresponding to dataset."""
        if self.attrs["POLE_LON"] != 0:
            raise ValueError("Invalid POLE_LON: %f." % self.attrs["POLE_LON"])
        proj = self.attrs["MAP_PROJ"]
        if proj in (0, 1, 102, 3, 4, 5, 6, 105, 203):
            raise NotImplementedError("Projection code %d." % proj)
        proj = {2: "polarstereo"}[proj]
        return getattr(self, "_crs_pyproj_%s" % proj)

    @property
    def _crs_pyproj_polarstereo(self):
        """Return the CRS (pyproj) corresponding to dataset.

        This method handles the specific case of polar stereographic
        projections.

        """
        if self.attrs["MAP_PROJ"] != 2:
            raise ValueError("Invalid value for MAP_PROJ.")
        if self.attrs["MAP_PROJ_CHAR"] != "Polar Stereographic":
            raise ValueError("Invalid value for MAP_PROJ_CHAR.")
        if self.attrs["STAND_LON"] != self.attrs["CEN_LON"]:
            raise ValueError("Inconsistency in central longitude values.")
        if self.attrs["POLE_LAT"] not in (90, -90):
            raise ValueError("Invalid value for attribute POLE_LAT.")
        for which in ("TRUELAT1", "TRUELAT2", "MOAD_CEN_LAT"):
            if round(self.attrs[which], 4) != round(self.attrs["CEN_LAT"], 4):
                raise ValueError("Inconsistency in true latitude values.")
        proj = dict(
            proj="stere",
            lat_0=self.attrs["POLE_LAT"],
            lat_ts=self.attrs["TRUELAT1"],
            lon_0=self.attrs["CEN_LON"],
        )
        return pyproj.CRS.from_dict(proj)

    @property
    def crs_cartopy(self):
        """Return the CRS (cartopy) corresponding to dataset."""
        # We let self.crs_pyproj do all the quality checking
        crs_pyproj = self.crs_pyproj
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proj = crs_pyproj.to_dict()
        if proj["proj"] == "stere":
            return cartopy.crs.Stereographic(
                central_longitude=proj["lon_0"],
                central_latitude=proj["lat_0"],
                false_easting=proj["x_0"],
                false_northing=proj["y_0"],
                true_scale_latitude=proj["lat_ts"],
                globe=cartopy.crs.Globe(datum=proj["datum"]),
            )
        else:
            raise ValueError("Unsupported projection: %s." % proj["proj"])
