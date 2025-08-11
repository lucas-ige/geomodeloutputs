# Copyright (c) 2024-2025, Institut des Géosciences de l'Environnement, France.
#
# License: BSD 3-clause "new" or "revised" license (BSD-3-Clause).

"""Module geomodeloutputs: accessors to add functionality to datasets."""

import xarray as xr
from ._accessors_generic import GenericDatasetAccessor
from ._genutils import method_cacher


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

    def time_coord(self, varname):
        """Return the name of the time coordinate associated with variable."""
        return self.myself.time_coord(varname)
