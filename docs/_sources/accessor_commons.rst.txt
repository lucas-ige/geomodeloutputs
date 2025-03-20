.. Documentation of the geomodeloutputs Python package.
   Copyright (c) 2025-now, Institut des Géosciences de l'Environnement, France.
   License: CC BY 4.0

Utility functions
##################

.. autofunction:: geomodeloutputs.xarray_accessors.open_dataset
    :no-index:

.. autofunction:: geomodeloutputs.xarray_accessors.open_mfdataset
    :no-index:

Shared functionality
####################

Some functionality is implemented for all models. For example, the method :code:`units_nice` is common to all models,
so you can use, :code:`ds.lmdz.units_nice`, :code:`ds.elmerice.units_nice`, and of course :code:`ds.wizard.units_nice`.

.. automethod:: geomodeloutputs.xarray_accessors.GenericDatasetAccessor.units_nice
    :no-index:

.. autoproperty:: geomodeloutputs.xarray_accessors.GenericDatasetAccessor.crs_pyproj
    :no-index:

.. autoproperty:: geomodeloutputs.xarray_accessors.GenericDatasetAccessor.crs_cartopy
    :no-index:
