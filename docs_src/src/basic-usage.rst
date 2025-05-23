.. Documentation of the geomodeloutputs Python package.
   Copyright (c) 2024-now, Institut des Géosciences de l'Environnement, France.
   License: CC BY 4.0

Basic usage
###########

Let us look at an example
=========================

Imagine you ran an Elmer/Ice simulation over Greenland, and you want to make a map of the surface elevation at the end
of the simulation (variable :code:`orog` in file :code:`output.nc`). Without geomodeloutputs, you might write a script
like this:

.. code-block:: python

  import xarray as xr
  import matplotlib.pyplot as plt

  ds = xr.open_dataset("output.nc")
  meshname = "greenland"
  tri = Triangulation(ds["x"][-1,:], ds["y"][-1,:], ds[meshname + "_face_nodes"])
  plt.tripcolor(tri, ds["orog"][-1,:])
  plt.savefig("orog.png")
  ds.close()

In order to plot Elmer/Ice output data, which are given on an unstructured mesh, you first have to create a
Triangulation object from three of the output's variables: x, y, and a third variable whose name depends on the name
that was given to the mesh when Elmer/Ice was run. To plot Elmer/Ice output data, you therefore have to:

 - remember how to correctly create the Triangulation instance.

 - know the name of the mesh, which you will probably have to hard-code in your script. Your script will therefore not
   be usable with outputs of other Elmer/Ice simulations, if they have a different mesh name.

Using geomodeloutputs, you might write a script like this instead:

.. code-block:: python

  import xarray as xr
  import geomodeloutputs
  import matplotlib.pyplot as plt

  ds = xr.open_dataset("output.nc").elmerice
  plt.tripcolor(ds.triangulation, ds["orog"][-1,:])
  plt.savefig("orog.png")
  ds.close()

In this example, you can see that:

 - the NetCDF file is opened, accessed, and closed in exactly the same way as in the previous example, except that we
   add the name of the model after calling `xr.open_dataset` to unlock the Elmer/Ice functionality.

 - the details of the calculation of the Triangulation instance are now hidden.

 - you do not need to know the name of the mesh (it is automatically detected by geomodeloutputs). Your script will
   therefore be usable "as is" with other Elmer/Ice output files.

In fact, if you need to know it, geomodeloutputs can give you the name of the mesh:

.. code-block:: python

  >>> print(ds.meshname)

  "greenland"

The examples above show the philosophy of geomodeloutputs:

 - do not change the functionality of xarray.

 - hide uninteresting technical details and automate their calculation.

 - make manipulating outputs from known models (Elmer/Ice, MAR, LMDz, WRF, ...) more convenient.

The general philosophy of geomodeloutputs
=========================================

The general philosophy of geomodeloutputs is to add functionality to NetCDF files opened as :code:`xarray.Dataset`
instances, without changing how xarray works.

The functionality added by geomodeloutputs can be accessed through new attributes of Dataset instances. There is one
such new attribute for each model supported by geomodeloutputs. Think of these attributes as drawers: one drawer for
each model. These drawers are actually called "accessors" in xarray jargon.

For example, the added functionality for Elmer/Ice outputs can be accessed via :code:`ds.elmerice.*`, and the added
functionality for LMDz outputs can be accessed via :code:`ds.lmdz.*`.

Each accessor also features the usual xarray interface. For example, :code:`ds.lmdz["snow"]` is equivalent to
:code:`ds["snow"]`.

The wizard
==========

Sometimes one wants to write code that is model-agnostic (ie. code that works transparently with outputs of different
models). For that, geomodeloutputs provides the :code:`wizard` accessor that works for all relevant models. For
example, :code:`ds.wizard.units_nice` correspond to :code:`ds.elmerice.units_nice` for Elmer/Ice outputs and to
:code:`ds.lmdz.units_nice` for LMDz outputs.
