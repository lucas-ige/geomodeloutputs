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

  ds = xr.open_dataset("output.nc")
  plt.tripcolor(ds.elmerice.triangulation, ds["orog"][-1,:])
  plt.savefig("orog.png")
  ds.close()

In this example, you can see that:

 - the NetCDF file is opened, accessed, and closed in exactly the same way as in the previous example.

 - the details of the calculation of the Triangulation instance are now hidden.

 - you do not need to know the name of the mesh (it is automatically detected by geomodeloutputs). Your script will
   therefore be usable "as is" with other Elmer/Ice output files.

In fact, if you need to know it, geomodeloutputs can give you the name of the mesh:

.. code-block:: python

  >>> print(ds.elmerice.meshname)

  "greenland"

The examples above show the philosophy of geomodeloutputs:

 - do not change the functionality of xarray.

 - hide uninteresting technical details and automate their calculation.

 - make manipulating outputs from known models (Elmer/Ice, MAR, LMDz, WRF, ...) more convenient.
