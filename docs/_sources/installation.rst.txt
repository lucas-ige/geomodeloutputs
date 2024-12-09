.. Documentation of the geomodeloutputs Python package.
   Copyright (c) 2024-now, Institut des Géosciences de l'Environnement, France.
   License: CC BY 4.0

Installation instructions
#########################

Python dependencies
===================

The geomodeloutputs package only works with Python 3. It imports the following third-party packages, which must
therefore be installed:

 * numpy
 * pandas
 * xarray
 * pyproj
 * matplotlib
 * cartopy

Any version of these packages should be fine, as long as they are not obscenely old.

Installing the package
======================

To install the geomodeloutputs package, open a terminal, move to the directory where you want to install it (here we
use :code:`~/python-packages` as an example), and clone the git repository:

.. code-block:: sh

  git clone git@github.com:lucas-ige/geomodeloutputs.git ./geomodeloutputs

Then make sure that python will find this directory when importing packages. To do that, update the PYTHONPATH
environment variable. If you use the Bash shell (which is the default on many GNU/Linux distributions):

.. code-block:: sh

  echo "export \$PYTHONPATH=\$PYTHONPATH:~/python-packages" >> ~/.bash_profile

If you use the Z shell (which is the default on modern MacOS installations):

.. code-block:: sh

  echo "export \$PYTHONPATH=\$PYTHONPATH:~/python-packages" >> ~/.zprofile

Testing your installation
=========================

Open a Python shell and import geomodeloutputs. If it works, you can start using it!
