.. Documentation of the geomodeloutputs Python package.
   Copyright (c) 2024-now, Institut des Géosciences de l'Environnement, France.

Installation instructions
#########################

Python dependencies
===================

This part needs to be completed.

Installation instructions
=========================

To install the geomodeloutputs Python package, open a terminal, move to the
directory where you want to install it (here we use :code:`~/python-packages`
as an example), and clone the git repository:

.. code-block:: sh

    git clone git@github.com:lucas-ige/geomodeloutputs.git ./geomodeloutputs

Then you need to make sure that python will find this directory when importing
packages. To do that, update the PYTHONPATH environment variable. If you use
the Bash shell (which is the default on many GNU/Linux distributions):

.. code-block:: sh

    echo "export \$PYTHONPATH=\$PYTHONPATH:~/python-packages" >> ~/.bash_profile

If you use the Z shell (which is the default on modern MacOS installation):

.. code-block:: sh

    echo "export \$PYTHONPATH=\$PYTHONPATH:~/python-packages" >> ~/.zprofile

Testing your installation
=========================

This part needs to be completed.
