# Copyright (c) 2024-2025, Institut des Géosciences de l'Environnement, France.
#
# License: BSD 3-clause "new" or "revised" license (BSD-3-Clause).

"""Module geomodeloutputs: init file."""

import sys

if sys.version_info.major != 3:
    raise RuntimeError("The geomodelouputs package only works with Python 3.")

from ._genutils import open_dataset, open_mfdataset
