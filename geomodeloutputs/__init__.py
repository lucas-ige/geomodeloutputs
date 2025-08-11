# Copyright (c) 2024-2025, Institut des Géosciences de l'Environnement, France.
#
# License: BSD 3-clause "new" or "revised" license (BSD-3-Clause).

"""Module geomodeloutputs: init file."""

import sys

if sys.version_info.major != 3:
    raise RuntimeError("The geomodelouputs package only works with Python 3.")

from ._genutils import open_dataset, open_mfdataset
from ._accessors_generic import WizardDatasetAccessor
from ._accessors_elmerice import ElmerIceDatasetAccessor
from ._accessors_lmdz import LMDzDatasetAccessor
from ._accessors_mar import MARDatasetAccessor
from ._accessors_wrf import WRFDatasetAccessor
