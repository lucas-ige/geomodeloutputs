# Copyright (c) 2024-2025, Institut des Géosciences de l'Environnement, France.
#
# License: BSD 3-clause "new" or "revised" license (BSD-3-Clause).

"""Module geomodeloutputs: accessors to add functionality to datasets."""

import numpy as np
import xarray as xr
import pyproj
from matplotlib.tri import Triangulation
import cartopy
from ._genutils import method_cacher
from ._accessors_generic import GenericDatasetAccessor

@xr.register_dataset_accessor("elmerice")
class ElmerIceDatasetAccessor(GenericDatasetAccessor):

    @property
    @method_cacher
    def epsg(self):
        """Return the EPSG code associated with file."""
        epsg = self.attrs["projection"].split(":")
        # Here we account for a typo (espg) in the XIOS configuration files
        # that were used in ISMIP6 simulations, and potentially others
        if len(epsg) != 2 or epsg[0] not in ("epsg", "espg"):
            raise ValueError("Invalid value for projection global attribute.")
        return int(epsg[1])

    @property
    @method_cacher
    def icesheet(self):
        """Return name of icesheet, inferred from global attributes."""
        if self.epsg == 3031:
            return "Antarctica"
        elif self.epsg == 3413:
            return "Greenland"
        else:
            raise RuntimeError("Could not infer name of icesheet.")

    @property
    @method_cacher
    def crs_pyproj(self):
        """Return the CRS (pyproj) corresponding to dataset."""
        return pyproj.CRS.from_epsg(self.epsg)

    @property
    @method_cacher
    def crs_cartopy(self):
        """Return the CRS (cartopy) corresponding to the file."""
        if self.epsg == 3031:
            return cartopy.crs.SouthPolarStereo(
                central_longitude=0, true_scale_latitude=-71)
        elif self.epsg == 3413:
            return cartopy.crs.NorthPolarStereo(
                central_longitude=-45, true_scale_latitude=70)
        else:
            raise ValueError("Unsupported projection: %d." % self.epsg)

    @property
    @method_cacher
    def meshname(self):
        """Return the name of the mesh (or None if it cannot be guessed)."""
        dims = self.sizes.keys()
        candidates = [d for d in dims if len(d) > 6 and
                      d.startswith("n") and d.endswith("_edge")]
        if len(candidates) != 1:
            return None
        meshname = candidates[0][1:-5]
        for which in ("face", "node", "vertex"):
            if "n%s_%s" % (meshname, which) not in dims:
                return None
        return meshname

    @property
    @method_cacher
    def dimname_edge(self):
        """Return the name of the dimension that holds the number of edges."""
        name = self.meshname
        if name is not None:
            name = "n%s_edge" % name
        return name

    @property
    @method_cacher
    def dimname_face(self):
        """Return the name of the dimension that holds the number of faces."""
        name = self.meshname
        if name is not None:
            name = "n%s_face" % name
        return name

    @property
    @method_cacher
    def dimname_node(self):
        """Return the name of the dimension that holds the number of nodes."""
        name = self.meshname
        if name is not None:
            name = "n%s_node" % name
        return name

    @property
    @method_cacher
    def dimname_vertex(self):
        """Return the name of the dim. that holds the number of vertices."""
        name = self.meshname
        if name is not None:
            name = "n%s_vertex" % name
        return name

    @property
    @method_cacher
    def triangulation(self):
        """Return Triangulation object corresponding to data."""
        return Triangulation(
            self["x"].values[0,:],
            self["y"].values[0,:],
            self[self.meshname + "_face_nodes"].values)

    @property
    @method_cacher
    def map_face_node(self):
        """An array giving, for each face, the indices of its vertices.

        The indices are given in Python convention (ie. starting at 0).

        """
        n_faces = self.sizes[self.dimname_face]
        n_vertices = self.sizes[self.dimname_vertex]
        varname = self.meshname + "_face_nodes"
        out = np.array(self[varname].values)
        if out.shape != (n_faces, n_vertices):
            raise ValueError("Map array has invalid shape.")
        start_index = int(self[varname].start_index)
        if start_index > 0:
            out -= start_index
        elif start_index != 0:
            raise ValueError("Negative start index.")
        return out

    def node2face(self, values):
        """Convert given node values to face values.

        Parameters
        ----------
        values : numpy.array
            The array of node values to convert.

        Returns
        -------
        numpy.array
            The corresponding array of face values.

        """
        if len(values) != self.sizes[self.dimname_node]:
            raise ValueError("Bad length for input.")
        map_ = self.map_face_node
        out = np.zeros(map_.shape[0])
        for i in range(len(out)):
            nodevals = [values[v] for v in map_[i,:]]
            out[i] = sum(nodevals) / len(nodevals)
        return out
