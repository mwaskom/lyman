import heapq
from itertools import product
import numpy as np
import nibabel as nib


class SurfaceMeasure(object):
    """Object for computing distance along a surface mesh.

    Adapted from Nikolaas Oosterhof's code in PyMVPA.

    """
    def __init__(self, verts, faces):
        """Initialize the measure object and cache vertex neighbors.

        Parameters
        ----------
        verts : n x 3 float array
            Array of vertex coordinates. Vertices will be identifed by their
            row index in this array.
        faces : n x 3 int array
            Array of triangles defined by triples of vertex ids.

        """
        self.verts = verts
        self.faces = faces
        self.n_v = len(verts)
        self.n_f = len(faces)

        face_ids = range(self.n_f)
        vert_ids = range(self.n_v)
        neighbors = {v: {} for v in vert_ids}

        for i, j in product(face_ids, range(3)):

            p, q = faces[i][j], faces[i][(j + 1) % 3]

            if p in neighbors and q in neighbors[p]:
                continue

            pv, qv = verts[[p, q]]

            d = pv[0] - qv[0], pv[1] - qv[1], pv[2] - qv[2]
            d = np.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)

            neighbors[p][q] = d
            neighbors[q][p] = d

        self.neighbors = neighbors

    @classmethod
    def from_file(cls, fname):
        """Initialize the measure object from a Freesurfer surface file."""
        f, v = nib.freesurfer.read_geometry(fname)
        return cls(f, v)

    def __call__(self, vert, maxdistance=np.inf):
        """Return the distances from input to other vertices.

        Parameters
        ----------
        vert : int
            ID of vertex to compute distances from.
        maxdistance : float
            Return distances to neighbors that are closer than this threshold.

        Returns
        -------
        f_dist : dict
            Keys are ids of other vertices and values are distances (vertex
            coordinate units) from ``vert`` to each vertex.

        """
        neighbors = self.neighbors

        t_dist = {vert: 0}
        f_dist = {}
        candidates = []

        heapq.heappush(candidates, (0, vert))

        while candidates:

            d, i = heapq.heappop(candidates)

            if i in f_dist:
                continue

            for n_i, n_d in neighbors[i].items():

                d_new = d + n_d

                if d_new > maxdistance:
                    continue

                if d_new < t_dist.get(n_i, np.inf):
                    t_dist[n_i] = d_new
                    heapq.heappush(candidates, (d_new, n_i))

            f_dist[i] = t_dist[i]

        return f_dist
