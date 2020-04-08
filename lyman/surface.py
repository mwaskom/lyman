import os
import os.path as op
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

    @classmethod
    def from_names(cls, subj, hemi, surf, subjects_dir=None):
        """Initialize from Freesurfer-style names."""
        if subjects_dir is None:
            subjects_dir = os.environ["SUBJECTS_DIR"]
        fname = op.join(subjects_dir, subj, "surf", "{}.{}".format(hemi, surf))
        return cls.from_file(fname)

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


def vol_to_surf(data_img, subject, hemi, surf="graymid",
                null_value=0, cortex_only=True, subjects_dir=None):
    """Sample data from a volume image onto a surface mesh.

    This function assumes that ``data_img`` is in register with the anatomy
    and loads data from the Freesurfer data directory hierarchy.

    Parameters
    ----------
    data_img : nibabel image
        Input volume image; can be 3D or 4D.
    subject : string
        Subject ID to locate data in data directory.
    hemi : lh | rh
        Hemisphere code; with ``surf`` finds surface mesh geometry file.
    surf : string
        Surface name, with ``hemi`` finds surface mesh geometry file.
    null_value : float
        Value to use for surface vertices that are outside the volume field
        of view.
    cortex_only : bool
        If True, vertices outside the Freesurfer-defined cortex label are
        assigned ``null_value``.
    subjects_dir : string
        Path to the Freesurfer data directory root; if absent, get from the
        SUBJECTS_DIR environment variable.

    Returns
    -------
    surf_data : n_vert (x n_tp) array
        Data as a vector (for 3D inputs) or matrix (for 4D inputs).

    """
    from numpy.linalg import inv

    # Locate the Freesurfer files
    if subjects_dir is None:
        subjects_dir = os.environ["SUBJECTS_DIR"]
    anat_file = op.join(subjects_dir, subject, "mri", "orig.mgz")
    surf_file = op.join(subjects_dir, subject,
                        "surf", "{}.{}".format(hemi, surf))

    # Convert input to MGH format and load volumes
    data = data_img.get_fdata()
    data_img = nib.MGHImage(data, data_img.affine, data_img.header)
    anat_img = nib.load(anat_file)

    # Get affines that map to scanner and tkr spaces
    Tanat = anat_img.affine
    Tdata = data_img.affine
    Kanat = anat_img.header.get_vox2ras_tkr()

    # Compute the full transform
    # This takes surface xyz -> anat ijk -> scanner xyz -> data ijk
    xfm = inv(Tdata).dot(Tanat).dot(inv(Kanat))

    # Find volume coordinates corresponding to the surface vertices
    vertices, _ = nib.freesurfer.read_geometry(surf_file)
    vertices = nib.affines.apply_affine(xfm, vertices)
    i, j, k = np.round(vertices).astype(int).T

    # Find a mask for surfaces vertices that are in the volume FOV
    ii, jj, kk = data.shape[:3]
    fov = (np.in1d(i, np.arange(ii))
           & np.in1d(j, np.arange(jj))
           & np.in1d(k, np.arange(kk)))

    # Initialize the output array
    n_v = len(vertices)
    if len(data.shape) == 3:
        shape = (n_v,)
    else:
        shape = (n_v, data.shape[-1])
    surf_data = np.full(shape, null_value, np.float)

    # Sample from the volume array into the surface array
    surf_data[fov] = data[i[fov], j[fov], k[fov]]

    # Restrict vertices that are not part of the cortical surface
    if cortex_only:
        label_file = op.join(subjects_dir, subject, "label",
                             "{}.cortex.label".format(hemi))
        cortex_verts = nib.freesurfer.read_label(label_file)
        noncortical = np.ones(n_v, np.bool)
        noncortical[cortex_verts] = False
        surf_data[noncortical] = null_value

    return surf_data
