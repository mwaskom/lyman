import numpy as np
import nibabel as nib

import pytest

from .. import surface


class TestSurfaceMeasure(object):

    @pytest.fixture
    def surfdata(self, execdir):

        verts = np.array([[0, 0, 0],
                          [1, 0, 0],
                          [1, 1, 1],
                          [2, 0, 0],
                          [2, 2, 2]], np.float)

        faces = np.array([[0, 1, 2],
                          [0, 2, 3],
                          [2, 3, 4]], np.int)

        sqrt2 = np.sqrt(2)
        sqrt3 = np.sqrt(3)
        sqrt8 = np.sqrt(8)

        neighbors = {0: {1: 1.0, 2: sqrt3, 3: 2.0},
                     1: {0: 1.0, 2: sqrt2},
                     2: {0: sqrt3, 1: sqrt2, 3: sqrt3, 4: sqrt3},
                     3: {0: 2.0, 2: sqrt3, 4: sqrt8},
                     4: {2: sqrt3, 3: sqrt8}}

        fname = execdir.join("test.mesh")
        nib.freesurfer.write_geometry(fname, verts, faces)

        surfdata = dict(
            verts=verts,
            faces=faces,
            neighbors=neighbors,
            fname=fname,
        )
        return surfdata

    def test_surface_measure_neighbors(self, surfdata):

        sm = surface.SurfaceMeasure(surfdata["verts"], surfdata["faces"])

        for v, v_n in sm.neighbors.items():
            assert v_n == pytest.approx(surfdata["neighbors"][v])

    def test_surface_measure_neighbors_from_file(self, surfdata):

        sm = surface.SurfaceMeasure.from_file(surfdata["fname"])

        for v, v_n in sm.neighbors.items():
            assert v_n == pytest.approx(surfdata["neighbors"][v])

    def test_surface_measure_distance(self, surfdata):

        sm = surface.SurfaceMeasure(surfdata["verts"], surfdata["faces"])

        n = surfdata["neighbors"]
        d = {0: 0,
             1: n[0][1],
             2: n[0][2],
             3: n[0][3],
             4: n[0][2] + n[2][4]}

        assert sm(0) == pytest.approx(d)

    def test_surface_measure_distance_maxdistance(self, surfdata):

        sm = surface.SurfaceMeasure(surfdata["verts"], surfdata["faces"])

        n = surfdata["neighbors"]
        d = {0: 0,
             1: n[0][1]}

        assert sm(0, maxdistance=1.1) == pytest.approx(d)

    def test_surface_measure_smoke_distances(self, surfdata):

        sm = surface.SurfaceMeasure(surfdata["verts"], surfdata["faces"])
        for v in range(sm.n_v):
            assert isinstance(sm(v), dict)
