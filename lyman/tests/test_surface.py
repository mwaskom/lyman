import pytest

from .. import surface


class TestSurfaceMeasure(object):

    def test_surface_measure_neighbors(self, meshdata):

        sm = surface.SurfaceMeasure(meshdata["verts"], meshdata["faces"])

        for v, v_n in sm.neighbors.items():
            assert v_n == pytest.approx(meshdata["neighbors"][v])

    def test_surface_measure_neighbors_from_file(self, meshdata):

        sm = surface.SurfaceMeasure.from_file(meshdata["fname"])

        for v, v_n in sm.neighbors.items():
            assert v_n == pytest.approx(meshdata["neighbors"][v])

    def test_surface_measure_distance(self, meshdata):

        sm = surface.SurfaceMeasure(meshdata["verts"], meshdata["faces"])

        n = meshdata["neighbors"]
        d = {0: 0,
             1: n[0][1],
             2: n[0][2],
             3: n[0][3],
             4: n[0][2] + n[2][4]}

        assert sm(0) == pytest.approx(d)

    def test_surface_measure_distance_maxdistance(self, meshdata):

        sm = surface.SurfaceMeasure(meshdata["verts"], meshdata["faces"])

        n = meshdata["neighbors"]
        d = {0: 0,
             1: n[0][1]}

        assert sm(0, maxdistance=1.1) == pytest.approx(d)

    def test_surface_measure_smoke_distances(self, meshdata):

        sm = surface.SurfaceMeasure(meshdata["verts"], meshdata["faces"])
        for v in range(sm.n_v):
            assert isinstance(sm(v), dict)


class TestVolToSurf(object):

    def test_vol_to_surf(self, template):

        pass
