import nibabel as nib
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

    def test_surface_measure_neighbors_from_names(self, meshdata):

        sm = surface.SurfaceMeasure.from_names(
            meshdata["subj"], meshdata["hemi"], meshdata["surf"]
        )

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

        anat_img = nib.load(template["anat_file"])
        vert_img = nib.load(template["surf_file"])

        v, _ = nib.freesurfer.read_geometry(template["mesh_files"][0])
        n_verts = len(v,)
        n_frames = vert_img.shape[-1]

        surf_data = surface.vol_to_surf(
            anat_img, template["subject"], "lh", template["mesh_name"],
        )

        assert surf_data.shape == (n_verts,)

        null_value = -1
        surf_data = surface.vol_to_surf(
            anat_img, template["subject"], "lh", template["mesh_name"],
            null_value=null_value,
        )

        cortex = nib.freesurfer.read_label(template["label_files"]["lh"])
        for i, val in enumerate(surf_data):
            if i not in cortex.flat:
                assert val == null_value

        surf_data = surface.vol_to_surf(
            vert_img, template["subject"], "lh", template["mesh_name"],
        )

        assert surf_data.shape == (n_verts, n_frames)
