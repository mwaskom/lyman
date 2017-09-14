import os
from textwrap import dedent

import pytest

from traits.api import TraitError

from .. import frontend


class TestFrontend(object):

    def test_load_info_from_module(self, execdir):

        lyman_dir = execdir.mkdir("lyman")

        # Write a Python module to test import from disk
        module_text = dedent("""
        foo = "a"
        bar = 3
        buz = [1, 2, 3]
        """)
        module_fname = lyman_dir.join("test.py")
        with open(module_fname, "w") as fid:
            fid.write(module_text)

        expected = dict(foo="a", bar=3, buz=[1, 2, 3])

        module_vars = frontend.load_info_from_module("test", lyman_dir)
        assert module_vars == expected

        # Remove the file to test import from memory
        os.remove(module_fname)
        module_vars = frontend.load_info_from_module("test", lyman_dir)
        assert module_vars == expected

    def test_check_extra_vars(self):

        with pytest.raises(RuntimeError):
            module_vars = {"not_a_valid_trait": True}
            frontend.check_extra_vars(module_vars, frontend.ProjectInfo)

    @pytest.fixture
    def lyman_dir(self, execdir):

        lyman_dir = execdir.mkdir("lyman")

        scans = dedent("""
        subj01:
          sess01:
            exp_alpha: [run01, run02]
          ess01:
            exp_alpha: [run01]
            exp_beta: [run01, run01]
        """)

        project = dedent("""
        data_dir = "../datums"
        voxel_size = (2.5, 2.5, 2.5)
        """)

        experiment = dedent("""
        tr = .72
        """)

        model = dedent("""
        tr = 1.5
        contrasts = [("a-b", ["a", "b"], [1, -1])]
        """)

        model_bad = dedent("""
        contrasts = ["a-b", "b-a"]
        """)

        with open(lyman_dir.join("scans.yaml"), "w") as fid:
            fid.write(scans)

        with open(lyman_dir.join("project.py"), "w") as fid:
            fid.write(project)

        with open(lyman_dir.join("exp_alpha.py"), "w") as fid:
            fid.write(experiment)

        with open(lyman_dir.join("exp_alpha-model_a.py"), "w") as fid:
            fid.write(model)

        with open(lyman_dir.join("exp_alpha-model_b.py"), "w") as fid:
            fid.write(model_bad)

        return lyman_dir

    def test_lyman_info(self, lyman_dir, execdir):

        os.environ["LYMAN_DIR"] = str(lyman_dir)

        info = frontend.lyman_info()
        assert info.data_dir == execdir.join("datums")

        model_traits = frontend.ModelInfo().trait_get()
        assert info.trait_get(*model_traits.keys()) == model_traits

        info = frontend.lyman_info("exp_alpha")
        assert info.tr == .72

        info = frontend.lyman_info("exp_alpha", "model_a")
        assert info.tr == 1.5
        assert info.contrasts == [("a-b", ["a", "b"], [1, -1])]

        with pytest.raises(TraitError):
            frontend.lyman_info("exp_alpha", "model_b")

        lyman_dir_new = execdir.join("lyman2")
        lyman_dir.move(lyman_dir_new)

        info = frontend.lyman_info(lyman_dir=str(lyman_dir_new))
        assert info.voxel_size == (2.5, 2.5, 2.5)
