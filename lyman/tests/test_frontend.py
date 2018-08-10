import os
from textwrap import dedent
from nipype import Workflow, Node, Function
from traits.api import TraitError
import pytest
from .. import frontend


class TestFrontend(object):

    @pytest.fixture
    def lyman_dir(self, execdir):

        lyman_dir = execdir.mkdir("lyman")
        os.environ["LYMAN_DIR"] = str(lyman_dir)

        scans = dedent("""
        subj01:
          sess01:
            exp_alpha: [run01, run02]
          sess02:
            exp_alpha: [run01]
            exp_beta: [run01, run02]
        subj02:
          sess01:
            exp_beta: [run01, run03]
        """)

        project = dedent("""
        data_dir = "../datums"
        voxel_size = (2.5, 2.5, 2.5)
        """)

        experiment = dedent("""
        tr = .72
        smooth_fwhm = 2.5
        """)

        model = dedent("""
        smooth_fwhm = 4
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

        yield lyman_dir

        lyman_dir.remove()

    def test_info(self, lyman_dir, execdir):

        info = frontend.info()
        assert info.data_dir == execdir.join("datums")
        assert info.scan_info == {
            "subj01": {"sess01": {"exp_alpha": ["run01", "run02"]},
                       "sess02": {"exp_alpha": ["run01"],
                                  "exp_beta": ["run01", "run02"]}},
            "subj02": {"sess01": {"exp_beta": ["run01", "run03"]}},
        }

        model_traits = frontend.ModelInfo().trait_get()
        assert info.trait_get(*model_traits.keys()) == model_traits

        info = frontend.info("exp_alpha")
        assert info.tr == .72
        assert info.smooth_fwhm == 2.5

        info = frontend.info("exp_alpha", "model_a")
        assert info.smooth_fwhm == 4
        assert info.contrasts == [("a-b", ["a", "b"], [1, -1])]

        with pytest.raises(TraitError):
            frontend.info("exp_alpha", "model_b")

        with pytest.raises(RuntimeError):
            frontend.info(model="model_b")

        lyman_dir_new = execdir.join("lyman2")
        lyman_dir.move(lyman_dir_new)

        info = frontend.info(lyman_dir=str(lyman_dir_new))
        assert info.voxel_size == (2.5, 2.5, 2.5)

        lyman_dir_new.move(execdir.join("lyman"))

        del os.environ["LYMAN_DIR"]
        info = frontend.info()

    def test_subjects(self, lyman_dir, execdir):

        subjects = frontend.subjects()
        assert subjects == ["subj01", "subj02"]

        subjects = frontend.subjects("subj01")
        assert subjects == ["subj01"]

        subjects = frontend.subjects(["subj01"])
        assert subjects == ["subj01"]

        subjects = frontend.subjects(["subj01", "subj02"])
        assert subjects == ["subj01", "subj02"]

        subj_file = lyman_dir.join("group_one.txt")
        with open(subj_file, "w") as fid:
            fid.write("subj01")

        subjects = frontend.subjects("group_one")
        assert subjects == ["subj01"]

        subjects = frontend.subjects(["group_one"])
        assert subjects == ["subj01"]

        with pytest.raises(RuntimeError):
            frontend.subjects(["subj01", "subj03"])

        with pytest.raises(RuntimeError):
            frontend.subjects(["subj01", "subj02"], ["sess01", "sess02"])

        with pytest.raises(RuntimeError):
            frontend.subjects(["subj02"], ["sess01", "sess02"])

        del os.environ["LYMAN_DIR"]
        subjects = frontend.subjects()
        assert subjects == []

    def test_execute(self, lyman_dir, execdir):

        info = frontend.info(lyman_dir=lyman_dir)

        def f(x):
            return x ** 2
        assert f(2) == 4

        n1 = Node(Function("x", "y", f), "n1")
        n2 = Node(Function("x", "y", f), "n2")

        wf = Workflow("test", base_dir=info.cache_dir)
        wf.connect(n1, "y", n2, "x")
        wf.inputs.n1.x = 2

        cache_dir = execdir.join("cache").join("test")

        class args(object):
            graph = False
            n_procs = 1
            debug = False
            clear_cache = True
            execute = True

        frontend.execute(wf, args, info)
        assert not cache_dir.exists()

        args.debug = True
        frontend.execute(wf, args, info)
        assert cache_dir.exists()

        args.debug = False
        info.remove_cache = False
        frontend.execute(wf, args, info)
        assert cache_dir.exists()

        args.execute = False
        res = frontend.execute(wf, args, info)
        assert res is None

        args.execute = True
        fname = str(execdir.join("graph").join("workflow.dot"))
        args.graph = fname
        res = frontend.execute(wf, args, info)
        assert res == fname[:-4] + ".svg"

        args.graph = True
        args.stage = "preproc"
        res = frontend.execute(wf, args, info)
        assert res == cache_dir.join("preproc.svg")

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
