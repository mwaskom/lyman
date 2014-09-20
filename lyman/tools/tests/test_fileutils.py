import nose.tools as nt
from .. import fileutils as fu


class TestAddSuffix(object):

    def test_fname_only(self):

        fname = "buz.nii"
        nt.assert_equal(fu.add_suffix(fname, "bang"), "buz_bang.nii")

    def test_gzipped(self):

        fname = "buz.nii.gz"
        nt.assert_equal(fu.add_suffix(fname, "bang"), "buz_bang.nii.gz")

    def test_absolute_path(self):

        fname = "/foo/bar/buz.nii"
        nt.assert_equal(fu.add_suffix(fname, "bang"), "/foo/bar/buz_bang.nii")

    def test_relative_path(self):

        fname = "foo/bar/buz.nii"
        nt.assert_equal(fu.add_suffix(fname, "bang"), "foo/bar/buz_bang.nii")
