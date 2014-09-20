import os
import os.path as op
import time
import tempfile
from nipype.interfaces.base import Bunch

import nose.tools as nt

from .. import submission


class TestSubmitCmdLine(object):

    def test_submit(self):

        tmp_dir = tempfile.gettempdir()
        tmp_file = op.join(tmp_dir, "tmp_{}".format(time.time()))
        cmdline = ["touch", tmp_file]
        runtime = Bunch(cwd=os.getcwd(), environ=os.environ)
        submission.submit_cmdline(runtime, cmdline)
        assert op.exists(tmp_file)
        os.remove(tmp_file)

    def test_stdout(self):

        cmdline = ["echo", "hello test"]
        runtime = Bunch(cwd=os.getcwd(), environ=os.environ)
        out = submission.submit_cmdline(runtime, cmdline)
        nt.assert_equal(out.stdout, "hello test\n")

    def test_stdout_addition(self):

        cmdline = ["echo", "oh why hello"]
        runtime = Bunch(stdout="hello test\n",
                        cwd=os.getcwd(), environ=os.environ)
        out = submission.submit_cmdline(runtime, cmdline)
        nt.assert_equal(out.stdout, "hello test\noh why hello\n")

    def test_runtime_error(self):

        tmp_dir = tempfile.gettempdir()
        tmp_file = op.join(tmp_dir, "i_am_not_a_file")
        cmdline = ["cat", tmp_file]
        runtime = Bunch(cwd=os.getcwd(), environ=os.environ)
        with nt.assert_raises(RuntimeError):
            out = submission.submit_cmdline(runtime, cmdline)
            error = "cat: {}: No such file or directory\n".format(tmp_file)
            nt.assert_equal(out.stderr, error)

    def test_cwd(self):

        cmdline = ["pwd"]
        home_dir = op.expanduser("~")
        runtime = Bunch(cwd=home_dir, environ=os.environ)
        out = submission.submit_cmdline(runtime, cmdline)
        nt.assert_equal(out.stdout, home_dir + "\n")
