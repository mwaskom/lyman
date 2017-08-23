import os.path as op
import subprocess as sp
import nibabel as nib
from nipype.interfaces.base import BaseInterface


# TODO rename to LymanInterface, move to utils
class SimpleInterface(BaseInterface):

    def __init__(self, **inputs):

        super(SimpleInterface, self).__init__(**inputs)
        self._results = {}

    def _list_outputs(self):

        return self._results

    def define_output(self, field, fname):

        fname = op.abspath(fname)
        self._results[field] = fname
        return fname

    def write_image(self, field, fname, data, affine=None, header=None):

        fname = self.define_output(field, fname)
        if isinstance(data, nib.Nifti1Image):
            img = data
        else:
            img = nib.Nifti1Image(data, affine, header)
        img.to_filename(fname)
        return img

    def submit_cmdline(self, runtime, cmdline, **results):
        """Submit a command-line job and capture the output."""

        for attr in ["stdout", "stderr", "cmdline"]:
            if not hasattr(runtime, attr):
                setattr(runtime, attr, "")
        if not hasattr(runtime, "returncode"):
            runtime.returncode = 0
        elif runtime.returncode is None:
            runtime.returncode = 0

        if isinstance(cmdline, list):
            cmdline = " ".join(cmdline)

        proc = sp.Popen(cmdline,
                        stdout=sp.PIPE,
                        stderr=sp.PIPE,
                        shell=True,
                        cwd=runtime.cwd,
                        env=runtime.environ,
                        universal_newlines=True)

        stdout, stderr = proc.communicate()

        runtime.stdout += "\n" + stdout + "\n"
        runtime.stderr += "\n" + stderr + "\n"
        runtime.cmdline += "\n" + cmdline + "\n"
        runtime.returncode += proc.returncode

        if proc.returncode is None or proc.returncode != 0:
            message = "\n\nCommand:\n" + runtime.cmdline + "\n"
            message += "Standard output:\n" + runtime.stdout + "\n"
            message += "Standard error:\n" + runtime.stderr + "\n"
            message += "Return code: " + str(runtime.returncode)
            raise RuntimeError(message)

        for field, fname in results.items():
            self._results[field] = fname

        return runtime


def generate_iterables(scan_info, subjects, experiment, session=None):
    # TODO This is preproc-specific so move it there
    # TODO additionally we want to expand this to specify > 1 session
    # TODO also change the order of subjects and experiment?
    subject_iterables = subjects
    session_iterables = dict()
    run_iterables = dict()

    for subj in subjects:

        session_iterables[subj] = []

        for sess in scan_info[subj]:

            sess_key = subj, sess

            if session is not None and sess != session:
                continue

            if experiment in scan_info[subj][sess]:

                session_iterables[subj].append(sess_key)
                run_iterables[sess_key] = []

                for run in scan_info[subj][sess][experiment]:
                    run_key = subj, sess, run
                    run_iterables[sess_key].append(run_key)

    return subject_iterables, session_iterables, run_iterables
