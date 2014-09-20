import subprocess as sp


def submit_cmdline(runtime, cmdline):
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
                    env=runtime.environ)

    stdout, stderr = proc.communicate()

    runtime.stdout += stdout
    runtime.stderr += stderr
    runtime.cmdline += cmdline
    runtime.returncode += proc.returncode

    if proc.returncode is None or proc.returncode != 0:
        message = "Command:\n" + runtime.cmdline + "\n"
        message += "Standard output:\n" + runtime.stdout + "\n"
        message += "Standard error:\n" + runtime.stderr + "\n"
        message += "Return code: " + str(runtime.returncode)
        raise RuntimeError(message)

    return runtime
