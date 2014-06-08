import os
from nipype.utils.filemanip import fname_presuffix


def add_suffix(fname, suffix):
    """Insert a suffix into a filename before the extension."""
    out_fname = fname_presuffix(fname, suffix="_" + suffix,
                                use_ext=True)
    return out_fname


def nii_to_png(fname, suffix=""):
    """Return a path to write a local png based on an image."""
    out_fname = fname_presuffix(fname, suffix=suffix + ".png",
                                newpath=os.getcwd(),
                                use_ext=False)
    return out_fname
