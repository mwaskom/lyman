import os
from nipype.utils.filemanip import split_filename, fname_presuffix


def add_suffix(fname, suffix):
    """Insert a suffix into a filename before the extension."""
    path, name, ext = split_filename(fname)
    if path:
        return "{}/{}_{}{}".format(path, name, suffix, ext)
    else:    
        return "{}_{}{}".format(name, suffix, ext)


def nii_to_png(fname, suffix=""):
    """Return a path to write a local png based on an image."""
    out_fname = fname_presuffix(fname, suffix=suffix + ".png",
                                newpath=os.getcwd(),
                                use_ext=False)
    return out_fname
