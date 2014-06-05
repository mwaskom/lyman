from nipype.utils.filemanip import split_filename


def add_suffix(fname, suffix):

    path, name, ext = split_filename(fname)
    if path:
        return "{}/{}_{}{}".format(path, name, suffix, ext)
    else:    
        return "{}_{}{}".format(name, suffix, ext)
