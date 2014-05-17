"""
Estimate nonlinear normalization parameters from Freesurfer conformed space
to FSL's nonlinear MNI152 target using FLIRT and FNIRT.

See the docstring for the create_normalization_workflow function for more
information about the processing.

"""
import os
import os.path as op
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
import nibabel as nib
from nipy.labs import viz
import seaborn as sns

from nipype import IdentityInterface, Function, DataGrabber, DataSink
from nipype.interfaces import fsl
from nipype.interfaces import freesurfer as fs
from nipype import Node, Workflow


def create_anatwarp_workflow(data_dir=None, subjects=None, name="anatwarp"):
    """Set up the anatomical normalzation workflow.

    Your anatomical data must have been processed in Freesurfer.
    Unlike most lyman workflows, the DataGrabber and DataSink
    nodes are hardwired within the returned workflow, as this
    tightly integrates with the Freesurfer subjects directory
    structure.

    Parameters
    ----------
    data_dir : path
        top level of data hierarchy/FS subjects directory
    subjects : list of strings
        list of subject IDs
    name : alphanumeric string, optional
        workflow name

    """
    if data_dir is None:
        data_dir = os.environ["SUBJECTS_DIR"]
    if subjects is None:
        subjects = []

    # Get target images
    target_brain = fsl.Info.standard_image("avg152T1_brain.nii.gz")
    target_head = fsl.Info.standard_image("avg152T1.nii.gz")
    hires_head = fsl.Info.standard_image("MNI152_T1_1mm.nii.gz")
    target_mask = fsl.Info.standard_image(
        "MNI152_T1_2mm_brain_mask_dil.nii.gz")
    fnirt_cfg = os.path.join(
        os.environ["FSLDIR"], "etc/flirtsch/T1_2_MNI152_2mm.cnf")

    # Subject source node
    subjectsource = Node(IdentityInterface(fields=["subject_id"]),
                         iterables=("subject_id", subjects),
                         name="subjectsource")

    # Grab recon-all outputs
    datasource = Node(DataGrabber(infields=["subject_id"],
                                  outfields=["aseg", "head"],
                                  base_directory=data_dir,
                                  sort_filelist=True,
                                  template="%s/mri/%s.mgz"),
                      name="datagrabber")
    datasource.inputs.template_args = dict(aseg=[["subject_id", "aparc+aseg"]],
                                           head=[["subject_id", "orig"]])

    # Convert images to nifti storage and float representation
    cvtaseg = Node(fs.MRIConvert(out_type="niigz"),
                   name="convertaseg")

    cvthead = Node(fs.MRIConvert(out_type="niigz", out_datatype="float"),
                   name="converthead")

    # Turn the aparc+aseg into a brainmask
    makemask = Node(fs.Binarize(dilate=4, erode=3, min=0.5),
                    name="makemask")

    # Extract the brain from the orig.mgz using the mask
    skullstrip = Node(fsl.ApplyMask(),
                      name="skullstrip")

    # FLIRT brain to MNI152_brain
    flirt = Node(fsl.FLIRT(reference=target_brain),
                 name="flirt")

    sw = [-180, 180]
    for dim in ["x", "y", "z"]:
        setattr(flirt.inputs, "searchr_%s" % dim, sw)

    # FNIRT head to MNI152
    fnirt = Node(fsl.FNIRT(ref_file=target_head,
                           refmask_file=target_mask,
                           config_file=fnirt_cfg,
                           fieldcoeff_file=True),
                 name="fnirt")

    # Warp and rename the images
    warpbrain = Node(fsl.ApplyWarp(ref_file=target_head,
                                   interp="spline",
                                   out_file="brain_warp.nii.gz"),
                     name="warpbrain")

    warpbrainhr = Node(fsl.ApplyWarp(ref_file=hires_head,
                                     interp="spline",
                                     out_file="brain_warp_hires.nii.gz"),
                       name="warpbrainhr")

    # Generate a png summarizing the registration
    checkreg = Node(Function(input_names=["in_file"],
                             output_names=["out_file"],
                             function=warp_report,
                             imports=[
                                "import os",
                                "import os.path as op",
                                "import numpy as np",
                                "import scipy as sp",
                                "import matplotlib.pyplot as plt",
                                "import matplotlib.image as mplimg",
                                "import nibabel as nib",
                                "from nipy.labs import viz",
                                "import seaborn as sns"]),
                    name="checkreg")

    # Save relevant files to the data directory
    datasink = Node(DataSink(base_directory=data_dir,
                             parameterization=False,
                             substitutions=[
                             ("orig_out_masked_flirt.mat", "affine.mat"),
                             ("orig_out_fieldwarp", "warpfield"),
                             ("orig_out_masked", "brain"),
                             ("orig_out", "T1")]),
                    name="datasink")

    # Define and connect the workflow
    # -------------------------------

    normalize = Workflow(name=name)

    normalize.connect([
        (subjectsource, datasource,
            [("subject_id", "subject_id")]),
        (datasource, cvtaseg,
            [("aseg", "in_file")]),
        (datasource, cvthead,
            [("head", "in_file")]),
        (cvtaseg, makemask,
            [("out_file", "in_file")]),
        (cvthead, skullstrip,
            [("out_file", "in_file")]),
        (makemask, skullstrip,
            [("binary_file", "mask_file")]),
        (skullstrip, flirt,
            [("out_file", "in_file")]),
        (flirt, fnirt,
            [("out_matrix_file", "affine_file")]),
        (cvthead, fnirt,
            [("out_file", "in_file")]),
        (skullstrip, warpbrain,
            [("out_file", "in_file")]),
        (fnirt, warpbrain,
            [("fieldcoeff_file", "field_file")]),
        (skullstrip, warpbrainhr,
            [("out_file", "in_file")]),
        (fnirt, warpbrainhr,
            [("fieldcoeff_file", "field_file")]),
        (warpbrainhr, checkreg,
            [("out_file", "in_file")]),
        (subjectsource, datasink,
            [("subject_id", "container")]),
        (skullstrip, datasink,
            [("out_file", "normalization.@brain")]),
        (cvthead, datasink,
            [("out_file", "normalization.@t1")]),
        (flirt, datasink,
            [("out_file", "normalization.@brain_flirted")]),
        (flirt, datasink,
            [("out_matrix_file", "normalization.@affine")]),
        (warpbrain, datasink,
            [("out_file", "normalization.@brain_warped")]),
        (warpbrainhr, datasink,
            [("out_file", "normalization.@brain_hires")]),
        (fnirt, datasink,
            [("fieldcoeff_file", "normalization.@warpfield")]),
        (checkreg, datasink,
            [("out_file", "normalization.@reg_png")]),
    ])

    return normalize


def warp_report(in_file):
    """Plot the registration summary using nipy contours"""
    mni_file = op.join(os.environ["FSL_DIR"],
                       "data/standard/MNI152_T1_1mm_brain.nii.gz")
    mni_img = nib.load(mni_file)
    mni_data, mni_aff = mni_img.get_data(), mni_img.get_affine()
    sub_img = nib.load(in_file)
    sub_data, sub_aff = sub_img.get_data(), sub_img.get_affine()
    sub_data[sub_data < 1] = 0

    kwargs = dict(draw_cross=False, annotate=False)
    cut_coords = dict(x=(-45, -12, 12, 45),
                      y=(-55, -25, 5, 45),
                      z=(-30, -5, 20, 40))

    colors = sns.color_palette("bright")
    im_data = dict()
    for axis in ["x", "y", "z"]:
        f = plt.figure(figsize=(10, 2.5))
        coords = cut_coords[axis]
        slicer = viz.plot_anat(sub_data, sub_aff, slicer=axis,
                               cut_coords=coords, figure=f, **kwargs)
        slicer.contour_map(mni_data, mni_aff, colors=colors)
        fname = "slices_%s.png" % axis
        f.savefig(fname, facecolor="k", edgecolor="k")
        im_data[axis] = mplimg.imread(fname)

    concat_data = [im_data[axis] for axis in ["x", "y", "z"]]
    concat_data = np.concatenate(concat_data, axis=0)
    mplimg.imsave("warp_report.png", concat_data)
    return op.abspath("warp_report.png")


def mni_reg_qc(in_file):
    """Plot the registration using external FSL routines."""
    import os.path as op
    from subprocess import call
    from nipype.interfaces import fsl

    mni_targ = fsl.Info.standard_image("MNI152_T1_1mm_brain.nii.gz")

    planes = ["x", "y", "z"]
    options = []
    for plane in planes:
        for slice in ["%.2f" % i for i in .15, .3, .45, .5, .55, .7, .85]:
            if not(plane == "x" and slice == "0.50"):
                options.append((plane, slice))

    shots = ["%s-%s.png" % i for i in options]

    for i, shot in enumerate(shots):
        cmd = ["slicer",
               in_file,
               mni_targ,
               "-s .8",
               "-%s" % options[i][0],
               options[i][1],
               shot]

        call(cmd)

    for i in range(3):
        cmd = ["pngappend"]
        cmd.append(" + ".join(
            [s for s in shots if op.split(s)[1].startswith(planes[i])]))
        rowimg = "row-%d.png" % i
        cmd.append(rowimg)
        shots.append(rowimg)
        call(" ".join(cmd), shell=True)

    cmd = ["pngappend"]
    cmd.append(" - ".join(["row-%d.png" % i for i in range(3)]))
    out_file = op.join(op.abspath("."), "brain_warp_to_mni.png")
    cmd.append(out_file)
    call(" ".join(cmd), shell=True)

    return out_file
