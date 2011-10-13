import nipype.pipeline.engine as pe
import nipype.interfaces.fsl as fsl
import nipype.interfaces.freesurfer as fs
import nipype.interfaces.utility as util


def get_registration_workflow(name="registration", 
                              volume=False,
                              surface=False,
                              native_surf_smooth=True,
                              norm_surf_smooth=True):

    registration = pe.Workflow(name=name)

    if not surface:
        norm_surf_smooth, native_surf_smooth = False, False

    # Define the inputs for the registation workflow
    infields = []
    if volume:
        infields.extend(["vol_source", "warpfield", "fsl_affine"])
    if surface:
        infields.extend(["surf_source", "subject_id", "tkreg_affine"])
    if (norm_surf_smooth or native_surf_smooth):
        infields.extend(["smooth_fwhm"])
    inputnode = pe.Node(util.IdentityInterface(fields=infields),
                        name="inputspec")

    if volume:

        mni152 = fsl.Info.standard_image("avg152T1_brain.nii.gz")
        applywarp = pe.MapNode(fsl.ApplyWarp(ref_file=mni152,
                                             interp="spline",
                                             ),
                                iterfield=["in_file", "premat"],
                             name="applywarp")
        
        registration.connect([
            (inputnode, applywarp, [("vol_source", "in_file"),
                                    ("warpfield", "field_file"),
                                    ("fsl_affine", "premat")]),
                ])

    if surface:
        
        hemisource = pe.Node(util.IdentityInterface(fields=["hemi"]),
                             iterables=("hemi",["lh","rh"]),
                             name="hemisource")

        surfproject = pe.MapNode(fs.SampleToSurface(sampling_range=(0,1,.1),
                                                    sampling_units="frac",
                                                    cortex_mask=True),
                                 iterfield=["source_file", "reg_file"],
                                 name="surfproject")
        surfproject.inputs.sampling_method="average"

        surftransform = pe.MapNode(fs.SurfaceTransform(target_subject="fsaverage",
                                                       reshape=True),
                                   iterfield=["source_file"],
                                   name="surftransform")

        cvtnormsurf = pe.MapNode(fs.MRIConvert(out_type="niigz"),
                                 iterfield=["in_file"],
                                 name="convertnormsurf")
        
        registration.connect([
            (inputnode,    surfproject,    [("surf_source", "source_file"),
                                            ("subject_id", "subject_id"),
                                            ("tkreg_affine", "reg_file")]),
            (hemisource,   surfproject,    [("hemi", "hemi")]),
            (surfproject,  surftransform,  [("out_file", "source_file")]),
            (inputnode,    surftransform,  [("subject_id", "source_subject")]),
            (hemisource,   surftransform,  [("hemi", "hemi")]),
            ])

    if norm_surf_smooth:

        smoothnormsurf = pe.MapNode(fs.SurfaceSmooth(subject_id="fsaverage",
                                                     reshape=True),
                                    iterfield=["in_file"],
                                    name="smoothnormsurf")
        registration.connect([
            (surftransform,smoothnormsurf, [("out_file", "in_file")]),
            (hemisource,   smoothnormsurf, [("hemi", "hemi")]),
            (inputnode,    smoothnormsurf, [("smooth_fwhm", "fwhm")]),
            (smoothnormsurf, cvtnormsurf,  [("out_file", "in_file")]),
            ])

    elif surface:
        registration.connect(surftransform, "out_file", cvtnormsurf, "in_file")

    if native_surf_smooth:

        smoothnatsurf = pe.MapNode(fs.SurfaceSmooth(),
                                   iterfield=["in_file"],
                                   name="smoothnativesurf")

        registration.connect([
            (surfproject,  smoothnatsurf,  [("out_file", "in_file")]),
            (hemisource,   smoothnatsurf,  [("hemi", "hemi")]),
            (inputnode,    smoothnatsurf,  [("subject_id", "subject_id"),
                                            ("smooth_fwhm", "fwhm")]),
            ])

    outfields = []
    if volume:
        outfields.append("warped_image")
    if surface:
        outfields.extend(["surface_image", "surface_image_fsaverage"])
   
    outputnode = pe.Node(util.IdentityInterface(fields=outfields),
                         name="outputspec")

    if volume:
        registration.connect([
            (applywarp, outputnode, [("out_file", "warped_image")]),
            ])
    if surface:
        if native_surf_smooth:
            registration.connect([
                (smoothnatsurf, outputnode, [("out_file", "surface_image")]),
                ])
        else:
            registration.connect([
                (surfproject, outputnode, [("out_file", "surface_image")]),
                ])
        registration.connect([
            (cvtnormsurf,   outputnode, [("out_file", "surface_image_fsaverage")]),
            ])
    
    return registration, inputnode, outputnode
