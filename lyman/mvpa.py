from __future__ import division
import os
import os.path as op
from glob import glob
import numpy as np
import scipy as sp
import nibabel as nib
import nipy.modalities.fmri.hemodynamic_models as hrf
import moss
from lyman import gather_project_info, gather_experiment_info


def iterated_deconvolution(data, evs, tr=2, hpf_cutoff=128, filter_data=True,
                           copy_data=False, split_confounds=True,
                           hrf_model="canonical", fir_bins=12):
    """Deconvolve stimulus events from an ROI data matrix.

    Parameters
    ----------
    data : ntp x n_feat
        array of fMRI data from an ROI
    evs : sequence of n x 3 arrays
        list of (onset, duration, amplitude) event specifications
    tr : int
        time resolution in seconds
    hpf_cutoff : float or None
        filter cutoff in seconds or None to skip filter
    filter_data : bool
        if False data is assumed to have been filtered
    copy_data : if False data is filtered in place
    split_confounds : boolean
        if true, confound regressors are separated by event type
    hrf_model : string
        nipy hrf_model specification name
    fir_bins : none or int
        number of bins if hrf_model is "fir"

    Returns
    -------
    coef_array : n_ev x n_feat array
        array of deconvolved parameter estimates

    """
    # Possibly filter the data
    ntp = data.shape[0]
    if hpf_cutoff is not None:
        F = moss.fsl_highpass_matrix(ntp, hpf_cutoff, tr)
        if filter_data:
            if copy_data:
                data = data.copy()
            data[:] = np.dot(F, data)
    # Demean by feature
    data -= data.mean(axis=0)

    # Devoncolve the parameter estimate for each event
    coef_list = []
    for X_i in event_designs(evs, ntp, tr, split_confounds,
                             hrf_model, fir_bins):
        # Filter each design matrix
        if hpf_cutoff is not None:
            X_i = np.dot(F, X_i)
        X_i -= X_i.mean(axis=0)
        # Fit an OLS model
        beta_i, _, _, _ = np.linalg.lstsq(X_i, data)
        # Select the relevant betas
        if hrf_model == "fir":
            coef_list.append(np.hstack(beta_i[:fir_bins]))
        else:
            coef_list.append(beta_i[0])

    return np.vstack(coef_list)


def event_designs(evs, ntp, tr=2, split_confounds=True,
                  hrf_model="canonical", fir_bins=12):
    """Generator function to return event-wise design matrices.

    Parameters
    ----------
    evs : sequence of n x 3 arrays
        list of (onset, duration, amplitude) event secifications
    ntp : int
        total number of timepoints in experiment
    tr : int
        time resolution in seconds
    split_confounds : boolean
        if true, confound regressors are separated by event type
    hrf_model : string
        nipy hrf_model specification name
    fir_bins : none or int
        number of bins if hrf_model is "fir"

    Yields
    ------
    design_mat : ntp x (2 or n event + 1) array
        yields a design matrix to deconvolve each event
        with the event of interest as the first column

    """
    n_cond = len(evs)
    master_sched = moss.make_master_schedule(evs)

    # Create a vector of frame onset times
    frametimes = np.linspace(0, ntp * tr - tr, ntp)

    # Set up FIR bins, maybe
    if hrf_model == "fir":
        fir_delays = np.linspace(0, tr * (fir_bins - 1), fir_bins)
    else:
        fir_delays = None

    # Generator loop
    for ii, row in enumerate(master_sched):
        # Unpack the schedule row
        time, dur, amp, ev_id, stim_idx = row

        # Generate the regressor for the event of interest
        ev_interest = np.atleast_2d(row[:3]).T
        design_mat, _ = hrf.compute_regressor(ev_interest,
                                              hrf_model,
                                              frametimes,
                                              fir_delays=fir_delays)

        # Build the confound regressors
        if split_confounds:
            # Possibly one for each event type
            for cond in range(n_cond):
                cond_idx = master_sched[:, 3] == cond
                conf_sched = master_sched[cond_idx]
                if cond == ev_id:
                    conf_sched = np.delete(conf_sched, stim_idx, 0)
                conf_reg, _ = hrf.compute_regressor(conf_sched[:, :3].T,
                                                    hrf_model,
                                                    frametimes,
                                                    fir_delays=fir_delays)
                design_mat = np.column_stack((design_mat, conf_reg))
        else:
            # Or a single confound regressor
            conf_sched = np.delete(master_sched, ii, 0)
            conf_reg, _ = hrf.compute_regressor(conf_sched[:, :3].T,
                                                hrf_model,
                                                frametimes,
                                                fir_delays=fir_delays)
            design_mat = np.column_stack((design_mat, conf_reg))

        yield design_mat


def extract_dataset(evs, timeseries, mask, tr=2, frames=None):
    """Extract model and targets for single run of fMRI data.

    Parameters
    ----------
    evs : event sequence
        each element in the sequence is n_ev x 3 array of
        onset, duration, amplitude
    timeseries : 4D numpy array
        BOLD data
    mask : 3D boolean array
        ROI mask
    tr : int
        acquistion TR (in seconds)
    frames : sequence of ints, optional
        extract frames relative to event onsets or at onsets if None

    Returns
    -------
    X : (n_frame) x n_samp x n_feat array
        model matrix (zscored by feature)
        if n_frame is 1, matrix is 2D
    y : n_ev vector
        target vector

    """
    sched = moss.make_master_schedule(evs)

    if frames is None:
        frames = [0]

    # Double check mask datatype
    if not mask.dtype == np.bool:
        raise ValueError("Mask must be boolean array")

    # Initialize the outputs
    X = np.zeros((len(frames), sched.shape[0], mask.sum()))
    y = sched[:, 3].astype(int)

    # Extract the ROI into a 2D n_tr x n_feat
    roi_data = timeseries[mask].T

    # Build the data array
    for i, frame in enumerate(frames):
        onsets = (sched[:, 0] / tr).astype(int)
        onsets += frame
        X[i, ...] = sp.stats.zscore(roi_data[onsets])

    return X.squeeze(), y


def fmri_dataset(subj, exp_name, mask_name, event_file,
                 event_names=None, frames=None, force_extract=False):
    """Build decoding dataset from predictable lyman outputs.

    This function will make use of the LYMAN_DIR environment variable
    to access information about where the relevant data live, so that
    must be set properly.

    If it finds an existing dataset file in the predictable location,
    it will use that file unless ``force_extract`` is True. Extraction
    will always write a dataset file, possibly overwriting old data.

    Parameters
    ----------
    subj : string
        subject id
    exp_name : string
        lyman experiment name where timecourse data can be found
        in analysis hierarchy
    mask_name : string
        name of ROI mask that can be found in data hierachy
    event_file : string
        event file name in data hierachy
    event_names : list of strings, optional
        list of event names if do not want to use all event
        specifications in event file
    frames : sequence of ints, optional
        extract frames relative to event onsets or at onsets if None
    force_extract : boolean, optional
        enforce that a dataset is created from nifti files even
        if a corresponding dataset file already exists

    Returns
    -------
    data : dictionary
        dictionary with X, y, and runs entries

    """
    project = gather_project_info()
    exp = gather_experiment_info(exp_name)

    # Find the relevant disk location
    data_file = op.join(project["ANALYSIS_DIR"],
                        exp_name, subj, "mvpa",
                        mask_name + "-" + event_file + ".npz")

    # Make sure the target location exists
    try:
        os.mkdir(op.split(data_file)[0])
    except OSError:
        pass

    if op.exists(data_file) and not force_extract:
        data_obj = np.load(data_file)
        return dict(data_obj.items())

    # Determine number of runs from glob
    ts_dir = op.join(project["analysis_dir"], exp_name, subj,
                     "reg", "epi", "unsmoothed")
    n_runs = len(glob(op.join(ts_dir, "run_*")))

    # Initialize outputs
    X, y, runs = [], [], []

    # Load mask file
    mask_file = op.join(project["data_dir"], subj, "masks",
                        "%s.nii.gz" % mask_name)
    mask_data = nib.load(mask_file).get_data().astype(bool)

    # Load the event information
    event_fpath = op.join(project["data_dir"], subj, "events",
                          "%s.npz" % event_file)
    event_data = np.load(event_fpath)
    if event_names is None:
        event_names = event_data.keys()

    # Make each runs' dataset
    for r_i in range(n_runs):
        ts_file = op.join(ts_dir, "run_%d" % (r_i + 1),
                          "timeseries_xfm.nii.gz")
        ts_data = nib.load(ts_file).get_data()

        evs = [event_data[ev][r_i] for ev in event_names]

        # Use the basic extractor function
        X_i, y_i = extract_dataset(evs, ts_data, mask_data,
                                   exp["TR"], frames)

        # Just add to list
        X.append(X_i)
        y.append(y_i)
        runs.append(np.ones_like(y_i) * r_i)

    # Stick the list items together for final dataset
    if frames is not None and len(frames > 1):
        X = np.concatenate(X, axis=1)
    else:
        X = np.concatenate(X, axis=0)
    y = np.concatenate(y)
    runs = np.concatenate(runs)

    # Save to disk and return
    data_dict = dict(X=X, y=y, runs=runs)
    np.savez(data_file, **data_dict)
    return data_dict
