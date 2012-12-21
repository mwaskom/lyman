from __future__ import division
import os
import os.path as op
from glob import glob
import hashlib
import re

import numpy as np
import scipy as sp
import nibabel as nib
import nipy.modalities.fmri.hemodynamic_models as hrf

from sklearn.cross_validation import (cross_val_score,
                                      LeaveOneOut, LeaveOneLabelOut)

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
    elif not hasattr(frames, "__len__"):
        frames = [frames]

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


def fmri_dataset(subj, problem, roi_name, mask_name=None,
                 exp_name=None, frames=None):
    """Build decoding dataset from predictable lyman outputs.

    This function will make use of the LYMAN_DIR environment variable
    to access information about where the relevant data live, so that
    must be set properly.

    This function caches its results and, on repeated calls,
    hashes the arguments and checks those against the hash value
    associated with the stored data. The hashing process considers
    the timestamp on the relevant data files, but not the data itself.

    Parameters
    ----------
    subj : string
        subject id
    problem : string
        problem name corresponding to set of event types
    roi_name : string
        ROI name associated with data
    mask_name : string, optional
        name of ROI mask that can be found in data hierachy,
        uses roi_name if absent
    exp_name : string, optional
        lyman experiment name where timecourse data can be found
        in analysis hierarchy
    frames : int or sequence of ints, optional
        extract frames relative to event onsets or at onsets if None

    Returns
    -------
    data : dictionary
        dictionary with X, y, and runs entries

    """
    project = gather_project_info()
    if exp_name is None:
        exp_name = project["default_exp"]
    exp = gather_experiment_info(exp_name)

    if roi_name is None:
        roi_name = mask_name

    # Find the relevant disk location for the dataaset file
    ds_file = op.join(project["analysis_dir"],
                      exp_name, subj, "mvpa",
                      problem, roi_name, "dataset.npz")

    # Make sure the target location exists
    try:
        os.makedirs(op.dirname(ds_file))
    except OSError:
        pass

    # Get paths to the relevant files
    mask_file = op.join(project["data_dir"], subj, "masks",
                        "%s.nii.gz" % mask_name)
    problem_file = op.join(project["data_dir"], subj, "events",
                           "%s.npz" % problem)
    ts_dir = op.join(project["analysis_dir"], exp_name, subj,
                     "reg", "epi", "unsmoothed")
    n_runs = len(glob(op.join(ts_dir, "run_*")))
    ts_files = [op.join(ts_dir, "run_%d" % (r_i + 1),
                        "timeseries_xfm.nii.gz") for r_i in range(n_runs)]

    # Get the hash value for this dataset
    ds_hash = hashlib.sha1()
    ds_hash.update(mask_name)
    ds_hash.update(str(op.getmtime(mask_file)))
    ds_hash.update(str(op.getmtime(problem_file)))
    for ts_file in ts_files:
        ds_hash.update(str(op.getmtime(ts_file)))
    ds_hash.update(str(frames))
    ds_hash = ds_hash.hexdigest()

    # If the file exists and the hash matches, convert to a dict and return
    if op.exists(ds_file):
        ds_obj = np.load(ds_file)
        if ds_hash == str(ds_obj["hash"]):
            dataset = dict(ds_obj.items())
            for k, v in dataset.items():
                if v.dtype.kind == "S":
                    dataset[k] = str(v)
            return dataset

    # Othersies, initialize outputs
    X, y, runs = [], [], []

    # Load mask file
    mask_data = nib.load(mask_file).get_data().astype(bool)

    # Load the event information
    problem_data = np.load(problem_file)
    event_names = problem_data["event_names"]

    # Make each runs' dataset
    for r_i in range(n_runs):
        ts_data = nib.load(ts_files[r_i]).get_data()

        evs = [problem_data[ev][r_i] for ev in event_names]

        # Use the basic extractor function
        X_i, y_i = extract_dataset(evs, ts_data, mask_data,
                                   exp["TR"], frames)

        # Just add to list
        X.append(X_i)
        y.append(y_i)
        runs.append(np.ones_like(y_i) * r_i)

    # Stick the list items together for final dataset
    if frames is not None and len(frames) > 1:
        X = np.concatenate(X, axis=1)
    else:
        X = np.concatenate(X, axis=0)
    y = np.concatenate(y)
    runs = np.concatenate(runs)

    # Save to disk and return
    dataset = dict(X=X, y=y, runs=runs, roi_name=roi_name, subj=subj,
                   event_names=event_names, problem=problem, frames=frames,
                   hash=ds_hash)
    np.savez(ds_file, **dataset)
    return dataset


def load_datasets(problem, roi_name, mask_name=None, frames=None,
                  collapse=None, exp_name=None, subjects=None, dv=None):
    """Load datasets for a group of subjects, possibly in parallel.

    Parameters
    ----------
    problem : string
        problem name corresponding to set of event types
    roi_name : string
        ROI name associated with data
    mask_name : string, optional
        name of ROI mask that can be found in data hierachy,
        uses roi_name if absent
    frames : int or sequence
        frames relative to stimulus onsets in event file to extract
    collapse : int or slice
        if int, returns that element in first dimension
        if slice, take mean over the slice (both relative to
        frames, not to the actual onsets) otherwise return each frame
    exp_name : string, optional
        lyman experiment name where timecourse data can be found
        in analysis hierarchy
    subjects : sequence of strings, optional
        sequence of subjects to return; if none reads subjects.txt file
        from lyman directory and uses all defined there
    dv : IPython cluster direct view, optional
        if provided, executes in parallel using the cluster

    Returns
    -------
    data : list of dicts
       list of mvpa dictionaries

    """
    if subjects is None:
        subj_file = op.join(os.environ["LYMAN_DIR"], "subjects.txt")
        subjects = np.loadtxt(subj_file, str)

    # Allow to run in serial or parallel
    if dv is None:
        import __builtin__
        map = __builtin__.map
    else:
        map = dv.map_sync

    # Set up lists for the map to work
    problem = [problem for s in subjects]
    roi_name = [roi_name for s in subjects]
    mask_name = [mask_name for s in subjects]
    frames = [frames for s in subjects]
    exp_name = [exp_name for s in subjects]

    # Actually do the loading
    data = map(fmri_dataset, subjects, problem, roi_name, mask_name,
               exp_name, frames)

    # Potentially collapse across some stimulus frames
    for dset in data:
        if collapse is not None:
            if isinstance(collapse, int):
                dset["X"] = dset["X"][collapse]
            else:
                dset["X"] = dset["X"][collapse].mean(axis=0)
        dset["collapse"] = collapse

    return data


def _results_fname(dataset, model, split_pred, split_name, exp_name, shuffle):
    """Get a path to where files storing decoding results will live."""
    project = gather_project_info()
    if exp_name is None:
        exp_name = project["default_exp"]

    roi_name = dataset["roi_name"]
    problem = dataset["problem"]
    subj = dataset["subj"]

    res_path = op.join(project["analysis_dir"],
                       exp_name, subj, "mvpa",
                       problem, roi_name)

    model_str = re.match("(.+)\(", str(model)).group(1)
    collapse = dataset["collapse"]

    collapse_str, split_str, shuffle_str = "", "", ""
    if collapse is not None:
        if isinstance(collapse, slice):
            collapse_str = "%s-%s" % (collapse.start, collapse.stop)
        else:
            collapse_str = str(collapse)
    if split_pred is not None and split_name is None:
        split_str = "split"
    if shuffle:
        shuffle_str = "shuffle"

    res_fname = "_".join([model_str, collapse_str, split_str, shuffle_str])
    res_fname = re.sub("_{2,}", "_", res_fname)
    res_fname = res_fname.strip("_") + ".npz"
    res_fname = op.join(res_path, res_fname)

    return res_fname


def _hash_decoder(ds, model, random_seed=None):
    """Hash the inputs to a decoding analysis."""
    ds_hash = hashlib.sha1()
    ds_hash.update(ds["X"].data)
    ds_hash.update(ds["y"].data)
    ds_hash.update(ds["runs"])
    ds_hash.update(str(model))
    if random_seed is not None:
        ds_hash.update(str(random_seed))
    return ds_hash.hexdigest()


def _decode_subject(dataset, model, split_pred=None, split_name=None,
                    cv_method="run", n_jobs=1):
    """Internal decoding function to allow for simpler testing."""
    # TODO maybe move this to moss?

    # Get direct references to the data
    X = dataset["X"]
    y = dataset["y"]
    runs = dataset["runs"]

    # Set up the cross-validation
    indices = True if split_pred is None else False
    if cv_method == "run":
        cv = LeaveOneLabelOut(runs, indices=indices)
    elif cv_method == "sample":
        cv = LeaveOneOut(len(y), indices=indices)
    else:
        cv = cv_method

    if X.ndim < 3:
        X = [X]

    # Do the decoding
    scores = []
    for X_i in X:
        if split_pred is None:
            score = cross_val_score(model, X_i, y, cv=cv,
                                    n_jobs=n_jobs).mean()
            scores.append(score)
        else:
            n_bins = len(np.unique(split_pred))
            bin_scores = [[] for i in range(n_bins)]
            for train, test in cv:
                model.fit(X_i[train], y[train])
                for bin in range(n_bins):
                    idx = np.logical_and(test, split_pred == bin)
                    bin_score = model.score(X_i[idx], y[idx])
                    bin_scores[bin].append(bin_score)
            scores.append(np.mean(bin_scores, axis=1))
    scores = np.atleast_1d(np.squeeze(scores))
    return scores


def decode_subject(dataset, model, split_pred=None, split_name=None,
                   cv_method="run", exp_name=None, n_jobs=1):
    """Perform decoding on a single dataset.

    This function hashes the relevant inputs and uses that to store
    persistant data over multiple executions.

    Parameters
    ----------
    dataset : dict
        decoding dataset
    model : scikit-learn estimator
        model to decode with
    spit_pred : array or sequence of arrays, optional
        bin prediction accuracies by the index values in the array.
        can pass one array to use for all datasets, or a list
        of arrays with the same length as the dataset list.
        splits will form last axis of returned accuracy array.
        n_jobs will have no effect when this is used, but can
        still run in parallel over subjects using IPython.
    split_name : string
        name to associate with split results file
    cv_method : run | sample | cv arg for cross_val_score
        cross validate over runs, over samples (leave-one-out)
        or otherwise something that can be provided to the cv
        argument for sklearn.cross_val_score
    exp_name : string
        name of experiment, if not default
    n_jobs : int, optional
        number of jobs for sklean internal parallelization
    dv : IPython cluster direct view, optional
        IPython cluster to decode in parallel

    Return
    ------
    scores : array
        squeezed array of scores with (n_split, n_tp) dimensions

    """
    # Get a path to the results will live
    res_file = _results_fname(dataset, model, split_pred, split_name,
                              exp_name, False)

    # Hash the inputs to the decoder
    decoder_hash = _hash_decoder(dataset, model)

    # If the file exists and the hash matches, load and return
    if op.exists(res_file):
        res_obj = np.load(res_file)
        if decoder_hash == str(res_obj["hash"]):
            return res_obj["scores"]

    # Do the decoding with a private function so we can test it
    # without dealing with all the persistance stuff
    scores = _decode_subject(dataset, model, split_pred, split_name,
                             cv_method, n_jobs)

    # Save the scores to disk
    res_dict = dict(scores=scores, hash=decoder_hash)
    np.savez(res_file, **res_dict)

    return scores


def decode_group(datasets, model, split_pred=None, split_name=None,
                 cv_method="run", exp_name=None, n_jobs=1, dv=None):
    """Perform decoding on a sequence of datasets.

    Parameters
    ----------
    datasets : sequence of dicts
        one dataset per subject
    model : scikit-learn estimator
        model to decode with
    spit_pred : array or sequence of arrays, optional
        bin prediction accuracies by the index values in the array.
        can pass one array to use for all datasets, or a list
        of arrays with the same length as the dataset list.
        splits will form last axis of returned accuracy array.
        n_jobs will have no effect when this is used, but can
        still run in parallel over subjects using IPython.
    split_name : string
        name to associate with split results file
    cv_method : run | sample | cv arg for cross_val_score
        cross validate over runs, over samples (leave-one-out)
        or otherwise something that can be provided to the cv
        argument for sklearn.cross_val_score
    exp_name : string
        name of experiment, if not default
    n_jobs : int, optional
        number of jobs for sklean internal parallelization
    dv : IPython cluster direct view, optional
        IPython cluster to decode in parallel

    Return
    ------
    all_scores : array
        array where first dimension is subjects and second
        dimension may be timepoints

    """
    if dv is None:
        import __builtin__
        map = __builtin__.map
    else:
        map = dv.map_sync

    # Set up the lists for the map
    model = [model for d in datasets]

    try:
        if len(np.array(cv_method)) != len(datasets):
            cv_method = [cv_method for d in datasets]
    except TypeError:
        cv_method = [cv_method for d in datasets]

    try:
        len(split_pred[0])
    except TypeError:
        split_pred = [split_pred for d in datasets]
    split_name = [split_name for d in datasets]

    exp_name = [exp_name for d in datasets]
    n_jobs = [n_jobs for d in datasets]

    # Do the decoding
    all_scores = map(decode_subject, datasets, model,
                     split_pred, split_name, cv_method, exp_name, n_jobs)
    return np.array(all_scores)


def classifier_permutations(datasets, model, n_iter=1000, cv_method="run",
                            random_seed=None, exp_name=None, dv=None):
    """Do a randomization test on a set of classifiers with cached results.

    The randomizations can be distributed over an IPython cluster using
    the ``dv`` argument. Note that unlike the decode_group function,
    the parallelization occurs within, rather than over subjects.

    Parameters
    ----------
    datasets : list of dictionaries
        each item in the list is an mvpa dictionary
    n_iter : int
        number of permutation iterations
    cv_method : run | sample | cv arg for cross_val_score
        cross validate over runs, over samples (leave-one-out)
        or otherwise something that can be provided to the cv
        argument for sklearn.cross_val_score
    random_state : int
        seed for random state to obtain stable permutations
    exp_name : string
        experiment name if not default
    dv : IPython direct view
        view onto IPython cluster for parallel execution over iterations

    Returns
    -------
    group_scores : list of arrays
        permutation array for each item in datasets

    """
    group_scores = []
    for i_s, data in enumerate(datasets):

        # Get a path to the results will live
        res_file = _results_fname(data, model, None, None, exp_name, True)

        # Hash the inputs to the decoder
        decoder_hash = _hash_decoder(data, model, random_seed)

        # If the file exists and the hash matches, load and return
        if op.exists(res_file):
            res_obj = np.load(res_file)
            if decoder_hash == str(res_obj["hash"]):
                return res_obj["scores"]

        # Otherwise, do the test for this dataset
        p_vals, scores = moss.randomize_classifier(data, model, n_iter,
                                                   cv_method, random_seed,
                                                   return_dist=True, dv=dv)

        # Save the scores to disk
        res_dict = dict(scores=scores, hash=decoder_hash)
        np.savez(res_file, **res_dict)

        group_scores.append(scores)

    return np.array(group_scores)
