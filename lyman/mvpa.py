from __future__ import division
import os
import os.path as op
from glob import glob
import hashlib
import re

import numpy as np
import scipy as sp
from scipy import stats
from scipy.interpolate import interp1d
import pandas as pd
import nibabel as nib
from sklearn.cross_validation import (KFold,
                                      LeaveOneOut,
                                      LeaveOneLabelOut)

import moss
from lyman import gather_project_info, gather_experiment_info


def extract_dataset(sched, timeseries, mask, tr=2, frames=None,
                    upsample=None, event_names=None):
    """Extract model and targets for single run of fMRI data.

    Parameters
    ----------
    sched : event sequence DataFrame
        must contain `condition` and `onsets` columns
    timeseries : 4D numpy array
        BOLD data
    mask : 3D boolean array
        ROI mask
    tr : int
        acquistion TR (in seconds)
    frames : sequence of ints, optional
        extract frames relative to event onsets or at onsets if None
    upsample : int
        upsample the raw timeseries by this factor using cubic spline
        interpolation
    event_names : list of strings
        list of condition names to use, otherwise uses sorted unique
        values in sched.condition

    Returns
    -------
    X : (n_frame) x n_samp x n_feat array
        model matrix (zscored by feature)
        if n_frame is 1, matrix is 2D
        NOTE: this array can have NaNs; use mask to remove
    y : n_ev vector
        target vector
    mask : n_feat boolean vector
        False where the variance for that feature is 0

    """
    # Set up the extraction frames
    if frames is None:
        frames = [0]
    elif not hasattr(frames, "__len__"):
        frames = [frames]
    frames = np.asarray(frames)

    if upsample is not None:
        n_frames = len(frames) * upsample
        frames = np.linspace(frames.min() * upsample,
                             (frames.max() + 1) * upsample,
                             n_frames + 1)[:-1]

    # Double check mask datatype
    if not mask.dtype == np.bool:
        raise ValueError("Mask must be boolean array")

    # Initialize the outputs
    if event_names is None:
        event_names = sorted(sched.condition.unique())
    else:
        sched = sched[sched.condition.isin(event_names)]
        event_names = list(event_names)
    X = np.zeros((len(frames), sched.shape[0], mask.sum()))
    y = sched.condition.map(lambda x: event_names.index(x))

    # Extract the ROI into a 2D n_tr x n_feat
    roi_data = timeseries[mask].T

    # Possibly upsample the raw data
    if upsample is None:
        upsample = 1
    else:
        time_points = len(roi_data)
        x = np.linspace(0, time_points - 1, time_points)
        xx = np.linspace(0, time_points,
                         time_points * upsample + 1)[:-upsample]
        interpolator = interp1d(x, roi_data, "cubic", axis=0)
        roi_data = interpolator(xx)

    # Build the data array
    for i, frame in enumerate(frames):
        onsets = np.array(sched.onset / tr).astype(int) * upsample
        onsets += int(frame)
        X[i, ...] = sp.stats.zscore(roi_data[onsets])

    # Find a mask to only use features with nonzero variance
    good_features = np.all([(np.var(X_i, axis=0) > 0) for X_i in X], axis=0)

    return X.squeeze(), y, good_features


def extract_subject(subj, problem, roi_name, mask_name=None, frames=None,
                    collapse=None, confounds=None, upsample=None,
                    smoothed=False, exp_name=None, event_names=None):
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
        problem name corresponding to design file name
    roi_name : string
        ROI name associated with data
    mask_name : string, optional
        name of ROI mask that can be found in data hierachy,
        uses roi_name if absent
    frames : int or sequence of ints, optional
        extract frames relative to event onsets or at onsets if None
    collapse : int, slice, or (subj x frames | frames) array
        if int, returns that element in first dimension
        if slice, take mean over the slice (both relative to
        frames, not to the actual onsets)
        if array, take weighted average of each frame (possibly
        with different weights by subject) otherwise return each frame
    confounds : string or list of strings
        column name(s) in schedule datafame to be regressed out of the
        data matrix during extraction
    upsample : int
        upsample the raw timeseries by this factor using cubic spline
        interpolation
    smoothed : bool
        whether to use the spatially smoothed timeseries data
    exp_name : string, optional
        lyman experiment name where timecourse data can be found
        in analysis hierarchy (uses default if None)
    event_names : list of strings
        list of condition names to use, otherwise uses sorted unique
        values in the condition field of the event schedule

    Returns
    -------
    data : dictionary
        dictionary with X, y, and runs entries, along with metadata

    """
    project = gather_project_info()
    exp = gather_experiment_info(exp_name)
    if exp_name is None:
        exp_name = project["default_exp"]

    if mask_name is None:
        mask_name = roi_name

    if smoothed:
        roi_name += "_smoothed"

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
    design_file = op.join(project["data_dir"], subj, "design",
                          "%s.csv" % problem)
    smoothing = "smoothed" if smoothed else "unsmoothed"
    ts_dir = op.join(project["analysis_dir"], exp_name, subj,
                     "reg", "epi", smoothing)
    n_runs = len(glob(op.join(ts_dir, "run_*")))
    ts_files = [op.join(ts_dir, "run_%d/timeseries_xfm.nii.gz" % r_i)
                for r_i in range(1, n_runs + 1)]

    # Get the hash value for this dataset
    ds_hash = hashlib.sha1()
    ds_hash.update(mask_name)
    ds_hash.update(str(op.getmtime(mask_file)))
    ds_hash.update(str(op.getmtime(design_file)))
    for ts_file in ts_files:
        ds_hash.update(str(op.getmtime(ts_file)))
    ds_hash.update(np.asarray(frames).data)
    ds_hash.update(str(confounds))
    ds_hash.update(str(upsample))
    ds_hash.update(str(event_names))
    ds_hash = ds_hash.hexdigest()

    # If the file exists and the hash matches, convert to a dict and return
    if op.exists(ds_file):
        with np.load(ds_file) as ds_obj:
            if ds_hash == str(ds_obj["hash"]):
                dataset = dict(ds_obj.items())
                for k, v in dataset.items():
                    if v.dtype.kind == "S":
                        dataset[k] = str(v)
                # Possibly perform temporal compression
                _temporal_compression(collapse, dataset)
                return dataset

    # Otherwise, initialize outputs
    X, y, runs, use = [], [], [], []

    # Load mask file
    mask_data = nib.load(mask_file).get_data().astype(bool)

    # Load the event information
    sched = pd.read_csv(design_file)

    # Get a list of event names
    if event_names is None:
        event_names = sorted(sched.condition.unique())
    else:
        sched = sched[sched.condition.isin(event_names)]

    # Make each runs' dataset
    for r_i, sched_r in sched.groupby("run"):
        ts_data = nib.load(ts_files[int(r_i - 1)]).get_data()

        # Use the basic extractor function
        X_i, y_i, use_i = extract_dataset(sched_r, ts_data,
                                          mask_data, exp["TR"],
                                          frames, upsample, event_names)

        # Just add to list
        X.append(X_i)
        y.append(y_i)
        use.append(use_i)

    # Find the voxels that are good in every run and make a final mask
    good_features = np.all(use, axis=0)
    mask_data[mask_data] = good_features

    # Stick the list items together for final dataset
    if frames is not None and len(frames) > 1:
        X = np.concatenate(X, axis=1)
    else:
        X = np.concatenate(X, axis=0)
    y = np.concatenate(y)
    runs = sched.run

    # Apply the feature mask
    X = np.atleast_3d(X)[:, :, good_features].squeeze()

    # Regress the confound vector out from the data matrix
    if confounds is not None:
        X = np.atleast_3d(X)
        confounds = np.asarray(sched[confounds])
        confounds = stats.zscore(confounds.reshape(X.shape[1], -1))
        denom = confounds / np.dot(confounds.T, confounds)
        for X_i in X:
            X_i -= np.dot(X_i.T, confounds).T * denom
        X = X.squeeze()

    # Save to disk and return
    dataset = dict(X=X, y=y, runs=runs, roi_name=roi_name, subj=subj,
                   event_names=event_names, problem=problem, frames=frames,
                   confounds=confounds, upsample=upsample, smoothed=smoothed,
                   hash=ds_hash, mask=mask_data, mask_name=mask_name)
    np.savez(ds_file, **dataset)

    # Possibly perform temporal compression
    _temporal_compression(collapse, dataset)

    return dataset


def _temporal_compression(collapse, dset):
    """Either select a single frame or take the mean over several frames."""
    if collapse is not None:
        if isinstance(collapse, int):
            dset["X"] = dset["X"][collapse]
        elif isinstance(collapse, slice):
            dset["X"] = dset["X"][collapse].mean(axis=0)
        else:
            dset["X"] = np.average(dset["X"], axis=0, weights=collapse)

    dset["collapse"] = collapse


def extract_group(problem, roi_name, mask_name=None, frames=None,
                  collapse=None, confounds=None, upsample=None, smoothed=False,
                  exp_name=None, event_names=None, subjects=None, dv=None):
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
    collapse : int, slice, or (subj x frames | frames) array
        if int, returns that element in first dimension
        if slice, take mean over the slice (both relative to
        frames, not to the actual onsets)
        if array, take weighted average of each frame (possibly
        with different weights by subject) otherwise return each frame
    confounds : sequence of arrays, optional
        list ofsubject-specific obs x n arrays of confounding variables
        to regress out of the data matrix during extraction
    upsample : int, optional
        upsample the raw timeseries by this factor using cubic splines
    smoothed : bool, optional
        whether to extract the spatially smoothed dataset
    exp_name : string, optional
        lyman experiment name where timecourse data can be found
        in analysis hierarchy (uses default if None)
    event_names : list of strings
        list of condition names to use, otherwise uses sorted unique
        values in the condition field of the event schedule
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
    subjects = list(subjects)

    # Allow to run in serial or parallel
    if dv is None:
        import __builtin__
        map = __builtin__.map
    else:
        map = dv.map_sync

    # Try to make frames a list, if possible
    if hasattr(frames, "tolist"):
        frames = frames.tolist()
    if hasattr(collapse, "tolist"):
        collapse = collapse.tolist()

    # Set up lists for the map to work
    problem = [problem for _ in subjects]
    roi_name = [roi_name for _ in subjects]
    mask_name = [mask_name for _ in subjects]
    frames = [frames for _ in subjects]
    if np.ndim(collapse) < 2:
        collapse = [collapse for _ in subjects]
    confounds = [confounds for _ in subjects]
    upsample = [upsample for _ in subjects]
    smoothed = [smoothed for _ in subjects]
    exp_name = [exp_name for _ in subjects]
    event_names = [event_names for _ in subjects]

    # Actually do the loading
    data = map(extract_subject, subjects, problem, roi_name, mask_name,
               frames, collapse, confounds, upsample, smoothed, exp_name,
               event_names)

    return data


def _results_fname(dataset, model, split_pred, trialwise, logits, shuffle,
                   exp_name):
    """Get a path to where files storing decoding results will live."""
    project = gather_project_info()
    if exp_name is None:
        exp_name = project["default_exp"]

    roi_name = dataset["roi_name"]
    collapse = dataset["collapse"]
    problem = dataset["problem"]
    subj = dataset["subj"]

    res_path = op.join(project["analysis_dir"],
                       exp_name, subj, "mvpa",
                       problem, roi_name)

    try:
        model_str = "_".join([i[0] for i in model.steps])
    except AttributeError:
        model_str = model.__class__.__name__

    collapse_str, split_str, trial_str, logit_str, shuffle_str = ("", "", "",
                                                                  "", "")
    if collapse is not None:
        if isinstance(collapse, slice):
            collapse_str = "%s-%s" % (collapse.start, collapse.stop)
        elif isinstance(collapse, int):
            collapse_str = str(collapse)
        else:
            collapse_str = "weighted"
    if split_pred is not None:
        split_str = "split"
        if hasattr(split_pred, "name"):
            if split_pred.name is None:
                split_str = split_pred.name
    if trialwise:
        trial_str = "trialwise"
    if logits:
        logit_str = "logits"
    if shuffle:
        shuffle_str = "shuffle"

    res_fname = "_".join([model_str, collapse_str, split_str,
                          trial_str, logit_str, shuffle_str])
    res_fname = re.sub("_{2,}", "_", res_fname)
    res_fname = res_fname.strip("_") + ".npz"
    res_fname = op.join(res_path, res_fname)

    return res_fname


def _hash_decoder(ds, model, split_pred=None, n_iter=None, random_seed=None):
    """Hash the inputs to a decoding analysis."""
    ds_hash = hashlib.sha1()
    try:
        ds_hash.update(ds["X"].data)
    except AttributeError:
        ds_hash.update(ds["X"].copy().data)
    ds_hash.update(ds["y"].data)
    ds_hash.update(ds["runs"])
    ds_hash.update(str(model))
    if split_pred is not None:
        ds_hash.update(np.array(split_pred).data)
    if n_iter is not None:
        ds_hash.update(str(n_iter))
    if random_seed is not None:
        ds_hash.update(str(random_seed))
    return ds_hash.hexdigest()


def _decode_subject(dataset, model, cv="run", split_pred=None,
                    trialwise=False, logits=False):
    """Internal function for classification."""
    if split_pred is not None and trialwise:
        raise ValueError("Cannot use both `split_pred` and `trialwise`.")

    # Unpack the dataset
    X = dataset["X"]
    y = dataset["y"]
    runs = dataset["runs"]
    if X.ndim < 3:
        X = [X]

    # Set up the cross-validation
    if cv == "run":
        cv_ = LeaveOneLabelOut(runs)
    elif cv == "sample":
        cv_ = LeaveOneOut(len(y))
    elif isinstance(cv, int):
        cv_ = KFold(len(y), cv)
    else:
        raise ValueError("CV argument was not understood")

    # Cross-validate the model over frames
    scores = np.empty((len(X), len(y)))
    for i, X_i in enumerate(X):
        for train, test in cv_:
            if logits:
                ps = model.fit(X_i[train], y[train]).predict_proba(X_i[test])
                rows = np.arange(len(ps))
                ps = ps[rows, y[test]]
                ps = np.log(ps) - np.log(1 - ps)
            else:
                ps = model.fit(X_i[train], y[train]).predict(X_i[test])
                ps = ps == y[test]
            scores[i, test] = ps

    # Possibly bin by trial splits
    if split_pred is not None:
        n_bins = len(np.unique(split_pred))
        split_scores = np.empty((len(X), n_bins))
        for i, bin in enumerate(np.unique(split_pred)):
            split_scores[:, i] = scores[:, split_pred == bin].mean()
        scores = split_scores
    elif not trialwise:
        scores = scores.mean(axis=1)

    return np.atleast_1d(scores.squeeze())


def decode_subject(dataset, model, cv="run", split_pred=None,
                   trialwise=False, logits=False, exp_name=None):
    """Perform decoding on a single dataset.

    This function hashes the relevant inputs and uses that to store
    persistant data over multiple executions.

    Parameters
    ----------
    dataset : dict
        decoding dataset
    model : scikit-learn estimator
        model to decode with
    cv : "run" | "sample" | k
        cross validate over runs, over samples (leave-one-out), or
        over `k` folds.
    spit_pred : pandas series or array
        bin prediction accuracies by the index values in the array.
        note: cannot be used with `trialwise`.
    trialwise : bool, optional
        if False, return accuracy/logit on each trial; otherwise take
        mean, possibly over frame. note: cannot be used with `split_pred`.
    logits : bool, optional
        return continuous target logit value for each observation
    exp_name : string
        name of experiment, if not default

    Return
    ------
    scores : array
        squeezed array of scores with shape (n_frames, (n_splits | n_trials))

    """
    # Ensure some inputs
    if split_pred is not None:
        split_pred = np.asarray(split_pred)

    # Get a path to the results will live
    res_file = _results_fname(dataset, model, split_pred, trialwise,
                              logits, False, exp_name)

    # Hash the inputs to the decoder
    decoder_hash = _hash_decoder(dataset, model, split_pred)

    # If the file exists and the hash matches, load and return
    if op.exists(res_file):
        with np.load(res_file) as res_obj:
            if decoder_hash == str(res_obj["hash"]):
                return res_obj["scores"]

    # Do the decoding with a private function so we can test it
    # without dealing with all the persistance stuff
    scores = _decode_subject(dataset, model, cv, split_pred, trialwise, logits)

    # Save the scores to disk
    res_dict = dict(scores=scores, hash=decoder_hash)
    try:
        os.makedirs(op.dirname(res_file))
    except OSError:
        pass
    np.savez(res_file, **res_dict)

    return scores


def decode_group(datasets, model, cv="run", split_pred=None,
                 trialwise=False, logits=False, exp_name=None, dv=None):
    """Perform decoding on a sequence of datasets.

    Parameters
    ----------
    datasets : sequence of dicts
        one dataset per subject
    model : scikit-learn estimator
        model to decode with
    cv : "run" | "sample" | k
        cross validate over runs, over samples (leave-one-out), or
        over `k` folds.
    spit_pred : series/array or sequence of seires/arrays, optional
        bin prediction accuracies by the index values in the array.
        can pass one array to use for all datasets, or a list
        of arrays with the same length as the dataset list.
        splits will form last axis of returned accuracy array.
        note: cannot be used with `trialwise`.
    trialwise : bool, optional
        if False, return accuracy/logit on each trial; otherwise take
        mean, possibly over frame. note: cannot be used with `split_pred`.
    logits : bool, optional
        return continuous target logit value for each observation
    exp_name : string
        name of experiment, if not default
    dv : IPython cluster direct view, optional
        IPython cluster to decode in parallel

    Return
    ------
    all_scores : array
        array with possible dimensions in (subj, frame, split, trial)

    """
    if dv is None:
        import __builtin__
        map = __builtin__.map
    else:
        map = dv.map_sync

    # Set up the lists for the map
    model = [model for d in datasets]
    cv = [cv for _ in datasets]
    if split_pred is None or not np.iterable(split_pred[0]):
        split_pred = [split_pred for _ in datasets]
    trialwise = [trialwise for _ in datasets]
    logits = [logits for _ in datasets]
    exp_name = [exp_name for _ in datasets]

    # Do the decoding
    all_scores = map(decode_subject, datasets, model, cv, split_pred,
                     trialwise, logits, exp_name)

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
        res_file = _results_fname(data, model, None, False,
                                  False, True, exp_name)

        # Hash the inputs to the decoder
        decoder_hash = _hash_decoder(data, model, n_iter=n_iter,
                                     random_seed=random_seed)

        # If the file exists and the hash matches, load and return
        if op.exists(res_file):
            with np.load(res_file) as res_obj:
                if decoder_hash == str(res_obj["hash"]):
                    group_scores.append(res_obj["scores"])
                    continue

        # Otherwise, do the test for this dataset
        p_vals, scores = moss.randomize_classifier(data, model, n_iter,
                                                   cv_method, random_seed,
                                                   return_dist=True, dv=dv)

        # Save the scores to disk
        res_dict = dict(scores=scores, hash=decoder_hash)
        np.savez(res_file, **res_dict)

        group_scores.append(scores)

    return np.array(group_scores)


def model_coefs(datasets, model, flat=True, exp_name=None):
    """Fit a model on all data and save the learned model weights.

    This does not work for datasets with > 1 frames.

    Parameters
    ----------
    datasets : list of dicts
        group mvpa datasets
    model : scikit-learn estimator
        decoding model
    flat : bool
        if False return in original data space (with voxels outside
        mask represented as NaN. otherwise return straight from model
    exp_name : string or None
        experiment name, otherwise uses project default

    Returns
    -------
    out_coefs : list of arrays
        model coefficients; form is determined by `flat` parameter

    """
    project = gather_project_info()
    if exp_name is None:
        exp_name = project["default_exp"]

    out_coefs = []

    # Iterate through the datasets
    for dset in datasets:
        subj = dset["subj"]

        # Get a header and affine matrix for the EPI space
        smoothed = "smoothed" if dset["smoothed"] else "unsmoothed"
        epi_file = op.join(project["analysis_dir"], exp_name, subj,
                           "preproc/run_1/example_func.nii.gz")
        epi_img = nib.load(epi_file)
        epi_header, epi_affine = epi_img.get_header(), epi_img.get_affine()

        # Check if we need to do anything
        decoder_hash = _hash_decoder(dset, model, None)
        coef_file = _results_fname(dset, model, None, False,
                                   False, False, exp_name)
        coef_file = coef_file.strip(".npz") + "_coef.npz"
        coef_nifti = coef_file.strip(".npz") + ".nii.gz"
        if op.exists(coef_file) and op.exists(coef_nifti):
            with np.load(coef_file) as res_obj:
                if decoder_hash == str(res_obj["hash"]):
                    if flat:
                        data = res_obj["data"]
                    else:
                        data = nib.load(coef_nifti).get_data()
                    out_coefs.append(data)
                    continue

        # Determine the mask
        mask = dset["mask"]

        # Get the mask dimensions
        x, y, z = mask.shape

        # Fit the model and extract the learned model weights
        model = model.fit(dset["X"], dset["y"])
        if hasattr(model, "estimators_"):
            coef = np.array([e.coef_.ravel() for e in model.estimators_])
        else:
            coef = model.coef_
        coef_data = np.zeros((x, y, z, len(coef))) * np.nan
        coef_data[mask] = coef.T

        # Save the data both as a npz and nifti
        coef_dict = dict(data=coef, hash=decoder_hash)
        np.savez(coef_file, **coef_dict)
        coef_nifti = coef_file.strip(".npz") + ".nii.gz"
        coef_img = nib.Nifti1Image(coef_data, epi_affine, epi_header)
        nib.save(coef_img, coef_nifti)

    return out_coefs
