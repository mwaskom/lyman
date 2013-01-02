import os
import os.path as op
from glob import glob
import hashlib
import numpy as np
import scipy as sp
import nibabel as nib
import nitime as nit

from lyman import gather_project_info

def extract_subject(subj, mask_name, summary_func=np.mean,
                    exp_name=None):
    """Extract timeseries from within a mask, summarizing flexibly.

    Parameters
    ----------
    subj : string
        subject name
    mask_name : string
        name of mask in data hierarchy
    summary_func : callable or None
        callable to reduce data over voxel dimensions. can take an
        ``axis`` argument to operate over each frame, if this
        argument does not exist the function will be called on the
        n_tr x n_voxel array. if None, simply returns all voxels.
    exp_name : string
        experiment name, if not using the default experiment

    Returns
    -------
    data : dict with ndarray
        datta array is n_runs x n_timepoint x n_dimension,
        data are not otherwise altered

    """
    project = gather_project_info()
    if exp_name is None:
        exp_name = project["default_exp"]

    # Get a path to the file where 
    cache_dir = op.join(project["analysis_dir"],
                      exp_name, subj, "evoked")

    try:
        os.makedirs(cache_dir)
    except OSError:
        pass

    if summary_func is None:
        func_name = ""
    else:
        func_name = summary_func.__name__
    cache_fname = mask_name + "_" + func_name
    cache_fname = cache_fname.strip("_") + ".npz"
    cache_file = op.join(cache_dir, cache_fname)

    # Get paths to the relevant files
    mask_file = op.join(project["data_dir"], subj, "masks",
                        "%s.nii.gz" % mask_name)
    ts_dir = op.join(project["analysis_dir"], exp_name, subj,
                     "reg", "epi", "unsmoothed")
    n_runs = len(glob(op.join(ts_dir, "run_*")))
    ts_files = [op.join(ts_dir, "run_%d" % (r_i + 1),
                        "timeseries_xfm.nii.gz") for r_i in range(n_runs)]

    # Get the hash value for this extraction
    cache_hash = hashlib.sha1()
    cache_hash.update(mask_name)
    cache_hash.update(str(op.getmtime(mask_file)))
    for ts_file in ts_files:
        cache_hash.update(str(op.getmtime(ts_file)))
    cache_hash = cache_hash.hexdigest()

    # If the file exists and the hash matches, return the data
    if op.exists(cache_file):
        cache_obj = np.load(cache_file)
        if cache_hash == str(cache_obj["hash"]):
            return dict(cache_obj.items())

    # Otherwise, do the extraction
    data = []
    mask = nib.load(mask_file).get_data().astype(bool)
    for run, ts_file in enumerate(ts_files):
        ts_data = nib.load(ts_file).get_data()
        roi_data = ts_data[mask].T

        if summary_func is None:
            data.append(roi_data)
            continue

        # Try to use the axis argument to summarize over voxels
        try:
            roi_data = summary_func(roi_data, axis=1)
        # Catch a TypeError and just call the function 
        # This lets us do e.g. a PCA
        except TypeError:
            roi_data = summary_func(roi_data)

        data.append(roi_data)

    data = map(np.squeeze, data)

    # Save the results and return them
    data_dict = dict(data=data, subj=subj, hash=cache_hash)
    np.savez(cache_file, **data_dict)

    return data_dict


def extract_group(mask_name, summary_func=np.mean,
                  exp_name=None, subjects=None, dv=None):
    """Extract timeseries from within a mask, summarizing flexibly.

    Parameters
    ----------
    mask_name : string
        name of mask in data hierarchy
    summary_func : callable or None
        callable to reduce data over voxel dimensions. can take an
        ``axis`` argument to operate over each frame, if this
        argument does not exist the function will be called on the
        n_tr x n_voxel array. if None, simply returns all voxels.
    exp_name : string
        experiment name, if not using the default experiment
    subjects : sequence of strings
        subjects to operate over if not using default subject list
    dv : IPython cluster direct view
        if provided with view on cluster, executes in parallel over
        subjects

    Returns
    -------
    data : list of dicts with ndarrays
        each array is squeezed n_runs x n_timepoint x n_dimension
        data is not otherwise altered

    """
    if dv is None:
        import __builtin__
        _map = __builtin__.map
    else:
        _map = dv.map_sync

    if subjects is None:
        subj_file = op.join(os.environ["LYMAN_DIR"], "subjects.txt")
        subjects = np.loadtxt(subj_file, str)

    mask_name = [mask_name for s in subjects]
    summary_func = [summary_func for s in subjects]
    exp_name = [exp_name for s in subjects]

    data = _map(extract_subject, subjects, mask_name,
                summary_func, exp_name)
    for d in data:
        d["data"] = np.asarray(d["data"])

    return data


def calculate_evoked(data, n_bins, onsets=None, problem=None, tr=2,
                     calc_method="FIR", offset=0, percent_change=True,
                     correct_baseline=True):

    project = gather_project_info()
    event_template = op.join(project["data_dir"], "%s",
                             "events/%s.npz" % problem)
    evoked = []
    for i, data_i in enumerate(data):

        if problem is not None:
            subj = data_i["subj"]
            event_obj = np.load(event_template % subj)
            ev_data = [event_obj[name] for name in event_obj["event_names"]]
            onsets = [[r[:, 0] for r in d] for d in ev_data]
            
        event_list = []
        data_list = []
        for run, run_data in enumerate(data_i["data"]):

            events_i = np.zeros_like(run_data)
            for ev_id, ev_onsets in enumerate(onsets, 1):
                run_onsets = ev_onsets[run] + offset
                onset_frames = (run_onsets / tr).astype(int)
                events_i[onset_frames] = ev_id
            event_list.append(events_i)

            if percent_change:
                run_data = nit.utils.percent_change(run_data, ax=0)
            data_list.append(run_data)

        events = np.concatenate(event_list)
        events_ts = nit.TimeSeries(events, sampling_interval=tr)
        data = np.concatenate(data_list)
        data_ts = nit.TimeSeries(data, sampling_interval=tr)

        analyzer = nit.analysis.EventRelatedAnalyzer(
            data_ts, events_ts, n_bins)

        evoked_data = getattr(analyzer, calc_method).data
        if correct_baseline:
            evoked_data = evoked_data - evoked_data[:, 0, None]
        evoked.append(evoked_data)

    return np.array(evoked)


def integrate_evoked(evoked):

    int_evoked = []
    if np.array(evoked).ndim < 3:
        evoked = [evoked]
    for data in evoked:
        int_data = sp.integrate.trapz(data, axis=-1)
        int_evoked.append(int_data)
    return np.array(int_evoked)
