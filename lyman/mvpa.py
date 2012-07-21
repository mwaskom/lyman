from __future__ import division
import numpy as np
import scipy as sp
import nipy.modalities.fmri.hemodynamic_models as hrf
import moss


def iterated_deconvolution(data, evs, tr=2, hpf_cutoff=128, filter_data=True,
                           copy_data=False, split_confounds=True,
                           hrf_model="canonical"):
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
    for X_i in event_designs(evs, ntp, tr, split_confounds, hrf_model):
        # Filter each design matrix
        if hpf_cutoff is not None:
            X_i = np.dot(F, X_i)
        X_i -= X_i.mean(axis=0)
        # Fit an OLS model
        beta_i, _, _, _ = np.linalg.lstsq(X_i, data)
        # Take the beta for the first regressor
        coef_list.append(beta_i[0])

    return np.array(coef_list)


def event_designs(evs, ntp, tr=2, split_confounds=True,
                  hrf_model="canonical"):
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

    # Generator loop
    for ii, row in enumerate(master_sched):
        # Unpack the schedule row
        time, dur, amp, ev_id, stim_idx = row

        # Generate the regressor for the event of interest
        ev_interest = np.atleast_2d(row[:3]).T
        design_mat, _ = hrf.compute_regressor(ev_interest,
                                              hrf_model,
                                              frametimes)

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
                                                    frametimes)
                design_mat = np.column_stack((design_mat, conf_reg))
        else:
            # Or a single confound regressor
            conf_sched = np.delete(master_sched, ii, 0)
            conf_reg, _ = hrf.compute_regressor(conf_sched[:, :3].T,
                                                hrf_model,
                                                frametimes)
            design_mat = np.column_stack((design_mat, conf_reg))

        yield design_mat
