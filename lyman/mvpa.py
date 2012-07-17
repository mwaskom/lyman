from __future__ import division
import numpy as np
import scipy as sp
import nipy.modalities.fmri.hemodynamic_models as hrf


def iterated_deconvolution(schedule, data):
    """Deconvolve stimulus events from an ROI data matrix."""
    pass


def event_designs(evs, ntp, tr=2, hrf_model="canonical",
                  split_confounds=True):
    """Generator function to return event-wise design matrices."""
    evs = np.asarray(evs)
    n_ev = len(evs)
    ev_ids = [np.zeros(evs[:, 0].size) * i for i in range(n_ev)]
    ev_ids = np.concatenate(ev_ids)
    stim_idxs = np.concatenate([np.arange(len(ev)) for ev in evs])
    master_sched = np.row_stack(evs)
    master_sched = np.column_stack((master_sched, ev_ids, stim_idxs))
    timesorter = np.argsort(master_sched[:, 0])
    master_sched = master_sched[timesorter]

    frametimes = np.linspace(0, ntp - tr, ntp / tr)

    n_cond = len(evs)
    for ii, row in enumerate(master_sched):
        time, dur, amp, ev_id, stim_idx = row
        ev_interest = evs[ev_id][stim_idx]
        ev_interest = np.atleast_2d(ev_interest).T

        design_mat, _ = hrf.compute_regressor(ev_interest,
                                              hrf_model,
                                              frametimes)

        if split_confounds:
            for cond in range(n_cond):
                conf_sched = master_sched[master_sched[:, 3] == cond]
                if n_cond == ev_id:
                    if ii < len(master_sched):
                        conf_sched = conf_sched[:stim_idx, stim_idx + 1:]
                    else:
                        conf_sched = conf_sched[:-1]
                conf_reg, _ = hrf.compute_regressor(conf_sched[:, :3].T,
                                                    hrf_model,
                                                    frametimes)
                design_mat = np.column_stack((design_mat, conf_reg))
        else:
            if ii < len(master_sched):
                conf_sched = master_sched[:ii, ii + 1:]
            else:
                conf_sched = master_sched[:-1]
            conf_reg, _  = hrf.compute_regressor(conf_sched[:, :3].T,
                                                 hrf_model,
                                                 frametimes)
            design_mat = np.column_stack((design_mat, conf_reg))

        yield design_mat


def deconvolve_event(X, data):
    """Given a design matrix for a particular event return a row of betas."""
    pass
