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
    master_sched = np.vstack(evs)
    master_sched = np.column_stack((master_sched[:, 0], ev_ids, stim_idxs))
    timesorter = np.argsort(master_sched[:, 0])
    master_sched = master_sched[timesorter]

    frametimes = np.linspace(0, ntp - tr, ntp / tr)

    for ii, (time, ev_id, stim_idx) in enumerate(master_sched):
        ev_interest = evs[ev_id][stim_idx]
        ev_interest = np.atleast_2d(ev_interest).T

        reg_interest, _ = hrf.compute_regressor(ev_interest,
                                                hrf_model,
                                                frametimes)

        yield reg_interest


def deconvolve_event(X, data):
    """Given a design matrix for a particular event return a row of betas."""
    pass
