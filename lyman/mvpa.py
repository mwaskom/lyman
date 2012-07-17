import numpy as np
import scipy as sp
import nipy.modalities.fmri.hemodynamic_models as hrf

def iterated_deconvolution(schedule, data):
    """Deconvolve stimulus events from an ROI data matrix."""
    pass


def event_designs(evs, hrf_model="canonical", split_confounds=True):
    """Generator function to return event-wise design matrices."""
    n_ev = len(evs)
    ev_ids = [np.zeros_like(evs[:, 0]) * i for i in range(n_ev)]
    ev_ids = np.concatenate(ev_ids)
    stim_idx = np.concatenate([np.arange(len(ev)) for ev in evs])
    master_sched = np.vstack(evs)
    master_sched = np.hstack((master_sched[:, 0], ev_ids, stim_idx))
    timesorter = np.argsort(master_sched[:, 0])
    master_sched = master_sched[timesorter]

    for time, ev_id, stim_idx in master_sched:
        ev_interest = evs[ev_id][stim_idx]
        

def deconvolve_event(X, data):
    """Given a design matrix for a particular event return a row of betas."""
    pass
