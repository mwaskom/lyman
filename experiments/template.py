source_template = "%s/bold/scan??.nii.gz"

nruns = 6

frames_to_toss = 6

slice_time_correction = True
interleaved = False
slice_order = "up"

smooth_fwhm = 6

hpf_cutoff = 128
TR = 2.

hrf_model = "dgamma"
hrf_derivs = False

parfile_base_dir = "/mindhive/gablab/fluid/Data"
parfile_template = "%(subject_id)s/parfiles/IQ_r%(run)d_d1_%(event)s_%(subject_id)s.txt"
units = "secs"

events = ["easy", "hard"]

cont01 = ["easy", "T", events, [1,0]]
cont02 = ["hard", "T", events, [0,1]]
cont03 = ["easy-hard", "T", events, [1,-1]]
cont04 = ["hard-easy", "T", events, [-1,1]]
