"""
"""
import re

template_args = dict(timeseries=[["subject_id", "bold", ["DK_run?"]]])

source_template = "%s/%s/%s.nii.gz"

smooth_fwhm = 6

highpass_sigma = 128
TR = 2.
units = "secs"

frames_to_toss = 6

slice_time_correction = True
interleaved = False
slice_order = "up"

n_runs = 6

fsl_bases = {"dgamma":{"derivs":False}}
spm_bases = {"hrf":[0,0]}

parfile_base_dir = "/mindhive/gablab/fluid/Data"
parfile_template = "%s/parfiles/IQ_r%d_d%d_%s_%s.txt"
parfile_args = ["subject_id", "run_number", "day", "event", "subject_id"]

events = ["easy", "hard"]

cont01 = ["easy", "T", events, [1,0]]
cont02 = ["hard", "T", events, [0,1]]
cont03 = ["easy-hard", "T", events, [1,-1]]
cont04 = ["hard-easy", "T", events, [-1,1]]

convars = [var for var in dir() if re.match("cont\d+",var)]
convars.sort()

contrasts = [locals()[con] for con in convars]
