# import libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import (create_eog_epochs, create_ecg_epochs, compute_proj_ecg, compute_proj_eog)

# %matplotlib inline

# load the data

sample_data_folder = '/Users/christinadelta/datasets/eeg_testing_data'
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file)
# raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True)

raw.crop(0, 60).load_data() # we'll use the 60 sec of the data for now and load to memory

'''
SSP projectors are included in this dataset. This is because the recording system used for this dataset,
isolates environmental noise in SSP projectors, this way (reasonably) clean data can be viewed in
real-time during acquisition. Empty room recording was also acquired for this dataset.
Thus, a new projector will be created'''

# first extract the system projector (already ceated)
system_proj =raw.info['projs']
raw.del_proj()

# load empty room recording
empty_room_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                               'ernoise_raw.fif')
empty_room_raw = mne.io.read_raw_fif(empty_room_file)

# remove the system craeted ssp projectors from the empty room recording (these are created automatically)
empty_room_raw.del_proj()

# Visualise empty-room noise
# take a look at the spectrum of the empty-room noise for each sensor separately:
for average in (False, True):
    empty_room_raw.plot_psd(average=average, dB=False, xscale='log')

# Create new projectors using the empty room noise
empty_room_projs = mne.compute_proj_raw(empty_room_raw, n_grad=3, n_mag=3) # 3 projectors
mne.viz.plot_projs_topomap(empty_room_projs, colorbar=True, vlim='joint',
                           info=empty_room_raw.info) # colormap is computed jointly for each projector for a given channel type


# How do projectors affect the signal?
# visualise the system-projectors and empty-room projector on the signal:
mags = mne.pick_types(raw.info, meg='mag')
for title, projs in [('system', system_proj),
                     ('subject-specific', empty_room_projs[3:])]:
    raw.add_proj(projs, remove_existing=True)
    fig = raw.plot(proj=True, order=mags, duration=1, n_channels=2)
    fig.subplots_adjust(top=0.9)  # make room for title
    fig.suptitle('{} projectors'.format(title), size='xx-large', weight='bold')
