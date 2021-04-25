'''
Run ICA
A tip that I found throuth the MNE webpage: To tun an ICA equivalent to EEGLab's runica() with
'pca', n options to reduce dimensionality via PCA before running the ICA algorithm, I need to
set n_components=n during initialization and pass n_pca_components=n to apply() method.'''

# import libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs, corrmap)

# %matplotlib inline

# load the data
sample_data_folder = '/Users/christinadelta/datasets/eeg_testing_data'
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file)
# raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True)

raw.crop(0, 60).load_data() # we'll use the 60 sec of the data for now and load to memory

# load events
events_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                       'sample_audvis_raw-eve.fif')
events = mne.read_events(events_file)

# EOG and ECG artefact repair
# First visualise the artifacts to repair

# pick some channels that clearly show heartbeats and blinks
regexp = r'(MEG [12][45][123]1|EEG 00.)'
artifact_picks = mne.pick_channels_regexp(raw.ch_names, regexp=regexp)
raw.plot(order=artifact_picks, n_channels=len(artifact_picks),
         show_scrollbars=False)

# get a summary of how the eog (ocular) artefact manifests acrros channel types by creating eog epochs and
eog_epochs = create_eog_epochs(raw).average()
eog_epochs.apply_baseline(baseline =(None, -0.2))
eog_epochs.plot_joint()

# get a summary of how the ecg (hartbeat) artefact manifests acrros channel types by creating ecg epochs and
ecg_epochs = create_ecg_epochs(raw).average()
ecg_epochs.apply_baseline(baseline =(None, -0.2))
ecg_epochs.plot_joint()

# before running the ICA filter the data to remove low-frequency drifts which can negatively
# affect the fit of the ICA algorithm
# lowpass filter the data to remove slow drifts
filter_raw = raw.copy() # create a copy of the raw data
filter_raw.load_data().filter(l_freq=1., h_freq=None) # add the copy to memory and apply low_freq filtering

# Fit ICA algorithm
ica = ICA(n_components=20, random_state=97) # i think 20 components are good for this dataset
ica.fit(filter_raw)

# Look at the ICs to see what the have captured (we can use the original unfiltered raw here):
raw.load_data()
ica.plot_sources(raw, show_scrollbars=False)

# plot the components on topomaps
ica.plot_components()

# or plot the raw (ica trainned) data to see differences
ica.plot_components(inst=filter_raw)

# It is obvious from the ica sources plot that the first component captures the blinks
# pretty well and the second captures the heartbeats. But let's take a closer look:

# First plot an overlay of the original signal against the reconstructed signal excluding the artefacts:
# for blinks:
ica.plot_overlay(raw, exclude=[0], picks='eeg')

# for heartbeats
ica.plot_overlay(raw, exclude=[1], picks='mag')

# plot diagnostics of the bad components
ica.plot_properties(raw, picks=[0,1])

# exclude the bad components and plot the IC data again
ica.exclude = [0,1] # blinks and heartbeats components

# reconstruct the raw using ica.apply() and plot side by side with raw
reconstr_raw = raw.copy() # first make a copy
ica.apply(reconstr_raw)

# plot raw
raw.plot(order=artifact_picks, n_channels=len(artifact_picks),
        show_scrollbars=False)

# plot reconstructed
reconstr_raw.plot(order=artifact_picks, n_channels=len(artifact_picks),
                 show_scrollbars=False)
