# import libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import mne

# %matplotlib inline

# load the data

sample_data_folder = '/Users/christinadelta/datasets/eeg_testing_data'
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True)

raw.crop(0, 60).load_data() # we'll use the 60 sec of the data for now and load to memory

# load events
# events_file = os.path.join(sample_data_folder, 'MEG', 'sample',
#                                       'sample_audvis_raw-eve.fif')
# events = mne.read_events(events_file)

# keep only eeg channels
raw.pick(['EEG 0{:02}'.format(n) for n in range(41,60)])

'''
Set or change the refernce channel
If a scalp electrode was used as reference but was not saved with the raw data we need to add it back to
the dataset before re-referencing.
For example if Fp1 was used as reference but it wasn't included in the data file, we need to
add back Fp1 as a flat channel prior to re-referencing using the add_reference_channels() method.
Here, since the electrodes are not set according to the 10-20 or 10-10 system but ar numbered instead,
EEE_999 is added as the missing reference and then sets the reference to EEG_050.'''

# use a single channel reference (left earlobe)
# raw.set_eeg_reference(ref_channels=['A1'])

# use average of mastoid channels as reference
# raw.set_eeg_reference(ref_channels=['M1', 'M2'])

# plot the raw
raw.plot()

# add new reference channel (flat). By default mne.add_reference_channels() createds a copy of the data.
# If we want to alter the raw itself, add parameter: copy=False
raw_newref = mne.add_reference_channels(raw, ref_channels=['EEG 999'])
raw_newref.plot()

# set reference to EEG_050
raw_newref.set_eeg_reference(ref_channels=['EEG 050'])
raw_newref.plot()

'''
Two things to notice:
The EEG 050 electrode is flat (all zeros), while the former reference (EEG 999) now has non-zero values
The EEG 053 bad channel is not affected by the re-referencing
Set average reference instead (especially in cases of source modelling)
NOTE: this will not affect the channels marked as bad, nor will it include bad channels when computing
the average, but it affects the Raw object, so make a copy first'''

# use the average of all channels as reference
raw_avg_ref = raw.copy().set_eeg_reference(ref_channels='average')
raw_avg_ref.plot()

# create the average reference as projector (highly recommended)
raw.set_eeg_reference('average', projection =True)
print(raw.info['projs'])

# visualise original reference and average with projector to look at differences and advantages of 'averaging'
for title, proj in zip(['Original', 'Average'], [False, True]):
    fig = raw.plot(proj=proj, n_channels=len(raw))
    # make room for title
    fig.subplots_adjust(top=0.9)
    fig.suptitle('{} reference'.format(title), size='xx-large', weight='bold')
