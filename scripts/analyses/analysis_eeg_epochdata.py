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
raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False,
                          preload=True).crop(tmax=60)# we'll use the 60 sec of the data for now
# raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True, verbose=False)

# load events
events_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                       'sample_audvis_raw-eve.fif')
events = mne.read_events(events_file)

# or using the trigger channel
trig_events = mne.find_events(raw, stim_channel='STI 014')

# to epoch data we need the (filtered) eeg data and events file. This will create an Epochs obejct:
# first create an events didct. Note that if this is not passed when epoching, mne will create one automatically
events_dict = {'audt/left': 1, 'audt/right': 2, 'vis/left': 3, 'vis/right': 4,
              'face': 5, 'resp': 32}
epoched = mne.Epochs(raw, events, tmin=-0.3, tmax=0.7, event_id=events_dict,
                    preload=True)

print(epoched.event_id)
del raw # free memory

# view the dropped epochs to see why epoches were dropped
print(epoched.drop_log)

# visualise epochs
# epoched.plot()

# or specify smaller portion of the epoched data
epoched.plot(n_epochs=10)


# selecting epochs

print(epoched['face'])

# pool across left and right condition
print(epoched['audt'])
assert len(epoched['audt']) == (len(epoched['audt/left']) +
                                   len(epoched['audt/right']))
# pool across auditory and visual
print(epoched['left'])
assert len(epoched['left']) == (len(epoched['audt/left']) +
                               len(epoched['vis/left']))

# pool conditions by passing multiple tags as a list
print(epoched[['right', 'bottom']]) # as long as one of the tags is is present in the object, there wont be errors

# select epochs by index
# Epoch objects can be indexed with integer, slices or lists of integers. This method ignores event labels
print(epoched[:10]) # epochs 0-9
print(epoched[1:9:2]) # epochs 2,4,6,8

# we can also index strings:
print(epoched['resp'][:4]) # print the first 4 buttopress epochs

# Select, drop and reorder channels
epochs_eeg = epoched.copy().pick_types(meg=False, eeg=True)
print(epochs_eeg.ch_names)

# plot eeg epochs
# reorder channels
new_order = ['EEG 002', 'STI 014', 'EOG 061', 'MEG 2521']
epochs_subset = epoched.copy().reorder_channels(new_order)
print(epochs_subset.ch_names)

del epochs_subset




# Rename channels
# This involves taking a dictionary where thw keys are existing channel names and the
# values are the new name

epoched.rename_channels({'EOG 061': 'blinkchannel'})

epoched.set_channel_types({'EEG 060': 'ecg'})
print(list(zip(epoched.ch_names, epoched.get_channel_types()))[-4:])

# set the back to the correct values
epoched.rename_channels({'blinkchannel': 'EOG 061'})
epoched.set_channel_types({'EEG 060': 'eeg'})
