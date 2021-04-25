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

# raw.crop(0, 60).load_data() # we'll use the 60 sec of the data for now and load to memory

# load events
events_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                       'sample_audvis_raw-eve.fif')
events = mne.read_events(events_file)

'''
Artefact detection and removal:
There are 3 things we can do with artifacts detected in the M/EEG signal:
1) ignore it and run analysis
2) crop the artifact and run analysis
3) repair signal in the infected portion of the signal and run analysis. This is done by
suppressing the artifactual part and leaving the signal of interest intact'''

# First step is to detect the artefact
# first remove projectors (from the sample data) to inspect data in the original form
ssp_projectors = raw.info['projs']
raw.del_proj()

# low frequency drifts are easily detected with visual inspection
mag_channels = mne.pick_types(raw.info, meg='mag') # show all the magnetometer channels
raw.plot(duration = 60, order= mag_channels, n_channels=len(mag_channels), remove_dc=False)

# inspect powerline noise using the psd plot
fig= raw.plot_psd(tmax=np.inf, fmax=250, average=True)

# Clear noise at 60, 120 and 180 Hz (also at 240Hz).
# Inspect heartbeat artefacts
# use the mne.preprocessing module to find and remove ecg artefacts
ecg_epochs = mne.preprocessing.create_ecg_epochs(raw)

ecg_epochs.plot_image(combine='mean')

# The horizontal streaks int he magnetometer image plot reflect the heartbeat artefact
# in superimposed low-frequency drifts. To avoid this apply baseline:
avg_ecg_epochs = ecg_epochs.average().apply_baseline((-0.5, -0.2))

# visualise spatial pattern
avg_ecg_epochs.plot_topomap(times=np.linspace(-0.05, 0.05, 11))

# get an erp plot by providing the times for scalp field maps manually
avg_ecg_epochs.plot_joint(times=[-0.25, -0.025, 0, 0.025, 0.25])

# or leaving times blank. times will be choosen automatically based on peaks in the signal
avg_ecg_epochs.plot_joint()

# To detect eog artifacts use use the mne.preprocessing.create_eog_epochs function
eog_epochs = mne.preprocessing.create_eog_epochs(raw, baseline=(-0.5, -0.2))
eog_epochs.plot_image(combine='mean')
eog_epochs.average().plot_joint()

# Interpolate bad channels
# Bad channels can be stored in a list of bad cahnnels in the Raw info object.

print(raw.info['bads']) # two bad channels

# take a look at the bad EEG channels:
picks = mne.pick_channels_regexp(raw.ch_names, regexp='EEG 05.')

raw.plot(order=picks, n_channels = len(picks))

# and also look at the bad MEG channel
picks = mne.pick_channels_regexp(raw.ch_names, regexp='MEG 2..3')
raw.plot(order=picks, n_channels=len(picks))

# interpolate the two bad channels
# This can be done using the interpolate_bads() method.
# choose data channels (meg, or eeg?)
eeg_data = raw.copy().pick_types(meg=False, eeg=True, exclude=[])
eeg_interpolate = eeg_data.copy().interpolate_bads(reset_bads=False) # to plot the data before and after interpolation
# eeg_interpolate = eeg_data.copy().interpolate_bads(reset_bads=True)

# plot the original and interpolated eeg
for title, data in zip(['original', 'interpolated'], [eeg_data, eeg_interpolate]):
    fig = data.plot(butterfly=True, color='#00000022', bad_color='r')
    fig.subplots_adjust(top=0.9)
    fig.suptitle(title, size='xx-large', weight='bold')

'''
Reject bad data
Annotate bad data and reject data
to reject by annotation parameter use the interactive raw.plot()'''

# plot only the eeg data (in case we have meg and eeg)
eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True)
fig = raw.plot(order=eeg_picks)
fig.canvas.key_press_event('a') # this is done manually

'''
I love creating annotations manually. This is only done using ipython (wont work on this notebok).
In the annotations window create all the annotations (they better start with bad_ because MNE
detects tehm and deletes those parts automatically) and then click on the annotation of interest, and
in the eeg plot click on the part of the signal that you want to annotate.'''

# To remove an annotation just click on it (with 2 fingers on mousepad, for macbook)
# generate anotations with code
# I don't like this method
# for blinks
eog_events = mne.preprocessing.find_eog_events(raw)
onsets = eog_events[:,0]/raw.info['sfreq'] - 0.25
durations = [0.5]*len(eog_events)
descriptions = ['bad blink'] * len(eog_events)

blink_annot = mne.Annotations(onsets, durations, descriptions, orig_time=raw.info['meas_date'])

raw.set_annotations(blink_annot)

# plot eeg only
eeg_picks = mne.pick_types(raw.info, meg = False, eeg=True)
raw.plot(events=eog_events, order=eeg_picks)

'''
Reject epochs based on channel amplitude
That is, if the signal amplitude exceeds a threshold, remove it by specifying maximum
and minimum peak-to-peak amplitudes (flat and reject criteria)'''

#specify flat and reject criteria. Note that threshold are different based on device and so on..
reject_criteria = dict(mag=3000e-15,     # 3000 fT
                       grad=3000e-13,    # 3000 fT/cm
                       eeg=100e-6,       # 100 µV
                       eog=200e-6)       # 200 µV

flat_criteria = dict(mag=1e-15,          # 1 fT
                     grad=1e-13,         # 1 fT/cm
                     eeg=1e-6)           # 1 µV

# reject epochs and plot
epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.5, reject_tmax=0,
                    reject=reject_criteria, flat=flat_criteria,
                    reject_by_annotation=False, preload=True)

#plot
epochs.plot_drop_log()

# reject the epochs with the annotations that were created earlier (in the above rejection we
# removed only channel threshold citeria)
epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.5, reject_tmax=0,
                    reject=reject_criteria, flat=flat_criteria, preload=True)

epochs.plot_drop_log()

# With the annotations + rejection criteria 92 epochs were rejected. Without the
# annotaions 81 epochs were rejected
# look at the epochs drop log
print(epochs.drop_log)

# Filter and resample
# look at the drifts and noise
eeg_channels = mne.pick_types(raw.info, meg = False, eeg=True)
raw.plot(duration=60, order = eeg_channels, proj=False, n_channels=len(eeg_channels), remove_dc=False)

# Try two different highpass filters of 0.1Hz and 0.2Hz and see which works better
for cutoff in (0.1, 0.2):
    raw_highpass = raw.copy().filter(l_freq=cutoff, h_freq=None)
    fig = raw_highpass.plot(duration=60, order=eeg_channels, proj=False,
                            n_channels=len(eeg_channels), remove_dc=False)
    fig.subplots_adjust(top=0.9)
    fig.suptitle('High-pass filtered at {} Hz'.format(cutoff), size='xx-large',
                 weight='bold')

# get the filter parameters (0.2 Hz looks better)
filter_params = mne.filter.create_filter(raw.get_data(), raw.info['sfreq'],
                                         l_freq=0.2, h_freq=None)

mne.viz.plot_filter(filter_params, raw.info['sfreq'], flim=(0.01, 5))

'''
Resample
EEG and MEG recordings are notable for their high temporal precision, and are often
recorded with sampling rates around 1000 Hz or higher. This is good when precise timing of
events is important to the experimental design or analysis plan, but also consumes more memory and
computational resources when processing the data. In cases where high-frequency components of the signal
are not of interest and precise timing is not needed (e.g., computing EOG or ECG projectors on a
long recording), downsampling the signal can be a useful time-saver.'''

raw_downsampled = raw.copy().resample(sfreq=200)

for data, title in zip([raw, raw_downsampled], ['Original', 'Downsampled']):
    fig = data.plot_psd(average=True)
    fig.subplots_adjust(top=0.9)
    fig.suptitle(title)
    plt.setp(fig.axes, xlim=(0, 300))

'''
To avoid the reduction in temporal precision of events that comes with resampling a Raw object,
and also avoid the edge artifacts that come with filtering an Epochs or Evoked object, the best
practice is to low-pass filter the Raw data at or below 1/3 of the desired sample rate, then decimate
the data after epoching, by either passing the decim parameter to the Epochs constructor, or using
the decimate() method after the Epochs have been created.'''

current_sfreq = raw.info['sfreq']
desired_sfreq = 90  # Hz
decim = np.round(current_sfreq / desired_sfreq).astype(int)
obtained_sfreq = current_sfreq / decim
lowpass_freq = obtained_sfreq / 3.

raw_filtered = raw.copy().filter(l_freq=None, h_freq=lowpass_freq)
events = mne.find_events(raw_filtered)
epochs = mne.Epochs(raw_filtered, events, decim=decim)

print('desired sampling frequency was {} Hz; decim factor of {} yielded an '
      'actual sampling frequency of {} Hz.'
      .format(desired_sfreq, decim, epochs.info['sfreq']))
