"""
@author: Nicolas Legrand (legrand@cyceron.fr)

Preprocess and frequency analysis

"""

import mne
import pandas as pd
import os
import ntpath
import pickle
import numpy as np
from mne.preprocessing import ICA
import autoreject
from functools import partial
from mne.time_frequency import tfr_morlet

data_path   = '/media/legrand/DATAPART1/ENGRAMME/GROUPE_2/EEG/'   # Setup path
Names       = os.listdir(data_path)                               # Subjects ID
wd_path     = '/media/legrand/DATAPART1/EEG_wd/'                  # Working directory

# Parameter
tmin, tmax  = -1.0 , 3.5 # Epochs limits
drop        = ['E43', 'E49', 'E56', 'E63', 'E68', 'E73', 'E81', 'E88', 'E94', 'E99', 'E107', 'E113', 'E120']
fname       = {'eeg' : '_t.fil.edf', 'eprime' : '_t.txt'}

chan_rename = {'EEG ' + str(i):'E'+str(i) for i in range(1,129)} # Montage
chan_rename['EEG VREF'] = 'Cz'

# Set channels types
mapping = {'E1'     : "eog", 'E8'     : "eog", 'E14'    : "eog",
           'E21'    : "eog", 'E25'    : "eog", 'E32'    : "eog",
           'E125'   : "eog", 'E126'   : "eog", 'E127'   : "eog",
           'E128'   : "eog", 'E48'    : "emg", 'E119'   : "emg",
           'E17'    : "misc",'Cz'     : "misc"}

# Interpolation
reconstruct = {'32CVI' : ['E48'],                 '31NLI' : ['E33', 'E125'],
               '34LME' : ['E123'],                '35QSY' : ['E116'],
               '36LSA' : [],                      '37BMA' : ['E108'],
               '38MAX' : [],                      '39BDA' : [],
               '40MMA' : ['E39', 'E57'],          '41BAL' : ['E32', 'E114'],
               '42SPE' : ['E22'],                 '44SMU' : ['E114'],
               '45MJA' : [],                      '46SQU' : [],
               '47HMA' : ['E45'],                 '50JOC' : ['E46', 'E48', 'E77'],
               '52PFA' : ['E64'],                 '53SMA' : ['E27'],
               '55MNI' : [],                      '56BCL' : [],
               '57NCO' : [],                      '58BAN' : ['E30'],
               '59DIN' : [],                      '60CAN' : ['E115']}

eog = {}
eog['TNT'] = {'31NLI' : 'E25','32CVI' : 'E8', '34LME' : 'E14', '35QSY' : 'E25',
              '36LSA' : 'E25', '37BMA' : 'E8', '38MAX' : 'E25', '39BDA' : 'E25',
              '40MMA' : 'E25', '41BAL' : 'E21', '42SPE' : 'E25', '44SMU' : 'E8',
              '45MJA' : 'E14', '46SQU' : 'E25', '47HMA' : 'E25', '50JOC' : 'E25',
              '52PFA' : 'E8', '53SMA' : 'E21', '55MNI' : 'E25', '56BCL' : 'E25',
              '57NCO' : 'E25', '58BAN' : 'E25', '59DIN' : 'E25', '60CAN' : 'E25'}
# %%
def run_filter(subject, task, overwrite):

    # Load edf file
    subject_path = data_path + subject + '/' + subject + fname['eeg']
    raw = mne.io.read_raw_edf(subject_path, preload=True)

    # Rename channels
    mne.channels.rename_channels(raw.info, chan_rename)

    # Set montage
    montage = mne.channels.read_montage('GSN-HydroCel-129')
    mne.io.Raw.set_montage(raw, montage)
    raw.set_channel_types(mapping=mapping)

    # Drop channels
    raw.drop_channels(drop)

    # Save raw data
    out_raw   = wd_path + task + '/1_raw/' + subject + '-raw.fif'
    raw.save(out_raw, overwrite = overwrite)

    # Filter
    raw.filter(None, 40, l_trans_bandwidth='auto', h_trans_bandwidth='auto', n_jobs = 8,
               filter_length='auto', phase='zero', fir_window='hamming', fir_design='firwin')

    # Interpolate bad channels
    raw.info['bads'] = reconstruct[subject]
    raw.interpolate_bads(reset_bads=True)

    # Set EEG average reference
    raw.set_eeg_reference('average', projection=True)

    # Save data
    out_rawfilter   = wd_path + 'TNT/2_rawfilter/' + subject + '-raw.fif'
    raw.save(out_rawfilter, overwrite = overwrite)


# %%
def run_epochs(subject, task):

    # Load filtered data
    input_path = wd_path + task + '/2_rawfilter/' + subject + '-raw.fif'
    raw = mne.io.read_raw_fif(input_path)

    # Load e-prime df
    eprime_df   = data_path + subject + '/' + subject + fname['eprime']
    eprime      = pd.read_table(eprime_df, skiprows = 1)
    eprime      = eprime[['Cond1', 'Cond2', 'Image.OnsetTime', 'Black.RESP', 'ImageCentre', 'ImageFond']]

    # Revome training rows after pause for TNT
    eprime = eprime.drop(eprime.index[[97, 195, 293]])
    eprime.reset_index(inplace = True)

    # Find stim presentation in raw data
    events = mne.find_events(raw, stim_channel='STI 014')

    # Compensate for delay (as measured manually with photodiod)
    events[:, 0] += int(.015 * raw.info['sfreq'])

    # Keep only Obj Pres triggers
    events = events[events[:,2] == 7, :]

    # Match stim presentation with conditions from eprime df
    for i in range(len(events)):
        if eprime['Cond1'][i] == 'Think':
            if eprime['Cond2'][i] == 'Emotion':
                events[i,2] = 1
            else:
                events[i,2] = 2
        elif eprime['Cond1'][i] == 'No-Think':
            if eprime['Cond2'][i] == 'Emotion':
                events[i,2] = 3
            else:
                events[i,2] = 4
        else:
            events[i,2] = 5

    # Set event id
    id = {'Think/EMO': 1, 'Think/NEU': 2, 'No-Think/EMO': 3, 'No-Think/NEU': 4}

    # Treshold for rejecting bad epochs
    reject = dict(eeg=350e-6)

    # Epoch raw data
    epochs = mne.Epochs(raw, events, id, tmin, tmax, reject = reject, proj=True)

    # Save epochs
    epochs.save(wd_path + task + '/3_epochs/' + subject + '-epo.fif')

# %%
def run_autoreject(subject, task):

    # Import data
    input_path = wd_path + task + '/3_epochs/' + subject + '-epo.fif'
    epochs     = mne.read_epochs(input_path)

    # Autoreject
    picks = mne.pick_types(epochs.info, eeg=True)  # Find indices of all EEG channels

    thresh_func     = partial(autoreject.compute_thresholds,
                              picks=picks,
                              n_jobs=4,
                              method='bayesian_optimization')

    ar              = autoreject.LocalAutoRejectCV(picks=picks,
                                                   thresh_func=thresh_func)
    epochs_clean = ar.fit_transform(epochs)

    # Save epoch data
    out_epoch = wd_path + task + '/4_autoreject/' + subject + '-epo.fif'
    epochs_clean.save(out_epoch)

# %%
def run_ICA(subject, task):

    input_path = wd_path + task + '/4_autoreject/' + subject + '-epo.fif'
    epochs = mne.read_epochs(input_path)

    # ICA correction
    ica = ICA(n_components=0.95, method='fastica')

    picks = mne.pick_types(epochs.info, meg=False, eeg=True, eog=False, stim=False, exclude='bads')

    ica.fit(epochs, picks=picks, decim=10)

    # Identify bad components
    n_max_eog = 3  # maximum number of components to reject

    eog_inds, scores = ica.find_bads_eog(epochs, ch_name = eog[task][subject])

    eog_inds = eog_inds[:n_max_eog]
    ica.exclude += eog_inds
    ica.apply(epochs)

    # Save scores
    np.save(wd_path + task + '/5_ICA/' + subject + '-scores.npy', scores)
    # Save ICA
    ica.save(wd_path + task + '/5_ICA/' + subject + '-ica.fif')
    # Save epochs
    epochs.save(wd_path + task + '/5_ICA/' + subject + '-epo.fif')

# %% Time Frequency
decim           = 20
freqs           = [5, 6, 7, 8, 9] # Define frequencies of interest
n_cycles        = 5.0

def run_erf(subject, task):

    # Epochs
    eeg_path  = wd_path + task + '/5_ICA/' + subject + '-epo.fif'
    epochs    = mne.read_epochs(eeg_path)

    # Load eprime file
    eprime_path = data_path + subject + '/' + subject + '_t.txt'
    eprime = pd.read_table(eprime_path, skiprows = 1)
    eprime = eprime[['Cond1', 'Cond2', 'Image.OnsetTime', 'ImageFond', 'Black.RESP']]
    eprime = eprime.drop(eprime.index[[97, 195, 293]])
    eprime.reset_index(inplace = True)

    # Droped epochs
    eprime = eprime.loc[epochs.selection]

    # Remove criterion (items forgotten before the TNT task)
    Criterion   = pd.read_table('/media/legrand/DATAPART1/ENGRAMME/GROUPE_2/COMPORTEMENT/' + subject + '/TNT/criterion.txt', encoding='latin1', nrows = 78)
    forgotten   = [ntpath.basename(i) for i in Criterion[' Image'][Criterion[' FinalRecall'] == 0]]

    if len(forgotten):
        epochs.drop(eprime['ImageFond'].str.contains('|'.join(forgotten)))
        eprime      = eprime[~eprime['ImageFond'].str.contains('|'.join(forgotten))]

    # Load condition
    nothink         = epochs['No-Think/EMO']
    think           = epochs[(eprime['Black.RESP'] == 3) & (eprime['Cond1'] == 'Think') & (eprime['Cond2'] == 'Emotion')]

    # Morlet wavelets
    tfr_nothink     = tfr_morlet(nothink, freqs,n_jobs = 16,
                                 n_cycles=n_cycles, decim=decim,
                                 return_itc=False)

    tfr_think       = tfr_morlet(think, freqs,n_jobs = 16,
                                 n_cycles=n_cycles, decim=decim,
                                 return_itc=False)

    # Crop
    tfr_nothink.crop(tmin = -0.5, tmax = 3.0)
    tfr_think.crop(tmin = -0.5, tmax = 3.0)

    # Save results
    TOT = [tfr_nothink, tfr_think]
    with open(wd_path + task + '/8_ERF_theta/' + subject + '-ERF.txt', "wb") as fp:
       pickle.dump(TOT, fp)

# %%
### LOOP ###
############
for subject in Names:
#    run_filter(subject, 'TNT', overwrite=True)
    run_epochs(subject, 'TNT')
#    run_autoreject(subject, 'TNT')
#    run_ICA(subject,'TNT')
#    run_erf(subject,'TNT')
