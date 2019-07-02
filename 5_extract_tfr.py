# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 13:59:38 2018

@author: legrand
"""
import mne
import ntpath
import numpy as np
import os
import pandas as pd
from mne.time_frequency import tfr_multitaper

path = 'E:/EEG_wd/Machine_learning/'  # Setup path
outpath = 'C:/Users/legrand/Dropbox/Productions/Machine_learning/'

Names = os.listdir('E:/EEG_wd/Machine_learning/TNT/All_frequencies_morlet/')

Names = sorted(list(set([subject[:5] for subject in Names])))

# ======================
# %% Extract TNT
# ======================


def data_tnt(subject):
    """Extract TNT epochs.

    Extract epochs from TNT recording, removing criterion and non encoded items

    Input
    -----
    * subject: string
        Subject reference.

    Output
    -----
    * epoch_TNT: mne Epochs type.

    * eprime: Pandas Dataframe
    """
    data_path = 'E:/ENGRAMME/GROUPE_2/EEG/'
    criterion_path = 'E:/ENGRAMME/GROUPE_2/COMPORTEMENT/'

    # Load preprocessed epochs
    in_epoch = path + 'TNT/5_autoreject/' + subject + '-epo.fif'
    epochs_TNT = mne.read_epochs(in_epoch, preload=True)
    epochs_TNT.pick_types(emg=False, eeg=True, stim=False, eog=False,
                          misc=False, exclude='bads')

    # Load e-prime file
    eprime_df = data_path + subject + '/' + subject + '_t.txt'
    eprime = pd.read_csv(eprime_df, skiprows=1, sep='\t')
    eprime = eprime[['Cond1', 'Cond2', 'Image.OnsetTime', 'ImageFond',
                     'Black.RESP', 'ListImage.Cycle']]
    eprime = eprime.drop(eprime.index[[97, 195, 293]])
    eprime['ListImage.Cycle'] = eprime['ListImage.Cycle'] - 1
    eprime.reset_index(inplace=True)

    # Droped epochs_TNT
    eprime = eprime[[not i for i in epochs_TNT.drop_log]]

    # Remove criterion
    Criterion = pd.read_csv(criterion_path + subject + '/TNT/criterion.txt',
                            encoding='latin1', sep='\t', nrows=78)
    forgotten = [ntpath.basename(i)
                 for i in Criterion[' Image'][Criterion[' FinalRecall'] == 0]]

    if len(forgotten):
        epochs_TNT.drop(eprime['ImageFond'].str.contains('|'.join(forgotten)))
        eprime = eprime[~eprime['ImageFond'].str.contains('|'.join(forgotten))]

    return epochs_TNT, eprime

# =========================================
# %% Extract Frequencies - Think - No-Think
# =========================================


def extract_frequencies(subject, freqs, decim):
    """Filter TNT epochs using multitaper.

    Input
    -----
    * subject: str
        Subject reference.

    freqs: array like
        Frequency range to extract.

    decim: int
        Decimation parameter

    Output
    ------

    Save -tfr.h5 in the '/All_frequencies_multitaper' directory

    """
    n_cycles = freqs/2

    # TNT
    tnt, tnt_df = data_tnt(subject)

    this_tfr = tfr_multitaper(tnt, freqs, n_cycles=n_cycles,
                              n_jobs=6, decim=decim, average=False,
                              return_itc=False)

    this_tfr = this_tfr.crop(-0.5, 3.0)

    this_tfr = this_tfr.apply_baseline(mode='percent',
                                       baseline=(-0.5, 0))

    np.save(path + 'TNT/All_frequencies_multitaper/'
            + subject + '.npy', this_tfr._data)


for subject in Names:
    extract_frequencies(subject, freqs=np.arange(3, 30, 1), decim=20)
