# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 13:59:38 2018

@author: legrand
"""
import os
import mne
import pickle
import numpy as np
import seaborn as sns
from scipy import stats
import scipy.io as sio
from mne.viz import plot_topomap
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from mne.stats import permutation_cluster_1samp_test
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Setup path
data_path = '/media/legrand/DATAPART1/ENGRAMME/GROUPE_2/EEG/'

# Working directory
wd_path = '/media/legrand/DATAPART1/EEG_wd/'

# Subjects ID
Names           = os.listdir(data_path)

# %% Import index of cardiac control
bpm = []
with open('bpm.txt', "rb") as fp:
    bpm = pickle.load(fp)

#%% Store in array

def store_array(contrasts):
    data = np.array([c.data for c in contrasts])

    connectivity = None
    tail = 0 # 0.  # for two sided test

    # set cluster threshold
    p_thresh    = 0.01 / (1 + (tail == 0))
    n_samples   = len(data)
    threshold   = -stats.t.ppf(p_thresh, n_samples - 1)

    # Make a triangulation between EEG channels locations to
    # use as connectivity for cluster level stat
    connectivity = mne.channels.find_ch_connectivity(contrast.info, 'eeg')[0]

    data = np.transpose(data, (0, 2, 1))  # transpose for clustering

    random_state = 42
    cluster_stats = permutation_cluster_1samp_test(data,
                                                   threshold=threshold,
                                                   verbose=True,
                                                   connectivity=connectivity,
                                                   out_type='indices',
                                                   check_disjoint=True,
                                                   step_down_p=0.05,
                                                   seed=random_state)

    T_obs, clusters, p_values, _ = cluster_stats
    good_cluster_inds = np.where(p_values < 0.05)[0]

    print("Good clusters: %s" % good_cluster_inds)

    return good_cluster_inds, data, clusters, T_obs, p_values

# %% Plot
def plottopo(contrasts, cluster, clu_idx, band):

    sns.set_context("paper", font_scale=1.4)

    times = contrasts[0].times * 1e3
    pos = mne.find_layout(contrasts[0].info).pos

    T_obs_min = T_obs.min()/2
    T_obs_max = -T_obs_min

    # unpack cluster information, get unique indices
    time_inds, space_inds = np.squeeze(clusters[clu_idx])
    ch_inds     = np.unique(space_inds)
    time_inds   = np.unique(time_inds)

    # get topography for T0 stat
    T_obs_map = T_obs[time_inds, ...].mean(axis=0)

    signals = data[..., ch_inds].mean(axis=-1)
    sig_times = times[time_inds]

    # create spatial mask
    mask = np.zeros((T_obs_map.shape[0], 1), dtype=bool)
    mask[ch_inds, :] = True

    # initialize figure
    fig, ax_topo = plt.subplots(1, 1, figsize=(5, 5))

    # plot average test statistic and mark significant sensors
    image, _ = plot_topomap(T_obs_map, pos, mask=mask, axes=ax_topo,
                            vmin=T_obs_min, vmax=T_obs_max,
                            show=False, mask_params = dict(marker='o', markerfacecolor='w',
                                                           markeredgecolor='k',
                                                           linewidth=0, markersize=6))

    # advanced matplotlib for showing image with figure and colorbar
    # in one plot
    divider = make_axes_locatable(ax_topo)

    # add axes for colorbar
    ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(image, cax=ax_colorbar, format='%0.1f')
    ax_topo.set_xlabel('Averaged t-map\n({:0.1f} - {:0.1f} ms)'.format(
        *sig_times[[0, -1]]
    ))

    # add new axis for time courses and plot time courses
    ax_signals = divider.append_axes('bottom', size='30%', pad=0.5)

    ax_signals = sns.tsplot(signals,
                            time = times,
                            ci = 68,
                            color='b')

    ax_signals = sns.tsplot(signals,
                            time = times,
                            err_style = None,
                            color='k')

    ax_signals.plot(sig_times,
                    signals[:,time_inds].mean(0),
                    color='k',
                    linewidth = 5)

    # add information
    ax_signals.axvline(0, color='k', linestyle='--')
    ax_signals.axhline(0, color='k', linestyle='--')

    ax_signals.set_xlim([times[0], times[-1]])
    ax_signals.set_xlabel('Time after cue onset [ms]')
    ax_signals.set_ylabel(r'$\Delta$ Power')

    # plot significant time range
    ax_signals.fill_between(sig_times,
                            signals[:,time_inds].mean(0),
                            color='b')

    # clean up viz
    fig.tight_layout(pad=0.5, w_pad=0)
    fig.subplots_adjust(bottom=.05)
    plt.savefig('EEG_' + band + '.svg', dpi = 600)
    plt.show()

# %% Plot correlation with BPM
def plot_ecg_corr(data, clust, band):

    # unpack cluster information, get unique indices
    time_inds, space_inds = np.squeeze(clust)
    ch_inds = np.unique(space_inds)
    time_inds = np.unique(time_inds)
    sns.set_context("paper", font_scale=2)
    sns.set_style("white")

    # get signals at significant sensors
    sig = []
    for i in range(24):
        su = data[i, time_inds[0]:time_inds[-1], ch_inds].mean((0, 1))
        sig.append(su)

    mean_signal = np.asarray(sig)
    sns.jointplot(bpm[2],
                  mean_signal * 10e9,
                  size = 4,
                  color = 'indigo',
                  stat_func = spearmanr,
                  kind="reg")

    plt.savefig('EEG_BPM' + band + '.svg', dpi = 600, bbox_inches='tight')
    plt.show()

    # Save results for the correlation toolbox
    sio.savemat('CORR.mat', {'bpm': bpm[2],
                             'eeg':mean_signal})

# %% Global results

# Loop across each TFR/subjects
contrasts = []
for subject in Names:

    # Read ERF
    TOT = []
    with open(wd_path + 'Cardiac_control/TNT/8_ERF_theta/' + subject + '-ERF.txt', "rb") as fp:
        TOT = pickle.load(fp)

    TOT[-1].apply_baseline(mode='mean', baseline=(None, 0))
    TOT[0].apply_baseline(mode='mean', baseline=(None, 0))

    # Set contrast between the two conditions
    contrast = TOT[0] - TOT[-1]

    # Average over all frequencies
    contrast._data = np.mean(contrast._data[:,:,:], 1)
    contrasts.append(contrast) # Store in contrasts

# Clusters statistics
good_cluster_inds, data, clusters, T_obs, p_values = store_array(contrasts)

# Plot topomap
plottopo(contrasts, clusters, good_cluster_inds[0], '5-9HZ')

# Plot correlation
plot_ecg_corr(data, clusters[good_cluster_inds[0]], '5-9HZ')
