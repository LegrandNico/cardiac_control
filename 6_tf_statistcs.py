# Author: Nicolas Legrand (legrand@cyceron.fr)

import matplotlib.pyplot as plt
import mne
from mne.stats import permutation_cluster_1samp_test
from mne.viz import plot_topomap
import numpy as np
import os
import pandas as pd
import pickle
from scipy import stats
import seaborn as sns

path = 'E:/EEG_wd/Machine_learning/'  # Setup path
outpath = 'C:/Users/legrand/Dropbox/Productions/Machine_learning/'

Names = os.listdir(path + 'TNT/All_frequencies_multitaper/')

Names = sorted(list(set([subject[:5] for subject in Names])))

cwd = os.getcwd()

# =============================================================================
# %% TF plot
# =============================================================================


def tfr_permutation(data, title, threshold, tail):
    """Plot and test the event-related perturbation.

    Input
    -----

    * data: numpy asarray

    * title: string
        Name of the file used for saving.

    * threshold: float
        Threshold value used to find cluster.

    * tail: int
        -1, 1 for one-tailed tests.

    Output
    -----
    Save matplotlib figure as 'title' .svg

    """
    n_permutations = 5000
    T_obs, clusters, cluster_p_values, H0 = \
        permutation_cluster_1samp_test(data,
                                       n_permutations=n_permutations,
                                       threshold=threshold,
                                       tail=tail)

    # Create new stats image with only significant clusters
    T_obs_plot = np.nan * np.ones_like(T_obs)
    for c, p_val in zip(clusters, cluster_p_values):
        if p_val <= 0.05:
            T_obs_plot[c] = T_obs[c]

    plt.figure(figsize=(8, 4))
    plt.imshow(data.mean(0), cmap=plt.cm.get_cmap('RdBu_r', 12), vmin=-0.15,
               vmax=0.15, extent=[-0.5, 3, 3, 30], interpolation='gaussian',
               aspect='auto', origin='lower')
    clb = plt.colorbar()
    clb.ax.set_title('% change')
    plt.contour(~np.isnan(T_obs_plot), colors=["w"], extent=[-0.5, 3, 3, 30],
                linewidths=[2], corner_mask=False, antialiased=True,
                levels=[.5])
    plt.axvline(x=0, linestyle='--', linewidth=2, color='k')
    plt.ylabel('Frequencies', size=15)
    plt.xlabel('Time (s)', size=15)
    plt.savefig(cwd + '/Figures/' + title + '.svg', dpi=300)

# =============================================================================
# %% Plot topo
# =============================================================================


def topoplot(data, freq, title, threshold, tail):
    """Plot and test the event-related perturbation.

    Input
    -----

    * data: numpy asarray

    * title: string
        Name of the file used for saving.

    * threshold: float
        Threshold value used to find cluster.

    * tail: int
        -1, 1 for one-tailed tests.

    Output
    -----
    Save matplotlib figure as 'title' .svg

    """
    fig, axs = plt.subplots(1, 7, figsize=(15, 5), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.5, wspace=.001)

    axs = axs.ravel()

    for i, rg in enumerate(range(25, 175, 25)):

        this_data = data[:, :, freq, rg:rg+25].mean((2, 3))

        connectivity = mne.channels.find_ch_connectivity(tnt.info, 'eeg')[0]

        cluster_stats = permutation_cluster_1samp_test(this_data,
                                                       threshold=threshold,
                                                       verbose=True,
                                                       connectivity=connectivity,
                                                       out_type='indices',
                                                       n_jobs=1,
                                                       tail=tail,
                                                       check_disjoint=True,
                                                       step_down_p=0.05,
                                                       seed=42)

        T_obs, clusters, p_values, _ = cluster_stats
        good_cluster_inds = np.where(p_values < 0.05)[0]

        # Extract mask and indices of active sensors in the layout
        mask = np.zeros((T_obs.shape[0], 1), dtype=bool)
        if len(clusters):
            for clus in good_cluster_inds:
                mask[clusters[clus], :] = True

        evoked = mne.EvokedArray(T_obs[:, np.newaxis],
                                 tnt.average().info, tmin=0.)

        evoked.plot_topomap(ch_type='eeg', times=0, scalings=1,
                            time_format=None, cmap=plt.cm.get_cmap('RdBu_r', 12), vmin=-6., vmax=6,
                            units='t values', mask = mask, axes = axs[i],
                            size=3, show_names=lambda x: x[4:] + ' ' * 20,
                            time_unit='s', show=False)

        plt.savefig(cwd + '/Figures/' + title + '_topo.svg')

# =============================================================================
# %% Think / No-Think
# =============================================================================

total_tfr = []
for subject in Names:

    tfr = np.load(path + 'TNT/All_frequencies_multitaper/' + subject + '.npy')

    tnt_df = pd.read_csv(path + 'TNT/All_frequencies_multitaper/'
                         + subject + '.txt')

    think = tfr[(tnt_df.Cond1 == "Think") & (tnt_df.Cond2 == "Emotion")].mean(0)
    nothink = tfr[(tnt_df.Cond1 == "No-Think") & (tnt_df.Cond2 == "Emotion")].mean(0)

    total_tfr.append(nothink - think)

np.save(path + 'TNT_emotion_percent.npy', np.asarray(total_tfr))

total_tfr = np.load(path + 'TNT_emotion_percent.npy')

tfr_permutation(np.asarray(total_tfr).mean(1), 'TNT', threshold=-2.5)

data = np.asarray(total_tfr)

theta = np.arange(0, 6)
topoplot(data, theta, 'Theta_TNT', threshold=-2.5, tail=-1)

alpha = np.arange(6, 10)
topoplot(data, alpha, 'Alpha_TNT', threshold=-2.5, tail=-1)

lowbeta = np.arange(10, 17)
topoplot(data, lowbeta, 'LowBeta_TNT', threshold=-2.5, tail=-1)

highbeta = np.arange(17, 27)
topoplot(data, highbeta, 'HighBeta_TNT', threshold=-2.5, tail=-1)

# Import BPM
bpm = []
with open('bpm.txt', "rb") as fp:
    bpm = pickle.load(fp)

bpm = bpm[2]

# Extract averaged power over time and frequency of interest
pw = total_tfr[:, :, theta, 75:150].mean((1, 2, 3))
stats.mannwhitneyu(pw[bpm < np.median(bpm)], pw[bpm >= np.median(bpm)])
th = pd.DataFrame({'Low control': pw[bpm < np.median(bpm)], 'High control':pw[bpm >= np.median(bpm)]}).melt()
th['Frequency'] = 'Theta'

# Extract averaged power over time and frequency of interest
pw = total_tfr[:, :, alpha, 75:150].mean((1, 2, 3))
stats.mannwhitneyu(pw[bpm < np.median(bpm)], pw[bpm >= np.median(bpm)])
al = pd.DataFrame({'Low control': pw[bpm < np.median(bpm)], 'High control':pw[bpm >= np.median(bpm)]}).melt()
al['Frequency'] = 'Alpha'


# Extract averaged power over time and frequency of interest
pw = total_tfr[:, :, lowbeta, 75:150].mean((1, 2, 3))
stats.mannwhitneyu(pw[bpm < np.median(bpm)], pw[bpm >= np.median(bpm)])
lb = pd.DataFrame({'Low control': pw[bpm < np.median(bpm)], 'High control':pw[bpm >= np.median(bpm)]}).melt()
lb['Frequency'] = 'Low-beta'


# Extract averaged power over time and frequency of interest
pw = total_tfr[:, :, highbeta, 75:150].mean((1, 2, 3))
stats.mannwhitneyu(pw[bpm < np.median(bpm)], pw[bpm >= np.median(bpm)])
hb = pd.DataFrame({'Low control': pw[bpm < np.median(bpm)], 'High control':pw[bpm >= np.median(bpm)]}).melt()
hb['Frequency'] = 'High-beta'


data = pd.concat([th, al, lb, hb], ignore_index=True)

data.to_csv('EEG_TNT.txt')

# =============================================================================
# %% Intrusion / Non-intrusion
# =============================================================================

total_tfr = []
for subject in Names:

    tfr = np.load(path + 'TNT/All_frequencies_multitaper/' + subject + '.npy')

    tnt_df = pd.read_csv(path + 'TNT/All_frequencies_multitaper/'
                         + subject + '.txt')

    intrusion = tfr[(tnt_df['Black.RESP'] != 1) & (tnt_df.Cond1 == "No-Think") & (tnt_df.Cond2 == "Emotion")].mean(0)
    nonintrusion = tfr[(tnt_df['Black.RESP'] == 1) & (tnt_df.Cond1 == "No-Think") & (tnt_df.Cond2 == "Emotion")].mean(0)

    total_tfr.append(nonintrusion - intrusion)

np.save(path + 'Intrusions_emotion_percent.npy', np.asarray(total_tfr))


total_tfr = np.load(path + 'Intrusions_emotion_percent.npy')

pick = []
for subject in Names:

    tnt_df = pd.read_csv(path + 'TNT/All_frequencies_multitaper/'
                         + subject + '.txt')

    int = sum((tnt_df['Black.RESP'] != 1) & (tnt_df.Cond1 == "No-Think") & (tnt_df.Cond2 == "Emotion"))
    ni = sum((tnt_df['Black.RESP'] == 1) & (tnt_df.Cond1 == "No-Think") & (tnt_df.Cond2 == "Emotion"))

    if (int > 5) & (ni > 5):
        pick.append(True)
    else:
        pick.append(False)

total_tfr = total_tfr[pick]


# =============================================================================
# Plots
# =============================================================================

tfr_permutation(np.asarray(total_tfr).mean(1), 'NonIntrusions', threshold=2.5, tail=1)

data = total_tfr

theta = np.arange(0, 6)
topoplot(data, theta, 'Theta_NonIntrusions', threshold=2.5, tail=1)

alpha = np.arange(6, 10)
topoplot(data, alpha, 'Alpha_NonIntrusions', threshold=2.5, tail=1)

lowbeta = np.arange(10, 17)
topoplot(data, lowbeta, 'LowBeta_NonIntrusions', threshold=2.5, tail=1)

highbeta = np.arange(17, 27)
topoplot(data, highbeta, 'HighBeta_NonIntrusions', threshold=2.5, tail=1)



plt.plot(total_tfr[:, pick_rf, :6, :].mean((1, 2)).T)

pick_rf   = mne.pick_channels(tnt.ch_names, electrodes['Midline'])
data = total_tfr[:, pick_rf, 17:22, :].mean((1, 2))
data = pd.melt(pd.DataFrame(data))
sns.lineplot(data=data, x='variable', y='value')

t, p = stats.ttest_1samp(data, axis=0, popmean=0)
plt.plot(t.T)

# Import BPM
bpm = []
with open('bpm.txt', "rb") as fp:
    bpm = pickle.load(fp)

bpm = bpm[2]
bpm = bpm[pick]

# Extract averaged power over time and frequency of interest
pw = total_tfr[:, :, lowbeta, 75:100].mean((1, 2, 3))
stats.mannwhitneyu(pw[bpm < np.median(bpm)], pw[bpm >= np.median(bpm)])
lb = pd.DataFrame({'Low control': pw[bpm < np.median(bpm)], 'High control':pw[bpm >= np.median(bpm)]}).melt()
lb['Frequency'] = 'Low-beta'


# Extract averaged power over time and frequency of interest
pw = total_tfr[:, :, highbeta, 75:100].mean((1, 2, 3))
stats.mannwhitneyu(pw[bpm < np.median(bpm)], pw[bpm >= np.median(bpm)])
hb = pd.DataFrame({'Low control': pw[bpm < np.median(bpm)], 'High control':pw[bpm >= np.median(bpm)]}).melt()
hb['Frequency'] = 'High-beta'


data = pd.concat([lb, hb], ignore_index=True)

data.to_csv('EEG_Intrusions.txt')
