# Does the heart forget? Modulation of cardiac activity induced by inhibitory control over emotional memories.

Legrand, N., Etard, O., Vandevelde, A., Pierre, M., Viader, F., Clochon, P., Doidy, F., Peschanski, D., Eustache, F. & Gagnepain, P. (2018). Preprint version 3.0, doi: https://www.biorxiv.org/content/10.1101/376954v3

# Abstract

*Effort to suppress past experiences from conscious awareness can lead to forgetting. It remains largely unknown whether emotions, including their physiological causes, are also impacted by such memory suppression. In two studies, we measured in healthy participants the aftereffect of suppressing negative memories on cardiac response. Results of Study 1 revealed that an efficient control of memories was associated with a long-term inhibition of the cardiac deceleration normally induced by disgusting stimuli. Attempts to suppress sad memories, on the opposite, aggravated cardiac response, an effect that was largely related to the inability to forget this specific material. In Study 2, we found using electroencephalography that a prominent neural marker of inhibitory control, a suppression of the 5-9 Hz frequency band, was related to the subsequent inhibition of the cardiac response. These results demonstrate that suppressing memories also influence the cardiac system, opening new avenues for treating intrusive memories.*

This repository contains data, scripts and Jupyter notebook accessible through Binder detailing analysis from the preprint version of the paper. The currents scripts have not been peer-reviewed, so we do not recommend to use them on your own data for now. If you judge that some codes would benefit from specific clarifications or improvements do not hesitate to contact us (legrand@cyceron.fr).

# Data

Behavioral data from Study 1 (n=28) and Study 2 (n=24) are provided in `data/Emotion.csv`, `data/Recall.csv` and `data/Intrusions.csv`. Preprocessed ECG are provided in `ECG*.txt` files.

# Notebooks

Figures and statistical models can be reproduced via two Jupyter Notebook (`Behavioral.ipynb` and `ECG.ipynb`).

# Scripts

`EEG.py` implement preprocessing, `EEG_statistics.py` implement statistics and plotting.

## EEG.py

Preprocesses:

* run_filter(): read raw data, exclude unused channels, interpolate bad ones and filter.

* run_epochs(): epoch raw data according to the experimental design, reject epochs with V > 350e-6.

* run_autoreject(): apply [Autoreject](https://autoreject.github.io/) to the epoched data.

* run_ICA(): reduce blinks with ICA based on the selected EOG channel.

* run_ERF(): extract TFR between 5 and 9Hz using Morlet wavelets.

## EEG_statistics.py

* store_array(): permutation cluster statistics.

* plottopo(): Plot topomap of the contrasted two conditions base on the selected cluster.

* plot_ecg_corr(): Plot the correlation between the averaged cardiac modulation (provided in `bpm.txt`) and the decrease of frequency power.

