# Does the heart forget? Modulation of cardiac activity induced by inhibitory control over emotional memories.

Legrand, N., Etard, O., Vandevelde, A., Pierre, M., Viader, F., Clochon, P., Doidy, F., Peschanski, D., Eustache, F. & Gagnepain, P. (2018). Preprint version 1.0, doi: https://doi.org/10.1101/376954

This repository contain data, scripts and Jupyter notebook accessibles through Binder detailling analysis from the preprint version of the paper. The currents scripts has not been peer-reviewed, so we do not recommand to use them on your own data for now. If you judge that some codes would benefit from specific clarifications or improvements do not hesitate to contact us (legrand@cyceron.fr).

# Abstract

*The subjective construction surrounding the perception of negative experience is partly build upon bodily afferent information, comprising heart, gut or respiratory signals. While this bottom-up influence has been extensively described, the opposite pathway, the putative influence of cognitive processes over autonomic response, is still debatable. However, emotion regulation and the ability to control maladjusted physiological response associated with thoughts and memories is a recurrent concern for most psychiatric disorders and mental health as a whole. Direct suppression (i.e. exerting inhibitory control toward unwanted memories) has been proposed as a possible solution to perform such regulation. But this method also holds debates as it could putatively worsen the negative symptoms when unsuccessful, and may be ineffective on the physiological roots of emotions. Here, we tested the hypothesis that direct suppression can influence physiological markers of emotions in two studies by training healthy participants to control and suppress the retrieval of distressing memories using the “Think/No-Think” paradigm. We measured their cardiac reaction toward the manipulated memories before and after the experimental procedure. Our results revealed that an efficient control of memories was associated with a long-term suppression of the cardiac deceleration normally induced by disgusting stimuli. In the second study, this difference was paralleled by an increase of subjective valence and correlated with the decrease of the 5-9 Hz frequency band during the suppression trials, indicating a reduced memory reactivation. These results support the notion that cognitive control over unwanted emotional memories can influence autonomic processes to achieve emotional regulation, and open avenues for possible markers and cognitive therapeutics to reduce the impact of distressing intrusive memories on mental health.*

# Data

Behavioral data from Study 1 (n=28) and Study 2 (n=24) are provided in `data/Emotion.csv`, `data/Recall.csv` and `data/Intrusions.csv`. Preprocessed ECG are privided in `ECG*.txt` files.

# Notebooks

Figures and statistical models can be reproduced via two Jupyter Notebook (`Behavioral.ipynb` and `ECG.ipynb`).

# Scripts

`EEG.py` implement preprocessing, `EEG_statistics.py` implement statistics and plotting.

## EEG.py

Preprocesses are divided into 5 steps:

* run_filter(): read raw data, exclude unused channles, interpolate bad ones and filter.

* run_epochs(): epoch raw data according to the experimental design, reject epochs with V > 350e-6.

* run_autoreject(): apply [Autoreject](https://autoreject.github.io/) to the epoched data.

* run_ICA(): reduce blinks with ICA based on the selected EOG channel.

* run_ERF(): extract TFR between 5 and 9Hz using Morlet wavelets.

## EEG_statistics.py

* store_array(): permutation cluster statistics.

* plottopo(): Plot topomap of the contrasted two conditions base on the selected cluster.

* plot_ecg_corr() : Plot the correlation between the averaged cardiac modulation (provided in `bpm.txt`) and the decrease of frequency power.

