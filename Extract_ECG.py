import scipy.io
from scipy.ndimage import gaussian_filter1d
import numpy as np
import pandas as pd
import os
from statistics import median

group           = '1'
path            = '/media/legrand/DATAPART1/ENGRAMME/GROUPE_' + group
Session_num     = '1'
Names           = os.listdir(path + '/COMPORTEMENT/')

TOT, TOT_Outliers, TOT_Outliers_DF = [], [], []
output_df = pd.DataFrame([])

def normalized_MAD(mat, k = 2.2414):
    """"Return bpm mean with outliers detection based on the
    Normalized Median Absolute Deviation"""

    X = np.nanmean(mat, 0) # Mean BPM along the 10 seconds of presentation
    M = median(X)
    MAD = median(abs(X - M))
    MADN = MAD*1.4826
    rejected = (abs(X-M))/MADN > k
    mean_bpm = np.nanmean(mat[:,~rejected],1)
    return rejected, mean_bpm

for i_sub, nsub in enumerate(Names):

    # Load data frame
    Eval_Emo = pd.read_table(path + '/COMPORTEMENT/' + nsub + '/TNT/Eval_Emo_' + Session_num + '.txt')
    Eval_Emo.rename(columns=lambda x: x.strip(), inplace=True) # Remove spaces in cols names

    # Load BIOPAC file
    if group == '1':
        x = scipy.io.loadmat(path + '/EMO_' + Session_num + '/' + nsub + '_EMO' + Session_num + '.mat')
        if len(x['data'][1,:]) == 6: # ECG artifacts were manually corrected
            stim_vec = x['data'][:,4]
            bpm_vec  = x['data'][:,3]
        else:
            stim_vec = x['data'][:,3]
            bpm_vec  = x['data'][:,4]
    else:
        x = scipy.io.loadmat(path + '/ECG/' + nsub + '/' + nsub + '_EMO' + Session_num + '.mat')
        if len(x['data'][1,:]) == 7: # ECG artifacts were manually corrected
            stim_vec = x['data'][:,5]
            bpm_vec  = x['data'][:,4]
        else:
            stim_vec = x['data'][:,4]
            bpm_vec  = x['data'][:,5]

    # %% Set 1 at each stim presentation, else = 0
    for i in range(len(stim_vec)):
        if stim_vec[i] != 0:
            stim_vec[i] = 1
            stim_vec[i+1 : i+400] = 0

    stim_sample = np.where(stim_vec != 0)

    # %%
    if (group == '2') & ((nsub == '31NLI') & (Session_num == '1')): # Specific case
        stim_sample = stim_sample[0][5:]
    else:
        stim_sample = stim_sample[0][2:]

    # %% Find Stim presentation in df
    Eval_Emo['Pres_Stim'] = Eval_Emo['TR']
    # Add stim timing to df
    if group == '1':
        for i in range(len(Eval_Emo)):
            Eval_Emo.iloc[i, 8] = stim_sample[0]  # Assign a sample start in Eval table
            # Move to the next onset
            if (i+1) % 12 == 0:
                if Eval_Emo['TR'][i] == ' NaN':
                    if nsub == '06DJO' and Session_num == '2':
                        stim_sample = stim_sample[3:]
                    else:
                        stim_sample = stim_sample[4:]
                else:
                    if nsub == '06DJO' and Session_num == '2':
                        stim_sample = stim_sample[4:]
                    else:
                        stim_sample = stim_sample[5:]
            else:
                if Eval_Emo['TR'][i] == ' NaN':
                        stim_sample = stim_sample[1:]
                else:
                    if nsub == '06DJO' and Session_num == '2':
                        stim_sample = stim_sample[1:]
                    else:
                        stim_sample = stim_sample[2:]
    else:
        for i in range(len(Eval_Emo)):
            Eval_Emo.iloc[i, 7] = stim_sample[0]  # Assign a sample start in Eval table
            # Move to the next onset
            if (i+1) % 12 == 0:
                if Eval_Emo['TR'][i] == ' NaN':
                    stim_sample = stim_sample[3:]
                else:
                    stim_sample = stim_sample[4:]
            else:
                if Eval_Emo['TR'][i] == ' NaN':
                    stim_sample = stim_sample[1:]
                else:
                    stim_sample = stim_sample[2:]

    # Check correct matching
    if len(stim_sample)>0:
        print('Error in general stim detection')

    # %% Check coherent timing (Biopac - Psychtoolbox)
    for i in range(len(Eval_Emo)):
        diff = ((Eval_Emo['Pres_Stim'][i]/2000)-(Eval_Emo['Pres_Stim'][0]/2000)) - (Eval_Emo['Start Time'][i] - Eval_Emo['Start Time'][0]);
        if  abs(diff) >1:
            print('Inconsistent stim detection for ' + str(nsub) + ' at stim ' + str(i))

    # %% Remove artifacts from data
    if group == '1':
        if Session_num == '1':
             if nsub == '04PLA':
                 Eval_Emo = Eval_Emo.drop(Eval_Emo.index[[37,38,49]])
        else:
            if nsub == '21MVA':
                Eval_Emo = Eval_Emo.drop(Eval_Emo.index[[44,54,55,56]])
            if nsub == '23RGR':
                Eval_Emo = Eval_Emo.drop(Eval_Emo.index[[69, 70, 71]])
            if nsub == '30OCE':
                Eval_Emo = Eval_Emo.drop(Eval_Emo.index[18])

    # %% Remove forgotten items from criterion
    if Session_num == 2:
        try:
            Criterion   = pd.read_table(path + '/COMPORTEMENT/' + nsub + '/TNT/criterion.txt', encoding='latin1')
            forgotten   = [os.path.basename(i) for i in Criterion[' Image'][Criterion[' FinalRecall'] == 0]]
            if len(forgotten) > 0:
                Eval_Emo    = Eval_Emo[~Eval_Emo['img'].str.contains('|'.join(forgotten))]
        except ValueError:
            print(str(nsub) + " Criterion.txt not found")

    # %% Extract stimuli induced HR
    if group == '1':
        CodeEmo = ['Neu', 'disgust', 'sadness']
    else:
        CodeEmo  = ['Neutral', 'Emotion']
    CodeCond = [' Think', ' No-Think', ' Baseline']
    Res = np.ndarray(shape = (20000, len(CodeEmo) * len(CodeCond)))
    Res[:] = np.NAN
    Outliers, Outliers_df = [], []
    for i_Emo, Emo in enumerate(CodeEmo):
        for i_Cond, Cond in enumerate(CodeCond):

            # Filter emotions
            if group == '1':
                Ev = Eval_Emo[Eval_Emo['img'].str.contains(Emo)]
            else:
                Ev = Eval_Emo[Eval_Emo.nLoop == Emo]

            # Filter conditions
            Ev = Ev[Ev.Condition == Cond]
            Ev.reset_index(inplace = True)

            # Store in array
            mat = np.ndarray(shape = (20000, 12))
            mat[:] = np.NAN

            for i in Ev.index:

                # Extract 0-10s after stim presentation
                HRtmpt = bpm_vec[int(Ev.Pres_Stim[i]):int(Ev.Pres_Stim[i] + 20000)]

                # Check for artifacts using arbitrary thresholds
                if any(HRtmpt<40) or any(HRtmpt>120):
                    print('Artifacts in ' + nsub + ' ' + Emo)
                    Ev.drop(i, inplace=True)
                else:
                    HRtmpt = gaussian_filter1d(HRtmpt, 2 ** 10) # Smooth
                    HRtmpt = HRtmpt-HRtmpt[0]                   # Normalize at T(0)
                    mat[:,i] = HRtmpt                           # Store in array

            # Remove NaNs from matrice
            mat = mat[:,~np.any(np.isnan(mat), axis=0)]

            # Store BPM
            rejected, mean_bpm = normalized_MAD(mat) # Filter outliers with MAD rule
            Res[:, (i_Emo * 3) + i_Cond] = mean_bpm

            # Store outliers
            Outliers.append(list([mat[:, rejected], mat[:, ~rejected]]))
            out_df = Ev[rejected]
            Outliers_df.append(out_df)

            # Store emotional evaluation and response time in a dataframe
            val = pd.to_numeric(Ev.Valence, errors='coerce')
            val = val[(~rejected) & (~pd.isnull(val))].mean()
            res = pd.to_numeric(Ev.TR, errors='coerce')
            res = res[(~rejected) & (~pd.isnull(res))].mean()
            output_df = output_df.append(pd.DataFrame({'Subject': nsub,
                                                        'Condition': Cond[1:],
                                                        'Emotion': Emo,
                                                        'Valence':val,
                                                        'TR': res}, index=[0]), ignore_index=True)
    TOT.append(Res) # BPM
    TOT_Outliers.append(Outliers) # Arrays of droped BPM
    TOT_Outliers_DF.append(Outliers_df) # Df of droped items
    print('Subject ' + str(i_sub) + ' done')

# %% Save results
import pickle

with open("Group" + group + "_Sess" + Session_num + ".txt", "wb") as fp:   # Raw ECG data
   pickle.dump(TOT, fp)

with open("Group" + group + "_Sess" + Session_num + "_Outliers.txt", "wb") as fp:   # Outliers
   pickle.dump(TOT_Outliers, fp)

with open("Group" + group + "_Sess" + Session_num + "_Outliers_df.txt", "wb") as fp:   # Outliers
   pickle.dump(TOT_Outliers_DF, fp)

with open("Group" + group + "_Sess" + Session_num + "_Eval_Emo.txt", "wb") as fp:   # Data frame
   pickle.dump(output_df, fp)
