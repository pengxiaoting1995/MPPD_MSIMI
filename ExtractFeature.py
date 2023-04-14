import itertools
import os
import math
import random
from itertools import groupby
import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def judge_R_num(data, fs, r_num):
    # 0.6s~1s
    r_num_max = len(data) / fs / 0.5
    r_num_min = len(data) / fs / 1.2
    if r_num < r_num_max and r_num > r_num_min:
        results = True
    else:
        results = False
    return results


def interval_rand(a, b):
    return a + (b - a) * np.random.random()


def detectR(denoise_data, fs):
    _, rpeaks = nk.ecg_peaks(denoise_data, sampling_rate=fs)
    RR = [rpeaks['ECG_R_Peaks'][i + 1] - rpeaks['ECG_R_Peaks'][i] for i in range(len(rpeaks['ECG_R_Peaks']) - 1)]
    down = 120;
    up = 1800
    r_exist = all(_ <= up for _ in RR) & all(_ >= down for _ in RR)
    counter = 1
    while r_exist == False:
        ECG_R_Peaks = find_peaks(x=denoise_data, threshold=None, distance=interval_rand(200, 600), prominence=None,
                                 width=None, wlen=None, rel_height=None, plateau_size=None)
        rpeaks = {'ECG_R_Peaks': ECG_R_Peaks, 'sampling_rate': fs}
        if counter >= 100000:
            ECG_R_Peaks = find_peaks(x=denoise_data, threshold=None, distance=300, prominence=None, width=None,
                                     wlen=None, rel_height=None, plateau_size=None)
            rpeaks = {'ECG_R_Peaks': ECG_R_Peaks, 'sampling_rate': fs}
            break
        counter = counter + 1
    return rpeaks

    r_reasonable = judge_R_num(data=ecg_signal, fs=fs, r_num=len(rpeaks['ECG_R_Peaks']))
    while r_reasonable == False:
        ECG_R_Peaks = find_peaks(x=ecg_signal, threshold=None, distance=interval_rand(200, 600), prominence=None,
                                 width=None, wlen=None, rel_height=None, plateau_size=None)
        rpeaks = {'ECG_R_Peaks': ECG_R_Peaks, 'sampling_rate': fs}
        r_reasonable = judge_R_num(data=ecg_signal, fs=fs, r_num=len(rpeaks['ECG_R_Peaks']))
    return rpeaks


def morphology_feature(denoise_data, rpeaks):
    ECG_R_Peaks = rpeaks['ECG_R_Peaks']
    interval = 1. / rpeaks['sampling_rate']
    signal_dwt, waves_dwt = nk.ecg_delineate(denoise_data, rpeaks, sampling_rate=rpeaks['sampling_rate'], method="dwt",
                                             show=False, show_type='all')

    length_wave = len(waves_dwt['ECG_P_Onsets']) - 2

    ECG_P_Onsets = waves_dwt['ECG_P_Onsets']
    ECG_P_Peaks = waves_dwt['ECG_P_Peaks']
    ECG_P_Offsets = waves_dwt['ECG_P_Offsets']
    ECG_R_Onsets = waves_dwt['ECG_R_Onsets']
    ECG_Q_Peaks = waves_dwt['ECG_Q_Peaks']
    ECG_S_Peaks = waves_dwt['ECG_S_Peaks']
    ECG_R_Offsets = waves_dwt['ECG_R_Offsets']
    ECG_T_Onsets = waves_dwt['ECG_T_Onsets']
    ECG_T_Peaks = waves_dwt['ECG_T_Peaks']
    ECG_T_Offsets = waves_dwt['ECG_T_Offsets']
    length_peaks = [len(ECG_P_Onsets), len(ECG_P_Peaks), len(ECG_P_Offsets), len(ECG_R_Onsets), len(ECG_Q_Peaks),
                    len(ECG_S_Peaks), len(ECG_R_Offsets), len(ECG_T_Onsets), len(ECG_T_Peaks), len(ECG_T_Offsets)]
    while len(set(length_peaks)) != 1:
        Num_list = list(map(abs, (np.array(length_peaks) - max(length_peaks)).tolist()))
        ECG_P_Onsets = (np.full([1, Num_list[0]], np.nan)).flatten().tolist() + waves_dwt['ECG_P_Onsets']
        ECG_P_Peaks = (np.full([1, Num_list[1]], np.nan)).flatten().tolist() + waves_dwt['ECG_P_Peaks']
        ECG_P_Offsets = (np.full([1, Num_list[2]], np.nan)).flatten().tolist() + waves_dwt['ECG_P_Offsets']
        ECG_R_Onsets = (np.full([1, Num_list[3]], np.nan)).flatten().tolist() + waves_dwt['ECG_R_Onsets']
        ECG_Q_Peaks = (np.full([1, Num_list[4]], np.nan)).flatten().tolist() + waves_dwt['ECG_Q_Peaks']
        ECG_S_Peaks = (np.full([1, Num_list[5]], np.nan)).flatten().tolist() + waves_dwt['ECG_S_Peaks']
        ECG_R_Offsets = (np.full([1, Num_list[6]], np.nan)).flatten().tolist() + waves_dwt['ECG_R_Offsets']
        ECG_T_Onsets = (np.full([1, Num_list[7]], np.nan)).flatten().tolist() + waves_dwt['ECG_T_Onsets']
        ECG_T_Peaks = (np.full([1, Num_list[8]], np.nan)).flatten().tolist() + waves_dwt['ECG_T_Peaks']
        ECG_T_Offsets = (np.full([1, Num_list[9]], np.nan)).flatten().tolist() + waves_dwt['ECG_T_Offsets']
        length_peaks = [len(ECG_P_Onsets), len(ECG_P_Peaks), len(ECG_P_Offsets), len(ECG_R_Onsets),
                        len(ECG_Q_Peaks), len(ECG_S_Peaks), len(ECG_R_Offsets), len(ECG_T_Onsets),
                        len(ECG_T_Peaks), len(ECG_T_Offsets)]
    jump = 0
    morphology_features_list = []
    for i in range(length_wave):
        if math.isnan(float(ECG_P_Onsets[i + 1])) or math.isnan(float(ECG_P_Offsets[i + 1])) or math.isnan(
                float(ECG_T_Onsets[i + 1])) or math.isnan(float(ECG_T_Offsets[i + 1])) or math.isnan(
            float(ECG_R_Onsets[i + 1])) or math.isnan(ECG_P_Onsets[i + 2]) or math.isnan(
            float(ECG_R_Offsets[i + 1])) or math.isnan(float(ECG_P_Peaks[i + 1])) or math.isnan(
            float(ECG_T_Peaks[i + 1])) or math.isnan(float(ECG_Q_Peaks[i + 1])) or math.isnan(
            float(ECG_S_Peaks[i + 1])):
            jump += 1
            continue
        else:
            Pduration = (ECG_P_Offsets[i + 1] - ECG_P_Onsets[i + 1]) * interval
            Tduration = (ECG_T_Offsets[i + 1] - ECG_T_Onsets[i + 1]) * interval
            QRSduration = (ECG_R_Offsets[i + 1] - ECG_R_Onsets[i + 1]) * interval
            PRinterval = (ECG_R_Onsets[i + 1] - ECG_P_Onsets[i + 1]) * interval
            PRsegment = np.absolute(ECG_R_Onsets[i + 1] - ECG_P_Offsets[i + 1]) * interval
            STinterval = (ECG_T_Offsets[i + 1] - ECG_R_Offsets[i + 1]) * interval
            STsegment = np.absolute(ECG_T_Onsets[i + 1] - ECG_R_Offsets[i + 1]) * interval
            TPinterval = (ECG_P_Onsets[i + 2] - ECG_T_Offsets[i + 1]) * interval
            QTinterval = (ECG_T_Offsets[i + 1] - ECG_R_Onsets[i + 1]) * interval
            Pamp = denoise_data[ECG_P_Peaks[i + 1]]
            Tamp = denoise_data[ECG_T_Peaks[i + 1]]
            Qamp = denoise_data[ECG_Q_Peaks[i + 1]]
            Ramp = denoise_data[ECG_R_Peaks[i + 1]]
            Samp = denoise_data[ECG_S_Peaks[i + 1]]
            morphology_pro = [Pduration, Tduration, PRinterval, PRsegment, QRSduration, STinterval, STsegment,
                              TPinterval, QTinterval, Pamp, Qamp, Ramp, Samp, Tamp]
            morphology_features_list.append(morphology_pro)
    morphology_features = np.array(morphology_features_list)
    morphology_list = pd.DataFrame(
        {'Pduration': [np.mean(morphology_features[:, 0])],
         'Tduration': [np.mean(morphology_features[:, 1])],
         'PRinterval': [np.mean(morphology_features[:, 2])],
         'PRsegment': [np.mean(morphology_features[:, 3])],
         'QRSduration': [np.mean(morphology_features[:, 4])],
         'STinterval': [np.mean(morphology_features[:, 5])],
         'STsegment': [np.mean(morphology_features[:, 6])],
         'TPinterval': [np.mean(morphology_features[:, 7])],
         'QTinterval': [np.mean(morphology_features[:, 8])],
         'Pamp': [np.mean(morphology_features[:, 9])],
         'Qamp': [np.mean(morphology_features[:, 10])],
         'Ramp': [np.mean(morphology_features[:, 11])],
         'Samp': [np.mean(morphology_features[:, 12])],
         'Tamp': [np.mean(morphology_features[:, 13])]})
    return morphology_list


def hrv_feature(rpeaks, fs):
    hrv_list = nk.hrv(rpeaks, sampling_rate=fs, show=False)
    return hrv_list
