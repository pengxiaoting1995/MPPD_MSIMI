import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import medfilt
import pywt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def normalized(origin_signal):
  y = origin_signal - np.mean(origin_signal)
  y = y / np.max(np.abs(y))
  return y


def mydenoise(data,fs):
    normalized_data = normalized(data)
    t1 = int(0.2 * fs)
    t2 = int(0.6 * fs)
    signal_1 = medfilt(normalized_data, t1 + 1)
    signal_2 = medfilt(signal_1, t2 + 1)
    denoise_data = normalized_data - signal_2
    return denoise_data

PATH= 'path includes raw ECG data'; # '/sourcedata/sub_001/baseline-data/sub_001_baseline-data_rest-ecg-100hz.csv'
OutPutFilePath = 'output path to save ECG data after denoising';
OutPcitureFilePath = 'output pictures for vildation';
# loading data
data_df = pd.read_csv(PATH,header=None)
# denoise
denoise_data = mydenoise(data_df.values,fs=500) # fs=500 or fs=100
# visualization
plt.figure(figsize=(20, 8))
plt.subplot(211)
plt.plot(data_df.values)
plt.title("oiginal")
plt.subplot(212)
plt.plot(denoise_data)
plt.title("denoise")
plt.show()
plt.savefig(OutPcitureFilePath)