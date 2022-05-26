## Synchronization of ephys and imaging data + analysis of spiking
# !/usr/bin/envpython3

# correct for figure colors and text sizing in illustrator

import numpy as np
from tkinter import filedialog
import matplotlib.pyplot as plt
import scipy.io as sp
import pandas as pd
from scipy.stats.stats import pearsonr

## set variables
# hardcode some variables for now

ephys_type = 1        # 0 = whole-cell, 1 = cell-attached
ref_ch = 0;           # 0 = no ref ch from imaging (i.e. AF594 red), 1 = ref ch

user_input = input("ScanImage framerate (Hz) and trigger delay (s):").split(',')


## import MatLab .mat file
file_path = filedialog.askopenfilename()
mat_contents = sp.loadmat(file_path)

ephys = mat_contents['unnamed1']    # raw ephys data
gcp = mat_contents['unnamed']       # raw GCaMP imaging data

ax1 = plt.subplot(211)
ax1.plot(ephys[:,0]/1000, ephys[:,1])   # convert time ms to s
ax1.set_title('electrophysiology')
ax1.set_xlabel('time (s)')
ax1.set_xlim(0, 100)
ax1.set_ylabel('mV')

ax2 = plt.subplot(212)
ax2.plot(gcp[:,0]/float(user_input[0]) +
         float(user_input[1]), gcp[:,1])    # convert frames to time via framerate input and add trigger offset
ax2.set_title('GCaMP6')
ax2.set_xlabel('time (s)')
ax2.set_xlim(0, 100)
ax2.set_ylabel('F (a.u.)')
plt.show()

## calculate power spectrum of data
# raw power spectrum calculation - check for 60Hz noise
ps = np.abs(np.fft.fft(ephys[:,1]))**2
timestep = 1/(0.25/1000)            # from digitizer; double check for errors
freqs = np.fft.fftfreq(ephys[:,1].size, timestep)
idx = np.argsort(freqs)

plt.plot(freqs[idx], ps[idx])
plt.xlabel('freq (Hz)')
plt.yscale('log')
plt.xscale('log')
# plt.xlim(0,int(timestep)/2)
plt.show()

## check correlation between ephys and imaging
# ** caution - requires interpolation or resampling due to lower sampling of imaging data

## try pandas dataframe
# create pandas dataframe for each set
df_ephys = pd.DataFrame(data = ephys[:,1],
                        index = pd.to_datetime((ephys[:,0]/1000), unit = 's'))
df_gcp = pd.DataFrame(data = gcp[:,1],
                      index = pd.to_datetime(gcp[:,0]/float(user_input[0])
                                             + float(user_input[1]), unit = 's'))
# resample to set rate (100ms default) and merge
df_ephys = df_ephys.resample('100ms').mean()
df_gcp = df_gcp.resample('100ms').mean()
df_corr = pd.concat((df_ephys, df_gcp), axis = 1)
df_corr.columns = ['ephys', 'gcamp']

df_corr.plot(x='ephys', y='gcamp')
plt.show()

## try interp in numpy
# t = np.linspace(0, 100, 1)
# ephys_interp = np.interp(t, 0.25/1000, ephys[:,1])
# gcp_interp = np.interp(t, float(user_input[0]), gcp[:,1])
#
# smooth_sigma = 10   # play around with this to avoid over rectifying signals -- careful
# ephys_interp, gcp_interp = abs(ephys_interp), abs(gcp_interp)
# ephys_interp, gcp_interp = gaussian_filter(ephys_interp, smooth_sigma), guassian_filter(gcp_interp, smooth_sigma)
#
# max_xc = 0
# best_shift = 0
# for shift in range(-10, 10):    # adjust search range
#     xc = (np.roll(ephys_interp, shift) * gcp_interp).sum()
#     if xc > max_xc:
#         max_xc = xc
#         best_shift = shift
#
# print('Best shift:', best_shift)

## try rolling correction
# all_series = {'patch': ephys[:,1], 'image': gcp[:,1]}
# series_list = [i for i in all_series.keys()]
#
# def max_correlation(patch, image):
#     if len(image)>=len(patch):
#         patch_old = patch
#         patch = image
#         image = patch_old
#     df = pd.DataFrame(dict(x=patch))
#     corr_vals = np.array(image)
#     def get_correlation(vals):
#         return pearsonr(vals, corr_vals)[0]
#     return df.rolling(window=len(corr_vals)).apply(get_correlation).max().values[0]
#
# corr_matrix = pd.DataFrame(index=series_list, columns=series_list)
# for i in series_list:
#     for j in series_list:
#         corr_matrix.loc[i,j]=max_correlation(all_series[i], all_series[j])
#
# print(corr_matrix)








