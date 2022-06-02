## Synchronization of ephys and imaging data + analysis of spiking
# !/usr/bin/env/python3

# correct for figure colors and text sizing in illustrator

import numpy as np
from tkinter import filedialog
import matplotlib.pyplot as plt
import scipy.io as sp
import pandas as pd
from scipy import signal

## set variables
# hardcode some variables for now

ephys_type = 1  # 0 = whole-cell, 1 = cell-attached
ref_ch = 0  # 0 = no ref ch from imaging (i.e. AF594 red), 1 = ref ch

user_input = input("ScanImage framerate (Hz) and trigger delay (s):").split(
    ',')  # user_input[0] = framerate, [1] = delay

## import MatLab .mat file
file_path = filedialog.askopenfilename()
mat_contents = sp.loadmat(file_path)

ephys = mat_contents['unnamed1']  # raw ephys data
gcp = mat_contents['unnamed']  # raw GCaMP imaging data

ax1 = plt.subplot(211)
ax1.plot(ephys[:, 0] / 1000, ephys[:, 1])  # convert time ms to s
ax1.set_title('electrophysiology')
ax1.set_xlabel('time (s)')
ax1.set_xlim(0, 100)
ax1.set_ylabel('mV')

ax2 = plt.subplot(212)
ax2.plot(gcp[:, 0] / float(user_input[0]) +
         float(user_input[1]), gcp[:, 1])  # convert frames to time via framerate input and add trigger offset
ax2.set_title('GCaMP6')
ax2.set_xlabel('time (s)')
ax2.set_xlim(0, 100)
ax2.set_ylabel('F (a.u.)')
plt.show()

## calculate power spectrum of ephys data
timestep = 1 / (0.25 / 1000)  # from digitizer; double check for errors

freq, Psden = signal.welch(ephys[:, 1], timestep, nperseg=4096)  #welch's periodogram w/ overlapping windowing
plt.plot(freq, Psden)
plt.xlim(0,100)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()

# raw fft - check for 60Hz noise
# ps = np.abs(np.fft.fft(ephys[:, 1])) ** 2
# freqs = np.fft.fftfreq(ephys[:, 1].size, timestep)
# idx = np.argsort(freqs)
# plt.plot(freqs[idx], ps[idx])
# plt.xlabel('freq (Hz)')
# plt.yscale('log')
# plt.xscale('log')
# # plt.xlim(0,int(timestep)/2)
# plt.show()

## check correlation between ephys and imaging
# **caution - requires resampling of ephys due to lower sampling rate of imaging data

## try pandas dataframe
# create pandas dataframe (df) for each set

df_ephys = pd.DataFrame(data=ephys[:, 1], index=pd.to_timedelta((ephys[:, 0]/1000), unit='s'))
df_gcp = pd.DataFrame(data=gcp[:, 1],
                      index=pd.to_timedelta(gcp[:, 0] / float(user_input[0]) + float(user_input[1]), unit='s'))

# resample to set rate (100ms default) and merge
# cannot resample below 1/framerate
df_ephys = df_ephys.resample('80ms').mean()
df_gcp = df_gcp.resample('80ms').mean()

# calculate cross correlation between ephys and imaging
corr = signal.correlate(df_ephys, df_gcp)
lag = signal.correlation_lags(len(df_gcp), len(df_ephys))
corr /= np.max(corr)

fig_crosscorr, (ax_ephys, ax_gcp, ax_corr) = plt.subplots(3, 1)
ax_ephys.plot(df_ephys)
ax_ephys.set_title('ephys')
ax_ephys.set_xlabel('time (ns)')
ax_gcp.plot(df_gcp)
ax_gcp.set_title('gcamp')
ax_gcp.set_xlabel('time (ns)')
ax_gcp.get_shared_x_axes().join(ax_gcp, ax_ephys)
ax_corr.plot(lag, corr)
ax_corr.set_title('Cross-correlated signal')
ax_corr.set_xlabel('Lag')
ax_ephys.margins(0, 0.1)
ax_gcp.margins(0, 0.1)
ax_corr.margins(0, 0.1)
fig_crosscorr.tight_layout()
plt.show()
