## Synchronization of ephys and imaging data + analysis of spiking
# !/usr/bin/env/python3

# correct for figure colors and text sizing in illustrator

import numpy as np
from tkinter import filedialog
import matplotlib.pyplot as plt
import scipy.io as sp
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks


## set variables
# hardcode some variables for now

ephys_type = 1  # 0 = whole-cell, 1 = cell-attached
ref_ch = 0  # 0 = no ref ch from imaging (i.e. AF594 red), 1 = ref ch

user_input = input("ScanImage framerate (Hz) and trigger delay (s):").split(
    ',')  # user_input[0] = framerate, [1] = delay


## define functions
# finds and annotates max peak
# borrowed from stackoverflow.com/questions/43374920/how-to-automatically-annotate-maximum-value-in-pyplot
def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)


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
timestep = 1 / (0.25 / 1000)  # sampling frequency (Hz), from digitizer; double check for errors

freq, Psden = signal.welch(ephys[:, 1], timestep, nperseg=4096)  #welch's periodogram w/ overlapping windowing
plt.plot(freq, Psden)
plt.xlim(0,200)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()

## identify max spike frequency in filtered data
x = freq[1:60]
y = Psden[1:60]        # only frequencies under 60Hz for max calculation to avoid 60 cycle noise
fig, ax = plt.subplots()
ax.plot(x,y)
annot_max(x,y)
#ax.set_ylim(0,0.5)
plt.show()

## detect and filter 60Hz noise
# Create/view notch filter
notch_freq = 60.0  # Frequency to be removed from signal (Hz)
quality_factor = 30.0  # Quality factor
b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, timestep)
freq, h = signal.freqz(b_notch, a_notch, fs = timestep)
plt.figure('filter')
plt.plot( freq, 20*np.log10(abs(h)))
plt.show()

# apply notch filter to signal
y_notched = signal.filtfilt(b_notch, a_notch, ephys[:,1])   # apply 60Hz bandpass on spike data
filtfreq, filtPsden = signal.welch(y_notched, timestep, nperseg=4096)   # run PSD on filtered data

# plot notch-filtered version of signal
ax1 = plt.subplot(211)
ax1.plot(ephys[:,0] / 1000, y_notched)
ax1.set_xlabel('time (s)')
ax1.set_xlim(0, 30)
ax1.set_ylabel('mV')
ax2 = plt.subplot(223)
ax2.plot(filtfreq, filtPsden)
ax2.set_xlabel('freq (Hz)')
ax2.set_xlim(0, 80)
ax2.set_ylabel('PSD [V**2/Hz]')
ax3 = plt.subplot(224)
ax3.plot(filtfreq, filtPsden)
ax3.set_xlabel('freq (Hz)')
ax3.set_xlim(0, 20)
ax3.set_ylabel('PSD [V**2/Hz]')
plt.show()

## identify max spike frequency in filtered data
# this is a redundant measure but may mark other oscillatory signals
x = filtfreq[1:100]
y = filtPsden[1:100]        # only frequencies under 100Hz for max calculation

fig, ax = plt.subplots()
ax.plot(x,y)
annot_max(x,y)
#ax.set_ylim(0,0.5)
plt.show()


## check correlation between ephys and imaging
# **caution - requires resampling of ephys due to lower sampling rate of imaging data
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
