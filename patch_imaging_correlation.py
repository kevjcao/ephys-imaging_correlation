## Synchronization of ephys and imaging data + analysis of spiking
# !/usr/bin/envpython3

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import scipy.io as sp

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
# raw power spectrum calculation **
ps = np.abs(np.fft.fft(ephys[:,1]))**2
timestep = 1/(0.25/1000)            # from digitizer
freqs = np.fft.fftfreq(ephys[:,1].size, timestep)
idx = np.argsort(freqs)

plt.plot(freqs[idx], ps[idx])
plt.xlabel('freq (Hz)')
plt.yscale('log')
plt.xscale('log')
# plt.xlim(0,int(timestep)/2)
plt.show()


