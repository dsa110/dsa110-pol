import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import correlate
from scipy.signal import savgol_filter as sf
from scipy.signal import convolve
from scipy.signal import fftconvolve
from scipy.ndimage import convolve1d
from scipy.signal import peak_widths
from scipy.stats import chi
from scipy.stats import norm


"""
This file contains helpers for filtering and computing signal-to-noise
"""
import json
f = open("directories.json","r")
dirs = json.load(f)
f.close()
logfile = dirs["logs"] + "filter_logfile.txt"

def get_SNR(I_tcal,I_w_t_filtcal,timestart,timestop):
    """
    Takes a time series and padded filter weights, and limits and computes the matched filter signal-to-noise
    """

    I_w_t_filtcal_unpadded=I_w_t_filtcal[timestart:timestop]
    I_trial_binned = (convolve(I_tcal,I_w_t_filtcal_unpadded))
    sigbin = np.argmax(I_trial_binned)
    sig0 = I_trial_binned[sigbin]
    I_binned = (convolve(I_tcal,I_w_t_filtcal_unpadded))
    noise = np.std(np.concatenate([I_binned[:sigbin-(timestop-timestart)*2],I_binned[sigbin+(timestop-timestart)*2:]]))
    return sig0/noise


