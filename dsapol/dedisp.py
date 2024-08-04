import numpy as np
import os


"""
Functions for brute force dedispersion
"""
import json
f = open(os.environ['DSAPOLDIR'] + "directories.json","r")
dirs = json.load(f)
f.close()
logfile = dirs["logs"] + "dedisp_logfile.txt"


def get_min_DM_step(n_t,fminGHz=1.307,fmaxGHz=1.493,res=32.7e-3):
    return np.around((res)*n_t/(4.15)/((1/fminGHz**2) - (1/fmaxGHz**2)),2)

def dedisperse(dyn_spec,DM,tsamp,freq_axis):
    """
    This function dedisperses a dynamic spectrum of shape nsamps x nchans by brute force without accounting for edge effects
    """
    f = open(logfile,"w")


    #get delay axis
    tdelays = -DM*4.15*(((np.min(freq_axis)*1e-3)**(-2)) - ((freq_axis*1e-3)**(-2)))#(8.3*(chanbw)*burst_DMs[i]/((freq_axis*1e-3)**3))*(1e-3) #ms
    tdelays_idx_hi = np.array(np.ceil(tdelays/tsamp),dtype=int)
    tdelays_idx_low = np.array(np.floor(tdelays/tsamp),dtype=int)
    tdelays_frac = tdelays/tsamp - tdelays_idx_low
    print("Dedispersing to DM: " + str(DM) + " pc/cc...",end='',file=f)#, DM delays (ms): " + str(tdelays) + "...",end='')
    nchans = len(freq_axis)
    #print(dyn_spec.shape)
    dyn_spec_DM = np.zeros(dyn_spec.shape)
    #print(tdelays_idx_hi)
    #print(tdelays_idx_low)
    #print(DM)

    #shift each channel
    for k in range(nchans):
        if tdelays_idx_low[k] >= 0:
            arrlow =  np.pad(dyn_spec[k,:],((0,tdelays_idx_low[k])),mode="constant",constant_values=0)[tdelays_idx_low[k]:]/nchans
        else:
            arrlow =  np.pad(dyn_spec[k,:],((np.abs(tdelays_idx_low[k]),0)),mode="constant",constant_values=0)[:tdelays_idx_low[k]]/nchans

        if tdelays_idx_hi[k] >= 0:
            arrhi =  np.pad(dyn_spec[k,:],((0,tdelays_idx_hi[k])),mode="constant",constant_values=0)[tdelays_idx_hi[k]:]/nchans
        else:
            arrhi =  np.pad(dyn_spec[k,:],((np.abs(tdelays_idx_hi[k]),0)),mode="constant",constant_values=0)[:tdelays_idx_hi[k]]/nchans

        dyn_spec_DM[k,:] = arrlow*(1-tdelays_frac[k]) + arrhi*(tdelays_frac[k])
    print("Done!",file=f)
    f.close()

    return dyn_spec_DM
