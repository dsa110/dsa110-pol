"""
DSA110-pol dsapol library
Authors: Myles Sherman, Liam Connor, Casey Law, Vikram Ravi, Dana Simard

This library contains functions for polarization analysis of FRBs. Functions herein were
designed for use with the DSA-110 file system and naming conventions, and therefore should be
used with caution. This is particularly applicable to calibration functions; functions for
interfacing with filterbank data should be portable to any system.

"""

import time
from scipy.signal import find_peaks
from scipy.signal import peak_widths
import copy
import numpy as np

from tqdm.contrib.slack import tqdm,trange
import numpy.ma as ma
from sigpyproc import FilReader
from sigpyproc.Filterbank import FilterbankBlock
from sigpyproc.Header import Header
from scipy.special import erf
from matplotlib import pyplot as plt
import pylab
import pickle
import json
from scipy.interpolate import interp1d
from scipy.stats import chi2
from scipy.stats import chi
from scipy.signal import savgol_filter as sf
from scipy.signal import convolve
from scipy.ndimage import convolve1d
from RMtools_1D.do_RMsynth_1D import run_rmsynth
from RMtools_1D.do_RMclean_1D import run_rmclean
ext= ".pdf"
import json
import os
import sys
f = open("directories.json","r")
dirs = json.load(f)
f.close()

DEFAULT_DATADIR = dirs["data"] + "testimgs/" #"/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/testimgs/"#"/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03/testimgs/" #Users can find datadirectories for all processed FRBs here; to access set datadir = DEFAULT_WDIR + trigname_label
logfile = dirs["logs"] + "dsapol_logfile.txt" #"/media/ubuntu/ssd/sherman/code/dsapol_logfiles/dsapol_logfile.txt"
from astropy.time import Time
from astropy.coordinates import EarthLocation
import astropy.units as u

import matplotlib.ticker as ticker
from concurrent.futures import ProcessPoolExecutor, as_completed, wait

#3C48 and 3C286 polynomial fit parameters from Perley-Butler 2013
coeffs_3C286 = [1.2481,-0.4507,-0.1798,0.0357]
coeffs_3C48 = [ 1.3253, -0.7553, -0.1914, 0.0498]
def PB_flux(coeffs,nu_GHz):
    logS = np.zeros(len(nu_GHz))
    for i in range(len(coeffs)):
        logS += coeffs[i]*(np.log10(nu_GHz)**(i))
    return 10**logS

#3C48 parameters
RA_3C48 = ((1 + 37/60 + 41.1/3600)*360/24) #degrees
DEC_3C48 = (33 + 9/60 + 32/3600) #degrees
RM_3C48 = -68 #rad/m^2
p_3C48 = 0.005
chip_3C48 = 25*np.pi/180 #rad


#3C286 parameters
RA_3C286 = (13 + 31/60 + 8.28811/3600)*360/24#degrees
DEC_3C286 = 30 + 30/60 + 32.96/3600#degrees
RM_3C286 = 0 #rad/m^2
p_3C286 = 0.0945 + 0.0077*(22/100)
chip_3C286 = 33*np.pi/180 #rad

#Reads in stokes parameter data from specified directory
#(Liam Connor)
def create_stokes_arr(sdir, nsamp=10240,verbose=False,dtype=np.float32,alpha=False,start=0):
    """
    This function reads in Stokes parameter data from a given directory. Stokes parameters
    are saved in high resolution filterbank files with the same prefix and numbered 0,1,2,3
    for I,Q,U,V respectively

    Inputs: sdir --> str,directory containing Stokes fil files
            nsamp --> int,number of time samples to read in
            verbose --> bool,if True prints additional information
    Outputs: size 4 list containing dynamic spectra for each stokes parameter

    """
    if verbose:
        print("Reading stokes parameters from " + sdir)
    stokes_arr=[]
    if alpha:
        labels=["I","Q","U","V"]
    for ii in range(4):
        if verbose:
            print("Reading stokes param..." + str(ii),end="")
        if alpha:
            fn = '%s_%s.fil'%(sdir,labels[ii])
        else:
            fn = '%s_%d.fil'%(sdir,ii)
        d = read_fil_data_dsa(fn, start=start, stop=nsamp,verbose=verbose)[0]
        #if d == 0:
            #print("Failed to read file: " + fn + ", returning 0")
         
        stokes_arr.append(d.astype(dtype))
        if verbose:
            print("Done!")
    stokes_arr = np.concatenate(stokes_arr).reshape(4, -1, nsamp)
    return stokes_arr

#read in Stokes I filterbank only
def create_I_arr(sdir, nsamp=10240,verbose=False,dtype=np.float32,alpha=False,start=0):
    """
    This function reads in Stokes parameter data from a given directory. Stokes parameters
    are saved in high resolution filterbank files with the same prefix and numbered 0,1,2,3
    for I,Q,U,V respectively

    Inputs: sdir --> str,directory containing Stokes fil files
            nsamp --> int,number of time samples to read in
            verbose --> bool,if True prints additional information
    Outputs: size 4 list containing dynamic spectra for each stokes parameter

    """
    if verbose:
        print("Reading stokes parameters from " + sdir)
    stokes_arr=[]
    ii =0
    if verbose:
        print("Reading stokes param..." + str(ii),end="")
    if alpha:
        fn = '%s_%s.fil'%(sdir,"I")
    else:
        fn = '%s_%d.fil'%(sdir,ii)
    d = read_fil_data_dsa(fn, start=start, stop=nsamp,verbose=verbose)[0]
    #if d == 0:
        #print("Failed to read file: " + fn + ", returning 0")
    return d.astype(np.float16)
    #stokes_arr.append(d)
    #print("Done!")
    #stokes_arr = np.concatenate(stokes_arr).reshape(4, -1, nsamp)
    #return stokes_arr

#Creates freq, time axes
def create_freq_time(sdir,nsamp=10240,alpha=False,start=0,verbose=False):#1500):
    """
    This function creates frequency and time axes from stokes filterbank headers

    Inputs: sdir --> str,directory containing Stokes fil files
            nsamp --> int,number of time samples to read in
    Outputs: (freq,dt) --> frequency and time axes respectively

    """
    freq=[]
    dt = []
    if alpha:
        labels=["I","Q","U","V"]
    for ii in range(4):
        if alpha:
            fn = '%s_%s.fil'%(sdir,labels[ii])
        else:
            fn = '%s_%d.fil'%(sdir,ii)
        d = read_fil_data_dsa(fn, start=start, stop=nsamp,verbose=verbose)
        freq.append(d[1])
        dt.append(d[2])
        #print(len(freq[ii]))
    #stokes_arr = np.concatenate(stokes_arr).reshape(4, -1, nsamp)
    return freq,dt

#Creates freq, time axes for stokes I only
def create_freq_time_I(sdir,nsamp=10240,alpha=False,start=0):#1500):
    """
    This function creates frequency and time axes from stokes filterbank headers

    Inputs: sdir --> str,directory containing Stokes fil files
            nsamp --> int,number of time samples to read in
    Outputs: (freq,dt) --> frequency and time axes respectively

    """
    freq=[]
    dt = []
    #for ii in range(4):
    ii = 0
    if alpha:
        fn = '%s_%s.fil'%(sdir,"I")
    else:
        fn = '%s_%d.fil'%(sdir,ii)
    d = read_fil_data_dsa(fn, start=start, stop=nsamp)
    freq.append(d[1])
    dt.append(d[2])
    #print(len(freq[ii]))
    #stokes_arr = np.concatenate(stokes_arr).reshape(4, -1, nsamp)
    return freq,dt

#Read stop - start samples from filterbank file fn
#(Liam Connor)
def read_fil_data_dsa(fn, start=0, stop=1,verbose=True):
    """
    This function reads a filterbank file specified and returns the data, axes and header. 

    Inputs: fn --> str, filterbank file
            start --> int, time sample to start reading
            stop --> int, time sample to stop reading
    Outputs: (data,freq,delta_t,header) --> 2D filterbank data, frequency axis, time axis, and header dictionary 

    """
    if verbose:
        print("Reading Filterbank File: " + fn)
    fil_obj = FilReader(fn)
    header = fil_obj.header
    delta_t = fil_obj.header['tsamp']
    fch1 = header['fch1']
    nchans = header['nchans']
    foff = header['foff']
    fch_f = fch1 + nchans*foff
    freq = np.linspace(fch1,fch_f,nchans)
    try:
        data = fil_obj.readBlock(start, stop)
    except(ValueError):
        if verbose:
            print("Read data failed, returning 0")
        data = 0

    return data, freq, delta_t, header

#Bin 2d (n_f x n_t) array by n samples on n_t axis
def avg_time(arr,n): #averages time axis over n samples
    """
    This function bins a 2D array of size (nchans x nsamples) along the time axis
    by n and outputs a 2D array of size (nchans x nsamples/n). Note nsamples%n must be 0

    Inputs: arr --> 2D array, size (nchans x nsamples)
            n --> int, number of time samples to bin by (average over)
    Outputs: 2D array of size (nchans x nsamples/n) 

    """
    if n == 1:
        return arr

    """
    if arr.shape[1]%n != 0:
        print("array size must be divisible by n")
        return
    """
    if arr.shape[1]%n != 0:
        arr = arr[:,arr.shape[1]%n:]

    return ((arr.transpose()).reshape(-1,n,arr.shape[0]).mean(1)).transpose()
    #return ((arr.transpose()).reshape(-1,n,arr.shape[0]).mean(1)).transpose()


#Bin 2d (n_f x n_t) array by n samples on n_f axis
def avg_freq(arr,n): #averages freq axis over n samples
    """
    This function bins a 2D array of size (nchans x nsamples) along the frequency axis
    by n and outputs a 2D array of size (nchans/n x nsamples). Note nchans%n must be 0

    Inputs: arr --> 2D array, size (nchans x nsamples)
            n --> int, number of frequency samples to bin by (average over)
    Outputs: 2D array of size (nchans/n x nsamples)

    """
    if n == 1:
        return arr
    """
    if arr.shape[0]%n != 0:
        print("array size must be divisible by n")
        return
    """
    if arr.shape[0]%n != 0:
        arr = arr[arr.shape[0]%n:,:]

    return ((arr).reshape(-1,n,arr.shape[1]).mean(1))

def find_bad_channels(I):
    bad_chans = []
    for chan in range(I.shape[0]):
        if np.all(I[chan,:] == 0):
            bad_chans.append(chan)
    return bad_chans

def fix_bad_channels(I,Q,U,V,bad_chans,iters = 100):
    Ifix = copy.deepcopy(I)
    Qfix = copy.deepcopy(Q)
    Ufix = copy.deepcopy(U)
    Vfix = copy.deepcopy(V)
    for i in range(iters):


        for bad_chan in bad_chans:
            low = np.max([bad_chan-10,0])
            hi = np.min([bad_chan+10,I.shape[0]])
            Ifix[bad_chan,:] = np.nanmean(Ifix[low:hi,:],axis=0)
            Qfix[bad_chan,:] = np.nanmean(Qfix[low:hi,:],axis=0)
            Ufix[bad_chan,:] = np.nanmean(Ufix[low:hi,:],axis=0)
            Vfix[bad_chan,:] = np.nanmean(Vfix[low:hi,:],axis=0)
            
    return (Ifix,Qfix,Ufix,Vfix)

#Takes data directory and stokes fil file prefix and returns I Q U V 2D arrays binned in time and frequency
def get_stokes_2D(datadir,fn_prefix,nsamps,n_t=1,n_f=1,n_off=3000,sub_offpulse_mean=True,fixchans=True,dtype=np.float32,alpha=False,start=0,verbose=False,fixchansfile='',fixchansfile_overwrite=False):
    """
    This function generates 2D dynamic spectra for each stokes parameter, taken from
    filterbank files in the specified directory. Optionally normalizes by subtracting off-pulse mean, but
    would not recommend this for calibrators.

    Inputs: datadir --> str, path to directory containing all 4 stokes filterbank files
            fn_prefix -->str, prefix of filterbank files, e.g. '220319aaeb_dev' for file '220319aaeb_dev_0/1/2/3.fil'
            n_t --> int, number of time samples to bin by (average over)
            n_f --> int, number of frequency samples to bin by (average over)
            n_off --> int, specifies index of end of off-pulse samples
            sub_offpulse_mean --> bool, if True, subtracts mean of data up to n_off samples
    Outputs: (I,Q,U,V,fobj,timeaxis,freq_arr,wav_arr) --> 2D dynamic spectra for I,Q,U,V; filterbank object for I file; time axis; frequency
                                                        and wavelength axes for each stokes param

    """
    sdir = datadir + fn_prefix 
    sarr = create_stokes_arr(sdir, nsamp=nsamps,dtype=dtype,alpha=alpha,start=start,verbose=verbose)
    freq,dt = create_freq_time(sdir, nsamp=nsamps,alpha=alpha,start=start,verbose=verbose)
    if alpha:
        fobj=FilReader(sdir+"_I.fil")
    else:
        fobj=FilReader(sdir+"_0.fil") #need example object for header data

    #Bin in time and frequency
    #n_t = 1#8
    #n_f = 1#32
    if verbose:
        print("Binning by " + str(n_t)  + " in time")
        print("Binning by " + str(n_f) + " in frequency")
    #timeaxis = np.arange(fobj.header.tstart*86400, fobj.header.tstart*86400 + fobj.header.tsamp*fobj.header.nsamples/n_t, fobj.header.tsamp*n_t)
    if fobj.header.nsamples == 5120:
        timeaxis = np.linspace(0,fobj.header.tsamp*(fobj.header.nsamples),(fobj.header.nsamples)//n_t)
    else:
        timeaxis = np.linspace(0,fobj.header.tsamp*(fobj.header.nsamples//4),(fobj.header.nsamples//4)//n_t)
    I,Q,U,V = avg_time(sarr[0],n_t),avg_time(sarr[1],n_t),avg_time(sarr[2],n_t),avg_time(sarr[3],n_t)
    I,Q,U,V = avg_freq(I,n_f),avg_freq(Q,n_f),avg_freq(U,n_f),avg_freq(V,n_f)

    if fixchans == True:
        #bad_chans = np.arange(I.shape[0])[np.all(I==0,axis=1)]#find_bad_channels(I)
        #(I,Q,U,V) = fix_bad_channels(I,Q,U,V,bad_chans)
        if fixchansfile != '':
            bad_chans = np.load(fixchansfile)
            good_chans = np.array(list(set(np.arange(I.shape[0])) - set(bad_chans)))
        else:
            bad_idxs = np.all(I==0,axis=1)
            good_idxs = np.logical_not(bad_idxs)
            bad_chans = np.arange(I.shape[0])[bad_idxs]
            good_chans = np.arange(I.shape[0])[good_idxs]
            if fixchansfile_overwrite:
                np.save(datadir + "/badchans.npy",bad_chans)
                print("Saving bad channels to " + datadir + "/badchans.npy")
        if verbose:
            print("Bad Channels: " + str(bad_chans))
        



        #mask
        #mask = np.zeros(I.shape)
        #mask[bad_chans,:] = 1
        #I = ma.masked_array(I,mask)
        #Q = ma.masked_array(Q,mask)
        #U = ma.masked_array(U,mask)
        #V = ma.masked_array(V,mask)
    else:
        bad_chans = np.array([],dtype=int)
    
    #mask
    mask = np.zeros(I.shape)
    mask[bad_chans,:] = 1
    I = ma.masked_array(I,mask)
    Q = ma.masked_array(Q,mask)
    U = ma.masked_array(U,mask)
    V = ma.masked_array(V,mask)

    #Subtract off-pulse mean
    if sub_offpulse_mean:
        """
        offpulse_I = np.mean(I[:,:n_off],axis=1,keepdims=True) 
        offpulse_Q = np.mean(Q[:,:n_off],axis=1,keepdims=True)
        offpulse_U = np.mean(U[:,:n_off],axis=1,keepdims=True)
        offpulse_V = np.mean(V[:,:n_off],axis=1,keepdims=True)
    
        offpulse_I_std = np.std(I[:,:n_off],axis=1,keepdims=True)
        offpulse_Q_std = np.std(Q[:,:n_off],axis=1,keepdims=True)
        offpulse_U_std = np.std(U[:,:n_off],axis=1,keepdims=True)
        offpulse_V_std = np.std(V[:,:n_off],axis=1,keepdims=True)
    
        if fixchans:
            offpulse_I_std[bad_chans,:] = 1
            offpulse_Q_std[bad_chans,:] = 1
            offpulse_U_std[bad_chans,:] = 1
            offpulse_V_std[bad_chans,:] = 1

        I = (I - offpulse_I)/offpulse_I_std
        Q = (Q - offpulse_Q)/offpulse_Q_std
        U = (U - offpulse_U)/offpulse_U_std
        V = (V - offpulse_V)/offpulse_V_std
        """

        """
        offpulse_I = np.mean(I[:,:n_off],axis=1,keepdims=False)
        offpulse_Q = np.mean(Q[:,:n_off],axis=1,keepdims=False)
        offpulse_U = np.mean(U[:,:n_off],axis=1,keepdims=False)
        offpulse_V = np.mean(V[:,:n_off],axis=1,keepdims=False)

        offpulse_I_std = np.std(I[:,:n_off],axis=1,keepdims=False)
        offpulse_Q_std = np.std(Q[:,:n_off],axis=1,keepdims=False)
        offpulse_U_std = np.std(U[:,:n_off],axis=1,keepdims=False)
        offpulse_V_std = np.std(V[:,:n_off],axis=1,keepdims=False)

        if fixchans:
            offpulse_I_std[bad_chans] = 1
            offpulse_Q_std[bad_chans] = 1
            offpulse_U_std[bad_chans] = 1
            offpulse_V_std[bad_chans] = 1

        """
        if fixchans:
            offpulse_I = np.zeros(I.shape[0])
            offpulse_Q = np.zeros(I.shape[0])
            offpulse_U = np.zeros(I.shape[0])
            offpulse_V = np.zeros(I.shape[0])

            offpulse_I[good_chans] = np.mean(I.data[good_chans,:n_off],axis=1,keepdims=False,dtype=np.float32).astype(dtype)
            offpulse_Q[good_chans] = np.mean(Q.data[good_chans,:n_off],axis=1,keepdims=False,dtype=np.float32).astype(dtype)
            offpulse_U[good_chans] = np.mean(U.data[good_chans,:n_off],axis=1,keepdims=False,dtype=np.float32).astype(dtype)
            offpulse_V[good_chans] = np.mean(V.data[good_chans,:n_off],axis=1,keepdims=False,dtype=np.float32).astype(dtype)


            offpulse_I_std = np.ones(I.shape[0])
            offpulse_Q_std = np.ones(I.shape[0])
            offpulse_U_std = np.ones(I.shape[0])
            offpulse_V_std = np.ones(I.shape[0])

            offpulse_I_std[good_chans] = np.std(I.data[good_chans,:n_off],axis=1,keepdims=False,dtype=np.float32).astype(dtype)
            offpulse_Q_std[good_chans] = np.std(Q.data[good_chans,:n_off],axis=1,keepdims=False,dtype=np.float32).astype(dtype)
            offpulse_U_std[good_chans] = np.std(U.data[good_chans,:n_off],axis=1,keepdims=False,dtype=np.float32).astype(dtype)
            offpulse_V_std[good_chans] = np.std(V.data[good_chans,:n_off],axis=1,keepdims=False,dtype=np.float32).astype(dtype)


        else:
            offpulse_I = np.mean(I.data[:,:n_off],axis=1,keepdims=False,dtype=np.float32).astype(dtype)
            offpulse_Q = np.mean(Q.data[:,:n_off],axis=1,keepdims=False,dtype=np.float32).astype(dtype)
            offpulse_U = np.mean(U.data[:,:n_off],axis=1,keepdims=False,dtype=np.float32).astype(dtype)
            offpulse_V = np.mean(V.data[:,:n_off],axis=1,keepdims=False,dtype=np.float32).astype(dtype)

            offpulse_I_std = np.std(I.data[:,:n_off],axis=1,keepdims=False,dtype=np.float32).astype(dtype)
            offpulse_Q_std = np.std(Q.data[:,:n_off],axis=1,keepdims=False,dtype=np.float32).astype(dtype)
            offpulse_U_std = np.std(U.data[:,:n_off],axis=1,keepdims=False,dtype=np.float32).astype(dtype)
            offpulse_V_std = np.std(V.data[:,:n_off],axis=1,keepdims=False,dtype=np.float32).astype(dtype)


        I = ((I - offpulse_I[..., np.newaxis])/offpulse_I_std[..., np.newaxis])
        Q = ((Q - offpulse_Q[..., np.newaxis])/offpulse_Q_std[..., np.newaxis])
        U = ((U - offpulse_U[..., np.newaxis])/offpulse_U_std[..., np.newaxis])
        V = ((V - offpulse_V[..., np.newaxis])/offpulse_V_std[..., np.newaxis])


        #I = ((I.transpose() - offpulse_I)/offpulse_I_std).transpose()
        #Q = ((Q.transpose() - offpulse_Q)/offpulse_Q_std).transpose()
        #U = ((U.transpose() - offpulse_U)/offpulse_U_std).transpose()
        #V = ((V.transpose() - offpulse_V)/offpulse_V_std).transpose()

    #calculate frequency and wavelength arrays (separate array for each stokes parameter, but should be the same)
    c = (3e8) #m/s
    freq_arr = []
    wav_arr = []
    for i in range(4):
        freq_arr.append(freq[i].reshape(-1,n_f).mean(1))
        wav_arr.append(c/(np.array(freq_arr[i])*(1e6)))

    
    return (I,Q,U,V,fobj,timeaxis,freq_arr,wav_arr,bad_chans)



#functions for rewriting data to filterbanks
def write_fil_data_dsa(arr,fn,fobj):
    """
    This function writes a 2D array to a filterbank file at the 
    given file path.

    Inputs: arr --> array-like, arbitrary 2D array
            fn --> str, path to filterbank file to write the array to; should end with '.fil'
            fobj --> filterbank object containing the object header (this can be obtained by reading a separate filterbank file, or using sigpyproc)
    """
    #create filterbank block object with identical header
    b = FilterbankBlock(arr,header=fobj.header)
    b.toFile(fn)
    return

def put_stokes_2D(I,Q,U,V,fobj,datadir,fn_prefix,suffix="polcal",alpha=False):
    """ 
    This function writes the provided Stokes dynamic spectra to 
    filterbank files in the given directory.

    Inputs: I,Q,U,V --> array-like, 2D dynamic spectra for I,Q,U,V
            fobj -->filterbank object containing the object header (this can be obtained by reading a separate filterbank file, or using sigpyproc)
            datadir --> str, path to directory to write filterbank files to
            fn_prefix --> str, prefix of filterbank files, e.g. '220319aaeb_dev' for file '220319aaeb_dev_0/1/2/3.fil'
            suffix --> str, suffix to be appended to end of filterbank file name (before 0/1/2/3)
            alpha --> bool, True if desired filterbanks should end with 'I,Q,U,V', False if desired filterbanks should end with '0,1,2,3' (default=False)
    """
    if alpha:
        sdir = datadir + fn_prefix + "_" + suffix
        print("Writing Stokes I to " + sdir + "_I.fil")
        write_fil_data_dsa(I,sdir + "_I.fil",fobj)
        print("Writing Stokes Q to " + sdir + "_Q.fil")
        write_fil_data_dsa(Q,sdir + "_Q.fil",fobj)
        print("Writing Stokes U to " + sdir + "_U.fil")
        write_fil_data_dsa(U,sdir + "_U.fil",fobj)
        print("Writing Stokes V to " + sdir + "_V.fil")
        write_fil_data_dsa(V,sdir + "_V.fil",fobj)

    else:
        sdir = datadir + fn_prefix + "_" + suffix
        print("Writing Stokes I to " + sdir + "_0.fil")
        write_fil_data_dsa(I,sdir + "_0.fil",fobj)
        print("Writing Stokes Q to " + sdir + "_1.fil")
        write_fil_data_dsa(Q,sdir + "_1.fil",fobj)
        print("Writing Stokes U to " + sdir + "_2.fil")
        write_fil_data_dsa(U,sdir + "_2.fil",fobj)
        print("Writing Stokes V to " + sdir + "_3.fil")
        write_fil_data_dsa(V,sdir + "_3.fil",fobj)
    return


#Takes data directory and stokes fil file prefix and returns I Q U V 2D arrays binned in time and frequency
def get_I_2D(datadir,fn_prefix,nsamps,n_t=1,n_f=1,n_off=3000,sub_offpulse_mean=True,fixchans=True,dtype=np.float32,alpha=False):
    """
    This function generates 2D dynamic spectra for each stokes parameter, taken from
    filterbank files in the specified directory. Optionally normalizes by subtracting off-pulse mean, but
    would not recommend this for calibrators.

    Inputs: datadir --> str, path to directory containing stokes I filterbank files
            fn_prefix -->str, prefix of filterbank files, e.g. '220319aaeb_dev' for file '220319aaeb_dev_0/1/2/3.fil'
            n_t --> int, number of time samples to bin by (average over)
            n_f --> int, number of frequency samples to bin by (average over)
            n_off --> int, specifies index of end of off-pulse samples
            sub_offpulse_mean --> bool, if True, subtracts mean of data up to n_off samples
            fixchans --> bool, if True, masks any channels with 0 rms, 0 mean (i.e. correlator issue)
            dtype --> dtype, default np.float32
            alpha --> bool, True if desired filterbanks should end with 'I', False if desired filterbank should end with '0' (default=False)
    Outputs: (I,fobj,timeaxis,freq_arr,wav_arr) --> 2D dynamic spectra for I; filterbank object for I file; time axis; frequency
                                                        and wavelength axes for each stokes param

    """
    sdir = datadir + fn_prefix
    I = create_I_arr(sdir, nsamp=nsamps,dtype=dtype,alpha=alpha)
    freq,dt = create_freq_time_I(sdir, nsamp=nsamps,alpha=alpha)
    if alpha:
        fobj=FilReader(sdir+"_I.fil")
    else:
        fobj=FilReader(sdir+"_0.fil") #need example object for header data

    #Bin in time and frequency
    #n_t = 1#8
    #n_f = 1#32
    print("Binning by " + str(n_t)  + " in time")
    print("Binning by " + str(n_f) + " in frequency")
    #timeaxis = np.arange(fobj.header.tstart*86400, fobj.header.tstart*86400 + fobj.header.tsamp*fobj.header.nsamples/n_t, fobj.header.tsamp*n_t)
    if fobj.header.nsamples == 5120:
        timeaxis = np.linspace(0,fobj.header.tsamp*(fobj.header.nsamples),(fobj.header.nsamples)//n_t)
    else:
        timeaxis = np.linspace(0,fobj.header.tsamp*(fobj.header.nsamples//4),(fobj.header.nsamples//4)//n_t)
    I= avg_time(I,n_t)#,avg_time(sarr[1],n_t),avg_time(sarr[2],n_t),avg_time(sarr[3],n_t)
    I= avg_freq(I,n_f)#,avg_freq(Q,n_f),avg_freq(U,n_f),avg_freq(V,n_f)


    if fixchans == True:
        bad_chans = find_bad_channels(I)
        #(I,Q,U,V) = fix_bad_channels(I,Q,U,V,bad_chans)
        print("Bad Channels: " + str(bad_chans))


        #mask
        mask = np.zeros(I.shape)
        mask[bad_chans,:] = 1
        I = ma.masked_array(I,mask)
        #Q = ma.masked_array(Q,mask)
        #U = ma.masked_array(U,mask)
        #V = ma.masked_array(V,mask)

    #Subtract off-pulse mean
    if sub_offpulse_mean:
        """
        offpulse_I = np.mean(I[:,:n_off],axis=1,keepdims=True) 
        offpulse_Q = np.mean(Q[:,:n_off],axis=1,keepdims=True)
        offpulse_U = np.mean(U[:,:n_off],axis=1,keepdims=True)
        offpulse_V = np.mean(V[:,:n_off],axis=1,keepdims=True)
    
        offpulse_I_std = np.std(I[:,:n_off],axis=1,keepdims=True)
        offpulse_Q_std = np.std(Q[:,:n_off],axis=1,keepdims=True)
        offpulse_U_std = np.std(U[:,:n_off],axis=1,keepdims=True)
        offpulse_V_std = np.std(V[:,:n_off],axis=1,keepdims=True)
    
        if fixchans:
            offpulse_I_std[bad_chans,:] = 1
            offpulse_Q_std[bad_chans,:] = 1
            offpulse_U_std[bad_chans,:] = 1
            offpulse_V_std[bad_chans,:] = 1

        I = (I - offpulse_I)/offpulse_I_std
        Q = (Q - offpulse_Q)/offpulse_Q_std
        U = (U - offpulse_U)/offpulse_U_std
        V = (V - offpulse_V)/offpulse_V_std
        """
        offpulse_I = np.mean(I[:,:n_off],axis=1,keepdims=False)
        #offpulse_Q = np.mean(Q[:,:n_off],axis=1,keepdims=False)
        #offpulse_U = np.mean(U[:,:n_off],axis=1,keepdims=False)
        #offpulse_V = np.mean(V[:,:n_off],axis=1,keepdims=False)

        offpulse_I_std = np.std(I[:,:n_off],axis=1,keepdims=False)
        #offpulse_Q_std = np.std(Q[:,:n_off],axis=1,keepdims=False)
        #offpulse_U_std = np.std(U[:,:n_off],axis=1,keepdims=False)
        #offpulse_V_std = np.std(V[:,:n_off],axis=1,keepdims=False)

        if fixchans:
            offpulse_I_std[bad_chans] = 1
         #   offpulse_Q_std[bad_chans] = 1
         #   offpulse_U_std[bad_chans] = 1
         #   offpulse_V_std[bad_chans] = 1

        I = ((I.transpose() - offpulse_I)/offpulse_I_std).transpose()
        #Q = ((Q.transpose() - offpulse_Q)/offpulse_Q_std).transpose()
        #U = ((U.transpose() - offpulse_U)/offpulse_U_std).transpose()
        #V = ((V.transpose() - offpulse_V)/offpulse_V_std).transpose()

    #calculate frequency and wavelength arrays (separate array for each stokes parameter, but should be the same)
    c = (3e8) #m/s
    freq_1D = freq[0].reshape(-1,n_f).mean(1)
    wav_1D = c/(np.array(freq_1D*1e6))

    """
    freq_arr = []
    wav_arr = []
    for i in range(4):
        freq_arr.append(freq[i].reshape(-1,n_f).mean(1))
        wav_arr.append(list(c/(np.array(freq_arr[i])*(1e6))))
    """

    return (I,fobj,timeaxis,freq_1D,wav_1D)

#Get frequency averaged stokes params vs time; note run get_stokes_2D first to get 2D I Q U V arrays
def get_stokes_vs_time(I,Q,U,V,width_native,t_samp,n_t,n_off=3000,plot=False,datadir=DEFAULT_DATADIR,label='',calstr='',ext=ext,show=False,normalize=True,buff=0,timeaxis=None,fobj=None,window=10):
    """
    This function calculates the frequency averaged time series for each stokes
    parameter. Outputs plots if specified in region around the pulse.

    Inputs: I,Q,U,V --> 2D arrays, dynamic spectra of I,Q,U,V generated with get_stokes_2D()
            width_native --> int, width in samples of pulse in native sampling rate; equivalent to ibox parameter
            t_samp --> float, sampling time
            n_t --> int, number of time samples to bin by (average over)
            n_off --> int, specifies index of end of off-pulse samples
            plot --> bool, if True, outputs plots of time series into datadir
            datadir --> str, path to directory to output images
            label --> str, trigname_nickname of FRB, e.g. '220319aaeb_Mark'
            calstr --> str, string specifying whether given data is calibrated or not, optional
            ext --> str, image file extension (png, pdf, etc.)
            show --> bool, if True, displays images with matplotlib
            normalize --> bool, if True subtracts off-pulse mean averaged over frequency and divides by off-pulse standard deviation
            buff --> int, number of samples to buffer either side of pulse if needed
            timeaxis --> array-like, time axis returned by get_stokes_2D
            fobj --> filterbank object containing the object header (this can be obtained by reading a separate filterbank file, or using sigpyproc)
            window --> int, window in samples around pulse to plot
    Outputs: (I_t,Q_t,U_t,V_t) --> 1D time series of each stokes parameter, IQUV respectively
                            

    """
    #use full timestream for calibrators
    if width_native == -1:
        timestart = 0
        timestop = I.shape[1]
        #suboffp = 0
    else:
        peak,timestart,timestop = find_peak(I,width_native,t_samp,n_t,buff=buff) 
        #suboffp = 1
    
    if normalize:
        I_t = (np.mean(I,axis=0) - np.mean(np.mean(I[:,:n_off],axis=0)))/np.std(np.mean(I[:,:n_off],axis=0))
        Q_t = (np.mean(Q,axis=0) - np.mean(np.mean(Q[:,:n_off],axis=0)))/np.std(np.mean(Q[:,:n_off],axis=0))
        U_t = (np.mean(U,axis=0) - np.mean(np.mean(U[:,:n_off],axis=0)))/np.std(np.mean(U[:,:n_off],axis=0))
        V_t = (np.mean(V,axis=0) - np.mean(np.mean(V[:,:n_off],axis=0)))/np.std(np.mean(V[:,:n_off],axis=0))
    else:
        I_t = (np.mean(I,axis=0))
        Q_t = (np.mean(Q,axis=0))
        U_t = (np.mean(U,axis=0))
        V_t = (np.mean(V,axis=0))


    if plot:
        f=plt.figure(figsize=(12,6))
        plt.plot(I_t,label="I")
        plt.plot(Q_t,label="Q")
        plt.plot(U_t,label="U")
        plt.plot(V_t,label="V")
        plt.axvline(timestart,color="red")
        plt.axvline(timestop,color="red")
        plt.grid()
        plt.legend()
        plt.xlim(timestart-window,timestop+window)
        plt.xlabel("Time Sample (" + str(t_samp*n_t) + " s sampling time)")
        plt.title(label)
        plt.savefig(datadir +label + "_time_"+ calstr + str(n_t) + "_binned" + ext)
        if show:
            plt.show()
        plt.close(f)
    
    
    return (I_t,Q_t,U_t,V_t)


#Get optimal SNR weights using binning that maximizes SNR
def get_weights(I,Q,U,V,width_native,t_samp,n_f,n_t,freq_test,timeaxis,fobj,n_off=3000,buff=0,n_t_weight=1,sf_window_weights=45,padded=True,norm=True):
    """
    This function calculates ideal weights by downsampling and smoothing the 
    pulse profile. This uses a Savitsky-Golay (Savgol) filter for smoothing and
    interpolates back to the initial time resolution.
    Input: I,Q,U,V --> 2D arrays, dynamic spectra of I,Q,U,V generated with get_stokes_2D()
           width_native --> int, width in samples of pulse in native sampling rate; equivalent to ibox parameter
           t_samp -->  float, sampling time
           n_f --> int, number of frequency samples binned by
            n_t --> int, number of time samples binned by (average over)
            freq_test --> list, frequency axes for each stokes parameter 
            n_off --> int, specifies index of end of off-pulse samples
            buff --> int, number of samples to buffer either side of pulse if needed
            n_t_weight --> int, number of time samples to bin time series by for smoothing
            sf_window_weights --> int, width of 3rd order SavGol filter in samples
            padded --> bool, pad the filter weights with zeros to the original number of timesamples; if False, returns a size timestop-timesart window around the peak
            norm --> bool, normalize weights so they sum to 1
    Output: weights --> 1D array of weights
    """
    (peak,timestart,timestop) = find_peak(I,width_native,t_samp,n_t,buff=buff)
    #Bin to optimal time binning
    Ib,Qb,Ub,Vb = avg_time(I,n_t_weight),avg_time(Q,n_t_weight),avg_time(U,n_t_weight),avg_time(V,n_t_weight)
    """
    if fobj.header.nsamples == 5120:
        timeaxisb = np.linspace(0,fobj.header.tsamp*(fobj.header.nsamples),(fobj.header.nsamples)//(n_t*n_t_weight))
    else:
        timeaxisb = np.linspace(0,fobj.header.tsamp*(fobj.header.nsamples//4),(fobj.header.nsamples//4)//(n_t*n_t_weight))
    """
    #this is simpler:
    timeaxis = np.arange(I.shape[1])
    #timaxisb = np.linspace(0,I.shape[1],Ib.shape[1])

    #Calculate weights
    (I_t,Q_t,U_t,V_t) = get_stokes_vs_time(Ib,Qb,Ub,Vb,width_native,t_samp,n_t*n_t_weight,n_off//n_t_weight,normalize=True,buff=buff) #note normalization always used to get SNR
    #timeaxisb = np.linspace(0,I.shape[1],len(I_t))
    timeaxist = np.zeros((1,timeaxis.shape[0]))
    timeaxist[0,:] = timeaxis

    if timeaxist.shape[1]%n_t_weight != 0:
        timeaxist = timeaxist[:,timeaxist.shape[1]%n_t_weight:]
    
    timeaxisb = (((timeaxist.transpose()).reshape(-1,n_t_weight,timeaxist.shape[0]).mean(1)).transpose())[0]
    #print(len(timeaxisb),len(I_t))
    #print(timeaxist)
    #print(timeaxisb)
    #print(((timeaxist.transpose()).reshape(-1,n_t_weight,timeaxist.shape[0]).mean(1)).transpose())

    #Interpolate to original sample rate
    fI = interp1d(timeaxisb,I_t,kind="linear",fill_value="extrapolate")
    fQ = interp1d(timeaxisb,Q_t,kind="linear",fill_value="extrapolate")
    fU = interp1d(timeaxisb,U_t,kind="linear",fill_value="extrapolate")
    fV = interp1d(timeaxisb,V_t,kind="linear",fill_value="extrapolate")


    #divide by sum to normalize to 1-- actually don't because you already do this at the end
    I_t_weight = fI(timeaxis)#/np.sum(fI(timeaxis))
    Q_t_weight = fQ(timeaxis)#/np.sum(fQ(timeaxis))
    U_t_weight = fU(timeaxis)#/np.sum(fU(timeaxis))
    V_t_weight = fV(timeaxis)#/np.sum(fV(timeaxis))

    #savgol filter
    if sf_window_weights > 3:
        I_t_weight = sf(I_t_weight,sf_window_weights,3)
        Q_t_weight = sf(Q_t_weight,sf_window_weights,3)
        U_t_weight = sf(U_t_weight,sf_window_weights,3)
        V_t_weight = sf(V_t_weight,sf_window_weights,3)
    else:
        print("Skip Savgol Filter, sf_window_weights <= 3")

    #take absolute value (negative weights meaningless)
    I_t_weight = np.abs(I_t_weight)

    #mark any points where I<0 as invalid
    I_t_weight[I.mean(0)<0] = 0
    #repeat over frequency
    #I_weight = np.abs(np.array([I_t_weight]*I.shape[0]))
    #Q_weight = np.abs(np.array([Q_t_weight]*Q.shape[0]))
    #U_weight = np.abs(np.array([U_t_weight]*U.shape[0]))
    #V_weight = np.abs(np.array([V_t_weight]*V.shape[0]))
    
    #I_weight = (np.array([I_t_weight]*I.shape[0]))
    #Q_weight = (np.array([Q_t_weight]*Q.shape[0]))
    #U_weight = (np.array([U_t_weight]*U.shape[0]))
    #V_weight = (np.array([V_t_weight]*V.shape[0]))

    #zero outside of window
    #(peak,timestart,timestop) = find_peak(I_t_weight,width_native,t_samp,n_t,buff=buff)
    if padded: 
        #I_t_weight[:np.argmax(I_t_weight)-buff] = 0
        #I_t_weight[np.argmax(I_t_weight)+buff:] = 0
        I_t_weight[:timestart] = 0
        I_t_weight[timestop:] = 0
    if not padded:
        I_t_weight = I_t_weight[timestart:timestop]
    #print(I_t_weight.shape)    

    if norm:
        norm_factor = np.sum(I_t_weight)
    else:
        norm_factor = 1
    return I_t_weight/norm_factor #/np.sum(I_t_weight)#,Q_weight,U_weight,V_weight


def get_weights_1D(I_t_init,Q_t_init,U_t_init,V_t_init,timestart,timestop,width_native,t_samp,n_f,n_t,freq_test,timeaxis,fobj,n_off=3000,buff=0,n_t_weight=1,sf_window_weights=45,padded=True,norm=True):
    """
    This function calculates ideal weights by downsampling and smoothing the
    pulse profile, taking pre-computed time-series as input. To use 2D filterbanks 
    as inputs, use get_weights. This uses a Savitsky-Golay (Savgol) filter for smoothing and
    interpolates back to the initial time resolution.
    Input: I_t_init,Q_t_init,U_t_init,V_t_init --> 1D arrays, frequency averaged time series of I,Q,U,V generated e.g. with get_stokes_vs_time()
           timestart, timestop --> int, bounding sample numbers of burst returned from find_peak()
           width_native --> int, width in samples of pulse in native sampling rate; equivalent to ibox parameter
           t_samp -->  float, sampling time
           n_f --> int, number of frequency samples binned by
            n_t --> int, number of time samples binned by (average over)
            freq_test --> list, frequency axes for each stokes parameter
            timeaxis --> array-like, time sample array returned e.g. from get_stokes_2D()
            fobj --> filterbank object containing the object header (this can be obtained by reading a separate filterbank file, or using sigpyproc)
            n_off --> int, specifies index of end of off-pulse samples
            buff --> int, number of samples to buffer either side of pulse if needed
            n_t_weight --> int, number of time samples to bin time series by for smoothing
            sf_window_weights --> int, width of 3rd order SavGol filter in samples
            padded --> bool, pad the filter weights with zeros to the original number of timesamples; if False, returns a size timestop-timesart window around the peak
            norm --> bool, normalize weights so they sum to 1
    Output: weights --> 1D array of weights
    """

    if timestart==-1 or timestop==-1:
        peak,timestart,timestop = find_peak((I_t_init,I_t_init),width_native,t_samp,n_t,buff=buff,pre_calc_tf=True)

    #downsample
    """
    I_t = np.zeros((1,len(I_t_init)))
    I_t[0,:] = I_t_init[I_t_init.shape[0]%n_t_weight:]
    I_t = (((I_t.transpose()).reshape(-1,n_t_weight,I_t.shape[0]).mean(1)).transpose())[0]
    Q_t = np.zeros((1,len(Q_t_init)))
    Q_t[0,:] = Q_t_init[Q_t_init.shape[0]%n_t_weight:]
    Q_t = (((Q_t.transpose()).reshape(-1,n_t_weight,Q_t.shape[0]).mean(1)).transpose())[0]
    U_t = np.zeros((1,len(U_t_init)))
    U_t[0,:] = U_t_init[U_t_init.shape[0]%n_t_weight:]
    U_t = (((U_t.transpose()).reshape(-1,n_t_weight,U_t.shape[0]).mean(1)).transpose())[0]
    V_t = np.zeros((1,len(V_t_init)))
    V_t[0,:] = V_t_init[V_t_init.shape[0]%n_t_weight:]
    V_t = (((V_t.transpose()).reshape(-1,n_t_weight,V_t.shape[0]).mean(1)).transpose())[0]
    """
   
    I_t = I_t_init[len(I_t_init)%n_t_weight:]
    I_t = I_t.reshape(len(I_t)//n_t_weight,n_t_weight).mean(1)
    Q_t = Q_t_init[len(Q_t_init)%n_t_weight:]
    Q_t = Q_t.reshape(len(Q_t)//n_t_weight,n_t_weight).mean(1)
    U_t = U_t_init[len(U_t_init)%n_t_weight:]
    U_t = U_t.reshape(len(U_t)//n_t_weight,n_t_weight).mean(1)
    V_t = V_t_init[len(V_t_init)%n_t_weight:]
    V_t = V_t.reshape(len(V_t)//n_t_weight,n_t_weight).mean(1)



    timeaxis = np.arange(I_t_init.shape[0])
    #timaxisb = np.linspace(0,I.shape[1],Ib.shape[1])

    #Calculate weights
    #(I_t,Q_t,U_t,V_t) = get_stokes_vs_time(Ib,Qb,Ub,Vb,width_native,t_samp,n_t*n_t_weight,n_off//n_t_weight,normalize=True,buff=buff) #note normalization always used to get SNR
    timeaxist = np.zeros((1,timeaxis.shape[0]))
    timeaxist[0,:] = timeaxis

    if timeaxist.shape[1]%n_t_weight != 0:
        timeaxist = timeaxist[:,timeaxist.shape[1]%n_t_weight:]

    timeaxisb = (((timeaxist.transpose()).reshape(-1,n_t_weight,timeaxist.shape[0]).mean(1)).transpose())[0]
    #timeaxisb = np.linspace(0,I_t.shape[0],len(I_t))
    #print(len(timeaxisb),len(I_t))


    #Interpolate to original sample rate
    fI = interp1d(timeaxisb,I_t,kind="linear",fill_value="extrapolate")
    fQ = interp1d(timeaxisb,Q_t,kind="linear",fill_value="extrapolate")
    fU = interp1d(timeaxisb,U_t,kind="linear",fill_value="extrapolate")
    fV = interp1d(timeaxisb,V_t,kind="linear",fill_value="extrapolate")


    #divide by sum to normalize to 1-- actually don't because you already do this at the end
    I_t_weight = fI(timeaxis)#/np.sum(fI(timeaxis))
    Q_t_weight = fQ(timeaxis)#/np.sum(fQ(timeaxis))
    U_t_weight = fU(timeaxis)#/np.sum(fU(timeaxis))
    V_t_weight = fV(timeaxis)#/np.sum(fV(timeaxis))

    #savgol filter
    #if sf_window_weights <= 3:
    #print("No SF Filter")
    if sf_window_weights > 3:
        I_t_weight = sf(I_t_weight,sf_window_weights,3)
        Q_t_weight = sf(Q_t_weight,sf_window_weights,3)
        U_t_weight = sf(U_t_weight,sf_window_weights,3)
        V_t_weight = sf(V_t_weight,sf_window_weights,3)

    #take absolute value (negative weights meaningless)
    I_t_weight = np.abs(I_t_weight)

    #mark any points where I<0 as invalid
    I_t_weight[I_t_init<0] = 0
    #repeat over frequency
    #I_weight = np.abs(np.array([I_t_weight]*I.shape[0]))
    #Q_weight = np.abs(np.array([Q_t_weight]*Q.shape[0]))
    #U_weight = np.abs(np.array([U_t_weight]*U.shape[0]))
    #V_weight = np.abs(np.array([V_t_weight]*V.shape[0]))

    #I_weight = (np.array([I_t_weight]*I.shape[0]))
    #Q_weight = (np.array([Q_t_weight]*Q.shape[0]))
    #U_weight = (np.array([U_t_weight]*U.shape[0]))
    #V_weight = (np.array([V_t_weight]*V.shape[0]))

    #zero outside of window
    #(peak,timestart,timestop) = find_peak(I_t_weight,width_native,t_samp,n_t,buff=buff)
    if padded:
        #I_t_weight[:np.argmax(I_t_weight)-buff] = 0
        #I_t_weight[np.argmax(I_t_weight)+buff:] = 0
        I_t_weight[:timestart] = 0
        I_t_weight[timestop:] = 0
    if not padded:
        I_t_weight = I_t_weight[timestart:timestop]
    #print(I_t_weight.shape)    

    if norm:
        norm_factor = np.sum(I_t_weight)
    else:
        norm_factor = 1
    return I_t_weight/norm_factor #/np.sum(I_t_weight)#,Q_weight,U_weight,V_weight

#Weight dynamic spectrum using get_weights
def weight_spectra_2D(I,Q,U,V,width_native,t_samp,n_f,n_t,freq_test,timeaxis,fobj,n_off=3000,buff=0,n_t_weight=1,sf_window_weights=45):
    """
    This function applies ideal filter weights to Stokes dynamic
    spectra. This is a helper function for get_stokes_vs_freq().

    Inputs: I,Q,U,V --> 2D arrays, dynamic spectra of I,Q,U,V generated with get_stokes_2D()
            width_native --> int, width in samples of pulse in native sampling rate; equivalent to ibox parameter
            t_samp --> float, sampling time
            n_f --> int, number of frequency samples to bin by (average over)
            n_t --> int, number of time samples to bin by (average over)
            freq_test --> list, frequency axes for each stokes parameter
            timeaxis --> array-like, time sample array returned e.g. from get_stokes_2D()
            fobj --> filterbank object containing the object header (this can be obtained by reading a separate filterbank file, or using sigpyproc)
            n_off --> int, specifies index of end of off-pulse samples
            buff --> int, number of samples to buffer either side of pulse if needed
            n_t_weight --> int, number of time samples to bin time series by for smoothing
            sf_window_weights --> int, width of 3rd order SavGol filter in samples
    Outputs: (I,Q,U,V) --> 2D filterbanks weighted for I,Q,U,V

    """
    peak,timestart,timestop = find_peak(I,width_native,t_samp,n_t,buff=buff)
    I_t_weights = get_weights(I,Q,U,V,width_native,t_samp,n_f,n_t,freq_test,timeaxis,fobj=fobj,n_off=n_off,buff=buff,n_t_weight=n_t_weight,sf_window_weights=sf_window_weights)

    I_rolled = np.roll(I,-peak,axis=1)
    Q_rolled = np.roll(Q,-peak,axis=1)
    U_rolled = np.roll(U,-peak,axis=1)
    V_rolled = np.roll(V,-peak,axis=1)
    I_t_weights_rolled = np.roll(I_t_weights,-np.argmax(I_t_weights))

    Iwr = convolve1d(I_rolled,I_t_weights_rolled,axis=1,mode="constant",cval=0)
    Qwr = convolve1d(Q_rolled,I_t_weights_rolled,axis=1,mode="constant",cval=0)
    Uwr = convolve1d(U_rolled,I_t_weights_rolled,axis=1,mode="constant",cval=0)
    Vwr = convolve1d(V_rolled,I_t_weights_rolled,axis=1,mode="constant",cval=0)
    I = np.roll(Iwr,peak,axis=1)
    Q = np.roll(Qwr,peak,axis=1)
    U = np.roll(Uwr,peak,axis=1)
    V = np.roll(Vwr,peak,axis=1)
    return I,Q,U,V


#Get time averaged (over given width) stokes params vs freq; note run get_stokes_2D first to get 2D I Q U V arrays
def get_stokes_vs_freq(I,Q,U,V,width_native,t_samp,n_f,n_t,freq_test,n_off=3000,plot=False,datadir=DEFAULT_DATADIR,label='',calstr='',ext=ext,show=False,normalize=False,buff=0,weighted=False,n_t_weight=1,timeaxis=None,fobj=None,sf_window_weights=45,input_weights=[]):
    """
    This function calculates the time averaged (over width of pulse) frequency spectra for each stokes
    parameter. Outputs plots if specified in region around the pulse.

    Inputs: I,Q,U,V --> 2D arrays, dynamic spectra of I,Q,U,V generated with plot_spectra_2D()
            width_native --> int, width in samples of pulse in native sampling rate; equivalent to ibox parameter
            t_samp --> float, sampling time
            n_f --> int, number of frequency samples to bin by (average over)
            n_t --> int, number of time samples to bin by (average over)
            freq_test --> list, frequency axes for each stokes parameter
            n_off --> int, specifies index of end of off-pulse samples
            plot --> bool, if True, outputs plots of time series into datadir
            datadir --> str, path to directory to output images
            label --> str, trigname_nickname of FRB, e.g. '220319aaeb_Mark'
            calstr --> str, string specifying whether given data is calibrated or not, optional
            ext --> str, image file extension (png, pdf, etc.)
            show --> bool, if True, displays images with matplotlib
            normalize --> bool, if True subtracts off-pulse mean on each channel and divides by off-pulse standard deviation
            buff --> int, number of samples to buffer either side of pulse if needed
            weighted --> bool, if True, obtains optimal spectrum by weighting by frequency averaged SNR; uses input_weights if non-empty
            n_t_weight --> int, number of time samples to bin time series by for smoothing
            timeaxis --> array-like, time sample array returned e.g. from get_stokes_2D()
            fobj --> filterbank object containing the object header (this can be obtained by reading a separate filterbank file, or using sigpyproc)
            sf_window_weights --> int, width of 3rd order SavGol filter in samples
            input_weights --> array-like, array of weights used to compute the frequency spectrum. If empty, manually computes weights with get_weights()
    Outputs: (I_f,Q_f,U_f,V_f) --> 1D frequency spectra of each stokes parameter, I,Q,U,V respectively


    """
    #use full timestream for calibrators
    if width_native == -1 or weighted:
        timestart = 0
        timestop = I.shape[1]
    else:
        peak,timestart,timestop = find_peak(I,width_native,t_samp,n_t,buff=buff)

    #I_copy = copy.deepcopy(I)
    #Q_copy = copy.deepcopy(Q)
    #U_copy = copy.deepcopy(U)
    #V_copy = copy.deepcopy(V)
    
    #optimal weighting
    if weighted:
        #I,Q,U,V = weight_spectra_2D(I,Q,U,V,width_native,t_samp,n_f,n_t,freq_test,timeaxis,fobj,n_off=n_off,buff=buff,n_t_weight=n_t_weight,sf_window_weights=sf_window_weights)
        """
        I_t_weights = get_weights(I,Q,U,V,width_native,t_samp,n_f,n_t,freq_test,timeaxis,fobj=fobj,n_off=n_off,buff=buff,n_t_weight=n_t_weight,sf_window_weights=sf_window_weights)
        
        I_rolled = np.roll(I,-peak,axis=1)
        Q_rolled = np.roll(Q,-peak,axis=1)
        U_rolled = np.roll(U,-peak,axis=1)
        V_rolled = np.roll(V,-peak,axis=1)
        I_t_weights_rolled = np.roll(I_t_weights,-np.argmax(I_t_weights))

        Iwr = convolve1d(I_rolled,I_t_weights_rolled,axis=1,mode="constant",cval=0)
        Qwr = convolve1d(Q_rolled,I_t_weights_rolled,axis=1,mode="constant",cval=0)
        Uwr = convolve1d(U_rolled,I_t_weights_rolled,axis=1,mode="constant",cval=0)
        Vwr = convolve1d(V_rolled,I_t_weights_rolled,axis=1,mode="constant",cval=0)
        I = np.roll(Iwr,peak,axis=1)
        Q = np.roll(Qwr,peak,axis=1)
        U = np.roll(Uwr,peak,axis=1)
        V = np.roll(Vwr,peak,axis=1)
        """
        if input_weights == []:
            I_t_weights=get_weights(I,Q,U,V,width_native,t_samp,n_f,n_t,freq_test,timeaxis,fobj,n_off=n_off,buff=buff,n_t_weight=n_t_weight,sf_window_weights=sf_window_weights)
        else:   
            I_t_weights = input_weights


        """    
        I_t_weights_2D = np.array([I_t_weights]*I.shape[0])        

        I = I*I_t_weights_2D
        Q = Q*I_t_weights_2D
        U = U*I_t_weights_2D
        V = V*I_t_weights_2D
        """
        #more efficient weighted average

        nzero = np.nonzero(I_t_weights)

        I_f = np.average(I[:,nzero][:,0,:],weights=I_t_weights[nzero],axis=1)
        Q_f = np.average(Q[:,nzero][:,0,:],weights=I_t_weights[nzero],axis=1)
        U_f = np.average(U[:,nzero][:,0,:],weights=I_t_weights[nzero],axis=1)
        V_f = np.average(V[:,nzero][:,0,:],weights=I_t_weights[nzero],axis=1)

        if normalize:
            """
            I_f = (I[:,timestart:timestop].sum(1) - I_copy[:,:n_off].mean(1))/np.std(np.mean(I_copy[:,:n_off],axis=0))#I[:,:n_off].std(1)
            Q_f = (Q[:,timestart:timestop].sum(1) - Q_copy[:,:n_off].mean(1))/np.std(np.mean(Q_copy[:,:n_off],axis=0))#Q[:,:n_off].std(1)
            U_f = (U[:,timestart:timestop].sum(1) - U_copy[:,:n_off].mean(1))/np.std(np.mean(U_copy[:,:n_off],axis=0))#U[:,:n_off].std(1)
            V_f = (V[:,timestart:timestop].sum(1) - V_copy[:,:n_off].mean(1))/np.std(np.mean(V_copy[:,:n_off],axis=0))#V[:,:n_off].std(1)
            """
            """
            I_f = (I[:,timestart:timestop].sum(1) - np.mean(I_copy[:,:n_off].mean(0)))/np.std(np.mean(I_copy[:,:n_off],axis=0))#I[:,:n_off].std(1)
            Q_f = (Q[:,timestart:timestop].sum(1) - np.mean(Q_copy[:,:n_off].mean(0)))/np.std(np.mean(Q_copy[:,:n_off],axis=0))#Q[:,:n_off].std(1)
            U_f = (U[:,timestart:timestop].sum(1) - np.mean(U_copy[:,:n_off].mean(0)))/np.std(np.mean(U_copy[:,:n_off],axis=0))#U[:,:n_off].std(1)
            V_f = (V[:,timestart:timestop].sum(1) - np.mean(V_copy[:,:n_off].mean(0)))/np.std(np.mean(V_copy[:,:n_off],axis=0))#V[:,:n_off].std(1)
            """
            I_f = (I_f - np.mean(I[:,:n_off].mean(0)))/np.std(np.mean(I[:,:n_off],axis=0))#I[:,:n_off].std(1)
            Q_f = (Q_f - np.mean(Q[:,:n_off].mean(0)))/np.std(np.mean(Q[:,:n_off],axis=0))#Q[:,:n_off].std(1)
            U_f = (U_f - np.mean(U[:,:n_off].mean(0)))/np.std(np.mean(U[:,:n_off],axis=0))#U[:,:n_off].std(1)
            V_f = (V_f - np.mean(V[:,:n_off].mean(0)))/np.std(np.mean(V[:,:n_off],axis=0))#V[:,:n_off].std(1)


        
        #else:
        #    I_f = I[:,timestart:timestop].sum(1)
        #    Q_f = Q[:,timestart:timestop].sum(1)
        #    U_f = U[:,timestart:timestop].sum(1)
        #    V_f = V[:,timestart:timestop].sum(1)
    
        
        


    else:
        if normalize:
            I_f = (I[:,timestart:timestop].mean(1) - np.mean(I[:,:n_off].mean(0)))/np.std(np.mean(I[:,:n_off],axis=0))#I[:,:n_off].std(1)
            Q_f = (Q[:,timestart:timestop].mean(1) - np.mean(Q[:,:n_off].mean(0)))/np.std(np.mean(Q[:,:n_off],axis=0))#Q[:,:n_off].std(1)
            U_f = (U[:,timestart:timestop].mean(1) - np.mean(U[:,:n_off].mean(0)))/np.std(np.mean(U[:,:n_off],axis=0))#U[:,:n_off].std(1)
            V_f = (V[:,timestart:timestop].mean(1) - np.mean(V[:,:n_off].mean(0)))/np.std(np.mean(V[:,:n_off],axis=0))#V[:,:n_off].std(1)
            """
            I_f = (I[:,timestart:timestop].mean(1) - I_copy[:,:n_off].mean(1))/np.std(np.mean(I_copy[:,:n_off],axis=0))#I[:,:n_off].std(1)
            Q_f = (Q[:,timestart:timestop].mean(1) - Q_copy[:,:n_off].mean(1))/np.std(np.mean(Q_copy[:,:n_off],axis=0))#Q[:,:n_off].std(1)
            U_f = (U[:,timestart:timestop].mean(1) - U_copy[:,:n_off].mean(1))/np.std(np.mean(U_copy[:,:n_off],axis=0))#U[:,:n_off].std(1)
            V_f = (V[:,timestart:timestop].mean(1) - V_copy[:,:n_off].mean(1))/np.std(np.mean(V_copy[:,:n_off],axis=0))#V[:,:n_off].std(1)
            """
        else:
            I_f = I[:,timestart:timestop].mean(1)
            Q_f = Q[:,timestart:timestop].mean(1)
            U_f = U[:,timestart:timestop].mean(1)
            V_f = V[:,timestart:timestop].mean(1)

    #if weighted:
    #    I_f = I_f*np.sum(I_t_weights[0,timestart:timestop])
    #    Q_f = Q_f*np.sum(Q_t_weights[0,timestart:timestop])
    #    U_f = U_f*np.sum(U_t_weights[0,timestart:timestop])
    #    V_f = V_f*np.sum(V_t_weights[0,timestart:timestop])

    if plot:
        f=plt.figure(figsize=(12,6))
        plt.plot(freq_test[0],I_f,label="I")
        plt.plot(freq_test[1],Q_f,label="Q")
        plt.plot(freq_test[2],U_f,label="U")
        plt.plot(freq_test[3],V_f,label="V")
        plt.grid()
        plt.xlabel("frequency (MHz)")
        #plt.xlim(1365,1375)
        plt.legend()
        plt.title(label)
        #plt.title(label)
        plt.savefig(datadir + label + "_frequency_" + calstr + str(n_f) + "_binned" + ext)
        if show:
            plt.show()
        plt.close(f)

    return (I_f,Q_f,U_f,V_f)


#Plot dynamic spectra, I Q U V
def plot_spectra_2D(I,Q,U,V,width_native,t_samp,n_t,n_f,freq_test,n_off=3000,datadir=DEFAULT_DATADIR,label='',calstr='',ext=ext,window=10,lim=500,show=False,buff=0,weighted=False,n_t_weight=1,timeaxis=None,fobj=None,sf_window_weights=45,cmap='viridis'):
    """
    This function plots the given dynamic spectra and outputs images in the specified directory. 
    The spectra are normalized by subtracting the time average in each frequency channel.
    Inputs: I,Q,U,V --> 2D arrays, dynamic spectra of I,Q,U,V generated with get_stokes_2D()
            width_native --> int, width in samples of pulse in native sampling rate; equivalent to ibox parameter
            t_samp --> float, sampling time
            n_t --> int, number of time samples spectra have been binned by (average over)
            n_f --> int, number of frequency samples spectra have been binned by (average over)
            freq_test --> list, frequency axes for each stokes parameter
            datadir --> str, path to directory to output images
            label --> str, trigname_nickname of FRB, e.g. '220319aaeb_Mark'
            calstr --> str, string specifying whether given data is calibrated or not, optional
            ext --> str, image file extension (png, pdf, etc.)
            window --> int, number of time samples on either side of pulse to plot
            lim --> int, absolute value of max and min of range to plot
            show --> bool, if True, displays images with matplotlib
            buff --> int, number of samples to buffer either side of pulse if needed

    """
    if width_native == -1:
        timestart = 0
        timestop = I.shape[1]
        window = 0
    else:
        peak,timestart,timestop = find_peak(I,width_native,t_samp,n_t,buff=buff)    

    #optimal weighting
    if weighted:
        print("weighting...")
        """
        I_t_weights = get_weights(I,Q,U,V,width_native,t_samp,n_f,n_t,freq_test,timeaxis,fobj=fobj,n_off=n_off,buff=buff,n_t_weight=n_t_weight,sf_window_weights=sf_window_weights)

        I_rolled = np.roll(I,-np.argmax(I.mean(0)),axis=1)
        Q_rolled = np.roll(Q,-np.argmax(Q.mean(0)),axis=1)
        U_rolled = np.roll(U,-np.argmax(U.mean(0)),axis=1)
        V_rolled = np.roll(V,-np.argmax(V.mean(0)),axis=1)
        I_t_weights_rolled = np.roll(I_t_weights,-np.argmax(I_t_weights))

        I = convolve1d(I_rolled,I_t_weights_rolled,axis=1,mode="constant",cval=0)
        Q = convolve1d(Q_rolled,I_t_weights_rolled,axis=1,mode="constant",cval=0)
        U = convolve1d(U_rolled,I_t_weights_rolled,axis=1,mode="constant",cval=0)
        V = convolve1d(V_rolled,I_t_weights_rolled,axis=1,mode="constant",cval=0)
        """
        I,Q,U,V = weight_spectra_2D(I,Q,U,V,width_native,t_samp,n_f,n_t,freq_test,timeaxis,fobj,n_off=n_off,buff=buff,n_t_weight=n_t_weight,sf_window_weights=sf_window_weights)
        print("Done weighting!")

    #Dynamic Spectra
    f=plt.figure(figsize=(25,15))
    pylab.subplot(2,2,1)
    plt.imshow(I - np.mean(I,1,keepdims=True),aspect="auto",vmin=-lim,vmax=lim,cmap=cmap)
    plt.xlim(timestart-window,timestop+window)
    plt.title(label + " I")
    plt.colorbar()
    plt.xlabel("Time Sample (" + str(t_samp*n_t) + " s sampling time)")
    plt.ylabel("frequency sample")

    pylab.subplot(2,2,2)
    plt.imshow(Q - np.mean(Q,1,keepdims=True),aspect="auto",vmin=-lim,vmax=lim,cmap=cmap)
    plt.xlim(timestart-window,timestop+window)
    plt.title(label + " Q")
    plt.colorbar()
    plt.xlabel("Time Sample (" + str(t_samp*n_t) + " s sampling time)")
    plt.ylabel("frequency sample")

    pylab.subplot(2,2,3)
    plt.imshow(U - np.mean(U,1,keepdims=True),aspect="auto",vmin=-lim,vmax=lim,cmap=cmap)
    plt.xlim(timestart-window,timestop+window)
    plt.title(label + " U")
    plt.colorbar()
    plt.xlabel("Time Sample (" + str(t_samp*n_t) + " s sampling time)")
    plt.ylabel("frequency sample")

    pylab.subplot(2,2,4)
    plt.imshow(V - np.mean(V,1,keepdims=True),aspect="auto",vmin=-lim,vmax=lim,cmap=cmap)
    plt.xlim(timestart-window,timestop+window)
    plt.title(label + " V")
    plt.colorbar()
    plt.xlabel("Time Sample (" + str(t_samp*n_t) + " s sampling time)")
    plt.ylabel("frequency sample")

    plt.savefig(datadir + label + "_freq-time_" + calstr + str(n_f) + "_binned" + ext)
    if show:
        plt.show()
    plt.close(f)
    #print("yo")
    return

def plot_spectra_1D(I_f,Q_f,U_f,V_f,n_f,freq_test,datadir=DEFAULT_DATADIR,label='',calstr='',ext=ext,show=False):
    """
    This function plots the given time averaged spectra and outputs images in the specified directory.
    Inputs: I_f,Q_f,U_f,V_f --> 1D array, time averaged spectra of I,Q,U,V generated with get_stokes_vs_freq()
            n_f --> int, number of frequency samples spectra have been binned by (average over)
            freq_test --> list, frequency axes for each stokes parameter
            datadir --> str, path to directory to output images
            label --> str, trigname_nickname of FRB, e.g. '220319aaeb_Mark'
            calstr --> str, string specifying whether given data is calibrated or not, optional
            ext --> str, image file extension (png, pdf, etc.)
            show --> bool, if True, displays images with matplotlib

    """
    f=plt.figure(figsize=(12,6))
    plt.plot(freq_test[0],I_f,label="I")
    plt.plot(freq_test[1],Q_f,label="Q")
    plt.plot(freq_test[2],U_f,label="U")
    plt.plot(freq_test[3],V_f,label="V")
    plt.grid()
    plt.xlabel("frequency (MHz)")
    #plt.xlim(1365,1375)
    plt.legend()
    plt.title(label)
    #plt.title(label)
    plt.savefig(datadir + label + "_frequency_" + calstr + str(n_f) + "_binned" + ext)
    if show:
        plt.show()
    plt.close(f)

def plot_timestream_1D(I_t,Q_t,U_t,V_t,width_native,t_samp,n_t,datadir='',label='',calstr='',ext=ext,show=False,buff=0):
    """
    This function plots the given frequency averaged time series and outputs images in the specified directory.
    The time axis is constrained to the region around the pulse calculated with find_peak().
    Inputs: I_t,Q_t,U_t,V_t --> 1D arrays, frequency averaged time series of I,Q,U,V generated with get_stokes_vs_time()
            width_native --> int, width in samples of pulse in native sampling rate; equivalent to ibox parameter
            t_samp --> float, sampling time
            n_t --> int, number of time samples spectra have been binned by (average over)
            datadir --> str, path to directory to output images
            label --> str, trigname_nickname of FRB, e.g. '220319aaeb_Mark'
            calstr --> str, string specifying whether given data is calibrated or not, optional
            ext --> str, image file extension (png, pdf, etc.)
            show --> bool, if True, displays images with matplotlib
            buff --> int, number of samples to buffer either side of pulse if needed

    """
    if width_native == -1:
        timestart = 0
        timestop = I.shape[1]
    else:
        peak,timestart,timestop = find_peak(I,width_native,t_samp,n_t,buff=buff)


    f=plt.figure(figsize=(12,6))
    plt.plot(I_t,label="I")
    plt.plot(Q_t,label="Q")
    plt.plot(U_t,label="U")
    plt.plot(V_t,label="V")
    plt.grid()
    plt.legend()
    plt.xlim(timestart,timestop)
    plt.xlabel("Time Sample (" + str(t_samp*n_t) + " s sampling time)")
    plt.title(label)
    plt.savefig(datadir +label + "_time_"+ calstr + str(n_t) + "_binned" + ext)
    if show:
        plt.show()
    plt.close(f)



#Convert width from 256 us samples to fil file native sampling rate (sampling time in seconds)
def convert_width(width_native,t_samp,n_t):
    """
    This function converts the given pulse width in 256 us samples to samples in 
    the fil file native sampling rate given
    Inputs: width_native --> int, width in samples of pulse in native sampling rate; equivalent to ibox parameter
            t_samp --> float, sampling time
            n_t --> int, number of time samples binned by
    Outputs: width --> int, width in samples of given sample rate
    """
    return int(np.ceil(width_native*(256e-6)/(n_t*t_samp)))

#Find peak, start sample, stop sample from I (frequency averaged)
def find_peak(I,width_native,t_samp,n_t,peak_range=None,pre_calc_tf=False,buff=0):
    """
    This function finds the peak sample in the intensity dynamic spectrum and the 
    start and end sample such that the SNR within the start and end is maximized.
    Inputs: I --> 2D array or 1D array, dynamic spectrum of I generated with get_stokes_2D() or 1D spectrum generated with get_stokes_vs_freq()
            width_native --> int, width in samples of pulse in native sampling rate; equivalent to ibox parameter
            t_samp --> float, sampling time
            n_t --> int, number of time samples binned by
            peak_range --> tuple, optionally specify time sample range in which to search for the peak
            pre_calc_tf --> bool, set if I is frequency averaged time series
            buff --> int, number of samples to buffer either side of pulse if needed
    Outpus: peak --> int, time sample location of peak
            timestart --> int, time sample at start of pulse
            timestop --> int, time sample at end of pulse
    """
    if isinstance(buff, int):
        buff1 = buff
        buff2 = buff
    else:
        buff1 = buff[0]
        buff2 = buff[1]

    #get width
    width = convert_width(width_native,t_samp,n_t)
    #frequency averaged
    if pre_calc_tf:
        I_t = I[0]
    else:
        I_t = I.mean(axis=0)
    if peak_range != None:
        peak = np.argmax(I_t[peak_range[0]:peak_range[1]]) + peak_range[0]
    else:
        peak = np.argmax(I_t)

    snr = np.zeros(width)
    for i in range(width):
        snr[i] = np.sum(I_t[peak-i:peak + width - i])

    offset = np.argmax(snr)
    timestart = peak-offset-buff1
    timestop = peak+width-offset+buff2+1
    return (peak, timestart, timestop)

#Calculate polarization angle vs frequency and time from 2D I Q U V arrays
def get_pol_fraction(I,Q,U,V,width_native,t_samp,n_t,n_f,freq_test,n_off=3000,plot=False,datadir=DEFAULT_DATADIR,label='',calstr='',ext=ext,pre_calc_tf=False,show=False,normalize=True,buff=0,full=False,weighted=False,n_t_weight=1,timeaxis=None,fobj=None,sf_window_weights=45,multipeaks=False,height=0.03,window=30,unbias=True,input_weights=[],allowed_err=1,unbias_factor=1,intL=None,intR=None):
    """
    This function calculates and plots the polarization fraction averaged over both time and 
    frequency, the total an polarized signal-to-noise, and the average polarization fraction within the peak.
    Inputs: I,Q,U,V --> 2D arrays, dynamic spectra of I,Q,U,V generated with get_stokes_2D()
            width_native --> int, width in samples of pulse in native sampling rate; equivalent to ibox parameter
            t_samp --> float, sampling time
            n_t --> int, number of time samples spectra have been binned by (average over)
            n_f --> int, number of frequency samples spectra have been binned by (average over)
            freq_test --> list, frequency axes for each stokes parameter
            n_off --> int, specifies index of end of off-pulse samples
            plot --> bool, set to plot and output images in specified directory
            datadir --> str, path to directory to output images
            label --> str, trigname_nickname of FRB, e.g. '220319aaeb_Mark'
            calstr --> str, string specifying whether given data is calibrated or not, optional
            pre_calc_tf --> bool, set if I,Q,U,V are given as tuples of the frequency and time averaged arrays, e.g. (I_t,I_f)
            ext --> str, image file extension (png, pdf, etc.)
            show --> bool, if True, displays images with matplotlib
            buff --> int, number of samples to buffer either side of pulse if needed
            full --> bool, if True, calculates polarization vs frequency and time before averaging
            weighted --> bool, if True, computes weighted average polarization fraction; if False, computes unweighted average
            n_t_weight --> int, number of time samples to bin time series by for smoothing
            timeaxis --> array-like, time sample array returned e.g. from get_stokes_2D()
            fobj --> filterbank object containing the object header (this can be obtained by reading a separate filterbank file, or using sigpyproc)
            sf_window_weights --> int, width of 3rd order SavGol filter in samples
            multipeaks --> bool, set True if sub-burst has multiple peaks, pol fraction will be computed between lower bound of first peak and upper bound of last peak
            height --> float, minimum height of sub-burst peak if multipeaks is True, default 0.03
            window --> int, window in samples around pulse to plot
            unbias --> bool, if True, unbiases linear polarization according to Simmons & Stewart, default True
            input_weights --> array-like, array of weights used to compute the frequency spectrum. If empty, manually computes weights with get_weights()
            allowed_err --> float, fractional overpolarization allowed, default 100%
            unbias_factor --> float, unbiasing offset term applied if unbias=True
    Outputs:pol_f --> 1D array, frequency dependent total polarization
            pol_t --> 1D array, time dependent total polarization
            avg_frac --> float, frequency and time averaged total polarization fraction
            sigma_frac --> float, error in average total polarization fraction
            snr_frac --> float, total polarization signal-to-noise
            L_f --> 1D array, frequency dependent linear polarization
            L_t --> 1D array, time dependent linear polarization
            avg_L --> float, frequency and time averaged linear polarization fraction
            sigma_L --> float, error in average linear polarization fraction
            snr_L --> float, linear polarization signal-to-noise
            C_f --> 1D array, frequency dependent circular polarization
            C_t --> 1D array, time dependent circular polarization
            avg_C_abs --> float, frequency and time averaged absolute value circular polarization fraction
            sigma_C_abs --> float, error in average absolute value circular polarization fraction
            snr_C --> float, circular polarization signal-to-noise
            snr --> float, intensity signal-to-noise
            outputs are returned as a list in the format: [(pol_f,pol_t,avg_frac,sigma_frac,snr_frac),(L_f,L_t,avg_L,sigma_L,snr_L),(C_f,C_t,avg_C_abs,sigma_C_abs,snr_C),(C_f,C_t,avg_C,sigma_C,snr_C),snr]
    """
    if isinstance(buff, int):
        buff1 = buff
        buff2 = buff
    else:
        buff1 = buff[0]
        buff2 = buff[1]

    if pre_calc_tf:
        (I_t,I_f) = I
        (Q_t,Q_f) = Q
        (U_t,U_f) = U
        (V_t,V_f) = V
    else:
        (I_t,Q_t,U_t,V_t) = get_stokes_vs_time(I,Q,U,V,width_native,t_samp,n_t,n_off=n_off,plot=False,normalize=normalize,buff=buff,window=window)
        (I_f,Q_f,U_f,V_f) = get_stokes_vs_freq(I,Q,U,V,width_native,t_samp,n_f,n_t,freq_test,n_off=n_off,plot=False,normalize=normalize,buff=buff,weighted=weighted,n_t_weight=n_t_weight,timeaxis=timeaxis,fobj=fobj,sf_window_weights=sf_window_weights)
    
    #if input_weights != []:
    #    timestart = 0
    #    timestop = len(I_t)
    if width_native == -1:#use full timestream for calibrators
        timestart = 0
        timestop = len(I_t)#I.shape[1]
    else:
        peak,timestart,timestop = find_peak(I,width_native,t_samp,n_t,pre_calc_tf=pre_calc_tf,buff=buff)

    if full:
        #linear polarization
        L = np.sqrt(Q**2 + U**2)
        
        if unbias:
            L_f = np.nanmean(L,axis=1)
            L_t = np.nanmean(L,axis=0)
            L_t[L_t**2 <= (unbias_factor*np.std(I_t[:n_off]))**2] = np.std(I_t[:n_off])
            L_t = np.sqrt(L_t**2 - np.std(I_t[:n_off])**2)
            L_f[L_f**2 <= (unbias_factor*np.std(I_t[:n_off]))**2] = np.std(I_t[:n_off])
            L_f = np.sqrt(L_f**2 - np.std(I_t[:n_off])**2)
            L_f = L_f#/I_f
            L_t = L_t#/I_t
        else:
            L_f = np.nanmean(L,axis=1)#/I_f
            L_t = np.nanmean(L,axis=0)#/I_t
        
        #circular polarization
        C_f = np.nanmean(V,axis=1)#/I_f
        C_t = np.nanmean(V,axis=0)#/I_t


        #total polarization
        pol_t = np.sqrt(L_t**2 + C_t**2)/I_t
        pol_f = np.sqrt(L_f**2 + C_f**2)/I_f

        C_t = C_t/I_t
        C_f = C_f/I_f

        L_t = L_t/I_t
        L_f = L_f/I_f


    else:
        #linear polarization
        if unbias:
            L_f = np.sqrt((np.array(Q_f)**2 + np.array(U_f)**2))
            L_t = np.sqrt((np.array(Q_t)**2 + np.array(U_t)**2))

            L_t[L_t**2 <= (unbias_factor*np.std(I_t[:n_off]))**2] = np.std(I_t[:n_off])
            L_t = np.sqrt(L_t**2 - np.std(I_t[:n_off])**2)
            L_f[L_f**2 <= (unbias_factor*np.std(I_t[:n_off]))**2] = np.std(I_t[:n_off])
            L_f = np.sqrt(L_f**2 - np.std(I_t[:n_off])**2)
            L_f = L_f#/I_f
            L_t = L_t#/I_t
        else:
            L_f = np.sqrt((np.array(Q_f)**2 + np.array(U_f)**2))#/I_f#(np.array(I_f)**2))
            L_t = np.sqrt((np.array(Q_t)**2 + np.array(U_t)**2))#/I_t#(np.array(I_t)**2))

        #circular polarization
        C_f = V_f#/I_f
        C_t = V_t#/I_t

        #total polarization
        pol_f = np.sqrt(C_f**2 + L_f**2)/I_f
        pol_t = np.sqrt(C_t**2 + L_t**2)/I_t

        C_f = C_f/I_f
        C_t = C_t/I_t

        L_f = L_f/I_f
        L_t = L_t/I_t

    if plot:
        f=plt.figure(figsize=(12,6))
        plt.plot(freq_test[0],pol_f,label=r'Time Averaged Total Polarization ($\sqrt{Q^2 + U^2 + V^2}/I$)')
        plt.plot(freq_test[0],L_f,label=r'Time Averaged Linear Polarization ($\sqrt{Q^2 + U^2}/I$)')
        plt.plot(freq_test[0],C_f,label=r'Time Averaged Circular Polarization ($V/I$)')
        plt.grid()
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Polarization Fraction")
        plt.ylim(-1-allowed_err,1+allowed_err)
        plt.title(label)
        plt.legend()
        plt.savefig(datadir + label + "_polfraction_frequency_"  + calstr  + str(n_f) + "_binned" + ext)
        if show:
            plt.show()
        plt.close(f)
    
        f=plt.figure(figsize=(12,6))
        plt.plot(np.arange(timestart,timestop),pol_t[timestart:timestop],label=r'Frequency Averaged Total Polarization ($\sqrt{Q^2 + U^2 + V^2}/I$)')
        plt.plot(np.arange(timestart,timestop),L_t[timestart:timestop],label=r'Frequency Averaged Linear Polarization ($\sqrt{Q^2 + U^2}/I$)')
        plt.plot(np.arange(timestart,timestop),C_t[timestart:timestop],label=r'Frequency Averaged Circular Polarization ($V/I$)')
        plt.grid()
        plt.xlabel("Time Sample (" + str(t_samp*n_t) + " s sampling time)")
        plt.ylabel("Polarization Fraction")
        plt.ylim(-1-allowed_err,1+allowed_err)
        plt.title(label)
        plt.legend()
        plt.savefig(datadir + label + "_polfraction_time_" + calstr + str(n_t) + "_binned" + ext)
        if show:
            plt.show()
        plt.close(f)

    #allowed_err = 0.1 #allowed error above 100 %
    #avg_frac = (np.mean(pol_t[timestart:timestop][pol_t[timestart:timestop]<1]))
    #avg_frac = (np.mean(pol_t[timestart:timestop]))
    

    if weighted:
        if input_weights == []:
            I_t_weights=get_weights(I,Q,U,V,width_native,t_samp,n_f,n_t,freq_test,timeaxis,fobj,n_off=n_off,buff=buff,n_t_weight=n_t_weight,sf_window_weights=sf_window_weights)
            I_t_weights_unpadded = get_weights(I,Q,U,V,width_native,t_samp,n_f,n_t,freq_test,timeaxis,fobj,n_off=n_off,buff=buff,n_t_weight=n_t_weight,sf_window_weights=sf_window_weights,padded=False)
        else:
            I_t_weights = input_weights
            I_t_weights_unpadded = np.trim_zeros(input_weights)


        if multipeaks and (intL == None or intR == None):
            pks,props = find_peaks(I_t_weights,height=height)
            FWHM,heights,intL,intR = peak_widths(I_t_weights,pks)
            intL = intL[0]
            intR = intR[-1]
        elif (intL == None or intR == None):
            FWHM,heights,intL,intR = peak_widths(I_t_weights,[np.argmax(I_t_weights)])
        
        intL = int(intL)
        intR = int(intR)

        #average,error
        off_pulse_I = np.nanstd(I_t[:n_off])
        off_pulse_Q = np.nanstd(Q_t[:n_off])
        off_pulse_U = np.nanstd(U_t[:n_off])
        off_pulse_V = np.nanstd(V_t[:n_off])
        L_t_biased = np.sqrt(Q_t**2 + U_t**2)
        T_t_biased = np.sqrt(Q_t**2 + U_t**2  + V_t**2)



        pol_t_cut1 = ((pol_t)[intL:intR])
        pol_t_cut = pol_t_cut1[np.abs(pol_t_cut1) < 1+allowed_err]
        I_t_weights_pol_cut = I_t_weights[intL:intR]
        I_t_weights_pol_cut = I_t_weights_pol_cut[np.abs(pol_t_cut1) < 1+allowed_err]
        avg_frac = np.nansum(pol_t_cut*I_t_weights_pol_cut)/np.nansum(I_t_weights_pol_cut)
        #sigma_frac = np.nansum(I_t_weights_pol_cut*np.sqrt((pol_t_cut - avg_frac)**2))/np.nansum(I_t_weights_pol_cut)
        sigma_frac = np.sqrt((Q_t*off_pulse_Q/I_t/T_t_biased)**2 +   (U_t*off_pulse_U/I_t/T_t_biased)**2+ (V_t*off_pulse_V/I_t/T_t_biased)**2  + (T_t_biased*off_pulse_I/((I_t)**2)))
        sigma_frac = (sigma_frac[intL:intR])[np.abs(pol_t_cut1) < 1+allowed_err]
        sigma_frac = np.sqrt(np.nansum((I_t_weights_pol_cut*sigma_frac)**2))/np.nansum(I_t_weights_pol_cut)

        L_t_cut1 = ((L_t)[intL:intR])
        L_t_cut = L_t_cut1[np.abs(pol_t_cut1) < 1+allowed_err]
        I_t_weights_L_cut = I_t_weights[intL:intR]
        I_t_weights_L_cut = I_t_weights_L_cut[np.abs(pol_t_cut1) < 1+allowed_err]
        avg_L = np.nansum(L_t_cut*I_t_weights_L_cut)/np.nansum(I_t_weights_L_cut)
        #sigma_L = np.nansum(I_t_weights_L_cut*np.sqrt((L_t_cut - avg_L)**2))/np.nansum(I_t_weights_L_cut)
        sigma_L = np.sqrt((Q_t*off_pulse_Q/I_t/L_t_biased)**2 +   (U_t*off_pulse_U/I_t/L_t_biased)**2  + (L_t_biased*off_pulse_I/((I_t)**2)))
        sigma_L = (sigma_L[intL:intR])[np.abs(pol_t_cut1) < 1+allowed_err]
        sigma_L = np.sqrt(np.nansum((I_t_weights_pol_cut*sigma_L)**2))/np.nansum(I_t_weights_pol_cut)


        C_t_cut1 = (np.abs(C_t)[intL:intR])
        C_t_cut = C_t_cut1[np.abs(pol_t_cut1) < 1+allowed_err]
        I_t_weights_C_cut = I_t_weights[intL:intR]
        I_t_weights_C_cut = I_t_weights_C_cut[np.abs(pol_t_cut1) < 1+allowed_err]
        avg_C_abs = np.nansum(C_t_cut*I_t_weights_C_cut)/np.nansum(I_t_weights_C_cut)
        #sigma_C_abs = np.nansum(I_t_weights_C_cut*np.sqrt((C_t_cut - avg_C_abs)**2))/np.nansum(I_t_weights_C_cut)
        sigma_C_abs = np.sqrt((off_pulse_V/I_t)**2  + (V_t*off_pulse_I/((I_t)**2))**2)
        sigma_C_abs = (sigma_C_abs[intL:intR])[np.abs(pol_t_cut1) < 1+allowed_err]
        sigma_C_abs = np.sqrt(np.nansum((I_t_weights_pol_cut*sigma_C_abs)**2))/np.nansum(I_t_weights_pol_cut)

        
        C_t_cut1 = ((C_t)[intL:intR])
        C_t_cut = C_t_cut1[np.abs(pol_t_cut1) < 1+allowed_err]
        I_t_weights_C_cut = I_t_weights[intL:intR]
        I_t_weights_C_cut = I_t_weights_C_cut[np.abs(pol_t_cut1) < 1+allowed_err]
        avg_C = np.nansum(C_t_cut*I_t_weights_C_cut)/np.nansum(I_t_weights_C_cut)
        sigma_C = np.nansum(I_t_weights_C_cut*np.sqrt((C_t_cut - avg_C)**2))/np.nansum(I_t_weights_C_cut)
        sigma_C = sigma_C_abs
        
        I_trial_binned = convolve(I_t,I_t_weights_unpadded)
        sigbin = np.argmax(I_trial_binned)
        sig0 = I_trial_binned[sigbin]
        I_binned = convolve(I_t,I_t_weights_unpadded)
        noise = np.std(np.concatenate([I_binned[:sigbin-(timestop-timestart)*2],I_binned[sigbin+(timestop-timestart)*2:]]))
        snr = sig0/noise
        #print((sig0,noise,snr))

        T_trial_binned = convolve(pol_t*I_t,I_t_weights_unpadded)
        sig0 = T_trial_binned[sigbin]
        Q_binned = convolve(Q_t,I_t_weights_unpadded)
        noise = np.std(np.concatenate([Q_binned[:sigbin-(timestop-timestart)*2],Q_binned[sigbin+(timestop-timestart)*2:]]))
        snr_frac = sig0/noise
        #print((sig0,noise,snr_frac))

        L_trial_binned = convolve(L_t*I_t,I_t_weights_unpadded)
        sig0 = L_trial_binned[sigbin]
        snr_L = sig0/noise
        #print((sig0,noise,snr_L))

        C_trial_binned = np.convolve(np.abs(C_t)*I_t,I_t_weights_unpadded)
        sig0 = C_trial_binned[sigbin]
        V_binned = convolve(V_t,I_t_weights_unpadded)
        noise = np.std(np.concatenate([V_binned[:sigbin-(timestop-timestart)*2],V_binned[sigbin+(timestop-timestart)*2:]]))
        snr_C = sig0/noise
        #print((sig0,noise,snr_C))

        

    else:
        if input_weights == []:
            I_t_weights=get_weights(I,Q,U,V,width_native,t_samp,n_f,n_t,freq_test,timeaxis,fobj,n_off=n_off,buff=buff,n_t_weight=n_t_weight,sf_window_weights=sf_window_weights)
            I_t_weights_unpadded = get_weights(I,Q,U,V,width_native,t_samp,n_f,n_t,freq_test,timeaxis,fobj,n_off=n_off,buff=buff,n_t_weight=n_t_weight,sf_window_weights=sf_window_weights,padded=False)
        else:
            I_t_weights = input_weights
            I_t_weights_unpadded = np.trim_zeros(input_weights)


        if multipeaks and (intL == None or intR == None):
            pks,props = find_peaks(I_t,height=height)
            FWHM,heights,intL,intR = peak_widths(I_t,pks)
            intL = intL[0]
            intR = intR[-1]
        elif (intL == None or intR == None):
            FWHM,heights,intL,intR = peak_widths(I_t,[np.argmax(I_t)])
        
        intL = int(intL)
        intR = int(intR)

        #average,error
        off_pulse_I = np.nanstd(I_t[:n_off])
        off_pulse_Q = np.nanstd(Q_t[:n_off])
        off_pulse_U = np.nanstd(U_t[:n_off])
        off_pulse_V = np.nanstd(V_t[:n_off])
        L_t_biased = np.sqrt(Q_t**2 + U_t**2)
        T_t_biased = np.sqrt(Q_t**2 + U_t**2  + V_t**2)

        avg_frac = np.nanmean((pol_t[intL:intR])[np.abs(pol_t)[intL:intR]<=1+allowed_err])
        avg_L = np.nanmean((L_t[intL:intR])[np.abs(L_t)[intL:intR]<=1+allowed_err])
        avg_C_abs = np.nanmean((np.abs(C_t)[intL:intR])[np.abs(C_t)[intL:intR]<=1+allowed_err])
        avg_C = np.nanmean(((C_t)[intL:intR])[(C_t)[intL:intR]<=1+allowed_err])

        #RMS error
        sigma_frac = np.sqrt((Q_t*off_pulse_Q/I_t/T_t_biased)**2 +   (U_t*off_pulse_U/I_t/T_t_biased)**2+ (V_t*off_pulse_V/I_t/T_t_biased)**2  + (T_t_biased*off_pulse_I/((I_t)**2)))
        sigma_frac = (sigma_frac[intL:intR])[np.abs(pol_t)[intL:intR] < 1+allowed_err]
        sigma_frac = np.sqrt(np.nansum((sigma_frac)**2))/len(sigma_frac)

        sigma_L = np.sqrt((Q_t*off_pulse_Q/I_t/L_t_biased)**2 +   (U_t*off_pulse_U/I_t/L_t_biased)**2  + (L_t_biased*off_pulse_I/((I_t)**2)))
        sigma_L = (sigma_L[intL:intR])[np.abs(pol_t)[intL:intR] < 1+allowed_err]
        sigma_L = np.sqrt(np.nansum((sigma_L)**2))/len(sigma_L)

        sigma_C_abs = np.sqrt((off_pulse_V/I_t)**2  + (V_t*off_pulse_I/((I_t)**2))**2)
        sigma_C_abs = (sigma_C_abs[intL:intR])[np.abs(pol_t)[intL:intR] < 1+allowed_err]
        sigma_C_abs = np.sqrt(np.nansum((sigma_C_abs)**2))/len(sigma_C_abs)
        sigma_C = sigma_C_abs

        #SNR
        sig0 = np.nanmean(I_t[timestart:timestop])
        I_t_cut1 = I_t[timestart%(timestop-timestart):]
        I_t_cut = I_t_cut1[:(len(I_t_cut1)-(len(I_t_cut1)%(timestop-timestart)))]
        I_t_binned = I_t_cut.reshape(len(I_t_cut)//(timestop-timestart),timestop-timestart).mean(1)
        sigbin = np.argmax(I_t_binned)
        noise = (np.nanstd(np.concatenate([I_t_cut[:sigbin],I_t_cut[sigbin+1:]])))
        snr = sig0/noise

        sig0 = np.nanmean((pol_t*I_t)[timestart:timestop])
        pol_t_cut1 = (pol_t*I_t)[timestart%(timestop-timestart):]
        pol_t_cut = pol_t_cut1[:(len(pol_t_cut1)-(len(pol_t_cut1)%(timestop-timestart)))]
        pol_t_binned = pol_t_cut.reshape(len(pol_t_cut)//(timestop-timestart),timestop-timestart).mean(1)
        sigbin = np.argmax(pol_t_binned)
        noise = (np.nanstd(np.concatenate([pol_t_cut[:sigbin],pol_t_cut[sigbin+1:]])))
        snr_frac = sig0/noise

        sig0 = np.nanmean((L_t*I_t)[timestart:timestop])
        L_t_cut1 = (L_t*I_t)[timestart%(timestop-timestart):]
        L_t_cut = L_t_cut1[:(len(L_t_cut1)-(len(L_t_cut1)%(timestop-timestart)))]
        L_t_binned = L_t_cut.reshape(len(L_t_cut)//(timestop-timestart),timestop-timestart).mean(1)
        sigbin = np.argmax(L_t_binned)
        noise = (np.nanstd(np.concatenate([L_t_cut[:sigbin],L_t_cut[sigbin+1:]])))
        snr_L = sig0/noise

        sig0 = np.nanmean((np.abs(C_t)*I_t)[timestart:timestop])
        C_t_cut1 = (np.abs(C_t)*I_t)[timestart%(timestop-timestart):]
        C_t_cut = C_t_cut1[:(len(C_t_cut1)-(len(C_t_cut1)%(timestop-timestart)))]
        C_t_binned = C_t_cut.reshape(len(C_t_cut)//(timestop-timestart),timestop-timestart).mean(1)
        sigbin = np.argmax(C_t_binned)
        noise = (np.nanstd(np.concatenate([C_t_cut[:sigbin],C_t_cut[sigbin+1:]])))
        snr_C = sig0/noise



    #snr_frac = np.nanmean(pol_t[timestart:timestop])/np.nanstd(pol_t[:n_off])
    #snr_L = np.nanmean(L_t[timestart:timestop])/np.nanstd(L_t[:n_off])
    #snr_C = np.nanmean(C_t[timestart:timestop])/np.nanstd(C_t[:n_off])

    return [(pol_f,pol_t,avg_frac,sigma_frac,snr_frac),(L_f,L_t,avg_L,sigma_L,snr_L),(C_f,C_t,avg_C_abs,sigma_C_abs,snr_C),(C_f,C_t,avg_C,sigma_C,snr_C),snr]


#Calculate polarization fraction vs time from 2D I Q U V arrays
def get_pol_fraction_vs_time(I,Q,U,V,width_native,t_samp,n_t,n_off=3000,plot=False,datadir=DEFAULT_DATADIR,label='',calstr='',ext=ext,pre_calc_tf=False,show=False,normalize=True,buff=0,full=False,weighted=False,n_t_weight=1,timeaxis=None,fobj=None,sf_window_weights=45,multipeaks=False,height=0.03,window=30,unbias=True,input_weights=[],allowed_err=1,unbias_factor=1):
    """
    This function calculates and plots the polarization fraction averaged over
    frequency, and the average polarization fraction within the peak.
    Inputs: I,Q,U,V --> 2D arrays, dynamic spectra of I,Q,U,V generated with get_stokes_2D()
            width_native --> int, width in samples of pulse in native sampling rate; equivalent to ibox parameter
            t_samp --> float, sampling time
            n_t --> int, number of time samples spectra have been binned by (average over)
            n_off --> int, specifies index of end of off-pulse samples
            plot --> bool, set to plot and output images in specified directory
            datadir --> str, path to directory to output images
            label --> str, trigname_nickname of FRB, e.g. '220319aaeb_Mark'
            calstr --> str, string specifying whether given data is calibrated or not, optional
            pre_calc_tf --> bool, set if I,Q,U,V are given as tuples of the frequency and time averaged arrays, e.g. (I_t,I_f)
            ext --> str, image file extension (png, pdf, etc.)
            show --> bool, if True, displays images with matplotlib
            buff --> int, number of samples to buffer either side of pulse if needed
            full --> bool, if True, calculates polarization vs frequency and time before averaging
            weighted --> bool, if True, computes weighted average polarization fraction; if False, computes unweighted average
            n_t_weight --> int, number of time samples to bin time series by for smoothing
            timeaxis --> array-like, time sample array returned e.g. from get_stokes_2D()
            fobj --> filterbank object containing the object header (this can be obtained by reading a separate filterbank file, or using sigpyproc)
            sf_window_weights --> int, width of 3rd order SavGol filter in samples
            multipeaks --> bool, set True if sub-burst has multiple peaks, pol fraction will be computed between lower bound of first peak and upper bound of last peak
            height --> float, minimum height of sub-burst peak if multipeaks is True, default 0.03
            window --> int, window in samples around pulse to plot
            unbias --> bool, if True, unbiases linear polarization according to Simmons & Stewart, default True
            input_weights --> array-like, array of weights used to compute the frequency spectrum. If empty, manually computes weights with get_weights()
            allowed_err --> float, fractional overpolarization allowed, default 100%
            unbias_factor --> float, unbiasing offset term applied if unbias=True
    Outputs:pol_t --> 1D array, time dependent total polarization
            avg_frac --> float, frequency and time averaged total polarization fraction
            sigma_frac --> float, error in average total polarization fraction
            snr_frac --> float, total polarization signal-to-noise
            L_t --> 1D array, time dependent linear polarization
            avg_L --> float, frequency and time averaged linear polarization fraction
            sigma_L --> float, error in average linear polarization fraction
            snr_L --> float, linear polarization signal-to-noise
            C_t --> 1D array, time dependent circular polarization
            avg_C_abs --> float, frequency and time averaged absolute value circular polarization fraction
            sigma_C_abs --> float, error in average absolute value circular polarization fraction
            snr_C --> float, circular polarization signal-to-noise
            snr --> float, intensity signal-to-noise
            outputs are returned as a list in the format: [(pol_t,avg_frac,sigma_frac,snr_frac),(L_t,avg_L,sigma_L,snr_L),(C_t,avg_C_abs,sigma_C_abs,snr_C),(C_t,avg_C,sigma_C,snr_C),snr]

    """
    if isinstance(buff, int):
        buff1 = buff
        buff2 = buff
    else:
        buff1 = buff[0]
        buff2 = buff[1]

    if pre_calc_tf:
        (I_t,I_f) = I
        (Q_t,Q_f) = Q
        (U_t,U_f) = U
        (V_t,V_f) = V
    else:
        (I_t,Q_t,U_t,V_t) = get_stokes_vs_time(I,Q,U,V,width_native,t_samp,n_t,n_off=n_off,plot=False,normalize=normalize,buff=buff,window=window)

    #if input_weights != []:
    #    timestart = 0
    #    timestop = len(I_t)
    if width_native == -1:#use full timestream for calibrators
        timestart = 0
        timestop = len(I_t)#I.shape[1]
    else:
        peak,timestart,timestop = find_peak(I,width_native,t_samp,n_t,pre_calc_tf=pre_calc_tf,buff=buff)

    if full:
        #linear polarization
        L = np.sqrt(Q**2 + U**2)

        if unbias:
            L_t = np.nanmean(L,axis=0)
            L_t[L_t**2 <= (unbias_factor*np.std(I_t[:n_off]))**2] = np.std(I_t[:n_off])
            L_t = np.sqrt(L_t**2 - np.std(I_t[:n_off])**2)
            L_t = L_t#/I_t
        else:
            L_t = np.nanmean(L,axis=0)#/I_t

        #circular polarization
        C_t = np.nanmean(V,axis=0)#/I_t


        #total polarization
        pol_t = np.sqrt(L_t**2 + C_t**2)/I_t

        C_t = C_t/I_t

        L_t = L_t/I_t

    else:
        #linear polarization
        if unbias:
            L_t = np.sqrt((np.array(Q_t)**2 + np.array(U_t)**2))

            L_t[L_t**2 <= (unbias_factor*np.std(I_t[:n_off]))**2] = np.std(I_t[:n_off])
            L_t = np.sqrt(L_t**2 - np.std(I_t[:n_off])**2)
            L_t = L_t#/I_t
        else:
            L_t = np.sqrt((np.array(Q_t)**2 + np.array(U_t)**2))#/I_t#(np.array(I_t)**2))

        #circular polarization
        C_t = V_t#/I_t

        #total polarization
        pol_t = np.sqrt(C_t**2 + L_t**2)/I_t

        C_t = C_t/I_t

        L_t = L_t/I_t

    if plot:
        f=plt.figure(figsize=(12,6))
        plt.plot(np.arange(timestart,timestop),pol_t[timestart:timestop],label=r'Frequency Averaged Total Polarization ($\sqrt{Q^2 + U^2 + V^2}/I$)')
        plt.plot(np.arange(timestart,timestop),L_t[timestart:timestop],label=r'Frequency Averaged Linear Polarization ($\sqrt{Q^2 + U^2}/I$)')
        plt.plot(np.arange(timestart,timestop),C_t[timestart:timestop],label=r'Frequency Averaged Circular Polarization ($V/I$)')
        plt.grid()
        plt.xlabel("Time Sample (" + str(t_samp*n_t) + " s sampling time)")
        plt.ylabel("Polarization Fraction")
        plt.ylim(-1-allowed_err,1+allowed_err)
        plt.title(label)
        plt.legend()
        plt.savefig(datadir + label + "_polfraction_time_" + calstr + str(n_t) + "_binned" + ext)
        if show:
            plt.show()
        plt.close(f)

    if weighted:
        if input_weights == []:
            I_t_weights=get_weights(I,Q,U,V,width_native,t_samp,n_f,n_t,freq_test,timeaxis,fobj,n_off=n_off,buff=buff,n_t_weight=n_t_weight,sf_window_weights=sf_window_weights)
            I_t_weights_unpadded = get_weights(I,Q,U,V,width_native,t_samp,n_f,n_t,freq_test,timeaxis,fobj,n_off=n_off,buff=buff,n_t_weight=n_t_weight,sf_window_weights=sf_window_weights,padded=False)
        else:
            I_t_weights = input_weights
            I_t_weights_unpadded = np.trim_zeros(input_weights)


        if multipeaks:
            pks,props = find_peaks(I_t_weights,height=height)
            FWHM,heights,intL,intR = peak_widths(I_t_weights,pks)
            intL = intL[0]
            intR = intR[-1]
        else:
            FWHM,heights,intL,intR = peak_widths(I_t_weights,[np.argmax(I_t_weights)])
        intL = int(intL)
        intR = int(intR)

        #average,error
        off_pulse_I = np.nanstd(I_t[:n_off])
        off_pulse_Q = np.nanstd(Q_t[:n_off])
        off_pulse_U = np.nanstd(U_t[:n_off])
        off_pulse_V = np.nanstd(V_t[:n_off])
        L_t_biased = np.sqrt(Q_t**2 + U_t**2)
        T_t_biased = np.sqrt(Q_t**2 + U_t**2  + V_t**2)



        pol_t_cut1 = ((pol_t)[intL:intR])
        pol_t_cut = pol_t_cut1[np.abs(pol_t_cut1) < 1+allowed_err]
        I_t_weights_pol_cut = I_t_weights[intL:intR]
        I_t_weights_pol_cut = I_t_weights_pol_cut[np.abs(pol_t_cut1) < 1+allowed_err]
        avg_frac = np.nansum(pol_t_cut*I_t_weights_pol_cut)/np.nansum(I_t_weights_pol_cut)
        #sigma_frac = np.nansum(I_t_weights_pol_cut*np.sqrt((pol_t_cut - avg_frac)**2))/np.nansum(I_t_weights_pol_cut)
        sigma_frac = np.sqrt((Q_t*off_pulse_Q/I_t/T_t_biased)**2 +   (U_t*off_pulse_U/I_t/T_t_biased)**2+ (V_t*off_pulse_V/I_t/T_t_biased)**2  + (T_t_biased*off_pulse_I/((I_t)**2)))
        sigma_frac = (sigma_frac[intL:intR])[np.abs(pol_t_cut1) < 1+allowed_err]
        sigma_frac = np.sqrt(np.nansum((I_t_weights_pol_cut*sigma_frac)**2))/np.nansum(I_t_weights_pol_cut)

        L_t_cut1 = ((L_t)[intL:intR])
        L_t_cut = L_t_cut1[np.abs(pol_t_cut1) < 1+allowed_err]
        I_t_weights_L_cut = I_t_weights[intL:intR]
        I_t_weights_L_cut = I_t_weights_L_cut[np.abs(pol_t_cut1) < 1+allowed_err]
        avg_L = np.nansum(L_t_cut*I_t_weights_L_cut)/np.nansum(I_t_weights_L_cut)
        #sigma_L = np.nansum(I_t_weights_L_cut*np.sqrt((L_t_cut - avg_L)**2))/np.nansum(I_t_weights_L_cut)
        sigma_L = np.sqrt((Q_t*off_pulse_Q/I_t/L_t_biased)**2 +   (U_t*off_pulse_U/I_t/L_t_biased)**2  + (L_t_biased*off_pulse_I/((I_t)**2)))
        sigma_L = (sigma_L[intL:intR])[np.abs(pol_t_cut1) < 1+allowed_err]
        sigma_L = np.sqrt(np.nansum((I_t_weights_pol_cut*sigma_L)**2))/np.nansum(I_t_weights_pol_cut)


        C_t_cut1 = (np.abs(C_t)[intL:intR])
        C_t_cut = C_t_cut1[np.abs(pol_t_cut1) < 1+allowed_err]
        I_t_weights_C_cut = I_t_weights[intL:intR]
        I_t_weights_C_cut = I_t_weights_C_cut[np.abs(pol_t_cut1) < 1+allowed_err]
        avg_C_abs = np.nansum(C_t_cut*I_t_weights_C_cut)/np.nansum(I_t_weights_C_cut)
        #sigma_C_abs = np.nansum(I_t_weights_C_cut*np.sqrt((C_t_cut - avg_C_abs)**2))/np.nansum(I_t_weights_C_cut)
        sigma_C_abs = np.sqrt((off_pulse_V/I_t)**2  + (V_t*off_pulse_I/((I_t)**2))**2)
        sigma_C_abs = (sigma_C_abs[intL:intR])[np.abs(pol_t_cut1) < 1+allowed_err]
        sigma_C_abs = np.sqrt(np.nansum((I_t_weights_pol_cut*sigma_C_abs)**2))/np.nansum(I_t_weights_pol_cut)

        C_t_cut1 = ((C_t)[intL:intR])
        C_t_cut = C_t_cut1[np.abs(pol_t_cut1) < 1+allowed_err]
        I_t_weights_C_cut = I_t_weights[intL:intR]
        I_t_weights_C_cut = I_t_weights_C_cut[np.abs(pol_t_cut1) < 1+allowed_err]
        avg_C = np.nansum(C_t_cut*I_t_weights_C_cut)/np.nansum(I_t_weights_C_cut)
        sigma_C = np.nansum(I_t_weights_C_cut*np.sqrt((C_t_cut - avg_C)**2))/np.nansum(I_t_weights_C_cut)
        sigma_C = sigma_C_abs

        I_trial_binned = convolve(I_t,I_t_weights_unpadded)
        sigbin = np.argmax(I_trial_binned)
        sig0 = I_trial_binned[sigbin]
        I_binned = convolve(I_t,I_t_weights_unpadded)
        noise = np.std(np.concatenate([I_binned[:sigbin-(timestop-timestart)*2],I_binned[sigbin+(timestop-timestart)*2:]]))
        snr = sig0/noise
        print((sig0,noise,snr))

        T_trial_binned = convolve(pol_t*I_t,I_t_weights_unpadded)
        sig0 = T_trial_binned[sigbin]
        Q_binned = convolve(Q_t,I_t_weights_unpadded)
        noise = np.std(np.concatenate([Q_binned[:sigbin-(timestop-timestart)*2],Q_binned[sigbin+(timestop-timestart)*2:]]))
        snr_frac = sig0/noise
        print((sig0,noise,snr_frac))

        L_trial_binned = convolve(L_t*I_t,I_t_weights_unpadded)
        sig0 = L_trial_binned[sigbin]
        snr_L = sig0/noise
        print((sig0,noise,snr_L))

        C_trial_binned = np.convolve(np.abs(C_t)*I_t,I_t_weights_unpadded)
        sig0 = C_trial_binned[sigbin]
        V_binned = convolve(V_t,I_t_weights_unpadded)
        noise = np.std(np.concatenate([V_binned[:sigbin-(timestop-timestart)*2],V_binned[sigbin+(timestop-timestart)*2:]]))
        snr_C = sig0/noise
        print((sig0,noise,snr_C))


    else:
        #average,error
        off_pulse_I = np.nanstd(I_t[:n_off])
        off_pulse_Q = np.nanstd(Q_t[:n_off])
        off_pulse_U = np.nanstd(U_t[:n_off])
        off_pulse_V = np.nanstd(V_t[:n_off])
        L_t_biased = np.sqrt(Q_t**2 + U_t**2)
        T_t_biased = np.sqrt(Q_t**2 + U_t**2  + V_t**2)

        avg_frac = np.nanmean((pol_t[intL:intR])[np.abs(pol_t)[intL:intR]<=1+allowed_err])
        avg_L = np.nanmean((L_t[intL:intR])[np.abs(L_t)[intL:intR]<=1+allowed_err])
        avg_C_abs = np.nanmean((np.abs(C_t)[intL:intR])[np.abs(C_t)[intL:intR]<=1+allowed_err])
        avg_C = np.nanmean(((C_t)[intL:intR])[(C_t)[intL:intR]<=1+allowed_err])

        #RMS error
        sigma_frac = np.sqrt((Q_t*off_pulse_Q/I_t/T_t_biased)**2 +   (U_t*off_pulse_U/I_t/T_t_biased)**2+ (V_t*off_pulse_V/I_t/T_t_biased)**2  + (T_t_biased*off_pulse_I/((I_t)**2)))
        sigma_frac = (sigma_frac[intL:intR])[np.abs(pol_t)[intL:intR] < 1+allowed_err]
        sigma_frac = np.sqrt(np.nansum((sigma_frac)**2))/len(sigma_frac)

        sigma_L = np.sqrt((Q_t*off_pulse_Q/I_t/L_t_biased)**2 +   (U_t*off_pulse_U/I_t/L_t_biased)**2  + (L_t_biased*off_pulse_I/((I_t)**2)))
        sigma_L = (sigma_L[intL:intR])[np.abs(pol_t)[intL:intR] < 1+allowed_err]
        sigma_L = np.sqrt(np.nansum((sigma_L)**2))/len(sigma_L)

        sigma_C_abs = np.sqrt((off_pulse_V/I_t)**2  + (V_t*off_pulse_I/((I_t)**2))**2)
        sigma_C_abs = (sigma_C_abs[intL:intR])[np.abs(pol_t)[intL:intR] < 1+allowed_err]
        sigma_C_abs = np.sqrt(np.nansum((sigma_C_abs)**2))/len(sigma_C_abs)
        sigma_C = sigma_C_abs

        #SNR
        sig0 = np.nanmean(I_t[timestart:timestop])
        I_t_cut1 = I_t[timestart%(timestop-timestart):]
        I_t_cut = I_t_cut1[:(len(I_t_cut1)-(len(I_t_cut1)%(timestop-timestart)))]
        I_t_binned = I_t_cut.reshape(len(I_t_cut)//(timestop-timestart),timestop-timestart).mean(1)
        sigbin = np.argmax(I_t_binned)
        noise = (np.nanstd(np.concatenate([I_t_cut[:sigbin],I_t_cut[sigbin+1:]])))
        snr = sig0/noise

        sig0 = np.nanmean((pol_t*I_t)[timestart:timestop])
        pol_t_cut1 = (pol_t*I_t)[timestart%(timestop-timestart):]
        pol_t_cut = pol_t_cut1[:(len(pol_t_cut1)-(len(pol_t_cut1)%(timestop-timestart)))]
        pol_t_binned = pol_t_cut.reshape(len(pol_t_cut)//(timestop-timestart),timestop-timestart).mean(1)
        sigbin = np.argmax(pol_t_binned)
        noise = (np.nanstd(np.concatenate([pol_t_cut[:sigbin],pol_t_cut[sigbin+1:]])))
        snr_frac = sig0/noise

        sig0 = np.nanmean((L_t*I_t)[timestart:timestop])
        L_t_cut1 = (L_t*I_t)[timestart%(timestop-timestart):]
        L_t_cut = L_t_cut1[:(len(L_t_cut1)-(len(L_t_cut1)%(timestop-timestart)))]
        L_t_binned = L_t_cut.reshape(len(L_t_cut)//(timestop-timestart),timestop-timestart).mean(1)
        sigbin = np.argmax(L_t_binned)
        noise = (np.nanstd(np.concatenate([L_t_cut[:sigbin],L_t_cut[sigbin+1:]])))
        snr_L = sig0/noise

        sig0 = np.nanmean((np.abs(C_t)*I_t)[timestart:timestop])
        C_t_cut1 = (np.abs(C_t)*I_t)[timestart%(timestop-timestart):]
        C_t_cut = C_t_cut1[:(len(C_t_cut1)-(len(C_t_cut1)%(timestop-timestart)))]
        C_t_binned = C_t_cut.reshape(len(C_t_cut)//(timestop-timestart),timestop-timestart).mean(1)
        sigbin = np.argmax(C_t_binned)
        noise = (np.nanstd(np.concatenate([C_t_cut[:sigbin],C_t_cut[sigbin+1:]])))
        snr_C = sig0/noise



    #snr_frac = np.nanmean(pol_t[timestart:timestop])/np.nanstd(pol_t[:n_off])
    #snr_L = np.nanmean(L_t[timestart:timestop])/np.nanstd(L_t[:n_off])
    #snr_C = np.nanmean(C_t[timestart:timestop])/np.nanstd(C_t[:n_off])

    return [(pol_t,avg_frac,sigma_frac,snr_frac),(L_t,avg_L,sigma_L,snr_L),(C_t,avg_C_abs,sigma_C_abs,snr_C),(C_t,avg_C,sigma_C,snr_C),snr]



#1sigma PA error calculation 
def PA_error_NKC(mean,L0,sigma,siglevel=0.68,plot=False):
    """
    This function computes the error in the position angle given the 
    mean value, linear polarization, and off-pulse RMS. This follows the
    definition from NKC.
    Inputs: mean --> float, mean position angle in radians
            L0 --> float, linear polarization fraction
            sigma --> float, off-pulse RMS in Stokes I
            siglevel --> float, between 0 and 1; defines significance level for error estimate. Default is 1sigma errors (68% significance)
            plot --> bool, set to plot position angle distribution
    Outputs: avg_eror,upper_limit,lower_limit

    """

    PA_axis = np.linspace(mean-np.pi/2,mean+np.pi/2,1000)
    P0 = (L0/sigma/np.sqrt(2))
    #print(P0)
    eta_axis = P0*(np.cos(2*(PA_axis-mean)))
    #print(eta_axis*np.exp(eta_axis**2))
    Pdist = np.exp(-(P0**2)/2)*(1/np.sqrt(np.pi))*((1/np.sqrt(np.pi)) + eta_axis*np.exp(eta_axis**2)*(1+erf(eta_axis)) )
    Pdist = Pdist/np.sum(Pdist*(PA_axis[1]-PA_axis[0]))
    
    #cumulative dist
    cumdisthost = np.cumsum(Pdist)/np.sum(Pdist)
    #print("calculating " + str(((1-siglevel)/2)) + " and"+ str((1-((1-siglevel)/2))) + " percentiles")
    low = PA_axis[np.argmin(abs(cumdisthost-((1-siglevel)/2)))]
    upp = PA_axis[np.argmin(abs(cumdisthost-(1-((1-siglevel)/2))))]
    
    if plot:
        plt.figure()
        plt.plot(PA_axis,Pdist)
        plt.show()
    return (upp-low)/2,upp,low

def PA_error_NKC_array(means,L0s,sigma,siglevel=0.68):
    """
    This function computes the error in the position angle given an array of 
    mean values, linear polarization, and off-pulse RMS. This follows the
    definition from NKC.
    Inputs: means --> array-like, mean position angles in radians
            L0s --> array-like, linear polarization fractions
            sigma --> float, off-pulse RMS in Stokes I
            siglevel --> float, between 0 and 1; defines significance level for error estimate. Default is 1sigma errors (68% significance)
    Outputs: errors --> 1D array of errors
    """
    errs = []
    for i in range(len(means)):
        err,upp,low = PA_error_NKC(means[i],L0s[i],sigma,siglevel)
        errs.append(err)
    return np.array(errs)



def fitfunc(x,amp,scale):
    return amp*np.exp(-x/scale)


def PA_err_fit(LSNR,popt1=[ 0.2685898,  28.71010826],popt2=[0.93328853 ,9.92364144],boundary=19):
    if LSNR > boundary:
        return fitfunc(LSNR,popt1[0],popt1[1])
    else:
        return fitfunc(LSNR,popt2[0],popt2[1])
    
def PA_err_fit_array(LSNR_arr,popt1=[ 0.2685898,  28.71010826],popt2=[0.93328853 ,9.92364144],boundary=19):
    PAerrs = []
    for LSNR in LSNR_arr:
        PAerrs.append(PA_err_fit(LSNR,popt1,popt2,boundary))
    return np.array(PAerrs)


def get_pol_angle(I,Q,U,V,width_native,t_samp,n_t,n_f,freq_test,n_off=3000,plot=False,datadir=DEFAULT_DATADIR,label='',calstr='',ext=ext,pre_calc_tf=False,show=False,normalize=True,buff=0,weighted=False,n_t_weight=1,timeaxis=None,fobj=None,sf_window_weights=45,multipeaks=False,height=0.03,window=30,input_weights=[],unbias_factor=1,errormethod='MC',intL=None,intR=None):
    """
    This function calculates and plots the polarization angle averaged over both time and
    frequency, and the average polarization angle within the peak.
    Inputs: I,Q,U,V --> 2D arrays, dynamic spectra of I,Q,U,V generated with get_stokes_2D()
            width_native --> int, width in samples of pulse in native sampling rate; equivalent to ibox parameter
            t_samp --> float, sampling time
            n_t --> int, number of time samples spectra have been binned by (average over)
            n_f --> int, number of frequency samples spectra have been binned by (average over)
            freq_test --> list, frequency axes for each stokes parameter
            n_off --> int, specifies index of end of off-pulse samples
            plot --> bool, set to plot and output images in specified directory
            datadir --> str, path to directory to output images
            label --> str, trigname_nickname of FRB, e.g. '220319aaeb_Mark'
            calstr --> str, string specifying whether given data is calibrated or not, optional
            pre_calc_tf --> bool, set if I,Q,U,V are given as tuples of the frequency and time averaged arrays, e.g. (I_t,I_f)
            ext --> str, image file extension (png, pdf, etc.)
            show --> bool, if True, displays images with matplotlib
            buff --> int, number of samples to buffer either side of pulse if needed
    Outputs: PA_f --> 1D array, frequency dependent PA
             PA_t --> 1D array, time dependent PA
             avg --> float, frequency and time averaged PA
    """
    if pre_calc_tf:
        (I_t,I_f) = I
        (Q_t,Q_f) = Q
        (U_t,U_f) = U
        (V_t,V_f) = V
    else:
        (I_t,Q_t,U_t,V_t) = get_stokes_vs_time(I,Q,U,V,width_native,t_samp,n_t,n_off=n_off,plot=False,normalize=normalize,buff=buff,window=window)
        (I_f,Q_f,U_f,V_f) = get_stokes_vs_freq(I,Q,U,V,width_native,t_samp,n_f,n_t,freq_test,n_off=n_off,plot=False,normalize=normalize,buff=buff,weighted=weighted,n_t_weight=n_t_weight,timeaxis=timeaxis,fobj=fobj,sf_window_weights=sf_window_weights)


    if input_weights != []:
        timestart = 0
        timestop = len(I_t)
    elif width_native == -1:#use full timestream for calibrators
        timestart = 0
        timestop = len(I_t)#I.shape[1]
    else:
        peak,timestart,timestop = find_peak(I,width_native,t_samp,n_t,pre_calc_tf=pre_calc_tf,buff=buff)

    PA_f = 0.5*np.angle(Q_f +1j*U_f)#np.sqrt((np.array(Q_f)**2 + np.array(U_f)**2 + np.array(V_f)**2)/(np.array(I_f)**2))
    PA_t = 0.5*np.angle(Q_t +1j*U_t)#np.sqrt((np.array(Q_t)**2 + np.array(U_t)**2 + np.array(V_t)**2)/(np.array(I_t)**2))

    #errorbars
    L_t = np.sqrt(Q_t**2 + U_t**2)#*I_w_t_filt
    L_t[L_t**2 <= (unbias_factor*np.std(I_t[:n_off]))**2] = np.std(I_t[:n_off])
    L_t = np.sqrt(L_t**2 - np.std(I_t[:n_off])**2)
    if errormethod == 'NKC':
        PA_t_errs = PA_error_NKC_array(PA_t,L_t,np.std(I_t[:n_off]))
    elif errormethod == 'MC':
        PA_t_errs = PA_err_fit_array(L_t)
    else:
        print("Invalid PA error method")
        return None
    L_f = np.sqrt(Q_f**2 + U_f**2)
    L_f[L_f**2 <= (unbias_factor*np.std(I_t[:n_off]))**2] = np.std(I_t[:n_off])
    L_f = np.sqrt(L_f**2 - np.std(I_t[:n_off])**2)
    if errormethod == 'NKC':
        PA_f_errs = PA_error_NKC_array(PA_f,L_f,np.std(I_t[:n_off]))
    elif errormethod == 'MC':
        PA_f_errs = PA_err_fit_array(L_f)
    else:
        print("Invalid PA error method")
        return None

    if plot:
        f=plt.figure(figsize=(12,6))
        plt.errorbar(freq_test[0],PA_f,yerr=PA_f_errs,linestyle="",marker="o",capsize=5)
        plt.grid()
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Polarization Angle (rad)")
        #plt.ylim(-1,1)
        plt.title(label)
        plt.savefig(datadir + label + "_polangle_frequency_" + calstr + str(n_f) + "_binned" + ext)
        if show:
            plt.show()
        plt.close(f)

        f=plt.figure(figsize=(12,6))
        plt.errorbar(np.arange(timestart,timestop),PA_t[timestart:timestop],yerr=PA_t_errs[timestart:timestop],linestyle="",marker="o",capsize=5)
        plt.grid()
        plt.xlabel("Time Sample (" + str(t_samp*n_t) + " s sampling time)")
        plt.ylabel("Polarization Angle (rad)")
        #plt.ylim(-1,1)
        plt.title(label)
        plt.savefig(datadir + label + "_polangle_time_" + calstr + str(n_t) + "_binned" + ext)
        #plt.xlim(timestart,timestop)dd
        if show:
            plt.show()
        plt.close(f)

    
    if weighted:
        if input_weights == []:
            I_t_weights=get_weights(I,Q,U,V,width_native,t_samp,n_f,n_t,freq_test,timeaxis,fobj,n_off=n_off,buff=buff,n_t_weight=n_t_weight,sf_window_weights=sf_window_weights)
        else:
            I_t_weights = input_weights

 
        if multipeaks and (intL == None or intR == None):
            pks,props = find_peaks(I_t_weights,height=height)
            FWHM,heights,intL,intR = peak_widths(I_t_weights,pks)
            intL = intL[0]
            intR = intR[-1]
        elif (intL == None or intR == None):
            FWHM,heights,intL,intR = peak_widths(I_t_weights,[np.argmax(I_t_weights)])
        intL = int(intL)
        intR = int(intR)

        #average,error
        PA_t_cut1 = PA_t[intL:intR]
        PA_t_cut = PA_t_cut1#[np.abs(PA_t_cut1) < (2*np.pi)]
        I_t_weights_pol_cut = I_t_weights[intL:intR]
        I_t_weights_pol_cut = I_t_weights_pol_cut#[np.abs(PA_t_cut1) < (2*np.pi)]
        avg_PA = np.nansum(PA_t_cut*I_t_weights_pol_cut)/np.nansum(I_t_weights_pol_cut)
        #sigma_PA = np.nansum(I_t_weights_pol_cut*np.sqrt((PA_t_cut - avg_PA)**2))/np.nansum(I_t_weights_pol_cut)
        sigma_PA = np.sqrt(np.nansum((I_t_weights_pol_cut*(PA_t_errs[intL:intR]))**2))/np.nansum(I_t_weights_pol_cut)

    else:
        if multipeaks and (intL == None or intR == None):
            pks,props = find_peaks(I_t,height=height)
            FWHM,heights,intL,intR = peak_widths(I_t,pks)
            intL = intL[0]
            intR = intR[-1]
        elif (intL == None or intR == None):
            FWHM,heights,intL,intR = peak_widths(I_t,[np.argmax(I_t)])
        intL = int(intL)
        intR = int(intR)

        #avg_PA = np.mean(PA_t[timestart:timestop][PA_t[timestart:timestop]<1])
        avg_PA = np.nanmean(PA_t[intL:intR])
        #sigma_PA = np.nanstd(PA_t[intL:intR])
        sigma_PA = np.sqrt(np.nansum(((PA_t_errs[intL:intR]))**2))/(intR-intL)

    return PA_f,PA_t,PA_f_errs,PA_t_errs,avg_PA,sigma_PA

#Calculate polarization angle vs time from 2D I Q U V arrays
def get_pol_angle_vs_time(I,Q,U,V,width_native,t_samp,n_t,n_off=3000,plot=False,datadir=DEFAULT_DATADIR,label='',calstr='',ext=ext,pre_calc_tf=False,show=False,normalize=True,buff=0,weighted=False,n_t_weight=1,timeaxis=None,fobj=None,sf_window_weights=45,multipeaks=False,height=0.03,window=30,input_weights=[],unbias_factor=1):
    """
    This function calculates and plots the polarization angle averaged over both time and
    frequency, and the average polarization angle within the peak.
    Inputs: I,Q,U,V --> 2D arrays, dynamic spectra of I,Q,U,V generated with get_stokes_2D()
            width_native --> int, width in samples of pulse in native sampling rate; equivalent to ibox parameter
            t_samp --> float, sampling time
            n_t --> int, number of time samples spectra have been binned by (average over)
            n_off --> int, specifies index of end of off-pulse samples
            plot --> bool, set to plot and output images in specified directory
            datadir --> str, path to directory to output images
            label --> str, trigname_nickname of FRB, e.g. '220319aaeb_Mark'
            calstr --> str, string specifying whether given data is calibrated or not, optional
            pre_calc_tf --> bool, set if I,Q,U,V are given as tuples of the frequency and time averaged arrays, e.g. (I_t,I_f)
            ext --> str, image file extension (png, pdf, etc.)
            show --> bool, if True, displays images with matplotlib
            buff --> int, number of samples to buffer either side of pulse if needed
    Outputs: PA_t --> 1D array, time dependent PA
             avg --> float, frequency and time averaged PA
    """
    if pre_calc_tf:
        (I_t,I_f) = I
        (Q_t,Q_f) = Q
        (U_t,U_f) = U
        (V_t,V_f) = V
    else:
        (I_t,Q_t,U_t,V_t) = get_stokes_vs_time(I,Q,U,V,width_native,t_samp,n_t,n_off=n_off,plot=False,normalize=normalize,buff=buff,window=window)


    if plot and input_weights != []:
        timestart = 0
        timestop = len(I_t)
    elif plot and width_native == -1:#use full timestream for calibrators
        timestart = 0
        timestop = len(I_t)#I.shape[1]
    elif plot:
        peak,timestart,timestop = find_peak(I,width_native,t_samp,n_t,pre_calc_tf=pre_calc_tf,buff=buff)

    PA_t = 0.5*np.angle(Q_t +1j*U_t)#np.sqrt((np.array(Q_t)**2 + np.array(U_t)**2 + np.array(V_t)**2)/(np.array(I_t)**2))

    #errorbars
    L_t = np.sqrt(Q_t**2 + U_t**2)#*I_w_t_filt
    L_t[L_t**2 <= (unbias_factor*np.std(I_t[:n_off]))**2] = np.std(I_t[:n_off])
    L_t = np.sqrt(L_t**2 - np.std(I_t[:n_off])**2)
    assert(len(L_t) == len(PA_t))
    PA_t_errs = PA_error_NKC_array(PA_t,L_t,np.std(I_t[:n_off]))

    if plot:
        f=plt.figure(figsize=(12,6))
        plt.errorbar(np.arange(timestart,timestop),PA_t[timestart:timestop],yerr=PA_t_errs[timestart:timestop],linestyle="",marker="o",capsize=5)
        plt.grid()
        plt.xlabel("Time Sample (" + str(t_samp*n_t) + " s sampling time)")
        plt.ylabel("Polarization Angle (rad)")
        #plt.ylim(-1,1)
        plt.title(label)
        plt.savefig(datadir + label + "_polangle_time_" + calstr + str(n_t) + "_binned" + ext)
        #plt.xlim(timestart,timestop)dd
        if show:
            plt.show()
        plt.close(f)
    """
    if weighted:
        if input_weights == []:
            I_t_weights=get_weights(I,Q,U,V,width_native,t_samp,n_f,n_t,freq_test,timeaxis,fobj,n_off=n_off,buff=buff,n_t_weight=n_t_weight,sf_window_weights=sf_window_weights)
        else:
            I_t_weights = input_weights


        if multipeaks:
            pks,props = find_peaks(I_t_weights,height=height)
            FWHM,heights,intL,intR = peak_widths(I_t_weights,pks)
            intL = intL[0]
            intR = intR[-1]
        else:
            FWHM,heights,intL,intR = peak_widths(I_t_weights,[np.argmax(I_t_weights)])
        print("here: " + str((intL,intR)))
        intL = int(intL)
        intR = int(intR)
        print("here: " + str((intL,intR)))

        #average,error
        PA_t_cut1 = PA_t[intL:intR]
        PA_t_cut = PA_t_cut1#[np.abs(PA_t_cut1) < (2*np.pi)]
        I_t_weights_pol_cut = I_t_weights[intL:intR]
        I_t_weights_pol_cut = I_t_weights_pol_cut#[np.abs(PA_t_cut1) < (2*np.pi)]
        avg_PA = np.nansum(PA_t_cut*I_t_weights_pol_cut)/np.nansum(I_t_weights_pol_cut)
        #sigma_PA = np.nansum(I_t_weights_pol_cut*np.sqrt((PA_t_cut - avg_PA)**2))/np.nansum(I_t_weights_pol_cut)
        sigma_PA = np.sqrt(np.nansum((I_t_weights_pol_cut*(PA_t_errs[intL:intR]))**2))/np.nansum(I_t_weights_pol_cut)

    else:
        if multipeaks:
            pks,props = find_peaks(I_t,height=height)
            FWHM,heights,intL,intR = peak_widths(I_t,pks)
            intL = intL[0]
            intR = intR[-1]
        else:
            FWHM,heights,intL,intR = peak_widths(I_t,[np.argmax(I_t)])
        print("here: " + str((intL,intR)))
        intL = int(intL)
        intR = int(intR)
        print("here: " + str((intL,intR)))

        #avg_PA = np.mean(PA_t[timestart:timestop][PA_t[timestart:timestop]<1])
        avg_PA = np.nanmean(PA_t[intL:intR])
        #sigma_PA = np.nanstd(PA_t[intL:intR])
        sigma_PA = np.sqrt(np.nansum(((PA_t_errs[intL:intR]))**2))/(intR-intL)
    """
    return PA_t,PA_t_errs#,avg_PA,sigma_PA


"""
#function to compute error and upper limit on polarization
def cal_pol_upper_limit_sim():

    #Get ratio and phase errors from IQUV

    #gain calibrator
    if len(gain_obs_names) ==1:
        obs_name = gain_obs_names[0]
        label = gain_source + obs_name + suffix
        sdir = datadir + label
        (I_gain,Q_gain,U_gain,V_gain,fobj,timeaxis,freq_test,wav_test) = dsapol.get_stokes_2D(gain_dir,label,nsamps,n_t=n_t,n_f=n_f,n_off=-1,sub_offpulse_mean=False)
        (I_f,Q_f,U_f,V_f) = dsapol.get_stokes_vs_freq(I_gain,Q_gain,U_gain,V_gain,-1,fobj.header.tsamp,n_f,freq_test,datadir=datadir,label=label,plot=True,show=True,normalize=False)


        I_err_f = I_gain.std(1)/np.sqrt(I_gain.shape[1])
        Q_err_f = Q_gain.std(1)/np.sqrt(Q_gain.shape[1])
        U_err_f = U_gain.std(1)/np.sqrt(U_gain.shape[1])
        V_err_f = V_gain.std(1)/np.sqrt(V_gain.shape[1])
    
        #random trials for gain calibrator
        I_samps = []
        Q_samps = []
        U_samps = []
        V_samps = []

        for i in range(len(I_f)):
            I_samps.append(np.random.normal(I_f[i],I_err_f[i],trials))
            Q_samps.append(np.random.normal(Q_f[i],Q_err_f[i],trials))
            U_samps.append(np.random.normal(U_f[i],U_err_f[i],trials))
            V_samps.append(np.random.normal(V_f[i],V_err_f[i],trials))   
        I_samps = np.array(I_samps)
        Q_samps = np.array(Q_samps)
        U_samps = np.array(U_samps)
        V_samps = np.array(V_samps)


        ratio_samps = []
        for i in range(trials):
            ratio,ratio_fit_params = dsapol.gaincal(I_samps[:,i],Q_samps[:,i],U_samps[:,i],V_samps[:,i],freq_test,stokes=True,deg=deg,datadir=datadir,label=label,plot=False,show=False)
        
            if use_fit:
                ratio_fit = np.zeros(np.shape(freq_test[0]))
                for i in range(deg+1):
                    ratio_fit += ratio_fit_params[i]*(freq_test[0]**(deg-i))
                ratio_samps.append(ratio_fit)
            else:
                ratio_samps.append(ratio)
        ratio_samps = np.array(ratio_samps)



    else:

        pass


    #phase calibrator
    if len(phase_obs_names) == 1:
        obs_name = phase_obs_names[0]
        label = phase_source + obs_name + suffix
        sdir = datadir + label
        (I_phase,Q_phase,U_phase,V_phase,fobj,timeaxis,freq_test,wav_test) = dsapol.get_stokes_2D(phase_dir,label,nsamps,n_t=n_t,n_f=n_f,n_off=-1,sub_offpulse_mean=False)
        (I_f,Q_f,U_f,V_f) = dsapol.get_stokes_vs_freq(I_phase,Q_phase,U_phase,V_phase,-1,fobj.header.tsamp,n_f,freq_test,datadir=datadir,label=label,plot=True,show=True,normalize=False)


        I_err_f = I_phase.std(1)/np.sqrt(I_phase.shape[1])
        Q_err_f = Q_phase.std(1)/np.sqrt(Q_phase.shape[1])
        U_err_f = U_phase.std(1)/np.sqrt(U_phase.shape[1])
        V_err_f = V_phase.std(1)/np.sqrt(V_phase.shape[1])

        #random trials for gain calibrator
        I_samps = []
        Q_samps = []
        U_samps = []
        V_samps = []

        for i in range(len(I_f)):
            I_samps.append(np.random.normal(I_f[i],I_err_f[i],trials))
            Q_samps.append(np.random.normal(Q_f[i],Q_err_f[i],trials))
            U_samps.append(np.random.normal(U_f[i],U_err_f[i],trials))
            V_samps.append(np.random.normal(V_f[i],V_err_f[i],trials))
        I_samps = np.array(I_samps)
        Q_samps = np.array(Q_samps)
        U_samps = np.array(U_samps)
        V_samps = np.array(V_samps)

        phase_samps = []
        for i in range(trials):
            phase,phase_fit_params = dsapol.phasecal(I_samps[:,i],Q_samps[:,i],U_samps[:,i],V_samps[:,i],freq_test,stokes=True,deg=deg,datadir=datadir,label=label,plot=False,show=False)
            
            if use_fit:
                phase_fit = np.zeros(np.shape(freq_test[0]))
                for i in range(deg+1):
                    phase_fit += phase_fit_params[i]*(freq_test[0]**(deg-i))
                phase_samps.append(phase_fit)
            else:
                phase_samps.append(phase))
            
        phase_samps = np.array(phase_samps)
    else:
        pass
"""

    



#Functions for each step of calibration

#general fit for deg degree polynomial
def compute_fit(data,xaxis,deg):
    """
    This function computes the fit for a deg degree polynomial for each point on the specified xaxis
    Inputs: data --> 1D array, data to be fit 
            xaxis --> 1D array, xaxis corresponding to data, must have len(xaxis) == len(data)
            deg --> degree of polynomial to fit
    Returns: 
    """
    popt = np.polyfit(xaxis,np.nan_to_num(data,nan=np.nanmedian(data)),deg=deg)
    data_fit = np.zeros(np.shape(xaxis))
    for i in range(deg+1):
        data_fit += popt[i]*(xaxis**(deg-i))
    return data_fit,popt

#Takes observed data products for unpolarized calibrator and calculates ratio of complex gain magnitudes
#Inputs are either stokes parameters IQUV (set stokes=True) or xx,yy,xy,yx (set stokes=False)
def gaincal(xx_I_obs,yy_Q_obs,xy_U_obs,yx_V_obs,freq_test,stokes=True,deg=10,plot=False,datadir=DEFAULT_DATADIR,label='',show=False,sfwindow=-1):
    """
    This function takes observed IQUV for unpolarized gain calibrator (e.g. 3C48) and computes 
    |gxx|/|gyy| for Jones matrix. Optionall fits with deg degree polynomial vs frequency
    Inputs: I,Q,U,V --> 2D arrays, dynamic spectra of I,Q,U,V generated with get_stokes_2D()
            width_native --> int, width in samples of pulse in native sampling rate; equivalent to ibox parameter
            t_samp --> float, sampling time
            n_t --> int, number of time samples spectra have been binned by (average over)
            n_f --> int, number of frequency samples spectra have been binned by (average over)
            freq_test --> list, frequency axes for each stokes parameter
            n_off --> int, specifies index of end of off-pulse samples
            plot --> bool, set to plot and output images in specified directory
            datadir --> str, path to directory to output images
            label --> str, trigname_nickname of FRB, e.g. '220319aaeb_Mark'
            calstr --> str, string specifying whether given data is calibrated or not, optional
            pre_calc_tf --> bool, set if I,Q,U,V are given as tuples of the frequency and time averaged arrays, e.g. (I_t,I_f)
            ext --> str, image file extension (png, pdf, etc.)
            show --> bool, if True, displays images with matplotlib
            buff --> int, number of samples to buffer either side of pulse if needed
            sfwindow --> int, width of savgol filter window, must be odd
    """

    if stokes:
        I_obs = xx_I_obs
        Q_obs = yy_Q_obs
        U_obs = xy_U_obs
        V_obs = yx_V_obs
    else: 
        I_obs = 0.5*(xx_I_obs + yy_Q_obs)
        Q_obs = 0.5*(xx_I_obs - yy_Q_obs)
        U_obs = 0.5*(xy_U_obs + yx_V_obs)#np.real(xy_obs)
        V_obs = -1j*0.5*(xy_U_obs - yx_V_obs)#-np.imag(xy_obs)
    
    ratio = np.sqrt((I_obs + Q_obs)/(I_obs - Q_obs))
    ratio = np.nan_to_num(ratio,nan=np.nanmedian(ratio))
    ratio_fit,fit_params = compute_fit(ratio,freq_test[0],deg)
    
    #mask and savgol filter
    if sfwindow != -1:
        ratio_masked = np.ma.masked_outside(ratio,np.mean(ratio)-3*np.std(ratio),np.mean(ratio)+3*np.std(ratio))
        ratio_sf = sf(ratio_masked,sfwindow,3)

    if plot:
        f= plt.figure()
        plt.title(r'Gain Ratio ($g_{xx}/g_{yy}$) ' + label)
        plt.plot(freq_test[0],ratio,label="Calculated")
        plt.plot(freq_test[0],ratio_fit,label="Fit")
        if sfwindow != -1:
            plt.plot(freq_test[0],ratio_sf,label="Savgol Filter, " + str(sfwindow) + " sample window")
        plt.axhline(np.nanmedian(ratio),color="black",label="median")
        plt.legend()
        plt.xlabel("Frequency (MHz)")
        plt.grid()
        plt.savefig(datadir + "calibrator_gain_ratio_" + label + ext)
        if show:
            plt.show()
        plt.close(f) 

    if sfwindow != -1:
        return ratio,fit_params,ratio_sf
    return ratio,fit_params,None

#Cleans spurious peaks by downsampling and filtering
def cleangaincal_V2(xx_I_obs,yy_Q_obs,xy_U_obs,yx_V_obs,freq_test,stokes=True,deg=10,plot=False,datadir=DEFAULT_DATADIR,label='',show=False,sfwindow=-1,padwidth=10,peakheight=2,n_t_down=8,n_f=1,edgefreq=-1,breakfreq=-1):
   
    if stokes:
        I_obs = xx_I_obs
        Q_obs = yy_Q_obs
        U_obs = xy_U_obs
        V_obs = yx_V_obs
    else:
        I_obs = 0.5*(xx_I_obs + yy_Q_obs)
        Q_obs = 0.5*(xx_I_obs - yy_Q_obs)
        U_obs = 0.5*(xy_U_obs + yx_V_obs)#np.real(xy_obs)
        V_obs = -1j*0.5*(xy_U_obs - yx_V_obs)#-np.imag(xy_obs)

    #downsample
    I_obs = I_obs.reshape(len(I_obs)//n_t_down,n_t_down).mean(1)#avg_freq(I_obs,n_t_down)
    Q_obs = Q_obs.reshape(len(Q_obs)//n_t_down,n_t_down).mean(1)#avg_freq(Q_obs,n_t_down)
    U_obs = U_obs.reshape(len(U_obs)//n_t_down,n_t_down).mean(1)#avg_freq(U_obs,n_t_down)
    V_obs = V_obs.reshape(len(V_obs)//n_t_down,n_t_down).mean(1)#avg_freq(V_obs,n_t_down)
    #print(len(freq_test[0]))
    freq_test_down = []
    freq_test_down.append(freq_test[0].reshape(len(freq_test[0])//n_t_down,n_t_down).mean(1))
    freq_test2 = []
    freq_test2.append(freq_test[0].reshape(len(freq_test[0])//n_f,n_f).mean(1))
    

    #get gain cal
    ratio_use,tmp,tmp = gaincal(I_obs,Q_obs,U_obs,V_obs,freq_test_down,stokes=True,deg=deg,plot=False)
    ratio_norm = (ratio_use - np.mean(ratio_use))/np.std(ratio_use)

    #find peaks
    ratio_new = copy.deepcopy(ratio_use)
    #padwidth=10
    #peakheight=4
    pks = find_peaks(np.abs(np.pad(ratio_norm,pad_width=padwidth,mode='constant')),height=peakheight)[0]
    #print(pks)
    pks = np.array(pks)-padwidth#10
    wds = peak_widths(np.abs(np.pad(ratio_norm,pad_width=padwidth,mode='constant')),pks)[0]
    for i in range(len(pks)):
        pk = pks[i]
        wd = int(np.ceil(wds[i])) + 1
        if wd == 0: wd = 1

        low = pk-wd
        hi = pk+wd+1
        if low < 0 :
            low = 0
        if hi >= len(ratio_new):
            hi = len(ratio_new)-1
        #print((low,hi))
        ratio_new[low:hi] = np.mean(ratio_use)

    #plt.figure(figsize=(12,6))
    #plt.plot(freq_test[0],ratio_2)
    #plt.plot(freq_test[0],ratio_new)
    #plt.show()

    #interpolate
    f_ratio = interp1d(freq_test_down[0],ratio_new,kind="linear",fill_value="extrapolate")
    ratio_use_int = f_ratio(freq_test2[0])

    #savgol filter
    if sfwindow == 0:
        sfwindow = int(np.ceil(len(freq_test2[0])/100))
        if sfwindow%2 == 0:
            sfwindow += 1
        if sfwindow <= 1:
            sfwindow = 3
        order = 1
        ratio_use_sf = sf(ratio_use_int,sfwindow,order)
    elif sfwindow != -1:
        ratio_use_sf = sf(ratio_use_int,sfwindow,deg)
    else:
        ratio_use_sf = None

    #fit
    
    #optionally piecewise fit with given parameters
    if edgefreq != -1 and breakfreq != -1:
        edgeidx = np.argmin(np.abs(freq_test2[0]-edgefreq))
        breakidx = np.argmin(np.abs(freq_test2[0]-breakfreq))
        ratio_fit1p,fit_params1 = compute_fit(ratio_use_int[edgeidx:],freq_test2[0][edgeidx:],deg)
        ratio_fit2p,fit_params2 = compute_fit(ratio_use_int[:edgeidx],freq_test2[0][:edgeidx],deg)
        
        ratio_fit1 = np.zeros(len(ratio_use_int))
        for i in range(len(fit_params1)):
            ratio_fit1 += fit_params1[i]*(freq_test2[0]**(len(fit_params1)-i-1))

        ratio_fit2 = np.zeros(len(ratio_use_int))
        for i in range(len(fit_params2)):
            ratio_fit2 += fit_params2[i]*(freq_test2[0]**(len(fit_params2)-i-1))


        ratio_fit = np.zeros(len(ratio_fit1))
        ratio_fit[breakidx:] = ratio_fit1[breakidx:]
        ratio_fit[:breakidx] = ratio_fit2[:breakidx]
        fit_params = [fit_params1,fit_params2]
    else:
        ratio_fit,fit_params = compute_fit(ratio_use_int,freq_test2[0],deg)
    if plot:
        f= plt.figure()
        plt.title(r'Gain Ratio ($g_{xx}/g_{yy}$) ' + label)
        plt.plot(freq_test2[0],ratio_use_int,label="Calculated")
        
        if edgefreq != -1 and breakfreq != -1:
            c=plt.plot(freq_test2[0],ratio_fit,label="Fit")
            plt.plot(freq_test2[0],ratio_fit1,"--",color=c[0].get_color(),label="Fit")
            plt.plot(freq_test2[0],ratio_fit2,"--",color=c[0].get_color(),label="Fit")
        else:
            plt.plot(freq_test2[0],ratio_fit,label="Fit")
        if sfwindow != -1:#len(ratio_use_sf) > 0:
            plt.plot(freq_test2[0],ratio_use_sf,label="Savgol Filter, " + str(sfwindow) + " sample window")
        plt.axhline(np.nanmedian(ratio_use_int),color="black",label="median")
        plt.legend()
        plt.xlabel("Frequency (MHz)")
        plt.grid()
        plt.savefig(datadir + "calibrator_gain_ratio_clean" + label + ext)
        if show:
            plt.show()
        plt.close(f)
    return ratio_use_int,fit_params,ratio_use_sf
    

#Cleans spurious peaks by downsampling and filtering
def cleanphasecal_V2(xx_I_obs,yy_Q_obs,xy_U_obs,yx_V_obs,freq_test,stokes=True,deg=10,plot=False,datadir=DEFAULT_DATADIR,label='',show=False,sfwindow=-1,padwidth=10,peakheight=2,n_t_down=8,n_f=1,edgefreq=-1,breakfreq=-1):

    if stokes:
        I_obs = xx_I_obs
        Q_obs = yy_Q_obs
        U_obs = xy_U_obs
        V_obs = yx_V_obs
    else:
        I_obs = 0.5*(xx_I_obs + yy_Q_obs)
        Q_obs = 0.5*(xx_I_obs - yy_Q_obs)
        U_obs = 0.5*(xy_U_obs + yx_V_obs)#np.real(xy_obs)
        V_obs = -1j*0.5*(xy_U_obs - yx_V_obs)#-np.imag(xy_obs)

    #downsample
    I_obs = I_obs.reshape(len(I_obs)//n_t_down,n_t_down).mean(1)#avg_freq(I_obs,n_t_down)
    Q_obs = Q_obs.reshape(len(Q_obs)//n_t_down,n_t_down).mean(1)#avg_freq(Q_obs,n_t_down)
    U_obs = U_obs.reshape(len(U_obs)//n_t_down,n_t_down).mean(1)#avg_freq(U_obs,n_t_down)
    V_obs = V_obs.reshape(len(V_obs)//n_t_down,n_t_down).mean(1)#avg_freq(V_obs,n_t_down)
    #print(len(freq_test[0]))
    freq_test_down = []
    freq_test_down.append(freq_test[0].reshape(len(freq_test[0])//n_t_down,n_t_down).mean(1))
    freq_test2 = []
    freq_test2.append(freq_test[0].reshape(len(freq_test[0])//n_f,n_f).mean(1))

    #get gain cal
    phase_use,tmp,tmp = phasecal(I_obs,Q_obs,U_obs,V_obs,freq_test_down,stokes=True,deg=deg,plot=False)
    phase_norm = (phase_use - np.mean(phase_use))/np.std(phase_use)

    #find peaks
    phase_new = copy.deepcopy(phase_use)
    #padwidth=10
    #peakheight=4
    pks = find_peaks(np.abs(np.pad(phase_norm,pad_width=padwidth,mode='constant')),height=peakheight)[0]
    #print(pks)
    pks = np.array(pks)-padwidth#10
    wds = peak_widths(np.abs(np.pad(phase_norm,pad_width=padwidth,mode='constant')),pks)[0]
    for i in range(len(pks)):
        pk = pks[i]
        wd = int(np.ceil(wds[i])) + 1
        if wd == 0: wd = 1

        low = pk-wd
        hi = pk+wd+1
        if low < 0 :
            low = 0
        if hi >= len(phase_new):
            hi = len(phase_new)-1
        #print((low,hi))
        phase_new[low:hi] = np.mean(phase_use)

    #interpolate
    f_phase = interp1d(freq_test_down[0],phase_new,kind="linear",fill_value="extrapolate")
    phase_use_int = f_phase(freq_test2[0])

    #savgol filter
    if sfwindow == 0:
        sfwindow = int(np.ceil(len(freq_test2[0])/100))
        if sfwindow%2 == 0:
            sfwindow += 1
        if sfwindow <= 1:
            sfwindow = 3
        order = 1
        phase_use_sf = sf(phase_use_int,sfwindow,order)
    elif sfwindow != -1:
        phase_use_sf = sf(phase_use_int,sfwindow,deg)
    else:
        phase_use_sf = None

    #fit
    
    #optionally piecewise fit with given parameters
    if edgefreq != -1 and breakfreq != -1:
        edgeidx = np.argmin(np.abs(freq_test2[0]-edgefreq))
        breakidx = np.argmin(np.abs(freq_test2[0]-breakfreq))
        phase_fit1p,fit_params1 = compute_fit(phase_use_int[edgeidx:],freq_test2[0][edgeidx:],deg)
        phase_fit2p,fit_params2 = compute_fit(phase_use_int[:edgeidx],freq_test2[0][:edgeidx],deg)

        phase_fit1 = np.zeros(len(phase_use_int))
        for i in range(len(fit_params1)):
            phase_fit1 += fit_params1[i]*(freq_test2[0]**(len(fit_params1)-i-1))

        phase_fit2 = np.zeros(len(phase_use_int))
        for i in range(len(fit_params2)):
            phase_fit2 += fit_params2[i]*(freq_test2[0]**(len(fit_params2)-i-1))


        phase_fit = np.zeros(len(phase_fit1))
        phase_fit[breakidx:] = phase_fit1[breakidx:]
        phase_fit[:breakidx] = phase_fit2[:breakidx]
        fit_params = [fit_params1,fit_params2]
    else:
        phase_fit,fit_params = compute_fit(phase_use_int,freq_test2[0],deg)

    if plot:
        f= plt.figure()
        plt.title(r'Phase Difference ($\phi_{xx} - \phi_{yy}$) ' + label )
        plt.plot(freq_test2[0],phase_use_int,label="Calculated")
        if edgefreq != -1 and breakfreq != -1:
            c=plt.plot(freq_test2[0],phase_fit,label="Fit")
            plt.plot(freq_test2[0],phase_fit1,"--",color=c[0].get_color(),label="Fit")
            plt.plot(freq_test2[0],phase_fit2,"--",color=c[0].get_color(),label="Fit")
        else:
            plt.plot(freq_test2[0],phase_fit,label="Fit")
        if sfwindow != -1:
            plt.plot(freq_test2[0],phase_use_sf,label="Savgol Filter, " + str(sfwindow) + " sample window")
        plt.axhline(np.nanmedian(phase_use_int),color="black",label="median")
        plt.legend()
        plt.xlabel("Frequency (MHz)")
        plt.grid()
        plt.savefig(datadir + "calibrator_phase_diff_" + label + ext)
        if show:
            plt.show()
        plt.close(f)

    return phase_use_int,fit_params,phase_use_sf

#Calculates absolute GY gain from observation of 3C48 and Perly-Butler 2013 polynomial fits
def absgaincal(gain_dir,source_name,obs_name,n_t,n_f,nsamps,deg,ibeam,suffix="_dev",plot=False,show=False,padwidth=10,peakheight=2,n_t_down=32,beamsize_as=14,Lat=37.23,Lon=-118.2851,centerbeam=125):

    if source_name == "3C48":
        RA = RA_3C48
        DEC = DEC_3C48
        RM = RM_3C48
        p = p_3C48
        chip = chip_3C48
    elif source_name == "3C286":
        RA = RA_3C286
        DEC = DEC_3C286
        RM = RM_3C286
        p = p_3C286
        chip = chip_3C286
    else:
        print("Please use valid calibrator")
        return -1


    #read in 3C48 observation
    (Igainuc,Qgainuc,Ugainuc,Vgainuc,fobj,timeaxis,freq_test,wav_test,badchans) = get_stokes_2D(gain_dir,source_name + obs_name + "_dev",5,n_t=n_t,n_f=n_t_down,sub_offpulse_mean=False)

    #PA correction
    Ical,Qcal,Ucal,Vcal,ParA = calibrate_angle(Igainuc,Qgainuc,Ugainuc,Vgainuc,fobj,ibeam,RA,DEC)

    #compute simulated I,Q,U and XX,YY (https://science.nrao.edu/facilities/vla/docs/manuals/obsguide/modes/pol)
    I_sim = PB_flux(coeffs_3C48,freq_test[0]*1e-3) #Jy
    Q_sim = I_sim*(p*np.cos(chip)*np.cos(2*ParA) + p*np.sin(chip)*np.sin(2*ParA))
    U_sim = I_sim*(-p*np.cos(chip)*np.sin(2*ParA) + p*np.sin(chip)*np.cos(2*ParA))
    V_sim = np.zeros(I_sim.shape)

    #apply the measured RM to predict what signal should look like
    I_sim,Q_sim,U_sim,V_sim = calibrate_RM(I_sim,Q_sim,U_sim,V_sim,-RM,0,freq_test,stokes=True)
    XX_sim = 0.5*(I_sim + Q_sim)
    YY_sim = 0.5*(I_sim - Q_sim)

    #compute XX, YY observed after cleaning I for peaks
    #find peaks
    I_new = copy.deepcopy(Igainuc.mean(1))
    I_norm = (Igainuc.mean(1) - np.mean(Igainuc.mean(1)))/np.std(Igainuc.mean(1))
    pks = find_peaks(np.abs(np.pad(I_norm,pad_width=padwidth,mode='constant')),height=peakheight)[0]
    pks = np.array(pks)-padwidth#10
    wds = peak_widths(np.abs(np.pad(I_norm,pad_width=padwidth,mode='constant')),pks)[0]
    for i in range(len(pks)):
        pk = pks[i]
        wd = int(np.ceil(wds[i])) + 1
        if wd == 0: wd = 1

        low = pk-wd
        hi = pk+wd+1
        if low < 0 :
            low = 0
        if hi >= len(I_new):
            hi = len(I_new)-1
        I_new[low:hi] = np.mean(Igainuc.mean(1))

    Q_new = copy.deepcopy(Qgainuc.mean(1))
    Q_norm = (Qgainuc.mean(1) - np.mean(Qgainuc.mean(1)))/np.std(Qgainuc.mean(1))
    pks = find_peaks(np.abs(np.pad(Q_norm,pad_width=padwidth,mode='constant')),height=peakheight)[0]
    pks = np.array(pks)-padwidth#10
    wds = peak_widths(np.abs(np.pad(Q_norm,pad_width=padwidth,mode='constant')),pks)[0]
    for i in range(len(pks)):
        pk = pks[i]
        wd = int(np.ceil(wds[i])) + 1
        if wd == 0: wd = 1

        low = pk-wd
        hi = pk+wd+1
        if low < 0 :
            low = 0
        if hi >= len(I_new):
            hi = len(I_new)-1
        Q_new[low:hi] = np.mean(Qgainuc.mean(1))


    XX = 0.5*(I_new + Q_new)
    YY = 0.5*(I_new - Q_new)
    if plot:

        plt.figure(figsize=(12,6))
        c=plt.plot(freq_test[0]*1e-3,I_new,label="I")
        plt.plot(freq_test[0]*1e-3,I_sim,'--',label="Predicted I (Jy)",color=c[0].get_color())
        c=plt.plot(freq_test[0]*1e-3,XX,label=r'XX (W)')
        plt.plot(freq_test[0]*1e-3,XX_sim,'--',label="Predicted XX (Jy)",color=c[0].get_color())
        c=plt.plot(freq_test[0]*1e-3,YY,label=r'YY (W)')
        plt.plot(freq_test[0]*1e-3,YY_sim,'--',label="Predicted YY (Jy)",color=c[0].get_color())
        plt.legend()
        plt.ylim(0,100)
        plt.savefig(source_name + obs_name + "_fluxprediction.pdf")
        if show:
            plt.show()
        

    #estimated gains
    GX = np.sqrt(XX/XX_sim)
    GY = np.sqrt(YY/YY_sim)

    import numpy.ma as ma 
    GX = ma.masked_invalid(GX, copy=True)
    GY = ma.masked_invalid(GY, copy=True)

    if plot:
        plt.figure(figsize=(12,6))
        plt.plot(freq_test[0],GX,label="GX")
        plt.plot(freq_test[0],GY,label="GY")
        plt.ylabel(r'$Gain = \sqrt{A_{eff}} (m\sqrt{\Omega})$')
        plt.xlabel("Frequency (MHz)")
        plt.legend()
        plt.savefig(source_name + obs_name + "_GYGX.pdf")
        if show:
            plt.show()
        
    #interpolate
    (Igainuc,Qgainuc,Ugainuc,Vgainuc,fobj,timeaxis,freq_test_fullres,wav_test,badchans) = get_stokes_2D(gain_dir,source_name + obs_name + "_dev",5,n_t=n_t,n_f=n_f,sub_offpulse_mean=False)

    idx = np.isfinite(GX)
    f_GX= interp1d(freq_test[0][idx],GX[idx],kind="linear",fill_value="extrapolate")
    GX_fullres = f_GX(freq_test_fullres[0])


    idx = np.isfinite(GY)
    f_GY= interp1d(freq_test[0][idx],GY[idx],kind="linear",fill_value="extrapolate")
    GY_fullres = f_GY(freq_test_fullres[0])

    #polyfit
    popt = np.polyfit(freq_test_fullres[0],GY_fullres,deg=deg)
    #print(len(popt))
    GY_fit = np.zeros(len(GY_fullres))
    for i in range(len(popt)):
        GY_fit += popt[i] * (freq_test_fullres[0]**(deg - i))

    if plot:
        plt.figure(figsize=(12,6))
        plt.plot(freq_test_fullres[0],GY_fullres,label="GY")
        plt.plot(freq_test_fullres[0],GY_fit,label="fit")
        plt.ylabel(r'$Gain = \sqrt{A_{eff}} (m\sqrt{\Omega})$')
        plt.xlabel("Frequency (MHz)")
        plt.legend()
        plt.savefig(source_name + obs_name + "_GY_fit.pdf")
        if show:
            plt.show()
    return GY_fullres,GY_fit,popt



#deprecated
#Cleans spurious peaks by downsampling and filtering
def cleangaincal(xx_I_obs,yy_Q_obs,xy_U_obs,yx_V_obs,freq_test,stokes=True,deg=10,plot=False,datadir=DEFAULT_DATADIR,label='',show=False,sfwindow=-1,padwidth=10,peakheight=2,n_t_down=8,n_f=1):
   
    if stokes:
        I_obs = xx_I_obs
        Q_obs = yy_Q_obs
        U_obs = xy_U_obs
        V_obs = yx_V_obs
    else:
        I_obs = 0.5*(xx_I_obs + yy_Q_obs)
        Q_obs = 0.5*(xx_I_obs - yy_Q_obs)
        U_obs = 0.5*(xy_U_obs + yx_V_obs)#np.real(xy_obs)
        V_obs = -1j*0.5*(xy_U_obs - yx_V_obs)#-np.imag(xy_obs)

    #normalize
    Igain_norm = (I_obs - np.mean(I_obs))/np.std(I_obs)
    Qgain_norm = (Q_obs - np.mean(Q_obs))/np.std(Q_obs)
    Ugain_norm = (U_obs - np.mean(U_obs))/np.std(U_obs)
    Vgain_norm = (V_obs - np.mean(V_obs))/np.std(V_obs)
    
    #get gain cal
    ratio_use,tmp,tmp = gaincal(xx_I_obs,yy_Q_obs,xy_U_obs,yx_V_obs,freq_test,stokes=True,deg=deg,plot=False)
    ratio_norm = (ratio_use - np.mean(ratio_use))/np.std(ratio_use)


    #find peaks and set to average value
    pks = find_peaks(np.abs(np.pad(Igain_norm,pad_width=padwidth,mode='constant')),height=peakheight)[0]
    pks2 = find_peaks(np.abs(np.pad(ratio_norm,pad_width=padwidth,mode='constant')),height=peakheight)[0]
    pks = np.union1d(pks,pks2)
    pks = pks - padwidth

    wds = peak_widths(np.abs(Igain_norm),pks)[0]#peak_widths((I.mean(1) - np.mean(I.mean(1)))/np.std(I.mean(1)),pks)[0]

    Igain_new = copy.deepcopy(I_obs)
    Qgain_new = copy.deepcopy(Q_obs)
    Ugain_new = copy.deepcopy(U_obs)
    Vgain_new = copy.deepcopy(V_obs)
    for i in range(len(pks)):
        pk = pks[i]
        wd = int(np.ceil(wds[i])) + 1
        if wd == 0: wd = 1

        low = pk-wd
        hi = pk+wd+1
        if low < 0 :
            low = 0
        if hi >= len(Igain_new):
            hi = len(Igain_new)-1

        Igain_new[low:hi] = np.mean(I_obs)
        Qgain_new[low:hi]= np.mean(Q_obs)
        Ugain_new[low:hi] = np.mean(U_obs)
        Vgain_new[low:hi]= np.mean(V_obs)


    #recalculate ratio
    ratio_new = (np.sqrt((Igain_new+Qgain_new)/(Igain_new-Qgain_new)))
    ratio_new = np.nan_to_num(ratio_new,nan=np.nanmedian(ratio_new))

    #downsample to average over flattened peaks
    ratio_use_down = ratio_new.reshape(len(ratio_new)//n_t_down,n_t_down).mean(1)
    freq_test_down = []
    freq_test_down.append(freq_test[0].reshape(len(ratio_use)//n_t_down,n_t_down).mean(1))

    #interpolate to desired resolution
    freq_test2 = [freq_test[0].reshape(len(freq_test[0])//n_f,n_f).mean(1)]*4
    f_ratio = interp1d(freq_test_down[0],ratio_use_down,kind="linear",fill_value="extrapolate")
    ratio_use_int = f_ratio(freq_test2[0])

    #savgol filter
    if sfwindow == 0:
        sfwindow = int(np.ceil(len(freq_test2[0])/100))
        if sfwindow%2 == 0:
            sfwindow += 1
        if sfwindow <= 1:
            sfwindow = 3
        deg = 1
        ratio_use_sf = sf(ratio_use_int,sfwindow,deg)
    elif sfwindow != -1:
        ratio_use_sf = sf(ratio_use_int,sfwindow,deg)
    else:
        ratio_use_sf = None

    #fit
    ratio_fit,fit_params = compute_fit(ratio_use_int,freq_test2[0],deg)
    
    if plot:
        f= plt.figure()
        plt.title(r'Gain Ratio ($g_{xx}/g_{yy}$) ' + label)
        plt.plot(freq_test2[0],ratio_use_int,label="Calculated")
        plt.plot(freq_test2[0],ratio_fit,label="Fit")
        if sfwindow != -1:#len(ratio_use_sf) > 0:
            plt.plot(freq_test2[0],ratio_use_sf,label="Savgol Filter, " + str(sfwindow) + " sample window")
        plt.axhline(np.nanmedian(ratio_use_int),color="black",label="median")
        plt.legend()
        plt.xlabel("Frequency (MHz)")
        plt.grid()
        plt.savefig(datadir + "calibrator_gain_ratio_clean" + label + ext)
        if show:
            plt.show()
        plt.close(f)
    return ratio_use_int,fit_params,ratio_use_sf

#Takes observed data products for linear polarized calibrator and calculates phase difference
def phasecal(xx_I_obs,yy_Q_obs,xy_U_obs,yx_V_obs,freq_test,stokes=True,deg=10,plot=False,datadir=DEFAULT_DATADIR,label='',show=False,sfwindow=-1):
    
    if stokes:
        I_obs = xx_I_obs
        Q_obs = yy_Q_obs
        U_obs = xy_U_obs
        V_obs = yx_V_obs
    else: 
        I_obs = 0.5*(xx_I_obs + yy_Q_obs)
        Q_obs = 0.5*(xx_I_obs - yy_Q_obs)
        U_obs = 0.5*(xy_U_obs + yx_V_obs)#np.real(xy_obs)
        V_obs = -1j*0.5*(xy_U_obs - yx_V_obs)#-np.imag(xy_obs)
    
    phase_diff = np.angle(U_obs + 1j*V_obs)
    phase_diff = np.nan_to_num(phase_diff,nan=np.nanmedian(phase_diff))
    phase_diff_fit,fit_params = compute_fit(phase_diff,freq_test[0],deg)

    #mask and savgol filter
    if sfwindow != -1:
        phase_masked = np.ma.masked_outside(phase_diff,np.mean(phase_diff)-3*np.std(phase_diff),np.mean(phase_diff)+3*np.std(phase_diff))
        phase_sf = sf(phase_masked,sfwindow,3)

    if plot:
        f= plt.figure()
        plt.title(r'Phase Difference ($\phi_{xx} - \phi_{yy}$) ' + label )
        plt.plot(freq_test[0],phase_diff,label="Calculated")
        plt.plot(freq_test[0],phase_diff_fit,label="Fit")
        if sfwindow != -1:
            plt.plot(freq_test[0],phase_sf,label="Savgol Filter, " + str(sfwindow) + " sample window")
        plt.axhline(np.nanmedian(phase_diff),color="black",label="median")
        plt.legend()
        plt.xlabel("Frequency (MHz)")
        plt.grid()
        plt.savefig(datadir + "calibrator_phase_diff_" + label + ext)
        if show:
            plt.show()
        plt.close(f)

    if sfwindow != -1:
        return phase_diff,fit_params,phase_sf
    return phase_diff,fit_params,None


#Cleans spurious peaks by downsampling and filtering
def cleanphasecal(xx_I_obs,yy_Q_obs,xy_U_obs,yx_V_obs,freq_test,stokes=True,deg=10,plot=False,datadir=DEFAULT_DATADIR,label='',show=False,sfwindow=-1,padwidth=10,peakheight=2,n_t_down=8,n_f=1):

    if stokes:
        I_obs = xx_I_obs
        Q_obs = yy_Q_obs
        U_obs = xy_U_obs
        V_obs = yx_V_obs
    else:
        I_obs = 0.5*(xx_I_obs + yy_Q_obs)
        Q_obs = 0.5*(xx_I_obs - yy_Q_obs)
        U_obs = 0.5*(xy_U_obs + yx_V_obs)#np.real(xy_obs)
        V_obs = -1j*0.5*(xy_U_obs - yx_V_obs)#-np.imag(xy_obs)

    #normalize
    Iphase_norm = (I_obs - np.mean(I_obs))/np.std(I_obs)
    Qphase_norm = (Q_obs - np.mean(Q_obs))/np.std(Q_obs)
    Uphase_norm = (U_obs - np.mean(U_obs))/np.std(U_obs)
    Vphase_norm = (V_obs - np.mean(V_obs))/np.std(V_obs)

    #get gain cal
    phase_use,tmp,tmp = phasecal(xx_I_obs,yy_Q_obs,xy_U_obs,yx_V_obs,freq_test,stokes=True,deg=deg,plot=False)
    phase_norm = (phase_use - np.mean(phase_use))/np.std(phase_use)

    #find peaks and set to average value
    pks = find_peaks(np.abs(np.pad(Uphase_norm,pad_width=padwidth,mode='constant')),height=peakheight)[0]
    pks2 = find_peaks(np.abs(np.pad(Vphase_norm,pad_width=padwidth,mode='constant')),height=peakheight)[0]
    pks = np.union1d(pks,pks2)
    pks = pks - padwidth
    
    wds = peak_widths(np.abs(Iphase_norm),pks)[0]#peak_widths((I.mean(1) - np.mean(I.mean(1)))/np.std(I.mean(1)),pks)[0]
   
    Iphase_new = copy.deepcopy(I_obs)
    Qphase_new = copy.deepcopy(Q_obs)
    Uphase_new = copy.deepcopy(U_obs)
    Vphase_new = copy.deepcopy(V_obs)
    for i in range(len(pks)):
        pk = pks[i]
        wd = int(np.ceil(wds[i])) + 1
        if wd == 0: wd = 1

        low = pk-wd
        hi = pk+wd+1
        if low < 0 :
            low = 0
        if hi >= len(Iphase_new):
            hi = len(Iphase_new)-1

        Iphase_new[low:hi] = np.mean(I_obs)
        Qphase_new[low:hi]= np.mean(Q_obs)
        Uphase_new[low:hi] = np.mean(U_obs)
        Vphase_new[low:hi]= np.mean(V_obs)

    #recalculate ratio
    phase_new = (np.angle(Uphase_new + 1j*Vphase_new))
    phase_new = np.nan_to_num(phase_new,nan=phase_new)
   

    #downsample to average over flattened peaks
    phase_use_down = phase_new.reshape(len(phase_new)//n_t_down,n_t_down).mean(1)
    freq_test_down = []
    freq_test_down.append(freq_test[0].reshape(len(phase_use)//n_t_down,n_t_down).mean(1))

    #interpolate to desired resolution
    freq_test2 = [freq_test[0].reshape(len(freq_test[0])//n_f,n_f).mean(1)]*4
    f_phase = interp1d(freq_test_down[0],phase_use_down,kind="linear",fill_value="extrapolate")
    phase_use_int = f_phase(freq_test2[0])

    #savgol filter
    if sfwindow == 0:
        sfwindow = int(np.ceil(len(freq_test2[0])/100))
        if sfwindow%2 == 0:
            sfwindow += 1
        if sfwindow <= 1:
            sfwindow = 3
        deg = 1
        phase_use_sf = sf(phase_use_int,sfwindow,deg)
    elif sfwindow != -1:
        phase_use_sf = sf(phase_use_int,sfwindow,deg)
    else:
        phase_use_sf = None

    #fit
    phase_fit,fit_params = compute_fit(phase_use_int,freq_test2[0],deg)

    if plot:
        f= plt.figure()
        plt.title(r'Phase Difference ($\phi_{xx} - \phi_{yy}$) ' + label )
        plt.plot(freq_test2[0],phase_use_int,label="Calculated")
        plt.plot(freq_test2[0],phase_fit,label="Fit")
        if sfwindow != -1:
            plt.plot(freq_test2[0],phase_use_sf,label="Savgol Filter, " + str(sfwindow) + " sample window")
        plt.axhline(np.nanmedian(phase_use_int),color="black",label="median")
        plt.legend()
        plt.xlabel("Frequency (MHz)")
        plt.grid()
        plt.savefig(datadir + "calibrator_phase_diff_" + label + ext)
        if show:
            plt.show()
        plt.close(f)

    return phase_use_int,fit_params,phase_use_sf


#Takes directory with all unpolarized calibrator observations and calibrator name and computes average gain ratio
#vs frequency
def gaincal_full(datadir,source_name,obs_names,n_t,n_f,nsamps,deg,suffix="_dev",average=False,plot=False,show=False,sfwindow=-1,clean=True,padwidth=10,peakheight=2,n_t_down=8,mask=[],edgefreq=-1,breakfreq=-1):
    #if cleaning, always read at full resolution
    if clean:
        n_f_out = n_f
        n_f = 1
        

    ratio_all = []
    if sfwindow != -1:
        ratio_sf_all = []
    fit_params_all = []
    for i in range(len(obs_names)):
        label = source_name + obs_names[i] + suffix
        sdir = datadir + label
        (I,Q,U,V,fobj,timeaxis,freq_test,wav_test,badchans) = get_stokes_2D(datadir,label,nsamps,n_t=n_t,n_f=n_f,n_off=-1,sub_offpulse_mean=False)
        if mask != []:
            I = ma.masked_array(I,mask)
            Q = ma.masked_array(Q,mask)
            U = ma.masked_array(U,mask)
            V = ma.masked_array(V,mask)

        (I_f,Q_f,U_f,V_f) = get_stokes_vs_freq(I,Q,U,V,-1,fobj.header.tsamp,n_f,n_t,freq_test,plot=plot,datadir=datadir,label=label)
        (I_t,Q_t,U_t,V_t) = get_stokes_vs_time(I,Q,U,V,-1,fobj.header.tsamp,n_t,plot=plot,datadir=datadir,label=label,normalize=False)
        if clean:
            ratio,fit_params,ratio_sf = cleangaincal_V2(I_f,Q_f,U_f,V_f,freq_test,stokes=True,deg=deg,plot=plot,datadir=datadir,label=label,sfwindow=sfwindow,padwidth=padwidth,peakheight=peakheight,n_t_down=n_t_down,n_f=n_f_out,edgefreq=edgefreq,breakfreq=breakfreq)
        else:
            ratio,fit_params,ratio_sf = gaincal(I_f,Q_f,U_f,V_f,freq_test,stokes=True,deg=deg,plot=plot,datadir=datadir,label=label,sfwindow=sfwindow)
        
        ratio_all.append(ratio)
        if not average:
            fit_params_all.append(fit_params)
        if sfwindow != -1:
            ratio_sf_all.append(ratio_sf)

    #if cleaned, get correct freq_test
    #(tmp,tmp,tmp,tmp,tmp,tmp,freq_test,tmp) = get_stokes_2D(datadir,label,nsamps,n_t=n_t,n_f=n_f_out,n_off=-1,sub_offpulse_mean=False)
    if clean:
        freq_test = [(freq_test[0].reshape(len(freq_test[0])//n_f_out,n_f_out).mean(1))]*4

    #average together
    if average:
        avg_ratio = np.nanmean(np.array(ratio_all),axis=0)
        avg_ratio_fit,avg_fit_params = compute_fit(avg_ratio,freq_test[0],deg)#np.nanmean(np.array(fit_params_all),axis=0)
        if breakfreq != -1 and edgefreq != -1:
            avg_fit_params = fit_params
        if sfwindow != -1 and not clean:
            ratio_masked = np.ma.masked_outside(avg_ratio,np.mean(avg_ratio)-3*np.std(avg_ratio),np.mean(avg_ratio)+3*np.std(avg_ratio))
            ratio_sf = sf(ratio_masked,sfwindow,3)

        if plot:
            f= plt.figure()
            plt.title(r'Gain Ratio ($g_{xx}/g_{yy}$) ' + label)
            for i in range(len(obs_names)):
                plt.plot(freq_test[0],ratio_all[i],'--',label="Calculated, " + obs_names[i])
            p=plt.plot(freq_test[0],avg_ratio,'-',color="red",label="Averaged Over Observations")
            plt.plot(freq_test[0],avg_ratio_fit,'-',color="gray",label="Fit")
            if sfwindow != -1:
                plt.plot(freq_test[0],ratio_sf,'-',color="green",label="Savgol Filter")
            plt.axhline(np.nanmedian(avg_ratio),color="black",label="median")
            plt.legend()
            plt.xlabel("Frequency (MHz)")
            plt.grid()
            plt.savefig(datadir + "calibrator_gain_ratio_avg_clean_" + source_name + suffix + ext)
            if show:
                plt.show()
            plt.close(f)
        if sfwindow != -1:
            return avg_ratio,avg_fit_params,ratio_sf
        else:
            return avg_ratio,avg_fit_params,None
    elif len(ratio_all) == 1:
        if sfwindow != -1:
            return ratio_all[0],fit_params_all[0],ratio_sf
        else:
            return ratio_all[0],fit_params_all[0],None
    else:
        if sfwindow != -1:
            return ratio_all,fit_params_all,ratio_sf
        else:
            return ratio_all,fit_params_all,None



#Takes directory with all linear calibrator observations and calibrator name and computes average phase difference
#vs frequency
def phasecal_full(datadir,source_name,obs_names,n_t,n_f,nsamps,deg,suffix="_dev",average=False,plot=False,show=False,sfwindow=-1,clean=True,padwidth=10,peakheight=2,n_t_down=8,mask=[],edgefreq=-1,breakfreq=-1):
    
    #if cleaning, always use n_f = 1
    if clean:
        n_f_out = n_f
        n_f = 1

    phase_diff_all = []
    if sfwindow != -1:
        phase_sf_all = []
    fit_params_all = []
    for i in range(len(obs_names)):
        label = source_name + obs_names[i] + suffix
        sdir = datadir + label
        (I,Q,U,V,fobj,timeaxis,freq_test,wav_test,badchans) = get_stokes_2D(datadir,label,nsamps,n_t=n_t,n_f=n_f,n_off=-1,sub_offpulse_mean=False)
        if mask != []:
            I = ma.masked_array(I,mask)
            Q = ma.masked_array(Q,mask)
            U = ma.masked_array(U,mask)
            V = ma.masked_array(V,mask)
        (I_f,Q_f,U_f,V_f) = get_stokes_vs_freq(I,Q,U,V,-1,fobj.header.tsamp,n_f,n_t,freq_test,plot=plot,datadir=datadir,label=label)
        (I_t,Q_t,U_t,V_t) = get_stokes_vs_time(I,Q,U,V,-1,fobj.header.tsamp,n_t,plot=plot,datadir=datadir,label=label,normalize=False)
        if clean:
            phase_diff,fit_params,phase_sf = cleanphasecal_V2(I_f,Q_f,U_f,V_f,freq_test,stokes=True,deg=deg,plot=plot,datadir=datadir,label=label,sfwindow=sfwindow,padwidth=padwidth,peakheight=peakheight,n_t_down=n_t_down,n_f=n_f_out,edgefreq=edgefreq,breakfreq=breakfreq)
        else:
            phase_diff,fit_params,phase_sf = phasecal(I_f,Q_f,U_f,V_f,freq_test,stokes=True,deg=deg,plot=plot,datadir=datadir,label=label)
        phase_diff_all.append(phase_diff)
        if not average:
            fit_params_all.append(fit_params)
        if sfwindow != -1:
            phase_sf_all.append(phase_sf)

    if clean:
        freq_test = [(freq_test[0].reshape(len(freq_test[0])//n_f_out,n_f_out).mean(1))]*4


    #average together
    if average:
        avg_phase_diff = np.nanmean(np.array(phase_diff_all),axis=0)
        avg_phase_diff_fit,avg_fit_params = compute_fit(avg_phase_diff,freq_test[0],deg)#np.nanmean(np.array(fit_params_all),axis=0)
        if breakfreq != -1 and edgefreq != -1:
            avg_fit_params = fit_params
        if sfwindow != -1 and not clean:
            phase_masked = np.ma.masked_outside(avg_phase_diff,np.mean(avg_phase_diff)-3*np.std(avg_phase_diff),np.mean(avg_phase_diff)+3*np.std(avg_phase_diff))
            phase_sf = sf(phase_masked,sfwindow,3)

        if plot:
            f= plt.figure()
            plt.title(r'Phase Difference ($\phi_{xx} - \phi_{yy}$) ' + label )
            for i in range(len(obs_names)):
                plt.plot(freq_test[0],phase_diff_all[i],'--',label="Calculated, " + obs_names[i])
            p=plt.plot(freq_test[0],avg_phase_diff,'-',color="red",label="Averaged Over Observations")
            plt.plot(freq_test[0],avg_phase_diff_fit,'-',color="gray",label="Fit")
            if sfwindow != -1:
                plt.plot(freq_test[0],phase_sf,'-',color="green",label="Savgol Filter")
            plt.axhline(np.nanmedian(avg_phase_diff),color="black",label="median")
            plt.legend()
            plt.xlabel("Frequency (MHz)")
            plt.grid()
            plt.savefig(datadir + "calibrator_phase_diff_avg_" + source_name + suffix + ext)
            if show:
                plt.show()
            plt.close(f)

        if sfwindow != -1:
            return avg_phase_diff,avg_fit_params,phase_sf
        else:
            return avg_phase_diff,avg_fit_params,None
    elif len(phase_diff_all) == 1:
        if sfwindow != -1:
            return phase_diff_all[0],fit_params_all[0],phase_sf
        else:
            return phase_diff_all[0],fit_params_all[0],None
    else:
        if sfwindow != -1:
            return phase_diff_all,fit_params_all,phase_sf
        else:
            return phase_diff_all,fit_params_all,None



#Takes datx a products for unpolarized calibrator and linearly polarized calibrator and returns Jones matrix 
# assuming gxy = gyx = 0 and given gxx
def get_calmatrix_from_ratio_phasediff(ratio,phase_diff,gyy_mag=1,gyy_phase=0):#(gain_calibrator,phase_calibrator,gxx_mag=1,gxx_phase=0):
    #ratio = gaincal(gain_calibrator[0],gain_calibrator[1],gain_calibrator[2],gain_calibrator[3])
    #phase_diff = phasecal(phase_calibrator[0],phase_calibrator[1],phase_calibrator[2],phase_calibrator[3])
    
    #Calculate gyy
    #gxx = gxx_mag*np.exp(1j*gxx_phase)*np.ones(np.shape(ratio))
    #gyy = (gxx_mag/ratio)*np.exp(1j*(gxx_phase-phase_diff))
    gxx = ratio*gyy_mag*np.exp(1j*(phase_diff+gyy_phase))
    gyy = gyy_mag*np.exp(1j*gyy_phase)*np.ones(np.shape(ratio))

    return [gxx,gyy]

#Takes data products for target and returns calibrated Stokes parameters
def calibrate(xx_I_obs,yy_Q_obs,xy_U_obs,yx_V_obs,calmatrix,stokes=True,multithread=False,maxProcesses=100,idx=np.nan):
    #calculate Stokes parameters
    if stokes:
        I_obs = xx_I_obs
        Q_obs = yy_Q_obs
        U_obs = xy_U_obs
        V_obs = yx_V_obs
    else: 
        I_obs = 0.5*(xx_I_obs + yy_Q_obs)
        Q_obs = 0.5*(xx_I_obs - yy_Q_obs)
        U_obs = 0.5*(xy_U_obs + yx_V_obs)#np.real(xy_obs)
        V_obs = -1j*0.5*(xy_U_obs - yx_V_obs)#-np.imag(xy_obs)
        
    #get gain params
    (gxx,gyy) = calmatrix

    #if 2D, duplicate gxx gyy
    if len(I_obs.shape) == 2:
        #print("2D Calibration")
        gxx_cal = np.transpose(np.tile(gxx,(I_obs.shape[1],1)))
        gyy_cal = np.transpose(np.tile(gyy,(I_obs.shape[1],1)))
        print(gxx_cal.shape,gyy_cal.shape,I_obs.shape)
    else:
        print("1D Calibration")
        gxx_cal = gxx
        gyy_cal = gyy
        print(gxx_cal.shape,gyy_cal.shape,I_obs.shape)


    if multithread:
        I_true = np.zeros(I_obs.shape)
        Q_true = np.zeros(Q_obs.shape)
        U_true = np.zeros(U_obs.shape)
        V_true = np.zeros(V_obs.shape)

        executor = ProcessPoolExecutor(maxProcesses)
        chunk_size = I_obs.shape[1]//maxProcesses
        task_list = []
        for i in range(maxProcesses):

            I_obs_i = I_obs[:,i*chunk_size:(i+1)*chunk_size]
            Q_obs_i = Q_obs[:,i*chunk_size:(i+1)*chunk_size]
            U_obs_i = U_obs[:,i*chunk_size:(i+1)*chunk_size]
            V_obs_i = V_obs[:,i*chunk_size:(i+1)*chunk_size]

            task_list.append(executor.submit(calibrate,I_obs_i,Q_obs_i,U_obs_i,V_obs_i,calmatrix,True,False,1,i))

        if I_obs.shape[1]%maxProcesses != 0:
            i = maxProcesses
            I_obs_i = I_obs[:,maxProcesses*chunk_size:]
            Q_obs_i = Q_obs[:,maxProcesses*chunk_size:]
            U_obs_i = U_obs[:,maxProcesses*chunk_size:]
            V_obs_i = V_obs[:,maxProcesses*chunk_size:]

            task_list.append(executor.submit(calibrate,I_obs_i,Q_obs_i,U_obs_i,V_obs_i,calmatrix,True,False,1,i))

        for future in as_completed(task_list):
            I_true_i,Q_true_i,U_true_i,V_true_i,i = future.result()

            if i == maxProcesses:
                I_true[:,maxProcesses*chunk_size:] = I_true_i
                Q_true[:,maxProcesses*chunk_size:] = Q_true_i
                U_true[:,maxProcesses*chunk_size:] = U_true_i
                V_true[:,maxProcesses*chunk_size:] = V_true_i
            else:
                I_true[:,i*chunk_size:(i+1)*chunk_size] = I_true_i
                Q_true[:,i*chunk_size:(i+1)*chunk_size] = Q_true_i
                U_true[:,i*chunk_size:(i+1)*chunk_size] = U_true_i
                V_true[:,i*chunk_size:(i+1)*chunk_size] = V_true_i

        executor.shutdown(wait=True)

    else:

        I_true = 0.5*((((np.abs(1/gxx_cal)**2))*(I_obs + Q_obs)) + (((np.abs(1/gyy_cal)**2))*(I_obs - Q_obs)))
        Q_true = 0.5*((((np.abs(1/gxx_cal)**2))*(I_obs + Q_obs)) - (((np.abs(1/gyy_cal)**2))*(I_obs - Q_obs)))
    
        xy_obs = U_obs + 1j*V_obs
        xy_cal = xy_obs / (gxx_cal*np.conj(gyy_cal))#* np.exp(-1j * (np.angle(gxx_cal)-np.angle(gyy_cal)))
        U_true, V_true = xy_cal.real, xy_cal.imag
    if ~np.isnan(idx):
        return (I_true,Q_true,U_true,V_true,idx)
    return (I_true,Q_true,U_true,V_true)


#Takes data products for target and returns calibrated Stokes parameters; implemented with matrices
def calibrate_matrix(xx_I_obs,yy_Q_obs,xy_U_obs,yx_V_obs,calmatrix,stokes=True):
    #calculate Stokes parameters
    if stokes:
        I_obs = xx_I_obs
        Q_obs = yy_Q_obs
        U_obs = xy_U_obs
        V_obs = yx_V_obs
    else: 
        I_obs = 0.5*(xx_I_obs + yy_Q_obs)
        Q_obs = 0.5*(xx_I_obs - yy_Q_obs)
        U_obs = 0.5*(xy_U_obs + yx_V_obs)#np.real(xy_obs)
        V_obs = -1j*0.5*(xy_U_obs - yx_V_obs)#-np.imag(xy_obs)
    
    I_true = np.zeros(np.shape(I_obs))
    Q_true = np.zeros(np.shape(I_obs))
    U_true = np.zeros(np.shape(I_obs))
    V_true = np.zeros(np.shape(I_obs))
    
    #get gain params
    (gxx,gyy) = calmatrix
    
    #Get true stokes at each frequency
    for i in range(len(I_obs)):
        #Jones matrix
        J = np.array([[gxx[i],0],[0,gyy[i]]])
        J_dagger = np.conj(np.transpose(J))
        J_inv = np.linalg.inv(J)
        J_dagger_inv = np.linalg.inv(J_dagger)
        
        #Coherency matrix
        C_obs = np.array([[I_obs[i] + Q_obs[i], U_obs[i] + 1j*V_obs[i]],[U_obs[i] - 1j*V_obs[i], I_obs[i] - Q_obs[i]]])
        
        #True coherency matrix
        tmp = np.dot(C_obs,J_dagger_inv)
        C_true = np.dot(J_inv,tmp)
        
        #Get true stokes
        I_true[i] = 0.5*(C_true[0,0] + C_true[1,1])
        Q_true[i] = 0.5*(C_true[0,0] - C_true[1,1])
        U_true[i] = np.real(C_true[0,1])
        V_true[i] = np.imag(C_true[0,1])
    
    return (I_true,Q_true,U_true,V_true)
    

#Apply parallactic angle correction
def calibrate_angle(xx_I_obs,yy_Q_obs,xy_U_obs,yx_V_obs,fobj,ibeam,RA,DEC,beamsize_as=14,Lat=37.23,Lon=-118.2851,centerbeam=125,stokes=True,verbose=False):
    #calculate Stokes parameters
    if stokes:
        I_obs = xx_I_obs
        Q_obs = yy_Q_obs
        U_obs = xy_U_obs
        V_obs = yx_V_obs
    else:
        I_obs = 0.5*(xx_I_obs + yy_Q_obs)
        Q_obs = 0.5*(xx_I_obs - yy_Q_obs)
        U_obs = 0.5*(xy_U_obs + yx_V_obs)#np.real(xy_obs)
        V_obs = -1j*0.5*(xy_U_obs - yx_V_obs)#-np.imag(xy_obs)

    #antenna properties
    chi_ant = np.pi/2
    synth_beamsize = (np.pi/180) * beamsize_as / 3600  #rad, need to verify beam size with Vikram and Liam
    #centerbeam = 125
    #Lat = 37.23 #degrees
    #Lon = -118.2951 # degrees
    observing_location = EarthLocation(lat=Lat*u.deg, lon=Lon*u.deg)

    #compute parallactic angle for the observation to correct 
    #ids = "221025aanu"
    #nickname = "220912A"
    #FRBidx=19
    #ibeam = 163
    #datadir = "/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/"+ids + "_" + nickname + "/"
    #(I,Q,U,V,fobj,timeaxis,freq_test,wav_test) = dsapol.get_stokes_2D(datadir,ids + "_dev",20480,n_t=10240,n_f=6144,n_off=int(12000//n_t),sub_offpulse_mean=True)


    #OVRO
    ##Lat = 37.23 #degrees
    #Lon = -118.2951 # degrees
    #observing_location = EarthLocation(lat=Lat*u.deg, lon=Lon*u.deg)


    #RA = FRB_RA[FRBidx]
    #DEC = FRB_DEC[FRBidx]

    if verbose:
        print("FRB RA: " + str(RA) + " degrees")
        print("FRB DEC: " + str(DEC) + " degrees")

    RA = RA*np.pi/180
    DEC = DEC*np.pi/180

    #use hour angle for beam nearest center to estimate actual elevation
    observing_time = Time(fobj.header.tstart, format='mjd', location=observing_location)#Time(fobj.header.tstart, format='mjd', location=observing_location)
    GST = Time(fobj.header.tstart, format='mjd').to_datetime().hour + Time(fobj.header.tstart, format='mjd').to_datetime().minute/60 + Time(fobj.header.tstart, format='mjd').to_datetime().second/3600#observing_time.sidereal_time('mean').hour
    LST = (GST + Lon*24/360)%24
    if verbose:
        print(observing_time.isot)
        print(LST)

    HA = (LST*360/24)*np.pi/180 - RA #rad


    elev = np.arcsin(np.sin(Lat*np.pi/180)*np.sin(DEC) + np.cos(Lat*np.pi/180)*np.cos(DEC)*np.cos(HA))
    if verbose:
        print("elevation est. : " + str(elev*180/np.pi) + " degrees")

    #apply correction to azimuth due to beam offset
    az = np.arcsin(-np.cos(DEC)*np.sin(HA)/np.cos(elev))
    az_corr = (ibeam - centerbeam)*synth_beamsize
    
    if verbose:
        print("initial azimuth est. : " + str(az*180/np.pi) + " degrees")
        print("corrected azimuth est. : " + str((az-az_corr)*180/np.pi) + " degrees")

    ParA = np.arcsin(np.sin(HA)*np.cos(Lat*np.pi/180)/np.cos(elev)) + chi_ant
    if verbose:
        print("initial parallactic angle est : " + str(ParA*180/np.pi) + " degrees")

    #ParA = np.arcsin(-np.sin(az-az_corr)*np.cos(Lat*np.pi/180)/np.cos(DEC)) + chi_ant
    arg = -np.sin(az-az_corr)*np.cos(Lat*np.pi/180)/np.cos(DEC)
    if arg > 1:
        arg = 1
    elif arg < -1:
        arg = -1

    ParA = np.arcsin(arg) + chi_ant
    
    if verbose:
        print("corrected parallactic angle est : " + str(ParA*180/np.pi) + " degrees")

    #calibrate
    Qcal = Q_obs*np.cos(2*ParA) - U_obs*np.sin(2*ParA)
    Ucal = Q_obs*np.sin(2*ParA) + U_obs*np.cos(2*ParA)

    return I_obs,Qcal,Ucal,V_obs,ParA

    




#Estimate RM by maximizing SNR over given trial RM; wav is wavelength array
def faradaycal(I,Q,U,V,freq_test,trial_RM,trial_phi,plot=False,datadir=DEFAULT_DATADIR,calstr="",label="",n_f=1,n_t=1,show=False,fit_window=100,err=True,matrixmethod=False,multithread=False,maxProcesses=10,numbatch=1,mt_offset=0,sendtodir='',monitor=False): 
    if multithread: assert(numbatch*maxProcesses <= len(trial_RM)) #assert(len(trial_RM)%(maxProcesses*numbatch) == 0)
    #Get wavelength axis
    c = (3e8) #m/s
    wav = c/(freq_test[0]*(1e6))#wav_test[0]

    #Calculate polarization        
    P = Q + 1j*U
    
    if plot:
        f=plt.figure(figsize=(12,6))
        plt.plot(wav,np.real(P),label="Q")
        plt.plot(wav,np.imag(P),label="U")
        plt.grid()
        plt.xlabel("wavelength (m)")
        plt.legend()
        plt.savefig(datadir + label + "_QU_before_" +calstr + str(n_f) + "_binned" + ext)
        if show:
            plt.show()
        plt.close(f)
    
    #SNR matrix
    SNRs = np.zeros((len(trial_RM),len(trial_phi)))
    
    #matrix implementation
    if matrixmethod:
        RMmesh,wavmesh = np.meshgrid(trial_RM,wav) 
         
        for j in range(len(trial_phi)):
       
            cterm = np.matrix(np.cos(2*RMmesh*((wavmesh**2) - np.mean(wav**2)) + trial_phi[j]))
            sterm = np.matrix(np.sin(2*RMmesh*((wavmesh**2) - np.mean(wav**2)) + trial_phi[j]))
            SNRs[:,j] = np.array(np.abs(np.matrix(Q)*cterm + np.matrix(U)*sterm + 1j*(np.matrix(Q)*sterm - np.matrix(U)*cterm)))[0,:]
    
    else:


        if multithread:

            if monitor: itrf = tqdm(range(numbatch),token=os.environ["RMSYNTHTOKEN"],channel="rmsynthstatus",position=0,desc=str(numbatch) + " Batches")
            else: itrf = range(numbatch)
            for j in itrf:

                #create executor
                executor = ProcessPoolExecutor(maxProcesses)

                task_list = []
                trialsize = int(len(trial_RM)//(numbatch*maxProcesses))
                for i in range(maxProcesses*j,maxProcesses*(j+1)):
                    trial_RM_i = trial_RM[i*trialsize:(i+1)*trialsize]

                    #start thread
                    task_list.append(executor.submit(faradaycal,I,Q,U,V,freq_test,trial_RM_i,trial_phi,
                                            False,datadir,calstr,label,n_f,n_t,False,fit_window,False,
                                            False,False,1,1,i,sendtodir,monitor))

                if j == numbatch-1 and len(trial_RM)%(numbatch*maxProcesses) != 0:
                    trial_RM_i = trial_RM[numbatch*maxProcesses:]

                    task_list.append(executor.submit(faradaycal,I,Q,U,V,freq_test,trial_RM_i,trial_phi,
                                            False,datadir,calstr,label,n_f,n_t,False,fit_window,False,
                                            False,False,1,1,maxProcesses*numbatch,sendtodir,monitor))


                #wait for tasks to complete
                #wait(task_list)
                #print("done submitting tasks")
                for future in as_completed(task_list):
                    tmp,tmp,SNRs_i,tmp,i = future.result()
                    if i == maxProcesses*numbatch:
                        SNRs[numbatch*maxProcesses:,0] = SNRs_i
                    else:
                        SNRs[i*trialsize:(i+1)*trialsize,0] = SNRs_i


                    if sendtodir != '':
                        prevSNRs = np.load(sendtodir + "/SNRs.npy")
                        prevSNRs = np.concatenate([prevSNRs,SNRs_i])
                        np.save(sendtodir + "/SNRs.npy",prevSNRs)


                        prevRMs = np.load(sendtodir + "/trialRM.npy")
                        if i == maxProcesses*numbatch:
                            prevRMs = np.concatenate([prevRMs,trial_RM[maxProcesses*numbatch:]])
                        else:
                            prevRMs = np.concatenate([prevRMs,trial_RM[i*trialsize:(i+1)*trialsize]])
                        np.save(sendtodir + "/trialRM.npy",prevRMs)
                executor.shutdown(wait=True)
            

   
        else:
            for i in range(len(trial_RM)):
                for j in range(len(trial_phi)):
                    RM_i = trial_RM[i]
                    phi_j = trial_phi[j]

                    P_trial = P*np.exp(-1j*((2*RM_i*(((wav)**2) - np.mean((wav)**2))) + phi_j))

                    SNRs[i,j] = np.abs(np.mean(P_trial))
    

    if plot:

        f=plt.figure(figsize=(12,6))
        plt.grid()  
        plt.plot(trial_RM,SNRs)
        plt.xlabel("Trial RM")  
        plt.ylabel("Dispersion Function F(" + r'$\phi$' + ")")
        plt.title(label)
        plt.savefig(datadir + label + "_faraday1D_" + calstr + str(n_f) + "_binned" + ext)
        if show:
            plt.show()
        plt.close(f)
        
       
        if len(trial_RM) > 1 and len(trial_phi) >1 and (not matrixmethod):
            f=plt.figure(figsize=(12,6))
            plt.imshow(SNRs,aspect='auto')
            plt.colorbar()
            plt.ylabel("trial RM")
            plt.xlabel("trial phi")
            #plt.yticks(ticks=np.arange(0,len(trial_RM),20),labels=np.around(trial_RM[::20],1))
            #plt.xticks(ticks=np.arange(0,len(trial_phi),1),labels=np.around(trial_phi[::1],1))
            plt.savefig(datadir + label + "_faraday2D_" + calstr + str(n_f) + "_binned" + ext)
            if show:
                plt.show()
            plt.close(f)
    
            f=plt.figure(figsize=(12,6))
            plt.imshow(SNRs,aspect='auto')
            plt.colorbar()
            plt.ylabel("trial RM")
            plt.xlabel("trial phi")
            plt.xscale("log")
            #plt.yticks(ticks=np.arange(0,len(trial_RM),20),labels=np.around(trial_RM[::20],1))
            #plt.xticks(ticks=np.arange(0,len(trial_phi),1),labels=np.around(trial_phi[::1],1))
            plt.savefig(datadir + label + "_faraday2Dlog_" + calstr + str(n_f) + "_binned" + ext)
            if show:
                plt.show()
            plt.close(f)

    (max_RM_idx,max_phi_idx) = np.unravel_index(np.argmax(SNRs),np.shape(SNRs))
    P_derot = P*np.exp(-1j*2*trial_RM[max_RM_idx]*(wav**2))*np.exp(-1j*2*trial_RM[max_RM_idx]*(np.mean(wav**2)))
    if plot:
        f=plt.figure(figsize=(12,6))
        plt.plot(wav,np.real(P_derot),label="Q")
        plt.plot(wav,np.imag(P_derot),label="U")
        plt.grid()
        plt.xlabel("wavelength (m)")
        plt.legend()
        plt.title("De-rotated")
        plt.savefig(datadir + label + "_QU_after_" + calstr + str(n_f) + "_binned" + ext)
        if show:
            plt.show()
        plt.close(f)


    if err:
        #Estimate error by fitting parabola and finding FWHM
        fit_sample_RM = trial_RM[max_RM_idx-fit_window:max_RM_idx+fit_window]
        fit_sample_SNRs = SNRs[max_RM_idx-fit_window:max_RM_idx+fit_window]
    
        sigmas = np.zeros(len(fit_sample_RM))
        for i in range(len(fit_sample_RM)):
            sigmas[i]=(faraday_error(Q,U,freq_test,fit_sample_RM[i],0))



        popt,pcov = np.polyfit(fit_sample_RM,fit_sample_SNRs,2,w=1/sigmas,cov=True)
        fit_RM = np.linspace(trial_RM[max_RM_idx-fit_window],trial_RM[max_RM_idx+fit_window],1000)
        fit = np.zeros(1000)
        for i in range(len(popt)):
            fit += popt[i]*(fit_RM**(len(popt)-1-i))


        HWHM_RM = trial_RM[max_RM_idx]
        step = 0.001
        while SNR_fit(HWHM_RM,popt) > np.max(SNRs[:,0])/2:
            HWHM_RM += step


        RM_err = (HWHM_RM - trial_RM[max_RM_idx])

        if plot:

            f=plt.figure(figsize=(12,6))
            plt.grid()
            plt.plot(fit_sample_RM,fit_sample_SNRs,'o',label="Peak")
            plt.plot(fit_RM,fit,label="Parabolic Fit")
            #plt.plot(trial_RM,SNRs)
            plt.axvline(trial_RM[max_RM_idx]+RM_err,color="red")
            plt.axvline(trial_RM[max_RM_idx]-RM_err,label="Parabolic Fit HWHM, " + str(np.around(RM_err,2)) + "",color="red")
            plt.xlabel("Trial RM")
            plt.ylabel("Dispersion Function F(" + r'$\phi$' + ")")
            plt.title(label)
            plt.savefig(datadir + label + "_faraday1Derr_" + calstr + str(n_f) + "_binned" + ext)
            if show:
                plt.show()
            plt.close(f)



    else:
        RM_err = None

    return (trial_RM[max_RM_idx],trial_phi[max_phi_idx],SNRs[:,0],RM_err,mt_offset)

def SNR_fit(RM,popt):
    fit = 0
    for i in range(len(popt)):
        fit += popt[i]*(RM**(len(popt)-1-i))
    return fit

#Estimate of error on Faraday dispersion function points for given RM estimate; note this
#takes uncalibrated Q and U
def faraday_error(Q_f,U_f,freq_test,RM,phi=0):
    #Get wavelength axis
    c = (3e8) #m/s
    wav = c/(freq_test[0]*(1e6))#wav_test[0]
    
    P_f_cal_RM = (Q_f + 1j*U_f)*np.exp(-1j*((2*(RM-100)*((np.array(wav))**2)) + phi))

    P_mag = np.abs(np.sum(P_f_cal_RM))
    #print(P_mag)
    sigma_Q = np.sqrt(np.mean(np.real(P_f_cal_RM)**2))
    sigma_U = np.sqrt(np.mean(np.imag(P_f_cal_RM)**2))
    #print(sigma_Q,sigma_U)
    sigma = sigma_Q
    SNR = P_mag*np.sqrt((len(wav) -2))/sigma
    #print(SNR)
    sigma_lambda = np.sqrt(np.sum(np.array(wav)**4)/(len(wav) - 1))
    #print(sigma_lambda)
    sigma_chi = 0.5*sigma/P_mag
    #print(sigma_chi)
    sigma_RM = sigma_chi/(sigma_lambda*np.sqrt(len(wav)- 2))
    #print(sigma_RM)
    #print(len(wav_test[0]))
    #print(np.sum(np.array(wav_test[0])**4))
    method1HWHM = sigma_RM
    return method1HWHM

#Specific Faraday calibration to get SNR spectrum in range around peak
def faradaycal_SNR(I,Q,U,V,freq_test,trial_RM,trial_phi,width_native,t_samp,plot=False,datadir=DEFAULT_DATADIR,calstr="",label="",n_f=1,n_t=1,show=False,err=True,buff=0,weighted=False,n_t_weight=1,timeaxis=None,sf_window_weights=45,n_off=3000,full=False,input_weights=[],timestart_in=-1,timestop_in=-1,matrixmethod=False,multithread=False,maxProcesses=10,numbatch=1,mt_offset=0,sendtodir='',monitor=False):
    if multithread: assert(numbatch*maxProcesses <= len(trial_RM))
    #Get wavelength axis
    c = (3e8) #m/s
    wav = c/(freq_test[0]*(1e6))#wav_test[0]
    #wavall = np.array([wav]*np.shape(I)[1]).transpose()
    #print(np.any(np.isnan(wavall)))
    #use full timestream for calibrators
    if width_native == -1:
        timestart = 0
        timestop = I.shape[1]
    elif input_weights != []:
        #print("Using Custom Input Weights")
        timestart = timestart_in 
        timestop = timestop_in
    else:
        peak,timestart,timestop = find_peak(I,width_native,t_samp,n_t,buff=buff)


    if weighted:
        #get weights
        if input_weights!=[]:
            I_t_weights = input_weights
        else:
            I_t_weights=get_weights(I,Q,U,V,width_native,t_samp,n_f,n_t,freq_test,timeaxis,fobj=None,n_off=n_off,buff=buff,n_t_weight=n_t_weight,sf_window_weights=sf_window_weights,padded=False)
        I_t_weights_2D = np.array([I_t_weights]*I.shape[0])


    #Calculate polarization
    P = Q + 1j*U
    #print(P)
    #SNR matrix
    SNRs = np.zeros((len(trial_RM),len(trial_phi)))

    #get snr at RM =0  to estimate significance
    
    L0_t = np.sqrt(np.mean(Q,axis=0)**2 + np.mean(U,axis=0)**2)
    #eps = 1e-10
    if weighted:
        L_trial_binned = (convolve(L0_t,I_t_weights))
        sigbin = np.argmax(L_trial_binned)
        sig0 = L_trial_binned[sigbin]

        Q_binned = (convolve(Q.mean(0),I_t_weights))#,mode="same"))
        noise = np.std(np.concatenate([Q_binned[:sigbin-(timestop-timestart)*2],Q_binned[sigbin+(timestop-timestart)*2:]]))
        if plot:
            plt.figure(figsize=(12,6))
            plt.plot(L_trial_binned,alpha=0.5,color="gray")
            plt.plot(Q_binned)
            plt.axvline(sigbin-(timestop-timestart)*2)
            plt.axvline(sigbin+(timestop-timestart)*2)
            plt.plot(noise)
            plt.show()

    else:
        
        sig0 = np.mean(np.sqrt(np.mean(Q,axis=0)**2 + np.mean(U,axis=0)**2)[timestart:timestop])
        L_trial_cut1 = L0_t[timestart%(timestop-timestart):]
        L_trial_cut = L_trial_cut1[:(len(L_trial_cut1)-(len(L_trial_cut1)%(timestop-timestart)))]
        L_trial_binned = L_trial_cut.reshape(len(L_trial_cut)//(timestop-timestart),timestop-timestart).mean(1)
        sigbin = np.argmax(L_trial_binned)
        #noise = (np.std(np.concatenate([L_trial_cut[:sigbin],L_trial_cut[sigbin+1:]])))
        Q_cut = Q.mean(0)[timestart%(timestop-timestart):]
        Q_cut = Q_cut[:(len(Q_cut)-(len(Q_cut)%(timestop-timestart)))]
        Q_binned = Q_cut.reshape(len(Q_cut)//(timestop-timestart),timestop-timestart).mean(1)
        Qnoise = (np.std(np.concatenate([Q_binned[:sigbin],Q_binned[sigbin+1:]])))
        U_cut = U.mean(0)[timestart%(timestop-timestart):]
        U_cut = U_cut[:(len(U_cut)-(len(U_cut)%(timestop-timestart)))]
        U_binned = U_cut.reshape(len(U_cut)//(timestop-timestart),timestop-timestart).mean(1)
        Unoise = (np.std(np.concatenate([U_binned[:sigbin],U_binned[sigbin+1:]])))
        noise = Qnoise#np.sqrt((Q_binned[sigbin]*Qnoise)**2  + (U_binned[sigbin]*Unoise)**2)/sig0
    #print(noise)        
    snr0 = sig0/noise

    P_cut = P[:,timestart:timestop]
    wavall_cut = np.array([wav]*np.shape(P_cut)[1]).transpose()
    #if weighted:
    #    I_t_weights_cut = I_t_weights#I_t_weights[timestart:timestop]
    #print(P_cut.shape,wavall_cut.shape)

    #if full return full time vs RM matrix to estimate RM variation over the pulse width
    if full:
        SNRs_full = np.zeros((len(trial_RM),timestop-timestart))

    #matrix method
    if matrixmethod:
        RMmesh,wavmesh = np.meshgrid(trial_RM,wav)

        for j in range(len(trial_phi)):

            cterm = np.matrix(np.cos(2*RMmesh*((wavmesh**2) - np.mean(wav**2)) + trial_phi[j]))
            sterm = np.matrix(np.sin(2*RMmesh*((wavmesh**2) - np.mean(wav**2)) + trial_phi[j]))
            
            sig_all = np.zeros(len(trial_RM))
            for tidx in range(len(I_t_weights)):
                
                
                Q_cut = np.matrix(np.real(P_cut[:,tidx]))
                U_cut = np.matrix(np.imag(P_cut[:,tidx]))
                L_cut = np.array(np.abs(Q_cut*cterm + U_cut*sterm + 1j*(Q_cut*sterm - U_cut*cterm)))[0,:]
                
                if full:
                    SNRs_full[:,tidx] = L_cut#np.array(np.abs(Q_cut*cterm + U_cut*sterm + 1j*(Q_cut*sterm - U_cut*cterm)))[0,:]/len(wavall_cut)
                
                if weighted:
                    sig_all[tidx] += L_cut*I_t_weights[tidx]
                else:
                    sig_all[tidx] += L_cut
            SNRs[:,j] = sig_all/noise

    else:

        if multithread:


            if monitor: itrf = tqdm(range(numbatch),token=os.environ["RMSYNTHTOKEN"],channel="rmsynthstatus",position=0,desc="2D RM Synthesis;" + str(numbatch) + " Batches")
            else: itrf = range(numbatch)
            for j in itrf:

                #create executor
                #with ProcessPoolExecutor(maxProcesses) as executor:
                executor = ProcessPoolExecutor(maxProcesses)

                task_list = []
                trialsize = int(len(trial_RM)//(numbatch*maxProcesses))
                for i in range(maxProcesses*j,maxProcesses*(j+1)):
                    trial_RM_i = trial_RM[i*trialsize:(i+1)*trialsize]

                    #start thread
                    task_list.append(executor.submit(faradaycal_SNR,I,Q,U,V,freq_test,trial_RM_i,trial_phi,width_native,
                                                                                                t_samp,False,datadir,calstr,label,n_f,n_t,False,
                                                                                                False,buff,weighted,n_t_weight,timeaxis,
                                                                                                sf_window_weights,n_off,full,input_weights,
                                                                                                timestart_in,timestop_in,False,False,1,1,i,sendtodir,monitor))
                        
                        
                if j == numbatch-1 and len(trial_RM)%(numbatch*maxProcesses) != 0:
                    trial_RM_i = trial_RM[numbatch*maxProcesses:]


                    task_list.append(executor.submit(faradaycal_SNR,I,Q,U,V,freq_test,trial_RM_i,trial_phi,width_native,
                                                                                                t_samp,False,datadir,calstr,label,n_f,n_t,False,
                                                                                                False,buff,weighted,n_t_weight,timeaxis,
                                                                                                sf_window_weights,n_off,full,input_weights,
                                                                                                timestart_in,timestop_in,False,False,1,1,numbatch*maxProcesses,sendtodir,monitor))

                    """

                    if full:
                        tmp,tmp,SNRs_i,tmp,tmp,tmp,tmp,tmp,SNRs_full_i,tmp,i = faradaycal_SNR(I,Q,U,V,freq_test,trial_RM_i,trial_phi,
                                                    width_native,t_samp,False,datadir,calstr,label,n_f,n_t,
                                                    show,False,buff,weighted,n_t_weight,timeaxis,fobj,sf_window_weights,
                                                    n_off,full,input_weights,timestart_in,timestop_in,False,False,
                                                    1,1,i)
                        SNRs_full[i*trialsize:(i+1)*trialsize,:] = SNRs_full_i
                        SNRs[i*trialsize:(i+1)*trialsize,0] = SNRs_i
                    else:
                        tmp,tmp,SNRs_i,tmp,tmp,tmp,tmp,tmp,i = faradaycal_SNR(I,Q,U,V,freq_test,trial_RM_i,trial_phi,
                                                    width_native,t_samp,False,datadir,calstr,label,n_f,n_t,
                                                    show,False,buff,weighted,n_t_weight,timeaxis,fobj,sf_window_weights,
                                                    n_off,full,input_weights,timestart_in,timestop_in,False,False,
                                                    1,1,i)
                        SNRs[i*trialsize:(i+1)*trialsize,0] = SNRs_i
                    """ 
                
                    

                #wait for tasks to complete
                #wait(task_list)
                for future in as_completed(task_list):
                    
                    if full:
                        tmp,tmp,SNRs_i,tmp,tmp,tmp,tmp,tmp,SNRs_full_i,tmp,i = future.result()
                            
                        if i == numbatch*maxProcesses:
                            SNRs_full[numbatch*maxProcesses:,:] = SNRs_full_i
                        else:
                            SNRs_full[i*trialsize:(i+1)*trialsize,:] = SNRs_full_i
                      
                    else:
                        tmp,tmp,SNRs_i,tmp,tmp,tmp,tmp,tmp,i = future.result()
                            
                        
                        
                    if i == numbatch*maxProcesses:
                        SNRs[numbatch*maxProcesses:,0] = SNRs_i
                    else:
                        SNRs[i*trialsize:(i+1)*trialsize,0] = SNRs_i

                    if sendtodir != '':
                        prevSNRs = np.load(sendtodir + "/SNRs.npy")
                        prevSNRs = np.concatenate([prevSNRs,SNRs_i])
                        np.save(sendtodir + "/SNRs.npy",prevSNRs)

                        prevSNRs = np.load(sendtodir + "/SNRs_full.npy")
                        prevSNRs = np.concatenate([prevSNRs,SNRs_full_i],axis=0)                            
                        np.save(sendtodir + "/SNRs_full.npy",prevSNRs)

                        prevRMs = np.load(sendtodir + "/trialRM.npy")
                        if i == maxProcesses*numbatch:
                            prevRMs = np.concatenate([prevRMs,trial_RM[maxProcesses*numbatch:]])
                        else:
                            prevRMs = np.concatenate([prevRMs,trial_RM[i*trialsize:(i+1)*trialsize]])
                        np.save(sendtodir + "/trialRM.npy",prevRMs)


                executor.shutdown(wait=True)
        else:
            for i in range(len(trial_RM)):
                for j in range(len(trial_phi)):
                    RM_i = trial_RM[i]
                    phi_j = trial_phi[j]

                    P_trial = P_cut*np.exp(-1j*((2*RM_i*(((wavall_cut)**2) - np.mean(wav**2))) + phi_j))
                    Q_trial_t = np.mean(np.real(P_trial),axis=0)
                    U_trial_t = np.mean(np.imag(P_trial),axis=0)
                    #print(timestop-timestart)
                    #L_trial = np.sqrt(np.real(P_trial)**2 + np.imag(P_trial)**2)
                    #L_trial_t = np.mean(L_trial,axis=0)

                    L_trial_t = np.sqrt(Q_trial_t**2 + U_trial_t**2)
                    if full:
                        SNRs_full[i,:] = L_trial_t

                    #sig = np.mean(L_trial_t[timestart:timestop])#np.abs(np.mean(np.mean(P_trial,axis=0)[timestart:timestop]))#
                    if weighted:
                        sig = np.sum(L_trial_t*I_t_weights)
                    else:
                        sig = np.mean(L_trial_t)

                    SNRs[i,j] = sig/noise#np.abs(np.sum(P_trial))

    (max_RM_idx,max_phi_idx) = np.unravel_index(np.argmax(SNRs),np.shape(SNRs))
    max_RM_idx = np.argmax(SNRs)
    #print(max_RM_idx)

    significance = ((np.max(SNRs)-snr0)/snr0)

    if err:
        """
        #lower = trial_RM[max_RM_idx - np.argmin(np.abs((SNRs[:max_RM_idx,0])-(np.max(SNRs[:,0]) - 1))[::-1])]
        #upper = trial_RM[max_RM_idx]-trial_RM[np.argmin(np.abs((SNRs[max_RM_idx:,0])-(np.max(SNRs[:,0]) - 1)))] + trial_RM[max_RM_idx]

        check_lower = (SNRs[:max_RM_idx,0])[::-1]
        lower_idx = max_RM_idx - 1 - np.argmin(np.abs(check_lower-(SNRs[max_RM_idx] - 1)))
        lower = trial_RM[lower_idx]
        check_upper = (SNRs[max_RM_idx:,0])
        upper_idx = max_RM_idx + np.argmin(np.abs(check_upper-(SNRs[max_RM_idx] - 1)))
        upper = trial_RM[upper_idx]

        check_lower = SNRs[:np.argmax(SNRs)][::-1]
        lower_idx = len(check_lower)-1
        for i in range(len(check_lower)):#ts in testsnrs:
            if check_lower[i] < np.max(SNRs) -1:
                lower_idx = np.argmax(SNRs) - i
                break
        lower = trial_RM[lower_idx]
        check_upper = SNRs[np.argmax(SNRs):]
        upper_idx = len(check_upper)-1
        for i in range(len(check_upper)):#ts in testsnrs:
            if check_upper[i] < np.max(SNRs) -1:
                upper_idx = np.argmax(SNRs) - i
                break
        upper = trial_RM[upper_idx]
        RMerr = (upper-lower)/2
        """
        RMerr,upper,lower = faraday_error_SNR(SNRs,trial_RM,trial_RM[max_RM_idx])
        if plot:

            f=plt.figure(figsize=(12,6))
            plt.grid()
            plt.plot(trial_RM,SNRs)
            plt.axvline(lower,color="red")
            plt.axvline(upper,color="red")
            plt.xlabel("Trial RM")
            plt.ylabel("SNR")
            plt.title(label)
            plt.savefig(datadir + label + "_faradaySNR1D_" + calstr + str(n_f) + "_binned" + ext)
            if show:
                plt.show()
    else:
        RMerr = None
        upper = None
        lower = None
    
    
    if sendtodir != '':
        np.save(sendtodir + "/result.npy",np.array([trial_RM[max_RM_idx],RMerr,upper,lower]))
    
    if full:
        #for full time-dependent analysis, get RM vs time
        #print(trial_RM,trial_RM.shape)
        #print(SNRs_full,SNRs_full.shape)
        peak_RMs = trial_RM[np.argmax(SNRs_full,axis=0)]
        return (trial_RM[max_RM_idx],trial_phi[max_phi_idx],SNRs[:,0],RMerr,upper,lower,significance,noise,SNRs_full,peak_RMs,mt_offset)
    else:
        return (trial_RM[max_RM_idx],trial_phi[max_phi_idx],SNRs[:,0],RMerr,upper,lower,significance,noise,mt_offset)

    

#Calculate Error for SNR method
def faraday_error_SNR(SNRs,trial_RM_zoom,RMdet):
    tmp = SNRs[np.argmax(SNRs):]   
    for i in range(len(tmp)):
        if tmp[i] < (np.max(SNRs)-1):
            break
    upperRM = trial_RM_zoom[np.argmax(SNRs)+i]

    tmp = SNRs[:np.argmax(SNRs)]
    tmp = tmp[::-1]
    for i in range(len(tmp)):
        if tmp[i] < (np.max(SNRs)-1):
            break
    lowerRM = trial_RM_zoom[np.argmax(SNRs)-i]

    return ((upperRM-lowerRM)/2),upperRM,lowerRM

#Calculate initial estimate of RM from dispersion function, then get SNR spectrum to get true estimate and error
def faradaycal_full(I,Q,U,V,freq_test,trial_RM,trial_phi,width_native,t_samp,n_trial_RM_zoom,zoom_window=75,plot=False,datadir=DEFAULT_DATADIR,calstr="",label="",n_f=1,n_t=1,n_off=3000,show=False,fit_window=100,buff=0,normalize=True,DM=-1,weighted=False,n_t_weight=1,timeaxis=None,fobj=None,RM_tools=False,trial_RM_tools=np.linspace(-1e6,1e6,int(1e6)),sigma_clean=2):
    if len(I.shape) == 2:
        (I_f,Q_f,U_f,V_f) = get_stokes_vs_freq(I,Q,U,V,width_native,t_samp,n_f,n_t,freq_test,plot=False,normalize=normalize,buff=buff,weighted=weighted,n_t_weight=n_t_weight,timeaxis=timeaxis,fobj=fobj)
    else:
        (I_f,Q_f,U_f,V_f) = (I,Q,U,V)

    #get RMSF
    RMSF = []
    wav = (3e8)/(freq_test[0]*1e6)
    for rm in trial_RM:
        RMSF.append(np.sum(np.exp(-1j*2*rm*((wav**2) - np.mean(wav**2)))))


    #run initial faraday peak finding
    print("Initial RM synthesis...")
    (RM, phi,SNRs,RMerr) = faradaycal(I_f,Q_f,U_f,V_f,freq_test,trial_RM,trial_phi,plot=plot,datadir=datadir,calstr=calstr,label=label,n_f=n_f,n_t=n_t,show=show,fit_window=fit_window,err=True)
    
    #get RM tools estimate and plot with RM synthesis estimate
    if RM_tools:
        print("RM tools synthesis...")
        out=run_rmsynth([freq_test[0]*1e6,I_f,Q_f,U_f,np.std(I[:,:n_off],axis=1),np.std(Q[:,:n_off],axis=1),np.std(U[:,:n_off],axis=1)],phiMax_radm2=np.max(trial_RM),dPhi_radm2=np.abs(trial_RM_tools[1]-trial_RM_tools[0]))

        print("Cleaning...")
        out=run_rmclean(out[0],out[1],sigma_clean)
        print("RM Tools estimate: " + str(out[0]["phiPeakPIchan_rm2"]) + "\pm" + str(out[0]["dPhiPeakPIchan_rm2"]) + " rad/m^2")

        if plot:
            f=plt.figure(figsize=(12,6))
            plt.grid()
            plt.plot(trial_RM,SNRs,label="RM synthesis")
            plt.plot(out[1]["phiArr_radm2"],np.abs(out[1]["cleanFDF"]),alpha=0.5,label="RM Tools")
            plt.xlabel("Trial RM")
            plt.ylabel("Dispersion Function F(" + r'$\phi$' + ")")
            plt.title(label)
            plt.legend()
            plt.savefig(datadir + label + "_RMtools_" + calstr + str(n_f) + "_binned" + ext)
            if show:
                plt.show()
            plt.close(f)

    #estimate p-value
    sigma_Q = np.std(Q[:,:n_off])
    sigma_U = np.std(U[:,:n_off])
    sigma = (sigma_Q + sigma_U)/2

    (peak,timestart,timestop) = find_peak(I,width_native,t_samp,n_t,buff=buff)
    peak_chi2 = (np.max(SNRs)**2)/((Q.shape[0]/(timestop-timestart))*(sigma**2))

    p_val = chi2.sf(peak_chi2,df=2)
    print(r'Initial Estimate: ' + str(RM) + r'$\pm$' + str(RMerr) + ' rad/m^2, p-value: ' + str(p_val))
    print("SHAPE:" + str(I.shape))
    if len(I.shape)==2:
        #narrow down search
        print("Fine S/N synthesis")
        trial_RM_zoom = np.linspace(RM-zoom_window,RM+zoom_window,n_trial_RM_zoom)
        (RM,phi,SNRs,RMerr,upper,lower,significance,noise) = faradaycal_SNR(I,Q,U,V,freq_test,trial_RM_zoom,trial_phi,width_native,t_samp,plot=plot,datadir=datadir,calstr=calstr,label=label,n_f=n_f,n_t=n_t,show=show,buff=buff)
        print(r'Refined Estimate: '  + str(RM) + r'$\pm$' + str(RMerr) + r' rad/m^2')
    
        if RM_tools:

            L0_t = np.sqrt(np.mean(Q,axis=0)**2 + np.mean(U,axis=0)**2)
            sig0 = np.mean(np.sqrt(np.mean(Q,axis=0)**2 + np.mean(U,axis=0)**2)[timestart:timestop])
            L_trial_cut1 = L0_t[timestart%(timestop-timestart):]
            L_trial_cut = L_trial_cut1[:(len(L_trial_cut1)-(len(L_trial_cut1)%(timestop-timestart)))]
            L_trial_binned = L_trial_cut.reshape(len(L_trial_cut)//(timestop-timestart),timestop-timestart).mean(1)
            sigbin = np.argmax(L_trial_binned)
            noise = (np.std(np.concatenate([L_trial_cut[:sigbin],L_trial_cut[sigbin+1:]])))

        #get RMSF
        RMSF_SNR = []
        for rm in trial_RM_zoom:
            RMSF_SNR.append(np.mean(np.exp(-1j*2*rm*((wav**2) - np.mean(wav**2)))))

    else:
        print("Need 2D spectra for fine estimate")

    if plot:
        plt.figure(figsize=(12,6))
        plt.plot(trial_RM,RMSF,label="Synthesis")
        if len(I.shape)==2:
            plt.plot(trial_RM_zoom,RMSF_SNR,label="S/N Method")
        plt.grid()
        plt.xlabel("RM (rad/m^2)")
        plt.ylabel("RMSF")
        plt.title(label)
        plt.savefig(datadir + label + "_RMSF_" + calstr + str(n_f) + "_binned" + ext)
        if show:
            plt.show()
        plt.close(f)


    #Estimate true error from Brentjens/deBruyn eqn 52 and 61
    sigma_q = np.mean(np.std(Q[:,:n_off],axis=1))
    sigma_u = np.mean(np.std(U[:,:n_off],axis=1))
    sigma = (sigma_q + sigma_u)/2

    wav2 = ((3e8)/(freq_test[0]*(1e6)))**2
    waverr = (np.max(wav2)-np.min(wav2))/(2*np.sqrt(3))#np.sqrt(np.sum(wavs**2)/(len(wav2)-1))
    chierr = 0.5*sigma/np.abs(np.mean((Q_f + 1j*U_f)*np.exp(-2*1j*RM*wav2)))
    RMerr2 = chierr/(waverr*np.sqrt(len(wav2)-1))
    print("Brentjens/deBruyn Error: " + str(RMerr2) + "rad/m^2")

    if plot and len(I.shape)==2:
        f=plt.figure(figsize=(12,6))
        plt.grid()
        plt.plot(trial_RM_zoom,SNRs,label="S/N Method")#,label="RM synthesis")
        plt.axvline(upper,color="red",label=r'$1\sigma$ Error')
        plt.axvline(lower,color="red")
        plt.axvline(RM + RMerr2,color="green",label="Brentjens/deBruyn Error")
        plt.axvline(RM - RMerr2,color="green")
        plt.legend()
        plt.xlabel("Trial RM")
        plt.ylabel("SNR")
        plt.title(label)
        plt.savefig(datadir + label + "_BrentjensError_" + calstr + str(n_f) + "_binned" + ext)
        if show:
            plt.show()
        plt.close(f)
        
        
        if RM_tools:
            f=plt.figure(figsize=(12,6))
            plt.grid()
            plt.plot(trial_RM_zoom,SNRs,label="S/N Method")#,label="RM synthesis")
            plt.plot(out[1]["phiArr_radm2"],np.abs(out[1]["cleanFDF"])/noise,alpha=0.5,label="RM Tools")
            plt.axvline(upper,color="red",label=r'$1\sigma$ Error')
            plt.axvline(lower,color="red")
            plt.axvline(RM + RMerr2,color="green",label="Brentjens/deBruyn Error")
            plt.axvline(RM - RMerr2,color="green")
            plt.legend()
            plt.xlabel("Trial RM")
            plt.ylabel("SNR")
            plt.xlim(np.min(trial_RM_zoom),np.max(trial_RM_zoom))
            plt.title(label)
            plt.savefig(datadir + label + "_BrentjensError_RMtools_" + calstr + str(n_f) + "_binned" + ext)
            if show:
                plt.show()
            plt.close(f)


    #Estimate B field
    if DM != -1:
        B = RM/((0.81)*DM) #uG
    else:
        B = None

    return (RM,phi,SNRs,RMerr2,p_val,B)


#Apply faraday calibration
def calibrate_RM(xx_I_obs,yy_Q_obs,xy_U_obs,yx_V_obs,RM,phi,freq_test,stokes=True):
    #Get wavelength axis
    c = (3e8) #m/s
    wav = c/(freq_test[0]*(1e6))#wav_test[0]

    #calculate Stokes parameters
    if stokes:
        I_obs = xx_I_obs
        Q_obs = yy_Q_obs
        U_obs = xy_U_obs
        V_obs = yx_V_obs
    else:
        I_obs = 0.5*(xx_I_obs + yy_Q_obs)
        Q_obs = 0.5*(xx_I_obs - yy_Q_obs)
        U_obs = 0.5*(xy_U_obs + yx_V_obs)#np.real(xy_obs)
        V_obs = -1j*0.5*(xy_U_obs - yx_V_obs)#-np.imag(xy_obs)
    
    #if 2D, duplicate RM, phi
    if len(I_obs.shape) == 2:
        print("2D Calibration")
        #RM_cal = np.transpose(np.tile(RM,(I_obs.shape[1],1)))
        #phi_cal = np.transpose(np.tile(phi,(I_obs.shape[1],1)))
        #print(RM_cal.shape,phi_cal.shape,I_obs.shape)
        wav_cal = np.transpose(np.tile(wav,(I_obs.shape[1],1)))
    else:
        print("1D Calibration")
        #RM_cal = RM
        #phi_cal = phi
        #print(RM_cal.shape,phi_cal.shape,I_obs.shape)
        wav_cal = wav
    P_obs = Q_obs + 1j*U_obs
    P_true = P_obs*np.exp(-1j*((2*RM*((wav_cal)**2)) + phi))
    
    I_true = I_obs
    V_true = V_obs
    Q_true = np.real(P_true)
    U_true = np.imag(P_true)

    return (I_true,Q_true,U_true,V_true)



#read calibration data from pickle file; note cal data saved in a pickle file
def get_calibs(gaincal_fn,phasecal_fn,freq_test,use_fit=True):
    print("Gain calibration from " + gaincal_fn)
    print("Phase calibration from " + phasecal_fn)

    gain_dat = pickle.load(open(gaincal_fn,'rb'))
    phase_dat = pickle.load(open(phasecal_fn,'rb'))

    if use_fit:
        gain_params = gain_dat["ratio fit params"]
        phase_params = phase_dat["phidiff fit params"]

        ratio = np.zeros(len(freq_test[0]))
        deg = len(gain_params)
        for i in range(len(gain_params)):
            ratio += gain_params[i]*(freq_test[0]**(len(gain_params)-i-1))

        phidiff = np.zeros(len(freq_test[0]))
        deg = len(phase_params)
        for i in range(len(phase_params)):
            phidiff += phase_params[i]*(freq_test[0]**(len(phase_params)-i-1))

    else:
        ratio = gain_dat["ratio"]
        phidiff = phase_dat["phidiff"]

    return ratio,phidiff,deg


#find beam for calibrator observation
#function to get beam of calibrator
def find_beam(file_suffix,shape=(16,7680,256),path=dirs['data'],plot=False,show=False):
    #file_suffix = ""
    print("Starting find beam process...",file=f)#'/media/ubuntu/ssd/sherman/code/testoutput.txt')
    d = np.zeros(shape)
    i = 0
    for corrs in ['corr03', 'corr04', 'corr05', 'corr06', 'corr07',
                  'corr08', 'corr10', 'corr11', 'corr12', 'corr14',
                  'corr15', 'corr16', 'corr18', 'corr19', 'corr21',
                  'corr22']:
        data = np.loadtxt(path + corrs + file_suffix + '.out').reshape((7680,256))
        #print(data.sum())
        d[i,:,:] = data
        #print(corrs)
        i += 1

    if plot:
        f=plt.figure(figsize=(12,6))
        plt.imshow(d.mean(axis=0),aspect="auto")
        plt.xlabel("beam")
        plt.ylabel("time sample")
        #plt.xlim(100,120)
        plt.savefig(path +"_corrplots_" + file_suffix + ext)
        if show:
            plt.show()
        plt.close(f)

        f=plt.figure()
        plt.plot(d.mean(axis=0).mean(axis=0))
        plt.xlabel("beam")
        plt.savefig(path +"_beamspectrum_" + file_suffix + ext)
        if show:
            plt.show()

        plt.close(f)
    
    return (np.argmax(d.mean(axis=0).mean(axis=0)),d)


def arr_to_list_check(x):
    if type(x) != list:
        print("Converting array to list")
        return np.array(np.real(x),dtype=float).tolist()
        
    elif type(x[0]) == complex:
        return [np.real(i) for i in x]
    else:
        return x
import csv
def read_polcal(date):
    filename = "/media/ubuntu/ssd/sherman/code/POLCAL_PARAMETERS_" + date + ".csv"

    with open(filename,'r') as csvfile:
        reader = csv.reader(csvfile,delimiter=",")
        for row in reader:
            print(row[0])
            if row[0] == "|gxx|/|gyy|":
                ratio = np.array(row[1:],dtype="float")
            elif row[0] == "|gxx|/|gyy| fit":
                ratio_fit = np.array(row[1:],dtype="float")
            if row[0] == "phixx-phiyy":
                phase = np.array(row[1:],dtype="float")
            if row[0] == "phixx-phiyy fit":
                phase_fit = np.array(row[1:],dtype="float")
            if row[0] == "|gyy|":
                gainY = np.array(row[1:],dtype="float")
            if row[0] == "|gyy| fit":
                gainY_fit = np.array(row[1:],dtype="float")
            if row[0] == "gxx":
                gxx = np.array(row[1:],dtype="complex")
            if row[0] == "gyy":
                gyy = np.array(row[1:],dtype="complex")
            if row[0] == "freq_axis":
                freq_axis = np.array(row[1:],dtype="float")
    return (ratio,ratio_fit,phase,phase_fit,gainY,gainY_fit,gxx,gyy)

def RM_error_fit(x,a=35.16852537, b=-0.08341036):
    return (a*np.exp(b*(x)))

#Plotting Functions

def FRB_quick_analysis(ids,nickname,ibeam,width_native,buff,RA,DEC,caldate,n_t,n_f,beamsize_as=14,Lat=37.23,Lon=-118.2851,centerbeam=125,weighted=True,n_t_weight=2,sf_window_weights=7,RMcal=True,trial_RM=np.linspace(-1e6,1e6,int(2e6)),trial_phi=[0],n_trial_RM_zoom=5000,zoom_window=1000,fit_window=100,plot=True,show=False,datadir=None):
    if datadir is None:
        datadir = dirs['data'] + ids + "_" + nickname + "/"
    outdata = dict()
    outdata["inputs"] = dict()
    outdata["inputs"]["n_t"] = n_t
    outdata["inputs"]["n_f"] = n_f
    outdata["inputs"]["buff"] = buff
    outdata["inputs"]["avger_w"] = n_t_weight
    outdata["inputs"]["sf_window_weights"] = sf_window_weights
    outdata["inputs"]["width_native"] = width_native
    outdata["inputs"]["ids"] = ids
    outdata["inputs"]["nickname"] = nickname
    outdata["inputs"]["datadir"] = datadir
    
    #read cal parameters from file
    (ratio,ratio_fit,phase,phase_fit,gainY,gainY_fit,gxx,gyy) = read_polcal(caldate)
    
    #Read data
    #datadir="/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/"+ids + "_" + nickname + "/"
    (I_fullres,Q_fullres,U_fullres,V_fullres,fobj,timeaxis,freq_test_fullres,wav_test,badchans) = get_stokes_2D(datadir,ids + "_dev",20480,n_t=n_t,n_f=1,n_off=int(12000//n_t),sub_offpulse_mean=True)
    
    #gain/phase and PA calibrate
    Ical_fullres,Qcal_fullres,Ucal_fullres,Vcal_fullres = calibrate(I_fullres,Q_fullres,U_fullres,V_fullres,(gxx,gyy),stokes=True)
    Ical_fullres,Qcal_fullres,Ucal_fullres,Vcal_fullres,ParA_fullres = calibrate_angle(Ical_fullres,Qcal_fullres,Ucal_fullres,Vcal_fullres,fobj,ibeam,RA,DEC)

    #downsample in frequency
    I = avg_freq(I_fullres,n_f)
    Q = avg_freq(Q_fullres,n_f)
    U = avg_freq(U_fullres,n_f)
    V = avg_freq(V_fullres,n_f)

    Ical = avg_freq(Ical_fullres,n_f)
    Qcal = avg_freq(Qcal_fullres,n_f)
    Ucal = avg_freq(Ucal_fullres,n_f)
    Vcal = avg_freq(Vcal_fullres,n_f)
    
    if freq_test_fullres[0].shape[0]%n_f != 0:
            for i in range(4):
                freq_test_fullres[i] = freq_test_fullres[i][freq_test_fullres[i].shape[0]%n_f:]

    freq_test =  [freq_test_fullres[0].reshape(len(freq_test_fullres[0])//n_f,n_f).mean(1)]*4
    
    #get window
    (peak,timestart,timestop) = find_peak(I,width_native,fobj.header.tsamp,n_t=n_t,peak_range=None,pre_calc_tf=False,buff=buff)
    t = 32.7*n_t*np.arange(0,I.shape[1])


    if plot:
        plot_spectra_2D(I,Q,U,V,width_native,fobj.header.tsamp,n_t,n_f,freq_test,lim=np.percentile(I,90),show=show,buff=int(64/n_t),weighted=False,window=int(64/n_t),datadir=datadir,ext=".pdf",label=ids + "_" + nickname + "_uncal_")
        plot_spectra_2D(Ical,Qcal,Ucal,Vcal,width_native,fobj.header.tsamp,n_t,n_f,freq_test,lim=np.percentile(I,90),show=show,buff=int(64/n_t),weighted=False,window=int(64/n_t),datadir=datadir,ext=".pdf",label=ids + "_" + nickname + "_cal_")


    if weighted:
        #get filter, FWHM
        I_w_t_filt = get_weights(I,Q,U,V,width_native,fobj.header.tsamp,n_f,n_t,freq_test,timeaxis,fobj,n_off=int(12000/n_t),buff=buff,n_t_weight=n_t_weight,sf_window_weights=sf_window_weights)
        FWHM,heights,intL,intR = peak_widths(I_w_t_filt,[np.argmax(I_w_t_filt)])
        print("FWHM: " + str((intR-intL)*n_t*32.7) + " us")
        wind = (intL-intR)*3
        if plot:
            plt.figure(figsize=(12,6))
            plt.plot(t,I_w_t_filt)
            plt.axvline(32.7*n_t*intL)
            plt.axvline(32.7*n_t*intR)
            plt.xlim(32.7*n_t*timestart - wind*32.7*n_t,32.7*n_t*timestop + wind*32.7*n_t)
            plt.savefig(datadir + ids + "_" + nickname + "_filterplot.png")
            if show:
                plt.show()

    #get IQUV vs time and freq
    (I_tcal,Q_tcal,U_tcal,V_tcal) = get_stokes_vs_time(Ical,Qcal,Ucal,Vcal,width_native,fobj.header.tsamp,n_t,n_off=int(12000//n_t),plot=plot,show=show,datadir=datadir,normalize=True,buff=buff,window=3,label=ids + "_" + nickname + "_cal_")
    (I_fcal,Q_fcal,U_fcal,V_fcal) = get_stokes_vs_freq(Ical,Qcal,Ucal,Vcal,width_native,fobj.header.tsamp,n_f,n_t,freq_test,n_off=int(12000/n_t),plot=plot,show=show,datadir=datadir,normalize=True,buff=buff,weighted=weighted,n_t_weight=n_t_weight,timeaxis=timeaxis,fobj=fobj,sf_window_weights=sf_window_weights,label=ids + "_" + nickname + "_cal_")

    #estimate polarization fractions before RM
    
    [(pol_f,pol_t,avg,sigma_frac,snr_frac),(L_f,L_t,avg_L,sigma_L,snr_L),(C_f_unbiased,C_t_unbiased,avg_C_abs,sigma_C_abs,snr_C),(C_f,C_t,avg_C,sigma_C,snr_C),snr] = get_pol_fraction(Ical,Qcal,Ucal,Vcal,width_native,fobj.header.tsamp,n_t,n_f,freq_test,n_off=int(12000/n_t),plot=plot,show=show,datadir=datadir,normalize=True,buff=buff,full=False,weighted=weighted,n_t_weight=n_t_weight,timeaxis=timeaxis,fobj=fobj,sf_window_weights=sf_window_weights,label=ids + "_" + nickname + "_cal_")
    PA_f,PA_t,PA_f_errs,PA_t_errs,avg_PA,PA_err = get_pol_angle(Ical,Qcal,Ucal,Vcal,width_native,fobj.header.tsamp,n_t,n_f,freq_test,n_off=int(12000/n_t),plot=plot,show=show,datadir=datadir,normalize=True,buff=buff,weighted=weighted,n_t_weight=n_t_weight,timeaxis=timeaxis,fobj=fobj,sf_window_weights=sf_window_weights,label=ids + "_" + nickname + "_cal_")
    print(f'SNR: ${snr}\sigma$'.format(snr=snr))
    print(r'Total Polarization: ${avg} \pm {err}$'.format(avg=avg,err=sigma_frac))
    print(r'Linear Polarization: ${avg} \pm {err}$'.format(avg=avg_L,err=sigma_L))
    print(r'Circular Polarization: ${avg} \pm {err}$'.format(avg=avg_C_abs,err=sigma_C_abs))
    print(r'Signed Circular Polarization: ${avg} \pm {err}$'.format(avg=avg_C,err=sigma_C))
    print(r'Position Angle: ${avg}^\circ \pm {err}^\circ$'.format(avg=avg_PA*180/np.pi,err=PA_err*180/np.pi))
    outdata["calibrated"] = dict()
    outdata["calibrated"]["polarization"] = [avg,sigma_frac,snr_frac]
    outdata["calibrated"]["linear polarization"] = [avg_L,sigma_L,snr_L]
    outdata["calibrated"]["circular polarization"] = [avg_C,sigma_C,snr_C]

    #RM synthesis
    if RMcal:
        (I_fcal_fullres,Q_fcal_fullres,U_fcal_fullres,V_fcal_fullres) = get_stokes_vs_freq(Ical_fullres,Qcal_fullres,Ucal_fullres,Vcal_fullres,width_native,fobj.header.tsamp,n_f,n_t,freq_test_fullres,n_off=int(12000/n_t),plot=plot,show=show,datadir=datadir,normalize=True,buff=buff,weighted=weighted,n_t_weight=n_t_weight,timeaxis=timeaxis,fobj=fobj,sf_window_weights=sf_window_weights,label=ids + "_" + nickname + "_cal_")
        RM1,phi1,RMsnrs1,RMerr1 = faradaycal(I_fcal_fullres,Q_fcal_fullres,U_fcal_fullres,V_fcal_fullres,freq_test_fullres,trial_RM,trial_phi,plot=plot,show=show,datadir=datadir,fit_window=fit_window,err=True,label=ids + "_" + nickname + "_cal_")
        print(r'RM Synthesis estimate: ${avg} \pm {err} rad/m^2$'.format(avg=RM1,err=RMerr1))
        
        trial_RM2 = np.linspace(RM1-zoom_window,RM1+zoom_window,n_trial_RM_zoom)

        RM2,phi2,RMsnrs2,RMerr2,upp,low,sig,noise = faradaycal_SNR(Ical_fullres,Qcal_fullres,Ucal_fullres,Vcal_fullres,freq_test_fullres,trial_RM2,trial_phi,width_native,fobj.header.tsamp,plot=plot,n_f=1,n_t=n_t,show=show,datadir=datadir,err=True,buff=buff,weighted=weighted,n_off=int(12000/n_t),n_t_weight=n_t_weight,timeaxis=timeaxis,fobj=fobj,sf_window_weights=sf_window_weights,full=False,label=ids + "_" + nickname + "_cal_")
        print(r'RM S/N method estimate: ${avg} \pm {err} rad/m^2$, Linear S/N = {SNR}'.format(avg=RM2,err=RMerr2,SNR=np.max(RMsnrs2)))

        RMerrnew =  RM_error_fit(np.max(RMsnrs2))
        print(r'Simulated RM error: {err}'.format(err=RMerrnew))

        #calibrate and re-estimate polarization
        (IcalRM,QcalRM,UcalRM,VcalRM)=calibrate_RM(Ical,Qcal,Ucal,Vcal,RM2,0,freq_test,stokes=True)
        (I_tcal,Q_tcal,U_tcal,V_tcal) = get_stokes_vs_time(IcalRM,QcalRM,UcalRM,VcalRM,width_native,fobj.header.tsamp,n_t,n_off=int(12000//n_t),plot=plot,show=show,datadir=datadir,normalize=True,buff=buff,window=3,label=ids + "_" + nickname + "_RMcal_")
        (I_fcal,Q_fcal,U_fcal,V_fcal) = get_stokes_vs_freq(IcalRM,QcalRM,UcalRM,VcalRM,width_native,fobj.header.tsamp,n_f,n_t,freq_test,n_off=int(12000/n_t),plot=plot,show=show,datadir=datadir,normalize=True,buff=buff,weighted=weighted,n_t_weight=n_t_weight,timeaxis=timeaxis,fobj=fobj,sf_window_weights=sf_window_weights,label=ids + "_" + nickname + "_RMcal_")
        [(pol_f,pol_t,avg,sigma_frac,snr_frac),(L_f,L_t,avg_L,sigma_L,snr_L),(C_f_unbiased,C_t_unbiased,avg_C_abs,sigma_C_abs,snr_C),(C_f,C_t,avg_C,sigma_C,snr_C),snr] = get_pol_fraction(IcalRM,QcalRM,UcalRM,VcalRM,width_native,fobj.header.tsamp,n_t,n_f,freq_test,n_off=int(12000/n_t),plot=plot,show=show,datadir=datadir,normalize=True,buff=buff,full=False,weighted=weighted,n_t_weight=n_t_weight,timeaxis=timeaxis,fobj=fobj,sf_window_weights=sf_window_weights,label=ids + "_" + nickname + "_cal_")
        PA_f,PA_t,PA_f_errs,PA_t_errs,avg_PA,PA_err = get_pol_angle(IcalRM,QcalRM,UcalRM,VcalRM,width_native,fobj.header.tsamp,n_t,n_f,freq_test,n_off=int(12000/n_t),plot=plot,show=show,datadir=datadir,normalize=True,buff=buff,weighted=weighted,n_t_weight=n_t_weight,timeaxis=timeaxis,fobj=fobj,sf_window_weights=sf_window_weights,label=ids + "_" + nickname + "_cal_")
        print(f'SNR: ${snr}\sigma$'.format(snr=snr))
        print(r'Total Polarization: ${avg} \pm {err}$'.format(avg=avg,err=sigma_frac))
        print(r'Linear Polarization: ${avg} \pm {err}$'.format(avg=avg_L,err=sigma_L))
        print(r'Circular Polarization: ${avg} \pm {err}$'.format(avg=avg_C_abs,err=sigma_C_abs))
        print(r'Signed Circular Polarization: ${avg} \pm {err}$'.format(avg=avg_C,err=sigma_C))
        print(r'Position Angle: ${avg}^\circ \pm {err}^\circ$'.format(avg=avg_PA*180/np.pi,err=PA_err*180/np.pi))
        
        outdata["RMcalibrated"] = dict()
        outdata["RMcalibrated"]["RM"] = [RM2,RMerrnew]
        outdata["RMcalibrated"]["polarization"] = [avg,sigma_frac,snr_frac]
        outdata["RMcalibrated"]["linear polarization"] = [avg_L,sigma_L,snr_L]
        outdata["RMcalibrated"]["circular polarization"] = [avg_C,sigma_C,snr_C]

        fname_json = datadir + ids + "_" + nickname +  "_initial_polanalysis_out_initial.json"
        print("Writing output data to " + fname_json)
        with open(fname_json, "w") as outfile:
            json.dump(outdata, outfile)

        return (RM2,RMerrnew),(avg,sigma_frac,snr_frac),(avg_L,sigma_L,snr_L),(avg_C,sigma_C,snr_C),(avg_PA,PA_err)
    else:
        fname_json = datadir + ids + "_" + nickname +  "_initial_polanalysis_out_initial.json"
        print("Writing output data to " + fname_json)
        with open(fname_json, "w") as outfile:
            json.dump(outdata, outfile)
        return (avg,sigma_frac,snr_frac),(avg_L,sigma_L,snr_L),(avg_C,sigma_C,snr_C),(avg_PA,PA_err)


from matplotlib.widgets import Slider, Button
from matplotlib.widgets import CheckButtons
from matplotlib.widgets import SpanSelector
from matplotlib.widgets import Button
from matplotlib.widgets import TextBox

from RMtools_1D.do_RMsynth_1D import run_rmsynth
from RMtools_1D.do_RMclean_1D import run_rmclean
from RMtools_1D.do_QUfit_1D_mnest import run_qufit
#interactive functions
def int_get_downsampling_params(I_init,Q_init,U_init,V_init,ids,nickname,init_width_native,t_samp,freq_test_init,timeaxis,fobj,n_off=3000,mask_flag=False,datadir=None):
    if datadir is None:
        datadir = dirs['data'] + ids + "_" + nickname + "/"
    #function for tuning the following paramters:
    #n_t
    #n_f
    #n_t_weight
    #sf_window_weights
    #buff
    legend_loc = (0.35,1.01)
    I = copy.deepcopy(I_init)
    Q = copy.deepcopy(Q_init)
    U = copy.deepcopy(U_init)
    V = copy.deepcopy(V_init)

    #TIME
    fig = plt.figure(figsize=(72,84))
    ax = plt.subplot2grid(shape=(6,2),loc=(0,0))
    final_complist_min = []
    final_complist_max = []
    complist_min = []
    complist_max = []
    donelist = []
    #function to highlight components
    def update_component(xmin,xmax):
        if len(donelist) == 0:
            complist_min.append(xmin)
            complist_max.append(xmax)
            print("Current comp bounds: " + str(xmin) + "-" + str(xmax))
    span = SpanSelector(ax,update_component,"horizontal",interactive=True,props=dict(facecolor='red', alpha=0.5))
    span.set_active(False)
    span.set_visible(True)
    print(ax.lines)
    ax1 = plt.subplot2grid(shape=(6,2),loc=(4,0))
    ax1.set_xlabel("Frequency (MHz)")
    ax1.set_xlim(np.min(freq_test_init[0]),np.max(freq_test_init[0]))
    
    ax2 = plt.subplot2grid(shape=(6,2),loc=(0,1))
    ax2.set_title("RM Analysis")
    ax2.set_xlabel("RM ($rad/m^2$)")
    ax2.set_ylabel(r'$F(\phi)$')

    ax3 = plt.subplot2grid(shape=(6,2),loc=(4,1))
    ax3_1 = ax3.twinx()
    ax3.set_xlabel(r'RM ($rad/m^2$)')
    ax3.set_ylabel("Linear S/N")
    ax3_1.set_ylabel(r'$F(\phi)$')

    (I_t_init,Q_t_init,U_t_init,V_t_init) = get_stokes_vs_time(I,Q,U,V,init_width_native,fobj.header.tsamp,1,n_off=n_off,plot=False,show=False,datadir=datadir,normalize=True)
    I_t = copy.deepcopy(I_t_init)
    Q_t = copy.deepcopy(Q_t_init)
    U_t = copy.deepcopy(U_t_init)
    V_t = copy.deepcopy(V_t_init)

    peak = int(15280)
    timestart = int(peak - (5e-3)/(32.7e-6))
    timestop = int(peak + (5e-3)/(32.7e-6))
    #t = 32.7*np.arange(timestop-timestart)/1000
    ax.plot(I_t[timestart:timestop],label="I")
    ax.plot(Q_t[timestart:timestop],label="Q")
    ax.plot(U_t[timestart:timestop],label="U")
    ax.plot(V_t[timestart:timestop],label="V")
    ax.legend(ncol=2,loc=legend_loc)
    ax.set_xlabel("Time Sample (Sampling Time {t} $\mu s$)".format(t=np.around(32.7,2)))
    ax.set_xlim(0,timestop-timestart)

    #create sliders and checkbutton => SCREEN 1, n_t, multiple components, done
    slidewidth = 0.35#0.65
    slideleft = 0.1#0.5-slidewidth#(0.5-slidewidth)#(1-slidewidth)/2
    slideheight = 0.03
    offset = 0.36- slideheight*3.5
    buttonwidth = 0.1
    offset2 = 0.08
    

    #TIME AXES
    axn_t = plt.axes([slideleft,offset2+offset + slideheight*7,slidewidth,slideheight])
    axmulti = plt.axes([slideleft,offset2+0.22 + 0.1 + 2*0.085 + 0.02,buttonwidth,0.1])
    axnextcomp = plt.axes([slideleft+buttonwidth+0.01,offset2+0.22 + 0.1 + 2*0.085 + 0.02,buttonwidth,0.1-(0.03/2)])
    axdone = plt.axes([slideleft+2*buttonwidth+0.02,offset2+ 0.22 + 0.1 + 2*0.085 + 0.02,buttonwidth,0.1-(0.03/2)])
    axn_t_weight = plt.axes([slideleft,offset2+offset + slideheight*2,slidewidth,slideheight])
    axsf_window_weights = plt.axes([slideleft,offset2+offset + slideheight*3,slidewidth,slideheight])
    axbuff1 = plt.axes([slideleft,offset2+offset + slideheight*4,slidewidth,slideheight])
    axbuff2 = plt.axes([slideleft,offset2+offset + slideheight*5,slidewidth,slideheight])
    axwidth = plt.axes([slideleft,offset2+offset + slideheight*6,slidewidth,slideheight])
    

    #FREQ AXES
    axn_f = plt.axes([slideleft,offset2+offset + slideheight*1,slidewidth,slideheight])
    #axdone2 = plt.axes([0.625, 0.25 + slideheight*1 + 0.13 - (0.03/2) - 0.05 - 0.01,0.15,0.1-(0.03/2)])

    #RM AXES
    ax_RMmin = plt.axes([slideleft+0.45,offset2+0.22 + 0.1 + 2*0.085 + 0.02,buttonwidth,0.1])
    ax_RMmax = plt.axes([slideleft+0.45+buttonwidth+buttonwidth/2,offset2+0.22 + 0.1 + 2*0.085 + 0.02,buttonwidth,0.1])
    ax_RMtrials = plt.axes([slideleft+0.45,offset2+0.22 + 0.1 + 2*0.085 + 0.02-0.1-0.01,buttonwidth,0.1])
    ax_RMresult = plt.axes([slideleft+0.45,offset2+0.22 + 0.1 + 2*0.085 + 0.02-0.2-0.02,buttonwidth,0.1])
    ax_RMerror = plt.axes([slideleft+0.45+buttonwidth+buttonwidth/2,offset2+0.22 + 0.1 + 2*0.085 + 0.02-0.2-0.02,buttonwidth,0.1])
    ax_RMapply = plt.axes([slideleft+0.45+buttonwidth+buttonwidth/2,offset2+0.22 + 0.1 + 2*0.085 + 0.02-0.1-0.01,buttonwidth,0.1])
    ax_RMrun = plt.axes([slideleft+0.45+2*buttonwidth+buttonwidth,offset2+0.22 + 0.1 + 2*0.085 + 0.02-0.2-0.02,buttonwidth,0.1])
    ax_RMzoomrange = plt.axes([slideleft+0.45+2*buttonwidth+buttonwidth,offset2+0.22 + 0.1 + 2*0.085 + 0.02,buttonwidth,0.1])
    ax_RMzoomtrials = plt.axes([slideleft+0.45+2*buttonwidth+buttonwidth,offset2+0.22 + 0.1 + 2*0.085 + 0.02-0.1-0.01,buttonwidth,0.1])
    ax_reset = plt.axes([slideleft,0.9,buttonwidth,0.1])

    #POL AXES
    axTPOL = plt.axes([slideleft,0.05,buttonwidth*3/4,0.1])
    axLPOL = plt.axes([slideleft+buttonwidth+0.01,0.05,buttonwidth*3/4,0.1])
    axCPOL = plt.axes([slideleft+2*buttonwidth+0.02,0.05,buttonwidth*3/4,0.1])
    axSCPOL = plt.axes([slideleft+3*buttonwidth+0.03,0.05,buttonwidth*3/4,0.1])
    axTSNR = plt.axes([slideleft+4*buttonwidth+0.04,0.05,buttonwidth*3/4,0.1])
    axLSNR = plt.axes([slideleft+5*buttonwidth+0.05,0.05,buttonwidth*3/4,0.1])
    axCSNR = plt.axes([slideleft+6*buttonwidth+0.06,0.05,buttonwidth*3/4,0.1])
    axSNR = plt.axes([slideleft+7*buttonwidth+0.07,0.05,buttonwidth*3/4,0.1])

    #make buttons
    n_t_slider = Slider(axn_t,r'$n_t$',1,64,1,valstep=1)
    multi_button = CheckButtons(axmulti,labels=["multi"])
    nextcomp_button = Button(axnextcomp,label="next")
    done_button = Button(axdone,label="done")
    #deactivate others
    n_t_weight_slider = Slider(axn_t_weight,r'$n_{tw}$',0,8,0,valstep=1)
    n_t_weight_slider.valtext.set_text(1)
    sf_window_weights_slider = Slider(axsf_window_weights,"sf window",3,21,3,valstep=2)
    buff1_slider = Slider(axbuff1,"left buffer",0,20,0,valstep=1)
    buff2_slider = Slider(axbuff2,"right buffer",0,20,0,valstep=1)
    width_slider = Slider(axwidth,"ibox",0,40,init_width_native,valstep=1)
    n_t_weight_slider.set_active(False)
    sf_window_weights_slider.set_active(False)
    buff1_slider.set_active(False)
    buff2_slider.set_active(False)
    width_slider.set_active(False)

    #done_button2 = Button(axdone2,label="done")
    n_f_slider = Slider(axn_f,r'$n_f$',0,8,0,valstep=1)
    n_f_slider.valtext.set_text(1)
    n_f_slider.set_active(False)
    #done_button2.set_active(False)
    
    RMmax_input = TextBox(ax_RMmax, 'RM Max', initial="1e6")
    RMmin_input = TextBox(ax_RMmin, 'RM Min', initial="-1e6")
    RMtrials_input = TextBox(ax_RMtrials, 'RM Trials', initial="2e6")
    RMapply_button = Button(ax_RMapply,label="apply")
    RMrun_button = Button(ax_RMrun,label="run")
    RMresult_input = TextBox(ax_RMresult, 'Result', initial="")
    RMerror_input = TextBox(ax_RMerror, 'Error', initial="")
    RMzoomrange_input = TextBox(ax_RMzoomrange, 'Zoom\nRange',initial="1000")
    RMzoomtrials_input = TextBox(ax_RMzoomtrials, 'Zoom\nTrials',initial="5000")
    RMmax_input.set_active(False)
    RMmin_input.set_active(False)
    RMtrials_input.set_active(False)
    RMapply_button.set_active(False)
    RMrun_button.set_active(False)
    RMresult_input.set_active(False)
    RMerror_input.set_active(False)
    RMzoomrange_input.set_active(False)
    RMzoomtrials_input.set_active(False)

    reset_button = Button(ax_reset,label="reset")


    TPOL_input = TextBox(axTPOL, 'T/I', initial="")
    LPOL_input = TextBox(axLPOL, 'L/I', initial="")
    CPOL_input = TextBox(axCPOL, '|V|/I', initial="")
    SCPOL_input = TextBox(axSCPOL, 'V/I', initial="")
    TSNR_input = TextBox(axTSNR, 'T/I\nS/N', initial="")
    LSNR_input = TextBox(axLSNR, 'L/I\nS/N', initial="")
    CSNR_input = TextBox(axCSNR, 'V/I\nS/N', initial="")
    SNR_input = TextBox(axSNR, 'S/N', initial="")

    #function for n_t slider
    def update_n_t(val):
        n_t = n_t_slider.val

        for i in range(2,len(ax.lines)):
            ax.lines.pop()
            ax.set_prop_cycle(None)
            # recompute the ax.dataLim
            ax.relim()
            # update ax.viewLim using the new dataLim
            ax.autoscale_view()

        I_t = I_t_init[len(I_t_init)%n_t:]
        I_t = I_t.reshape(len(I_t)//n_t,n_t).mean(1)
        Q_t = Q_t_init[len(Q_t_init)%n_t:]
        Q_t = Q_t.reshape(len(Q_t)//n_t,n_t).mean(1)
        U_t = U_t_init[len(U_t_init)%n_t:]
        U_t = U_t.reshape(len(U_t)//n_t,n_t).mean(1)
        V_t = V_t_init[len(V_t_init)%n_t:]
        V_t = V_t.reshape(len(V_t)//n_t,n_t).mean(1)

        peak = int(15280/n_t)
        timestart = int(peak - (5e-3)/(n_t*32.7e-6))
        timestop = int(peak + (5e-3)/(n_t*32.7e-6))
        #t = (32.7*n_t*(timestop-timestart)/1000)
        ax.plot(I_t[timestart:timestop],label="I")
        ax.plot(Q_t[timestart:timestop],label="Q")
        ax.plot(U_t[timestart:timestop],label="U")
        ax.plot(V_t[timestart:timestop],label="V")
        ax.legend(ncol=2,loc=legend_loc)
        ax.set_xlabel("Time Sample (Sampling Time {t} $\mu s$)".format(t=np.around(32.7*n_t,2)))
        ax.set_xlim(0,timestop-timestart)
        return

    #function to select multiple components, activates span
    def update_multi(val):
        #span = span_lst[0]
        n_t_slider.set_active(not multi_button.get_status())
        span.set_active(not span.get_active())
        return

    """
    final_complist_min = []
    final_complist_max = []
    complist_min = []
    complist_max = []
    donelist = []
    #function to highlight components
    def update_component(xmin,xmax):
        if len(donelist) == 0:
            complist_min.append(xmin)
            complist_max.append(xmax)
            print("Current comp bounds: " + str(xmin) + "-" + str(xmax))
    """

    comp_widths = []
    comp_weights = []
    comp_n_t_weights = []
    comp_sf_window_weights = []
    comp_buffs = []
    comp_weights_list = [0]
    comp_timestarts = []
    comp_timestops =[]
    comp_TPOLs = []
    comp_LPOLs = []
    comp_CPOLs = []
    comp_SCPOLs = []
    comp_TSNRs = []
    comp_LSNRs = []
    comp_CSNRs = []
    comp_SNRs = []
    comp_multipeaks = []


    TPOL_str = []
    LPOL_str = []
    CPOL_str = []
    SCPOL_str = []
    TSNR_str = []
    LSNR_str = []
    CSNR_str = []
    SNR_str = []

    #function to switch to next component
    def update_nextcomp(val):
        #span = span_lst[0]
        if span.get_active() and len(donelist) == 0:
            final_complist_min.append(complist_min[-1])
            final_complist_max.append(complist_max[-1])
            for i in range(len(complist_min)):
                complist_min.pop()
                complist_max.pop()
            ax.axvspan(final_complist_min[-1], final_complist_max[-1], alpha=0.5, color='red')
            ax.axvline(final_complist_min[-1],color="red")
            ax.axvline(final_complist_max[-1],color="red")

            
            print("Component " + str(len(final_complist_min)) + " Bounds:" + str(final_complist_max[-1]) + "-" + str(final_complist_max[-1]))
        elif len(donelist) == 1:
            if len(comp_weights_list) < len(final_complist_min):
                comp_weights_list.append(comp_weights_list[-1]+1)
                n_t = n_t_slider.val
                n_t_weight = 2**(n_t_weight_slider.val)
                n_t_weight_slider.valtext.set_text(n_t_weight)
                sf_window_weights = sf_window_weights_slider.val
                buff1 = buff1_slider.val
                buff2 = buff2_slider.val
                buff= [buff1,buff2]
                width_native = width_slider.val

                for i in range(2,len(ax.lines)):
                    ax.lines.pop()
                    ax.set_prop_cycle(None)
                    ax.relim()
                    # update ax.viewLim using the new dataLim
                    ax.autoscale_view()
                I_t = I_t_init[len(I_t_init)%n_t:]
                I_t = I_t.reshape(len(I_t)//n_t,n_t).mean(1)
                Q_t = Q_t_init[len(Q_t_init)%n_t:]
                Q_t = Q_t.reshape(len(Q_t)//n_t,n_t).mean(1)
                U_t = U_t_init[len(U_t_init)%n_t:]
                U_t = U_t.reshape(len(U_t)//n_t,n_t).mean(1)
                V_t = V_t_init[len(V_t_init)%n_t:]
                V_t = V_t.reshape(len(V_t)//n_t,n_t).mean(1)

                #peak = int(15280/n_t)
                #timestart = int(peak - (1e-3)/(n_t*32.7e-6))
                #timestop = int(peak + (1e-3)/(n_t*32.7e-6))
                #mask all but first burst
                for i in range(len(final_complist_min)):
                    if i != comp_weights_list[-1]:
                        print((i,int(final_complist_min[i]),int(final_complist_max[i])))
                        mask = np.zeros(len(I_t))
                        TSART = int(int(15280/n_t) - (5e-3)/(n_t*32.7e-6)) + int(final_complist_min[i])
                        TSTOP = int(int(15280/n_t) - (5e-3)/(n_t*32.7e-6)) + int(final_complist_max[i])
                        mask[TSART:TSTOP] = 1
                        I_t = ma.masked_array(I_t,mask)
                        Q_t = ma.masked_array(Q_t,mask)
                        U_t = ma.masked_array(U_t,mask)
                        V_t = ma.masked_array(V_t,mask)

                (peak,timestart1,timestop1) = find_peak((I_t,I_t),width_native,t_samp,n_t,buff=buff,pre_calc_tf=True)
                timestart2 = timestart1 - int( (1e-3)/(n_t*32.7e-6))
                timestop2 = timestop1 + int( (1e-3)/(n_t*32.7e-6))
                I_t_weights=get_weights_1D(I_t,Q_t,U_t,V_t,timestart1,timestop1,width_native,t_samp,1,n_t,freq_test_init,timeaxis,fobj,n_off=n_off,buff=buff,n_t_weight=n_t_weight,sf_window_weights=sf_window_weights,padded=True)
                I_t_weights_cut = I_t_weights[timestart2:timestop2]

                #get polarization of component
                [(p_f,p_t,avg,sigma_frac,snr_frac),(L_f,L_t,avg_L,sigma_L,snr_L),(C_f,C_t,avg_C,sigma_C,snr_C),(C_f_u,C_t_u,avg_C_u,sigma_C_u,snr_C_u),snr] = get_pol_fraction([I_t,np.ones(1)],[Q_t,np.zeros(1)],[U_t,np.zeros(1)],[V_t,np.zeros(1)],width_native,t_samp,n_t,1,np.ones(1),n_off=int(12000/n_t),plot=False,pre_calc_tf=True,show=False,normalize=True,buff=[buff1_slider.val,buff2_slider.val],weighted=True,n_t_weight=1,timeaxis=timeaxis,fobj=fobj,unbias=True,input_weights=I_t_weights,allowed_err=1,unbias_factor=1)
                
                """
                TPOL_input.set_val(str(np.around(avg,2)) + "(" + str(int(np.around(sigma_frac*100,0))) + ")")
                LPOL_input.set_val(str(np.around(avg_L,2)) + "(" + str(int(np.around(sigma_L*100,0))) + ")")
                CPOL_input.set_val(str(np.around(avg_C,2)) + "(" + str(int(np.around(sigma_C*100,0))) + ")")
                SCPOL_input.set_val(str(np.around(avg_C_u,2)) + "(" + str(int(np.around(sigma_C_u*100,0))) + ")")
                
                SNR_input.set_val(np.around(snr,2))
                TSNR_input.set_val(np.around(snr_frac,2))
                LSNR_input.set_val(np.around(snr_L,2))
                CSNR_input.set_val(np.around(snr_C,2))
              

                TPOL_str.append(str(np.around(avg,2)) + "(" + str(int(np.around(sigma_frac*100,0))) + ")")
                LPOL_str.append(str(np.around(avg_L,2)) + "(" + str(int(np.around(sigma_L*100,0))) + ")")
                CPOL_str.append(str(np.around(avg_C,2)) + "(" + str(int(np.around(sigma_C*100,0))) + ")")
                SCPOL_str.append(str(np.around(avg_C_u,2)) + "(" + str(int(np.around(sigma_C_u*100,0))) + ")")

                SNR_str.append(str(np.around(snr,2)))
                TSNR_str.append(str(np.around(snr_frac,2)))
                LSNR_str.append(str(np.around(snr_L,2)))
                CSNR_str.append(str(np.around(snr_C,2)))


                TPOL_input.set_val(',\n'.join(TPOL_str))
                LPOL_input.set_val(',\n'.join(LPOL_str))
                CPOL_input.set_val(',\n'.join(CPOL_str))
                SCPOL_input.set_val(',\n'.join(SCPOL_str))


                TSNR_input.set_val(',\n'.join(TSNR_str))
                LSNR_input.set_val(',\n'.join(LSNR_str))
                CSNR_input.set_val(',\n'.join(CSNR_str))
                SNR_input.set_val(',\n'.join(SNR_str))
                """

                if len(comp_weights) < len(comp_weights_list):
                    print((len(comp_weights),len(comp_weights_list)))
                    comp_widths.append(width_native)
                    comp_weights.append(copy.deepcopy(I_t_weights))
                    comp_n_t_weights.append(n_t_weight)
                    comp_sf_window_weights.append(sf_window_weights)
                    comp_buffs.append(buff)
                    comp_timestarts.append(timestart1)
                    comp_timestops.append(timestop1)
                    comp_TPOLs.append(avg)
                    comp_LPOLs.append(avg_L)
                    comp_CPOLs.append(avg_C)
                    comp_SCPOLs.append(avg_C_u)
                    comp_TSNRs.append(snr_frac)
                    comp_LSNRs.append(snr_L)
                    comp_CSNRs.append(snr_C)
                    comp_SNRs.append(snr)
                    TPOL_str.append(str(np.around(avg,2)) + "(" + str(int(np.around(sigma_frac*100,0))) + ")")
                    LPOL_str.append(str(np.around(avg_L,2)) + "(" + str(int(np.around(sigma_L*100,0))) + ")")
                    CPOL_str.append(str(np.around(avg_C,2)) + "(" + str(int(np.around(sigma_C*100,0))) + ")")
                    SCPOL_str.append(str(np.around(avg_C_u,2)) + "(" + str(int(np.around(sigma_C_u*100,0))) + ")")

                    SNR_str.append(str(np.around(snr,2)))
                    TSNR_str.append(str(np.around(snr_frac,2)))
                    LSNR_str.append(str(np.around(snr_L,2)))
                    CSNR_str.append(str(np.around(snr_C,2)))
                else:
                    comp_widths[-1] = width_native
                    comp_weights[-1] = copy.deepcopy(I_t_weights)
                    comp_n_t_weights[-1] = n_t_weight
                    comp_sf_window_weights[-1] = sf_window_weights
                    comp_buffs[-1] = buff
                    comp_timestarts[-1] = timestart1
                    comp_timestops[-1] = timestop1
                    comp_TPOLs[-1] = avg
                    comp_LPOLs[-1] = avg_L
                    comp_CPOLs[-1] = avg_C
                    comp_SCPOLs[-1] = avg_C_u
                    comp_TSNRs[-1] = snr_frac
                    comp_LSNRs[-1] = snr_L
                    comp_CSNRs[-1] = snr_C
                    comp_SNRs[-1] = snr
                    TPOL_str[-1] = str(np.around(avg,2)) + "(" + str(int(np.around(sigma_frac*100,0))) + ")"
                    LPOL_str[-1] = str(np.around(avg_L,2)) + "(" + str(int(np.around(sigma_L*100,0))) + ")"
                    CPOL_str[-1] = str(np.around(avg_C,2)) + "(" + str(int(np.around(sigma_C*100,0))) + ")"
                    SCPOL_str[-1] = str(np.around(avg_C_u,2)) + "(" + str(int(np.around(sigma_C_u*100,0))) + ")"

                    SNR_str[-1] = str(np.around(snr,2))
                    TSNR_str[-1] = str(np.around(snr_frac,2))
                    LSNR_str[-1] = str(np.around(snr_L,2))
                    CSNR_str[-1] = str(np.around(snr_C,2))

                TPOL_input.set_val(',\n'.join(TPOL_str))
                LPOL_input.set_val(',\n'.join(LPOL_str))
                CPOL_input.set_val(',\n'.join(CPOL_str))
                SCPOL_input.set_val(',\n'.join(SCPOL_str))


                TSNR_input.set_val(',\n'.join(TSNR_str))
                LSNR_input.set_val(',\n'.join(LSNR_str))
                CSNR_input.set_val(',\n'.join(CSNR_str))
                SNR_input.set_val(',\n'.join(SNR_str))

                ax.plot(I_t_weights_cut*np.max(I_t)/np.max(I_t_weights_cut),linewidth=4,label="weights",color="purple")
                ax.plot(I_t[timestart2:timestop2],label="I")
                ax.plot(Q_t[timestart2:timestop2],label="Q")
                ax.plot(U_t[timestart2:timestop2],label="U")
                ax.plot(V_t[timestart2:timestop2],label="V")
                ax.legend(ncol=2,loc=legend_loc)
                ax.set_xlabel("Time Sample (Sampling Time {t} $\mu s$)".format(t=np.around(32.7*n_t,2)))
                ax.set_xlim(0,timestop2-timestart2)
            else:
                print("No more components")

        return

    #function to save results
    freq_avg_arrays = []
    def update_done(val):
        #span = span_lst[0]
        if len(donelist) == 0:
            donelist.append(1)
            n_t = n_t_slider.val
            width_slider.set_active(True)
            width_native = width_slider.val
            n_t_weight_slider.set_active(True)
            sf_window_weights_slider.set_active(True)
            buff1_slider.set_active(True)
            buff2_slider.set_active(True)
            span.set_visible(False)
            span.set_active(False)
        
            for i in range(2,len(ax.lines)):
                print(ax.lines)
                print(i)
                ax.lines.pop()
                ax.set_prop_cycle(None)
                ax.relim()
                # update ax.viewLim using the new dataLim
                ax.autoscale_view()
            I_t = I_t_init[len(I_t_init)%n_t:]
            I_t = I_t.reshape(len(I_t)//n_t,n_t).mean(1)
            Q_t = Q_t_init[len(Q_t_init)%n_t:]
            Q_t = Q_t.reshape(len(Q_t)//n_t,n_t).mean(1)
            U_t = U_t_init[len(U_t_init)%n_t:]
            U_t = U_t.reshape(len(U_t)//n_t,n_t).mean(1)
            V_t = V_t_init[len(V_t_init)%n_t:]
            V_t = V_t.reshape(len(V_t)//n_t,n_t).mean(1)

            #peak = int(15280/n_t)
            #timestart = int(peak - (1e-3)/(n_t*32.7e-6))
            #timestop = int(peak + (1e-3)/(n_t*32.7e-6))
            #mask all but first burst
            for i in range(1,len(final_complist_min)):
                print((i,int(final_complist_min[i]),int(final_complist_max[i])))
                mask = np.zeros(len(I_t))
                TSART = int(int(15280/n_t) - (5e-3)/(n_t*32.7e-6)) + int(final_complist_min[i])
                TSTOP = int(int(15280/n_t) - (5e-3)/(n_t*32.7e-6)) + int(final_complist_max[i])
                mask[TSART:TSTOP] = 1
                I_t = ma.masked_array(I_t,mask)
                Q_t = ma.masked_array(Q_t,mask)
                U_t = ma.masked_array(U_t,mask)
                V_t = ma.masked_array(V_t,mask)
            (peak,timestart1,timestop1) = find_peak((I_t,I_t),width_native,t_samp,n_t,buff=0,pre_calc_tf=True)
            timestart2 = timestart1 - int( (1e-3)/(n_t*32.7e-6))
            timestop2 = timestop1 + int( (1e-3)/(n_t*32.7e-6))
            I_t_weights=get_weights_1D(I_t,Q_t,U_t,V_t,timestart1,timestop1,width_native,t_samp,1,n_t,freq_test_init,timeaxis,fobj,n_off=n_off,buff=0,n_t_weight=1,sf_window_weights=3,padded=True,norm=False)
            I_t_weights_cut = I_t_weights[timestart2:timestop2]

            #get polarization of component
            [(p_f,p_t,avg,sigma_frac,snr_frac),(L_f,L_t,avg_L,sigma_L,snr_L),(C_f,C_t,avg_C,sigma_C,snr_C),(C_f_u,C_t_u,avg_C_u,sigma_C_u,snr_C_u),snr] = get_pol_fraction([I_t,np.ones(1)],[Q_t,np.zeros(1)],[U_t,np.zeros(1)],[V_t,np.zeros(1)],width_native,t_samp,n_t,1,np.ones(1),n_off=int(12000/n_t),plot=False,pre_calc_tf=True,show=False,normalize=True,buff=[buff1_slider.val,buff2_slider.val],weighted=True,n_t_weight=1,timeaxis=timeaxis,fobj=fobj,unbias=True,input_weights=I_t_weights,allowed_err=1,unbias_factor=1)

            """
            TPOL_input.set_val(str(np.around(avg,2)) + "(" + str(int(np.around(sigma_frac*100,0))) + ")")
            LPOL_input.set_val(str(np.around(avg_L,2)) + "(" + str(int(np.around(sigma_L*100,0))) + ")")
            CPOL_input.set_val(str(np.around(avg_C,2)) + "(" + str(int(np.around(sigma_C*100,0))) + ")")
            SCPOL_input.set_val(str(np.around(avg_C_u,2)) + "(" + str(int(np.around(sigma_C_u*100,0))) + ")")
            
            SNR_input.set_val(np.around(snr,2))
            TSNR_input.set_val(np.around(snr_frac,2))
            LSNR_input.set_val(np.around(snr_L,2))
            CSNR_input.set_val(np.around(snr_C,2))

            TPOL_str.append(str(np.around(avg,2)) + "(" + str(int(np.around(sigma_frac*100,0))) + ")")
            LPOL_str.append(str(np.around(avg_L,2)) + "(" + str(int(np.around(sigma_L*100,0))) + ")")
            CPOL_str.append(str(np.around(avg_C,2)) + "(" + str(int(np.around(sigma_C*100,0))) + ")")
            SCPOL_str.append(str(np.around(avg_C_u,2)) + "(" + str(int(np.around(sigma_C_u*100,0))) + ")")

            SNR_str.append(str(np.around(snr,2)))
            TSNR_str.append(str(np.around(snr_frac,2)))
            LSNR_str.append(str(np.around(snr_L,2)))
            CSNR_str.append(str(np.around(snr_C,2)))
            """

            if len(comp_weights) < len(comp_weights_list):
                print((len(comp_weights),len(comp_weights_list)))
                comp_weights.append(copy.deepcopy(I_t_weights))
                comp_widths.append(width_native)
                comp_n_t_weights.append(1)
                comp_sf_window_weights.append(3)
                comp_buffs.append(0)
                comp_timestarts.append(timestart1)
                comp_timestops.append(timestop1)
                comp_TPOLs.append(avg)
                comp_LPOLs.append(avg_L)
                comp_CPOLs.append(avg_C)
                comp_SCPOLs.append(avg_C_u)
                comp_TSNRs.append(snr_frac)
                comp_LSNRs.append(snr_L)
                comp_CSNRs.append(snr_C)
                comp_SNRs.append(snr)
                TPOL_str.append(str(np.around(avg,2)) + "(" + str(int(np.around(sigma_frac*100,0))) + ")")
                LPOL_str.append(str(np.around(avg_L,2)) + "(" + str(int(np.around(sigma_L*100,0))) + ")")
                CPOL_str.append(str(np.around(avg_C,2)) + "(" + str(int(np.around(sigma_C*100,0))) + ")")
                SCPOL_str.append(str(np.around(avg_C_u,2)) + "(" + str(int(np.around(sigma_C_u*100,0))) + ")")

                SNR_str.append(str(np.around(snr,2)))
                TSNR_str.append(str(np.around(snr_frac,2)))
                LSNR_str.append(str(np.around(snr_L,2)))
                CSNR_str.append(str(np.around(snr_C,2)))
            else:
                comp_widths.append(width_native)
                comp_weights[-1] = copy.deepcopy(I_t_weights)
                comp_n_t_weights[-1] = 1#n_t_weight
                comp_sf_window_weights[-1] = 5#sf_window_weights
                comp_buffs[-1] = 0#buff
                comp_timestarts[-1] = timestart1
                comp_timestops[-1] = timestop1
                comp_TPOLs[-1] = avg
                comp_LPOLs[-1] = avg_L
                comp_CPOLs[-1] = avg_C
                comp_SCPOLs[-1] = avg_C_u
                comp_TSNRs[-1] = snr_frac
                comp_LSNRs[-1] = snr_L
                comp_CSNRs[-1] = snr_C
                comp_SNRs[-1] = snr
                TPOL_str[-1] = str(np.around(avg,2)) + "(" + str(int(np.around(sigma_frac*100,0))) + ")"
                LPOL_str[-1] = str(np.around(avg_L,2)) + "(" + str(int(np.around(sigma_L*100,0))) + ")"
                CPOL_str[-1] = str(np.around(avg_C,2)) + "(" + str(int(np.around(sigma_C*100,0))) + ")"
                SCPOL_str[-1] = str(np.around(avg_C_u,2)) + "(" + str(int(np.around(sigma_C_u*100,0))) + ")"

                SNR_str[-1] = str(np.around(snr,2))
                TSNR_str[-1] = str(np.around(snr_frac,2))
                LSNR_str[-1] = str(np.around(snr_L,2))
                CSNR_str[-1] = str(np.around(snr_C,2))

            TPOL_input.set_val(',\n'.join(TPOL_str))
            LPOL_input.set_val(',\n'.join(LPOL_str))
            CPOL_input.set_val(',\n'.join(CPOL_str))
            SCPOL_input.set_val(',\n'.join(SCPOL_str))


            TSNR_input.set_val(',\n'.join(TSNR_str))
            LSNR_input.set_val(',\n'.join(LSNR_str))
            CSNR_input.set_val(',\n'.join(CSNR_str))
            SNR_input.set_val(',\n'.join(SNR_str))
            
            ax.plot(I_t_weights_cut*np.max(I_t)/np.max(I_t_weights_cut),linewidth=4,label="weights",color="purple")
            ax.plot(I_t[timestart2:timestop2],label="I")
            ax.plot(Q_t[timestart2:timestop2],label="Q")
            ax.plot(U_t[timestart2:timestop2],label="U")
            ax.plot(V_t[timestart2:timestop2],label="V")
            ax.legend(ncol=2,loc=legend_loc)
            ax.set_xlabel("Time Sample (Sampling Time {t} $\mu s$)".format(t=np.around(32.7*n_t,2)))
            ax.set_xlim(0,timestop2-timestart2)

        elif len(donelist) == 1 and (not (multi_button.get_status())[0] or  len(comp_weights_list) == len(final_complist_min)):
            print(comp_weights)
            comp_weights_sum = np.sum(np.array(comp_weights),axis=0)
            """
            copy.deepcopy(comp_weights[0])
            print(comp_weights)
            for i in range(1,len(comp_weights_list)):
                comp_weights_sum += comp_weights[i]
            """
            comp_weights_sum = comp_weights_sum/np.sum(comp_weights_sum)
            comp_weights.append(comp_weights_sum)# = comp_weights["final"]/np.sum(comp_weights["final"])
            
            print(comp_weights_sum)
            donelist.append(2)
        
            n_t = n_t_slider.val
            """
            n_t_weight_slider.set_active(True)
            sf_window_weights_slider.set_active(True)
            buff1_slider.set_active(True)
            buff2_slider.set_active(True)
            span.set_visible(False)
            span.set_active(False)
            """
            for i in range(2,len(ax.lines)):
                ax.lines.pop()
                ax.set_prop_cycle(None)
                ax.relim()
                # update ax.viewLim using the new dataLim
                ax.autoscale_view()
            I_t = I_t_init[len(I_t_init)%n_t:]
            I_t = I_t.reshape(len(I_t)//n_t,n_t).mean(1)
            Q_t = Q_t_init[len(Q_t_init)%n_t:]
            Q_t = Q_t.reshape(len(Q_t)//n_t,n_t).mean(1)
            U_t = U_t_init[len(U_t_init)%n_t:]
            U_t = U_t.reshape(len(U_t)//n_t,n_t).mean(1)
            V_t = V_t_init[len(V_t_init)%n_t:]
            V_t = V_t.reshape(len(V_t)//n_t,n_t).mean(1)


            #get total polarization
            n_t = n_t_slider.val
            [(p_f,p_t,avg,sigma_frac,snr_frac),(L_f,L_t,avg_L,sigma_L,snr_L),(C_f,C_t,avg_C,sigma_C,snr_C),(C_f_u,C_t_u,avg_C_u,sigma_C_u,snr_C_u),snr] = get_pol_fraction([I_t,np.ones(1)],[Q_t,np.zeros(1)],[U_t,np.zeros(1)],[V_t,np.zeros(1)],init_width_native,t_samp,n_t,1,np.ones(1),n_off=int(12000/n_t),plot=False,pre_calc_tf=True,show=False,normalize=True,buff=[buff1_slider.val,buff2_slider.val],weighted=True,n_t_weight=1,timeaxis=timeaxis,fobj=fobj,unbias=True,input_weights=comp_weights_sum,allowed_err=1,unbias_factor=1)

            TPOL_str.append(str(np.around(avg,2)) + "(" + str(int(np.around(sigma_frac*100,0))) + ")")
            LPOL_str.append(str(np.around(avg_L,2)) + "(" + str(int(np.around(sigma_L*100,0))) + ")")
            CPOL_str.append(str(np.around(avg_C,2)) + "(" + str(int(np.around(sigma_C*100,0))) + ")")
            SCPOL_str.append(str(np.around(avg_C_u,2)) + "(" + str(int(np.around(sigma_C_u*100,0))) + ")")

            SNR_str.append(str(np.around(snr,2)))
            TSNR_str.append(str(np.around(snr_frac,2)))
            LSNR_str.append(str(np.around(snr_L,2)))
            CSNR_str.append(str(np.around(snr_C,2)))

            TPOL_input.set_val(',\n'.join(TPOL_str))
            LPOL_input.set_val(',\n'.join(LPOL_str))
            CPOL_input.set_val(',\n'.join(CPOL_str))
            SCPOL_input.set_val(',\n'.join(SCPOL_str))


            TSNR_input.set_val(',\n'.join(TSNR_str))
            LSNR_input.set_val(',\n'.join(LSNR_str))
            CSNR_input.set_val(',\n'.join(CSNR_str))
            SNR_input.set_val(',\n'.join(SNR_str))

            peak = int(15280/n_t)
            timestart = int(peak - (5e-3)/(n_t*32.7e-6))
            timestop = int(peak + (5e-3)/(n_t*32.7e-6))
            ax.plot((comp_weights_sum*np.max(I_t)/np.max(comp_weights_sum))[timestart:timestop],linewidth=4,label="weights",color="purple")
            ax.plot(I_t[timestart:timestop],label="I")
            ax.plot(Q_t[timestart:timestop],label="Q")
            ax.plot(U_t[timestart:timestop],label="U")
            ax.plot(V_t[timestart:timestop],label="V")
            #ax.plot((comp_weights[0]*np.max(I_t)/np.max(comp_weights[0]))[timestart:timestop],linewidth=4,label="weights")
            #ax.plot((comp_weights[1]*np.max(I_t)/np.max(comp_weights[1]))[timestart:timestop],linewidth=4,label="weights")
            ax.legend(ncol=2,loc=legend_loc)
            ax.set_xlabel("Time Sample (Sampling Time {t} $\mu s$)".format(t=np.around(32.7*n_t,2)))
            ax.set_xlim(0,timestop-timestart)
            print("Resulting Params:")
            print("n_t_weights: " + str(comp_n_t_weights))
            print("sf_window_weights: " + str(comp_sf_window_weights))
            print("buffs: "  +str(comp_buffs))


            #plot frequency stuff
            print("Computing Spectra...")
            n_f_slider.set_active(True)
            #done_button2.set_active(True)
            I = avg_time(I_init,n_t)
            Q = avg_time(Q_init,n_t)
            U = avg_time(U_init,n_t)
            V = avg_time(V_init,n_t)

            (I_f,Q_f,U_f,V_f) = get_stokes_vs_freq(I,Q,U,V,init_width_native,fobj.header.tsamp,1,n_t,freq_test_init,n_off=n_off,plot=False,show=False,normalize=True,weighted=True,timeaxis=timeaxis,fobj=fobj,input_weights=comp_weights[-1])

            freq_avg_arrays.append(copy.deepcopy(I_f))
            freq_avg_arrays.append(copy.deepcopy(Q_f))
            freq_avg_arrays.append(copy.deepcopy(U_f))
            freq_avg_arrays.append(copy.deepcopy(V_f))

            fmin = np.min(freq_test_init[0])
            fmax = np.max(freq_test_init[0])
            ax1.plot(freq_test_init[0],I_f,label="I")
            ax1.plot(freq_test_init[0],Q_f,label="Q")
            ax1.plot(freq_test_init[0],U_f,label="U")
            ax1.plot(freq_test_init[0],V_f,label="V")
            ax1.set_xlabel("Frequency (MHz)")
            ax1.set_xlim(fmin,fmax)
            #ax1.legend(ncol=2,loc="upper right")
            print("Done!")


        elif len(donelist) == 1 and len(comp_weights_list) < len(final_complist_min):
            print("Still have " + str(len(final_complist_min)-len(comp_weights_list) + 1) + " components left")

        elif len(donelist) == 2:
            donelist.append(2)
            n_f_slider.set_active(False)
            n_f = 2**(n_f_slider.val)
            freq_test = (freq_test_init[0])[len(freq_test_init[0])%n_f:]
            freq_test = freq_test.reshape(len(freq_test)//n_f,n_f).mean(1)
            freq_test = [freq_test]*4
            fmin = np.min(freq_test)
            fmax = np.max(freq_test)

            I_f_init = freq_avg_arrays[0]
            Q_f_init = freq_avg_arrays[1]
            U_f_init = freq_avg_arrays[2]
            V_f_init = freq_avg_arrays[3]
        
            I_f = I_f_init[len(I_f_init)%n_f:]
            I_f = I_f.reshape(len(I_f)//n_f,n_f).mean(1)
            Q_f = Q_f_init[len(Q_f_init)%n_f:]
            Q_f = Q_f.reshape(len(Q_f)//n_f,n_f).mean(1)
            U_f = U_f_init[len(U_f_init)%n_f:]
            U_f = U_f.reshape(len(U_f)//n_f,n_f).mean(1)
            V_f = V_f_init[len(V_f_init)%n_f:]
            V_f = V_f.reshape(len(V_f)//n_f,n_f).mean(1)
            
            ax1.clear()
            ax1.plot(freq_test[0],I_f,label="I")
            ax1.plot(freq_test[0],Q_f,label="Q")
            ax1.plot(freq_test[0],U_f,label="U")
            ax1.plot(freq_test[0],V_f,label="V")
            ax1.set_xlabel("Frequency (MHz)")
            #ax1.legend(ncol=2,loc="upper right")
            ax1.set_xlim(fmin,fmax)
            print("Resulting Params:")
            print("n_f: " + str(2**(n_f_slider.val)))
        
            #activate RM synthesis
            RMmin_input.set_active(True)
            RMmax_input.set_active(True)
            RMtrials_input.set_active(True)
            RMrun_button.set_active(True)
            RMresult_input.set_active(True)
            RMerror_input.set_active(True)
            RMzoomrange_input.set_active(True)
            RMzoomtrials_input.set_active(True)
            return



    n_t_slider.on_changed(update_n_t)
    multi_button.on_clicked(update_multi)
    nextcomp_button.on_clicked(update_nextcomp)
    done_button.on_clicked(update_done)
    
    #once activated, select filter weights
    def update_time(val):
        n_t = n_t_slider.val
        n_t_weight = 2**(n_t_weight_slider.val)
        n_t_weight_slider.valtext.set_text(n_t_weight)
        sf_window_weights = sf_window_weights_slider.val
        buff1 = buff1_slider.val
        buff2 = buff2_slider.val
        buff = [buff1,buff2]
        width_native = width_slider.val

        for i in range(2,len(ax.lines)):
            ax.lines.pop()
            ax.set_prop_cycle(None)
            ax.relim()
            # update ax.viewLim using the new dataLim
            ax.autoscale_view()
        I_t = I_t_init[len(I_t_init)%n_t:]
        I_t = I_t.reshape(len(I_t)//n_t,n_t).mean(1)
        Q_t = Q_t_init[len(Q_t_init)%n_t:]
        Q_t = Q_t.reshape(len(Q_t)//n_t,n_t).mean(1)
        U_t = U_t_init[len(U_t_init)%n_t:]
        U_t = U_t.reshape(len(U_t)//n_t,n_t).mean(1)
        V_t = V_t_init[len(V_t_init)%n_t:]
        V_t = V_t.reshape(len(V_t)//n_t,n_t).mean(1)

        #peak = int(15280/n_t)
        #timestart = int(peak - (5e-3)/(n_t*32.7e-6))
        #timestop = int(peak + (5e-3)/(n_t*32.7e-6))
        #mask all but first burst
        for i in range(len(final_complist_min)):
            if i != comp_weights_list[-1]:
                print((i,int(final_complist_min[i]),int(final_complist_max[i])))
                mask = np.zeros(len(I_t))
                TSART = int(int(15280/n_t) - (5e-3)/(n_t*32.7e-6)) + int(final_complist_min[i])
                TSTOP = int(int(15280/n_t) - (5e-3)/(n_t*32.7e-6)) + int(final_complist_max[i])
                mask[TSART:TSTOP] = 1
                I_t = ma.masked_array(I_t,mask)
                Q_t = ma.masked_array(Q_t,mask)
                U_t = ma.masked_array(U_t,mask)
                V_t = ma.masked_array(V_t,mask)

        (peak,timestart1,timestop1) = find_peak((I_t,I_t),width_native,t_samp,n_t,buff=buff,pre_calc_tf=True)
        timestart2 = timestart1 - int( (1e-3)/(n_t*32.7e-6))
        timestop2 = timestop1 + int( (1e-3)/(n_t*32.7e-6))
        #I_t_weights=get_weights(I,Q,U,V,width_native,t_samp,n_f,n_t,freq_test,timeaxis,fobj,n_off=n_off,buff=buff,n_t_weight=n_t_weight,sf_window_weights=sf_window_weights)        
        I_t_weights=get_weights_1D(I_t,Q_t,U_t,V_t,timestart1,timestop1,width_native,t_samp,1,n_t,freq_test_init,timeaxis,fobj,n_off=n_off,buff=buff,n_t_weight=n_t_weight,sf_window_weights=sf_window_weights,padded=True,norm=False)
        I_t_weights_cut = I_t_weights[timestart2:timestop2]

        #get polarization of component
        [(p_f,p_t,avg,sigma_frac,snr_frac),(L_f,L_t,avg_L,sigma_L,snr_L),(C_f,C_t,avg_C,sigma_C,snr_C),(C_f_u,C_t_u,avg_C_u,sigma_C_u,snr_C_u),snr] = get_pol_fraction([I_t,np.ones(1)],[Q_t,np.zeros(1)],[U_t,np.zeros(1)],[V_t,np.zeros(1)],width_native,t_samp,n_t,1,np.ones(1),n_off=int(12000/n_t),plot=False,pre_calc_tf=True,show=False,normalize=True,buff=buff,weighted=True,timeaxis=timeaxis,fobj=fobj,unbias=True,input_weights=I_t_weights,allowed_err=1,unbias_factor=1)

        """
        TPOL_input.set_val(str(np.around(avg,2)) + "(" + str(int(np.around(sigma_frac*100,0))) + ")")
        LPOL_input.set_val(str(np.around(avg_L,2)) + "(" + str(int(np.around(sigma_L*100,0))) + ")")
        CPOL_input.set_val(str(np.around(avg_C,2)) + "(" + str(int(np.around(sigma_C*100,0))) + ")")
        SCPOL_input.set_val(str(np.around(avg_C_u,2)) + "(" + str(int(np.around(sigma_C_u*100,0))) + ")")
                
        SNR_input.set_val(np.around(snr,2))
        TSNR_input.set_val(np.around(snr_frac,2))
        LSNR_input.set_val(np.around(snr_L,2))
        CSNR_input.set_val(np.around(snr_C,2))
        

        TPOL_str.append(str(np.around(avg,2)) + "(" + str(int(np.around(sigma_frac*100,0))) + ")")
        LPOL_str.append(str(np.around(avg_L,2)) + "(" + str(int(np.around(sigma_L*100,0))) + ")")
        CPOL_str.append(str(np.around(avg_C,2)) + "(" + str(int(np.around(sigma_C*100,0))) + ")")
        SCPOL_str.append(str(np.around(avg_C_u,2)) + "(" + str(int(np.around(sigma_C_u*100,0))) + ")")

        SNR_str.append(str(np.around(snr,2)))
        TSNR_str.append(str(np.around(snr_frac,2)))
        LSNR_str.append(str(np.around(snr_L,2)))
        CSNR_str.append(str(np.around(snr_C,2)))


        TPOL_input.set_val(',\n'.join(TPOL_str))
        LPOL_input.set_val(',\n'.join(LPOL_str))
        CPOL_input.set_val(',\n'.join(CPOL_str))
        SCPOL_input.set_val(',\n'.join(SCPOL_str))


        TSNR_input.set_val(',\n'.join(TSNR_str))
        LSNR_input.set_val(',\n'.join(LSNR_str))
        CSNR_input.set_val(',\n'.join(CSNR_str))
        SNR_input.set_val(',\n'.join(SNR_str))
        """

        if len(comp_weights) < len(comp_weights_list):
            print((len(comp_weights),len(comp_weights_list)))
            comp_widths.append(width_native)
            comp_weights.append(copy.deepcopy(I_t_weights))
            comp_n_t_weights.append(n_t_weight)
            comp_sf_window_weights.append(sf_window_weights)
            comp_buffs.append(buff)
            comp_timestarts.append(timestart1)
            comp_timestops.append(timestop1)
            comp_TPOLs.append(avg)
            comp_LPOLs.append(avg_L)
            comp_CPOLs.append(avg_C)
            comp_SCPOLs.append(avg_C_u)
            comp_TSNRs.append(snr_frac)
            comp_LSNRs.append(snr_L)
            comp_CSNRs.append(snr_C)
            comp_SNRs.append(snr)
            TPOL_str.append(str(np.around(avg,2)) + "(" + str(int(np.around(sigma_frac*100,0))) + ")")
            LPOL_str.append(str(np.around(avg_L,2)) + "(" + str(int(np.around(sigma_L*100,0))) + ")")
            CPOL_str.append(str(np.around(avg_C,2)) + "(" + str(int(np.around(sigma_C*100,0))) + ")")
            SCPOL_str.append(str(np.around(avg_C_u,2)) + "(" + str(int(np.around(sigma_C_u*100,0))) + ")")

            SNR_str.append(str(np.around(snr,2)))
            TSNR_str.append(str(np.around(snr_frac,2)))
            LSNR_str.append(str(np.around(snr_L,2)))
            CSNR_str.append(str(np.around(snr_C,2)))

        else:
            comp_widths[-1] = width_native
            comp_weights[-1] = copy.deepcopy(I_t_weights)
            comp_n_t_weights[-1] = n_t_weight
            comp_sf_window_weights[-1] = sf_window_weights
            comp_buffs[-1] = buff
            comp_timestarts[-1] = timestart1
            comp_timestops[-1] = timestop1
            comp_TPOLs[-1] = avg
            comp_LPOLs[-1] = avg_L
            comp_CPOLs[-1] = avg_C
            comp_SCPOLs[-1] = avg_C_u
            comp_TSNRs[-1] = snr_frac
            comp_LSNRs[-1] = snr_L
            comp_CSNRs[-1] = snr_C
            comp_SNRs[-1] = snr
            TPOL_str[-1] = str(np.around(avg,2)) + "(" + str(int(np.around(sigma_frac*100,0))) + ")"
            LPOL_str[-1] = str(np.around(avg_L,2)) + "(" + str(int(np.around(sigma_L*100,0))) + ")"
            CPOL_str[-1] = str(np.around(avg_C,2)) + "(" + str(int(np.around(sigma_C*100,0))) + ")"
            SCPOL_str[-1] = str(np.around(avg_C_u,2)) + "(" + str(int(np.around(sigma_C_u*100,0))) + ")"

            SNR_str[-1] = str(np.around(snr,2))
            TSNR_str[-1] = str(np.around(snr_frac,2))
            LSNR_str[-1] = str(np.around(snr_L,2))
            CSNR_str[-1] = str(np.around(snr_C,2))

        TPOL_input.set_val(',\n'.join(TPOL_str))
        LPOL_input.set_val(',\n'.join(LPOL_str))
        CPOL_input.set_val(',\n'.join(CPOL_str))
        SCPOL_input.set_val(',\n'.join(SCPOL_str))


        TSNR_input.set_val(',\n'.join(TSNR_str))
        LSNR_input.set_val(',\n'.join(LSNR_str))
        CSNR_input.set_val(',\n'.join(CSNR_str))
        SNR_input.set_val(',\n'.join(SNR_str))

        ax.plot(I_t_weights_cut*np.max(I_t)/np.max(I_t_weights_cut),linewidth=4,label="weights",color="purple")
        ax.plot(I_t[timestart2:timestop2],label="I")
        ax.plot(Q_t[timestart2:timestop2],label="Q")
        ax.plot(U_t[timestart2:timestop2],label="U")
        ax.plot(V_t[timestart2:timestop2],label="V")
        ax.legend(ncol=2,loc=legend_loc)
        ax.set_xlabel("Time Sample (Sampling Time {t} $\mu s$)".format(t=np.around(32.7*n_t,2)))
        ax.set_xlim(0,timestop2-timestart2)
    
    
    n_t_weight_slider.on_changed(update_time)
    sf_window_weights_slider.on_changed(update_time)
    buff1_slider.on_changed(update_time)
    buff2_slider.on_changed(update_time)
    width_slider.on_changed(update_time)


    def update_n_f(val):
        n_f = 2**(n_f_slider.val)
        n_f_slider.valtext.set_text(n_f)

        I_f_init = freq_avg_arrays[0]
        Q_f_init = freq_avg_arrays[1]
        U_f_init = freq_avg_arrays[2]
        V_f_init = freq_avg_arrays[3]

        ax1.clear()
        freq_test = (freq_test_init[0])[len(freq_test_init[0])%n_f:]
        freq_test = freq_test.reshape(len(freq_test)//n_f,n_f).mean(1)
        freq_test = [freq_test]*4
        fmin = np.min(freq_test)
        fmax = np.max(freq_test)
        I_f = I_f_init[len(I_f_init)%n_f:]
        I_f = I_f.reshape(len(I_f)//n_f,n_f).mean(1)
        Q_f = Q_f_init[len(Q_f_init)%n_f:]
        Q_f = Q_f.reshape(len(Q_f)//n_f,n_f).mean(1)
        U_f = U_f_init[len(U_f_init)%n_f:]
        U_f = U_f.reshape(len(U_f)//n_f,n_f).mean(1)
        V_f = V_f_init[len(V_f_init)%n_f:]
        V_f = V_f.reshape(len(V_f)//n_f,n_f).mean(1)

        ax1.plot(freq_test[0],I_f,label="I")
        ax1.plot(freq_test[0],Q_f,label="Q")
        ax1.plot(freq_test[0],U_f,label="U")
        ax1.plot(freq_test[0],V_f,label="V")
        ax1.set_xlabel("Frequency (MHz)")
        #ax1.legend(ncol=2,loc="upper right")
        ax1.set_xlim(fmin,fmax)

    n_f_slider.on_changed(update_n_f)
    #done_button2.on_clicked(update_done)

    RM_params = [-1e6,1e6,2e6]
    def update_RMmin(text):
        RM_params[0] = float(text)
        print(float(text))
        return
    
    def update_RMmax(text):
        RM_params[1] = float(text)
        print(float(text))
        return

    def update_RMtrials(text):
        RM_params[2] = float(text)
        print(float(text))
        return

    RMzoom_params = [1000,5000]
    def update_RMzoomrange(text):
        RMzoom_params[0] = float(text)
        print(float(text))
        return

    def update_RMzoomtrials(text):
        RMzoom_params[1] = float(text)
        print(float(text))
        return

    RMresults = []
    RMerrors = []

    RMzoomresults = []
    RMzoomerrors = []
    RMrunlist = []
    
    RMsnrs = []
    RMzoomsnrs = []
    sigflag=[]

    trial_RMtools=[]
    trial_RMtoolszoom = []
    def update_RMrun(val):
        #do RM synthesis

        n_f = 2**(n_f_slider.val)
        n_t = n_t_slider.val

        I_f_init = freq_avg_arrays[0]
        Q_f_init = freq_avg_arrays[1]
        U_f_init = freq_avg_arrays[2]
        V_f_init = freq_avg_arrays[3]
        

        #RM tools
        I = avg_time(I_init,n_t)
        Q = avg_time(Q_init,n_t)
        U = avg_time(U_init,n_t)
        V = avg_time(V_init,n_t)

        n_off = int(12000/n_t)
        
        Ierr = np.std(I[:,:n_off],axis=1)
        Ierr[Ierr.mask] = np.nan
        Ierr = Ierr.data

        Qerr = np.std(Q[:,:n_off],axis=1)
        Qerr[Qerr.mask] = np.nan
        Qerr = Qerr.data

        Uerr = np.std(U[:,:n_off],axis=1)
        Uerr[Uerr.mask] = np.nan
        Uerr = Uerr.data

        I_f_rmtools = I_f_init.data
        I_f_rmtools[I_f_init.mask] = np.nan

        Q_f_rmtools = Q_f_init.data
        Q_f_rmtools[Q_f_init.mask] = np.nan

        U_f_rmtools = U_f_init.data
        U_f_rmtools[U_f_init.mask] = np.nan

        if len(RMrunlist) == 0:
            RMrunlist.append(0)
            #RM synthesis
            trial_RM = np.linspace(RM_params[0],RM_params[1],int(RM_params[2]))
            trial_phi = [0] 
            RM1,phi1,RMsnrs1,RMerr1 = faradaycal(I_f_init,Q_f_init,U_f_init,V_f_init,freq_test_init,trial_RM,trial_phi,plot=False,show=False,fit_window=100,err=True)

            trial_RM_tools = np.linspace(RM_params[0],RM_params[1],int(1e4))
            out=run_rmsynth([freq_test_init[0]*1e6,I_f_rmtools,Q_f_rmtools,U_f_rmtools,Ierr,Qerr,Uerr],phiMax_radm2=np.max(trial_RM_tools),dPhi_radm2=np.abs(trial_RM_tools[1]-trial_RM_tools[0]))

            print("Cleaning...")
            out=run_rmclean(out[0],out[1],2)
            print("RM Tools estimate: " + str(out[0]["phiPeakPIchan_rm2"]) + "\pm" + str(out[0]["dPhiPeakPIchan_rm2"]) + " rad/m^2")


            RMresult_input.set_val(str(np.around(RM1,2)))
            RMerror_input.set_val(str(np.around(RMerr1,2)))
        
            RMresults.append(RM1)
            RMresults.append(out[0]["phiPeakPIchan_rm2"])
            
            RMsnrs.append(RMsnrs1)
            RMsnrs.append(np.abs(out[1]["cleanFDF"]))

            RMerrors.append(RMerr1)
            RMerrors.append(out[0]["dPhiPeakPIchan_rm2"])
            
            #plot result
            ax2.plot(trial_RM,RMsnrs1,label="RM Synthesis",color="black")
            ax2.plot(out[1]["phiArr_radm2"],np.abs(out[1]["cleanFDF"]),alpha=0.5,label="RM Tools",color="blue")
            ax2.legend(ncol=2,loc="upper right")
        
            RMapply_button.set_active(True)
            trial_RMtools.append(out[1]["phiArr_radm2"])
        elif len(RMrunlist) == 1:
            RMrunlist.append(1)
            #RM synthesis
            trial_RM = np.linspace(RMresults[0]-RMzoom_params[0],RMresults[0]+RMzoom_params[0],int(RMzoom_params[1]))
            trial_phi = [0]
            RM1,phi1,RMsnrs1,RMerr1 = faradaycal(I_f_init,Q_f_init,U_f_init,V_f_init,freq_test_init,trial_RM,trial_phi,plot=False,show=False,fit_window=100,err=True)
            RMzoomresults.append(RM1)
            RMzoomerrors.append(RMerr1)
        

            if np.abs(RM1) < 1E3:
                out=run_rmsynth([freq_test_init[0]*1e6,I_f_rmtools,Q_f_rmtools,U_f_rmtools,Ierr,Qerr,Uerr],phiMax_radm2=np.max(trial_RM),dPhi_radm2=np.abs(trial_RM[1]-trial_RM[0]))

                print("Cleaning...")
                out=run_rmclean(out[0],out[1],2)
                print("RM Tools estimate: " + str(out[0]["phiPeakPIchan_rm2"]) + "\pm" + str(out[0]["dPhiPeakPIchan_rm2"]) + " rad/m^2")
                RMzoomresults.append(out[0]["phiPeakPIchan_rm2"])
                RMzoomerrors.append(out[0]["dPhiPeakPIchan_rm2"])

                L1=ax3_1.plot(out[1]["phiArr_radm2"],np.abs(out[1]["cleanFDF"]),label="RM Tools",color="blue")
                trial_RMtoolszoom.append(out[1]["phiArr_radm2"])
            else:
                print("RM Magnitude out of range for RM tools")
                RMzoomresults.append(np.nan)
                RMzoomerrors.append(np.nan)
                trial_RMtoolszoom.append(np.nan)
            #S/N method
            RM2,phi2,RMsnrs2,RMerr2,upp,low,sig,QUnoise = faradaycal_SNR(I,Q,U,V,freq_test_init,trial_RM,trial_phi,init_width_native,fobj.header.tsamp,plot=False,n_f=n_f,n_t=n_t,show=False,err=True,buff=comp_buffs[-1],weighted=True,n_off=int(12000/n_t),input_weights=(comp_weights[-1])[np.min(comp_timestarts):np.max(comp_timestops)],timestart_in=np.min(comp_timestarts),timestop_in=np.max(comp_timestops))
            RMerr_fit = RM_error_fit(np.max(RMsnrs2))
            RMresult_input.set_val(str(np.around(RM2,2)))
            RMerror_input.set_val(str(np.around(RMerr_fit,2)))
            RMzoomresults.append(RM2)
            RMzoomerrors.append(RMerr_fit)
            
            RMzoomsnrs.append(RMsnrs1)
            RMzoomsnrs.append(RMsnrs2)
            if np.abs(RM1) < 1E3:
                RMzoomsnrs.append(np.abs(out[1]["cleanFDF"]))
            #plot result
            L2=ax3.plot(trial_RM,RMsnrs2,label="S/N Method",color="orange",linewidth=4)
            L3=ax3_1.plot(trial_RM,RMsnrs1,label="RM Synthesis",color="black")
            ax3.legend(L1+L2+L3,["RM Tools","S/N Method","RM Synthesis"],loc="upper right")
            ax3.set_xlim(np.min(trial_RM),np.max(trial_RM))
            RMrunlist.append(1)
            if np.max(RMsnrs2) > 9:
                sigflag.append(True)
            RMapply_button.set_active(True)

    
    comp_TPOLs_cal = []#.append(avg)
    comp_LPOLs_cal = []#.append(avg_L)
    comp_CPOLs_cal = []#.append(avg_C)
    comp_SCPOLs_cal = []#.append(avg_C_u)
    comp_TSNRs_cal = []#.append(snr_frac)
    comp_LSNRs_cal = []#.append(snr_L)
    comp_CSNRs_cal = []#.append(snr_C)
    comp_SNRs_cal = []#.append(snr)

    TPOL_str_cal = []#.append(str(np.around(avg,2)) + "(" + str(int(np.around(sigma_frac*100,0))) + ")")
    LPOL_str_cal = []#.append(str(np.around(avg_L,2)) + "(" + str(int(np.around(sigma_L*100,0))) + ")")
    CPOL_str_cal = []#.append(str(np.around(avg_C,2)) + "(" + str(int(np.around(sigma_C*100,0))) + ")")
    SCPOL_str_cal = []#.append(str(np.around(avg_C_u,2)) + "(" + str(int(np.around(sigma_C_u*100,0))) + ")")

    SNR_str_cal = []#.append(str(np.around(snr,2)))
    TSNR_str_cal = []#.append(str(np.around(snr_frac,2)))
    LSNR_str_cal = []#.append(str(np.around(snr_L,2)))
    CSNR_str_cal = []#.append(str(np.around(snr_C,2)))
    def update_RMapply(val):
        if len(RMrunlist) == 1:
            RM = RMresults[0]
        else:
            RM = RMzoomresults[-1]
        print("Clearing Axes...")
        
        #clear axes and pol text boxes
        for i in range(2,len(ax.lines)):
            ax.lines.pop()
            ax.set_prop_cycle(None)
            ax.relim()
            # update ax.viewLim using the new dataLim
            ax.autoscale_view()

        ax1.clear()

        TPOL_input.set_val("")#str(np.around(avg*100,2)) + "(" + str(int(np.around(sigma_frac*10000,0))) + ") %")
        LPOL_input.set_val("")#(str(np.around(avg_L*100,2)) + "(" + str(int(np.around(sigma_L*10000,0))) + ") %")
        CPOL_input.set_val("")#(str(np.around(avg_C*100,2)) + "(" + str(int(np.around(sigma_C*10000,0))) + ") %")
        SCPOL_input.set_val("")#(str(np.around(avg_C_u*100,2)) + "(" + str(int(np.around(sigma_C_u*10000,0))) + ") %")

        SNR_input.set_val("")#(np.around(snr,2))
        TSNR_input.set_val("")#(np.around(snr_frac,2))
        LSNR_input.set_val("")#(np.around(snr_L,2))
        CSNR_input.set_val("")#(np.around(snr_C,2))

        #downsample, then RM calibrate
        n_t = n_t_slider.val
        n_f = 2**(n_f_slider.val)
        I = avg_time(I_init,n_t)
        Q = avg_time(Q_init,n_t)
        U = avg_time(U_init,n_t)
        V = avg_time(V_init,n_t)
        
        I = avg_freq(I,n_f)
        Q = avg_freq(Q,n_f)
        U = avg_freq(U,n_f)
        V = avg_freq(V,n_f)


        freq_test = (freq_test_init[0])[len(freq_test_init[0])%n_f:]
        freq_test = freq_test.reshape(len(freq_test)//n_f,n_f).mean(1)
        freq_test = [freq_test]*4

        print("RM calibrating...")
        (I,Q,U,V) = calibrate_RM(I,Q,U,V,RM,0,freq_test,stokes=True)
        (I_t,Q_t,U_t,V_t) = get_stokes_vs_time(I,Q,U,V,init_width_native,fobj.header.tsamp,n_t,n_off=n_off,plot=False,show=False,datadir=datadir,normalize=True)
        (I_f,Q_f,U_f,V_f) = get_stokes_vs_freq(I,Q,U,V,init_width_native,fobj.header.tsamp,n_f,n_t,freq_test,n_off=n_off,plot=False,show=False,normalize=True,weighted=True,timeaxis=timeaxis,fobj=fobj,input_weights=comp_weights[-1])


        #replot
        peak = int(15280/n_t)
        timestart = int(peak - (5e-3)/(n_t*32.7e-6))
        timestop = int(peak + (5e-3)/(n_t*32.7e-6))
        ax.plot((comp_weights[-1]*np.max(I_t)/np.max(comp_weights[-1]))[timestart:timestop],linewidth=4,label="weights",color="purple")
        ax.plot(I_t[timestart:timestop],label="I")
        ax.plot(Q_t[timestart:timestop],label="Q")
        ax.plot(U_t[timestart:timestop],label="U")
        ax.plot(V_t[timestart:timestop],label="V")
        ax.legend(ncol=2,loc=legend_loc)
        ax.set_xlabel("Time Sample (Sampling Time {t} $\mu s$)".format(t=np.around(32.7*n_t,2)))
        ax.set_xlim(0,timestop-timestart)

        fmin = np.min(freq_test[0])
        fmax = np.max(freq_test[0])
        ax1.plot(freq_test[0],I_f,label="I")
        ax1.plot(freq_test[0],Q_f,label="Q")
        ax1.plot(freq_test[0],U_f,label="U")
        ax1.plot(freq_test[0],V_f,label="V")
        ax1.set_xlabel("Frequency (MHz)")
        ax1.set_xlim(fmin,fmax)
    
        #recalculate polarization (how do you want to do this?)
        for i in range(len(comp_weights)):
            #get weights
            I_t_weights = comp_weights[i]
            if i == len(comp_weights)-1:
                width_native = comp_widths[i-1]
                buff = comp_widths[i-1]
            else:
                width_native = comp_widths[i]
                buff = comp_buffs[i]

            #compute pol fraction
            [(p_f,p_t,avg,sigma_frac,snr_frac),(L_f,L_t,avg_L,sigma_L,snr_L),(C_f,C_t,avg_C,sigma_C,snr_C),(C_f_u,C_t_u,avg_C_u,sigma_C_u,snr_C_u),snr] = get_pol_fraction([I_t,np.ones(1)],[Q_t,np.zeros(1)],[U_t,np.zeros(1)],[V_t,np.zeros(1)],width_native,t_samp,n_t,1,np.ones(1),n_off=int(12000/n_t),plot=False,pre_calc_tf=True,show=False,normalize=True,buff=buff,weighted=True,timeaxis=timeaxis,fobj=fobj,unbias=True,input_weights=I_t_weights,allowed_err=1,unbias_factor=1)

            #write in text box
            comp_TPOLs_cal.append(avg)
            comp_LPOLs_cal.append(avg_L)
            comp_CPOLs_cal.append(avg_C)
            comp_SCPOLs_cal.append(avg_C_u)
            comp_TSNRs_cal.append(snr_frac)
            comp_LSNRs_cal.append(snr_L)
            comp_CSNRs_cal.append(snr_C)
            comp_SNRs_cal.append(snr)
    
            TPOL_str_cal.append(str(np.around(avg,2)) + "(" + str(int(np.around(sigma_frac*100,0))) + ")")
            LPOL_str_cal.append(str(np.around(avg_L,2)) + "(" + str(int(np.around(sigma_L*100,0))) + ")")
            CPOL_str_cal.append(str(np.around(avg_C,2)) + "(" + str(int(np.around(sigma_C*100,0))) + ")")
            SCPOL_str_cal.append(str(np.around(avg_C_u,2)) + "(" + str(int(np.around(sigma_C_u*100,0))) + ")")

            SNR_str_cal.append(str(np.around(snr,2)))
            TSNR_str_cal.append(str(np.around(snr_frac,2)))
            LSNR_str_cal.append(str(np.around(snr_L,2)))
            CSNR_str_cal.append(str(np.around(snr_C,2)))
        
        TPOL_input.set_val(',\n'.join(TPOL_str_cal))
        LPOL_input.set_val(',\n'.join(LPOL_str_cal))
        CPOL_input.set_val(',\n'.join(CPOL_str_cal))
        SCPOL_input.set_val(',\n'.join(SCPOL_str_cal))


        TSNR_input.set_val(',\n'.join(TSNR_str_cal))
        LSNR_input.set_val(',\n'.join(LSNR_str_cal))
        CSNR_input.set_val(',\n'.join(CSNR_str_cal))
        SNR_input.set_val(',\n'.join(SNR_str_cal))


    RMmax_input.on_submit(update_RMmax)
    RMmin_input.on_submit(update_RMmin)
    RMtrials_input.on_submit(update_RMtrials)
    RMapply_button.on_clicked(update_RMapply)
    RMrun_button.on_clicked(update_RMrun)
    RMzoomtrials_input.on_submit(update_RMzoomtrials)
    RMzoomrange_input.on_submit(update_RMzoomrange)



    def update_reset(val):
        #ax.clear()
        for i in range(2,len(ax.lines)):
            ax.lines.pop()
            ax.set_prop_cycle(None)

            ax.relim()
            # update ax.viewLim using the new dataLim
            ax.autoscale_view()
        
        ax1.clear()
        ax2.clear()
        ax3.clear()

        n_t_slider.val = 1
        n_t = 1

        peak = int(15280)
        timestart = int(peak - (5e-3)/(32.7e-6))
        timestop = int(peak + (5e-3)/(32.7e-6))
        #t = 32.7*np.arange(timestop-timestart)/1000
        ax.plot(I_t[timestart:timestop],label="I")
        ax.plot(Q_t[timestart:timestop],label="Q")
        ax.plot(U_t[timestart:timestop],label="U")
        ax.plot(V_t[timestart:timestop],label="V")
        ax.legend(ncol=2,loc=legend_loc)
        ax.set_xlabel("Time Sample (Sampling Time {t} $\mu s$)".format(t=np.around(32.7,2)))
        ax.set_xlim(0,timestop-timestart)


        ax1.set_xlabel("Frequency (MHz)")
        ax1.set_xlim(np.min(freq_test_init[0]),np.max(freq_test_init[0]))

        ax2.set_title("RM Analysis")
        ax2.set_xlabel("RM (rad/m^2)")
        ax2.set_ylabel(r'$F(\phi)$')

        ax3.set_xlabel(r'RM ($rad/m^2$)')
        ax3.set_ylabel("Linear S/N")
        ax3_1.set_ylabel(r'$F(\phi)$')

        #reset button activity values
        n_t_weight_slider.set_active(False)
        sf_window_weights_slider.set_active(False)
        buff1_slider.set_active(False)
        buff2_slider.set_active(False)
        width_slider.set_active(False)
        n_f_slider.set_active(False)
        RMmax_input.set_active(False)
        RMmin_input.set_active(False)
        RMtrials_input.set_active(False)
        RMapply_button.set_active(False)
        RMrun_button.set_active(False)
        RMresult_input.set_active(False)
        RMerror_input.set_active(False)
        RMzoomrange_input.set_active(False)
        RMzoomtrials_input.set_active(False)

        final_complist_min.clear() # = []
        final_complist_max.clear() # = []
        complist_min.clear() # = []
        complist_max.clear() # = []
        donelist.clear() # = []

        #span.set_active(False)
        #span.set_visible(True)
        #span_lst[0] = SpanSelector(ax,update_component,"horizontal",interactive=True,props=dict(facecolor='red', alpha=0.5))
        #span_lst[0].set_active(False)
        #span_lst[0].set_visible(True)


        comp_widths.clear() # = []
        comp_weights.clear() # = []
        comp_n_t_weights.clear() # = []
        comp_sf_window_weights.clear() # = []
        comp_buffs.clear() # = []
        comp_weights_list.clear() # = [0]
        comp_weights_list.append(0)
        comp_timestarts.clear() # = []
        comp_timestops.clear() # =[]

        freq_avg_arrays.clear() # = []
        RMresults.clear() # = []
        RMerrors.clear() # = []

        RMzoomresults.clear() # = []
        RMzoomerrors.clear() # = []
        RMrunlist.clear() # = []

        RMsnrs.clear() # = []
        RMzoomsnrs.clear() # = []
        sigflag.clear()
        
        return

    reset_button.on_clicked(update_reset)


    plt.show()
    plt.close(fig)


    #return weights, average weights, and everything needed to reproduce them
    outdict =dict()
  
    outdict["component weights"] = comp_weights[:-1]
    outdict["weights"] = comp_weights[-1]
    outdict["n_t"] = n_t_slider.val
    outdict["n_f"] = 2**(n_f_slider.val)#n_f_slider.val
    outdict["n_t_weights"] = comp_n_t_weights
    outdict["buff"] = comp_buffs#[buff1_slider.val,buff2_slider.val]
    outdict["width"] = comp_widths
    outdict["sf_window_weights"] = comp_sf_window_weights
    outdict["timestarts"] = comp_timestarts
    outdict["timestops"] = comp_timestops
    outdict["num_components"] = len(comp_n_t_weights)
    outdict["RM_results"] = RMresults
    outdict["RM_errors"] = RMerrors
    outdict["RM_snrs"] = RMsnrs
    outdict["RM_zoomresults"] = RMzoomresults
    outdict["RM_zoomerrors"] = RMzoomerrors
    outdict["RM_zoomsnrs"] = RMzoomsnrs


    #output plots
    n_t = n_t_slider.val
    n_f = 2**(n_f_slider.val)
    
    buff1 = comp_buffs[0]
    buff2 = comp_buffs[1]
    buff = []
    if isinstance(buff1, int):
        buff.append(buff1)
    else:
        buff.append(buff1[0])

    if isinstance(buff2, int):
        buff.append(buff2)
    else:
        buff.append(buff2[0])

    I = avg_time(I_init,n_t)
    Q = avg_time(Q_init,n_t)
    U = avg_time(U_init,n_t)
    V = avg_time(V_init,n_t)

    I = avg_freq(I,n_f)  
    Q = avg_freq(Q,n_f)
    U = avg_freq(U,n_f)
    V = avg_freq(V,n_f)

    freq_test = (freq_test_init[0])[len(freq_test_init[0])%n_f:]
    freq_test = freq_test.reshape(len(freq_test)//n_f,n_f).mean(1)
    freq_test = [freq_test]*4

    I,Q,U,V = calibrate_RM(I,Q,U,V,RMzoomresults[-1],0,freq_test,stokes=True) #total derotation

    pol_summary_plot(I,Q,U,V,ids,nickname,comp_widths[0],t_samp,n_t_slider.val,2**(n_f_slider.val),freq_test,timeaxis,fobj,n_off=int(12000/n_t),buff=buff,weighted=True,n_t_weight=2,sf_window_weights=7,show=False,input_weights=comp_weights[-1],intL=-1,intR=-1,multipeaks=multi_button.get_status(),wind=n_t,suffix="",mask_flag=mask_flag,sigflag=sigflag[0],plot_weights=False)
    trial_RM1 = np.linspace(RM_params[0],RM_params[1],int(RM_params[2]))
    trial_RM2 = np.linspace(RMresults[0]-RMzoom_params[0],RMresults[0]+RMzoom_params[0],int(RMzoom_params[1]))
    RM_summary_plot(ids,nickname,RMsnrs,RMzoomsnrs,RMzoomresults[-1],RMzoomerrors[-1],trial_RM1,trial_RM2,trial_RMtools[0],trial_RMtoolszoom[0],threshold=9,suffix="",show=False)
    
    return outdict


#plotting functions
def RM_summary_plot(ids,nickname,RMsnrs,RMzoomsnrs,RM,RMerror,trial_RM1,trial_RM2,trial_RMtools1,trial_RMtools2,threshold=9,suffix="",show=True,title="",figsize=(38,28),datadir=None):
    if datadir is None:
        #datadir="/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/"+ids + "_" + nickname + "/"
        datadir=dirs['data'] +ids + "_" + nickname + "/"

    #parse results
    RMsynth_snrs, RMtools_snrs = RMsnrs
    if len(RMzoomsnrs) == 3:
        RMsynth_zoomsnrs, RMtools_zoomsnrs, RMSNR_zoomsnrs = RMzoomsnrs
    else:
        RMsynth_zoomsnrs, RMSNR_zoomsnrs = RMzoomsnrs


    fig, axs = plt.subplots(2,1,figsize=figsize, gridspec_kw={'height_ratios': [1, 1]})

    ax4= axs[0]# = plt.subplot2grid(shape=(2, 2), loc=(0, 0), colspan=2)
    ax5 =axs[1]#= plt.subplot2grid(shape=(2, 2), loc=(1, 0), colspan=2,rowspan=2)

    #set title
    if title != None:
        if title != "":
            ax4.set_title(title)
        elif nickname == "220912A" and ids == "221018aaaj":
            ax4.set_title("FRB20220912A Burst 1 RM Analysis")
        elif nickname == "220912A":
            ax4.set_title("FRB20220912A Burst 2 RM Analysis")
        else:
            ax4.set_title("FRB20" + ids[:6] + " RM Analysis")

    #full range plot
    ax4.plot(trial_RM1,RMsynth_snrs,label="RM synthesis",color="black")
    ax4.plot(trial_RMtools1,RMtools_snrs,alpha=0.5,label="RM Tools",color="blue") 
    ax4.legend(loc="upper right")
    ax4.set_xlabel(r'$RM\,(rad/m^2)$')
    ax4.set_ylabel(r'$F(\phi)$')
    ax4.set_xlim(np.min(trial_RM1),np.max(trial_RM1))

    #zoom range plot
    lns = []
    labs = []
    l1 = ax5.plot(trial_RM2,RMSNR_zoomsnrs,label="S/N Method",color="orange",linewidth=4)
    lns.append(l1[0])
    labs.append("S/N Method")
    l3 = ax5.axvline(RM+RMerror,color="red",label="RM Error",linewidth=2)
    lns.append(l3)
    labs.append("RM Error")
    ax5.axvline(RM-RMerror,color="red",linewidth=2)

    ax5_1 = ax5.twinx()
    l4 = ax5_1.plot(trial_RM2,RMsynth_zoomsnrs,label="RM Synthesis",color="black")
    lns.append(l4[0])
    labs.append("RM Synthesis")
    if len(RMzoomsnrs)==3:
        l2=ax5_1.plot(trial_RMtools2,RMtools_zoomsnrs,label="RM Tools",color="blue")
        lns.append(l2[0])
        labs.append("RM Tools")
    ax5.set_xlim(np.min(trial_RM2),np.max(trial_RM2))
    l6 = ax5.axhline(threshold,color="purple",linestyle="--",label=r'${t}\sigma$ threshold'.format(t=threshold),linewidth=2)
    lns.append(l6)
    ax5_1.set_ylabel(r'$F(\phi)$')

    ax5.legend(lns, labs, loc="upper right")
    ax5.set_xlabel("$RM\,(rad/m^2)$")
    ax5.set_ylabel("Linear S/N")



    plt.savefig(datadir + ids + "_" + nickname + "_RMsummary_plot" + suffix + ".pdf")
    if show:
        plt.show()
    plt.close(fig)
    

    return



def pol_summary_plot(I,Q,U,V,ids,nickname,width_native,t_samp,n_t,n_f,freq_test,timeaxis,fobj,n_off=3000,buff=0,weighted=True,n_t_weight=2,sf_window_weights=7,show=True,input_weights=[],intL=-1,intR=-1,multipeaks=False,wind=1,suffix="",mask_flag=False,sigflag=True,plot_weights=False,timestart=-1,timestop=-1,short_labels=True,unbias_factor=1,add_title=False,mask_th=0.005,SNRCUT=None,ghostPA=False,showspectrum=True,figsize=(42,36),datadir=None):
    """
    given calibrated I Q U V, generates plot w/ PA, pol profile and spectrum, Stokes I dynamic spectrum
    """
    if datadir is None:
        datadir = dirs['data'] + ids + "_" + nickname + "/"
    #use full timestream for calibrators
    if width_native == -1:
        timestart = 0
        timestop = I.shape[1]
    elif timestart == -1 and timestop == -1:
        peak,timestart,timestop = find_peak(I,width_native,t_samp,n_t,buff=buff)

    #optimal weighting
    if weighted:
        if input_weights == []:
            I_t_weights=get_weights(I,Q,U,V,width_native,t_samp,n_f,n_t,freq_test,timeaxis,fobj,n_off=n_off,buff=buff,n_t_weight=n_t_weight,sf_window_weights=sf_window_weights)
            if multipeaks:
                pks,props = find_peaks(I_t_weights,height=0.01)
                #pks = pks[1:]
                FWHM,heights,intL,intR = peak_widths(I_t_weights,pks)
                intL = intL[0]
                intR = intR[-1]
            else:
                FWHM,heights,intL,intR = peak_widths(I_t_weights,[np.argmax(I_t_weights)])
        else:
            I_t_weights = input_weights

    else:
        intL = timestart
        intR = timestop
        I_t_weights = [] 

    #get IQUV vs time and freq
    (I_t,Q_t,U_t,V_t) = get_stokes_vs_time(I,Q,U,V,width_native,fobj.header.tsamp,n_t,n_off=n_off,plot=False,show=False,datadir=datadir,normalize=True,buff=buff,window=3,label="")
    (I_f,Q_f,U_f,V_f) = get_stokes_vs_freq(I,Q,U,V,width_native,fobj.header.tsamp,n_f,n_t,freq_test,n_off=n_off,plot=False,show=False,datadir=datadir,normalize=True,buff=buff,weighted=weighted,input_weights=I_t_weights,n_t_weight=n_t_weight,timeaxis=timeaxis,fobj=fobj,sf_window_weights=sf_window_weights,label="")
    [(pol_f,pol_t,avg,sigma_frac,snr_frac),(L_f,L_t,avg_L,sigma_L,snr_L),(C_f_unbiased,C_t_unbiased,avg_C_abs,sigma_C_abs,snr_C),(C_f,C_t,avg_C,sigma_C,snr_C),snr] = get_pol_fraction(I,Q,U,V,width_native,fobj.header.tsamp,n_t,n_f,freq_test,n_off=n_off,plot=False,show=False,datadir=datadir,normalize=True,buff=buff,full=False,weighted=weighted,input_weights=I_t_weights,n_t_weight=n_t_weight,timeaxis=timeaxis,fobj=fobj,sf_window_weights=sf_window_weights,label="")

    L_t = L_t*I_t
    L_f = L_f*I_f

    C_t = C_t*I_t
    C_f = C_f*I_f

    C_t_unbiased = C_t_unbiased*I_t
    C_f_unbiased = C_f_unbiased*I_f

    
    L_t = np.sqrt(Q_t**2 + U_t**2)
    L_t[L_t**2 <= (unbias_factor*np.std(I_t[:n_off]))**2] = np.std(I_t[:n_off])
    L_t = np.sqrt(L_t**2 - np.std(I_t[:n_off])**2)
    L_f = np.sqrt(Q_f**2 + U_f**2)
    L_f[L_f**2 <= (unbias_factor*np.std(I_t[:n_off]))**2] = np.std(I_t[:n_off])
    L_f = np.sqrt(L_f**2 - np.std(I_t[:n_off])**2)


    C_t = V_t
    C_f = V_f


    #get PA vs time and freq
    PA_f,PA_t,PA_f_errs,PA_t_errs,avg_PA,PA_err = get_pol_angle(I,Q,U,V,width_native,fobj.header.tsamp,n_t,n_f,freq_test,n_off=n_off,plot=False,show=False,datadir=datadir,normalize=True,buff=buff,weighted=weighted,input_weights=I_t_weights,n_t_weight=n_t_weight,timeaxis=timeaxis,fobj=fobj,sf_window_weights=sf_window_weights,label="")

    #plot

    #Summary plot
    t = (t_samp*1e6)*n_t*np.arange(0,I.shape[1])
    tshifted = (t - np.argmax(I_t)*(t_samp*1e6)*n_t)/1000
    tpeak = (np.argmax(I_t)*(t_samp*1e6)*n_t)/1000
    
    if showspectrum:
        fig= plt.figure(figsize=figsize)
        ax0 = plt.subplot2grid(shape=(8, 8), loc=(0, 0), colspan=4)
        ax1 = plt.subplot2grid(shape=(8, 8), loc=(1, 0), colspan=4,rowspan=2,sharex=ax0)
        ax2 = plt.subplot2grid(shape=(8, 8), loc=(3, 0), colspan=4, rowspan=4)
        ax3 = plt.subplot2grid(shape=(8, 8), loc=(3, 4), rowspan=4,colspan=2)
        ax6 = plt.subplot2grid(shape=(8, 8), loc=(3,6), rowspan=4,colspan=1)
    else:
        smallfigsize = (figsize[0]//2,figsize[1])
        fig= plt.figure(figsize=smallfigsize)
        ax0 = plt.subplot2grid(shape=(8, 4), loc=(0, 0), colspan=4)
        ax1 = plt.subplot2grid(shape=(8, 4), loc=(1, 0), colspan=4,rowspan=2,sharex=ax0)
        ax2 = plt.subplot2grid(shape=(8, 4), loc=(3, 0), colspan=4, rowspan=4)
        #ax3 = plt.subplot2grid(shape=(8, 8), loc=(3, 4), rowspan=4,colspan=2)
        #ax6 = plt.subplot2grid(shape=(8, 8), loc=(3,6), rowspan=4,colspan=1)

    intL = int(intL)
    intR = int(intR)

    if ghostPA:
        ax0.errorbar(tshifted,(180/np.pi)*PA_t,yerr=(180/np.pi)*PA_t_errs,fmt='o',color="blue",markersize=10,linewidth=2,alpha=0.15)

    if sigflag:
        if short_labels:
            l = "PPA"
        else:
            l = "Intrinsic PPA"
        if mask_flag:
            ax0.errorbar(tshifted[intL:intR][I_t_weights[intL:intR] > mask_th],(180/np.pi)*PA_t[intL:intR][I_t_weights[intL:intR] > mask_th],yerr=(180/np.pi)*PA_t_errs[intL:intR][I_t_weights[intL:intR] > mask_th],fmt='o',label=l,color="blue",markersize=10,linewidth=2)
        elif SNRCUT!=None:
            ax0.errorbar(tshifted[intL:intR][L_t[intL:intR]>=SNRCUT],(180/np.pi)*PA_t[intL:intR][L_t[intL:intR]>=SNRCUT],yerr=(180/np.pi)*PA_t_errs[intL:intR][L_t[intL:intR]>=SNRCUT],fmt='o',label=l,color="blue",markersize=10,linewidth=2)
        else:
            ax0.errorbar(tshifted[intL:intR],(180/np.pi)*PA_t[intL:intR],yerr=(180/np.pi)*PA_t_errs[intL:intR],fmt='o',label=l,color="blue",markersize=10,linewidth=2)
    else:
        if short_labels:
            l = "PA"
        else:
            l = "Measured PA"

        if mask_flag:
            ax0.errorbar(tshifted[intL:intR][I_t_weights[intL:intR] > mask_th],(180/np.pi)*PA_t[intL:intR][I_t_weights[intL:intR] > mask_th],yerr=(180/np.pi)*PA_t_errs[intL:intR][I_t_weights[intL:intR] > mask_th],fmt='o',label=l,color="blue",markersize=10,linewidth=2)
        elif SNRCUT != None:
            ax0.errorbar(tshifted[intL:intR][L_t[intL:intR]>=SNRCUT],(180/np.pi)*PA_t[intL:intR][L_t[intL:intR]>=SNRCUT],yerr=(180/np.pi)*PA_t_errs[intL:intR][L_t[intL:intR]>=SNRCUT],fmt='o',label=l,color="blue",markersize=10,linewidth=2)
        else:
            ax0.errorbar(tshifted[intL:intR],(180/np.pi)*PA_t[intL:intR],yerr=(180/np.pi)*PA_t_errs[intL:intR],fmt='o',label=l,color="blue",markersize=10,linewidth=2)

    ax0.set_xlim(((timestart - np.argmax(I_t))*(t_samp*1e6)*n_t)/1000-wind,((timestop - np.argmax(I_t))*(t_samp*1e6)*n_t)/1000 +wind)
    if sigflag:
        ax0.set_ylabel(r'PPA ($^\circ$)')
    else:
        ax0.set_ylabel(r'PA ($^\circ$)')


    #ax0.set_ylim(-1.1*95,1.1*95)
    ax0.set_ylim(-1.4*95,1.1*95)
    #ax0.legend(loc="upper right")

    if showspectrum:
        if ghostPA:
            ax6.errorbar((180/np.pi)*PA_f,freq_test[0],xerr=(180/np.pi)*PA_f_errs,fmt='o',color="blue",markersize=10,linewidth=2,alpha=0.15)
    
        if sigflag:
            if SNRCUT != None:
                ax6.errorbar((180/np.pi)*PA_f[L_f >=SNRCUT],freq_test[0][L_f >= SNRCUT],xerr=(180/np.pi)*PA_f_errs[L_f >= SNRCUT],fmt='o',label="Intrinsic PPA",color="blue",markersize=10,linewidth=2)
            else:
                ax6.errorbar((180/np.pi)*PA_f,freq_test[0],xerr=(180/np.pi)*PA_f_errs,fmt='o',label="Intrinsic PPA",color="blue",markersize=10,linewidth=2)
        else:
            if SNRCUT != None:
                ax6.errorbar((180/np.pi)*PA_f[L_f >=SNRCUT],freq_test[0][L_f >= SNRCUT],xerr=(180/np.pi)*PA_f_errs[L_f >= SNRCUT],fmt='o',label="Measured PA",color="blue",markersize=10,linewidth=2)
            else:
                ax6.errorbar((180/np.pi)*PA_f,freq_test[0],xerr=(180/np.pi)*PA_f_errs,fmt='o',label="Measured PA",color="blue",markersize=10,linewidth=2)

        if sigflag:
            ax6.set_xlabel(r'PPA ($^\circ$)')
        else:
            ax6.set_xlabel(r'PA ($^\circ$)')
        #ax6.set_xlabel("degrees")
        ax6.set_xlim(-1.1*95,1.1*95)
        ax6.set_xlim(-1.4*95,1.1*95)
        ax6.set_ylim(np.min(freq_test[0]),np.max(freq_test[0]))
        #ax6.tick_params(axis='x', labelrotation = 45)

    #pol fracs
    if short_labels:
        ax1.step(tshifted,I_t,label=r'$I$',color="black",linewidth=3)#label=r'Intensity (I)',color="black",linewidth=3)
        #plt.plot(t,T_t,label=r'Total Polarization ($\sqrt{Q^2 + U^2 + V^2}/I$)')
        ax1.step(tshifted,L_t,label=r'$L$',color="blue",linewidth=2.5)#label=r'Linear Polarization (L)',color="blue",linewidth=2.5)
        ax1.step(tshifted,C_t,label=r'$V$',color="orange",linewidth=2)#label=r'Circular Polarization (V)',color="orange",linewidth=2)
    else:
        ax1.step(tshifted,I_t,label=r'Intensity (I)',color="black",linewidth=3)
        #plt.plot(t,T_t,label=r'Total Polarization ($\sqrt{Q^2 + U^2 + V^2}/I$)')
        ax1.step(tshifted,L_t,label=r'Linear Polarization (L)',color="blue",linewidth=2.5)
        ax1.step(tshifted,C_t,label=r'Circular Polarization (V)',color="orange",linewidth=2)

    if plot_weights and weighted:
        ax1.step(tshifted,I_t_weights*np.max(I_t)/np.max(I_t_weights),label=r'Weights',color="red",linewidth=4)
    ax1.legend(loc="upper right")
    ax1.set_ylabel(r'S/N')
    ax1.set_xlim(((timestart - np.argmax(I_t))*(t_samp*1e6)*n_t)/1000-wind,((timestop - np.argmax(I_t))*(t_samp*1e6)*n_t)/1000 +wind)

    ax2.set_xlabel(r'Time ($m s$)')
    ax2.set_ylabel(r'Frequency (MHz)')
    ax2.set_xlim(timestart- wind*1000/((t_samp*1e6)*n_t),timestop+ wind*1000/((t_samp*1e6)*n_t))
    #ax2.tick_params(axis='x', labelrotation = 45)

    if showspectrum:
        color1=ax3.step(I_f,freq_test[0],label=r'Total Polarization ($\sqrt{Q^2 + U^2 + V^2}/I$)',color="black",linewidth=3)
        color2 = ax3.step(L_f,freq_test[0],label=r'Linear Polarization ($\sqrt{Q^2 + U^2}/I$)',color="blue",linewidth=2.5)
        color3=ax3.step(C_f,freq_test[0],label=r'Circular Polarization ($V/I$)',color="orange",linewidth=2)
        ax3.set_ylim(np.min(freq_test[0]),np.max(freq_test[0]))
        ax3.set_xlabel(r'S/N')
        #ax3.tick_params(axis='x', labelrotation = 45)

    ticklabelsx = np.array(ax1.get_xticks(),dtype=int)[1:-1]
    ticksx = np.array(np.argmax(I_t) + np.array(ticklabelsx)/(32.7*n_t/1000),dtype=float)
    print(ticklabelsx)
    print(ticksx)

    if showspectrum:
        ticklabelsy =np.array(ax3.get_yticks(),dtype=int)[1:-1]
        print(ticklabelsy)
        ticksy= np.array(((ticklabelsy - np.max(freq_test[0]))/(freq_test[0][1]-freq_test[0][0])),dtype=int)#(np.max(freq_test[0]) -np.min(freq_test[0]))
        print(ticksy)
        

    ax2.imshow(I,aspect="auto",vmin=0,vmax=np.percentile(I,99),interpolation="nearest")
    ax2.set_xticks(ticksx,np.around(ticklabelsx,1))
    if not showspectrum:
        ticksy = ax2.get_yticks()[2:-1]
        ticklabelsy = np.array(np.around(freq_test[0][np.array(ticksy,dtype=int)],0),dtype=int)
    ax2.set_yticks(ticksy,np.around(ticklabelsy,1))


    fig.tight_layout()
    if add_title:
        if nickname == "220912A" and ids== "221018aaaj":
            fig.text(0.5,1, "FRB20220912A Burst 1",ha ='center')
        elif nickname == "220912A":
            fig.text(0.5,1, "FRB20220912A Burst 2",ha ='center')
        else:
            fig.text(0.5, 1, "FRB20" + ids[:6], ha='center')
    ax1.xaxis.set_major_locator(ticker.NullLocator())
    if showspectrum:
        ax3.yaxis.set_major_locator(ticker.NullLocator())
        ax6.yaxis.set_major_locator(ticker.NullLocator())
    fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(wspace=0)
    plt.savefig(datadir + ids + "_" + nickname + "_pol_summary_plot"+ suffix + ".pdf")
    if show:
        plt.show()
    plt.close(fig)

    return I_f,Q_f,U_f,L_f,V_f



#compute the unbiased linear polarization
def L_unbias(It,Qt,Ut,n_off,unbias_factor=1):

    L_tcal_trm = np.sqrt(Qt**2 + Ut**2)
    L_tcal_trm[L_tcal_trm**2 <= (unbias_factor*np.std(It[:n_off]))**2] = np.std(It[:n_off])
    L_tcal_trm = np.sqrt(L_tcal_trm**2 - np.std(It[:n_off])**2)
    return L_tcal_trm
