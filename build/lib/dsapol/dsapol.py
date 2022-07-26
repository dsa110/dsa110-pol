"""
Library of functions for polarization analysis of 
DSA-110 data. 
"""


from scipy.signal import find_peaks
from scipy.signal import peak_widths
import copy
import numpy as np


import numpy.ma as ma
from sigpyproc import FilReader
from sigpyproc.Filterbank import FilterbankBlock
from sigpyproc.Header import Header
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
DEFAULT_DATADIR = "/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/testimgs/"#"/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03/testimgs/" #Users can find datadirectories for all processed FRBs here; to access set datadir = DEFAULT_WDIR + trigname_label

from astropy.time import Time
from astropy.coordinates import EarthLocation
import astropy.units as u



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
def create_stokes_arr(sdir, nsamp=10240,verbose=False):
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
    for ii in range(4):
        print("Reading stokes param..." + str(ii),end="")
        fn = '%s_%d.fil'%(sdir,ii)
        d = read_fil_data_dsa(fn, start=0, stop=nsamp)[0]
        #if d == 0:
            #print("Failed to read file: " + fn + ", returning 0")
         
        stokes_arr.append(d)
        print("Done!")
    stokes_arr = np.concatenate(stokes_arr).reshape(4, -1, nsamp)
    return stokes_arr

#Creates freq, time axes
def create_freq_time(sdir,nsamp=10240):#1500):
    """
    This function creates frequency and time axes from stokes filterbank headers

    Inputs: sdir --> str,directory containing Stokes fil files
            nsamp --> int,number of time samples to read in
    Outputs: (freq,dt) --> frequency and time axes respectively

    """
    freq=[]
    dt = []
    for ii in range(4):
        fn = '%s_%d.fil'%(sdir,ii)
        d = read_fil_data_dsa(fn, start=0, stop=nsamp)
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
def get_stokes_2D(datadir,fn_prefix,nsamps,n_t=1,n_f=1,n_off=3000,sub_offpulse_mean=True,fixchans=True):
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
    sarr = create_stokes_arr(sdir, nsamp=nsamps)
    freq,dt = create_freq_time(sdir, nsamp=nsamps)
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
    I,Q,U,V = avg_time(sarr[0],n_t),avg_time(sarr[1],n_t),avg_time(sarr[2],n_t),avg_time(sarr[3],n_t)
    I,Q,U,V = avg_freq(I,n_f),avg_freq(Q,n_f),avg_freq(U,n_f),avg_freq(V,n_f)

    if fixchans == True:
        bad_chans = find_bad_channels(I)
        (I,Q,U,V) = fix_bad_channels(I,Q,U,V,bad_chans)
        print("Bad Channels: " + str(bad_chans))


    #Subtract off-pulse mean
    if sub_offpulse_mean:
        offpulse_I = np.mean(I[:,:n_off],axis=1,keepdims=True) 
        offpulse_Q = np.mean(Q[:,:n_off],axis=1,keepdims=True)
        offpulse_U = np.mean(U[:,:n_off],axis=1,keepdims=True)
        offpulse_V = np.mean(V[:,:n_off],axis=1,keepdims=True)
    
        offpulse_I_std = np.std(I[:,:n_off],axis=1,keepdims=True)
        offpulse_Q_std = np.std(Q[:,:n_off],axis=1,keepdims=True)
        offpulse_U_std = np.std(U[:,:n_off],axis=1,keepdims=True)
        offpulse_V_std = np.std(V[:,:n_off],axis=1,keepdims=True)
    
        I = (I - offpulse_I)/offpulse_I_std
        Q = (Q - offpulse_Q)/offpulse_Q_std
        U = (U - offpulse_U)/offpulse_U_std
        V = (V - offpulse_V)/offpulse_V_std
   

    #calculate frequency and wavelength arrays (separate array for each stokes parameter, but should be the same)
    c = (3e8) #m/s
    freq_arr = []
    wav_arr = []
    for i in range(4):
        freq_arr.append(freq[i].reshape(-1,n_f).mean(1))
        wav_arr.append(list(c/(np.array(freq_arr[i])*(1e6))))

    
    return (I,Q,U,V,fobj,timeaxis,freq_arr,wav_arr)


#Get frequency averaged stokes params vs time; note run get_stokes_2D first to get 2D I Q U V arrays
def get_stokes_vs_time(I,Q,U,V,width_native,t_samp,n_t,n_off=3000,plot=False,datadir=DEFAULT_DATADIR,label='',calstr='',ext=ext,show=False,normalize=True,buff=0,weighted=False,n_t_weight=1,timeaxis=None,fobj=None,sf_window_weights=45,window=10):
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

    #optimal weighting
#    if weighted:
#        I,Q,U,V = weight_spectra_2D(I,Q,U,V,width_native,t_samp,n_f,n_t,freq_test,timeaxis,fobj,n_off=n_off,buff=buff,n_t_weight=n_t_weight,sf_window_weights=sf_window_weights)

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


        #I = I*I_t_weights
        #Q = Q*I_t_weights
        #U = U*I_t_weights
        #V = V*I_t_weights
        """

    if plot:
        f=plt.figure(figsize=(12,6))
        plt.plot(I_t,label="I")
        plt.plot(Q_t,label="Q")
        plt.plot(U_t,label="U")
        plt.plot(V_t,label="V")
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
    (peak,timestart,timestop) = find_peak(I,width_native,t_samp,n_t,buff=buff)
    #Bin to optimal time binning
    Ib,Qb,Ub,Vb = avg_time(I,n_t_weight),avg_time(Q,n_t_weight),avg_time(U,n_t_weight),avg_time(V,n_t_weight)
    if fobj.header.nsamples == 5120:
        timeaxisb = np.linspace(0,fobj.header.tsamp*(fobj.header.nsamples),(fobj.header.nsamples)//(n_t*n_t_weight))
    else:
        timeaxisb = np.linspace(0,fobj.header.tsamp*(fobj.header.nsamples//4),(fobj.header.nsamples//4)//(n_t*n_t_weight))

    #Calculate weights
    (I_t,Q_t,U_t,V_t) = get_stokes_vs_time(Ib,Qb,Ub,Vb,width_native,t_samp,n_t*n_t_weight,n_off//n_t_weight,normalize=True,buff=buff) #note normalization always used to get SNR
    
    #Interpolate to original sample rate
    fI = interp1d(timeaxisb,I_t,kind="linear",fill_value="extrapolate")
    fQ = interp1d(timeaxisb,Q_t,kind="linear",fill_value="extrapolate")
    fU = interp1d(timeaxisb,U_t,kind="linear",fill_value="extrapolate")
    fV = interp1d(timeaxisb,V_t,kind="linear",fill_value="extrapolate")


    #divide by sum to normalize to 1
    I_t_weight = fI(timeaxis)/np.sum(fI(timeaxis))
    Q_t_weight = fQ(timeaxis)/np.sum(fQ(timeaxis))
    U_t_weight = fU(timeaxis)/np.sum(fU(timeaxis))
    V_t_weight = fV(timeaxis)/np.sum(fV(timeaxis))

    #savgol filter
    I_t_weight = sf(I_t_weight,sf_window_weights,3)
    Q_t_weight = sf(Q_t_weight,sf_window_weights,3)
    U_t_weight = sf(U_t_weight,sf_window_weights,3)
    V_t_weight = sf(V_t_weight,sf_window_weights,3)

    #take absolute value (negative weights meaningless)
    I_t_weight = np.abs(I_t_weight)

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
def get_stokes_vs_freq(I,Q,U,V,width_native,t_samp,n_f,n_t,freq_test,n_off=3000,plot=False,datadir=DEFAULT_DATADIR,label='',calstr='',ext=ext,show=False,normalize=False,buff=0,weighted=False,n_t_weight=1,timeaxis=None,fobj=None,sf_window_weights=45):
    """
    This function calculates the time averaged (over width of pulse) frequency spectra for each stokes
    parameter. Outputs plots if specified in region around the pulse.

    Inputs: I,Q,U,V --> 2D arrays, dynamic spectra of I,Q,U,V generated with get_stokes_2D()
            width_native --> int, width in samples of pulse in native sampling rate; equivalent to ibox parameter
            t_samp --> float, sampling time
            n_f --> int, number of frequency samples to bin by (average over)
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
            weigthed --> bool, if True, obtains optimal spectrum by weighting by frequency averaged SNR
    Outputs: (I_f,Q_f,U_f,V_f) --> 1D frequency spectra of each stokes parameter, IQUV respectively


    """
    #use full timestream for calibrators
    if width_native == -1:
        timestart = 0
        timestop = I.shape[1]
    else:
        peak,timestart,timestop = find_peak(I,width_native,t_samp,n_t,buff=buff)

    I_copy = copy.deepcopy(I)
    Q_copy = copy.deepcopy(Q)
    U_copy = copy.deepcopy(U)
    V_copy = copy.deepcopy(V)
    
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
        I_t_weights=get_weights(I,Q,U,V,width_native,t_samp,n_f,n_t,freq_test,timeaxis,fobj,n_off=n_off,buff=buff,n_t_weight=n_t_weight,sf_window_weights=sf_window_weights)
        I_t_weights_2D = np.array([I_t_weights]*I.shape[0])        

        I = I*I_t_weights_2D
        Q = Q*I_t_weights_2D
        U = U*I_t_weights_2D
        V = V*I_t_weights_2D
    
        if normalize:
            """
            I_f = (I[:,timestart:timestop].sum(1) - I_copy[:,:n_off].mean(1))/np.std(np.mean(I_copy[:,:n_off],axis=0))#I[:,:n_off].std(1)
            Q_f = (Q[:,timestart:timestop].sum(1) - Q_copy[:,:n_off].mean(1))/np.std(np.mean(Q_copy[:,:n_off],axis=0))#Q[:,:n_off].std(1)
            U_f = (U[:,timestart:timestop].sum(1) - U_copy[:,:n_off].mean(1))/np.std(np.mean(U_copy[:,:n_off],axis=0))#U[:,:n_off].std(1)
            V_f = (V[:,timestart:timestop].sum(1) - V_copy[:,:n_off].mean(1))/np.std(np.mean(V_copy[:,:n_off],axis=0))#V[:,:n_off].std(1)
            """
            I_f = (I[:,timestart:timestop].sum(1) - np.mean(I_copy[:,:n_off].mean(0)))/np.std(np.mean(I_copy[:,:n_off],axis=0))#I[:,:n_off].std(1)
            Q_f = (Q[:,timestart:timestop].sum(1) - np.mean(Q_copy[:,:n_off].mean(0)))/np.std(np.mean(Q_copy[:,:n_off],axis=0))#Q[:,:n_off].std(1)
            U_f = (U[:,timestart:timestop].sum(1) - np.mean(U_copy[:,:n_off].mean(0)))/np.std(np.mean(U_copy[:,:n_off],axis=0))#U[:,:n_off].std(1)
            V_f = (V[:,timestart:timestop].sum(1) - np.mean(V_copy[:,:n_off].mean(0)))/np.std(np.mean(V_copy[:,:n_off],axis=0))#V[:,:n_off].std(1)


        else:
            I_f = I[:,timestart:timestop].sum(1)
            Q_f = Q[:,timestart:timestop].sum(1)
            U_f = U[:,timestart:timestop].sum(1)
            V_f = V[:,timestart:timestop].sum(1)
    
    else:
        if normalize:
            I_f = (I[:,timestart:timestop].mean(1) - np.mean(I_copy[:,:n_off].mean(0)))/np.std(np.mean(I_copy[:,:n_off],axis=0))#I[:,:n_off].std(1)
            Q_f = (Q[:,timestart:timestop].mean(1) - np.mean(Q_copy[:,:n_off].mean(0)))/np.std(np.mean(Q_copy[:,:n_off],axis=0))#Q[:,:n_off].std(1)
            U_f = (U[:,timestart:timestop].mean(1) - np.mean(U_copy[:,:n_off].mean(0)))/np.std(np.mean(U_copy[:,:n_off],axis=0))#U[:,:n_off].std(1)
            V_f = (V[:,timestart:timestop].mean(1) - np.mean(V_copy[:,:n_off].mean(0)))/np.std(np.mean(V_copy[:,:n_off],axis=0))#V[:,:n_off].std(1)
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
def get_pol_fraction(I,Q,U,V,width_native,t_samp,n_t,n_f,freq_test,n_off=3000,plot=False,datadir=DEFAULT_DATADIR,label='',calstr='',ext=ext,pre_calc_tf=False,show=False,normalize=True,buff=0,full=False,weighted=False,n_t_weight=1,timeaxis=None,fobj=None,sf_window_weights=45,multipeaks=False,height=0.03,window=30,unbias=True,input_weights=[]):
    """
    This function calculates and plots the polarization fraction averaged over both time and 
    frequency, and the average polarization fraction within the peak.
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
    Outputs: pol_f --> 1D array, frequency dependent polarization
             pol_t --> 1D array, time dependent polarization
             avg --> float, frequency and time averaged polarization fraction
    """
    if isinstance(buff, int):
        buff1 = buff
        buff2 = buff
    else:
        buff1 = buff[0]
        buff2 = buff[1]

    allowed_err = 0.1 #allowed error above 100 %
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

    if full:
        #total polarization
        pol = np.sqrt(Q**2 + U**2 + V**2)

        if unbias:
            pol_f = np.nanmean(pol,axis=1)
            pol_t = np.nanmean(pol,axis=0)
            pol_t[pol_t**2 <= np.std(I_t[:n_off])**2] = np.std(I_t[:n_off])
            pol_t = np.sqrt(pol_t**2 - np.std(I_t[:n_off])**2)
            pol_f[pol_f**2 <= np.std(I_t[:n_off])**2] = np.std(I_t[:n_off])
            pol_f = np.sqrt(pol_f**2 - np.std(I_t[:n_off])**2)
            pol_f = pol_f/I_f
            pol_t = pol_t/I_t
        else:
            pol_f = np.nanmean(pol,axis=1)/I_f
            pol_t = np.nanmean(pol,axis=0)/I_t
        
        #linear polarization
        L = np.sqrt(Q**2 + U**2)
        
        if unbias:
            L_f = np.nanmean(L,axis=1)
            L_t = np.nanmean(L,axis=0)
            L_t[L_t**2 <= np.std(I_t[:n_off])**2] = np.std(I_t[:n_off])
            L_t = np.sqrt(L_t**2 - np.std(I_t[:n_off])**2)
            L_f[L_f**2 <= np.std(I_t[:n_off])**2] = np.std(I_t[:n_off])
            L_f = np.sqrt(L_f**2 - np.std(I_t[:n_off])**2)
            L_f = L_f/I_f
            L_t = L_t/I_t
        else:
            L_f = np.nanmean(L,axis=1)/I_f
            L_t = np.nanmean(L,axis=0)/I_t
        
        #circular polarization
        C_f = np.nanmean(V,axis=1)/I_f
        C_t = np.nanmean(V,axis=0)/I_t
    else:
        #print(I_f)
        #total polarization
        if unbias:
            pol_f = np.sqrt((np.array(Q_f)**2 + np.array(U_f)**2 + np.array(V_f)**2))
            pol_t = np.sqrt((np.array(Q_t)**2 + np.array(U_t)**2 + np.array(V_t)**2))

            pol_t[pol_t**2 <= np.std(I_t[:n_off])**2] = np.std(I_t[:n_off])
            pol_t = np.sqrt(pol_t**2 - np.std(I_t[:n_off])**2)
            pol_f[pol_f**2 <= np.std(I_t[:n_off])**2] = np.std(I_t[:n_off])
            pol_f = np.sqrt(pol_f**2 - np.std(I_t[:n_off])**2)
            pol_f = pol_f/I_f
            pol_t = pol_t/I_t
        else:
            pol_f = np.sqrt((np.array(Q_f)**2 + np.array(U_f)**2 + np.array(V_f)**2)/(np.array(I_f)**2))
            pol_t = np.sqrt((np.array(Q_t)**2 + np.array(U_t)**2 + np.array(V_t)**2)/(np.array(I_t)**2))
        
        #linear polarization
        if unbias:
            L_f = np.sqrt((np.array(Q_f)**2 + np.array(U_f)**2))
            L_t = np.sqrt((np.array(Q_t)**2 + np.array(U_t)**2))

            L_t[L_t**2 <= np.std(I_t[:n_off])**2] = np.std(I_t[:n_off])
            L_t = np.sqrt(L_t**2 - np.std(I_t[:n_off])**2)
            L_f[L_f**2 <= np.std(I_t[:n_off])**2] = np.std(I_t[:n_off])
            L_f = np.sqrt(L_f**2 - np.std(I_t[:n_off])**2)
            L_f = L_f/I_f
            L_t = L_t/I_t
        else:
            L_f = np.sqrt((np.array(Q_f)**2 + np.array(U_f)**2)/(np.array(I_f)**2))
            L_t = np.sqrt((np.array(Q_t)**2 + np.array(U_t)**2)/(np.array(I_t)**2))

        #circular polarization
        C_f = V_f/I_f
        C_t = V_t/I_t




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

    allowed_err = 0.1 #allowed error above 100 %
    #avg_frac = (np.mean(pol_t[timestart:timestop][pol_t[timestart:timestop]<1]))
    #avg_frac = (np.mean(pol_t[timestart:timestop]))
    
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
        intL = int(intL)
        intR = int(intR)

        #average,error
        pol_t_cut1 = ((pol_t)[intL:intR])
        pol_t_cut = pol_t_cut1[np.abs(pol_t_cut1) < 1+allowed_err]
        I_t_weights_pol_cut = I_t_weights[intL:intR]
        I_t_weights_pol_cut = I_t_weights_pol_cut[np.abs(pol_t_cut1) < 1+allowed_err]
        avg_frac = np.nansum(pol_t_cut*I_t_weights_pol_cut)/np.nansum(I_t_weights_pol_cut)
        sigma_frac = np.nansum(I_t_weights_pol_cut*np.sqrt((pol_t_cut - avg_frac)**2))/np.nansum(I_t_weights_pol_cut)

        L_t_cut1 = ((L_t)[intL:intR])
        L_t_cut = L_t_cut1[np.abs(pol_t_cut1) < 1+allowed_err]
        I_t_weights_L_cut = I_t_weights[intL:intR]
        I_t_weights_L_cut = I_t_weights_L_cut[np.abs(pol_t_cut1) < 1+allowed_err]
        avg_L = np.nansum(L_t_cut*I_t_weights_L_cut)/np.nansum(I_t_weights_L_cut)
        sigma_L = np.nansum(I_t_weights_L_cut*np.sqrt((L_t_cut - avg_L)**2))/np.nansum(I_t_weights_L_cut)


        C_t_cut1 = (np.abs(C_t)[intL:intR])
        C_t_cut = C_t_cut1[np.abs(pol_t_cut1) < 1+allowed_err]
        I_t_weights_C_cut = I_t_weights[intL:intR]
        I_t_weights_C_cut = I_t_weights_C_cut[np.abs(pol_t_cut1) < 1+allowed_err]
        avg_C = np.nansum(C_t_cut*I_t_weights_C_cut)/np.nansum(I_t_weights_C_cut)
        sigma_C = np.nansum(I_t_weights_C_cut*np.sqrt((C_t_cut - avg_C)**2))/np.nansum(I_t_weights_C_cut)

        """
        avg_frac = np.nansum((pol_t*I_t_weights)[intL:intR])/np.sum(I_t_weights[intL:intR])
        avg_L = np.nansum((L_t*I_t_weights)[intL:intR])/np.sum(I_t_weights[intL:intR])
        avg_C = np.nansum((C_t*I_t_weights)[intL:intR])/np.sum(I_t_weights[intL:intR])

        #error
        sigma_frac = np.nansum( (np.sqrt((pol_t-avg_frac)**2)*I_t_weights)[intL:intR])/np.sum(I_t_weights[intL:intR])
        sigma_L = np.nansum( (np.sqrt((L_t-avg_L)**2)*I_t_weights)[intL:intR])/np.sum(I_t_weights[intL:intR])
        sigma_C = np.nansum( (np.sqrt((C_t-avg_C)**2)*I_t_weights)[intL:intR])/np.sum(I_t_weights[intL:intR])
        """
        #snr
        I_w_t_filt_roll = np.roll(I_t_weights,-np.argmax(I_t_weights))
        I_t_roll = np.roll(I_t,-np.argmax(I_t))
        Q_t_roll = np.roll(Q_t,-np.argmax(Q_t))
        U_t_roll = np.roll(U_t,-np.argmax(U_t))
        V_t_roll = np.roll(V_t,-np.argmax(V_t))

        I_t_c = convolve(I_w_t_filt_roll,I_t_roll,mode="same")
        I_t_c=np.roll(I_t_c,-np.argmax(I_t_c)+np.argmax(I_t))


        Q_t_c = convolve(I_w_t_filt_roll,Q_t_roll,mode="same")
        Q_t_c=np.roll(Q_t_c,-np.argmax(Q_t_c)+np.argmax(Q_t))


        U_t_c = convolve(I_w_t_filt_roll,U_t_roll,mode="same")
        U_t_c=np.roll(U_t_c,-np.argmax(U_t_c)+np.argmax(U_t))


        V_t_c = convolve(I_w_t_filt_roll,V_t_roll,mode="same")
        V_t_c=np.roll(V_t_c,-np.argmax(V_t_c)+np.argmax(V_t))

        pol_t_c = np.sqrt(Q_t_c**2 + U_t_c**2 + V_t_c**2)
        L_t_c = np.sqrt(Q_t_c**2 + U_t_c**2)
        C_t_c = np.sqrt(V_t_c**2)

        noise_pol = np.std(np.concatenate([pol_t_c[:np.argmax(I_t_c)-int(buff1)],pol_t_c[np.argmax(I_t_c)+1+int(buff2):]]))
        noise_L = np.std(np.concatenate([L_t_c[:np.argmax(I_t_c)-int(buff1)],L_t_c[np.argmax(I_t_c)+1+int(buff2):]]))
        noise_C = np.std(np.concatenate([C_t_c[:np.argmax(I_t_c)-int(buff1)],C_t_c[np.argmax(I_t_c)+1+int(buff2):]]))

        snr_frac = np.max(pol_t_c)/noise_pol
        snr_L = np.max(L_t_c)/noise_L
        snr_C = np.max(C_t_c)/noise_C

    else:
    

        avg_frac = np.nanmean((pol_t[timestart:timestop])[np.abs(pol_t)[timestart:timestop]<=1+allowed_err])
        avg_L = np.nanmean((L_t[timestart:timestop])[np.abs(L_t)[timestart:timestop]<=1+allowed_err])
        avg_C = np.nanmean((np.abs(C_t)[timestart:timestop])[np.abs(C_t)[timestart:timestop]<=1+allowed_err])

        #RMS error
        sigma_frac = np.nanstd((pol_t[timestart:timestop])[(pol_t)[timestart:timestop]<=1+allowed_err])
        sigma_L = np.nanstd((L_t[timestart:timestop])[np.abs(L_t)[timestart:timestop]<=1+allowed_err])
        sigma_C = np.nanstd((np.abs(C_t)[timestart:timestop])[np.abs(C_t)[timestart:timestop]<=1+allowed_err])

        #SNR
        sig0 = np.nanmean(pol_t[timestart:timestop])
        pol_t_cut1 = pol_t[timestart%(timestop-timestart):]
        pol_t_cut = pol_t_cut1[:(len(pol_t_cut1)-(len(pol_t_cut1)%(timestop-timestart)))]
        pol_t_binned = pol_t_cut.reshape(len(pol_t_cut)//(timestop-timestart),timestop-timestart).mean(1)
        sigbin = np.argmax(pol_t_binned)
        noise = (np.nanstd(np.concatenate([pol_t_cut[:sigbin],pol_t_cut[sigbin+1:]])))
        snr_frac = sig0/noise

        sig0 = np.nanmean(L_t[timestart:timestop])
        L_t_cut1 = L_t[timestart%(timestop-timestart):]
        L_t_cut = L_t_cut1[:(len(L_t_cut1)-(len(L_t_cut1)%(timestop-timestart)))]
        L_t_binned = L_t_cut.reshape(len(L_t_cut)//(timestop-timestart),timestop-timestart).mean(1)
        sigbin = np.argmax(L_t_binned)
        noise = (np.nanstd(np.concatenate([L_t_cut[:sigbin],L_t_cut[sigbin+1:]])))
        snr_L = sig0/noise

        sig0 = np.nanmean(C_t[timestart:timestop])
        C_t_cut1 = C_t[timestart%(timestop-timestart):]
        C_t_cut = C_t_cut1[:(len(C_t_cut1)-(len(C_t_cut1)%(timestop-timestart)))]
        C_t_binned = C_t_cut.reshape(len(C_t_cut)//(timestop-timestart),timestop-timestart).mean(1)
        sigbin = np.argmax(C_t_binned)
        noise = (np.nanstd(np.concatenate([C_t_cut[:sigbin],C_t_cut[sigbin+1:]])))
        snr_C = sig0/noise




    #snr_frac = np.nanmean(pol_t[timestart:timestop])/np.nanstd(pol_t[:n_off])
    #snr_L = np.nanmean(L_t[timestart:timestop])/np.nanstd(L_t[:n_off])
    #snr_C = np.nanmean(C_t[timestart:timestop])/np.nanstd(C_t[:n_off])

    return [(pol_f,pol_t,avg_frac,sigma_frac,snr_frac),(L_f,L_t,avg_L,sigma_L,snr_L),(C_f,C_t,avg_C,sigma_C,snr_C)]

#Calculate polarization angle vs frequency and time from 2D I Q U V arrays
def get_pol_angle(I,Q,U,V,width_native,t_samp,n_t,n_f,freq_test,n_off=3000,plot=False,datadir=DEFAULT_DATADIR,label='',calstr='',ext=ext,pre_calc_tf=False,show=False,normalize=True,buff=0,weighted=False,n_t_weight=1,timeaxis=None,fobj=None,sf_window_weights=45,multipeaks=False,height=0.03,window=30,input_weights=[]):
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

    if plot:
        f=plt.figure(figsize=(12,6))
        plt.plot(freq_test[0],PA_f)
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
        plt.plot(np.arange(timestart,timestop),PA_t[timestart:timestop])
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
        PA_t_cut1 = ((PA_t%(2*np.pi))[intL:intR])
        PA_t_cut = PA_t_cut1[np.abs(PA_t_cut1) < (2*np.pi)]
        I_t_weights_pol_cut = I_t_weights[intL:intR]
        I_t_weights_pol_cut = I_t_weights_pol_cut[np.abs(PA_t_cut1) < (2*np.pi)]
        avg_PA = np.nansum(PA_t_cut*I_t_weights_pol_cut)/np.nansum(I_t_weights_pol_cut)
        sigma_PA = np.nansum(I_t_weights_pol_cut*np.sqrt((PA_t_cut - avg_PA)**2))/np.nansum(I_t_weights_pol_cut)

    else:
        #avg_PA = np.mean(PA_t[timestart:timestop][PA_t[timestart:timestop]<1])
        avg_PA = np.nanmean(PA_t[timestart:timestop])
        sigma_PA = np.nanstd(PA_t[timestart:timestop])
    return PA_f,PA_t,avg_PA,sigma_PA

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
    (Igainuc,Qgainuc,Ugainuc,Vgainuc,fobj,timeaxis,freq_test,wav_test) = get_stokes_2D(gain_dir,source_name + obs_name + "_dev",5,n_t=n_t,n_f=n_t_down,sub_offpulse_mean=False)

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
    (Igainuc,Qgainuc,Ugainuc,Vgainuc,fobj,timeaxis,freq_test_fullres,wav_test) = get_stokes_2D(gain_dir,source_name + obs_name + "_dev",5,n_t=n_t,n_f=n_f,sub_offpulse_mean=False)

    idx = np.isfinite(GX)
    f_GX= interp1d(freq_test[0][idx],GX[idx],kind="linear",fill_value="extrapolate")
    GX_fullres = f_GX(freq_test_fullres[0])


    idx = np.isfinite(GY)
    f_GY= interp1d(freq_test[0][idx],GY[idx],kind="linear",fill_value="extrapolate")
    GY_fullres = f_GY(freq_test_fullres[0])

    #polyfit
    popt = np.polyfit(freq_test_fullres[0],GY_fullres,deg=deg)
    print(len(popt))
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
        (I,Q,U,V,fobj,timeaxis,freq_test,wav_test) = get_stokes_2D(datadir,label,nsamps,n_t=n_t,n_f=n_f,n_off=-1,sub_offpulse_mean=False)
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
        (I,Q,U,V,fobj,timeaxis,freq_test,wav_test) = get_stokes_2D(datadir,label,nsamps,n_t=n_t,n_f=n_f,n_off=-1,sub_offpulse_mean=False)
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
    gxx = ratio*gyy_mag*np.exp(1j*(phase_diff-gyy_phase))
    gyy = gyy_mag*np.exp(1j*gyy_phase)*np.ones(np.shape(ratio))

    return [gxx,gyy]

#Takes data products for target and returns calibrated Stokes parameters
def calibrate(xx_I_obs,yy_Q_obs,xy_U_obs,yx_V_obs,calmatrix,stokes=True):
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
    
    I_true = 0.5*((((np.abs(1/gxx_cal)**2))*(I_obs + Q_obs)) + (((np.abs(1/gyy_cal)**2))*(I_obs - Q_obs)))
    Q_true = 0.5*((((np.abs(1/gxx_cal)**2))*(I_obs + Q_obs)) - (((np.abs(1/gyy_cal)**2))*(I_obs - Q_obs)))
    
    xy_obs = U_obs + 1j*V_obs
    xy_cal = xy_obs / (gxx_cal*np.conj(gyy_cal))#* np.exp(-1j * (np.angle(gxx_cal)-np.angle(gyy_cal)))
    U_true, V_true = xy_cal.real, xy_cal.imag
    
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
def calibrate_angle(xx_I_obs,yy_Q_obs,xy_U_obs,yx_V_obs,fobj,ibeam,RA,DEC,beamsize_as=14,Lat=37.23,Lon=-118.2851,centerbeam=125,stokes=True):
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

    print("FRB RA: " + str(RA) + " degrees")
    print("FRB DEC: " + str(DEC) + " degrees")

    RA = RA*np.pi/180
    DEC = DEC*np.pi/180

    #use hour angle for beam nearest center to estimate actual elevation
    observing_time = Time(fobj.header.tstart, format='mjd', location=observing_location)#Time(fobj.header.tstart, format='mjd', location=observing_location)
    GST = Time(fobj.header.tstart, format='mjd').to_datetime().hour + Time(fobj.header.tstart, format='mjd').to_datetime().minute/60 + Time(fobj.header.tstart, format='mjd').to_datetime().second/3600#observing_time.sidereal_time('mean').hour
    LST = (GST + Lon*24/360)%24
    print(observing_time.isot)
    print(LST)

    HA = (LST*360/24)*np.pi/180 - RA #rad


    elev = np.arcsin(np.sin(Lat*np.pi/180)*np.sin(DEC) + np.cos(Lat*np.pi/180)*np.cos(DEC)*np.cos(HA))
    print("elevation est. : " + str(elev*180/np.pi) + " degrees")

    #apply correction to azimuth due to beam offset
    az = np.arcsin(-np.cos(DEC)*np.sin(HA)/np.cos(elev))
    az_corr = (ibeam - centerbeam)*synth_beamsize
    print("initial azimuth est. : " + str(az*180/np.pi) + " degrees")
    print("corrected azimuth est. : " + str((az-az_corr)*180/np.pi) + " degrees")

    ParA = np.arcsin(np.sin(HA)*np.cos(Lat*np.pi/180)/np.cos(elev)) + chi_ant
    print("initial parallactic angle est : " + str(ParA*180/np.pi) + " degrees")

    #ParA = np.arcsin(-np.sin(az-az_corr)*np.cos(Lat*np.pi/180)/np.cos(DEC)) + chi_ant
    arg = -np.sin(az-az_corr)*np.cos(Lat*np.pi/180)/np.cos(DEC)
    if arg > 1:
        arg = 1
    elif arg < -1:
        arg = -1

    ParA = np.arcsin(arg) + chi_ant
    
    print("corrected parallactic angle est : " + str(ParA*180/np.pi) + " degrees")

    #calibrate
    Qcal = Q_obs*np.cos(2*ParA) - U_obs*np.sin(2*ParA)
    Ucal = Q_obs*np.sin(2*ParA) + U_obs*np.cos(2*ParA)

    return I_obs,Qcal,Ucal,V_obs,ParA

    




#Estimate RM by maximizing SNR over given trial RM; wav is wavelength array
def faradaycal(I,Q,U,V,freq_test,trial_RM,trial_phi,plot=False,datadir=DEFAULT_DATADIR,calstr="",label="",n_f=1,n_t=1,show=False,fit_window=100,err=True): 
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
        
       
        if len(trial_RM) > 1 and len(trial_phi) >1:
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

    return (trial_RM[max_RM_idx],trial_phi[max_phi_idx],SNRs[:,0],RM_err)

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
def faradaycal_SNR(I,Q,U,V,freq_test,trial_RM,trial_phi,width_native,t_samp,plot=False,datadir=DEFAULT_DATADIR,calstr="",label="",n_f=1,n_t=1,show=False,err=True,buff=0,weighted=False,n_t_weight=1,timeaxis=None,fobj=None,sf_window_weights=45,n_off=3000,full=False):
    #Get wavelength axis
    c = (3e8) #m/s
    wav = c/(freq_test[0]*(1e6))#wav_test[0]
    #wavall = np.array([wav]*np.shape(I)[1]).transpose()
    #print(np.any(np.isnan(wavall)))
    #use full timestream for calibrators
    if width_native == -1:
        timestart = 0
        timestop = I.shape[1]
    else:
        peak,timestart,timestop = find_peak(I,width_native,t_samp,n_t,buff=buff)

    if weighted:
        #get weights
        I_t_weights=get_weights(I,Q,U,V,width_native,t_samp,n_f,n_t,freq_test,timeaxis,fobj,n_off=n_off,buff=buff,n_t_weight=n_t_weight,sf_window_weights=sf_window_weights)
        I_t_weights_2D = np.array([I_t_weights]*I.shape[0])


    #Calculate polarization
    P = Q + 1j*U
    #print(P)
    #SNR matrix
    SNRs = np.zeros((len(trial_RM),len(trial_phi)))

    #get snr at RM =0  to estimate significance
    
    L0_t = np.sqrt(np.mean(Q,axis=0)**2 + np.mean(U,axis=0)**2)

    if weighted:
        L0_t_w = L0_t*I_t_weights
        sig0 = np.sum(L0_t_w[timestart:timestop])
        L_trial_binned = convolve(L0_t,I_t_weights)
        sigbin = np.argmax(L_trial_binned)
        noise = np.std(np.concatenate([L_trial_binned[:sigbin],L_trial_binned[sigbin+1:]]))

    else:
        
        sig0 = np.mean(np.sqrt(np.mean(Q,axis=0)**2 + np.mean(U,axis=0)**2)[timestart:timestop])
        L_trial_cut1 = L0_t[timestart%(timestop-timestart):]
        L_trial_cut = L_trial_cut1[:(len(L_trial_cut1)-(len(L_trial_cut1)%(timestop-timestart)))]
        L_trial_binned = L_trial_cut.reshape(len(L_trial_cut)//(timestop-timestart),timestop-timestart).mean(1)
        sigbin = np.argmax(L_trial_binned)
        noise = (np.std(np.concatenate([L_trial_cut[:sigbin],L_trial_cut[sigbin+1:]])))
    print(noise)        
    snr0 = sig0/noise

    P_cut = P[:,timestart:timestop]
    wavall_cut = np.array([wav]*np.shape(P_cut)[1]).transpose()
    if weighted:
        I_t_weights_cut = I_t_weights[timestart:timestop]
    print(P_cut.shape,wavall_cut.shape)


    #if full return full time vs RM matrix to estimate RM variation over the pulse width
    if full:
        SNRs_full = np.zeros((len(trial_RM),timestop-timestart))

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
                sig = np.sum(L_trial_t*I_t_weights_cut)
            else:
                sig = np.mean(L_trial_t)
            """
            L_trial_offp = np.concatenate([L_trial_t[:timestart],L_trial_t[timestop:]])
            L_trial_offp = L_trial_offp[:len(L_trial_offp) - (len(L_trial_offp)%(timestop-timestart))]
            print(L_trial_offp.reshape(len(L_trial_offp)//(timestop-timestart),timestop-timestart))
            L_trial_binned_offp = np.mean(L_trial_offp.reshape(len(L_trial_offp)//(timestop-timestart),timestop-timestart))
            #print(L_trial_offp)
            #print(L_trial_binned_offp)
            #binned_off_pulse =  #dsapol.avg_time(np.concatenate([L_trial_t[:timestart],L_trial_t[timestop:]])[:4096],timestop-timestart)
            noise = np.sqrt(np.mean(L_trial_binned_offp**2))
            print(noise)
            #print(sig,noise)

            if i == 0:
                L_trial_cut1 = L_trial_t[timestart%(timestop-timestart):]
                L_trial_cut = L_trial_cut1[:(len(L_trial_cut1)-(len(L_trial_cut1)%(timestop-timestart)))]

                L_trial_binned = L_trial_cut.reshape(len(L_trial_cut)//(timestop-timestart),timestop-timestart).mean(1)
                sigbin = np.argmax(L_trial_binned)
                noise = (np.std(np.concatenate([L_trial_cut[:sigbin],L_trial_cut[sigbin+1:]])))
            else:
                noise = noise
            """
            SNRs[i,j] = sig/noise#np.abs(np.sum(P_trial))


    (max_RM_idx,max_phi_idx) = np.unravel_index(np.argmax(SNRs),np.shape(SNRs))
    max_RM_idx = np.argmax(SNRs)
    print(max_RM_idx)

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

    if full:
        return (trial_RM[max_RM_idx],trial_phi[max_phi_idx],SNRs[:,0],RMerr,upper,lower,significance,SNRs_full)
    else:
        return (trial_RM[max_RM_idx],trial_phi[max_phi_idx],SNRs[:,0],RMerr,upper,lower,significance)

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
        (RM,phi,SNRs,RMerr,upper,lower,significance) = faradaycal_SNR(I,Q,U,V,freq_test,trial_RM_zoom,trial_phi,width_native,t_samp,plot=plot,datadir=datadir,calstr=calstr,label=label,n_f=n_f,n_t=n_t,show=show,buff=buff)
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
def find_beam(file_suffix,shape=(16,7680,256),path='/home/ubuntu/sherman/scratch_weights_update_2022-06-03/',plot=False,show=False):
    #file_suffix = ""
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


#Plotting Functions
def FRB_quick_analysis(ids,nickname,ibeam,width_native,buff,RA,DEC,gain_dir,gain_source_name,gain_obs_name,gain_beam,phase_dir,phase_source_name,phase_obs_name,phase_beam,n_t,n_f,sfwindow=-1,nsamps_cal=5,suffix="_dev",padwidth=10,peakheight=2,n_t_down=32,beamsize_as=14,Lat=37.23,Lon=-118.2851,centerbeam=125,edgefreq=-1,breakfreq=-1,use_fit=True,weighted=False,n_t_weight=2,sf_window_weights=7,RMcal=True,trial_RM=np.linspace(-1e6,1e6,int(2e6)),trial_phi=[0],n_trial_RM_zoom=5000,zoom_window=1000):



    #get calibration parameters
    #GY
    GY,GY_fit,GY_fit_params=absgaincal(gain_dir,gain_source_name,gain_obs_name,n_t,1,nsamps_cal,deg,gain_beam,suffix=suffix,plot=plot,show=show,padwidth=padwidth,peakheight=peakheight,n_t_down=n_t_down,beamsize_as=beamsize_as,Lat=Lat,Lon=Lon,centerbeam=centerbeam)

    #gain,phase
    (tmp,tmp,tmp,tmp,tmp,tmp,freq_test,tmp) = get_stokes_2D(gain_dir,gain_source_name + gain_obs_name + suffix ,nsamps,n_t=n_t,n_f=1)

    ratio_2,ratio_params_2,ratio_sf_2 = gaincal_full(gain_dir,gain_source_name,gain_obs_names,n_t,1,nsamps,deg,suffix=suffix,average=True,plot=False,show=False,sfwindow=sfwindow,clean=True,padwidth=padwidth,peakheight=peakheight+1,n_t_down=n_t_down,edgefreq=edgefreq,breakfreq=breakfreq)
    phase_2,phase_params_2,phase_sf_2 = phasecal_full(phase_dir,phase_source_name,phase_obs_names,n_t,1,nsamps,deg,suffix=suffix,average=True,plot=False,show=False,sfwindow=sfwindow,clean=True,padwidth=padwidth,peakheight=peakheight+1,n_t_down=n_t_down,edgefreq=-1,breakfreq=-1)#,edgefreq=-1,breakfreq=-1)

    if len(ratio_params_2) ==2:
        ratio_fit1 = np.zeros(len(ratio_2))
        for i in range(len(ratio_params_2[0])):
            ratio_fit1 += ratio_params_2[0][i]*(freq_test[0]**(len(ratio_params_2[0])-i-1))
        ratio_fit2 = np.zeros(len(ratio_2))
        for i in range(len(ratio_params_2[1])):
            ratio_fit2 += ratio_params_2[1][i]*(freq_test[0]**(len(ratio_params_2[1])-i-1))
        breakidx = np.argmin(np.abs(freq_test[0]-breakfreq))#np.argmin(np.abs(ratio_fit1 - ratio_fit2))
        ratio_fit3 = np.zeros(len(ratio_fit1))
        ratio_fit3[breakidx:] = ratio_fit1[breakidx:]
        ratio_fit3[:breakidx] = ratio_fit2[:breakidx]
        ratio_fit = ratio_fit3
    else:
        ratio_fit1 = np.zeros(len(ratio_2))
        for i in range(len(ratio_params_2)):
            ratio_fit1 += ratio_params_2[i]*(freq_test[0]**(len(ratio_params_2)-i-1))
        ratio_fit = ratio_fit1

    phase_fit = np.zeros(len(phase_2))
    for i in range(len(phase_params_2)):
        phase_fit += phase_params_2[i]*(freq_test[0]**(len(phase_params_2)-i-1))
    phase_fit = np.nanmedian(phase_fit)*np.ones(len(phase_fit))
    

    if use_fit:
        ratio_use = ratio_fit
        phase_use = phase_fit
        GY_use = GY_fit
    elif sfwindow != -1:
        ratio_use = ratio_sf_2
        phase_use = phase_sf_2
        GY_use = GY_fit
    else:
        ratio_use = ratio_2
        phase_use = phase_2
        GY_use = GY

    (gxx,gyy) = get_calmatrix_from_ratio_phasediff(ratio_fit,phase_fit,gyy_mag=GY_USE)

    print("Modelling phase as constant: " + str(np.mean(np.angle(gxx))*180/np.pi) + " degrees")
    print(str(deg) + " degree fit for gain: " + str(ratio_params_2))

    #Read data
    datadir="/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/"+ids + "_" + nickname + "/"
    (I_fullres,Q_fullres,U_fullres,V_fullres,fobj,timeaxis,freq_test_fullres,wav_test) = get_stokes_2D(datadir,ids + "_dev",20480,n_t=n_t,n_f=1,n_off=int(12000//n_t),sub_offpulse_mean=True)
    
    #gain/phase and PA calibrate
    Ical_fullres,Qcal_fullres,Ucal_fullres,Vcal_fullres = calibrate(I_fullres,Q_fullres,U_fullres,V_fullres,(gxx,gyy),stokes=True)
    Ical_fullres,Qcal_fullres,Ucal_fullres,Vcal_fullres,ParA_fullres = calibrate_angle(Ical_fullres,Qcal_fullres,Ucal_fullres,Vcal_fullres,fobj,ibeam,RA,DEC)

    #downsample in frequency
    I = dsapol.avg_freq(I_fullres,n_f)
    Q = dsapol.avg_freq(Q_fullres,n_f)
    U = dsapol.avg_freq(U_fullres,n_f)
    V = dsapol.avg_freq(V_fullres,n_f)

    Ical = dsapol.avg_freq(Ical_fullres,n_f)
    Qcal = dsapol.avg_freq(Qcal_fullres,n_f)
    Ucal = dsapol.avg_freq(Ucal_fullres,n_f)
    Vcal = dsapol.avg_freq(Vcal_fullres,n_f)
    
    if freq_test_fullres[0].shape[0]%n_f != 0:
            for i in range(4):
                freq_test_fullres[i] = freq_test_fullres[i][freq_test_fullres[i].shape[0]%n_f:]

    freq_test =  [freq_test_fullres[0].reshape(len(freq_test_fullres[0])//n_f,n_f).mean(1)]*4
    
    #get window
    (peak,timestart,timestop) = find_peak(I,width_native,fobj.header.tsamp,n_t=n_t,peak_range=None,pre_calc_tf=False,buff=buff)
    t = 32.7*n_t*np.arange(0,I.shape[1])


    if plot:
        plot_spectra_2D(I,Q,U,V,width_native,fobj.header.tsamp,n_t,n_f,freq_test,lim=np.percentile(I,90),show=show,buff=int(64/n_t),weighted=False,window=int(64/n_t),datadir=datadir,ext=".png",label="_uncal_")
        plot_spectra_2D(Ical,Qcal,Ucal,Vcal,width_native,fobj.header.tsamp,n_t,n_f,freq_test,lim=np.percentile(I,90),show=show,buff=int(64/n_t),weighted=False,window=int(64/n_t),datadir=datadir,ext=".png",label="_cal_")


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
    (I_tcal,Q_tcal,U_tcal,V_tcal) = get_stokes_vs_time(Ical,Qcal,Ucal,Vcal,width_native,fobj.header.tsamp,n_t,n_off=int(12000//n_t),plot=plot,show=show,datadir=datadir,normalize=True,buff=buff,window=3)
    (I_fcal,Q_fcal,U_fcal,V_fcal) = get_stokes_vs_freq(Ical,Qcal,Ucal,Vcal,width_native,fobj.header.tsamp,n_f,n_t,freq_test,n_off=int(12000/n_t),plot=plot,show=show,normalize=True,buff=buff,weighted=weighted,n_t_weight=n_t_weight,timeaxis=timeaxis,fobj=fobj,sf_window_weights=sf_window_weights)

    #estimate polarization fractions before RM
    
    [(p_f,p_t,avg,sigma_frac,snr_frac),(L_f,L_t,avg_L,sigma_L,snr_L),(C_f,C_t,avg_C,sigma_C,snr_C)] = get_pol_fraction(Ical,Qcal,Ucal,Vcal,width_native,fobj.header.tsamp,n_t,n_f,freq_test,n_off=int(12000/n_t),plot=plot,show=show,datadir=datadir,normalize=True,buff=buff,full=False,weighted=weighted,n_t_weight=n_t_weight,timeaxis=timeaxis,fobj=fobj,sf_window_weights=sf_window_weights)
    PA_f,PA_t,avg_PA,PA_err = dsapol.get_pol_angle(Ical,Qcal,Ucal,Vcal,width_native,fobj.header.tsamp,n_t,n_f,freq_test,n_off=int(12000/n_t),plot=plot,show=show,datadir=datadir,normalize=True,buff=buff,weighted=weighted,n_t_weight=n_t_weights,timeaxis=timeaxis,fobj=fobj,sf_window_weights=sf_window_weights)
    print(r'Total Polarization: ${avg} \pm {err}$'.format(avg=avg,err=sigma_frac))
    print(r'Linear Polarization: ${avg} \pm {err}$'.format(avg=avg_L,err=sigma_L))
    print(r'Circular Polarization: ${avg} \pm {err}$'.format(avg=avg_C,err=sigma_C))
    print(r'Position Angle: ${avg}^\circ \pm {err}^\circ$'.format(avg=avg_PA*180/np.pi,err=PA_err*180/np.pi))

    #RM synthesis
    if RMcal:
        (I_fcal_fullres,Q_fcal_fullres,U_fcal_fullres,V_fcal_fullres) = get_stokes_vs_freq(Ical_fullres,Qcal_fullres,Ucal_fullres,Vcal_fullres,width_native,fobj.header.tsamp,n_f,n_t,freq_test_fullres,n_off=int(12000/n_t),plot=plot,show=show,normalize=True,buff=buff,weighted=weighted,n_t_weight=n_t_weight,timeaxis=timeaxis,fobj=fobj,sf_window_weights=sf_window_weights)
        RM1,phi1,RMsnrs1,RMerr1 = faradaycal(I_fcal_fullres,Q_fcal_fullres,U_fcal_fullres,V_fcal_fullres,freq_test_fullres,trial_RM,trial_phi,plot=plot,show=show,fit_window=fit_window,err=True)
        print(r'RM Synthesis estimate: ${avg} \pm {err} rad/m^2$'.format(avg=RM1,err=RMerr1))
        
        trial_RM2 = np.linspace(RM1-zoom_window,RM1+zoom_window,n_trial_RM_zoom)

        RM2,phi2,RMsnrs2,RMerr2,upp,low,sig,RMsnrs2_full = faradaycal_SNR(Ical_fullres,Qcal_fullres,Ucal_fullres,Vcal_fullres,freq_test_fullres,trial_RM2,trial_phi,width_native,fobj.header.tsamp,plot=plot,n_f=1,n_t=n_t,show=show,datadir=datadir,err=True,buff=buff,weighted=weighted,n_off=int(12000/n_t),n_t_weight=n_t_weight,timeaxis=timeaxis,fobj=fobj,sf_window_weights=sf_window_weights,full=True)
        print(r'RM S/N method estimate: ${avg} \pm {err} rad/m^2$, Linear S/N = {SNR}'.format(avg=RM2,err=RMerr2,SNR=np.max(RMsnrs2)))

        #calibrate and re-estimate polarization
        (IcalRM,QcalRM,UcalRM,VcalRM)=calibrate_RM(Ical,Qcal,Ucal,Vcal,RM2,phi,freq_test,stokes=True)
        [(p_f,p_t,avg,sigma_frac,snr_frac),(L_f,L_t,avg_L,sigma_L,snr_L),(C_f,C_t,avg_C,sigma_C,snr_C)] = get_pol_fraction(IcalRM,QcalRM,UcalRM,VcalRM,width_native,fobj.header.tsamp,n_t,n_f,freq_test,n_off=int(12000/n_t),plot=plot,show=show,datadir=datadir,normalize=True,buff=buff,full=False,weighted=weighted,n_t_weight=n_t_weight,timeaxis=timeaxis,fobj=fobj,sf_window_weights=sf_window_weights)
        PA_f,PA_t,avg_PA,PA_err = dsapol.get_pol_angle(IcalRM,QcalRM,UcalRM,VcalRM,width_native,fobj.header.tsamp,n_t,n_f,freq_test,n_off=int(12000/n_t),plot=plot,show=show,datadir=datadir,normalize=True,buff=buff,weighted=weighted,n_t_weight=n_t_weights,timeaxis=timeaxis,fobj=fobj,sf_window_weights=sf_window_weights)
        print(r'Total Polarization: ${avg} \pm {err}$'.format(avg=avg,err=sigma_frac))
        print(r'Linear Polarization: ${avg} \pm {err}$'.format(avg=avg_L,err=sigma_L))
        print(r'Circular Polarization: ${avg} \pm {err}$'.format(avg=avg_C,err=sigma_C))
        print(r'Position Angle: ${avg}^\circ \pm {err}^\circ$'.format(avg=avg_PA180/np.pi,err=PA_err*180/np.pi))

    return -1

#deprecated
def FRB_plot_all(datadir,prefix,nickname,nsamps,n_t,n_f,n_off,width_native,cal=False,gain_dir='.',gain_source_name="",gain_obs_names=[],phase_dir='.',phase_source_name="",phase_obs_names=[],deg=10,suffix="_dev",use_fit=False,RM_in=None,phi_in=0,get_RM=True,RM_cal=True,trial_RM=np.linspace(-10000,10000,10000),trial_phi=[0],n_trial_RM_zoom=-1,zoom_window=75,fit_window=100,cal_2D=True,sub_offpulse_mean=True,window=10,lim=500,buff=0,DM=-1,weighted=False,n_t_weight=1,use_sf=False,sfwindow=-1,extra='',clean=True,padwidth=10,peakheight=2,n_t_down=8,sf_window_weights=45):
    if n_trial_RM_zoom == -1:
        n_trial_RM_zoom = len(trial_RM)
    
    #Get filterbank header
    hdr_dict = read_fil_data_dsa(datadir + prefix + "_0.fil")[-1] 

    label = prefix + "_" + nickname
    cal1str = ''
    RM_calstr = ''
    cal_2Dstr = ''
    cal_fitstr = ''
    if cal:
        cal1str = 'calibrated'
        hdr_dict["calibrated"] = True
    if RM_cal:
        RM_calstr = 'RM_calibrated'
        hdr_dict["RM calibrated"] = True
    if cal_2D:
        cal_2Dstr = '2D_calibrated'
        hdr_dict["2D calibrated"] = True
    if not use_fit:
        cal_fitstr = 'nocalfit'
    
    #Modify header parameters
    hdr_dict["tsamp"] = hdr_dict["tsamp"]*n_t
    hdr_dict["nsamples"] = int(hdr_dict["nsamples"]/n_t)
    hdr_dict["nsampleslist"] = [int(hdr_dict["nsamples"]/n_t)]
    hdr_dict["nchans"] = int(hdr_dict["nchans"]/n_f)


    calstr = "_" + cal1str + "_" + RM_calstr + "_" + cal_2Dstr + "_" + cal_fitstr + "_" + extra + "_"
    
    outdata =dict()
    outdata["calibration"]=dict()
   
    #Get 2D I Q U V array
    (I,Q,U,V,fobj,timeaxis,freq_test,wav_test) = get_stokes_2D(datadir=datadir,fn_prefix=prefix,nsamps=nsamps,n_t=n_t,n_f=n_f,n_off=n_off,sub_offpulse_mean=True)
    outdata["freq_test"] = arr_to_list_check(freq_test[0])    
    hdr_dict["fch1"] = np.max(freq_test[0])
    hdr_dict["foff"] = -np.abs(freq_test[0][1] - freq_test[0][0])
    
    
    t_samp = fobj.header.tsamp
    outdata["pre-cal"] = dict()
    outdata["pre-cal"]["I"] = arr_to_list_check(I)
    outdata["pre-cal"]["Q"] = arr_to_list_check(Q)
    outdata["pre-cal"]["U"] = arr_to_list_check(U)
    outdata["pre-cal"]["V"] = arr_to_list_check(V)

    if cal:
        outdata["calibration"] = dict()
        outdata["calibration"]["use fit"] = True
        #Get calibrator solutions
        ratio,ratio_fit_params,ratio_sf = gaincal_full(datadir=gain_dir,source_name=gain_source_name,obs_names=gain_obs_names,n_t=n_t,n_f=n_f,nsamps=nsamps,deg=deg,suffix=suffix,average=True,plot=True,sfwindow=sfwindow,clean=clean,padwidth=padwidth,peakheight=peakheight,n_t_down=n_t_down)


       

        if use_fit:
            ratio_fit = np.zeros(np.shape(freq_test[0]))
            for i in range(deg+1):
                ratio_fit += ratio_fit_params[i]*(freq_test[0]**(deg-i))
            ratio_use = ratio_fit
        elif use_sf:
            ratio_use = ratio_sf
        else:
            ratio_use = ratio
        outdata["calibration"]["gain cal"] = dict()
        outdata["calibration"]["gain cal source"] = gain_source_name
        outdata["calibration"]["gain cal observations"] = arr_to_list_check(gain_obs_names)
        outdata["calibration"]["gain cal"]["ratio"] = arr_to_list_check(ratio)
        outdata["calibration"]["gain cal"]["fit params"] = arr_to_list_check(ratio_fit_params)
        outdata["calibration"]["gain cal"]["savgol ratio"] = arr_to_list_check(ratio_sf)
        #hdr_dict["gain_cal_source"] = gain_source_name
        #hdr_dict["gain_cal_observations"] = arr_to_list_check(gain_obs_names)

        phase_diff,phase_fit_params,phase_sf = phasecal_full(datadir=phase_dir,source_name=phase_source_name,obs_names=phase_obs_names,n_t=n_t,n_f=n_f,nsamps=nsamps,deg=deg,suffix=suffix,average=True,plot=True,sfwindow=sfwindow,clean=clean,padwidth=padwidth,peakheight=peakheight,n_t_down=n_t_down) 
        if use_fit:
            phase_fit = np.zeros(np.shape(freq_test[0]))
            for i in range(deg+1):
                phase_fit += phase_fit_params[i]*(freq_test[0]**(deg-i))
            phase_use = phase_fit
        elif use_sf:
            phase_use = phase_sf
        else:
            phase_use = phase_diff
        outdata["calibration"]["phase cal"] = dict()
        outdata["calibration"]["phase cal source"] = phase_source_name
        outdata["calibration"]["phase cal observations"] = arr_to_list_check(phase_obs_names)
        outdata["calibration"]["phase cal"]["phase diff"] = arr_to_list_check(phase_diff)
        outdata["calibration"]["phase cal"]["fit params"] = arr_to_list_check(phase_fit_params)
        outdata["calibration"]["phase cal"]["savgol phase diff"] = arr_to_list_check(phase_sf)
        #hdr_dict["phase_cal_source"] = phase_source_name
        #hdr_dict["phase_cal_observations"] = arr_to_list_check(phase_obs_names)

        (gxx,gyy) = get_calmatrix_from_ratio_phasediff(ratio_use,phase_use)
        outdata["calibration"]["gxx"] = arr_to_list_check(gxx)
        outdata["calibration"]["gyy"] = arr_to_list_check(gyy)


        #calibrate if no RM cal
        if cal_2D:
            outdata["calibration"]["2D cal"] = True
            (I_cal,Q_cal,U_cal,V_cal) = calibrate(I,Q,U,V,(gxx,gyy),stokes=True)
            (I_f_cal,Q_f_cal,U_f_cal,V_f_cal) = get_stokes_vs_freq(I_cal,Q_cal,U_cal,V_cal,width_native,t_samp,n_f,n_t,freq_test,plot=(not RM_cal),datadir=datadir,label=label,calstr=calstr,buff=buff,weighted=weighted,n_t_weight=n_t_weight,timeaxis=timeaxis,fobj=fobj,sf_window_weights=sf_window_weights)
            (I_t_cal,Q_t_cal,U_t_cal,V_t_cal) = get_stokes_vs_time(I_cal,Q_cal,U_cal,V_cal,width_native,t_samp,n_t,plot=(not RM_cal),datadir=datadir,label=label,calstr=calstr,buff=buff) 
        
            if get_RM and RM_in == None:# RM_cal:
                (RM,phi,SNRs,RMerr,significance,B) = faradaycal_full(I_cal,Q_cal,U_cal,V_cal,freq_test,trial_RM,trial_phi,width_native,t_samp,n_trial_RM_zoom,zoom_window=zoom_window,plot=True,datadir=datadir,calstr=calstr,label=label,n_f=n_f,n_t=n_t,fit_window=fit_window,buff=buff,normalize=True,DM=DM,weighted=weighted,n_t_weight=n_t_weight,timeaxis=timeaxis,fobj=fobj,RM_tools=True,trial_RM_tools=trial_RM)
                if RM_cal:
                    (I_cal,Q_cal,U_cal,V_cal) = calibrate_RM(I_cal,Q_cal,U_cal,V_cal,RM,phi,freq_test,stokes=True)
                (I_f_cal,Q_f_cal,U_f_cal,V_f_cal) = get_stokes_vs_freq(I_cal,Q_cal,U_cal,V_cal,width_native,t_samp,n_f,n_t,freq_test,plot=True,datadir=datadir,label=label,calstr=calstr,buff=buff,weighted=weighted,n_t_weight=n_t_weight,timeaxis=timeaxis,fobj=fobj,sf_window_weights=sf_window_weights)
                (I_t_cal,Q_t_cal,U_t_cal,V_t_cal) = get_stokes_vs_time(I_cal,Q_cal,U_cal,V_cal,width_native,t_samp,n_t,plot=True,datadir=datadir,label=label,calstr=calstr,buff=buff)
                outdata["calibration"]["RM cal"] = dict()
                outdata["calibration"]["RM cal"]["RM"] = float(np.real(RM))
                outdata["calibration"]["RM cal"]["RMerr"] = float(np.real(RMerr))
                outdata["calibration"]["RM cal"]["phi"] = float(np.real(phi))
                outdata["calibration"]["RM cal"]["trial RM"] = arr_to_list_check(trial_RM)
                outdata["calibration"]["RM cal"]["trial phi"] = arr_to_list_check(trial_phi)
                outdata["calibration"]["RM cal"]["SNRs"] = arr_to_list_check(SNRs)
                outdata["calibration"]["RM cal"]["p-value"] = float(significance)
                if DM != -1:
                    outdata["calibration"]["RM cal"]["B field (uG)"] = float(B)
                #hdr_dict["RM"] = float(np.real(RM))
                #hdr_dict["RMerr"] = float(np.real(RMerr))
                #hdr_dict["phi"] = float(np.real(phi))
            elif RM_cal and RM_in != None:
                print("RM cal using input " + str(RM_in) + " rad/m^2")
                outdata["calibration"]["RM cal"] = dict()
                outdata["calibration"]["RM cal"]["RM"] = float(np.real(RM_in))
                outdata["calibration"]["RM cal"]["phi"] = float(np.real(phi_in))
                (I_cal,Q_cal,U_cal,V_cal) = calibrate_RM(I_cal,Q_cal,U_cal,V_cal,RM_in,phi_in,freq_test,stokes=True)
                (I_f_cal,Q_f_cal,U_f_cal,V_f_cal) = get_stokes_vs_freq(I_cal,Q_cal,U_cal,V_cal,width_native,t_samp,n_f,n_t,freq_test,plot=True,datadir=datadir,label=label,calstr=calstr,buff=buff,weighted=weighted,n_t_weight=n_t_weight,timeaxis=timeaxis,fobj=fobj,sf_window_weights=sf_window_weights)
                (I_t_cal,Q_t_cal,U_t_cal,V_t_cal) = get_stokes_vs_time(I_cal,Q_cal,U_cal,V_cal,width_native,t_samp,n_t,plot=True,datadir=datadir,label=label,calstr=calstr,buff=buff)
                if DM != -1:
                    outdata["calibration"]["RM cal"]["B field (uG)"] = float(RM_in/(0.81*DM))
        else:
            outdata["calibration"]["2D cal"] = False
            (I_cal,Q_cal,U_cal,V_cal) = (I,Q,U,V)#calibrate(I,Q,U,V,(gxx,gyy),stokes=True)
            (I_t_cal,Q_t_cal,U_t_cal,V_t_cal) = get_stokes_vs_time(I,Q,U,V,width_native,t_samp,n_t,plot=(not RM_cal),datadir=datadir,label=label,calstr=calstr,buff=buff)
            (I_f_cal,Q_f_cal,U_f_cal,V_f_cal) = get_stokes_vs_freq(I,Q,U,V,width_native,t_samp,n_f,n_t,freq_test,plot=(not RM_cal),datadir=datadir,label=label,calstr=calstr,buff=buff,weighted=weighted,n_t_weight=n_t_weight,timeaxis=timeaxis,fobj=fobj,sf_window_weights=sf_window_weights)
            (I_f_cal,Q_f_cal,U_f_cal,V_f_cal) = calibrate(I_f,Q_f,U_f,V_f,(gxx,gyy),stokes=True)
            
            if get_RM and RM_in == None:#RM_cal: 
                (RM,phi,SNRs,RMerr,significance,B) = faradaycal_full(I_f_cal,Q_f_cal,U_f_cal,V_f_cal,freq_test,trial_RM,trial_phi,width_native,t_samp,n_trial_RM_zoom,zoom_window=zoom_window,plot=True,datadir=datadir,calstr=calstr,label=label,n_f=n_f,n_t=n_t,fit_window=fit_window,buff=buff,DM=DM,weighted=weighted,n_t_weight=n_t_weight,timeaxis=timeaxis,fobj=fobj,RM_tools=True,trial_RM_tools=trial_RM)
                if RM_cal:
                    (I_f_cal,Q_f_cal,U_f_cal,V_f_cal) = calibrate_RM(I_f_cal,Q_f_cal,U_f_cal,V_f_cal,RM,phi,freq_test,stokes=True)
                outdata["calibration"]["RM cal"] = dict()
                outdata["calibration"]["RM cal"]["RM"] = float(np.real(RM))
                outdata["calibration"]["RM cal"]["phi"] = float(np.real(phi))
                outdata["calibration"]["RM cal"]["trial RM"] = arr_to_list_check(trial_RM)
                outdata["calibration"]["RM cal"]["trial phi"] =arr_to_list_check(trial_phi)
                outdata["calibration"]["RM cal"]["SNRs"] = arr_to_list_check(SNRs)
                outdata["calibration"]["RM cal"]["RM error"] = float(np.real(RM_err))
                outdata["calibration"]["RM cal"]["p-value"] = float(significance)
                if DM != -1:
                    outdata["calibration"]["RM cal"]["B field (uG)"] = float(B)
                #hdr_dict["RM"] = float(np.real(RM))
                #hdr_dict["RMerr"] = float(np.real(RMerr))
                #hdr_dict["phi"] = float(np.real(phi))
            elif RM_cal and RM_in != None:
                outdata["calibration"]["RM cal"] = dict()
                outdata["calibration"]["RM cal"]["RM"] = float(np.real(RM_in))
                outdata["calibration"]["RM cal"]["phi"] = float(np.real(phi_in))
                (I_cal,Q_cal,U_cal,V_cal) = calibrate_RM(I_cal,Q_cal,U_cal,V_cal,RM_in,phi_in,freq_test,stokes=True)

        outdata["post-cal"] = dict()
        outdata["post-cal"]["I"] = arr_to_list_check(I)
        outdata["post-cal"]["Q"] = arr_to_list_check(Q)
        outdata["post-cal"]["U"] = arr_to_list_check(U)
        outdata["post-cal"]["V"] = arr_to_list_check(V)
    
        outdata["post-cal"]["time"] = dict()
        outdata["post-cal"]["time"]["I_t"] = arr_to_list_check(I_t_cal)
        outdata["post-cal"]["time"]["Q_t"] = arr_to_list_check(Q_t_cal)
        outdata["post-cal"]["time"]["U_t"] = arr_to_list_check(U_t_cal)
        outdata["post-cal"]["time"]["V_t"] = arr_to_list_check(V_t_cal)
    
        outdata["post-cal"]["frequency"] = dict()
        outdata["post-cal"]["frequency"]["I_f"] = arr_to_list_check(I_f_cal)
        outdata["post-cal"]["frequency"]["Q_f"] = arr_to_list_check(Q_f_cal)
        outdata["post-cal"]["frequency"]["U_f"] = arr_to_list_check(U_f_cal)
        outdata["post-cal"]["frequency"]["V_f"] = arr_to_list_check(V_f_cal)

    else:
        (I_cal,Q_cal,U_cal,V_cal) = (I,Q,U,V)
        (I_t_cal,Q_t_cal,U_t_cal,V_t_cal) = get_stokes_vs_time(I,Q,U,V,width_native,t_samp,n_t,plot=(not RM_cal),datadir=datadir,label=label,calstr=calstr,buff=buff)
        (I_f_cal,Q_f_cal,U_f_cal,V_f_cal) = get_stokes_vs_freq(I,Q,U,V,width_native,t_samp,n_f,n_t,freq_test,plot=(not RM_cal),datadir=datadir,label=label,calstr=calstr,buff=buff,weighted=weighted,n_t_weight=n_t_weight,timeaxis=timeaxis,fobj=fobj,sf_window_weights=sf_window_weights)



        outdata["pre-cal"]["time"] = dict()
        outdata["pre-cal"]["time"]["I_t"] = arr_to_list_check(I_t_cal)
        outdata["pre-cal"]["time"]["Q_t"] = arr_to_list_check(Q_t_cal)
        outdata["pre-cal"]["time"]["U_t"] = arr_to_list_check(U_t_cal)
        outdata["pre-cal"]["time"]["V_t"] = arr_to_list_check(V_t_cal)
    
        outdata["pre-cal"]["frequency"] = dict()
        outdata["pre-cal"]["frequency"]["I_f"] = arr_to_list_check(I_f_cal)
        outdata["pre-cal"]["frequency"]["Q_f"] = arr_to_list_check(Q_f_cal)
        outdata["pre-cal"]["frequency"]["U_f"] = arr_to_list_check(U_f_cal)
        outdata["pre-cal"]["frequency"]["V_f"] = arr_to_list_check(V_f_cal)

    #Dynamic Spectra
    plot_spectra_2D(I_cal,Q_cal,U_cal,V_cal,width_native,t_samp,n_t,n_f,freq_test,datadir=datadir,label=label,calstr='calstr',ext=ext,window=window,lim=lim,buff=buff)

    #Polarization Angle and fraction
    PA_f,PA_t,avg_PA,sigma_PA=get_pol_angle((I_t_cal,I_f_cal),(Q_t_cal,Q_f_cal),(U_t_cal,U_f_cal),(V_t_cal,V_f_cal),width_native,t_samp,n_t,n_f,freq_test,plot=True,datadir=datadir,label=label,calstr=calstr,pre_calc_tf=True,buff=buff,weighted=weighted,n_t_weight=n_t_weight,timeaxis=timeaxis,fobj=fobj)
    (total,linear,circular)=get_pol_fraction((I_t_cal,I_f_cal),(Q_t_cal,Q_f_cal),(U_t_cal,U_f_cal),(V_t_cal,V_f_cal),width_native,t_samp,n_t,n_f,freq_test,plot=True,datadir=datadir,label=label,calstr=calstr,pre_calc_tf=True,buff=buff,weighted=weighted,n_t_weight=n_t_weight,timeaxis=timeaxis,fobj=fobj)
    outdata["PA"] = dict()
    outdata["PA"]["frequency"] = arr_to_list_check(PA_f)
    outdata["PA"]["time"] = arr_to_list_check(PA_t)
    outdata["PA"]["average"] = float(avg_PA)
    #hdr_dict["PA"] = float(avg_PA)
    print("PA: " + str(avg_PA))
    print(np.nanstd(PA_t))

    (pol_f,pol_t,avg_pol,sigma_pol,snr_pol) = total
    (L_f,L_t,avg_L,sigma_L,snr_L) = linear
    (C_f,C_t,avg_C,sigma_C,snr_C) = circular
    outdata["polarization"] = dict()
    outdata["polarization"]["frequency"] = arr_to_list_check(pol_f)
    outdata["polarization"]["time"] = arr_to_list_check(pol_t)
    outdata["polarization"]["average"] = float(avg_pol)
    outdata["polarization"]["error"] = float(sigma_pol)
    outdata["polarization"]["snr"] = float(snr_pol)
    print("Polarization:")
    print(avg_pol)
    print(sigma_pol)

    outdata["linear polarization"] = dict()
    outdata["linear polarization"]["frequency"] = arr_to_list_check(L_f)
    outdata["linear polarization"]["time"] = arr_to_list_check(L_t)
    outdata["linear polarization"]["average"] = float(avg_L)
    outdata["linear polarization"]["error"] = float(sigma_L)
    outdata["linear polarization"]["snr"] = float(snr_L)
    print("Linear:")
    print(avg_L)
    print(sigma_L)

    outdata["circular polarization"] = dict()
    outdata["circular polarization"]["frequency"] = arr_to_list_check(C_f)
    outdata["circular polarization"]["time"] = arr_to_list_check(C_t)
    outdata["circular polarization"]["average"] = float(avg_C)
    outdata["circular polarization"]["error"] = float(sigma_C)
    outdata["circular polarization"]["snr"] = float(snr_C)
    print("Circular:")
    print(avg_C)
    print(sigma_C)

    #hdr_dict["polarization"] = float(avg_pol)

    #print(outdata)
    #Save to json
    fname_json = datadir + label + calstr + "_polanalysis_out.json"
    print("Writing output data to " + fname_json)
    with open(fname_json, "w") as outfile:
        json.dump(outdata, outfile)


    #Write Calibrated Stokes params to filterbank
    headerI = Header(hdr_dict)
    headerI["Stokes"] = "I"
    fnameI = datadir + prefix +"_calibrated_0.fil"
    headerI["filenames"] = headerI["filename"] = headerI["basename"] = fnameI
    filI = FilterbankBlock(I_cal,headerI)
    filI.toFile(fnameI)

    headerQ = Header(hdr_dict)
    headerQ["Stokes"] = "Q"
    fnameQ = datadir + prefix + "_calibrated_1.fil"
    headerQ["filenames"] = headerQ["filename"] = headerQ["basename"] = fnameQ
    filQ =FilterbankBlock(Q_cal,headerQ)
    filQ.toFile(fnameQ)

    headerU = Header(hdr_dict)
    headerU["Stokes"] = "U"
    fnameU = datadir + prefix + "_calibrated_2.fil"
    headerU["filenames"] = headerU["filename"] = headerU["basename"] = fnameU
    filU = FilterbankBlock(U_cal,headerU)
    filU.toFile(fnameU)

    headerV = Header(hdr_dict)
    headerV["Stokes"] = "V"
    fnameV = datadir + prefix + "_calibrated_3.fil"
    headerV["filenames"] = headerV["filename"] = headerV["basename"] = fnameV
    filV = FilterbankBlock(V_cal,headerV)
    filV.toFile(fnameV)



    
    return outdata,fname_json,fnameI,fnameQ,fnameU,fnameV
