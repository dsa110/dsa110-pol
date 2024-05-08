import numpy as np
import glob
from datetime import datetime
from datetime import timedelta
from astropy.time import Time
import time
import os
import json
from scipy.interpolate import CubicSpline
from scipy.signal import correlate
from scipy.signal import savgol_filter as sf
from scipy.signal import convolve
from scipy.signal import fftconvolve
from scipy.ndimage import convolve1d
from scipy.signal import peak_widths
from scipy.stats import chi
from scipy.stats import norm
from scipy.stats import kstest
from scipy.optimize import curve_fit
import numpy.ma as ma
import csv
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
from astropy.coordinates import EarthLocation
from scipy.ndimage import convolve1d
import astropy.units as u
from dsapol import dsapol
import RMextract.getRM as gt
import logging
from RMtools_1D.do_RMsynth_1D import run_rmsynth
from RMtools_1D.do_RMclean_1D import run_rmclean
from RMtools_1D.do_QUfit_1D_mnest import run_qufit


"""
This file contains wrapper functions for getting Galactic and ionospheric RM estimates and 
running RM synthesis using the dsapol library
"""

#define longitude and latitude of OVRO
OVRO_lat = 37.2317 #deg
OVRO_lon = -118.2951 #deg
OVRO_height = 1216 #m, Big Pine


#function to get ionospheric RM using a file already downloaded from NASA archives
def get_rm_ion(RA,DEC,mjd,lat=OVRO_lat,lon=OVRO_lon,height=OVRO_height,window=1,prefixes=['COD0','CODG','IGSG','CORG','C1PG','UPCG']):

    """
    This function uses the RMextract library (added this 2024/05/04) to get
    the mean and uncertainty on the ionospheric RM
    """

    #define time range as window of ~1 hour around peak
    t = Time(mjd,format="mjd")#Time('2024-01-01T00:00:00',format='isot',scale ='utc')
    starttime = t.mjd*86400 - (window/2)*3600
    endtime = starttime + (window/2)*3600

    #define location of observatory (default OVRO)
    s = EarthLocation(lat=lat*u.deg,lon=lon*u.deg,height=height*u.m).itrs
    statpos = s.x.value,s.y.value,s.z.value
    
    #define pointing 
    pointing = RA,DEC





    #first try with new format, then old format
    #logger = logging.getLogger()
    #logger.disabled = True
    for prefix in prefixes:
        try:
            print("Trying " + prefix + " with new format")
            RMdict = gt.getRM(ionexPath='./IONEXdata/', radec=pointing, timestep=100, timerange = [starttime, endtime], stat_positions=[statpos,],prefix=prefix,newformat=True)
            break
        except:
                
            try:                    
                print("Trying " + prefix + " with old format")
                RMdict = gt.getRM(ionexPath='./IONEXdata/', radec=pointing, timestep=100, timerange = [starttime, endtime], stat_positions=[statpos,],prefix=prefix,newformat=False)
                break
            except:
                if prefix == prefixes[-1]: 
                    logger.disabled = False
                    return np.nan,np.nan
    #logger.disabled = False           
    print("success! " + prefix)
    #get mean and standard dev 
    RMs = RMdict['RM']['st1'].flatten()
    RMion = np.nanmedian(RMs)
    RMionerr = np.nanstd(RMs)
    return RMion,RMionerr
    
#function to run RM tools
def get_RM_tools(I_fcal,Q_fcal,U_fcal,V_fcal,Ical,Qcal,Ucal,Vcal,freq_test,n_t,maxRM_num_tools=1e6,dRM_tools=10000,n_off=2000):
    """
    This function uses the RM-tools module to run RM synthesis
    """

    #set masked channels to np.nan
    #print("getting errors for each stokes param...")
    Ierr = np.std(Ical[:,:n_off],axis=1)
    try:
        Ierr[Ierr.mask] = np.nan
        Ierr = Ierr.data
    except AttributeError:
        print("not a masked array")

    Qerr = np.std(Qcal[:,:n_off],axis=1)
    try:
        Qerr[Qerr.mask] = np.nan
        Qerr = Qerr.data
    except AttributeError:
        print("not a masked array")

    Uerr = np.std(Ucal[:,:n_off],axis=1)
    try:
        Uerr[Uerr.mask] = np.nan
        Uerr = Uerr.data
    except AttributeError:
        print("not a masked array")

    #print("getting nan masked data...")
    try:
        I_fcal_rmtools = I_fcal.data
        I_fcal_rmtools[I_fcal.mask] = np.nan
    except AttributeError:
        I_fcal_rmtools = I_fcal

    try:
        Q_fcal_rmtools = Q_fcal.data
        Q_fcal_rmtools[Q_fcal.mask] = np.nan
    except AttributeError:
        Q_fcal_rmtools = Q_fcal

    try:
        U_fcal_rmtools = U_fcal.data
        U_fcal_rmtools[U_fcal.mask] = np.nan
    except AttributeError:
        U_fcal_rmtools = U_fcal

    #print("starting RM-tools...")
    #run RM-tools
    out=run_rmsynth([freq_test[0]*1e6,I_fcal_rmtools,Q_fcal_rmtools,U_fcal_rmtools,Ierr,Qerr,Uerr],phiMax_radm2=maxRM_num_tools,dPhi_radm2=dRM_tools)

    #print("starting RM clean...")
    #run RM-clean
    out=run_rmclean(out[0],out[1],2)
    RM = float(out[0]["phiPeakPIchan_rm2"])
    RMerr = float(out[0]["dPhiPeakPIchan_rm2"])
    trial_RM = list(np.array(out[1]["phiArr_radm2"],dtype=float))
    RMsnrs = list(np.array(np.abs(out[1]["cleanFDF"]),dtype=float))
    #print("Done")
    return RM,RMerr,RMsnrs,trial_RM


#helper functions for parabolic fits
#New significance estimate
def L_sigma(Q,U,timestart,timestop,plot=False,weighted=False,I_w_t_filt=None):


    L0_t = np.sqrt(np.mean(Q,axis=0)**2 + np.mean(U,axis=0)**2)

    if weighted:
        L0_t_w = L0_t*I_w_t_filt
        L_trial_binned = convolve(L0_t,I_w_t_filt)
        sigbin = np.argmax(L_trial_binned)
        noise = np.std(np.concatenate([L_trial_binned[:sigbin],L_trial_binned[sigbin+1:]]))
        #print("weighted: " + str(noise))

    else:
        L_trial_cut1 = L0_t[timestart%(timestop-timestart):]
        L_trial_cut = L_trial_cut1[:(len(L_trial_cut1)-(len(L_trial_cut1)%(timestop-timestart)))]
        L_trial_binned = L_trial_cut.reshape(len(L_trial_cut)//(timestop-timestart),timestop-timestart).mean(1)
        sigbin = np.argmax(L_trial_binned)
        noise = (np.std(np.concatenate([L_trial_cut[:sigbin],L_trial_cut[sigbin+1:]])))
        #print("not weighted: " + str(noise))
    return noise

def L_pdf(x,df,width,abs_max=10,numpoints=10000,plot=False):

    delta = np.linspace(-abs_max,abs_max,2*numpoints)[1] - np.linspace(-abs_max,abs_max,2*numpoints)[0]
    y1 = chi.pdf(np.linspace(-abs_max,abs_max,2*numpoints),df=2)*delta
    y2 = copy.deepcopy(y1)
    if plot:
        plt.figure(figsize=(12,6))
        x1=chi.rvs(df=2,size=100000)
        h,b,p = plt.hist(x1,np.linspace(0,abs_max,int(numpoints/100)))
        plt.plot(np.linspace(-abs_max,abs_max,2*numpoints),np.max(h)*y2/np.max(y2))

    for i in range(width-1):
        y2 = convolve(y2,y1,mode="same")
        if plot:
            x1+=chi.rvs(df=2,size=100000)
            h,b,p = plt.hist(x1,np.linspace(0,abs_max,int(numpoints/100)))
            plt.plot(np.linspace(-abs_max,abs_max,2*numpoints),y2)#np.max(h)*y2/delta/np.max(y2/delta))

    fint = interp1d(np.linspace(-abs_max,abs_max,2*numpoints),y2/delta,fill_value="extrapolate")
    #print(x)
    if plot:
        plt.plot(x,np.max(h)*fint(x)/np.max(fint(x)),color="red",linewidth=2)
        plt.xlim(0,abs_max)
        plt.show()
    return fint(x)

def L_pvalue(dat,x,df,width,abs_max=10,numpoints=10000,sigma=1):
    pdf = L_pdf(x,df,width,abs_max=abs_max,numpoints=numpoints)

    idxdat = np.argmin(np.abs(x-dat*sigma))

    pvalue = np.sum(pdf[idxdat:])/np.sum(pdf) #probability that dat*sigma given this dist
    return pvalue

def fit_parabola(x,a,b,c):
    return -a*((x-c)**2) + b

def gauss_scint(x,bw,amp,off):
    return off + amp*np.exp(-np.log(2)*((x/bw)**2))


def lorentz_scint(x,bw,amp,off):
    return off + amp*(bw/(x**2 + (0.5*bw**2)))



#function to run 1D RM synthesis
def get_RM_1D(I_fcal,Q_fcal,U_fcal,V_fcal,Ical,Qcal,Ucal,Vcal,timestart,timestop,freq_test,nRM_num=int(2e6),minRM_num=-1e6,maxRM_num=1e6,n_off=2000,fit=False,fit_window=50,oversamps=5000,weights=None):
    """
    This function uses the manual 1D RM synthesis module to run RM synthesis
    """

    #make RM axis
    trial_RM = np.linspace(minRM_num,maxRM_num,int(nRM_num))
    trial_phi = [0]

    #run RM synthesis
    RM1,phi1,RMsnrs1,RMerr1 = dsapol.faradaycal(I_fcal,Q_fcal,U_fcal,V_fcal,freq_test,trial_RM,trial_phi,plot=False,show=False,fit_window=100,err=True)

    #if set, use better fit of FDF for error
    if fit:
        assert(weights is not None)
        
        poptpar,pcovpar = curve_fit(fit_parabola,trial_RM[np.argmax(RMsnrs1)-fit_window:np.argmax(RMsnrs1)+fit_window],
                RMsnrs1[np.argmax(RMsnrs1)-fit_window:np.argmax(RMsnrs1)+fit_window],
                p0=[1,1,RM1],sigma=1/RMsnrs1[np.argmax(RMsnrs1)-fit_window:np.argmax(RMsnrs1)+fit_window])
        RM1 = poptpar[2]
        
        FWHMRM1zoom,tmp,tmp,tmp = peak_widths(RMsnrs1,[np.argmax(RMsnrs1)])
        noisezoom = L_sigma(Qcal,Ucal,timestart,timestop,plot=False,weighted=True,I_w_t_filt=weights)
        RMerr1 = FWHMRM1zoom*(trial_RM[1]-trial_RM[0])*noisezoom/(2*np.max(RMsnrs1))

    return RM1,RMerr1,RMsnrs1,trial_RM
    
    


