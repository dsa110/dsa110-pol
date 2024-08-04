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
import matplotlib.ticker as ticker
from concurrent.futures import ProcessPoolExecutor, as_completed, wait
import contextlib
"""
This file contains wrapper functions for getting Galactic and ionospheric RM estimates and 
running RM synthesis using the dsapol library
"""

#define longitude and latitude of OVRO
OVRO_lat = 37.2317 #deg
OVRO_lon = -118.2951 #deg
OVRO_height = 1216 #m, Big Pine

import json
f = open(os.environ['DSAPOLDIR'] + "directories.json","r")
dirs = json.load(f)
f.close()
logfile = dirs["logs"] + "RMcal_logfile.txt" #"/media/ubuntu/ssd/sherman/code/dsapol_logfiles/RMcal_logfile.txt"

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
            with open(logfile,"w") as f:
                with contextlib.redirect_stdout(f):
                    #print("Trying " + prefix + " with new format")
                    RMdict = gt.getRM(ionexPath='./IONEXdata/', radec=pointing, timestep=100, timerange = [starttime, endtime], stat_positions=[statpos,],prefix=prefix,newformat=True)
            f.close()
            break
        except:
                
            try:                    
                #print("Trying " + prefix + " with old format")
                RMdict = gt.getRM(ionexPath='./IONEXdata/', radec=pointing, timestep=100, timerange = [starttime, endtime], stat_positions=[statpos,],prefix=prefix,newformat=False)
                break
            except:
                if prefix == prefixes[-1]: 
                    #logger.disabled = False
                    return np.nan,np.nan
    #logger.disabled = False           
    #print("success! " + prefix)
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


def trials_to_multiprocess(ntrials,maxthreads=100,maxbatches=10,maxtrials_per_thread=1000):
    """
    This function takes the number of RM trials and returns an optimized number of 
    batches and threads per batch to be used for RM synthesis. The number of batches is set to
    ~1/100 of the number of trials, but with a maximum of 10 batches and 100 threads. 
    """
    
    #get number of threads per batch
    for nthreads in range(1,maxthreads+1):
        trials_per_thread = ntrials//nthreads
        if trials_per_thread > maxtrials_per_thread:
            continue
        else:
            break

    #get number of batches
    if trials_per_thread > maxtrials_per_thread:
        nbatches = int(np.min([10,trials_per_thread//maxtrials_per_thread]))
    else:
        nbatches = 1

    return nbatches,nthreads

def future_callback_gR1(future,dname):#,fit,weights,trial_RM,fit_window,Qcal,Ucal,timestart,timestop,dname):
    print("RM Synthesis Complete")
    RM1,RMerr1,RMsnrs1,trial_RM = future.result()
    np.save(dirs['logs'] + "RM_files/" + dname + "result.npy",np.array([float(RM1),float(RMerr1)]))
    return
    """
    RM1,phi1,RMsnrs1,RMerr1,tmp = future.result()


    if fit:
        assert(weights is not None)
        print("fitting...")
        poptpar,pcovpar = curve_fit(fit_parabola,trial_RM[np.argmax(RMsnrs1)-fit_window:np.argmax(RMsnrs1)+fit_window],
                RMsnrs1[np.argmax(RMsnrs1)-fit_window:np.argmax(RMsnrs1)+fit_window],
                p0=[1,1,RM1],sigma=1/RMsnrs1[np.argmax(RMsnrs1)-fit_window:np.argmax(RMsnrs1)+fit_window])
        RM1 = poptpar[2]
        print("found RM: " + str(RM1))
        
        FWHMRM1zoom,tmp,tmp,tmp = peak_widths(RMsnrs1,[np.argmax(RMsnrs1)])
        noisezoom = L_sigma(Qcal,Ucal,timestart,timestop,plot=False,weighted=True,I_w_t_filt=weights)
        RMerr1 = FWHMRM1zoom*(trial_RM[1]-trial_RM[0])*noisezoom/(2*np.max(RMsnrs1))
        print("found err: " + str(RMerr1))
    #write results to file
    print("writing to file: " + str(np.array([RM1,RMerr1])))
    np.save(dirs['logs'] + "RM_files/" + dname + "result.npy",np.array([float(RM1),float(RMerr1)]))
    print("done")
    #np.save(dirs['logs'] + "RM_files/" + dname + "SNRs.npy",RMsnrs1)
    #np.save(dirs['logs'] + "RM_files/" + dname + "trialRM.npy",trial_RM)

    return 

    """


#function to run 1D RM synthesis
def get_RM_1D(I_fcal,Q_fcal,U_fcal,V_fcal,Ical,Qcal,Ucal,Vcal,timestart,timestop,freq_test,nRM_num=int(2e6),minRM_num=-1e6,maxRM_num=1e6,n_off=2000,fit=False,fit_window=50,oversamps=5000,weights=None,background=False,sendtodir="",monitor=False):
    """
    This function uses the manual 1D RM synthesis module to run RM synthesis
    """

    #make RM axis
    trial_RM = np.linspace(minRM_num,maxRM_num,int(nRM_num))
    trial_phi = [0]

    if background:
        print("Running 1D RM Synthesis in the background...")
    

        #make directory for output
        dname = "proc_1D_" + Time.now().isot + "/"
        os.mkdir(dirs['logs'] + "RM_files/" + dname)
        np.save(dirs['logs'] + "RM_files/" + dname + "result.npy",np.nan*np.ones(2))
        np.save(dirs['logs'] + "RM_files/" + dname + "SNRs.npy",np.zeros(0))
        np.save(dirs['logs'] + "RM_files/" + dname + "trialRM.npy",np.zeros(0))

        #create executor
        executor = ProcessPoolExecutor(5)
        t = executor.submit(get_RM_1D,I_fcal,Q_fcal,U_fcal,V_fcal,Ical,Qcal,Ucal,Vcal,timestart,timestop,freq_test,nRM_num,minRM_num,maxRM_num,n_off,fit,fit_window,oversamps,weights,False,dirs['logs'] + "RM_files/" + dname,True)
        t.add_done_callback(lambda future: future_callback_gR1(future,dname))


        #t = executor.submit(dsapol.faradaycal,I_fcal,Q_fcal,U_fcal,V_fcal,freq_test,trial_RM,trial_phi,False,dsapol.DEFAULT_DATADIR,"","",1,1,False,fit_window,True,False,True,10,10,0,dirs['logs'] + "RM_files/" + dname,True)    
        #t.add_done_callback(lambda future: future_callback_gR1(future,fit,weights,trial_RM,fit_window,Qcal,Ucal,timestart,timestop,dname))
        
        return dname

    #get opt. number of processes, batches
    nbatches,nthreads = trials_to_multiprocess(len(trial_RM))
    multithread = nthreads > 1

    #run RM synthesis
    RM1,phi1,RMsnrs1,RMerr1,tmp = dsapol.faradaycal(I_fcal,Q_fcal,U_fcal,V_fcal,freq_test,trial_RM,trial_phi,plot=False,show=False,fit_window=fit_window,err=True,matrixmethod=False,multithread=multithread,maxProcesses=nthreads,numbatch=nbatches,sendtodir=sendtodir,monitor=monitor)

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
    
   
def future_callback_gR2(future,dname):#fit,weights,trial_RM,fit_window,dname):
    print("RM Synthesis Complete")
    RM2,RMerr2,upp,low,RMsnrs2,SNRs_full,trial_RM = future.result()
    #RM2,RMerr2,upp,low = future.result()

    np.save(dirs['logs'] + "RM_files/" + dname + "result.npy",np.array([RM2,RMerr2,upp,low]))
    return
    """

    RM2,phi2,RMsnrs2,RMerr2,upp,low,sig,QUnoise,SNRs_full,peak_RMs,tmp = future.result()

    RMerr2 = dsapol.RM_error_fit(np.max(RMsnrs2))
    if fit:
        assert(weights is not None)

        poptpar,pcovpar = curve_fit(fit_parabola,trial_RM[np.argmax(RMsnrs2)-fit_window:np.argmax(RMsnrs2)+fit_window],
                RMsnrs2[np.argmax(RMsnrs2)-fit_window:np.argmax(RMsnrs2)+fit_window],
                p0=[1,1,RM2],sigma=1/RMsnrs2[np.argmax(RMsnrs2)-fit_window:np.argmax(RMsnrs2)+fit_window])
        RM2 = poptpar[2]



    #write results to file
    np.save(dirs['logs'] + "RM_files/" + dname + "result.npy",np.array([RM2,RMerr2,upp,low]))
    #np.save(dirs['logs'] + "RM_files/" + dname + "SNRs.npy",RMsnrs1)
    #np.save(dirs['logs'] + "RM_files/" + dname + "trialRM.npy",trial_RM)

    return
    """
#function to run 2D RM synthesis
def get_RM_2D(Ical,Qcal,Ucal,Vcal,timestart,timestop,width_native,t_samp,buff,n_f,n_t,freq_test,timeaxis,nRM_num=int(2e6),minRM_num=-1e6,maxRM_num=1e6,n_off=2000,fit=False,fit_window=50,oversamps=5000,weights=None,background=False,sendtodir="",monitor=False):
    """
    This function uses the manual 2D RM synthesis module to run RM synthesis
    """

    #make RM axis
    trial_RM = np.linspace(minRM_num,maxRM_num,int(nRM_num))
    trial_phi = [0]


    
    if background:
        print("Running 2D RM Synthesis in the background...")


        #make directory for output
        dname = "proc_2D_" + Time.now().isot + "/"
        os.mkdir(dirs['logs'] + "RM_files/" + dname)
        np.save(dirs['logs'] + "RM_files/" + dname + "result.npy",np.nan*np.ones(4))
        np.save(dirs['logs'] + "RM_files/" + dname + "SNRs.npy",np.zeros(0))
        np.save(dirs['logs'] + "RM_files/" + dname + "SNRs_full.npy",np.zeros((0,timestop-timestart)))
        np.save(dirs['logs'] + "RM_files/" + dname + "trialRM.npy",np.zeros(0))

        #create executor

        executor = ProcessPoolExecutor(5)
        t = executor.submit(get_RM_2D,Ical,Qcal,Ucal,Vcal,timestart,timestop,width_native,t_samp,buff,n_f,n_t,freq_test,timeaxis,nRM_num,minRM_num,maxRM_num,n_off,fit,fit_window,oversamps,weights,False,dirs['logs'] + "RM_files/" + dname,True)
        t.add_done_callback(lambda future: future_callback_gR2(future,dname))


        return dname


    #get opt. number of processes, batches
    nbatches,nthreads = trials_to_multiprocess(len(trial_RM))
    multithread = nthreads > 1
    
    #run RM synthesis
    RM2,phi2,RMsnrs2,RMerr2,upp,low,sig,QUnoise,SNRs_full,peak_RMs,tmp = dsapol.faradaycal_SNR(Ical,Qcal,Ucal,Vcal,freq_test,trial_RM,trial_phi,
                                                                                        width_native,t_samp,plot=False,n_f=n_f,n_t=n_t,
                                                                                        show=False,err=True,buff=buff,weighted=True,n_off=n_off,
                                                                                        timeaxis=timeaxis,full=True,
                                                                                        input_weights=weights[timestart:timestop],
                                                                                        timestart_in=timestart,timestop_in=timestop,matrixmethod=False,
                                                                                        multithread=multithread,maxProcesses=nthreads,numbatch=nbatches,sendtodir=sendtodir,monitor=monitor)
    
    #use error from the exponential fit
    RMerr2 = dsapol.RM_error_fit(np.max(RMsnrs2))

    #if set, use better fit of FDF for peak
    if fit:
        assert(weights is not None)
        poptpar,pcovpar = curve_fit(fit_parabola,trial_RM[np.argmax(RMsnrs2)-fit_window:np.argmax(RMsnrs2)+fit_window],
                RMsnrs2[np.argmax(RMsnrs2)-fit_window:np.argmax(RMsnrs2)+fit_window],
                p0=[1,1,RM2],sigma=1/RMsnrs2[np.argmax(RMsnrs2)-fit_window:np.argmax(RMsnrs2)+fit_window])
        RM2 = poptpar[2]
    return RM2,RMerr2,upp,low,RMsnrs2,SNRs_full,trial_RM


#plot the 2D RM spectrum
def plot_RM_2D(I_tcal,Q_tcal,U_tcal,V_tcal,n_off,n_t,timeaxis,timestart,timestop,RM,RMerr,trial_RM2,SNRs_full,Qnoise,show_calibrated=True,RMcal=np.nan,RMcalerr=np.nan,I_tcal_trm=None,Q_tcal_trm=None,U_tcal_trm=None,V_tcal_trm=None,rmbuff=500,cmapname='viridis',wind=5):
    """
    This function plots the RM spectrum and polarization profile
    """

    #get unbiased linear polarization
    if show_calibrated:
        L_tcal_trm = dsapol.L_unbias(I_tcal_trm,Q_tcal_trm,U_tcal_trm,n_off)    
    L_tcal = dsapol.L_unbias(I_tcal,Q_tcal,U_tcal,n_off)
    
    #Qnoise = np.std(Q_tcal.mean(0)[:n_off])

    #circular polarization
    C_tcal = V_tcal

    #make plot
    tshifted = (timeaxis - np.argmax(I_tcal)*(32.7)*n_t)/1000
    tpeak = (np.argmax(I_tcal)*(32.7)*n_t)/1000

    fig= plt.figure(figsize=(38,28))
    ax0 = plt.subplot2grid(shape=(9, 4), loc=(0, 0), colspan=4,rowspan=2)
    ax1 = plt.subplot2grid(shape=(9, 4), loc=(2, 0), colspan=4,rowspan=2,sharex=ax0)
    ax2 = plt.subplot2grid(shape=(9, 4), loc=(4, 0), colspan=4, rowspan=4)

    ax0.errorbar(tshifted[timestart:timestop],trial_RM2[SNRs_full.argmax(axis=0)],yerr=dsapol.RM_error_fit(SNRs_full.max(axis=0)/Qnoise),fmt='o',color="black",markersize=10,capsize=20,elinewidth=4,zorder=2,markeredgewidth=2)
    ax0.scatter(tshifted[timestart:timestop],trial_RM2[SNRs_full.argmax(axis=0)],c=SNRs_full.max(axis=0),marker='o',s=300,linewidth=2,zorder=3,vmin=np.nanpercentile(SNRs_full,75),vmax=np.nanmax(SNRs_full),cmap=cmapname,edgecolors='black',linewidths=2)

    ax0.set_xlim(((timestart - np.argmax(I_tcal_trm))*32.7*n_t)/1000-wind,((timestop - np.argmax(I_tcal_trm))*32.7*n_t)/1000 +wind)
    #ax0.set_xlim(((timestart -np.argmax(I_tcal_trm))*32.7*n_t)/1000,((timestop - np.argmax(I_tcal_trm))*32.7*n_t)/1000)
    ax0.set_ylabel(r'RM ($rad/m^2$)')
    ax0.set_ylim(np.min(trial_RM2)-10,np.max(trial_RM2)+10)
    ax0.set_ylim(RM-rmbuff,RM+rmbuff)

    ax1.step(tshifted,I_tcal,label=r'I',color="black",linewidth=3,where='post')
    if show_calibrated:
        ax1.step(tshifted,L_tcal_trm,label=r'L (RM calibrated)',color="blue",linewidth=2.5,where='post')
    ax1.step(tshifted,L_tcal,linestyle='--',label=r'L (RM uncalibrated)',color="blue",linewidth=2.5,where='post')
    ax1.step(tshifted,C_tcal,label=r'V',color="orange",linewidth=2,where='post')

    ax1.legend(loc="upper right",fontsize=50,frameon=True,framealpha=1)
    ax1.set_ylabel(r'S/N')
    ax1.set_xlim(((timestart -np.argmax(I_tcal))*32.7*n_t)/1000 - wind,((timestop -np.argmax(I_tcal))*32.7*n_t)/1000 + wind)
    ax2.set_xlabel(r'Time ($m s$)')
    ax2.set_ylabel(r'RM ($rad/m^2$)')


    ax2.imshow(SNRs_full[::-1,:],aspect="auto",interpolation="nearest",vmin=np.nanpercentile(SNRs_full,75),vmax=np.nanmax(SNRs_full),cmap=cmapname,
          extent=(((timestart -np.argmax(I_tcal_trm))*32.7*n_t)/1000,((timestop -np.argmax(I_tcal_trm))*32.7*n_t)/1000,
                 np.min(trial_RM2),np.max(trial_RM2)))
    ax2.set_xlim(((timestart -np.argmax(I_tcal))*32.7*n_t)/1000 - wind,((timestop -np.argmax(I_tcal))*32.7*n_t)/1000 + wind)
    ax0.axhline(RM,color="red",label=r'${{\rm RM}}_{{\rm peak}}={a}\pm{b}$ rad/m$^2$'.format(a=np.around(RM,2),b=np.around(RMerr,2)),linewidth=3,zorder=1)
    if show_calibrated:
        ax0.axhline(RMcal,color="magenta",label=r'${{\rm RM}}_{{\rm cal}}={a}\pm{b}$ rad/m$^2$'.format(a=np.around(RMcal,2),b=np.around(RMcalerr,2)),linewidth=3,zorder=1)
    ax0.legend(loc="upper right",fontsize=50,frameon=True,framealpha=1)

    fig.tight_layout()
    ax1.xaxis.set_major_locator(ticker.NullLocator())
    fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(wspace=0)
    #plt.savefig(datadir + ids + "_" + nickname + "_RMtime_summary_plot.pdf")
    plt.show()

    return
    
