from dsapol import dsapol

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
import copy
import glob
import csv


from numpy.ma import masked_array as ma
from scipy.stats import kstest
from scipy.optimize import curve_fit


from scipy.signal import find_peaks
from scipy.signal import peak_widths
import copy
import numpy as np

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
from RMtools_1D.do_QUfit_1D_mnest import run_qufit
from astropy.time import Time
from astropy.coordinates import EarthLocation
import astropy.units as u

"""
This file contains code for the Polarization Analysis and RM Synthesis Enabled for Calibration (PARSEC)
user interface to the dsa110-pol module. The interface will use Mercury and a jupyter notebook to 
create a web app, and will operate as a wrapper around dsapol.py. 
"""

"""
Plotting parameters
"""
fsize=35
fsize2=30
plt.rcParams.update({
                    'font.size': fsize,
                    'font.family': 'sans-serif',
                    'axes.labelsize': fsize,
                    'axes.titlesize': fsize,
                    'xtick.labelsize': fsize,
                    'ytick.labelsize': fsize,
                    'xtick.direction': 'in',
                    'ytick.direction': 'in',
                    'xtick.top': True,
                    'ytick.right': True,
                    'lines.linewidth': 1,
                    'lines.markersize': 5,
                    'legend.fontsize': fsize2,
                    'legend.borderaxespad': 0,
                    'legend.frameon': False,
                    'legend.loc': 'lower right'})

"""
Factors for RM, pol stuff
"""
RMSF_generated = False
unbias_factor = 1 #1.57
default_path = "/media/ubuntu/ssd/sherman/code/"


"""
Read FRB parameters
"""
FRB_RA = []
FRB_DEC = []
FRB_DM = []
FRBs= []
FRB_mjd = []
FRB_z = []
FRB_w = []
FRB_BEAM = []
FRB_IDS = []
def update_FRB_params(fname="DSA110-FRBs.csv",path=default_path):
    """
    This function updates the global FRB parameters from the provided file. File is a copy
    of the 'tablecsv' tab in the DSA110 FRB spreadsheet.
    """
    with open(path + fname,"r") as csvfile:
        reader = csv.reader(csvfile,delimiter=',')
        for row in reader:
            #print(row)
            if row[0] != "\ufeffname":
                FRBs.append(row[0])
            
                if row[1] != "":
                    FRB_mjd.append(float(row[1]))
                else:
                    FRB_mjd.append(-1)
                
                if row[4] != "":
                    FRB_DM.append(float(row[4]))
                elif row[3] != "":
                    FRB_DM.append(float(row[3]))
                else:
                    FRB_DM.append(-1)
            
                if row[6] != "":
                    FRB_w.append(float(row[6]))
                else:
                    FRB_w.append(-1)
                
                if row[7] != "":
                    FRB_z.append(float(row[7]))
                else:
                    FRB_z.append(-1)
                
                if row[8] != "":
                    FRB_RA.append(float(row[8]))
                else:
                    FRB_RA.append(-1)
                
                if row[9] != "":
                    FRB_DEC.append(float(row[9]))
                else:
                    FRB_DEC.append(-1)
                #print(row)
                if row[10] != "":
                    FRB_BEAM.append(float(row[10]))
                else:
                    FRB_BEAM.append(-1)
                
                if row[11] != "":
                    FRB_IDS.append(str(row[11]))
                else:
                    FRB_IDS.append(-1)
    return
update_FRB_params()

"""
Standard helper functions
"""
#New significance estimate
def L_sigma(Q,U,timestart,timestop,plot=False,weighted=False,I_w_t_filt=None):


    L0_t = np.sqrt(np.mean(Q,axis=0)**2 + np.mean(U,axis=0)**2)
    
    if weighted:
        L0_t_w = L0_t*I_w_t_filt
        L_trial_binned = convolve(L0_t,I_w_t_filt)
        sigbin = np.argmax(L_trial_binned)
        noise = np.std(np.concatenate([L_trial_binned[:sigbin],L_trial_binned[sigbin+1:]]))
        print("weighted: " + str(noise))

    else:
        L_trial_cut1 = L0_t[timestart%(timestop-timestart):]
        L_trial_cut = L_trial_cut1[:(len(L_trial_cut1)-(len(L_trial_cut1)%(timestop-timestart)))]
        L_trial_binned = L_trial_cut.reshape(len(L_trial_cut)//(timestop-timestart),timestop-timestart).mean(1)
        sigbin = np.argmax(L_trial_binned)
        noise = (np.std(np.concatenate([L_trial_cut[:sigbin],L_trial_cut[sigbin+1:]])))
        print("not weighted: " + str(noise))
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
    print(x)
    if plot:
        plt.plot(x,np.max(h)*fint(x)/np.max(fint(x)),color="red",linewidth=2)
        plt.xlim(0,abs_max)
        plt.show()
    return fint(x)
#pdf = L_pdf_weighted(np.linspace(0,m,npoint),m,hhfh,weights=I_w_t_filtnopad,abs_max=m,numpoints=npoint,plot=True)

def L_pdf_weighted(x,df,width,weights,abs_max=10,numpoints=10000,plot=False):
    
    
    
    axis = np.linspace(-abs_max,abs_max,2*numpoints)
    
    delta = axis[1] - axis[0]
    y1 = np.concatenate([np.zeros(numpoints),chi.pdf(axis[numpoints:]/np.abs(weights[0]),df=2)])/np.abs(weights[0])
    y2 = copy.deepcopy(y1)
    if plot:
        plt.figure(figsize=(12,6))
        #x1=chi.rvs(df=2,size=100000)*weights[0]
        #h,b,p = plt.hist(x1,np.linspace(0,0.005,int(numpoints/100)))
        plt.plot(axis,y2/np.max(y2))

        print((0,weights[0]))
    count = 1
    for i in range(1,width):
        #print(i)
        
        if np.abs(weights[i]) > 1e-6:
            print((i,np.abs(weights[i])))
            print("axis " + str((np.max(axis),np.min(axis))))
            #print(weights[i])
            #print(y2/weights[i])
            #print(weights[i])
            #print((axis[numpoints:]/np.abs(weights[i]))[(axis[numpoints:]/np.abs(weights[i]))>0])
            y3=np.concatenate([np.zeros(numpoints),chi.pdf(axis[numpoints:]/np.abs(weights[i]),df=2)])/np.abs(weights[i])
            y2 = delta*convolve(y2,y3,mode="same")
            if plot:
                #x1+=chi.rvs(df=2,size=100000)*np.abs(weights[i])
                #h,b,p = plt.hist(x1,np.linspace(0,0.005,int(numpoints/100)))
                plt.plot(axis,y3/np.max(y3))#np.max(h)*y2/np.max(y2))
                #plt.plot(axis,y2)
            count += 1
        else: 
            print("Skipping: " + str((i,weights[i])))
        if np.any(np.abs(y2) > 1.7976931348623157e+308):
            print("OVERFLOW")
            
        print(y2)
    fint = interp1d(axis,y2/delta,fill_value="extrapolate")
    #plt.figure()
    #plt.plot(x,fint(x),color="red",linewidth=2)
    #plt.show()
    #if plot:
     #   plt.plot(x,np.max(h)*fint(x)/np.max(fint(x)),color="red",linewidth=2)
      #  plt.xlim(0,abs_max)
    #plt.ylim(0,10)
    plt.xlim(-0.001,0.005)
    plt.show()
    
    print("Total count :  " + str(count) + "/" + str(len(weights)))
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



"""
General Layout:
    Mercury creates a web app server and will re-execute cells whenever a widget value is changed. However, it only re-executes cells below the widget that
    changed. Therefore we will layout the dashboard so that (1) the current state is stored as a global dictionary (2) a variable defining the current stage
    as an integer is stored as a global variable (3) the jupyter notebook will call a different plotting function depending on the stage to display the
    correct screen (4) all widgets will be defined in the notebook above the plotting cell
"""

state_dict = dict()
state_dict['current_state'] = 0
state_map = {'load':0,
             'dedisp':1,
             'polcal':2,
             'filter':3,
             'rmsynth':4,
             'pol':5,
             'summarize':6
             }

#Exception to quietly exit cell so that we can short circuit output cells
class StopExecution(Exception):
    def _render_traceback_(self):
        return []



"""
Load data state
"""
frbpath = "/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/"
def get_frbfiles(path=frbpath):
    frbfiles = glob.glob(path + '2*_*')
    return [frbfiles[i][frbfiles[i].index('us/2')+3:] for i in range(len(frbfiles))]


def load_screen(frbfiles_menu,n_t_slider,logn_f_slider,logibox_slider,buff_L_slider,buff_R_slider,RA_display,DEC_display,ibeam_display,loadbutton,path=frbpath):
    """
    This function updates the FRB loading screen
    """
    #first update state dict
    state_dict['base_n_t'] = n_t_slider.value
    state_dict['base_n_f'] = 2**logn_f_slider.value
    state_dict['ids'] = frbfiles_menu.value[:frbfiles_menu.value.index('_')]
    state_dict['nickname'] = frbfiles_menu.value[frbfiles_menu.value.index('_')+1:]
    state_dict['RA'] = FRB_RA[FRB_IDS.index(state_dict['ids'])]
    state_dict['DEC'] = FRB_DEC[FRB_IDS.index(state_dict['ids'])]
    state_dict['ibeam'] = int(FRB_BEAM[FRB_IDS.index(state_dict['ids'])])
    state_dict['DM0'] = FRB_DM[FRB_IDS.index(state_dict['ids'])]
    state_dict['datadir'] = path + state_dict['ids'] + "_" + state_dict['nickname'] + "/"
    state_dict['buff'] = [buff_L_slider.value,buff_R_slider.value]
    state_dict['width_native'] = 2**logibox_slider.value

    #update displays
    RA_display.data = state_dict['RA']
    DEC_display.data = state_dict['DEC']
    ibeam_display.data = state_dict['ibeam']

    #if button is clicked, load FRB data and go to next screen
    if loadbutton.clicked:

        #load data at base resolution
        (I,Q,U,V,fobj,timeaxis,freq_test,wav_test,badchans) = dsapol.get_stokes_2D(state_dict['datadir'],state_dict['ids'] + "_dev",5120,start=12800,n_t=state_dict['base_n_t'],n_f=state_dict['base_n_f'],n_off=int(2000//state_dict['base_n_t']),sub_offpulse_mean=True,fixchans=True)
        state_dict['base_I'] = I
        state_dict['base_Q'] = Q
        state_dict['base_U'] = U
        state_dict['base_V'] = V
        state_dict['fobj'] = fobj
        state_dict['base_freq_test'] = freq_test
        state_dict['base_wav_test'] = wav_test
        state_dict['badchans'] = badchans

        state_dict['current_state'] += 1


        

    return
    
"""
Dedispersion Tuning state
"""
def get_min_DM_step(n_t,fminGHz=1.307,fmaxGHz=1.493,res=32.7e-3):
    return np.around((res)*n_t/(4.15)/((1/fminGHz**2) - (1/fmaxGHz**2)),2)

def dedisperse(dyn_spec,DM,tsamp,freq_axis):
    """
    This function dedisperses a dynamic spectrum of shape nsamps x nchans by brute force without accounting for edge effects
    """

    #get delay axis
    tdelays = DM*4.15*(((np.min(freq_axis)*1e-3)**(-2)) - ((freq_axis*1e-3)**(-2)))#(8.3*(chanbw)*burst_DMs[i]/((freq_axis*1e-3)**3))*(1e-3) #ms
    tdelays_idx_hi = np.array(np.ceil(tdelays/tsamp),dtype=int)
    tdelays_idx_low = np.array(np.floor(tdelays/tsamp),dtype=int)
    tdelays_frac = tdelays/tsamp - tdelays_idx_low
    print("Trial DM: " + str(DM) + " pc/cc, DM delays (ms): " + str(tdelays) + "...",end='')
    nchans = len(freq_axis)
    print(dyn_spec.shape)
    dyn_spec_DM = np.zeros(dyn_spec.shape)

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
    print("Done!")
    return dyn_spec_DM


def dedisp_screen(n_t_slider,logn_f_slider,logwindow_slider,ddm_num,DM_input_display,DM_new_display,DMdonebutton):
    """
    This function updates the dedispersion screen when resolution
    or dm step are changed
    """

    

    #update DM step size
    state_dict['dDM'] = ddm_num.value
    ddm_num.step = get_min_DM_step(n_t_slider.value*state_dict['base_n_t'])

    #update new DM
    DM_new_display.data = DM_input_display.data + ddm_num.value
    state_dict['DM'] = DM_new_display.data

    #update time, freq resolution
    state_dict['window'] = 2**logwindow_slider.value
    state_dict['rel_n_t'] = n_t_slider.value
    state_dict['rel_n_f'] = (2**logn_f_slider.value)
    state_dict['n_t'] = n_t_slider.value*state_dict['base_n_t']
    state_dict['n_f'] = (2**logn_f_slider.value)*state_dict['base_n_f']
    state_dict['freq_test'] = [state_dict['base_freq_test'][0].reshape(len(state_dict['base_freq_test'][0])//(2**logn_f_slider.value),(2**logn_f_slider.value)).mean(1)]*4
    state_dict['I'] = dsapol.avg_time(state_dict['base_I'],n_t_slider.value)#state_dict['n_t'])
    state_dict['I'] = dsapol.avg_freq(state_dict['I'],2**logn_f_slider.value)#state_dict['n_f'])
    state_dict['Q'] = dsapol.avg_time(state_dict['base_Q'],n_t_slider.value)#state_dict['n_t'])
    state_dict['Q'] = dsapol.avg_freq(state_dict['Q'],2**logn_f_slider.value)#state_dict['n_f'])
    state_dict['U'] = dsapol.avg_time(state_dict['base_U'],n_t_slider.value)#state_dict['n_t'])
    state_dict['U'] = dsapol.avg_freq(state_dict['U'],2**logn_f_slider.value)#state_dict['n_f'])
    state_dict['V'] = dsapol.avg_time(state_dict['base_V'],n_t_slider.value)#state_dict['n_t'])
    state_dict['V'] = dsapol.avg_freq(state_dict['V'],2**logn_f_slider.value)#state_dict['n_f'])

    #dedisperse
    state_dict['I'] = dedisperse(state_dict['I'],state_dict['dDM'],(32.7e-3)*state_dict['n_t'],state_dict['freq_test'][0])
    state_dict['Q'] = dedisperse(state_dict['Q'],state_dict['dDM'],(32.7e-3)*state_dict['n_t'],state_dict['freq_test'][0])
    state_dict['U'] = dedisperse(state_dict['U'],state_dict['dDM'],(32.7e-3)*state_dict['n_t'],state_dict['freq_test'][0])
    state_dict['V'] = dedisperse(state_dict['V'],state_dict['dDM'],(32.7e-3)*state_dict['n_t'],state_dict['freq_test'][0])



    #get time series
    (state_dict['I_t'],state_dict['Q_t'],state_dict['U_t'],state_dict['V_t']) = dsapol.get_stokes_vs_time(state_dict['I'],state_dict['Q'],state_dict['U'],state_dict['V'],state_dict['width_native'],state_dict['fobj'].header.tsamp,state_dict['n_t'],n_off=int(12000//state_dict['n_t']),plot=False,show=False,normalize=True,buff=state_dict['buff'],window=30)
    state_dict['time_axis'] = 32.7*state_dict['n_t']*np.arange(0,len(state_dict['I_t']))

    #get timestart, timestop
    (state_dict['peak'],state_dict['timestart'],state_dict['timestop']) = dsapol.find_peak(state_dict['I'],state_dict['width_native'],state_dict['fobj'].header.tsamp,n_t=n_t_slider.value,peak_range=None,pre_calc_tf=False,buff=state_dict['buff'])

    #display dynamic spectrum
    fig, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 2]},figsize=(18,18))
    a0.step(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3,
            state_dict['I_t'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],label='I')
    a0.step(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3,
            state_dict['Q_t'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],label='Q')
    a0.step(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3,
            state_dict['U_t'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],label='U')
    a0.step(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3,
            state_dict['V_t'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],label='V')
    a0.set_xlim(32.7*state_dict['n_t']*state_dict['timestart']*1e-3 - state_dict['window']*32.7*state_dict['n_t']*1e-3,
            32.7*state_dict['n_t']*state_dict['timestop']*1e-3 + state_dict['window']*32.7*state_dict['n_t']*1e-3)
    a0.set_xticks([])
    a0.legend(loc="upper right")
    
    a1.imshow(state_dict['I'][:,state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],aspect='auto',
            extent=[32.7*state_dict['n_t']*state_dict['timestart']*1e-3 - state_dict['window']*32.7*state_dict['n_t']*1e-3,
                32.7*state_dict['n_t']*state_dict['timestop']*1e-3 + state_dict['window']*32.7*state_dict['n_t']*1e-3,
                np.min(state_dict['freq_test'][0]),np.max(state_dict['freq_test'][0])])
    a1.set_xlabel("Time (ms)")
    a1.set_ylabel("Frequency (MHz)")
    plt.subplots_adjust(hspace=0)
    plt.show(fig)


    if DMdonebutton.clicked:
        #dedisperse base dyn spectrum
        state_dict['base_I'] = dedisperse(state_dict['base_I'],state_dict['dDM'],(32.7e-3)*state_dict['base_n_t'],state_dict['base_freq_test'][0])
        state_dict['base_Q'] = dedisperse(state_dict['base_Q'],state_dict['dDM'],(32.7e-3)*state_dict['base_n_t'],state_dict['base_freq_test'][0])
        state_dict['base_U'] = dedisperse(state_dict['base_U'],state_dict['dDM'],(32.7e-3)*state_dict['base_n_t'],state_dict['base_freq_test'][0])
        state_dict['base_V'] = dedisperse(state_dict['base_V'],state_dict['dDM'],(32.7e-3)*state_dict['base_n_t'],state_dict['base_freq_test'][0])



        state_dict['current_state'] += 1
    return


"""
Calibration state
"""
def read_polcal(polcaldate,path=default_path):
    """
    This function reads in pol calibration parameters (gxx,gyy) from
    the calibration file provided
    """
    with open(path + polcaldate,'r') as csvfile:
        reader = csv.reader(csvfile,delimiter=",")
        for row in reader:
            if row[0] == "|gxx|/|gyy|":
                tmp_ratio = np.array(row[1:],dtype="float")
            elif row[0] == "|gxx|/|gyy| fit":
                tmp_ratio_fit = np.array(row[1:],dtype="float")
            if row[0] == "phixx-phiyy":
                tmp_phase = np.array(row[1:],dtype="float")
            if row[0] == "phixx-phiyy fit":
                tmp_phase_fit = np.array(row[1:],dtype="float")
            if row[0] == "|gyy|":
                tmp_gainY = np.array(row[1:],dtype="float")
            if row[0] == "|gyy| FIT":
                tmp_gainY_fit = np.array(row[1:],dtype="float")
            if row[0] == "gxx":
                gxx = np.array(row[1:],dtype="complex")
            if row[0] == "gyy":
                gyy = np.array(row[1:],dtype="complex")
            if row[0] == "freq_axis":
                freq_axis = np.array(row[1:],dtype="float")
    return gxx,gyy,freq_axis



def polcal_screen(polcaldate_menu,polcalbutton,ParA_display):
    """
    This function updates the polarization calibration screen
    whenever the cal file is selected
    """

    #update polcal parameters in state dict
    state_dict['gxx'],state_dict['gyy'],state_dict['cal_freq_axis'] = read_polcal(polcaldate_menu.value)

    #display
    fig=plt.figure(figsize=(18,14))
    plt.subplot(311)
    plt.xticks([])
    plt.ylabel(r'$|g_{xx}|/|g_{yy}|$')
    plt.plot(state_dict['cal_freq_axis'],np.abs(state_dict['gxx'])/np.abs(state_dict['gyy']))

    plt.subplot(312)
    plt.xticks([])
    plt.ylabel(r'$\angle g_{xx} - \angle g_{yy}$')
    plt.plot(state_dict['cal_freq_axis'],np.angle(state_dict['gxx'])-np.angle(state_dict['gyy']))

    plt.subplot(313)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel(r'$|g_{yy}|$')
    plt.plot(state_dict['cal_freq_axis'],np.abs(state_dict['gyy']))
    plt.subplots_adjust(hspace=0)
    plt.show()

    if polcalbutton.clicked:

        #calibrate at native resolution
        state_dict['base_Ical'],state_dict['base_Qcal'],state_dict['base_Ucal'],state_dict['base_Vcal'] = dsapol.calibrate(state_dict['base_I'],state_dict['base_Q'],state_dict['base_U'],state_dict['base_V'],(state_dict['gxx'],state_dict['gyy']),stokes=True)

        #parallactic angle calibration
        state_dict['base_Ical'],state_dict['base_Qcal'],state_dict['base_Ucal'],state_dict['base_Vcal'],state_dict['ParA[deg]'] = dsapol.calibrate_angle(state_dict['base_Ical'],state_dict['base_Qcal'],state_dict['base_Ucal'],state_dict['base_Vcal'],state_dict['fobj'],state_dict['ibeam'],state_dict['RA'],state_dict['DEC'])
        ParA_display.data = state_dict['ParA[deg]']

        #get downsampled versions
        state_dict['Ical'] = dsapol.avg_time(state_dict['base_Ical'],state_dict['rel_n_t'])
        state_dict['Ical'] = dsapol.avg_freq(state_dict['Ical'],state_dict['rel_n_f'])
        state_dict['Qcal'] = dsapol.avg_time(state_dict['base_Qcal'],state_dict['rel_n_t'])
        state_dict['Qcal'] = dsapol.avg_freq(state_dict['Qcal'],state_dict['rel_n_f'])
        state_dict['Ucal'] = dsapol.avg_time(state_dict['base_Ucal'],state_dict['rel_n_t'])
        state_dict['Ucal'] = dsapol.avg_freq(state_dict['Ucal'],state_dict['rel_n_f'])
        state_dict['Vcal'] = dsapol.avg_time(state_dict['base_Vcal'],state_dict['rel_n_t'])
        state_dict['Vcal'] = dsapol.avg_freq(state_dict['Vcal'],state_dict['rel_n_f'])


        #get time series
        (state_dict['I_tcal'],state_dict['Q_tcal'],state_dict['U_tcal'],state_dict['V_tcal']) = dsapol.get_stokes_vs_time(state_dict['Ical'],state_dict['Qcal'],state_dict['Ucal'],state_dict['Vcal'],state_dict['width_native'],state_dict['fobj'].header.tsamp,state_dict['n_t'],n_off=int(12000//state_dict['n_t']),plot=False,show=False,normalize=True,buff=state_dict['buff'],window=30)
        state_dict['time_axis'] = 32.7*state_dict['n_t']*np.arange(0,len(state_dict['I_tcal']))

        #get timestart, timestop
        (state_dict['peak'],state_dict['timestart'],state_dict['timestop']) = dsapol.find_peak(state_dict['Ical'],state_dict['width_native'],state_dict['fobj'].header.tsamp,n_t=state_dict['rel_n_t'],peak_range=None,pre_calc_tf=False,buff=state_dict['buff'])

        state_dict['current_state'] += 1

    return


"""
Filter Weights State
"""

def filter_screen():

    #display dynamic spectrum
    fig, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 2]},figsize=(18,18))
    a0.step(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3,
            state_dict['I_tcal'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],label='I')
    a0.step(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3,
            state_dict['Q_tcal'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],label='Q')
    a0.step(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3,
            state_dict['U_tcal'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],label='U')
    a0.step(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3,
            state_dict['V_tcal'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],label='V')
    a0.set_xlim(32.7*state_dict['n_t']*state_dict['timestart']*1e-3 - state_dict['window']*32.7*state_dict['n_t']*1e-3,
            32.7*state_dict['n_t']*state_dict['timestop']*1e-3 + state_dict['window']*32.7*state_dict['n_t']*1e-3)
    a0.set_xticks([])
    a0.legend(loc="upper right")

    a1.imshow(state_dict['Ical'][:,state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],aspect='auto',
            extent=[32.7*state_dict['n_t']*state_dict['timestart']*1e-3 - state_dict['window']*32.7*state_dict['n_t']*1e-3,
                32.7*state_dict['n_t']*state_dict['timestop']*1e-3 + state_dict['window']*32.7*state_dict['n_t']*1e-3,
                np.min(state_dict['freq_test'][0]),np.max(state_dict['freq_test'][0])])
    a1.set_xlabel("Time (ms)")
    a1.set_ylabel("Frequency (MHz)")
    plt.subplots_adjust(hspace=0)
    plt.show(fig)

    return
