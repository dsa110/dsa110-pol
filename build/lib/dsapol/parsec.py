from dsapol import polbeamform
from dsapol import polcal
from dsapol import RMcal
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
import pandas as pd
import pickle as pkl
from numpy.ma import masked_array as ma
from scipy.stats import kstest
from scipy.optimize import curve_fit
from syshealth.status_mon import get_rm

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
import os
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
Repo path
"""
repo_path = "/media/ubuntu/ssd/sherman/code/dsa110-pol/"

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
def update_FRB_params(fname="DSA110-FRBs-PARSEC_TABLE.csv",path=repo_path):
    """
    This function updates the global FRB parameters from the provided file. File is a copy
    of the 'tablecsv' tab in the DSA110 FRB spreadsheet.
    """
    with open(repo_path + 'data/' + fname,"r") as csvfile:
        reader = csv.reader(csvfile,delimiter=',')
        for row in reader:
            #print(row)
            if row[0] != "name" and row[0] != "": #\ufeff
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

        #print((0,weights[0]))
    count = 1
    for i in range(1,width):
        #print(i)
        
        if np.abs(weights[i]) > 1e-6:
            #print((i,np.abs(weights[i])))
            #print("axis " + str((np.max(axis),np.min(axis))))
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
        #else: 
        #    print("Skipping: " + str((i,weights[i])))
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
    
    #print("Total count :  " + str(count) + "/" + str(len(weights)))
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

# default values for state and polcal dicts and dataframe tables
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
state_dict['comps'] = dict()
state_dict['current_comp'] = 0
state_dict['n_comps'] = 1
state_dict['rel_n_t'] = 1
state_dict['rel_n_f'] = 2**5
df = pd.DataFrame(
    {
        r'${\rm buff}_{L}$': [np.nan],
        r'${\rm buff}_{R}$': [np.nan],
        r'$n_{tw}$':[np.nan],
        r'$sf_{ww}$':[np.nan],
        'lower mask limit':[np.nan],
        'upper mask limit':[np.nan],
        r'S/N':[np.nan]
    },
        index=['All']
    )
corrarray = ["corr03",
               "corr04",
               "corr05",
               "corr06",
               "corr07",
               "corr08",
               "corr10",
               "corr11",
               "corr12" ,
               "corr14", 
               "corr15", 
               "corr16", 
               "corr18", 
               "corr19", 
               "corr21", 
               "corr22"]
df_polcal = pd.DataFrame(
    {
        r'3C48': [],#*len(corrarray),
        r'3C48 beamformer weights':[],
        r'3C286': [],#*len(corrarray),
        r'3C286 beamformer weights':[]
    },
        index=[]#copy.deepcopy(corrarray)
    )
polcal_dict = dict()
polcal_dict['polcal_avail_3C48'],polcal_dict['polcal_avail_3C286'],polcal_dict['polcal_avail_bf_3C48'],polcal_dict['polcal_avail_bf_3C286'] = [],[],[],[]
#populate w/ calibrator files
vtimes3C48_init,vfiles3C48_init = polcal.get_voltages("3C48")
vtimes3C286_init,vfiles3C286_init = polcal.get_voltages("3C286")
mapping3C48_init = polcal.iso_voltages(vtimes3C48_init,vfiles3C48_init)
mapping3C286_init = polcal.iso_voltages(vtimes3C286_init,vfiles3C286_init)
for k in mapping3C48_init.keys():
    if len(mapping3C48_init[k]) == 0: continue
    #find corresponding bf weights
    bfweights3C48 = polcal.get_bfweights(mapping3C48_init[k],'3C48')
    bfweights3C286 = polcal.get_bfweights(mapping3C286_init[k],'3C286')
    #print(['<br/>'.join(mapping3C48_init[k]),
    #                         '<br/>'.join(bfweights3C48),
    #                         '<br/>'.join(mapping3C286_init[k]),
    #                         '<br/>'.join(bfweights3C286)])
    df_polcal.loc[str(k)] = ['<br/>'.join(mapping3C48_init[k]),
                             '<br/>'.join(bfweights3C48),
                             '<br/>'.join(mapping3C286_init[k]),
                             '<br/>'.join(bfweights3C286)]
    polcal_dict[str(k)] = dict()
    polcal_dict[str(k)]['3C48'] = mapping3C48_init[k]
    polcal_dict[str(k)]['3C286'] = mapping3C286_init[k]
    polcal_dict[str(k)]['3C48_bfweights'] = bfweights3C48
    polcal_dict[str(k)]['3C286_bfweights'] = bfweights3C286
df_beams = pd.DataFrame(
        {r'beam':[],
            r'mjd':[],
            r'beamformer weights':[]},
        index=[]
        )


#List of initial widget values updated whenever screen loads
frbpath = "/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/"
def get_frbfiles(path=frbpath):
    frbfiles = glob.glob(path + '2*_*')
    return [frbfiles[i][frbfiles[i].index('us/2')+3:] for i in range(len(frbfiles))]

frbfiles = get_frbfiles()
ids = frbfiles[0][:frbfiles[0].index('_')]
RA = FRB_RA[FRB_IDS.index(ids)]
DEC = FRB_DEC[FRB_IDS.index(ids)]
ibeam = int(FRB_BEAM[FRB_IDS.index(ids)])
mjd = FRB_mjd[FRB_IDS.index(ids)]
DMinit = FRB_DM[FRB_IDS.index(ids)]
RM_gal_init,RM_galerr_init = np.nan,np.nan#get_rm(radec=(RA,DEC),filename=repo_path + "/data/faraday2020v2.hdf5")
RM_ion_init,RM_ionerr_init = np.nan,np.nan#RMcal.get_rm_ion(RA,DEC,mjd)






polcalfiles_findbeams = polcal.get_beamfinding_files()
polcaldates = []
for k in polcal_dict.keys():
    if 'polcal' not in str(k):
        polcaldates.append(str(k))
polcalfiles_bf = polcal.get_avail_caldates()
polcalfiles_findbeams = polcal.get_beamfinding_files()

obs_files_3C48,obs_ids_3C48 = polcal.get_calfil_files('3C48',polcalfiles_findbeams[0],'3C48*0')
obs_files_3C286,obs_ids_3C286 = polcal.get_calfil_files('3C286',polcalfiles_findbeams[0],'3C286*0')

polcalfiles = glob.glob(default_path + 'POLCAL_PARAMETERS_*csv')
polcalfiles = [polcalfiles[i][polcalfiles[i].index('POLCAL'):] for i in range(len(polcalfiles))]



wdict = {'toggle_menu':'(0) Load Data', ############### (0) Load Data ##################
         'frbfiles_menu':frbfiles[0],
         'base_n_t_slider':1,
         'base_logn_f_slider':0,
         'logibox_slider_init':0,
         'buff_L_slider_init':1,
         'buff_R_slider_init':1,
         'RA_display':RA,
         'DEC_display':DEC,
         'ibeam_display':ibeam,
         'mjd_display':mjd,
         'DM_init_display':DMinit,
         'showlog':True,

         'n_t_slider':1, ############### (1) Dedispersion ##################
         'logn_f_slider':5,
         'logwindow_slider_init':5,
         'ddm_num':0,
         'DM_input_display':DMinit,
         'DM_new_display':DMinit,

         'polcaldate_create_menu':"", ############### (3) Calibration ##################
         'polcaldate_bf_menu':"",
         'polcaldate_findbeams_menu':polcalfiles_findbeams[0],
         'obsid3C48_menu':"",
         'obsid3C286_menu':"",
         'ParA_display':np.nan,
         'peakheight_slider':2,
         'peakwidth_slider':10,
         'sfflag':False,
         'sf_window_weight_cals':255,
         'sf_order_cals':5,
         'polyfitflag':False,
         'polyfitorder_slider':5,
         'edgefreq_slider':1370,
         'breakfreq_slider':1370,
         'ratio_peakheight_slider':3,
         'ratio_peakwidth_slider':10,
         'ratio_sfflag':False,
         'ratio_sf_window_weight_cals':257,
         'ratio_sf_order_cals':5,
         'ratio_polyfitflag':False,
         'ratio_polyfitorder_slider':5,
         'ratio_edgefreq_slider':1360,
         'ratio_breakfreq_slider':1360,
         'phase_peakheight_slider':3,
         'phase_peakwidth_slider':10,
         'phase_sfflag':False,
         'phase_sf_window_weight_cals':255,
         'phase_sf_order_cals':5,
         'phase_polyfitflag':False,
         'phase_polyfitorder_slider':5,
         'polcaldate_menu':"",
         'showlogcal':False,

         'ncomps_num':1, ############### (4) Filter Weights ##################
         'comprange_slider':[0,1],                  
         'avger_w_slider':1,
         'sf_window_weights_slider':3,
         'logibox_slider':0,
         'logwindow_slider':5,
         'buff_L_slider':1,
         'buff_R_slider':1,
         'n_t_slider_filt':1,
         'logn_f_slider_filt':5,

        'useRMTools':True, ################ (5) RM Synthesis ################
        'maxRM_num_tools':1e6,
        'dRM_tools':200,
        'useRMsynth':True,
        'nRM_num':int(2e6),
        'minRM_num':-1e6,
        'maxRM_num':1e6,
        'useRM2D':True,
        'nRM_num_zoom':5000,
        'RM_window_zoom':1000,
        'dRM_tools_zoom':0.4,
        'RM_gal_display':np.around(RM_gal_init,2),
        'RM_galerr_display':np.around(RM_galerr_init,2),
        'RM_ion_display':np.around(RM_ion_init,2),
        'RM_ionerr_display':np.around(RM_ionerr_init,2),
        'rmcomp_menu':'All',
        'rmcomp_menu_choices':['All','']
}

state_dict['RMcalibrated'] = dict()
state_dict['RMcalibrated']['RMsnrs1'] = np.nan*np.ones(int(wdict['nRM_num']))
state_dict['RMcalibrated']['RM_tools_snrs'] = np.nan*np.ones(int(2*wdict['maxRM_num']/wdict['dRM_tools']))
state_dict['RMcalibrated']['RMsnrs1zoom'] = np.nan*np.ones(int(wdict['nRM_num_zoom']))
state_dict['RMcalibrated']['RM_tools_snrszoom'] = np.nan*np.ones(int(2*wdict['RM_window_zoom']/wdict['dRM_tools_zoom']))
state_dict['RMcalibrated']['RMsnrs2'] = np.nan*np.ones(int(wdict['nRM_num_zoom']))
state_dict['RMcalibrated']["RM2"] = [np.nan,np.nan]
state_dict["RMcalibrated"]["RMerrfit"] = np.nan
state_dict["RMcalibrated"]["trial_RM1"] = np.linspace(wdict['minRM_num'],wdict['maxRM_num'],int(wdict['nRM_num']))
state_dict["RMcalibrated"]["trial_RM2"] = np.linspace(-wdict['RM_window_zoom'],wdict['RM_window_zoom'],int(wdict['nRM_num_zoom']))
state_dict["RMcalibrated"]["trial_RM_tools"] = np.arange(-wdict['maxRM_num'],wdict['maxRM_num'],wdict['dRM_tools'])
state_dict["RMcalibrated"]["trial_RM_toolszoom"] = np.arange(-wdict['RM_window_zoom'],wdict['RM_window_zoom'],wdict['dRM_tools_zoom'])




def update_wdict(objects,labels,param='value'):
    """
    This function takes a list of widget objects and a list of their names as strings and updates the wdict with their curent values
    """

    assert(len(objects)==len(labels))
    for i in range(len(objects)):
        if param == 'value':
            wdict[labels[i]] = objects[i].value
        elif param == 'data':
            wdict[labels[i]] = objects[i].data
    if 'DM_init_display' in labels:
        wdict['DM_input_display'] = wdict['DM_init_display']
        wdict['DM_new_display'] = wdict['DM_init_display'] + wdict['ddm_num']
    if 'DM_input_display' in labels: 
        wdict['DM_input_display'] = wdict['DM_init_display']
        objects[labels.index('DM_input_display')].data = wdict['DM_init_display']
    if 'DM_new_display' in labels:
        wdict['DM_new_display'] = wdict['DM_input_display'] + wdict['ddm_num']
        objects[labels.index('DM_new_display')].data = wdict['DM_input_display'] + wdict['ddm_num']
    if 'rmcomp_menu' in labels:
        if state_dict['n_comps'] > 1:
            wdict['rmcomp_menu_choices'] = [str(i) for i in range(state_dict['n_comps'])] + ['All','']
        else:
            wdict['rmcomp_menu_choices'] = ['All','']
    return

        


#Exception to quietly exit cell so that we can short circuit output cells
class StopExecution(Exception):
    def _render_traceback_(self):
        return []



"""
Load data state
"""
NOFFDEF = 2000
def load_screen(frbfiles_menu,n_t_slider,logn_f_slider,logibox_slider,buff_L_slider_init,buff_R_slider_init,RA_display,DEC_display,DM_init_display,ibeam_display,mjd_display,updatebutton,filbutton,loadbutton,path=frbpath):
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
    state_dict['buff'] = [buff_L_slider_init.value,buff_R_slider_init.value]
    state_dict['width_native'] = 2**logibox_slider.value
    state_dict['mjd'] = FRB_mjd[FRB_IDS.index(state_dict['ids'])]

    #update displays
    RA_display.data = state_dict['RA']
    DEC_display.data = state_dict['DEC']
    ibeam_display.data = state_dict['ibeam']
    mjd_display.data = state_dict['mjd']
    DM_init_display.data = state_dict['DM0']

    #see if filterbanks exist
    state_dict['fils'] = polbeamform.get_fils(state_dict['ids'],state_dict['nickname'])

    #find beamforming weights date
    state_dict['bfweights'] = polbeamform.get_bfweights(state_dict['ids'])

    #if update button is clicked, refresh FRB data from csv
    if updatebutton.clicked:
        update_FRB_params()

    #if button is clicked, load FRB data and go to next screen
    if loadbutton.clicked:

        #load data at base resolution
        (I,Q,U,V,fobj,timeaxis,freq_test,wav_test,badchans) = dsapol.get_stokes_2D(state_dict['datadir'],state_dict['ids'] + "_dev",5120,start=12800,n_t=state_dict['base_n_t'],n_f=state_dict['base_n_f'],n_off=int(NOFFDEF//state_dict['base_n_t']),sub_offpulse_mean=True,fixchans=True,verbose=False)
        
        #mask bad channels if not masked already
        """if len(badchans) > 0:
            m = np.zeros(I.shape)
            m[badchans,:] = 1
            I = ma(I,m)
            Q = ma(Q,m)
            U = ma(U,m)                
            V = ma(V,m)"""


        state_dict['base_I'] = I
        state_dict['base_Q'] = Q
        state_dict['base_U'] = U
        state_dict['base_V'] = V
        state_dict['fobj'] = fobj
        state_dict['base_freq_test'] = freq_test
        state_dict['base_wav_test'] = wav_test
        state_dict['base_time_axis'] = np.arange(I.shape[1])*32.7*state_dict['base_n_t']
        state_dict['badchans'] = badchans



        state_dict['current_state'] += 1

    #if filbutton is clicked, run the offline beamformer to make fil files
    if filbutton.clicked:
        status = polbeamform.make_filterbanks(state_dict['ids'],state_dict['nickname'],state_dict['bfweights'],state_dict['ibeam'],state_dict['mjd'],state_dict['DM0'])
        print("Submitted Job, status: " + str(status))#bfstatus_display.data = status

    #update widget dict
    update_wdict([frbfiles_menu,n_t_slider,logn_f_slider,logibox_slider,buff_L_slider_init,buff_R_slider_init],
                    ["frbfiles_menu","n_t_slider","logn_f_slider","logibox_slider","buff_L_slider_init","buff_R_slider_init"],
                    param='value')
    update_wdict([RA_display,DEC_display,DM_init_display,ibeam_display,mjd_display],
                    ["RA_display","DEC_display","DM_init_display","ibeam_display","mjd_display"],
                    param='data')
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
    tdelays = -DM*4.15*(((np.min(freq_axis)*1e-3)**(-2)) - ((freq_axis*1e-3)**(-2)))#(8.3*(chanbw)*burst_DMs[i]/((freq_axis*1e-3)**3))*(1e-3) #ms
    tdelays_idx_hi = np.array(np.ceil(tdelays/tsamp),dtype=int)
    tdelays_idx_low = np.array(np.floor(tdelays/tsamp),dtype=int)
    tdelays_frac = tdelays/tsamp - tdelays_idx_low
    #print("Trial DM: " + str(DM) + " pc/cc...",end='')#, DM delays (ms): " + str(tdelays) + "...",end='')
    nchans = len(freq_axis)
    #print(dyn_spec.shape)
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
    #print("Done!")

    return dyn_spec_DM



def dedisp_screen(n_t_slider,logn_f_slider,logwindow_slider_init,ddm_num,DM_input_display,DM_new_display,DMdonebutton):
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
    state_dict['window'] = 2**logwindow_slider_init.value
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
    (state_dict['I_t'],state_dict['Q_t'],state_dict['U_t'],state_dict['V_t']) = dsapol.get_stokes_vs_time(state_dict['I'],state_dict['Q'],state_dict['U'],state_dict['V'],state_dict['width_native'],state_dict['fobj'].header.tsamp,state_dict['n_t'],n_off=int(NOFFDEF//state_dict['n_t']),plot=False,show=False,normalize=True,buff=state_dict['buff'],window=30)
    state_dict['time_axis'] = 32.7*state_dict['n_t']*np.arange(0,len(state_dict['I_t']))

    #get timestart, timestop
    (state_dict['peak'],state_dict['timestart'],state_dict['timestop']) = dsapol.find_peak(state_dict['I'],state_dict['width_native'],state_dict['fobj'].header.tsamp,n_t=n_t_slider.value,peak_range=None,pre_calc_tf=False,buff=state_dict['buff'])

    #display dynamic spectrum
    fig, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 2]},figsize=(18,12))
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
    a0.legend(loc="upper right",fontsize=16)
    
    a1.imshow(state_dict['I'][:,state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],aspect='auto',
            extent=[32.7*state_dict['n_t']*state_dict['timestart']*1e-3 - state_dict['window']*32.7*state_dict['n_t']*1e-3,
                32.7*state_dict['n_t']*state_dict['timestop']*1e-3 + state_dict['window']*32.7*state_dict['n_t']*1e-3,
                np.min(state_dict['freq_test'][0]),np.max(state_dict['freq_test'][0])])
    a1.set_xlabel("Time (ms)")
    a1.set_ylabel("Frequency (MHz)")
    plt.subplots_adjust(hspace=0)
    plt.show(fig)


    if DMdonebutton.clicked:# and (state_dict['dDM'] != 0):
        
        if state_dict['dDM'] != 0:
            #dedisperse base dyn spectrum
            state_dict['base_I'] = dedisperse(state_dict['base_I'],state_dict['dDM'],(32.7e-3)*state_dict['base_n_t'],state_dict['base_freq_test'][0])
            state_dict['base_Q'] = dedisperse(state_dict['base_Q'],state_dict['dDM'],(32.7e-3)*state_dict['base_n_t'],state_dict['base_freq_test'][0])
            state_dict['base_U'] = dedisperse(state_dict['base_U'],state_dict['dDM'],(32.7e-3)*state_dict['base_n_t'],state_dict['base_freq_test'][0])
            state_dict['base_V'] = dedisperse(state_dict['base_V'],state_dict['dDM'],(32.7e-3)*state_dict['base_n_t'],state_dict['base_freq_test'][0])
        


        state_dict['current_state'] += 1
    #update widget dict
    update_wdict([n_t_slider,logn_f_slider,logwindow_slider_init,ddm_num],
                ["n_t_slider","logn_f_slider","logwindow_slider_init","ddm_num"],
                param='value')
    
    update_wdict([DM_input_display,DM_new_display],
                ["DM_input_display","DM_new_display"],
                param='data')

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


def polcal_screen(polcaldate_menu,polcaldate_create_menu,polcaldate_bf_menu,polcaldate_findbeams_menu,obsid3C48_menu,obsid3C286_menu,
        polcalbutton,polcopybutton,bfcal_button,findbeams_button,filcalbutton,ParA_display,
        edgefreq_slider,breakfreq_slider,sf_window_weight_cals,sf_order_cals,peakheight_slider,peakwidth_slider,polyfitorder_slider,
        ratio_edgefreq_slider,ratio_breakfreq_slider,ratio_sf_window_weight_cals,ratio_sf_order_cals,ratio_peakheight_slider,ratio_peakwidth_slider,ratio_polyfitorder_slider,
        phase_sf_window_weight_cals,phase_sf_order_cals,phase_peakheight_slider,phase_peakwidth_slider,phase_polyfitorder_slider,savecalsolnbutton,
                                                         sfflag,polyfitflag,ratio_sfflag,ratio_polyfitflag,phase_sfflag,phase_polyfitflag):
    
    """
    This function updates the polarization calibration screen
    whenever the cal file is selected
    """
    
    if polcaldate_menu.value != "":
        #update polcal parameters in state dict
        state_dict['gxx'],state_dict['gyy'],state_dict['cal_freq_axis'] = read_polcal(polcaldate_menu.value)
    state_dict['polcalfile'] = polcaldate_menu.value

    #look for new calibrator files
    vtimes3C48,vfiles3C48 = polcal.get_voltages("3C48")
    vtimes3C286,vfiles3C286 = polcal.get_voltages("3C286")
    mapping3C48 = polcal.iso_voltages(vtimes3C48,vfiles3C48)
    mapping3C286 = polcal.iso_voltages(vtimes3C286,vfiles3C286)
    for k in mapping3C48.keys():
        if len(mapping3C48[k]) == 0: continue
        #find corresponding bf weights
        bfweights3C48 = polcal.get_bfweights(mapping3C48[k],'3C48')
        bfweights3C286 = polcal.get_bfweights(mapping3C286[k],'3C286')
    
        df_polcal.loc[str(k)] = ['<br/>'.join(mapping3C48[k]),
                             '<br/>'.join(bfweights3C48),
                             '<br/>'.join(mapping3C286[k]),
                             '<br/>'.join(bfweights3C286)]
        polcal_dict[str(k)] = dict()
        polcal_dict[str(k)]['3C48'] = mapping3C48[k]
        polcal_dict[str(k)]['3C286'] = mapping3C286[k]
        polcal_dict[str(k)]['3C48_bfweights'] = bfweights3C48
        polcal_dict[str(k)]['3C286_bfweights'] = bfweights3C286


    #update calibrator date
    polcal_dict['polcal_create_file'] = polcaldate_create_menu.value

    #update find beam data
    polcal_dict['polcal_findbeams_file'] = polcaldate_findbeams_menu.value 
    obs_files_3C48,obs_ids_3C48 = polcal.get_calfil_files('3C48',polcaldate_findbeams_menu.value,'3C48*0')
    obs_files_3C286,obs_ids_3C286 = polcal.get_calfil_files('3C286',polcaldate_findbeams_menu.value,'3C286*0')

    #update polcal dict with calibrator files that have voltages, bf weights available
    if polcaldate_bf_menu.value != "":
        polcal_dict['polcal_avail_date'] = polcaldate_bf_menu.value
        polcal_dict['polcal_avail_3C48'],polcal_dict['polcal_avail_bf_3C48'] = polcal.get_all_calfiles(polcaldate_bf_menu.value,'3C48')
        polcal_dict['polcal_avail_3C286'],polcal_dict['polcal_avail_bf_3C286'] = polcal.get_all_calfiles(polcaldate_bf_menu.value,'3C286')
    else:
        polcal_dict['polcal_avail_3C48'],polcal_dict['polcal_avail_3C286'],polcal_dict['polcal_avail_bf_3C48'],polcal_dict['polcal_avail_bf_3C286'] = [],[],[],[]

    if polcalbutton.clicked and (state_dict['polcalfile'] != ""):


        f = open("tmpout.txt","w")
        print("start",file=f)

        #calibrate at native resolution
        state_dict['base_Ical'],state_dict['base_Qcal'],state_dict['base_Ucal'],state_dict['base_Vcal'] = dsapol.calibrate(state_dict['base_I'],state_dict['base_Q'],state_dict['base_U'],state_dict['base_V'],(state_dict['gxx'],state_dict['gyy']),stokes=True)
        print("done calibrating...",file=f)

        #parallactic angle calibration
        state_dict['base_Ical'],state_dict['base_Qcal'],state_dict['base_Ucal'],state_dict['base_Vcal'],state_dict['ParA'] = dsapol.calibrate_angle(state_dict['base_Ical'],state_dict['base_Qcal'],state_dict['base_Ucal'],state_dict['base_Vcal'],state_dict['fobj'],state_dict['ibeam'],state_dict['RA'],state_dict['DEC'])
        ParA_display.data = np.around(state_dict['ParA']*180/np.pi,2)
        print("done ParA calibrating...",file=f)
        #get downsampled versions
        state_dict['Ical'] = dsapol.avg_time(state_dict['base_Ical'],state_dict['rel_n_t'])
        state_dict['Ical'] = dsapol.avg_freq(state_dict['Ical'],state_dict['rel_n_f'])
        state_dict['Qcal'] = dsapol.avg_time(state_dict['base_Qcal'],state_dict['rel_n_t'])
        state_dict['Qcal'] = dsapol.avg_freq(state_dict['Qcal'],state_dict['rel_n_f'])
        state_dict['Ucal'] = dsapol.avg_time(state_dict['base_Ucal'],state_dict['rel_n_t'])
        state_dict['Ucal'] = dsapol.avg_freq(state_dict['Ucal'],state_dict['rel_n_f'])
        state_dict['Vcal'] = dsapol.avg_time(state_dict['base_Vcal'],state_dict['rel_n_t'])
        state_dict['Vcal'] = dsapol.avg_freq(state_dict['Vcal'],state_dict['rel_n_f'])
        print("done downsampling...",file=f)
        f.close()
        #get time series
        (state_dict['I_tcal'],state_dict['Q_tcal'],state_dict['U_tcal'],state_dict['V_tcal']) = dsapol.get_stokes_vs_time(state_dict['Ical'],state_dict['Qcal'],state_dict['Ucal'],state_dict['Vcal'],state_dict['width_native'],state_dict['fobj'].header.tsamp,state_dict['n_t'],n_off=int(NOFFDEF//state_dict['n_t']),plot=False,show=False,normalize=True,buff=state_dict['buff'],window=30)
        state_dict['time_axis'] = 32.7*state_dict['n_t']*np.arange(0,len(state_dict['I_tcal']))

        #get timestart, timestop
        (state_dict['peak'],state_dict['timestart'],state_dict['timestop']) = dsapol.find_peak(state_dict['Ical'],state_dict['width_native'],state_dict['fobj'].header.tsamp,n_t=state_dict['rel_n_t'],peak_range=None,pre_calc_tf=False,buff=state_dict['buff'])

        state_dict['current_state'] += 1

    #if copy button clicked, copy voltages from T3
    if polcopybutton.clicked:

        #copy voltages from T3 to scratch dir
        polcal.copy_voltages(polcal_dict[polcal_dict['polcal_create_file']]['3C48'],polcal_dict['polcal_create_file'],'3C48')
        polcal.copy_voltages(polcal_dict[polcal_dict['polcal_create_file']]['3C286'],polcal_dict['polcal_create_file'],'3C286')

        #copy beamformer weights from generated directory to scratch dir
        polcal.copy_bfweights(polcal_dict[polcal_dict['polcal_create_file']]['3C48_bfweights'])
        polcal.copy_bfweights(polcal_dict[polcal_dict['polcal_create_file']]['3C286_bfweights'])
    
    
    #if bfcal button clicked beamform calibration files at low res
    if bfcal_button.clicked:
        
        #beamform 3C48 and 3C286
        polcal.beamform_polcal(polcal_dict['polcal_avail_3C48'],polcal_dict['polcal_avail_bf_3C48'],'3C48',polcal_dict['polcal_avail_date'])
        polcal.beamform_polcal(polcal_dict['polcal_avail_3C286'],polcal_dict['polcal_avail_bf_3C286'],'3C286',polcal_dict['polcal_avail_date'])
    
    #if findbeam button clicked find the beams for each cal observation
    beam_dict_3C48 = dict()
    beam_dict_3C286 = dict()
    if findbeams_button.clicked:

        
        #run beam finder in the background
        #print("python " + repo_path + "/scripts/find_beams.py " + str('3C48') + " " + str(polcal_dict['polcal_findbeams_file']) + " > " + polcal.logfile + " 2>&1 &")
        os.system("python " + repo_path + "/scripts/find_beams.py " + str('3C48') + " " + str(polcal_dict['polcal_findbeams_file']) + " > " + polcal.logfile + " 2>&1 &")
        #print("python " + repo_path + "/scripts/find_beams.py " + str('3C286') + " " + str(polcal_dict['polcal_findbeams_file']) + " > " + polcal.logfile + " 2>&1 &")
        os.system("python " + repo_path + "/scripts/find_beams.py " + str('3C286') + " " + str(polcal_dict['polcal_findbeams_file']) + " > " + polcal.logfile + " 2>&1 &")

        
    #look to see if beam dicts have been saved 
    try:
        fs = os.listdir(polcal.output_path + "3C48_" + polcal_dict['polcal_findbeams_file'] + "/")
    except:
        fs = []
    if "3C48_" + polcal_dict['polcal_findbeams_file'] + "_beams.pkl" in fs:
        f = open(polcal.output_path + "3C48_" + polcal_dict['polcal_findbeams_file'] + "/3C48_" + polcal_dict['polcal_findbeams_file'] + "_beams.pkl","rb")
        beam_dict_3C48 = pkl.load(f)
        f.close()

    try:
        fs = os.listdir(polcal.output_path + "3C286_" + polcal_dict['polcal_findbeams_file'] + "/")
    except:
        fs = []
    if "3C286_" + polcal_dict['polcal_findbeams_file'] + "_beams.pkl" in fs:
        f = open(polcal.output_path + "3C286_" + polcal_dict['polcal_findbeams_file'] + "/3C286_" + polcal_dict['polcal_findbeams_file'] + "_beams.pkl","rb")
        beam_dict_3C286 = pkl.load(f)
        f.close()

    """
    #add best beams to dict
    if len(beam_dict_3C48.keys()) > 0:
        polcal_dict['cal_name_3C48_center'],polcal_dict['cal_beam_3C48_center'] = polcal.get_best_beam(beam_dict_3C48)
    if len(beam_dict_3C286.keys()) > 0:
        polcal_dict['cal_name_3C286_center'],polcal_dict['cal_beam_3C286_center'] = polcal.get_best_beam(beam_dict_3C286)
    """

    #but use the obs selected from the menu
    polcal_dict['cal_name_3C48'] = obsid3C48_menu.value
    polcal_dict['cal_name_3C286'] = obsid3C286_menu.value
    if polcal_dict['cal_name_3C48'] != '':
        polcal_dict['cal_beam_3C48'] = beam_dict_3C48[polcal_dict['cal_name_3C48']]['beam']
    else:
        polcal_dict['cal_beam_3C48'] = np.nan
    if polcal_dict['cal_name_3C286'] != '':
        polcal_dict['cal_beam_3C286'] = beam_dict_3C286[polcal_dict['cal_name_3C286']]['beam']
    else:
        polcal_dict['cal_beam_3C286'] = np.nan


    #update widget dict
    update_wdict([polcaldate_menu,polcaldate_create_menu,polcaldate_bf_menu,polcaldate_findbeams_menu,obsid3C48_menu,obsid3C286_menu,
                edgefreq_slider,breakfreq_slider,sf_window_weight_cals,sf_order_cals,peakheight_slider,peakwidth_slider,polyfitorder_slider,
                ratio_edgefreq_slider,ratio_breakfreq_slider,ratio_sf_window_weight_cals,ratio_sf_order_cals,ratio_peakheight_slider,
                ratio_peakwidth_slider,ratio_polyfitorder_slider,phase_sf_window_weight_cals,phase_sf_order_cals,phase_peakheight_slider,
                phase_peakwidth_slider,phase_polyfitorder_slider,sfflag,polyfitflag,ratio_sfflag,ratio_polyfitflag,phase_sfflag,phase_polyfitflag],
                ["polcaldate_menu","polcaldate_create_menu","polcaldate_bf_menu","polcaldate_findbeams_menu","obsid3C48_menu","obsid3C286_menu",
                "edgefreq_slider","breakfreq_slider","sf_window_weight_cals","sf_order_cals","peakheight_slider","peakwidth_slider","polyfitorder_slider",
                "ratio_edgefreq_slider","ratio_breakfreq_slider","ratio_sf_window_weight_cals","ratio_sf_order_cals","ratio_peakheight_slider",
                "ratio_peakwidth_slider","ratio_polyfitorder_slider","phase_sf_window_weight_cals","phase_sf_order_cals","phase_peakheight_slider",
                "phase_peakwidth_slider","phase_polyfitorder_slider","sfflag","polyfitflag","ratio_sfflag","ratio_polyfitflag","phase_sfflag","phase_polyfitflag"],
                param='value')
    update_wdict([ParA_display],
                ["ParA_display"],
                param='data')

   

    #plot
    plt.figure(figsize=(20,9))
    plt.subplot(121)
    for k in beam_dict_3C48.keys():
        df_beams.loc[str(k)] = [beam_dict_3C48[k]['beam'],beam_dict_3C48[k]['mjd'],beam_dict_3C48[k]['bf_weights']]
        if str(k) == polcal_dict['cal_name_3C48']:
            plt.plot(beam_dict_3C48[k]['beamspectrum'],label=k,linewidth=5)
        else:
            plt.plot(beam_dict_3C48[k]['beamspectrum'],label=k)
    plt.legend(loc='upper left')
    plt.title('3C48 ' + str(polcal_dict['polcal_findbeams_file']))
    plt.xlabel("Beam Number")
    plt.axvline(polcal.middle_beam,color='black')

    plt.subplot(122)
    for k in beam_dict_3C286.keys():
        df_beams.loc[str(k)] = [beam_dict_3C286[k]['beam'],beam_dict_3C286[k]['mjd'],beam_dict_3C286[k]['bf_weights']]
        if str(k) == polcal_dict['cal_name_3C286']:
            plt.plot(beam_dict_3C286[k]['beamspectrum'],label=k,linewidth=5)
        else:
            plt.plot(beam_dict_3C286[k]['beamspectrum'],label=k)
    plt.legend(loc='upper left')
    plt.title('3C286 ' + str(polcal_dict['polcal_findbeams_file']))
    plt.yticks([])
    plt.xlabel("Beam Number")
    plt.axvline(polcal.middle_beam,color='black')


    plt.subplots_adjust(wspace=0)
    plt.show()

    return beam_dict_3C48,beam_dict_3C286


def polcal_screen2(polcaldate_menu,polcaldate_create_menu,polcaldate_bf_menu,polcaldate_findbeams_menu,obsid3C48_menu,obsid3C286_menu,
        polcalbutton,polcopybutton,bfcal_button,findbeams_button,filcalbutton,ParA_display,
        edgefreq_slider,breakfreq_slider,sf_window_weight_cals,sf_order_cals,peakheight_slider,peakwidth_slider,polyfitorder_slider,
        ratio_edgefreq_slider,ratio_breakfreq_slider,ratio_sf_window_weight_cals,ratio_sf_order_cals,ratio_peakheight_slider,ratio_peakwidth_slider,ratio_polyfitorder_slider,
        phase_sf_window_weight_cals,phase_sf_order_cals,phase_peakheight_slider,phase_peakwidth_slider,phase_polyfitorder_slider,savecalsolnbutton,
                                                         sfflag,polyfitflag,ratio_sfflag,ratio_polyfitflag,phase_sfflag,phase_polyfitflag,beam_dict_3C48,beam_dict_3C286):


    #if make filt button pushed, make filterbanks for pol cals
    if filcalbutton.clicked:

        #make 3C48 filterbanks
        for k in beam_dict_3C48.keys():
            make_cal_filterbanks('3C48',polcal_dict['polcal_findbeams_file'],str(k)[4:],beam_dict_3C48[k]['bf_weights'],beam_dict_3C48[k]['beam'],beam_dict_3C48[k]['mjd'])

        #make 3C286 filterbanks
        for k in beam_dict_3C286.keys():
            make_cal_filterbanks('3C286',polcal_dict['polcal_findbeams_file'],str(k)[5:],beam_dict_3C286[k]['bf_weights'],beam_dict_3C286[k]['beam'],beam_dict_3C286[k]['mjd'])


    #if make solution button pushed, make solution and plot
    if ((polcal_dict['cal_name_3C48'] != "" or polcal_dict['cal_name_3C286'] != "") and polcal_dict['polcal_findbeams_file'] != "") or state_dict['polcalfile']:
        fig = plt.figure(figsize=(18,12))
        
        plt.subplot(311)
        plt.ylabel(r'$|g_{yy}|$')
        plt.xticks([])

        plt.subplot(312)
        plt.ylabel(r'$|g_{xx}|/|g_{yy}|$')
        plt.xticks([])

        plt.subplot(313)
        plt.ylabel(r'$\angle g_{xx} - \angle g_{yy}$')
        plt.xlabel(r'Frequency (MHz)')

        plt.subplots_adjust(hspace=0)
    
    if (polcal_dict['cal_name_3C48'] != "" or polcal_dict['cal_name_3C286'] != "") and polcal_dict['polcal_findbeams_file'] != "":
        
        #get previous cal solution
        last_caldate,last_calobs1,last_calobs2,last_calnum = polcal.get_last_calmeta()
        last_calnum =int(last_calnum)

        #read cal solution
        last_gxx,last_gyy,last_cal_freq_axis = read_polcal('POLCAL_PARAMETERS_' + last_caldate + '.csv')#,fit=False)
        #last_gxx_fit,last_gyy_fit,last_cal_freq_axis_fit = read_polcal('POLCAL_PARAMETERS_' + last_caldate + '.csv') 
        
        plt.subplot(311)
        plt.plot(last_cal_freq_axis,np.abs(last_gyy),color='grey',label='Old Soln (' + last_caldate + ')')
        #plt.plot(last_cal_freq_axis_fit,np.abs(last_gyy_fit),color='black',label='Old Fit Soln (' + last_caldate + ')')
        plt.ylabel(r'$|g_{yy}|$')
        plt.xticks([])

        plt.subplot(312)
        plt.plot(last_cal_freq_axis,np.abs(last_gxx)/np.abs(last_gyy),color='grey',label='Old Soln (' + last_caldate + ')')
        #plt.plot(last_cal_freq_axis_fit,np.abs(last_gxx_fit)/np.abs(last_gyy_fit),color='black',label='Old Fit Soln (' + last_caldate + ')')
        plt.ylabel(r'$|g_{xx}|/|g_{yy}|$')
        plt.xticks([])

        plt.subplot(313)
        plt.plot(last_cal_freq_axis,np.angle(last_gxx) - np.angle(last_gyy),color='grey',label='Old Soln (' + last_caldate + ')')
        #plt.plot(last_cal_freq_axis_fit,np.angle(last_gxx_fit) - np.angle(last_gyy_fit),color='black',label='Old Fit Soln (' + last_caldate + ')')
        plt.ylabel(r'$\angle g_{xx} - \angle g_{yy}$')
        plt.xlabel(r'Frequency (MHz)')

        plt.subplots_adjust(hspace=0)
    if polcal_dict['cal_name_3C48'] != "" and polcal_dict['polcal_findbeams_file'] != "":
        #make abs gain solution
        GY_fit,GY_fit_sf, GY_fullres,GY_fullres_sf,GY_fullres_i,freq_test = polcal.abs_gyy_solution(np.abs(last_gyy),last_calnum,polcal_dict['polcal_findbeams_file'],polcal_dict['cal_name_3C48'],beam_dict_3C48[polcal_dict['cal_name_3C48']]['beam'],
                                                            edgefreq=edgefreq_slider.value,
                                                            breakfreq=breakfreq_slider.value,
                                                            sf_window_weights=sf_window_weight_cals.value,sf_order=sf_order_cals.value,peakheight=peakheight_slider.value,
                                                            padwidth=peakwidth_slider.value,deg=polyfitorder_slider.value,sfflag=sfflag.value,polyfitflag=polyfitflag.value)
        polcal_dict['GY'] = GY_fullres
        polcal_dict['GY_fit'] = GY_fit
        polcal_dict['GY_fit_sf'] = GY_fit_sf
        polcal_dict['GY_sf'] = GY_fullres_sf
        polcal_dict['cal_freq_axis'] = freq_test[0]
        
        #make gain ratio solution
        ratio_fit,ratio_fit_sf,ratio_fullres,ratio_fullres_sf,ratio_fullres_i,freq_test = polcal.gain_solution(np.abs(last_gxx)/np.abs(last_gyy),last_calnum,polcal_dict['polcal_findbeams_file'],polcal_dict['cal_name_3C48'],beam_dict_3C48[polcal_dict['cal_name_3C48']]['beam'],
                                                            edgefreq=ratio_edgefreq_slider.value,
                                                            breakfreq=ratio_breakfreq_slider.value,
                                                            sf_window_weights=ratio_sf_window_weight_cals.value,sf_order=ratio_sf_order_cals.value,peakheight=ratio_peakheight_slider.value,
                                                            padwidth=ratio_peakwidth_slider.value,deg=ratio_polyfitorder_slider.value,sfflag=ratio_sfflag.value,polyfitflag=ratio_polyfitflag.value)
        polcal_dict['ratio'] = ratio_fullres
        polcal_dict['ratio_fit'] = ratio_fit
        polcal_dict['ratio_fit_sf'] = ratio_fit_sf
        polcal_dict['ratio_sf'] = ratio_fullres_sf
        polcal_dict['cal_freq_axis'] = freq_test[0]

        plt.subplot(311)
        plt.plot(freq_test[0],GY_fullres_i,label='New Soln')
        lw=1
        if (not polyfitflag.value) and (not sfflag.value): lw = 3
        c=plt.plot(freq_test[0],GY_fullres,label='Mean Soln',lw=lw)
        lw=1
        if (not polyfitflag.value) and (sfflag.value): lw = 3
        plt.plot(freq_test[0],GY_fullres_sf,label='Mean SF Soln',lw=lw)#,color=c[0].get_color(),linestyle='--',lw=lw)
        lw=1
        if (polyfitflag.value) and (not sfflag.value): lw = 3
        c=plt.plot(freq_test[0],GY_fit,label='Mean Fit Soln',lw=lw)
        lw=1
        if (polyfitflag.value) and (sfflag.value): lw = 3
        plt.plot(freq_test[0],GY_fit_sf,label='Mean Fit SF Soln',lw=lw)#,color=c[0].get_color(),linestyle='--',lw=lw)
        plt.ylabel(r'$|g_{yy}|$')
        plt.xticks([])
        plt.axvline(edgefreq_slider.value,color='black',linewidth=3)
        plt.axvline(breakfreq_slider.value,color='red',linewidth=3)
        plt.legend(loc='upper left',fontsize=20,frameon=True,ncol=3)
        
        plt.subplot(312)
        plt.plot(freq_test[0],ratio_fullres_i,label='New Soln')
        lw=1
        if (not ratio_polyfitflag.value) and (not ratio_sfflag.value): lw = 3
        c=plt.plot(freq_test[0],ratio_fullres,label='Mean Soln',lw=lw)
        lw=1
        if (not ratio_polyfitflag.value) and (ratio_sfflag.value): lw = 3
        plt.plot(freq_test[0],ratio_fullres_sf,label='Mean SF Soln',lw=lw)#,color=c[0].get_color(),linestyle='--',lw=lw)
        lw=1
        if (ratio_polyfitflag.value) and (not ratio_sfflag.value): lw = 3
        c=plt.plot(freq_test[0],ratio_fit,label='Mean Fit Soln',lw=lw)
        lw=1
        if (ratio_polyfitflag.value) and (ratio_sfflag.value): lw = 3
        plt.plot(freq_test[0],ratio_fit_sf,label='Mean Fit SF Soln',lw=lw)#,color=c[0].get_color(),linestyle='--',lw=lw)
        plt.ylabel(r'$|g_{xx}|/|g_{yy}|$')
        plt.xticks([])
        plt.axvline(ratio_edgefreq_slider.value,color='black',linewidth=3)
        plt.axvline(ratio_breakfreq_slider.value,color='red',linewidth=3)

    if polcal_dict['cal_name_3C286'] != "" and polcal_dict['polcal_findbeams_file'] != "":

        #make phase diff solution
        phase_fit,phase_fit_sf,phase_fullres,phase_fullres_sf,phase_fullres_i,freq_test = polcal.phase_solution(np.angle(last_gxx)-np.angle(last_gyy),last_calnum,polcal_dict['polcal_findbeams_file'],polcal_dict['cal_name_3C286'],beam_dict_3C286[polcal_dict['cal_name_3C286']]['beam'],
                                                            sf_window_weights=phase_sf_window_weight_cals.value,sf_order=phase_sf_order_cals.value,peakheight=phase_peakheight_slider.value,
                                                            padwidth=phase_peakwidth_slider.value,deg=phase_polyfitorder_slider.value,sfflag=phase_sfflag.value,polyfitflag=phase_polyfitflag.value)
        polcal_dict['phase'] = phase_fullres
        polcal_dict['phase_fit_sf'] = phase_fit_sf
        polcal_dict['phase_fit'] = phase_fit
        polcal_dict['phase_sf'] = phase_fullres_sf
        polcal_dict['cal_freq_axis'] = freq_test[0]

        plt.subplot(313)
        plt.plot(freq_test[0],phase_fullres_i,label='New Soln')
        lw=1
        if (not phase_polyfitflag.value) and (not phase_sfflag.value): lw = 3
        c=plt.plot(freq_test[0],phase_fullres,label='Mean Soln',lw=lw)
        lw=1
        if (not phase_polyfitflag.value) and (phase_sfflag.value): lw = 3
        plt.plot(freq_test[0],phase_fullres_sf,label='Mean SF Soln',lw=lw)#,color=c[0].get_color(),linestyle='--',lw=lw)
        lw=1
        if (phase_polyfitflag.value) and (not phase_sfflag.value): lw = 3
        c=plt.plot(freq_test[0],phase_fit,label='Mean Fit Soln',lw=lw)
        lw=1
        if (phase_polyfitflag.value) and (phase_sfflag.value): lw = 3
        plt.plot(freq_test[0],phase_fit_sf,label='Mean Fit SF Soln',lw=lw)#,color=c[0].get_color(),linestyle='--',lw=lw)
    #plt.show()


    #if save button clicked, write to file
    if savecalsolnbutton.clicked:
        if polyfitflag.value and sfflag.value: GY_USE=polcal_dict['GY_fit_sf']
        elif polyfitflag.value: GY_USE = polcal_dict['GY_fit']
        elif sfflag.value: GY_USE = polcal_dict['GY_sf'] 
        else: GY_USE = polcal_dict['GY']

        if ratio_polyfitflag.value and ratio_sfflag.value: ratio_USE = polcal_dict['ratio_fit_sf']
        elif ratio_polyfitflag.value: ratio_USE = polcal_dict['ratio_fit']
        elif ratio_sfflag.value: ratio_USE = polcal_dict['ratio_sf']
        else: ratio_USE = polcal_dict['ratio']

        if phase_polyfitflag.value and phase_sfflag.value: phase_USE = polcal_dict['phase_fit_sf']
        elif phase_polyfitflag.value: phase_USE = polcal_dict['phase_fit']
        elif phase_sfflag.value: phase_USE = polcal_dict['phase_sf']
        else: phase_USE = polcal_dict['phase']

        polcal_dict['gxx'],polcal_dict['gyy'] = dsapol.get_calmatrix_from_ratio_phasediff(ratio_USE,phase_USE,GY_USE)
        polcal_dict['new_cal_file'] = polcal.write_polcal_solution(polcal_dict['cal_name_3C48'],polcal_dict['cal_name_3C286'],last_calnum,
                                    polcal_dict['ratio'],polcal_dict['ratio_fit'],polcal_dict['ratio_sf'],polcal_dict['ratio_fit_sf'],
                                    polcal_dict['phase'],polcal_dict['phase_fit'],polcal_dict['phase_sf'],polcal_dict['phase_fit_sf'],
                                    polcal_dict['GY'],polcal_dict['GY_fit'],polcal_dict['GY_sf'],polcal_dict['GY_fit_sf'],
                                    polcal_dict['gxx'],polcal_dict['gyy'],polcal_dict['cal_freq_axis'])
        print(polcal_dict['new_cal_file'])
        
    #if using current cal solution, display
    #fig=plt.figure(figsize=(18,14))
    if polcaldate_menu.value != "":
        plt.subplot(312)
        #plt.xticks([])
        #plt.ylabel(r'$|g_{xx}|/|g_{yy}|$')
        plt.plot(state_dict['cal_freq_axis'],np.abs(state_dict['gxx'])/np.abs(state_dict['gyy']),color='magenta',linewidth=4)

        plt.subplot(313)
        #plt.ylabel(r'$\angle g_{xx} - \angle g_{yy}$')
        plt.plot(state_dict['cal_freq_axis'],np.angle(state_dict['gxx'])-np.angle(state_dict['gyy']),color='magenta',linewidth=4)
        #plt.xlabel("Frequency (MHz)")

        plt.subplot(311)
        plt.title(state_dict['polcalfile'])
        #plt.xticks([])
        #plt.ylabel(r'$|g_{yy}|$')
        plt.plot(state_dict['cal_freq_axis'],np.abs(state_dict['gyy']),color='magenta',linewidth=4)
    if ((polcal_dict['cal_name_3C48'] != "" or polcal_dict['cal_name_3C286'] != "") and polcal_dict['polcal_findbeams_file'] != "") or state_dict['polcalfile']:
        plt.subplots_adjust(hspace=0)
        plt.close()



    #update widget dict
    update_wdict([polcaldate_menu,polcaldate_create_menu,polcaldate_bf_menu,polcaldate_findbeams_menu,obsid3C48_menu,obsid3C286_menu,
                edgefreq_slider,breakfreq_slider,sf_window_weight_cals,sf_order_cals,peakheight_slider,peakwidth_slider,polyfitorder_slider,
                ratio_edgefreq_slider,ratio_breakfreq_slider,ratio_sf_window_weight_cals,ratio_sf_order_cals,ratio_peakheight_slider,
                ratio_peakwidth_slider,ratio_polyfitorder_slider,phase_sf_window_weight_cals,phase_sf_order_cals,phase_peakheight_slider,
                phase_peakwidth_slider,phase_polyfitorder_slider,sfflag,polyfitflag,ratio_sfflag,ratio_polyfitflag,phase_sfflag,phase_polyfitflag],
                ["polcaldate_menu","polcaldate_create_menu","polcaldate_bf_menu","polcaldate_findbeams_menu","obsid3C48_menu","obsid3C286_menu",
                "edgefreq_slider","breakfreq_slider","sf_window_weight_cals","sf_order_cals","peakheight_slider","peakwidth_slider","polyfitorder_slider",
                "ratio_edgefreq_slider","ratio_breakfreq_slider","ratio_sf_window_weight_cals","ratio_sf_order_cals","ratio_peakheight_slider",
                "ratio_peakwidth_slider","ratio_polyfitorder_slider","phase_sf_window_weight_cals","phase_sf_order_cals","phase_peakheight_slider",
                "phase_peakwidth_slider","phase_polyfitorder_slider","sfflag","polyfitflag","ratio_sfflag","ratio_polyfitflag","phase_sfflag","phase_polyfitflag"],
                param='value')
    update_wdict([ParA_display],
                ["ParA_display"],
                param='data')
    
    if ((polcal_dict['cal_name_3C48'] != "" or polcal_dict['cal_name_3C286'] != "") and polcal_dict['polcal_findbeams_file'] != "") or state_dict['polcalfile']:
        return fig
    
    return #beam_dict_3C48,beam_dict_3C286 #return these to prevent recalculating the beamformer weights isot


"""
Filter Weights State
"""

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
        

def filter_screen(logwindow_slider,logibox_slider,buff_L_slider,buff_R_slider,ncomps_num,comprange_slider,nextcompbutton,donecompbutton,avger_w_slider,sf_window_weights_slider):
    

    if nextcompbutton.clicked:

        #increment the component and reset bounds
        if state_dict['current_comp'] < state_dict['n_comps']-1:
            state_dict['current_comp'] += 1
        else: print('Done with components')

    #first check if resolution was changed
    state_dict['window'] = 2**logwindow_slider.value
    """
    if (n_t_slider_filt.value != state_dict['rel_n_t']) or (2**logn_f_slider_filt.value != state_dict['rel_n_f']): 
        state_dict['rel_n_t'] = n_t_slider_filt.value
        state_dict['rel_n_f'] = (2**logn_f_slider_filt.value)
        state_dict['n_t'] = n_t_slider_filt.value*state_dict['base_n_t']
        state_dict['n_f'] = (2**logn_f_slider_filt.value)*state_dict['base_n_f']
        state_dict['freq_test'] = [state_dict['base_freq_test'][0].reshape(len(state_dict['base_freq_test'][0])//(state_dict['rel_n_f']),(state_dict['rel_n_f'])).mean(1)]*4
        state_dict['I'] = dsapol.avg_time(state_dict['base_I'],state_dict['rel_n_t'])
        state_dict['I'] = dsapol.avg_freq(state_dict['I'],state_dict['rel_n_f'])
        state_dict['Q'] = dsapol.avg_time(state_dict['base_Q'],state_dict['rel_n_t'])
        state_dict['Q'] = dsapol.avg_freq(state_dict['Q'],state_dict['rel_n_f'])
        state_dict['U'] = dsapol.avg_time(state_dict['base_U'],state_dict['rel_n_t'])
        state_dict['U'] = dsapol.avg_freq(state_dict['U'],state_dict['rel_n_f'])
        state_dict['V'] = dsapol.avg_time(state_dict['base_V'],state_dict['rel_n_t'])
        state_dict['V'] = dsapol.avg_freq(state_dict['V'],state_dict['rel_n_f'])

        state_dict['Ical'] = dsapol.avg_time(state_dict['base_Ical'],state_dict['rel_n_t'])
        state_dict['Ical'] = dsapol.avg_freq(state_dict['Ical'],state_dict['rel_n_f'])
        state_dict['Qcal'] = dsapol.avg_time(state_dict['base_Qcal'],state_dict['rel_n_t'])
        state_dict['Qcal'] = dsapol.avg_freq(state_dict['Qcal'],state_dict['rel_n_f'])
        state_dict['Ucal'] = dsapol.avg_time(state_dict['base_Ucal'],state_dict['rel_n_t'])
        state_dict['Ucal'] = dsapol.avg_freq(state_dict['Ucal'],state_dict['rel_n_f'])
        state_dict['Vcal'] = dsapol.avg_time(state_dict['base_Vcal'],state_dict['rel_n_t'])
        state_dict['Vcal'] = dsapol.avg_freq(state_dict['Vcal'],state_dict['rel_n_f'])
    """

    #mask the current and previous component
    state_dict['n_comps'] = int(ncomps_num.value)
    state_dict['comps'][state_dict['current_comp']] = dict()
    Ip = copy.deepcopy(state_dict['Ical'])
    Qp = copy.deepcopy(state_dict['Qcal'])
    Up = copy.deepcopy(state_dict['Ucal'])
    Vp = copy.deepcopy(state_dict['Vcal'])
    for k in range(state_dict['current_comp']):
        mask = np.zeros(state_dict['Ical'].shape)
        mask[:,state_dict['comps'][k]['left_lim']:state_dict['comps'][k]['right_lim']] = 1
        Ip = ma(Ip,mask)
        Qp = ma(Qp,mask)
        Up = ma(Up,mask)
        Vp = ma(Vp,mask)
    n_off=int(NOFFDEF//state_dict['n_t'])

    
    #get weights for the current component
    state_dict['comps'][state_dict['current_comp']]['width_native'] = 2**logibox_slider.value
    state_dict['comps'][state_dict['current_comp']]['buff'] = [buff_L_slider.value,buff_R_slider.value]
    state_dict['comps'][state_dict['current_comp']]['avger_w'] = avger_w_slider.value
    state_dict['comps'][state_dict['current_comp']]['sf_window_weights'] = sf_window_weights_slider.value
    (state_dict['comps'][state_dict['current_comp']]['peak'],
    state_dict['comps'][state_dict['current_comp']]['timestart'],
    state_dict['comps'][state_dict['current_comp']]['timestop']) = dsapol.find_peak(Ip,state_dict['comps'][state_dict['current_comp']]['width_native'],
                                                                                    state_dict['fobj'].header.tsamp,n_t=state_dict['n_t'],
                                                                                    peak_range=None,pre_calc_tf=False,
                                                                                    buff=state_dict['comps'][state_dict['current_comp']]['buff'])
    state_dict['comps'][state_dict['current_comp']]['left_lim'] = np.argmin(np.abs(state_dict['time_axis']*1e-3 - (state_dict['comps'][state_dict['current_comp']]['timestart']*state_dict['n_t']*32.7e-3 - state_dict['window']*state_dict['n_t']*32.7e-3 + comprange_slider.value[0])))
    state_dict['comps'][state_dict['current_comp']]['right_lim'] = np.argmin(np.abs(state_dict['time_axis']*1e-3 - (state_dict['comps'][state_dict['current_comp']]['timestart']*state_dict['n_t']*32.7e-3 - state_dict['window']*state_dict['n_t']*32.7e-3 + comprange_slider.value[1])))

    (I_tcal,Q_tcal,U_tcal,V_tcal) = dsapol.get_stokes_vs_time(Ip,Qp,Up,Vp,state_dict['comps'][state_dict['current_comp']]['width_native'],
                                                state_dict['fobj'].header.tsamp,state_dict['n_t'],n_off=int(NOFFDEF//state_dict['n_t']),
                                                plot=False,show=False,normalize=True,buff=state_dict['comps'][state_dict['current_comp']]['buff'],window=30)

    state_dict['comps'][state_dict['current_comp']]['weights'] = dsapol.get_weights_1D(I_tcal,Q_tcal,U_tcal,V_tcal,
                                                                                state_dict['comps'][state_dict['current_comp']]['timestart'],
                                                                                state_dict['comps'][state_dict['current_comp']]['timestop'],
                                                                                state_dict['comps'][state_dict['current_comp']]['width_native'],
                                                                                state_dict['fobj'].header.tsamp,1,state_dict['n_t'],
                                                                                state_dict['freq_test'],state_dict['time_axis'],state_dict['fobj'],
                                                                                n_off=int(NOFFDEF//state_dict['n_t']),buff=state_dict['buff'],
                                                                                n_t_weight=state_dict['comps'][state_dict['current_comp']]['avger_w'],
                                                                sf_window_weights=state_dict['comps'][state_dict['current_comp']]['sf_window_weights'],
                                                                padded=True,norm=False)
    


    #initialize RM dict
    state_dict['comps'][state_dict['current_comp']]['RMcalibrated'] = dict()
    state_dict['comps'][state_dict['current_comp']]['RMcalibrated']['RMsnrs1'] = np.nan*np.ones(int(wdict['nRM_num']))
    state_dict['comps'][state_dict['current_comp']]['RMcalibrated']['RM_tools_snrs'] = np.nan*np.ones(int(2*wdict['maxRM_num']/wdict['dRM_tools']))
    state_dict['comps'][state_dict['current_comp']]['RMcalibrated']['RMsnrs1zoom'] = np.nan*np.ones(int(wdict['nRM_num_zoom']))
    state_dict['comps'][state_dict['current_comp']]['RMcalibrated']['RM_tools_snrszoom'] = np.nan*np.ones(int(2*wdict['RM_window_zoom']/wdict['dRM_tools_zoom']))
    state_dict['comps'][state_dict['current_comp']]['RMcalibrated']['RMsnrs2'] = np.nan*np.ones(int(wdict['nRM_num_zoom']))
    state_dict['comps'][state_dict['current_comp']]['RMcalibrated']["RM2"] = [np.nan,np.nan]
    state_dict['comps'][state_dict['current_comp']]["RMcalibrated"]["RMerrfit"] = np.nan
    state_dict['comps'][state_dict['current_comp']]["RMcalibrated"]["trial_RM1"] = np.linspace(wdict['minRM_num'],wdict['maxRM_num'],int(wdict['nRM_num']))
    state_dict['comps'][state_dict['current_comp']]["RMcalibrated"]["trial_RM2"] = np.linspace(-wdict['RM_window_zoom'],wdict['RM_window_zoom'],int(wdict['nRM_num_zoom']))
    state_dict['comps'][state_dict['current_comp']]["RMcalibrated"]["trial_RM_tools"] = np.arange(-wdict['maxRM_num'],wdict['maxRM_num'],wdict['dRM_tools'])
    state_dict['comps'][state_dict['current_comp']]["RMcalibrated"]["trial_RM_toolszoom"] = np.arange(-wdict['RM_window_zoom'],wdict['RM_window_zoom'],wdict['dRM_tools_zoom'])


    #compute S/N and display
    state_dict['comps'][state_dict['current_comp']]['S/N'] = get_SNR(I_tcal,state_dict['comps'][state_dict['current_comp']]['weights'],
                                                                    state_dict['comps'][state_dict['current_comp']]['timestart'],
                                                                    state_dict['comps'][state_dict['current_comp']]['timestop'])
    
    
  
    df.loc[str(state_dict['current_comp'])] = [state_dict['comps'][state_dict['current_comp']]['buff'][0],
                                                   state_dict['comps'][state_dict['current_comp']]['buff'][1],
                                                   state_dict['comps'][state_dict['current_comp']]['avger_w'],
                                                   state_dict['comps'][state_dict['current_comp']]['sf_window_weights'],
                                                   state_dict['comps'][state_dict['current_comp']]['left_lim'],
                                                   state_dict['comps'][state_dict['current_comp']]['right_lim'],
                                                   state_dict['comps'][state_dict['current_comp']]['S/N']]

    
    #display masked dynamic spectrum
    fig, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 2]},figsize=(18,12))
    a0.step(state_dict['time_axis'][state_dict['comps'][state_dict['current_comp']]['timestart']-state_dict['window']:state_dict['comps'][state_dict['current_comp']]['timestop']+state_dict['window']]*1e-3,
            I_tcal[state_dict['comps'][state_dict['current_comp']]['timestart']-state_dict['window']:state_dict['comps'][state_dict['current_comp']]['timestop']+state_dict['window']],label='I')
    a0.step(state_dict['time_axis'][state_dict['comps'][state_dict['current_comp']]['timestart']-state_dict['window']:state_dict['comps'][state_dict['current_comp']]['timestop']+state_dict['window']]*1e-3,
            Q_tcal[state_dict['comps'][state_dict['current_comp']]['timestart']-state_dict['window']:state_dict['comps'][state_dict['current_comp']]['timestop']+state_dict['window']],label='Q')
    a0.step(state_dict['time_axis'][state_dict['comps'][state_dict['current_comp']]['timestart']-state_dict['window']:state_dict['comps'][state_dict['current_comp']]['timestop']+state_dict['window']]*1e-3,
            U_tcal[state_dict['comps'][state_dict['current_comp']]['timestart']-state_dict['window']:state_dict['comps'][state_dict['current_comp']]['timestop']+state_dict['window']],label='U')
    a0.step(state_dict['time_axis'][state_dict['comps'][state_dict['current_comp']]['timestart']-state_dict['window']:state_dict['comps'][state_dict['current_comp']]['timestop']+state_dict['window']]*1e-3,
            V_tcal[state_dict['comps'][state_dict['current_comp']]['timestart']-state_dict['window']:state_dict['comps'][state_dict['current_comp']]['timestop']+state_dict['window']],label='V')
    a0.step(state_dict['time_axis'][state_dict['comps'][state_dict['current_comp']]['timestart']-state_dict['window']:state_dict['comps'][state_dict['current_comp']]['timestop']+state_dict['window']]*1e-3,
            state_dict['comps'][state_dict['current_comp']]['weights'][state_dict['comps'][state_dict['current_comp']]['timestart']-state_dict['window']:state_dict['comps'][state_dict['current_comp']]['timestop']+state_dict['window']]*np.max(I_tcal)/np.max(state_dict['comps'][state_dict['current_comp']]['weights'][state_dict['comps'][state_dict['current_comp']]['timestart']-state_dict['window']:state_dict['comps'][state_dict['current_comp']]['timestop']+state_dict['window']]),label='weights',color='purple',linewidth=4)
    a0.set_xlim(32.7*state_dict['n_t']*state_dict['timestart']*1e-3 - state_dict['window']*32.7*state_dict['n_t']*1e-3,
            32.7*state_dict['n_t']*state_dict['timestop']*1e-3 + state_dict['window']*32.7*state_dict['n_t']*1e-3)
    a0.set_xticks([])
    a0.legend(loc="upper right")
    a0.axvline(state_dict['time_axis'][state_dict['comps'][state_dict['current_comp']]['left_lim']]*1e-3,color='red')
    a0.axvline(state_dict['time_axis'][state_dict['comps'][state_dict['current_comp']]['right_lim']]*1e-3,color='red')

    a1.imshow(Ip[:,state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],aspect='auto',
            extent=[32.7*state_dict['n_t']*state_dict['timestart']*1e-3 - state_dict['window']*32.7*state_dict['n_t']*1e-3,
                32.7*state_dict['n_t']*state_dict['timestop']*1e-3 + state_dict['window']*32.7*state_dict['n_t']*1e-3,
                np.min(state_dict['freq_test'][0]),np.max(state_dict['freq_test'][0])])
    a1.set_xlabel("Time (ms)")
    a1.set_ylabel("Frequency (MHz)")
    plt.subplots_adjust(hspace=0)
    plt.show()



    if donecompbutton.clicked:
        
        #combine and normalize weights from each component
        state_dict['weights'] = np.zeros(len(state_dict['comps'][state_dict['current_comp']]['weights']))
        for i in range(state_dict['n_comps']):
            state_dict['weights'] += state_dict['comps'][i]['weights']
        state_dict['weights'] = state_dict['weights']/np.sum(state_dict['weights'])

        #compute full SNR
        state_dict['S/N'] = get_SNR(state_dict['I_tcal'],state_dict['weights'],state_dict['timestart'],state_dict['timestop'])

        #get edge conditions from first and last components
        ts1 = []
        ts2 = []
        b1 = []
        b2 = []
        for i in state_dict['comps'].keys():
            ts1.append(state_dict['comps'][i]['timestart'])
            ts2.append(state_dict['comps'][i]['timestop'])
            b1.append(state_dict['comps'][i]['buff'][0])
            b2.append(state_dict['comps'][i]['buff'][1])
        first = np.argmin(ts1)
        last = np.argmax(ts2)
        state_dict['timestart'] = ts1[first]
        state_dict['timestop'] = ts2[last]
        state_dict['buff'] = [b1[first],b2[last]]

        #add to dataframe
        df.loc["All"] = [state_dict['buff'][0],
                                                   state_dict['buff'][1],
                                                   np.nan,
                                                   np.nan,
                                                   np.nan,
                                                   np.nan,
                                                   state_dict['S/N']]

        #done with components, move to RM synth
        state_dict['current_state'] += 1

    #update widget dict
    update_wdict([logwindow_slider,logibox_slider,
                buff_L_slider,buff_R_slider,ncomps_num,comprange_slider,
                avger_w_slider,sf_window_weights_slider],
                ["logwindow_slider","logibox_slider",
                "buff_L_slider","buff_R_slider","ncomps_num","comprange_slider",
                "avger_w_slider","sf_window_weights_slider"],
                param='value')

    return





"""
RM Synthesis State
"""

def RM_screen(useRMTools,maxRM_num_tools,dRM_tools,useRMsynth,nRM_num,minRM_num,
                 maxRM_num,getRMbutton,useRM2D,nRM_num_zoom,RM_window_zoom,dRM_tools_zoom,
                 getRMbutton_zoom,RM_gal_display,RM_galerr_display,RM_ion_display,RM_ionerr_display,
                 getRMgal_button,getRMion_button,rmcomp_menu):
    
    #update component options
    update_wdict([rmcomp_menu],['rmcomp_menu'],
                param='value')

    #update RM displays
    if getRMgal_button.clicked:
        state_dict['RM_gal'],state_dict['RM_galerr'] = get_rm(radec=(state_dict['RA'],state_dict['DEC']),filename=repo_path + "/data/faraday2020v2.hdf5")
        state_dict['RM_gal'] = np.around(state_dict['RM_gal'],2)
        state_dict['RM_galerr'] = np.around(state_dict['RM_galerr'],2)
        RM_gal_display.data = state_dict['RM_gal']
        RM_galerr_display.data = state_dict['RM_galerr']
    
    if getRMion_button.clicked:
        state_dict['RM_ion'],state_dict['RM_ionerr'] = RMcal.get_rm_ion(state_dict['RA'],state_dict['DEC'],state_dict['mjd'])
        state_dict['RM_ion'] = np.around(state_dict['RM_ion'],2)
        state_dict['RM_ionerr'] = np.around(state_dict['RM_ionerr'],2)
        RM_ion_display.data = state_dict['RM_ion']
        RM_ionerr_display.data = state_dict['RM_ionerr']



    #if run button is clicked, run RM synthesis for all individual subcomponents and full burst
    if getRMbutton.clicked:

        
        #make dynamic spectra w/ high frequency resolution
        Ip_full = dsapol.avg_time(state_dict['base_Ical'],state_dict['rel_n_t'])
        Qp_full = dsapol.avg_time(state_dict['base_Qcal'],state_dict['rel_n_t'])
        Up_full = dsapol.avg_time(state_dict['base_Ucal'],state_dict['rel_n_t'])
        Vp_full = dsapol.avg_time(state_dict['base_Vcal'],state_dict['rel_n_t'])

        #plt.figure(figsize=(12,6))
        if state_dict['n_comps'] > 1:                
            #loop through each component
            for i in range(state_dict['n_comps']):
                #mask other components and get spectra
                Ip = copy.deepcopy(Ip_full)
                Qp = copy.deepcopy(Qp_full)
                Up = copy.deepcopy(Up_full)
                Vp = copy.deepcopy(Vp_full)
                for k in range(state_dict['n_comps']):
                    if i != k:
                        mask = np.zeros(Ip.shape)
                        mask[:,state_dict['comps'][k]['left_lim']:state_dict['comps'][k]['right_lim']] = 1
                        Ip = ma(Ip,mask)
                        Qp = ma(Qp,mask)
                        Up = ma(Up,mask)
                        Vp = ma(Vp,mask)
                If,Qf,Uf,Vf = dsapol.get_stokes_vs_freq(Ip,Qp,Up,Vp,state_dict['comps'][i]['width_native'],state_dict['fobj'].header.tsamp,state_dict['base_n_f'],state_dict['n_t'],state_dict['base_freq_test'],n_off=int(NOFFDEF/state_dict['n_t']),plot=False,show=False,normalize=True,buff=state_dict['comps'][i]['buff'],weighted=True,n_t_weight=state_dict['comps'][i]['avger_w'],timeaxis=state_dict['time_axis'],fobj=state_dict['fobj'],sf_window_weights=state_dict['comps'][i]['sf_window_weights'],input_weights=state_dict['comps'][i]['weights'])
                
                #STAGE 1: RM-TOOLS
                if useRMTools.value: #STAGE 1: RM-TOOLS
                    n_off = int(NOFFDEF/state_dict['n_t'])

                    RM,RMerr,state_dict['comps'][i]['RMcalibrated']['RM_tools_snrs'],state_dict['comps'][i]['RMcalibrated']['trial_RM_tools'] = RMcal.get_RM_tools(If,Qf,Uf,Vf,Ip,Qp,Up,Vp,state_dict['base_freq_test'],state_dict['n_t'],maxRM_num_tools=maxRM_num_tools.value,dRM_tools=dRM_tools.value,n_off=int(NOFFDEF/state_dict['n_t']))
                    state_dict['comps'][i]['RMcalibrated']['RM_tools'] = [RM,RMerr]
                #elif useRMsynth.value: #STAGE 2: 1D RM synthesis
                #plt.plot(state_dict['comps'][i]['trial_RM_tools'],state_dict['comps'][i]['RM_tools_snrs'])


                #STAGE 2: 1D RM synthesis
                if useRMsynth.value:
                    n_off = int(NOFFDEF/state_dict['n_t'])

                    RM,RMerr,state_dict['comps'][i]['RMcalibrated']['RMsnrs1'],state_dict['comps'][i]['RMcalibrated']['trial_RM1'] = RMcal.get_RM_1D(If,Qf,Uf,Vf,state_dict['base_freq_test'],nRM_num=nRM_num.value,minRM_num=minRM_num.value,maxRM_num=maxRM_num.value,n_off=n_off)
                    state_dict['comps'][i]['RMcalibrated']["RM1"] = [RM,RMerr]

                

        #RM-TOOLS for the full burst
        
        If,Qf,Uf,Vf = dsapol.get_stokes_vs_freq(Ip_full,Qp_full,Up_full,Vp_full,state_dict['width_native'],state_dict['fobj'].header.tsamp,state_dict['base_n_f'],state_dict['n_t'],state_dict['base_freq_test'],n_off=int(NOFFDEF/state_dict['n_t']),plot=False,show=False,normalize=True,buff=state_dict['buff'],weighted=True,timeaxis=state_dict['time_axis'],fobj=state_dict['fobj'],input_weights=state_dict['weights'])
        
        RM,RMerr,state_dict["RMcalibrated"]['RM_tools_snrs'],state_dict["RMcalibrated"]['trial_RM_tools'] = RMcal.get_RM_tools(If,Qf,Uf,Vf,Ip_full,Qp_full,Up_full,Vp_full,state_dict['base_freq_test'],state_dict['n_t'],maxRM_num_tools=maxRM_num_tools.value,dRM_tools=dRM_tools.value,n_off=int(NOFFDEF/state_dict['n_t']))
        state_dict["RMcalibrated"]['RM_tools'] = [RM,RMerr]

        #1D RM synthesis
        RM,RMerr,state_dict["RMcalibrated"]['RMsnrs1'],state_dict["RMcalibrated"]['trial_RM1'] = RMcal.get_RM_1D(If,Qf,Uf,Vf,state_dict['base_freq_test'],nRM_num=nRM_num.value,minRM_num=minRM_num.value,maxRM_num=maxRM_num.value,n_off=int(NOFFDEF/state_dict['n_t']))
        state_dict["RMcalibrated"]["RM1"] = [RM,RMerr]


        #plt.plot(state_dict['trial_RM_tools'],state_dict['RM_tools_snrs'])
        #plt.xlabel(r'RM ($rad/m^2$)')
        #plt.ylabel(r'F($\phi$)')            
        #plt.show()


    #plot

    if rmcomp_menu.value == 'All':
        dsapol.RM_summary_plot(state_dict['ids'],state_dict['nickname'],[state_dict['RMcalibrated']['RMsnrs1'],state_dict['RMcalibrated']['RM_tools_snrs']],[state_dict['RMcalibrated']['RMsnrs1zoom'],state_dict['RMcalibrated']['RM_tools_snrszoom'],state_dict['RMcalibrated']['RMsnrs2']],state_dict['RMcalibrated']["RM2"][0],state_dict["RMcalibrated"]["RMerrfit"],state_dict["RMcalibrated"]["trial_RM1"],state_dict["RMcalibrated"]["trial_RM2"],state_dict["RMcalibrated"]["trial_RM_tools"],state_dict["RMcalibrated"]["trial_RM_toolszoom"],threshold=9,suffix="_FORMAT_UPDATE_PARSEC",show=True,title='All Components')
    
    elif rmcomp_menu.value != '':
        i= int(rmcomp_menu.value)
        dsapol.RM_summary_plot(state_dict['ids'],state_dict['nickname'],[state_dict['comps'][i]['RMcalibrated']['RMsnrs1'],state_dict['comps'][i]['RMcalibrated']['RM_tools_snrs']],[state_dict['comps'][i]['RMcalibrated']['RMsnrs1zoom'],state_dict['comps'][i]['RMcalibrated']['RM_tools_snrszoom'],state_dict['comps'][i]['RMcalibrated']['RMsnrs2']],state_dict['comps'][i]['RMcalibrated']["RM2"][0],state_dict['comps'][i]["RMcalibrated"]["RMerrfit"],state_dict['comps'][i]["RMcalibrated"]["trial_RM1"],state_dict['comps'][i]["RMcalibrated"]["trial_RM2"],state_dict['comps'][i]["RMcalibrated"]["trial_RM_tools"],state_dict['comps'][i]["RMcalibrated"]["trial_RM_toolszoom"],threshold=9,suffix="_FORMAT_UPDATE_PARSEC",show=True,title='Component ' + rmcomp_menu.value)

    #update widget dict
    update_wdict([maxRM_num_tools,dRM_tools,nRM_num,minRM_num,maxRM_num,nRM_num_zoom,RM_window_zoom,dRM_tools_zoom,useRMTools,useRMsynth,useRM2D,rmcomp_menu],
                ['maxRM_num_tools','dRM_tools','nRM_num','minRM_num','maxRM_num','nRM_num_zoom','RM_window_zoom','dRM_tools_zoom','useRMTools','useRMsynth','useRM2D','rmcomp_menu'],param='value')
    
    update_wdict([RM_gal_display,RM_galerr_display,RM_ion_display,RM_ionerr_display],['RM_gal_display','RM_galerr_display','RM_ion_display','RM_ionerr_display'],param='data')
        
    
    return




