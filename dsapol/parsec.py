from rmtable import rmtable
from IPython.display import display,update_display
import contextlib
#from dsaT3.filplot_funcs import proc_cand_fil
from dsapol import customfilplotfuncs as cfpf
from dsapol import polbeamform
from dsapol import polcal
from dsapol import RMcal
from dsapol import dsapol
from dsapol import rmtablefuncs 
from dsapol import dedisp
from dsapol import filt
from dsapol import budget
from dsapol import scatscint
import numpy as np
import matplotlib
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
from bilby.core import result as bcresult
from scipy.signal import find_peaks
from scipy.signal import peak_widths
import copy
import numpy as np
import mercury as mr
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
from astropy.table import Table
from astropy.coordinates import EarthLocation,SkyCoord
import astropy.units as u
import os
import signal
from lmfit import minimize, Parameters, fit_report, Model,Parameter
from tqdm import tqdm
from scintillation import scint
from scattering import scat
from PIL import Image
"""
This file contains code for the Polarization Analysis and RM Synthesis Enabled for Calibration (PARSEC)
user interface to the dsa110-pol module. The interface will use Mercury and a jupyter notebook to 
create a web app, and will operate as a wrapper around dsapol.py. 
"""

"""
Plotting parameters
"""
fsize=30#35
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
import json
f = open(os.environ['DSAPOLDIR'] + "directories.json","r")
dirs = json.load(f)
f.close()
"""
Factors for RM, pol stuff
"""
RMSF_generated = False
unbias_factor = 1 #1.57
default_path = dirs["polcal"]#"/media/ubuntu/ssd/sherman/code/"

"""
Repo path
"""
repo_path = dirs["cwd"]#"/media/ubuntu/ssd/sherman/code/dsa110-pol/"

"""
Level 3 candidates path
"""
level3_path = dirs["candidates"] #"/dataz/dsa110/candidates/"

"""
FRB data path
"""
frbpath = dirs["data"] #"/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/"


"""
dsastorage paths
"""
dsastorageFRBDir = dirs["dsastorageFRBDir"]
dsastorageCALDir = dirs["dsastorageCALDir"]

"""
Read FRB parameters
"""
FRB_RA = []
FRB_DEC = []
FRB_DM = []
FRB_heimSNR = []
FRBs= []
FRB_mjd = []
FRB_z = []
FRB_w = []
FRB_BEAM = []
FRB_IDS = []
FRB_RM = []
FRB_RMerr = []
FRB_RMgal = []
FRB_RMgalerr = []
FRB_RMion = []
FRB_RMionerr = []
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
                
                if row[2] != "":
                    FRB_heimSNR.append(float(row[2]))
                else:
                    FRB_heimSNR.append(-1)

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
                    FRB_IDS.append("")

                if row[12] != "":
                    FRB_RM.append(float(row[12]))
                else:
                    FRB_RM.append(np.nan)

                if row[13] != "":
                    FRB_RMerr.append(float(row[13]))
                else:
                    FRB_RMerr.append(np.nan)

                if row[14] != "":
                    FRB_RMgal.append(float(row[14]))
                else:
                    FRB_RMgal.append(np.nan)

                if row[15] != "":
                    FRB_RMgalerr.append(float(row[15]))
                else:
                    FRB_RMgalerr.append(np.nan)

                if row[16] != "":
                    FRB_RMion.append(float(row[16]))
                else:
                    FRB_RMion.append(np.nan)

                if row[17] != "":
                    FRB_RMionerr.append(float(row[17]))
                else:
                    FRB_RMionerr.append(np.nan)
                #if no directory exists, make one
                if len(glob.glob(dirs['data']+FRB_IDS[-1] + "_" + FRBs[-1])) == 0:
                    os.system("mkdir " + dirs['data']+FRB_IDS[-1] + "_" + FRBs[-1])
    return
update_FRB_params()

def update_FRB_DM_params(FRB_name,DM,fname="DSA110-FRBs-PARSEC_TABLE.csv",path=repo_path):
    """
    This function updates the global FRB parameters from the provided file. File is a copy
    of the 'tablecsv' tab in the DSA110 FRB spreadsheet.
    """
    alldata = []
    with open(repo_path + 'data/' + fname,"r") as csvfile:
        reader = csv.reader(csvfile,delimiter=',')
        for row in reader:
            if row[0] == FRB_name:
                if row[4] != "":
                    row[4] = DM
                elif row[3] != "":
                    row[3] = DM
                else:
                    row[4] = row[3] = DM
            alldata.append(row)
    csvfile.close()

    with open(repo_path + 'data/' + fname,"w") as csvfile:
        writer = csv.writer(csvfile,delimiter=',')
        for row in alldata:
            writer.writerow(row)
    csvfile.close()
    

    update_FRB_params()
    return


"""
General Layout:
    Mercury creates a web app server and will re-execute cells whenever a widget value is changed. However, it only re-executes cells below the widget that
    changed. Therefore we will layout the dashboard so that (1) the current state is stored as a global dictionary (2) a variable defining the current stage
    as an integer is stored as a global variable (3) the jupyter notebook will call a different plotting function depending on the stage to display the
    correct screen (4) all widgets will be defined in the notebook above the plotting cell
"""

# default values for state and polcal dicts and dataframe tables
state_dict = dict()
##state_dict['current_state'] = 0
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
state_dict['base_df'] = 30.4E3 #Hz
state_dict['dDM'] = 0
state_dict['window'] = 2**5
state_dict['rel_n_t'] = 1
state_dict['rel_n_f'] = 2**5
state_dict['Iflux'] = np.nan 
state_dict['Iflux_err'] = np.nan
state_dict['Qflux'] = np.nan
state_dict['Qflux_err'] = np.nan
state_dict['Uflux'] = np.nan
state_dict['Uflux_err'] = np.nan 
state_dict['Vflux'] = np.nan 
state_dict['Vflux_err'] = np.nan
state_dict['noise_chan'] = np.nan 
state_dict['polint'] = np.nan 
state_dict['polint_err'] = np.nan
state_dict['Tpol'] = np.nan
state_dict['Tpol_err'] = np.nan
state_dict['Lpol'] = np.nan
state_dict['Lpol_err'] = np.nan
state_dict['Vpol'] = np.nan
state_dict['Vpol_err'] = np.nan
state_dict['absVpol'] = np.nan
state_dict['absVpol_err'] = np.nan
state_dict['avg_PA'] = np.nan
state_dict['PA_err'] = np.nan
state_dict['freq_test'] = [np.linspace(1311.25000003072,1498.75,6144)]*4
state_dict['base_n_t'] = 1
state_dict['base_n_f'] = 1 
state_dict['ids'] = ""
state_dict['nickname'] = ""
state_dict['RA'] = 0
state_dict['DEC'] = 0
state_dict['ibeam'] =np.nan
state_dict['DM0'] =np.nan 
state_dict['datadir'] ="./"
state_dict['buff'] = [0,0]
state_dict['width_native'] = 1 
state_dict['mjd'] = 0 
state_dict['base_I'] = ma(np.nan*np.ones((6144,5120)),np.zeros((6144,5120)))
state_dict['base_Q'] = ma(np.nan*np.ones((6144,5120)),np.zeros((6144,5120)))
state_dict['base_U'] = ma(np.nan*np.ones((6144,5120)),np.zeros((6144,5120)))
state_dict['base_V'] = ma(np.nan*np.ones((6144,5120)),np.zeros((6144,5120)))
#state_dict['fobj'] = None
state_dict['tsamp'] = 32.7e-6
state_dict['base_freq_test'] = [np.linspace(1311.25000003072,1498.75,6144)]*4 
state_dict['base_wav_test'] = [(3e8)/(np.linspace(1311.25000003072,1498.75,6144)*1e6)]*4
state_dict['base_time_axis'] = np.nan*np.ones(5120)
state_dict['badchans'] = []
state_dict['base_I_unnormalized'] = np.nan*np.ones((6144,5120))
state_dict['base_Q_unnormalized'] = np.nan*np.ones((6144,5120))
state_dict['base_U_unnormalized'] = np.nan*np.ones((6144,5120))
state_dict['base_V_unnormalized'] = np.nan*np.ones((6144,5120))
state_dict['base_Ical_unnormalized'] = np.nan*np.ones((6144,5120))
state_dict['base_Qcal_unnormalized'] = np.nan*np.ones((6144,5120))
state_dict['base_Ucal_unnormalized'] = np.nan*np.ones((6144,5120))
state_dict['base_Vcal_unnormalized'] = np.nan*np.ones((6144,5120))
state_dict['base_Ical_unnormalized_errs'] = np.nan*np.ones((6144,5120))
state_dict['base_Qcal_unnormalized_errs'] = np.nan*np.ones((6144,5120))
state_dict['base_Ucal_unnormalized_errs'] = np.nan*np.ones((6144,5120))
state_dict['base_Vcal_unnormalized_errs'] = np.nan*np.ones((6144,5120))
state_dict['base_Ical_f_unnormalized'] = np.nan*np.ones(6144)
state_dict['base_Qcal_f_unnormalized'] = np.nan*np.ones(6144)
state_dict['base_Ucal_f_unnormalized'] = np.nan*np.ones(6144)
state_dict['base_Vcal_f_unnormalized'] = np.nan*np.ones(6144)
state_dict['base_Ical_f_unnormalized_errs'] = np.nan*np.ones(6144)
state_dict['base_Qcal_f_unnormalized_errs'] = np.nan*np.ones(6144)
state_dict['base_Ucal_f_unnormalized_errs'] = np.nan*np.ones(6144)
state_dict['base_Vcal_f_unnormalized_errs'] = np.nan*np.ones(6144)        
state_dict['I_f'] = np.nan*np.ones(6144)
state_dict['Q_f'] = np.nan*np.ones(6144)
state_dict['U_f'] = np.nan*np.ones(6144)
state_dict['V_f'] = np.nan*np.ones(6144)
state_dict['I_f_unweighted'] = np.nan*np.ones(6144)
state_dict['Q_f_unweighted'] = np.nan*np.ones(6144)
state_dict['U_f_unweighted'] = np.nan*np.ones(6144)
state_dict['V_f_unweighted'] = np.nan*np.ones(6144)
state_dict['I_fcal'] = np.nan*np.ones(6144)
state_dict['Q_fcal'] = np.nan*np.ones(6144)
state_dict['U_fcal'] = np.nan*np.ones(6144)
state_dict['V_fcal'] = np.nan*np.ones(6144)
state_dict['I_fcal_unweighted'] = np.nan*np.ones(6144)
state_dict['Q_fcal_unweighted'] = np.nan*np.ones(6144)
state_dict['U_fcal_unweighted'] = np.nan*np.ones(6144)
state_dict['V_fcal_unweighted'] = np.nan*np.ones(6144)
state_dict['I_fcalRM'] = np.nan*np.ones(6144)
state_dict['Q_fcalRM'] = np.nan*np.ones(6144)
state_dict['U_fcalRM'] = np.nan*np.ones(6144)
state_dict['V_fcalRM'] = np.nan*np.ones(6144)

state_dict['I_tcal'] = np.nan*np.ones(5120)
state_dict['Q_tcal'] = np.nan*np.ones(5120)
state_dict['U_tcal'] = np.nan*np.ones(5120)
state_dict['V_tcal'] = np.nan*np.ones(5120)
state_dict['I_tcal_err'] = np.nan*np.ones(5120)
state_dict['Q_tcal_err'] = np.nan*np.ones(5120)
state_dict['U_tcal_err'] = np.nan*np.ones(5120)
state_dict['V_tcal_err'] = np.nan*np.ones(5120)

state_dict['Tpol'] = np.nan
state_dict['Tpol_err'] = np.nan
state_dict['Lpol'] = np.nan
state_dict['Lpol_err'] = np.nan
state_dict['Vpol'] = np.nan
state_dict['Vpol_err'] = np.nan
state_dict['absVpol'] = np.nan
state_dict['absVpol_err'] = np.nan
state_dict['snr'] = np.nan
state_dict['Tsnr'] =np.nan
state_dict['Lsnr']= np.nan
state_dict['Vsnr'] = np.nan
state_dict['Tclass'] = ""
state_dict['Lclass'] = ""
state_dict['Vclass'] = ""
state_dict['RM_gal'] = np.nan
state_dict['RM_galerr'] = np.nan
state_dict['RM_galRA'] = np.nan
state_dict['RM_galDEC'] = np.nan
state_dict['RM_ion'] = np.nan
state_dict['RM_ionerr'] = np.nan
state_dict['RM_ionRA'] = np.nan
state_dict['RM_ionDEC'] = np.nan
state_dict['RM_ionmjd'] = np.nan
state_dict['scatter_init'] = False
state_dict['RMhost'] = np.nan
state_dict['DMhost'] = np.nan
state_dict['qdat'] = None


state_dict['intervener_names_RM'] = []
state_dict['intervener_names_DM'] = []
state_dict['intervener_RMs']=[]
state_dict['intervener_RM_errs']=[]
state_dict['intervener_RMzs']=[]
state_dict['intervener_zs']=[]
state_dict['intervener_DMs']=[]
state_dict['intervener_DM_errs']=[]


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
polcal_dict['maxProcesses'] = 32
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


#table for scintillation bw fit values
df_scint = pd.DataFrame(
    {
        r'$\gamma$ (MHz)': [np.nan],#*len(corrarray),
        r'$\gamma$ Error (MHz)':[np.nan],
        r'm': [np.nan],#*len(corrarray),
        r'm Error':[np.nan],
        r'c': [np.nan],
        r'c Error':[np.nan]
    },
        index=["All"]#copy.deepcopy(corrarray)
    )

df_specidx = pd.DataFrame(
        {
            r'$\Gamma$':[np.nan],
            r'$\Gamma$ Error':[np.nan],
            r'F0':[np.nan],
            r'F0 Error':[np.nan]
        },
            index = ["All"]
        )
            

df_scat = pd.DataFrame(
        {
            r'x0 (ms)':[],
            r'x0 upper error (ms)':[],
            r'x0 lower error (ms)':[],
            r'amp':[],
            r'amp upper error':[],
            r'amp lower error':[],
            r'$\sigma$ (ms)':[],
            r'$\sigma$ upper error (ms)':[],
            r'$\sigma$ lower error (ms)':[],
            r'$\tau$ (ms)':[],
            r'$\tau$ upper error':[],
            r'$\tau$ lower error':[],
            r'f':[],
            r'f upper error':[],
            r'f lower error':[],
            r'BIC':[]
        },
        index=[]
        )


df_RM_budget = pd.DataFrame(
        {
            'RMobs':[""],
            'RMmw':[""],
            'RMion':[""],
            'RMint':[""],
            'RMhost':[""],
            'RMhost/(1+zhost)^2':[""],
            },
            index=['Budget']
        )


df_DM_budget = pd.DataFrame(
        {
            'DMobs':[""],
            'DMmw':[""],
            'DMhalo':[""],
            'DMigm':[""],
            'DMint':[""],
            'DMhost':[""],
            'DMhost/(1+zhost)':[""],
            },
            index=['Budget']
        )


#List of initial widget values updated whenever screen loads
def get_frbfiles(path=frbpath):
    frbfiles = glob.glob(path + '2*_*')
    return [frbfiles[i][frbfiles[i].index(frbpath)+len(frbpath):] for i in range(len(frbfiles))]

frbfiles = get_frbfiles()
ids = frbfiles[0][:frbfiles[0].index('_')]
RA = FRB_RA[FRB_IDS.index(ids)]
DEC = FRB_DEC[FRB_IDS.index(ids)]
ibeam = int(FRB_BEAM[FRB_IDS.index(ids)])
mjd = FRB_mjd[FRB_IDS.index(ids)]
DMinit = FRB_DM[FRB_IDS.index(ids)]
zinit = FRB_z[FRB_IDS.index(ids)]
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
         'z_display':zinit,
         'showlog':True,
         'polcalloadbutton':False,

         'n_t_slider':1, ############### (1) Dedispersion ##################
         'logn_f_slider':5,
         'logwindow_slider_init':5,
         'ddm_num':0,
         'DM_input_display':DMinit,
         'DM_new_display':DMinit,
         'DM_showerrs':False,
         'updateDM':False,
         #'DMINITIALIZED':False,

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
         'polcalprocs':32,

         'ncomps_num':1, ############### (4) Filter Weights ##################
         'comprange_slider':[0.25*5120*32.7E-3,0.75*5120*32.7E-3],                  
         'comprange_slider_max':5120*32.7E-3,
         'avger_w_slider':1,
         'sf_window_weights_slider':3,
         'logibox_slider':0,
         'logwindow_slider':5,
         'buff_L_slider':1,
         'buff_R_slider':1,
         'n_t_slider_filt':1,
         'logn_f_slider_filt':5,
         'multipeaks':False,
         'multipeaks_height_slider':0.5,
         'Iflux_display':np.nan,
         'Qflux_display':np.nan,
         'Uflux_display':np.nan,
         'Vflux_display':np.nan,
         'filt_showerrs':False,

        'scattermenu':['Component 0'],
        'scatfitmenu':'Nested Sampling',
        'scatfitmenu_choices':['LMFIT Non-Linear Least Squares','Scipy Non-Linear Least Squares','Nested Sampling','EMCEE Markov-Chain Monte Carlo'],
        'scattermenu_choices':['Component 0'],
        'scatterLbuffer_slider':0,
        'scatterRbuffer_slider':0,
        'scatter_init_all':False,
        'scatterbackground':False,
        'scatterweights':False,
        'scatterresume':False,
        'scattersamps':False,
        'scatter_nlive':500,
        'scatter_nwalkers':32,
        'scatter_nsteps':5000,
        'scatter_nburn':100,
        'scatter_nthin':15,
        'scatter_sliderrange':1,
        'x0_guess':80,
        'amp_guess':10,
        'sigma_guess':1,
        'tau_guess':1,
        'x0_guess_0':0.35e-3,
        'amp_guess_0':1,
        'sigma_guess_0':1,
        'tau_guess_0':1,
        'x0_guess_1':0.35e-3,
        'amp_guess_1':1,
        'sigma_guess_1':1,
        'tau_guess_1':1,
        'x0_guess_2':0.35e-3,
        'amp_guess_2':1,
        'sigma_guess_2':1,
        'tau_guess_2':1,
        'x0_guess_3':0.35e-3,
        'amp_guess_3':1,
        'sigma_guess_3':1,
        'tau_guess_3':1,
        'x0_guess_4':0.35e-3,
        'amp_guess_4':1,
        'sigma_guess_4':1,
        'tau_guess_4':1,

        'x0_range_0':[0.35e-3/2,3*0.35e-3/2],
        'amp_range_0':[1/2,3/2],
        'sigma_range_0':[1/2,3/2],
        'tau_range_0':[1/2,3/2],
        'x0_range_1':[0.35e-3/2,3*0.35e-3/2],
        'amp_range_1':[1/2,3/2],
        'sigma_range_1':[1/2,3/2],
        'tau_range_1':[1/2,3/2],
        'x0_range_2':[0.35e-3/2,3*0.35e-3/2],
        'amp_range_2':[1/2,3/2],
        'sigma_range_2':[1/2,3/2],
        'tau_range_2':[1/2,3/2],
        'x0_range_3':[0.35e-3/2,3*0.35e-3/2],
        'amp_range_3':[1/2,3/2],
        'sigma_range_3':[1/2,3/2],
        'tau_range_3':[1/2,3/2],
        'x0_range_4':[0.35e-3/2,3*0.35e-3/2],
        'amp_range_4':[1/2,3/2],
        'sigma_range_4':[1/2,3/2],
        'tau_range_4':[1/2,3/2],


        'scintmenu':'All',
        'scintmenu_choices':['All'],
        'scint_fit_range':1,
        'scintfitmenu':['LMFIT Non-Linear Least Squares'],
        'scintfitmenu_choices':['LMFIT Non-Linear Least Squares','Scipy Non-Linear Least Squares'],
        'gamma_guess':10, 
        'm_guess':1,
        'c_guess':0,
        
        'specidxmenu':'All',
        'specidxmenu_choices':['All'],
        'specidxfitmenu':['LMFIT Non-Linear Least Squares'],
        'specidxfitmenu_choices':['LMFIT Non-Linear Least Squares','Scipy Non-Linear Least Squares'],
        'specidx_guess':0,
        'F0_guess':1,


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
        'rmcomp_menu_choices':['All',''],
        'rmcal_menu':'No RM Calibration',
        'rmcal_menu_choices':['No RM Calibration'],
        'RMdisplay':np.nan,
        'RMerrdisplay':np.nan,
        'RMsynthbackground':False,
        'trialz':-1,
        'rmcal_input':0,
        'catalog_selection':['All'],
        'catalog_selection_choices':list(budget.SIMBAD_CATALOG_OPTIONS)+['All'],
        'galtype_selection':['All'],
        'galtype_selection_choices':list(budget.SIMBAD_GALAXY_OPTIONS)+['All'],
        'queryradius':1,
        'limitzrange':False,
        'zrange':1e-3,
        'cosmo_selection':'Planck18',
        'cosmo_selection_choices':budget.COSMOLOGY_OPTIONS,
        'galaxy_selection':[],
        'galaxy_selection_choices':[],
        'galaxy_masses':'',
        'galaxy_bfields':'',
        'mass_type':'500m',
        'mass_type_choices':['virial','200m','500m','200c','500c'],
        'Bfield_range':100,
        'Bfield_res': 0.2,
        'Bfield_display':np.nan,
        'Bfield_pos_err_display':np.nan,
        'Bfield_neg_err_display':np.nan,

        'showghostPA':True,
        'intLbuffer_slider':0,
        'intRbuffer_slider':0,
        'polcomp_menu':'All',
        'polcomp_menu_choices':['All'],
        'notesinput':"",
        'overwritefils':False
}


#create a mapping system for possible RMs to calibrate with
RMcaldict= {
        'RM-Tools':{
            'coarse':{
                'All Components':{
                    'RM':np.nan,
                    'Error':np.nan}},
            'zoom':{
                'All Components':{
                    'RM':np.nan,
                    'Error':np.nan}}},
        '1D-Synth':{
            'coarse':{
                'All Components':{
                    'RM':np.nan,
                    'Error':np.nan}},
            'zoom':{
                'All Components':{
                    'RM':np.nan,
                    'Error':np.nan}}},
        '2D-Synth':{
            'All Components':{
                'RM':np.nan,
                'Error':np.nan}}
            }

RMdf = pd.DataFrame(
    {
        'RM-Tools': [np.nan],
        'RM-Tools Error': [np.nan],
        '1D-Synth': [np.nan],
        '1D-Synth Error': [np.nan],
        '2D-Synth':[np.nan],
        '2D-Synth Error': [np.nan]
    },
        index=['All']
    )

poldf = pd.DataFrame(
        {
            'S/N': [np.nan],
            'T/I (%)': [np.nan],
            'T/I Error (%)': [np.nan],
            'T S/N': [np.nan],
            'T Class':[""],
            'L/I (%)': [np.nan],
            'L/I Error (%)': [np.nan],
            'L S/N': [np.nan],
            "L Class":[""],
            'V/I (%)': [np.nan],
            'V/I Error (%)': [np.nan],
            '|V|/I (%)': [np.nan],
            '|V|/I Error (%)': [np.nan],
            'V S/N': [np.nan],
            "V Class":[""],
            'PA (deg)': [np.nan],
            'PA Error (deg)': [np.nan]
        },
            index=['All']
        )

def make_rmcal_menu_choices():
    choices = ["No RM Calibration"]
    if  ~np.isnan(state_dict['RMinput']):
        choices.append("Previous RM Estimate: " + str(np.around(state_dict['RMinput'],2)) + "+-" + str(np.around(state_dict['RMinputerr'],2)) + r'rad/m^2')
    for k in RMcaldict.keys():
        for j in RMcaldict[k].keys():
            if j in ['coarse','zoom']:
                for l in RMcaldict[k][j].keys():
                    if ~np.isnan(RMcaldict[k][j][l]['RM']):
                        choice = str(k) + ' ' + '[' + str(j) + '] ' + str(l) + ': ' + str(np.around(RMcaldict[k][j][l]['RM'],2)) + r'+-' + str(np.around(RMcaldict[k][j][l]['Error'],2)) + r'rad/m^2'
                        choices.append(choice)
                    else: continue
            
            
            else:
                if ~np.isnan(RMcaldict[k][j]['RM']):
                    choice = str(k) + ' ' + str(j) + ': ' + str(np.around(RMcaldict[k][j]['RM'],2)) + r'+-' + str(np.around(RMcaldict[k][j]['Error'],2)) + r'rad/m^2'
                    choices.append(choice)
                else: continue
    choices.append("Input RM (ENTER VALUE BELOW)")

    return choices


def RM_from_menu(choice,rmcal_input=0):
    if choice == "No RM Calibration":
        return np.nan,np.nan
    if 'Input RM' in choice:
        return rmcal_input.value,np.nan
    if 'Previous RM Estimate' in choice:
        return state_dict['RMinput'],state_dict['RMinputerr']
    key1 = (choice[:8]).strip()
    if '[' in choice:
        key2 = (choice[choice.index('[')+1:choice.index(']')]).strip()
        key3 = (choice[choice.index(']')+1:choice.index(':')]).strip()
        return RMcaldict[key1][key2][key3]['RM'],RMcaldict[key1][key2][key3]['Error']
    else:
        key3 = (choice[9:choice.index(':')]).strip()
        return RMcaldict[key1][key3]['RM'],RMcaldict[key1][key3]['Error']



#RM dict
state_dict['RMcalibrated'] = dict()
state_dict['RMcalibrated']['RMsnrs1'] = np.nan*np.ones(int(wdict['nRM_num']))
state_dict['RMcalibrated']['RM_tools_snrs'] = np.nan*np.ones(int(2*wdict['maxRM_num']/wdict['dRM_tools']))
state_dict['RMcalibrated']['RMsnrs1zoom'] = np.nan*np.ones(int(wdict['nRM_num_zoom']))
state_dict['RMcalibrated']['RM_tools_snrszoom'] = np.nan*np.ones(int(2*wdict['RM_window_zoom']/wdict['dRM_tools_zoom']))
state_dict['RMcalibrated']['RMsnrs2'] = np.nan*np.ones(int(wdict['nRM_num_zoom']))
state_dict['RMcalibrated']["RM2"] = [np.nan,np.nan,np.nan,np.nan]
state_dict['RMcalibrated']["RM1"] = [np.nan,np.nan]
state_dict['RMcalibrated']["RM1zoom"] = [np.nan,np.nan]
state_dict['RMcalibrated']["RM_tools"] = [np.nan,np.nan]
state_dict['RMcalibrated']["RM_toolszoom"] = [np.nan,np.nan]
state_dict["RMcalibrated"]["RMerrfit"] = np.nan
state_dict["RMcalibrated"]["trial_RM1"] = np.linspace(wdict['minRM_num'],wdict['maxRM_num'],int(wdict['nRM_num']))
state_dict["RMcalibrated"]["trial_RM2"] = np.linspace(-wdict['RM_window_zoom'],wdict['RM_window_zoom'],int(wdict['nRM_num_zoom']))
state_dict["RMcalibrated"]["trial_RM_tools"] = np.arange(-wdict['maxRM_num'],wdict['maxRM_num'],wdict['dRM_tools'])
state_dict["RMcalibrated"]["trial_RM_toolszoom"] = np.arange(-wdict['RM_window_zoom'],wdict['RM_window_zoom'],wdict['dRM_tools_zoom'])
state_dict['RMcalibrated']['RMcal'] = np.nan
state_dict['RMcalibrated']['RMcalerr'] = np.nan
state_dict['RMcalibrated']['RMcalstring'] = "No RM Calibration"
state_dict['RMcalibrated']['RMFWHM'] = np.nan


#archive dict
RMtable_archive_df = (rmtablefuncs.make_FRB_RMTable(state_dict))#.to_pandas()#rmtable.RMTable({})
polspec_archive_df = (rmtablefuncs.make_FRB_polSpectra(state_dict))
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
    """
    if 'DM_init_display' in labels and not wdict['DMINITIALIZED']:
        wdict['DM_input_display'] = wdict['DM_init_display']
        wdict['DM_new_display'] = wdict['DM_init_display'] + wdict['ddm_num']
        wdict['DMINITIALIZED'] = True
    if 'DM_input_display' in labels: 
        #wdict['DM_input_display'] = wdict['DM_init_display']
        objects[labels.index('DM_input_display')].data = wdict['DM_input_display']
    if 'DM_new_display' in labels:
        wdict['DM_new_display'] = wdict['DM_input_display'] + wdict['ddm_num']
        objects[labels.index('DM_new_display')].data = wdict['DM_input_display'] + wdict['ddm_num']
    """
    if 'rmcomp_menu' in labels:
        if state_dict['n_comps'] > 1:
            wdict['rmcomp_menu_choices'] = [str(i) for i in range(state_dict['n_comps'])] + ['All','']
        else:
            wdict['rmcomp_menu_choices'] = ['All','']
    if 'rmcal_menu' in labels:

        #update menu options and dict for All components
        if ~np.isnan(state_dict['RMcalibrated']['RM_tools'][0]):
            RMcaldict['RM-Tools']['coarse']['All Components']['RM'] = state_dict['RMcalibrated']['RM_tools'][0]
            RMcaldict['RM-Tools']['coarse']['All Components']['Error'] = state_dict['RMcalibrated']['RM_tools'][1]
        if ~np.isnan(state_dict['RMcalibrated']['RM_toolszoom'][0]):
            RMcaldict['RM-Tools']['zoom']['All Components']['RM'] = state_dict['RMcalibrated']['RM_toolszoom'][0]
            RMcaldict['RM-Tools']['zoom']['All Components']['Error'] = state_dict['RMcalibrated']['RM_toolszoom'][1]
        if ~np.isnan(state_dict['RMcalibrated']['RM1'][0]):
            RMcaldict['1D-Synth']['coarse']['All Components']['RM'] = state_dict['RMcalibrated']['RM1'][0]
            RMcaldict['1D-Synth']['coarse']['All Components']['Error'] = state_dict['RMcalibrated']['RM1'][1]
        if ~np.isnan(state_dict['RMcalibrated']['RM1zoom'][0]):
            RMcaldict['1D-Synth']['zoom']['All Components']['RM'] = state_dict['RMcalibrated']['RM1zoom'][0]
            RMcaldict['1D-Synth']['zoom']['All Components']['Error'] = state_dict['RMcalibrated']['RM1zoom'][1]
        if ~np.isnan(state_dict['RMcalibrated']['RM2'][0]):
            RMcaldict['2D-Synth']['All Components']['RM'] = state_dict['RMcalibrated']['RM2'][0]
            RMcaldict['2D-Synth']['All Components']['Error'] = state_dict['RMcalibrated']['RM2'][1]
       
        #update menu options and dict for each peak
        if state_dict['n_comps'] > 1:
            for i in range(state_dict['n_comps']):
                if ~np.isnan(state_dict['comps'][i]['RMcalibrated']['RM_tools'][0]):
                    RMcaldict['RM-Tools']['coarse']['Component '+str(i)]['RM'] = state_dict['comps'][i]['RMcalibrated']['RM_tools'][0]
                    RMcaldict['RM-Tools']['coarse']['Component '+str(i)]['Error'] = state_dict['comps'][i]['RMcalibrated']['RM_tools'][1]
                if ~np.isnan(state_dict['comps'][i]['RMcalibrated']['RM_toolszoom'][0]):
                    RMcaldict['RM-Tools']['zoom']['Component '+str(i)]['RM'] = state_dict['comps'][i]['RMcalibrated']['RM_toolszoom'][0]
                    RMcaldict['RM-Tools']['zoom']['Component '+str(i)]['Error'] = state_dict['comps'][i]['RMcalibrated']['RM_toolszoom'][1]
                if ~np.isnan(state_dict['comps'][i]['RMcalibrated']['RM1'][0]):
                    RMcaldict['1D-Synth']['coarse']['Component '+str(i)]['RM'] = state_dict['comps'][i]['RMcalibrated']['RM1'][0]
                    RMcaldict['1D-Synth']['coarse']['Component '+str(i)]['Error'] = state_dict['comps'][i]['RMcalibrated']['RM1'][1]
                if ~np.isnan(state_dict['comps'][i]['RMcalibrated']['RM1zoom'][0]):
                    RMcaldict['1D-Synth']['zoom']['Component '+str(i)]['RM'] = state_dict['comps'][i]['RMcalibrated']['RM1zoom'][0]
                    RMcaldict['1D-Synth']['zoom']['Component '+str(i)]['Error'] = state_dict['comps'][i]['RMcalibrated']['RM1zoom'][1]
                if ~np.isnan(state_dict['comps'][i]['RMcalibrated']['RM2'][0]):
                    RMcaldict['2D-Synth']['Component '+str(i)]['RM'] = state_dict['comps'][i]['RMcalibrated']['RM2'][0]
                    RMcaldict['2D-Synth']['Component '+str(i)]['Error'] = state_dict['comps'][i]['RMcalibrated']['RM2'][1]

        

        wdict['rmcal_menu_choices'] = make_rmcal_menu_choices()

    #update scatter comps
    wdict['scattermenu_choices'] = ['Component ' + str(i) for i in range(state_dict['n_comps'])]

    #update scatter comps
    if state_dict['n_comps'] > 1:
        wdict['scintmenu_choices'] = ['All'] + ['Component ' + str(i) for i in range(state_dict['n_comps'])]
    else:
        wdict['scintmenu_choices'] = ['All']

    #update specidx comps
    if state_dict['n_comps'] > 1:
        wdict['specidxmenu_choices'] = ['All'] + ['Component ' + str(i) for i in range(state_dict['n_comps'])]
    else:
        wdict['specidxmenu_choices'] = ['All']

    #update polcomps
    if state_dict['n_comps'] > 1: 
        wdict['polcomp_menu_choices'] = ['All'] + ['Component ' + str(i) for i in range(state_dict['n_comps'])]
    else:
        wdict['polcomp_menu_choices'] = ['All']
    
    return

        


#Exception to quietly exit cell so that we can short circuit output cells
class StopExecution(Exception):
    def _render_traceback_(self):
        return []



"""
Load data state
"""
def restore_screen(savesessionbutton,restoresessionbutton):
    #if save/restore session button clicked
    if savesessionbutton.clicked:
        savestate(Time.now())

    if restoresessionbutton.clicked:
        try:
            restorestate()
        except OSError as ex:
            print("No cached state available")
    
    if len(glob.glob(dirs['cwd'] + '/interface/.current_state/cache_time.pkl')) > 0:
        f = open(dirs['cwd'] + '/interface/.current_state/cache_time.pkl','rb')
        cache = pkl.load(f)
        f.close()
        return "Cached Session: " + str(cache['frb']) + " (" + str(cache['cache_time']) + ")"
    else:
        return "Cached Session: None"

NOFFDEF = 2000
def load_screen(frbfiles_menu,n_t_slider,logn_f_slider,logibox_slider,buff_L_slider_init,buff_R_slider_init,RA_display,DEC_display,DM_init_display,ibeam_display,mjd_display,z_display,updatebutton,filbutton,loadbutton,polcalloadbutton,path=frbpath):
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
    state_dict['DM'] = state_dict['DM0'] + state_dict['dDM']
    state_dict['z'] = FRB_z[FRB_IDS.index(state_dict['ids'])]
    state_dict['RMinput'] = FRB_RM[FRB_IDS.index(state_dict['ids'])]
    state_dict['RMinputerr'] = FRB_RMerr[FRB_IDS.index(state_dict['ids'])]
    state_dict['datadir'] = path + state_dict['ids'] + "_" + state_dict['nickname'] + "/"
    state_dict['buff'] = [buff_L_slider_init.value,buff_R_slider_init.value]
    state_dict['width_native'] = 2**logibox_slider.value
    state_dict['mjd'] = FRB_mjd[FRB_IDS.index(state_dict['ids'])]
    state_dict['heimSNR'] = FRB_heimSNR[FRB_IDS.index(state_dict['ids'])]
    state_dict['RM_gal'] = FRB_RMgal[FRB_IDS.index(state_dict['ids'])]
    state_dict['RM_galerr'] = FRB_RMgalerr[FRB_IDS.index(state_dict['ids'])]
    if ~np.isnan(state_dict['RM_gal']): 
        state_dict['RM_galRA'] = state_dict['RA']
        state_dict['RM_galDEC'] = state_dict['DEC']
    state_dict['RM_ion'] = FRB_RMion[FRB_IDS.index(state_dict['ids'])]
    state_dict['RM_ionerr'] = FRB_RMionerr[FRB_IDS.index(state_dict['ids'])]
    if ~np.isnan(state_dict['RM_ion']):
        state_dict['RM_ionRA'] = state_dict['RA']
        state_dict['RM_ionDEC'] = state_dict['DEC']
        state_dict['RM_ionmjd'] = state_dict['mjd']

    #update displays
    RA_display.data = state_dict['RA']
    DEC_display.data = state_dict['DEC']
    ibeam_display.data = state_dict['ibeam']
    mjd_display.data = state_dict['mjd']
    DM_init_display.data = state_dict['DM0']
    z_display.data = state_dict['z']

    #see if filterbanks exist
    state_dict['fils'] = polbeamform.get_fils(state_dict['ids'],state_dict['nickname'])
    polcals_available = np.sum([((state_dict['datadir'] in str(f)) and ('polcal' in str(f))) for f in state_dict['fils']])
    
    #find beamforming weights date
    state_dict['bfweights'] = polbeamform.get_bfweights(state_dict['ids'])


    #if update button is clicked, refresh FRB data from csv
    if updatebutton.clicked:
        update_FRB_params()

    #if button is clicked, load FRB data and go to next screen
    if loadbutton.clicked:# and (not polcalloadbutton.value):
        if np.isnan(state_dict['z']):wdict['trialz'] = -1
        else: wdict['trialz'] = state_dict['z']
        #wdict['DMINITIALIZED'] = False
        #load data at base resolution
        if polcalloadbutton.value and (polcals_available > 0):
            fixchansfile = state_dict['datadir'] + '/badchans.npy'
            sub_offpulse_mean = False
            state_dict['suff'] = "_dev_polcal"
            fixchansfile_overwrite = False
        else:
            fixchansfile = ''
            sub_offpulse_mean = True
            state_dict['suff'] = "_dev"
            fixchansfile_overwrite = True
            if polcalloadbutton.value: 
                print("Pre-Calibrated Data Not Available, Loading Uncalibrated Stokes Parameters")
        (I,Q,U,V,fobj,timeaxis,freq_test,wav_test,badchans) = dsapol.get_stokes_2D(state_dict['datadir'],state_dict['ids'] + state_dict['suff'],5120,start=12800,n_t=state_dict['base_n_t'],n_f=state_dict['base_n_f'],n_off=int(NOFFDEF//state_dict['base_n_t']),sub_offpulse_mean=sub_offpulse_mean,fixchans=True,verbose=False,fixchansfile=fixchansfile,fixchansfile_overwrite=fixchansfile_overwrite)

        #update DM initialization
        wdict['DM_input_display'] = state_dict['DM0']#wdict['DM_init_display']
        wdict['DM_new_display'] = wdict['DM_init_display'] + wdict['ddm_num']

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
        state_dict['tsamp'] = fobj.header.tsamp
        #state_dict['fobj'] = fobj
        state_dict['base_freq_test'] = freq_test
        state_dict['base_wav_test'] = wav_test
        state_dict['base_time_axis'] = np.arange(I.shape[1])*32.7*state_dict['base_n_t']
        state_dict['badchans'] = badchans

        if polcalloadbutton.value and (polcals_available > 0):

            #save calibrated dyn spectra
            state_dict['base_Ical'] = copy.deepcopy(state_dict['base_I'])
            state_dict['base_Qcal'] = copy.deepcopy(state_dict['base_Q'])
            state_dict['base_Ucal'] = copy.deepcopy(state_dict['base_U'])
            state_dict['base_Vcal'] = copy.deepcopy(state_dict['base_V'])

            #get downsampled versions
            state_dict['I'] = dsapol.avg_time(state_dict['base_I'],state_dict['rel_n_t'])
            state_dict['I'] = dsapol.avg_freq(state_dict['I'],state_dict['rel_n_f'])
            state_dict['Q'] = dsapol.avg_time(state_dict['base_Q'],state_dict['rel_n_t'])
            state_dict['Q'] = dsapol.avg_freq(state_dict['Q'],state_dict['rel_n_f'])
            state_dict['U'] = dsapol.avg_time(state_dict['base_U'],state_dict['rel_n_t'])
            state_dict['U'] = dsapol.avg_freq(state_dict['U'],state_dict['rel_n_f'])
            state_dict['V'] = dsapol.avg_time(state_dict['base_V'],state_dict['rel_n_t'])
            state_dict['V'] = dsapol.avg_freq(state_dict['V'],state_dict['rel_n_f'])


            #default case: not RM calibrated
            state_dict['Ical'] = copy.deepcopy(state_dict['I'])
            state_dict['Qcal'] = copy.deepcopy(state_dict['Q'])
            state_dict['Ucal'] = copy.deepcopy(state_dict['U'])
            state_dict['Vcal'] = copy.deepcopy(state_dict['V'])

            state_dict['IcalRM'] = copy.deepcopy(state_dict['Ical'])
            state_dict['QcalRM'] = copy.deepcopy(state_dict['Qcal'])
            state_dict['UcalRM'] = copy.deepcopy(state_dict['Ucal'])
            state_dict['VcalRM'] = copy.deepcopy(state_dict['Vcal'])

            #get time series
            (state_dict['I_tcal'],state_dict['Q_tcal'],state_dict['U_tcal'],state_dict['V_tcal'],state_dict['I_tcal_err'],state_dict['Q_tcal_err'],state_dict['U_tcal_err'],state_dict['V_tcal_err']) = dsapol.get_stokes_vs_time(state_dict['Ical'],state_dict['Qcal'],state_dict['Ucal'],state_dict['Vcal'],state_dict['width_native'],state_dict['tsamp'],state_dict['n_t'],n_off=int(NOFFDEF//state_dict['n_t']),plot=False,show=False,normalize=True,buff=state_dict['buff'],window=30,error=True,badchans=state_dict['badchans'])

            state_dict['I_tcalRM'] = copy.deepcopy(state_dict['I_tcal'])
            state_dict['Q_tcalRM'] = copy.deepcopy(state_dict['Q_tcal'])
            state_dict['U_tcalRM'] = copy.deepcopy(state_dict['U_tcal'])
            state_dict['V_tcalRM'] = copy.deepcopy(state_dict['V_tcal'])

            state_dict['I_t'] = copy.deepcopy(state_dict['I_tcal'])
            state_dict['Q_t'] = copy.deepcopy(state_dict['Q_tcal'])
            state_dict['U_t'] = copy.deepcopy(state_dict['U_tcal'])
            state_dict['V_t'] = copy.deepcopy(state_dict['V_tcal'])

            state_dict['time_axis'] = 32.7*state_dict['n_t']*np.arange(0,len(state_dict['I_tcal']))

            #get timestart, timestop
            (state_dict['peak'],state_dict['timestart'],state_dict['timestop']) = dsapol.find_peak(state_dict['Ical'],state_dict['width_native'],state_dict['tsamp'],n_t=state_dict['rel_n_t'],peak_range=None,pre_calc_tf=False,buff=state_dict['buff'])

            #get UNWEIGHTED spectrum -- at this point, haven't gotten ideal filter weights yet
            (state_dict['I_fcal_unweighted'],state_dict['Q_fcal_unweighted'],state_dict['U_fcal_unweighted'],state_dict['V_fcal_unweighted']) = dsapol.get_stokes_vs_freq(state_dict['Ical'],state_dict['Qcal'],state_dict['Ucal'],state_dict['Vcal'],state_dict['width_native'],state_dict['tsamp'],
                                                        state_dict['n_f'],state_dict['n_t'],state_dict['freq_test'],
                                                        n_off=int(NOFFDEF//state_dict['n_t']),plot=False,
                                                        normalize=True,buff=state_dict['buff'],weighted=False,
                                                        fobj=FilReader(state_dict['datadir']+state_dict['ids'] + state_dict['suff'] + "_0.fil"))

            state_dict['I_fcalRM_unweighted'] = copy.deepcopy(state_dict['I_fcal_unweighted'])
            state_dict['Q_fcalRM_unweighted'] = copy.deepcopy(state_dict['Q_fcal_unweighted'])
            state_dict['U_fcalRM_unweighted'] = copy.deepcopy(state_dict['U_fcal_unweighted'])
            state_dict['V_fcalRM_unweighted'] = copy.deepcopy(state_dict['V_fcal_unweighted'])


            """#get UNWEIGHTED spectrum at max resolution -- at this point, haven't gotten ideal filter weights yet
            (state_dict['base_I_fcal_unweighted'],state_dict['base_Q_fcal_unweighted'],state_dict['base_U_fcal_unweighted'],state_dict['base_V_fcal_unweighted']) = dsapol.get_stokes_vs_freq(state_dict['base_Ical'],state_dict['base_Qcal'],state_dict['base_Ucal'],state_dict['base_Vcal'],state_dict['width_native'],state_dict['fobj'].header.tsamp,
                                                        state_dict['base_n_f'],state_dict['n_t'],state_dict['freq_test'],
                                                        n_off=int(NOFFDEF//state_dict['n_t']),plot=False,
                                                        normalize=True,buff=state_dict['buff'],weighted=False,
                                                        fobj=state_dict['fobj'])

            state_dict['base_I_fcalRM_unweighted'] = copy.deepcopy(state_dict['base_I_fcal_unweighted'])
            state_dict['base_Q_fcalRM_unweighted'] = copy.deepcopy(state_dict['base_Q_fcal_unweighted'])
            state_dict['base_U_fcalRM_unweighted'] = copy.deepcopy(state_dict['base_U_fcal_unweighted'])
            state_dict['base_V_fcalRM_unweighted'] = copy.deepcopy(state_dict['base_V_fcal_unweighted'])"""


        #state_dict['current_state'] += 1

    #if filbutton is clicked, run the offline beamformer to make fil files
    if filbutton.clicked:
        status = polbeamform.make_filterbanks(state_dict['ids'],state_dict['nickname'],state_dict['bfweights'],state_dict['ibeam'],state_dict['mjd'],state_dict['DM0'])
        print("Submitted Job, status: " + str(status))#bfstatus_display.data = status



    #plot filplot
    pngname = glob.glob(state_dict['datadir'] + state_dict['ids'] + "_parsec.png")
    if len(pngname) != 0:
        #show the existing image
        im = Image.open(pngname[0])
    else:
        #create the image, first find fil file
        filname = glob.glob(dirs['candidates'] + str(state_dict['ids']) + "/Level2/filterbank/" + str(state_dict['ids']) + "_" + str(state_dict['ibeam']) + ".fil")
        if len(filname) == 0:
            #couldn't find file on h23, try to copy from dsastorage
            
            #first try general dir
            os.system("scp " + dirs['dsastorageFRBDir'] + "/" + str(state_dict['ids']) + "/Level2/filterbank/" + str(state_dict['ids']) + "_" + str(state_dict['ibeam']) + ".fil" + state_dict['datadir'])
            filname = glob.glob(state_dict['datadir'] + state_dict['ids'] + "_" + str(state_dict['ibeam']) + ".fil")

        if len(filname) == 0:
            #try dsastorage backups
            
            if state_dict['ibeam'] < 64:
                corr = "corr01"
            elif 64 < state_dict['ibeam'] < 128:
                corr = "corr02"
            elif 128 < state_dict['ibeam'] < 192:
                corr = "corr09"
            else: #192 < state_dict['ibeam'] < 256:
                corr = "corr13"
            



            os.system("scp " + dirs['dsastorageFILDir'] + corr + "/20" + state_dict['ids'][:2] + "_" + str(int(state_dict['ids'][2:4])) + "_" + str(int(state_dict['ids'][4:6])) + "_*/fil_" + state_dict['ids'] + "/" + state_dict['ids'] + "_" + str(state_dict['ibeam']) + ".fil "+ state_dict['datadir'])
            filname = glob.glob(state_dict['datadir'] + state_dict['ids'] + "_" + str(state_dict['ibeam']) + ".fil")
        if len(filname) == 0:
            im = mr.Markdown("## No candidate plot for " + str(state_dict['ids']) + "_"+ str(state_dict['nickname']))

        else:
            c = SkyCoord(ra=state_dict['RA']*u.deg,dec=state_dict['DEC']*u.deg,frame='icrs')
            state_dict['gall'] = c.galactic.l.value
            state_dict['galb'] = c.galactic.b.value
            
            cfpf.custom_filplot(filname[0],
                                state_dict['DM0'],
                                state_dict['width_native'],
                                multibeam=None,
                                figname= state_dict['datadir'] + state_dict['ids'] + "_parsec.png",
                                ndm=32,
                                suptitle='candname:%s  DM:%0.1f  boxcar:%d  ibeam:%d \nMJD:%f  Ra/Dec=%0.1f,%0.1f Gal lon/lat=%0.1f,%0.1f' % (state_dict['ids'], state_dict['DM0'], state_dict['width_native'], state_dict['ibeam'], state_dict['mjd'], state_dict['RA'], state_dict['DEC'], state_dict['gall'], state_dict['galb']),
                                heimsnr=state_dict['heimSNR'],
                                ibeam=state_dict['ibeam'],
                                rficlean=False,
                                nfreq_plot=32,
                                classify=False,
                                heim_raw_tres=1,
                                showplot=False,
                                save_data=False,
                                candname=state_dict['ids'],
                                imjd=state_dict['mjd'],
                                injected=False,
                                fast_classify=False)
            im = Image.open(state_dict['datadir'] + state_dict['ids'] + "_parsec.png")
    
    #update widget dict
    update_wdict([frbfiles_menu,n_t_slider,logn_f_slider,logibox_slider,buff_L_slider_init,buff_R_slider_init,polcalloadbutton],
                    ["frbfiles_menu","n_t_slider","logn_f_slider","logibox_slider","buff_L_slider_init","buff_R_slider_init","polcalloadbutton"],
                    param='value')
    update_wdict([RA_display,DEC_display,DM_init_display,ibeam_display,mjd_display,z_display],
                    ["RA_display","DEC_display","DM_init_display","ibeam_display","mjd_display","z_display"],
                    param='data')
    return im
    
"""
Dedispersion Tuning state
"""
def dedisp_screen(n_t_slider,logn_f_slider,logwindow_slider_init,ddm_num,DMdonebutton,saveplotbutton,DM_showerrs):
    """
    This function updates the dedispersion screen when resolution
    or dm,where='post' step are changed
    """
    #if DMdonebutton.clicked:# and (state_dict['dDM'] != 0):
    """
    if DMdonebutton.clicked:
      
        if state_dict['dDM'] != 0:
            print("Dedispersing base spectra to" + str(state_dict['dDM']) + "...")
            #dedisperse base dyn spectrum
            state_dict['base_I'] = dedisp.dedisperse(state_dict['base_I'],state_dict['dDM'],(32.7e-3)*state_dict['base_n_t'],state_dict['base_freq_test'][0])
            state_dict['base_Q'] = dedisp.dedisperse(state_dict['base_Q'],state_dict['dDM'],(32.7e-3)*state_dict['base_n_t'],state_dict['base_freq_test'][0])
            state_dict['base_U'] = dedisp.dedisperse(state_dict['base_U'],state_dict['dDM'],(32.7e-3)*state_dict['base_n_t'],state_dict['base_freq_test'][0])
            state_dict['base_V'] = dedisp.dedisperse(state_dict['base_V'],state_dict['dDM'],(32.7e-3)*state_dict['base_n_t'],state_dict['base_freq_test'][0])
            
            #reset dm offset
            print("Resetting DM offset...")
            wdict['ddm_num'] = 0
            state_dict['dDM'] = 0
            wdict['DM_input_display'] = state_dict['DM']
            wdict['DM_new_display'] = state_dict['DM']
            #DM_input_display.data = wdict['DM_input_display'] 
     
        if state_dict['DM0'] != state_dict['DM']:
            if updateDM.value:
                #update the DM value in csv file
                update_FRB_DM_params(state_dict['nickname'],state_dict['DM'])

                #make a copy of the current filterbanks
                for i in range(4):
                    os.system("cp " + state_dict['datadir'] + state_dict['ids'] + state_dict['suff'] + "_" + str(i) + ".fil " + state_dict['datadir'] + state_dict['ids'] + state_dict['suff'] + "_DM_" + str(state_dict['DM']) + "_" + str(i) + ".fil ")

                #make new dedispersed filterbanks
                status = polbeamform.make_filterbanks(state_dict['ids'],state_dict['nickname'],state_dict['bfweights'],state_dict['ibeam'],state_dict['mjd'],state_dict['DM'])
                print("Submitted Job, status: " + str(status))#bfstatus_display.data = status
    """

    #dedisperse
    if ddm_num.value != state_dict['dDM']:
        #print("Dedispersing base spectra to" + str(ddm_num.value - state_dict['dDM']) + "...")
        #dedisperse base dyn spectrum
        state_dict['base_I'] = dedisp.dedisperse(state_dict['base_I'],ddm_num.value - state_dict['dDM'],(32.7e-3)*state_dict['base_n_t'],state_dict['base_freq_test'][0])
        state_dict['base_Q'] = dedisp.dedisperse(state_dict['base_Q'],ddm_num.value - state_dict['dDM'],(32.7e-3)*state_dict['base_n_t'],state_dict['base_freq_test'][0])
        state_dict['base_U'] = dedisp.dedisperse(state_dict['base_U'],ddm_num.value - state_dict['dDM'],(32.7e-3)*state_dict['base_n_t'],state_dict['base_freq_test'][0])
        state_dict['base_V'] = dedisp.dedisperse(state_dict['base_V'],ddm_num.value - state_dict['dDM'],(32.7e-3)*state_dict['base_n_t'],state_dict['base_freq_test'][0])

    #update DM step size
    state_dict['dDM'] = ddm_num.value
    #ddm_num.numeric.step = dedisp.get_min_DM_step(n_t_slider.value*state_dict['base_n_t'])

    #update new DM
    #DM_new_display.data = DM_input_display.data + ddm_num.value
    state_dict['DM'] = wdict['DM_input_display'] + state_dict['dDM']#new_display.data
    wdict['DM_new_display'] = state_dict['DM']




    #update time, freq resolution
    state_dict['window'] = 2**logwindow_slider_init.value
    state_dict['rel_n_t'] = n_t_slider.value
    state_dict['rel_n_f'] = (2**logn_f_slider.value)
    state_dict['n_t'] = n_t_slider.value*state_dict['base_n_t']
    state_dict['n_f'] = (2**logn_f_slider.value)*state_dict['base_n_f']
    state_dict['freq_test'] = [state_dict['base_freq_test'][0].reshape(len(state_dict['base_freq_test'][0])//(2**logn_f_slider.value),(2**logn_f_slider.value)).mean(1)]*4
    state_dict['freq_min'] = np.nanmin(state_dict['freq_test'][0])
    state_dict['freq_max'] = np.nanmax(state_dict['freq_test'][0])
    state_dict['freq_cntr'] = (state_dict['freq_max'] + state_dict['freq_min'])/2
    state_dict['I'] = dsapol.avg_time(state_dict['base_I'],n_t_slider.value)#state_dict['n_t'])
    state_dict['I'] = dsapol.avg_freq(state_dict['I'],2**logn_f_slider.value)#state_dict['n_f'])
    state_dict['Q'] = dsapol.avg_time(state_dict['base_Q'],n_t_slider.value)#state_dict['n_t'])
    state_dict['Q'] = dsapol.avg_freq(state_dict['Q'],2**logn_f_slider.value)#state_dict['n_f'])
    state_dict['U'] = dsapol.avg_time(state_dict['base_U'],n_t_slider.value)#state_dict['n_t'])
    state_dict['U'] = dsapol.avg_freq(state_dict['U'],2**logn_f_slider.value)#state_dict['n_f'])
    state_dict['V'] = dsapol.avg_time(state_dict['base_V'],n_t_slider.value)#state_dict['n_t'])
    state_dict['V'] = dsapol.avg_freq(state_dict['V'],2**logn_f_slider.value)#state_dict['n_f'])

    """
    #dedisperse
    print("post-base",state_dict['dDM'],ddm_num.value)
    state_dict['I'] = dedisp.dedisperse(state_dict['I'],state_dict['dDM'],(32.7e-3)*state_dict['n_t'],state_dict['freq_test'][0])
    state_dict['Q'] = dedisp.dedisperse(state_dict['Q'],state_dict['dDM'],(32.7e-3)*state_dict['n_t'],state_dict['freq_test'][0])
    state_dict['U'] = dedisp.dedisperse(state_dict['U'],state_dict['dDM'],(32.7e-3)*state_dict['n_t'],state_dict['freq_test'][0])
    state_dict['V'] = dedisp.dedisperse(state_dict['V'],state_dict['dDM'],(32.7e-3)*state_dict['n_t'],state_dict['freq_test'][0])
    """

    #get time series
    (state_dict['I_t'],state_dict['Q_t'],state_dict['U_t'],state_dict['V_t'],state_dict['I_t_err'],state_dict['Q_t_err'],state_dict['U_t_err'],state_dict['V_t_err']) = dsapol.get_stokes_vs_time(state_dict['I'],state_dict['Q'],state_dict['U'],state_dict['V'],state_dict['width_native'],state_dict['tsamp'],state_dict['n_t'],n_off=int(NOFFDEF//state_dict['n_t']),plot=False,show=False,normalize=True,buff=state_dict['buff'],window=30,error=True,badchans=state_dict['badchans'])
    state_dict['time_axis'] = 32.7*state_dict['n_t']*np.arange(0,len(state_dict['I_t']))

    #get timestart, timestop
    (state_dict['peak'],state_dict['timestart'],state_dict['timestop']) = dsapol.find_peak(state_dict['I'],state_dict['width_native'],state_dict['tsamp'],n_t=n_t_slider.value,peak_range=None,pre_calc_tf=False,buff=state_dict['buff'])
    wdict['comprange_slider_max'] = np.ceil((2*state_dict['window'] + state_dict['timestop'] - state_dict['timestart'])*state_dict['tsamp']*state_dict['n_t']*1e3)
    wdict['comprange_slider'] = [np.around(0.25*(2*state_dict['window'] + state_dict['timestop'] - state_dict['timestart'])*state_dict['tsamp']*state_dict['n_t']*1e3,2),np.around(0.75*(2*state_dict['window'] + state_dict['timestop'] - state_dict['timestart'])*state_dict['tsamp']*state_dict['n_t']*1e3,2)]

    #get UNWEIGHTED spectrum -- at this point, haven't gotten ideal filter weights yet
    (state_dict['I_f_unweighted'],state_dict['Q_f_unweighted'],state_dict['U_f_unweighted'],state_dict['V_f_unweighted']) = dsapol.get_stokes_vs_freq(state_dict['I'],state_dict['Q'],state_dict['U'],state_dict['V'],state_dict['width_native'],state_dict['tsamp'],
                                                        state_dict['n_f'],state_dict['n_t'],state_dict['freq_test'],
                                                        n_off=int(NOFFDEF//state_dict['n_t']),plot=False,
                                                        normalize=True,buff=state_dict['buff'],weighted=False,
                                                        fobj=FilReader(state_dict['datadir']+state_dict['ids'] + state_dict['suff'] + "_0.fil"))


    #display dynamic spectrum
    fig, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 2]},figsize=(18,12))
    c1=a0.step(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3,
            state_dict['I_t'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],label='I',where='post')
    c2=a0.step(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3,
            state_dict['Q_t'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],label='Q',where='post')
    c3=a0.step(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3,
            state_dict['U_t'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],label='U',where='post')
    c4=a0.step(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3,
            state_dict['V_t'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],label='V',where='post')
    if DM_showerrs.value:
        a0.errorbar(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3,
                state_dict['I_t'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],
                yerr=state_dict['I_t_err'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],color=c1[0].get_color(),alpha=1,linestyle='')
        a0.errorbar(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3,
                state_dict['Q_t'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],
                yerr=state_dict['Q_t_err'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],color=c2[0].get_color(),alpha=1,linestyle='')
        a0.errorbar(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3,
                state_dict['U_t'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],
                yerr=state_dict['U_t_err'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],color=c3[0].get_color(),alpha=1,linestyle='')
        a0.errorbar(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3,
                state_dict['V_t'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],
                yerr=state_dict['V_t_err'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],color=c4[0].get_color(),alpha=1,linestyle='')
    

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
    if saveplotbutton.clicked:
        try:
            plt.savefig(state_dict['datadir'] + state_dict['ids'] + "_" + state_dict['nickname'] + "_DMplot_PARSEC.pdf")
            print("Saved Figure to h24:" + state_dict['datadir'] + state_dict['ids'] + "_" + state_dict['nickname'] + "_DMplot_PARSEC.pdf")
        except Exception as ex:
            print("Save Failed: " + str(ex))
    plt.show()


    if DMdonebutton.clicked:
        if state_dict['DM0'] != state_dict['DM']:
            #update the DM value in csv file
            update_FRB_DM_params(state_dict['nickname'],state_dict['DM'])

            #make a copy of the current filterbanks
            for i in range(4):
                os.system("cp " + state_dict['datadir'] + state_dict['ids'] + state_dict['suff'] + "_" + str(i) + ".fil " + state_dict['datadir'] + state_dict['ids'] + state_dict['suff'] + "_DM_" + str(state_dict['DM']) + "_" + str(i) + ".fil ")

            #make new dedispersed filterbanks
            status = polbeamform.make_filterbanks(state_dict['ids'],state_dict['nickname'],state_dict['bfweights'],state_dict['ibeam'],state_dict['mjd'],state_dict['DM'])
            print("Submitted Job, status: " + str(status))#bfstatus_display.data = status

    #update widget dict
    update_wdict([n_t_slider,logn_f_slider,logwindow_slider_init,DM_showerrs,ddm_num],
                ["n_t_slider","logn_f_slider","logwindow_slider_init","DM_showerrs","ddm_num"],
                param='value')
    
    #update_wdict([DM_input_display,DM_new_display],
    #            ["DM_input_display","DM_new_display"],
    #            param='data')

    return #ddm_num


"""
Calibration state
"""

def polcal_screen(polcaldate_menu,polcaldate_create_menu,polcaldate_bf_menu,polcaldate_findbeams_menu,obsid3C48_menu,obsid3C286_menu,
        polcalbutton,polcopybutton,bfcal_button,findbeams_button,filcalbutton,ParA_display,
        edgefreq_slider,breakfreq_slider,sf_window_weight_cals,sf_order_cals,peakheight_slider,peakwidth_slider,polyfitorder_slider,
        ratio_edgefreq_slider,ratio_breakfreq_slider,ratio_sf_window_weight_cals,ratio_sf_order_cals,ratio_peakheight_slider,ratio_peakwidth_slider,ratio_polyfitorder_slider,
        phase_sf_window_weight_cals,phase_sf_order_cals,phase_peakheight_slider,phase_peakwidth_slider,phase_polyfitorder_slider,savecalsolnbutton,
                                                         sfflag,polyfitflag,ratio_sfflag,ratio_polyfitflag,phase_sfflag,phase_polyfitflag,saveplotbutton,polcalprocs,savepolcalfilbutton):
    
    """
    This function updates the polarization calibration screen
    whenever the cal file is selected
    """
    
    if polcaldate_menu.value != "":
        #update polcal parameters in state dict
        state_dict['gxx'],state_dict['gyy'],state_dict['cal_freq_axis'] = polcal.read_polcal(polcaldate_menu.value)
        
        #needd to downsample to base resolution
        state_dict['gxx'] = np.nanmean(state_dict['gxx'].reshape((len(state_dict['gxx'])//state_dict['base_n_f'],state_dict['base_n_f'])),axis=1)
        state_dict['gyy'] = np.nanmean(state_dict['gyy'].reshape((len(state_dict['gyy'])//state_dict['base_n_f'],state_dict['base_n_f'])),axis=1)
        state_dict['cal_freq_axis'] = np.nanmean(state_dict['cal_freq_axis'].reshape((len(state_dict['cal_freq_axis'])//state_dict['base_n_f'],state_dict['base_n_f'])),axis=1)

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
    polcal_dict['maxProcesses'] = int(polcalprocs.value)

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


        f = open(polcal.logfile,"w")
        print("start",file=f)

        #calibrate at native resolution
        state_dict['base_Ical'],state_dict['base_Qcal'],state_dict['base_Ucal'],state_dict['base_Vcal'] = dsapol.calibrate(state_dict['base_I'],state_dict['base_Q'],state_dict['base_U'],state_dict['base_V'],(state_dict['gxx'],state_dict['gyy']),stokes=True,multithread=True,maxProcesses=int(polcalprocs.value),bad_chans=state_dict['badchans'])
        print("done calibrating...",file=f)

        #parallactic angle calibration
        state_dict['base_Ical'],state_dict['base_Qcal'],state_dict['base_Ucal'],state_dict['base_Vcal'],state_dict['ParA'] = dsapol.calibrate_angle(state_dict['base_Ical'],state_dict['base_Qcal'],state_dict['base_Ucal'],state_dict['base_Vcal'],FilReader(state_dict['datadir']+state_dict['ids'] + state_dict['suff'] + "_0.fil"),state_dict['ibeam'],state_dict['RA'],state_dict['DEC'])
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
        
        
        #default case: not RM calibrated
        state_dict['IcalRM'] = copy.deepcopy(state_dict['Ical'])
        state_dict['QcalRM'] = copy.deepcopy(state_dict['Qcal'])
        state_dict['UcalRM'] = copy.deepcopy(state_dict['Ucal'])
        state_dict['VcalRM'] = copy.deepcopy(state_dict['Vcal'])
       
        print("done downsampling...",file=f)
        f.close()
        #get time series
        (state_dict['I_tcal'],state_dict['Q_tcal'],state_dict['U_tcal'],state_dict['V_tcal'],state_dict['I_tcal_err'],state_dict['Q_tcal_err'],state_dict['U_tcal_err'],state_dict['V_tcal_err']) = dsapol.get_stokes_vs_time(state_dict['Ical'],state_dict['Qcal'],state_dict['Ucal'],state_dict['Vcal'],state_dict['width_native'],state_dict['tsamp'],state_dict['n_t'],n_off=int(NOFFDEF//state_dict['n_t']),plot=False,show=False,normalize=True,buff=state_dict['buff'],window=30,error=True,badchans=state_dict['badchans'])

        state_dict['I_tcalRM'] = copy.deepcopy(state_dict['I_tcal'])
        state_dict['Q_tcalRM'] = copy.deepcopy(state_dict['Q_tcal'])
        state_dict['U_tcalRM'] = copy.deepcopy(state_dict['U_tcal'])
        state_dict['V_tcalRM'] = copy.deepcopy(state_dict['V_tcal'])

        state_dict['time_axis'] = 32.7*state_dict['n_t']*np.arange(0,len(state_dict['I_tcal']))

        #get timestart, timestop
        (state_dict['peak'],state_dict['timestart'],state_dict['timestop']) = dsapol.find_peak(state_dict['Ical'],state_dict['width_native'],state_dict['tsamp'],n_t=state_dict['rel_n_t'],peak_range=None,pre_calc_tf=False,buff=state_dict['buff'])


        #get UNWEIGHTED spectrum -- at this point, haven't gotten ideal filter weights yet
        (state_dict['I_fcal_unweighted'],state_dict['Q_fcal_unweighted'],state_dict['U_fcal_unweighted'],state_dict['V_fcal_unweighted']) = dsapol.get_stokes_vs_freq(state_dict['Ical'],state_dict['Qcal'],state_dict['Ucal'],state_dict['Vcal'],state_dict['width_native'],state_dict['tsamp'],
                                                        state_dict['n_f'],state_dict['n_t'],state_dict['freq_test'],
                                                        n_off=int(NOFFDEF//state_dict['n_t']),plot=False,
                                                        normalize=True,buff=state_dict['buff'],weighted=False,
                                                        fobj=FilReader(state_dict['datadir']+state_dict['ids'] + state_dict['suff'] + "_0.fil")) 

        state_dict['I_fcalRM_unweighted'] = copy.deepcopy(state_dict['I_fcal_unweighted'])
        state_dict['Q_fcalRM_unweighted'] = copy.deepcopy(state_dict['Q_fcal_unweighted'])
        state_dict['U_fcalRM_unweighted'] = copy.deepcopy(state_dict['U_fcal_unweighted'])
        state_dict['V_fcalRM_unweighted'] = copy.deepcopy(state_dict['V_fcal_unweighted'])

        """#get UNWEIGHTED spectrum at max resolution -- at this point, haven't gotten ideal filter weights yet
        (state_dict['base_I_fcal_unweighted'],state_dict['base_Q_fcal_unweighted'],state_dict['base_U_fcal_unweighted'],state_dict['base_V_fcal_unweighted']) = dsapol.get_stokes_vs_freq(state_dict['base_Ical'],state_dict['base_Qcal'],state_dict['base_Ucal'],state_dict['base_Vcal'],state_dict['width_native'],state_dict['fobj'].header.tsamp,
                                                        state_dict['base_n_f'],state_dict['n_t'],state_dict['freq_test'],
                                                        n_off=int(NOFFDEF//state_dict['n_t']),plot=False,
                                                        normalize=True,buff=state_dict['buff'],weighted=False,
                                                        fobj=state_dict['fobj'])

        state_dict['base_I_fcalRM_unweighted'] = copy.deepcopy(state_dict['base_I_fcal_unweighted'])
        state_dict['base_Q_fcalRM_unweighted'] = copy.deepcopy(state_dict['base_Q_fcal_unweighted'])
        state_dict['base_U_fcalRM_unweighted'] = copy.deepcopy(state_dict['base_U_fcal_unweighted'])
        state_dict['base_V_fcalRM_unweighted'] = copy.deepcopy(state_dict['base_V_fcal_unweighted'])"""

        #state_dict['current_state'] += 1
    if savepolcalfilbutton.clicked:
        if len(glob.glob(state_dict['datadir'] + '/badchans.npy'))>0:
            fixchansfile = state_dict['datadir'] + '/badchans.npy'
            fixchansfile_overwrite = False
        else:
            fixchansfile = ""
            fixchansfile_overwrite = True

        polcal.make_polcal_filterbanks(state_dict['datadir'],state_dict['datadir'],state_dict['ids'],state_dict['polcalfile'],state_dict['suff'],state_dict['suff']+"_polcal",20480,int(NOFFDEF)+12800,True,maxProcesses=polcal_dict['maxProcesses'],fixchans=True,fixchansfile=fixchansfile,fixchansfile_overwrite=fixchansfile_overwrite,verbose=False,background=True)

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
                phase_peakwidth_slider,phase_polyfitorder_slider,sfflag,polyfitflag,ratio_sfflag,ratio_polyfitflag,phase_sfflag,phase_polyfitflag,polcalprocs],
                ["polcaldate_menu","polcaldate_create_menu","polcaldate_bf_menu","polcaldate_findbeams_menu","obsid3C48_menu","obsid3C286_menu",
                "edgefreq_slider","breakfreq_slider","sf_window_weight_cals","sf_order_cals","peakheight_slider","peakwidth_slider","polyfitorder_slider",
                "ratio_edgefreq_slider","ratio_breakfreq_slider","ratio_sf_window_weight_cals","ratio_sf_order_cals","ratio_peakheight_slider",
                "ratio_peakwidth_slider","ratio_polyfitorder_slider","phase_sf_window_weight_cals","phase_sf_order_cals","phase_peakheight_slider",
                "phase_peakwidth_slider","phase_polyfitorder_slider","sfflag","polyfitflag","ratio_sfflag","ratio_polyfitflag","phase_sfflag","phase_polyfitflag","polcalprocs"],
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
    if saveplotbutton.clicked:
        try:
            plt.savefig(polcal.output_path + "3C48_" + polcal_dict['polcal_findbeams_file'] + "/" + str(polcal_dict['polcal_findbeams_file']) + "_polcal_beamselection_PARSEC.pdf")
            plt.savefig(polcal.output_path + "3C286_" + polcal_dict['polcal_findbeams_file'] + "/" + str(polcal_dict['polcal_findbeams_file']) + "_polcal_beamselection_PARSEC.pdf")
            print("Saved Figure to h24:" + polcal.output_path + "3C48_" + polcal_dict['polcal_findbeams_file'] + "/" + str(polcal_dict['polcal_findbeams_file']) + "_polcal_beamselection_PARSEC.pdf")
            print("Saved Figure to h24:" + polcal.output_path + "3C286_" + polcal_dict['polcal_findbeams_file'] + "/" + str(polcal_dict['polcal_findbeams_file']) + "_polcal_beamselection_PARSEC.pdf")
        except Exception as ex:
            print("Save Failed: " + str(ex))
    plt.show()

    return beam_dict_3C48,beam_dict_3C286


def polcal_screen2(polcaldate_menu,polcaldate_create_menu,polcaldate_bf_menu,polcaldate_findbeams_menu,obsid3C48_menu,obsid3C286_menu,
        polcalbutton,polcopybutton,bfcal_button,findbeams_button,filcalbutton,ParA_display,
        edgefreq_slider,breakfreq_slider,sf_window_weight_cals,sf_order_cals,peakheight_slider,peakwidth_slider,polyfitorder_slider,
        ratio_edgefreq_slider,ratio_breakfreq_slider,ratio_sf_window_weight_cals,ratio_sf_order_cals,ratio_peakheight_slider,ratio_peakwidth_slider,ratio_polyfitorder_slider,
        phase_sf_window_weight_cals,phase_sf_order_cals,phase_peakheight_slider,phase_peakwidth_slider,phase_polyfitorder_slider,savecalsolnbutton,
                                                         sfflag,polyfitflag,ratio_sfflag,ratio_polyfitflag,phase_sfflag,phase_polyfitflag,beam_dict_3C48,beam_dict_3C286,saveplotbutton):


    #if make filt button pushed, make filterbanks for pol cals
    if filcalbutton.clicked:

        #make 3C48 filterbanks
        for k in beam_dict_3C48.keys():
            make_cal_filterbanks('3C48',polcal_dict['polcal_findbeams_file'],str(k)[4:],beam_dict_3C48[k]['bf_weights'],beam_dict_3C48[k]['beam'],beam_dict_3C48[k]['mjd'])

        #make 3C286 filterbanks
        for k in beam_dict_3C286.keys():
            make_cal_filterbanks('3C286',polcal_dict['polcal_findbeams_file'],str(k)[5:],beam_dict_3C286[k]['bf_weights'],beam_dict_3C286[k]['beam'],beam_dict_3C286[k]['mjd'])


    #if make solution button pushed, make solution and plot
    if ((polcal_dict['cal_name_3C48'] != "" or polcal_dict['cal_name_3C286'] != "") and polcal_dict['polcal_findbeams_file'] != "") or state_dict['polcalfile'] != "":
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
        last_gxx,last_gyy,last_cal_freq_axis = polcal.read_polcal('POLCAL_PARAMETERS_' + last_caldate + '.csv')#,fit=False)
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
    if ((polcal_dict['cal_name_3C48'] != "" or polcal_dict['cal_name_3C286'] != "") and polcal_dict['polcal_findbeams_file'] != "") or state_dict['polcalfile'] != "":
        plt.subplots_adjust(hspace=0)
        if saveplotbutton.clicked:
            try:
                plt.savefig(polcal.default_path + "POLCAL_PARAMETERS_" + state_dict['polcalfile'] + ".pdf")
                print("Saved Figure to h24: " + polcal.default_path + "POLCAL_PARAMETERS_" + state_dict['polcalfile'] + ".pdf")
            except Exception as ex:
                print("Save Failed: " + str(ex))
        plt.show()


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
    
    if ((polcal_dict['cal_name_3C48'] != "" or polcal_dict['cal_name_3C286'] != "") and polcal_dict['polcal_findbeams_file'] != "") or state_dict['polcalfile'] != "":
        return fig
    
    return #beam_dict_3C48,beam_dict_3C286 #return these to prevent recalculating the beamformer weights isot


"""
Filter Weights State
"""

def filter_screen(logwindow_slider,logibox_slider,buff_L_slider,buff_R_slider,ncomps_num,comprange_slider,nextcompbutton,donecompbutton,avger_w_slider,sf_window_weights_slider,multipeaks,multipeaks_height_slider,fluxestbutton, Iflux_display,Qflux_display,Uflux_display,Vflux_display,filt_showerrs):
    

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
    if state_dict['current_comp'] not in state_dict['comps'].keys():
        state_dict['comps'][state_dict['current_comp']] = dict()
    Ip = copy.deepcopy(state_dict['Ical'])
    Qp = copy.deepcopy(state_dict['Qcal'])
    Up = copy.deepcopy(state_dict['Ucal'])
    Vp = copy.deepcopy(state_dict['Vcal'])
    mask = np.zeros(state_dict['Ical'].shape)
    for k in range(state_dict['current_comp']):
        #mask = np.zeros(state_dict['Ical'].shape)
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
                                                                                    state_dict['tsamp'],n_t=state_dict['n_t'],
                                                                                    peak_range=None,pre_calc_tf=False,
                                                                                    buff=state_dict['comps'][state_dict['current_comp']]['buff'])
    

    state_dict['comps'][state_dict['current_comp']]['left_lim'] = np.argmin(np.abs(state_dict['time_axis']*1e-3 - (state_dict['comps'][state_dict['current_comp']]['timestart']*state_dict['n_t']*32.7e-3 - state_dict['window']*state_dict['n_t']*32.7e-3 + comprange_slider.value[0])))
    state_dict['comps'][state_dict['current_comp']]['right_lim'] = np.argmin(np.abs(state_dict['time_axis']*1e-3 - (state_dict['comps'][state_dict['current_comp']]['timestart']*state_dict['n_t']*32.7e-3 - state_dict['window']*state_dict['n_t']*32.7e-3 + comprange_slider.value[1])))

    (I_tcal,Q_tcal,U_tcal,V_tcal,I_tcal_errs,Q_tcal_errs,U_tcal_errs,V_tcal_errs) = dsapol.get_stokes_vs_time(Ip,Qp,Up,Vp,state_dict['comps'][state_dict['current_comp']]['width_native'],
                                                state_dict['tsamp'],state_dict['n_t'],n_off=int(NOFFDEF//state_dict['n_t']),
                                                plot=False,show=False,normalize=True,buff=state_dict['comps'][state_dict['current_comp']]['buff'],window=30,error=True,badchans=state_dict['badchans'])

    state_dict['comps'][state_dict['current_comp']]['weights'] = dsapol.get_weights_1D(I_tcal,Q_tcal,U_tcal,V_tcal,
                                                                                state_dict['comps'][state_dict['current_comp']]['timestart'],
                                                                                state_dict['comps'][state_dict['current_comp']]['timestop'],
                                                                                state_dict['comps'][state_dict['current_comp']]['width_native'],
                                                                                state_dict['tsamp'],1,state_dict['n_t'],
                                                                                state_dict['freq_test'],state_dict['time_axis'],
                                                                                FilReader(state_dict['datadir']+state_dict['ids'] + state_dict['suff'] + "_0.fil"),
                                                                                n_off=int(NOFFDEF//state_dict['n_t']),buff=state_dict['buff'],
                                                                                n_t_weight=state_dict['comps'][state_dict['current_comp']]['avger_w'],
                                                                sf_window_weights=state_dict['comps'][state_dict['current_comp']]['sf_window_weights'],
                                                                padded=True,norm=False)

    #get calibrated WEIGHTED spectra
    (state_dict['comps'][state_dict['current_comp']]['I_fcal'],state_dict['comps'][state_dict['current_comp']]['Q_fcal'],state_dict['comps'][state_dict['current_comp']]['U_fcal'],state_dict['comps'][state_dict['current_comp']]['V_fcal']) = dsapol.get_stokes_vs_freq(Ip,Qp,Up,Vp,state_dict['comps'][state_dict['current_comp']]['width_native'],state_dict['tsamp'],
                                                        state_dict['n_f'],state_dict['n_t'],state_dict['freq_test'],n_off=int(NOFFDEF//state_dict['n_t']),plot=False,
                                                        normalize=True,buff=state_dict['comps'][state_dict['current_comp']]['buff'],weighted=True,input_weights=state_dict['comps'][state_dict['current_comp']]['weights'],
                                                        fobj=FilReader(state_dict['datadir']+state_dict['ids'] + state_dict['suff'] + "_0.fil"))
    state_dict['comps'][state_dict['current_comp']]['I_fcalRM'] = copy.deepcopy(state_dict['comps'][state_dict['current_comp']]['I_fcal'])
    state_dict['comps'][state_dict['current_comp']]['Q_fcalRM'] = copy.deepcopy(state_dict['comps'][state_dict['current_comp']]['Q_fcal'])
    state_dict['comps'][state_dict['current_comp']]['U_fcalRM'] = copy.deepcopy(state_dict['comps'][state_dict['current_comp']]['U_fcal'])
    state_dict['comps'][state_dict['current_comp']]['V_fcalRM'] = copy.deepcopy(state_dict['comps'][state_dict['current_comp']]['V_fcal'])





    if multipeaks.value:
        height = np.max(state_dict['comps'][state_dict['current_comp']]['weights'])*multipeaks_height_slider.value
        pks,props = find_peaks(state_dict['comps'][state_dict['current_comp']]['weights'],
                                height=height)
        FWHM,heights,intL,intR = peak_widths(state_dict['comps'][state_dict['current_comp']]['weights'],pks)
        state_dict['comps'][state_dict['current_comp']]['intL'] = intL[0]
        state_dict['comps'][state_dict['current_comp']]['intR'] = intR[-1]
        state_dict['comps'][state_dict['current_comp']]['FWHM'] = state_dict['comps'][state_dict['current_comp']]['intR'] - state_dict['comps'][state_dict['current_comp']]['intL']
    else:
        state_dict['comps'][state_dict['current_comp']]['FWHM'],heights,state_dict['comps'][state_dict['current_comp']]['intL'],state_dict['comps'][state_dict['current_comp']]['intR'] = peak_widths(state_dict['comps'][state_dict['current_comp']]['weights'],
                                    [np.argmax(state_dict['comps'][state_dict['current_comp']]['weights'])])

    state_dict['comps'][state_dict['current_comp']]['intL'] = int(state_dict['comps'][state_dict['current_comp']]['intL'])
    state_dict['comps'][state_dict['current_comp']]['intR'] = int(state_dict['comps'][state_dict['current_comp']]['intR'])
    state_dict['comps'][state_dict['current_comp']]['intLbuffer'] = 0
    state_dict['comps'][state_dict['current_comp']]['intRbuffer'] = 0
    state_dict['comps'][state_dict['current_comp']]['FWHM'] = int(state_dict['comps'][state_dict['current_comp']]['FWHM'])

    #compute S/N and display
    state_dict['comps'][state_dict['current_comp']]['S/N'] = filt.get_SNR(I_tcal,state_dict['comps'][state_dict['current_comp']]['weights'],
                                                                    state_dict['comps'][state_dict['current_comp']]['timestart'],
                                                                    state_dict['comps'][state_dict['current_comp']]['timestop'])
    
    
    #update tables  
    if state_dict['n_comps'] > 1:
        df.loc[str(state_dict['current_comp'])] = [state_dict['comps'][state_dict['current_comp']]['buff'][0],
                                                   state_dict['comps'][state_dict['current_comp']]['buff'][1],
                                                   state_dict['comps'][state_dict['current_comp']]['avger_w'],
                                                   state_dict['comps'][state_dict['current_comp']]['sf_window_weights'],
                                                   state_dict['comps'][state_dict['current_comp']]['left_lim'],
                                                   state_dict['comps'][state_dict['current_comp']]['right_lim'],
                                                   state_dict['comps'][state_dict['current_comp']]['S/N']]

        if str(state_dict['current_comp']) not in RMdf.index.tolist():
            RMdf.loc[str(state_dict['current_comp'])] = [np.nan]*6

        if 'RMcalibrated' not in state_dict['comps'][state_dict['current_comp']].keys():
            state_dict['comps'][state_dict['current_comp']]['RMcalibrated'] = dict()
            state_dict['comps'][state_dict['current_comp']]['RMcalibrated']['RMsnrs1'] = np.nan*np.ones(int(wdict['nRM_num']))
            state_dict['comps'][state_dict['current_comp']]['RMcalibrated']['RM_tools_snrs'] = np.nan*np.ones(int(2*wdict['maxRM_num']/wdict['dRM_tools']))
            state_dict['comps'][state_dict['current_comp']]['RMcalibrated']['RMsnrs1zoom'] = np.nan*np.ones(int(wdict['nRM_num_zoom']))
            state_dict['comps'][state_dict['current_comp']]['RMcalibrated']['RM_tools_snrszoom'] = np.nan*np.ones(int(2*wdict['RM_window_zoom']/wdict['dRM_tools_zoom']))
            state_dict['comps'][state_dict['current_comp']]['RMcalibrated']['RMsnrs2'] = np.nan*np.ones(int(wdict['nRM_num_zoom']))
            state_dict['comps'][state_dict['current_comp']]['RMcalibrated']["RM2"] = [np.nan,np.nan,np.nan,np.nan]
            state_dict['comps'][state_dict['current_comp']]['RMcalibrated']["RM1"] = [np.nan,np.nan]
            state_dict['comps'][state_dict['current_comp']]['RMcalibrated']["RM1zoom"] = [np.nan,np.nan]
            state_dict['comps'][state_dict['current_comp']]['RMcalibrated']["RM_tools"] = [np.nan,np.nan]
            state_dict['comps'][state_dict['current_comp']]['RMcalibrated']["RM_toolszoom"] = [np.nan,np.nan]
            state_dict['comps'][state_dict['current_comp']]["RMcalibrated"]["RMerrfit"] = np.nan
            state_dict['comps'][state_dict['current_comp']]["RMcalibrated"]["trial_RM1"] = np.linspace(wdict['minRM_num'],wdict['maxRM_num'],int(wdict['nRM_num']))
            state_dict['comps'][state_dict['current_comp']]["RMcalibrated"]["trial_RM2"] = np.linspace(-wdict['RM_window_zoom'],wdict['RM_window_zoom'],int(wdict['nRM_num_zoom']))
            state_dict['comps'][state_dict['current_comp']]["RMcalibrated"]["trial_RM_tools"] = np.arange(-wdict['maxRM_num'],wdict['maxRM_num'],wdict['dRM_tools'])
            state_dict['comps'][state_dict['current_comp']]["RMcalibrated"]["trial_RM_toolszoom"] = np.arange(-wdict['RM_window_zoom'],wdict['RM_window_zoom'],wdict['dRM_tools_zoom'])
        
        df.loc["All"] = [np.nan,
                                                   np.nan,
                                                   np.nan,
                                                   np.nan,
                                                   np.nan,
                                                   np.nan,
                                                   np.nan]
    else:
        df.loc["All"] = [state_dict['comps'][state_dict['current_comp']]['buff'][0],
                                                   state_dict['comps'][state_dict['current_comp']]['buff'][1],
                                                   state_dict['comps'][state_dict['current_comp']]['avger_w'],
                                                   state_dict['comps'][state_dict['current_comp']]['sf_window_weights'],
                                                   state_dict['comps'][state_dict['current_comp']]['left_lim'],
                                                   state_dict['comps'][state_dict['current_comp']]['right_lim'],
                                                   state_dict['comps'][state_dict['current_comp']]['S/N']]



    #update initial parameters for scattering analysis
    if not state_dict['scatter_init']:
        scatter_reset_initvals(state_dict['current_comp'])

    #display masked dynamic spectrum
    fig, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 2]},figsize=(18,12))
    c1=a0.step(state_dict['time_axis'][state_dict['comps'][state_dict['current_comp']]['timestart']-state_dict['window']:state_dict['comps'][state_dict['current_comp']]['timestop']+state_dict['window']]*1e-3,
            I_tcal[state_dict['comps'][state_dict['current_comp']]['timestart']-state_dict['window']:state_dict['comps'][state_dict['current_comp']]['timestop']+state_dict['window']],label='I',where='post')
    c2=a0.step(state_dict['time_axis'][state_dict['comps'][state_dict['current_comp']]['timestart']-state_dict['window']:state_dict['comps'][state_dict['current_comp']]['timestop']+state_dict['window']]*1e-3,
            Q_tcal[state_dict['comps'][state_dict['current_comp']]['timestart']-state_dict['window']:state_dict['comps'][state_dict['current_comp']]['timestop']+state_dict['window']],label='Q',where='post')
    c3=a0.step(state_dict['time_axis'][state_dict['comps'][state_dict['current_comp']]['timestart']-state_dict['window']:state_dict['comps'][state_dict['current_comp']]['timestop']+state_dict['window']]*1e-3,
            U_tcal[state_dict['comps'][state_dict['current_comp']]['timestart']-state_dict['window']:state_dict['comps'][state_dict['current_comp']]['timestop']+state_dict['window']],label='U',where='post')
    c4=a0.step(state_dict['time_axis'][state_dict['comps'][state_dict['current_comp']]['timestart']-state_dict['window']:state_dict['comps'][state_dict['current_comp']]['timestop']+state_dict['window']]*1e-3,
            V_tcal[state_dict['comps'][state_dict['current_comp']]['timestart']-state_dict['window']:state_dict['comps'][state_dict['current_comp']]['timestop']+state_dict['window']],label='V',where='post')
    a0.step(state_dict['time_axis'][state_dict['comps'][state_dict['current_comp']]['timestart']-state_dict['window']:state_dict['comps'][state_dict['current_comp']]['timestop']+state_dict['window']]*1e-3,
            state_dict['comps'][state_dict['current_comp']]['weights'][state_dict['comps'][state_dict['current_comp']]['timestart']-state_dict['window']:state_dict['comps'][state_dict['current_comp']]['timestop']+state_dict['window']]*np.max(I_tcal)/np.max(state_dict['comps'][state_dict['current_comp']]['weights'][state_dict['comps'][state_dict['current_comp']]['timestart']-state_dict['window']:state_dict['comps'][state_dict['current_comp']]['timestop']+state_dict['window']]),label='weights',color='purple',linewidth=4,where='post')

    if filt_showerrs.value:
        a0.errorbar(state_dict['time_axis'][state_dict['comps'][state_dict['current_comp']]['timestart']-state_dict['window']:state_dict['comps'][state_dict['current_comp']]['timestop']+state_dict['window']]*1e-3,
                I_tcal[state_dict['comps'][state_dict['current_comp']]['timestart']-state_dict['window']:state_dict['comps'][state_dict['current_comp']]['timestop']+state_dict['window']],
                yerr=I_tcal_errs[state_dict['comps'][state_dict['current_comp']]['timestart']-state_dict['window']:state_dict['comps'][state_dict['current_comp']]['timestop']+state_dict['window']],
                color=c1[0].get_color(),alpha=1,linestyle='')
        a0.errorbar(state_dict['time_axis'][state_dict['comps'][state_dict['current_comp']]['timestart']-state_dict['window']:state_dict['comps'][state_dict['current_comp']]['timestop']+state_dict['window']]*1e-3,
                Q_tcal[state_dict['comps'][state_dict['current_comp']]['timestart']-state_dict['window']:state_dict['comps'][state_dict['current_comp']]['timestop']+state_dict['window']],
                yerr=Q_tcal_errs[state_dict['comps'][state_dict['current_comp']]['timestart']-state_dict['window']:state_dict['comps'][state_dict['current_comp']]['timestop']+state_dict['window']],
                color=c2[0].get_color(),alpha=1,linestyle='')
        a0.errorbar(state_dict['time_axis'][state_dict['comps'][state_dict['current_comp']]['timestart']-state_dict['window']:state_dict['comps'][state_dict['current_comp']]['timestop']+state_dict['window']]*1e-3,
                U_tcal[state_dict['comps'][state_dict['current_comp']]['timestart']-state_dict['window']:state_dict['comps'][state_dict['current_comp']]['timestop']+state_dict['window']],
                yerr=U_tcal_errs[state_dict['comps'][state_dict['current_comp']]['timestart']-state_dict['window']:state_dict['comps'][state_dict['current_comp']]['timestop']+state_dict['window']],
                color=c3[0].get_color(),alpha=1,linestyle='')
        a0.errorbar(state_dict['time_axis'][state_dict['comps'][state_dict['current_comp']]['timestart']-state_dict['window']:state_dict['comps'][state_dict['current_comp']]['timestop']+state_dict['window']]*1e-3,
                U_tcal[state_dict['comps'][state_dict['current_comp']]['timestart']-state_dict['window']:state_dict['comps'][state_dict['current_comp']]['timestop']+state_dict['window']],
                yerr=U_tcal_errs[state_dict['comps'][state_dict['current_comp']]['timestart']-state_dict['window']:state_dict['comps'][state_dict['current_comp']]['timestop']+state_dict['window']],
                color=c4[0].get_color(),alpha=1,linestyle='')

    #plot weights from others
    for k in range(state_dict['current_comp']):

        a0.step(state_dict['time_axis'][state_dict['comps'][state_dict['current_comp']]['timestart']-state_dict['window']:state_dict['comps'][state_dict['current_comp']]['timestop']+state_dict['window']]*1e-3, 
                state_dict['comps'][k]['weights'][state_dict['comps'][state_dict['current_comp']]['timestart']-state_dict['window']:state_dict['comps'][state_dict['current_comp']]['timestop']+state_dict['window']]*state_dict['I_tcal'][state_dict['comps'][k]['peak']]/state_dict['comps'][k]['weights'][state_dict['comps'][k]['peak']],
                linewidth=4,color='purple',linestyle='--',where='post')
    
    a0.set_xlim(32.7*state_dict['n_t']*state_dict['timestart']*1e-3 - state_dict['window']*32.7*state_dict['n_t']*1e-3,
            32.7*state_dict['n_t']*state_dict['timestop']*1e-3 + state_dict['window']*32.7*state_dict['n_t']*1e-3)
    a0.set_xticks([])
    a0.legend(loc="upper right")
    a0.axvline(state_dict['time_axis'][state_dict['comps'][state_dict['current_comp']]['left_lim']]*1e-3,color='red')
    a0.axvline(state_dict['time_axis'][state_dict['comps'][state_dict['current_comp']]['right_lim']]*1e-3,color='red')
    a0.axvline(state_dict['time_axis'][state_dict['comps'][state_dict['current_comp']]['intL']]*1e-3,color='purple')
    a0.axvline(state_dict['time_axis'][state_dict['comps'][state_dict['current_comp']]['intR']]*1e-3,color='purple')
    if multipeaks.value:
        a0.axhline(height,color='purple',linestyle='--')

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
        state_dict['S/N'] = filt.get_SNR(state_dict['I_tcal'],state_dict['weights'],state_dict['timestart'],state_dict['timestop'])
    
        #get edge conditions from first and last components
        ts1 = []
        ts2 = []
        b1 = []
        b2 = []
        l1 = []
        l2 = []
        for i in state_dict['comps'].keys():
            ts1.append(state_dict['comps'][i]['timestart'])
            ts2.append(state_dict['comps'][i]['timestop'])
            b1.append(state_dict['comps'][i]['buff'][0])
            b2.append(state_dict['comps'][i]['buff'][1])
            l1.append(state_dict['comps'][i]['intL'])
            l2.append(state_dict['comps'][i]['intR'])
        first = np.argmin(ts1)
        last = np.argmax(ts2)
        state_dict['timestart'] = ts1[first]
        state_dict['timestop'] = ts2[last]
        state_dict['peak'] = np.argmax(np.nan_to_num(state_dict['I_tcal']))
        state_dict['buff'] = [b1[first],b2[last]]
        state_dict['intL'] = l1[first]
        state_dict['intR'] = l2[last]
        state_dict['FWHM'] = state_dict['intR'] - state_dict['intL']
        state_dict['intLbuffer'] = 0
        state_dict['intRbuffer'] = 0

        #get spectrum
        (state_dict['I_fcal'],state_dict['Q_fcal'],state_dict['U_fcal'],state_dict['V_fcal']) = dsapol.get_stokes_vs_freq(state_dict['Ical'],state_dict['Qcal'],state_dict['Ucal'],state_dict['Vcal'],state_dict['width_native'],state_dict['tsamp'],
                                                        state_dict['n_f'],state_dict['n_t'],state_dict['freq_test'],n_off=int(NOFFDEF//state_dict['n_t']),plot=False,
                                                        normalize=True,buff=state_dict['buff'],weighted=True,input_weights=state_dict['weights'],
                                                        fobj=FilReader(state_dict['datadir']+state_dict['ids'] + state_dict['suff'] + "_0.fil"))
        state_dict['I_fcalRM'] = copy.deepcopy(state_dict['I_fcal'])
        state_dict['Q_fcalRM'] = copy.deepcopy(state_dict['Q_fcal'])
        state_dict['U_fcalRM'] = copy.deepcopy(state_dict['U_fcal'])
        state_dict['V_fcalRM'] = copy.deepcopy(state_dict['V_fcal'])

        #add to dataframe
        df.loc["All"] = [state_dict['buff'][0],
                                                   state_dict['buff'][1],
                                                   np.nan,
                                                   np.nan,
                                                   np.nan,
                                                   np.nan,
                                                   state_dict['S/N']]

        #done with components, move to RM synth
        #state_dict['current_state'] += 1

        #update scatter dict stuff?

        #wdict['x0_guess'] = state_dict['peak']*32.7e-3
        #wdict['sigma_guess'] = state_dict['FWHM']*32.7e-3
        #wdict['tau_guess'] = (state_dict['intR'] - state_dict['peak'])*32.7e-3
        #wdict['amp_guess'] = state_dict['I_tcal'][int(state_dict['peak'])]

    if fluxestbutton.clicked:
        #get the non-normalized IQUV so we can estimate total flux in Jy
        if 'base_I_unnormalized' not in state_dict.keys():
            (I,Q,U,V,fobj,timeaxis,freq_test,wav_test,badchans) = dsapol.get_stokes_2D(state_dict['datadir'],state_dict['ids'] + "_dev",5120,start=12800,n_t=state_dict['base_n_t'],n_f=state_dict['base_n_f'],n_off=int(NOFFDEF//state_dict['base_n_t']),sub_offpulse_mean=False,fixchans=True,verbose=False)
            state_dict['base_I_unnormalized'] = I
            state_dict['base_Q_unnormalized'] = Q
            state_dict['base_U_unnormalized'] = U
            state_dict['base_V_unnormalized'] = V
      
        #calibrate
        state_dict['base_Ical_unnormalized'],state_dict['base_Qcal_unnormalized'],state_dict['base_Ucal_unnormalized'],state_dict['base_Vcal_unnormalized'] = dsapol.calibrate(state_dict['base_I_unnormalized'],state_dict['base_Q_unnormalized'],state_dict['base_U_unnormalized'],state_dict['base_V_unnormalized'],(state_dict['gxx'],state_dict['gyy']),stokes=True,multithread=True,maxProcesses=128,bad_chans=state_dict['badchans'])

        state_dict['base_Ical_unnormalized_errs'] = np.nanstd(state_dict['base_Ical_unnormalized'],axis=0)/state_dict['base_Ical_unnormalized'].shape[0]
        state_dict['base_Qcal_unnormalized_errs'] = np.nanstd(state_dict['base_Qcal_unnormalized'],axis=0)/state_dict['base_Qcal_unnormalized'].shape[0]
        state_dict['base_Ucal_unnormalized_errs'] = np.nanstd(state_dict['base_Ucal_unnormalized'],axis=0)/state_dict['base_Ucal_unnormalized'].shape[0]
        state_dict['base_Vcal_unnormalized_errs'] = np.nanstd(state_dict['base_Vcal_unnormalized'],axis=0)/state_dict['base_Vcal_unnormalized'].shape[0]

        #mean over frequency, take peak flux (i.e. not fluence)
        maxidx = np.argmax(np.abs(np.nanmean(state_dict['base_Ical_unnormalized'],axis=0)[int(state_dict['intL']):int(state_dict['intR'])]))
        state_dict['Iflux'] = np.abs(np.nanmean(state_dict['base_Ical_unnormalized'],axis=0)[maxidx]) #Jy
        state_dict['Iflux_err'] = np.nanstd(state_dict['base_Ical_unnormalized'],axis=0)[maxidx]/state_dict['base_Ical_unnormalized'].shape[0]
        state_dict['Qflux'] = np.abs(np.nanmean(state_dict['base_Qcal_unnormalized'],axis=0)[maxidx]) #Jy
        state_dict['Qflux_err'] = np.nanstd(state_dict['base_Qcal_unnormalized'],axis=0)[maxidx]/state_dict['base_Ical_unnormalized'].shape[0]
        state_dict['Uflux'] = np.abs(np.nanmean(state_dict['base_Ucal_unnormalized'],axis=0)[maxidx]) #Jy
        state_dict['Uflux_err'] = np.nanstd(state_dict['base_Ucal_unnormalized'],axis=0)[maxidx]/state_dict['base_Ical_unnormalized'].shape[0]
        state_dict['Vflux'] = np.abs(np.nanmean(state_dict['base_Vcal_unnormalized'],axis=0)[maxidx]) #Jy
        state_dict['Vflux_err'] = np.nanstd(state_dict['base_Vcal_unnormalized'],axis=0)[maxidx]/state_dict['base_Ical_unnormalized'].shape[0]
        state_dict['noise_chan'] = np.nanmedian(np.nanstd(state_dict['base_Ical_unnormalized'],axis=0)[int(state_dict['intL']):int(state_dict['intR'])])
        
        state_dict['polint'] = np.abs(np.sqrt(np.nanmean(state_dict['base_Qcal_unnormalized'],axis=0)**2 +
                                                        np.nanmean(state_dict['base_Ucal_unnormalized'],axis=0)**2 + 
                                                        np.nanmean(state_dict['base_Vcal_unnormalized'],axis=0)**2))[maxidx]
        state_dict['polint_err'] = np.nanstd(np.sqrt(state_dict['base_Qcal_unnormalized']**2 + 
                                                    state_dict['base_Ucal_unnormalized']**2 + 
                                                    state_dict['base_Vcal_unnormalized']**2),axis=0)[maxidx]/state_dict['base_Ical_unnormalized'].shape[0]

        Iflux_display.data = np.around(state_dict['Iflux'],2)
        Qflux_display.data = np.around(state_dict['Qflux'],2)
        Uflux_display.data = np.around(state_dict['Uflux'],2)
        Vflux_display.data = np.around(state_dict['Vflux'],2)


        #also get vs freq
        (state_dict['base_Ical_f_unnormalized'],
        state_dict['base_Qcal_f_unnormalized'],
        state_dict['base_Ucal_f_unnormalized'],
        state_dict['base_Vcal_f_unnormalized']) = dsapol.get_stokes_vs_freq(state_dict['base_Ical_unnormalized'],
                                                                                state_dict['base_Q_unnormalized'],
                                                                                state_dict['base_U_unnormalized'],
                                                                                state_dict['base_V_unnormalized'],
                                                                                state_dict['width_native'],state_dict['tsamp'],state_dict['base_n_f'],
                                                                                state_dict['n_t'],state_dict['base_freq_test'],n_off=int(NOFFDEF/state_dict['n_t']),
                                                                                plot=False,show=False,normalize=False,buff=state_dict['buff'],
                                                                                weighted=True,timeaxis=state_dict['time_axis'],fobj=FilReader(state_dict['datadir']+state_dict['ids'] + state_dict['suff'] + "_0.fil"),input_weights=state_dict['weights'])
        state_dict['base_Ical_f_unnormalized_errs'] = np.nanstd(state_dict['base_Ical_unnormalized'],axis=1)/state_dict['base_Ical_unnormalized'].shape[1]
        state_dict['base_Qcal_f_unnormalized_errs'] = np.nanstd(state_dict['base_Qcal_unnormalized'],axis=1)/state_dict['base_Ical_unnormalized'].shape[1]
        state_dict['base_Ucal_f_unnormalized_errs'] = np.nanstd(state_dict['base_Ucal_unnormalized'],axis=1)/state_dict['base_Ical_unnormalized'].shape[1]
        state_dict['base_Vcal_f_unnormalized_errs'] = np.nanstd(state_dict['base_Vcal_unnormalized'],axis=1)/state_dict['base_Ical_unnormalized'].shape[1]


    update_wdict([logwindow_slider,logibox_slider,
                buff_L_slider,buff_R_slider,ncomps_num,comprange_slider,
                avger_w_slider,sf_window_weights_slider,multipeaks,multipeaks_height_slider,filt_showerrs],
                ["logwindow_slider","logibox_slider",
                "buff_L_slider","buff_R_slider","ncomps_num","comprange_slider",
                "avger_w_slider","sf_window_weights_slider","multipeaks","multipeaks_height_slider","filt_showerrs"],
                param='value')

    update_wdict([Iflux_display,Qflux_display,Uflux_display,Vflux_display],
                 ['Iflux_display','Qflux_display','Uflux_display','Vflux_display'],
                 param='data')
    return


"""
Scattering Analysis State
"""
def scatter_reset_initvals(comp_num):#,x0_guess,sigma_guess,tau_guess,amp_guess,x0_range,sigma_range,tau_range,amp_range):
    wdict['x0_guess_' + str(comp_num)]  = np.around(state_dict['comps'][comp_num]['peak']*32.7e-3,2) #ms
    wdict['amp_guess_' + str(comp_num)] = np.around(state_dict['I_tcal'][state_dict['comps'][comp_num]['peak']],2) 
    wdict['sigma_guess_' + str(comp_num)] = np.around(state_dict['comps'][comp_num]['FWHM']*32.7e-3/2,2)  #ms
    wdict['tau_guess_' + str(comp_num)]  = np.around((state_dict['comps'][comp_num]['intR'] - state_dict['comps'][comp_num]['peak'])*32.7e-3,2)  #ms

    wdict['x0_range_' + str(comp_num)] =  [np.around(wdict['x0_guess_' + str(comp_num)]/2,2),np.around((3/2)*wdict['x0_guess_' + str(comp_num)],2)]
    wdict['amp_range_' + str(comp_num)] = [np.around(wdict['amp_guess_' + str(comp_num)]/2,2),np.around((3/2)*wdict['amp_guess_' + str(comp_num)],2)]
    wdict['sigma_range_' + str(comp_num)]  = [np.around(wdict['sigma_guess_' + str(comp_num)]/2,2),np.around((3/2)*wdict['sigma_guess_' + str(comp_num)],2)]
    wdict['tau_range_' + str(comp_num)] = [np.around(wdict['tau_guess_' + str(comp_num)]/2,2),np.around((3/2)*wdict['tau_guess_' + str(comp_num)],2)]
    return

def scatter_screen(scattermenu,scatfitmenu,x0_guess_comps,sigma_guess_comps,tau_guess_comps,amp_guess_comps,x0_range_sliders,sigma_range_sliders,tau_range_sliders,amp_range_sliders,calc_scat_button,save_scat_button,scatterbackground,refresh_button,scatterresume,scatterweights,scatter_nlive,scatter_init,scatter_sliderrange,scattersamps,scatter_nwalkers,scatter_nsteps,scatter_nburn,scatter_nthin):
    state_dict['scatter_init'] = True
    if scatter_init.clicked:
        state_dict['scatter_init'] = False

    state_dict['scatter_fit_comps'] = scatfitmenu.value


    #plot the full component
    if ~np.all(np.isnan(state_dict['I_tcal'])):
    
        #plot the selected components
        I_tcal_p = copy.deepcopy(state_dict['I_tcal'])
        I_tcal_errs_p = copy.deepcopy(state_dict['I_tcal_err'])
        time_axis_p = copy.deepcopy(state_dict['time_axis'])
        mask = np.zeros(state_dict['I_tcal'].shape)
        for k in range(state_dict['n_comps']):
            if 'Component ' + str(k) not in scattermenu.value:
                #mask = np.zeros(state_dict['Ical'].shape)
                mask[state_dict['comps'][k]['left_lim']:state_dict['comps'][k]['right_lim']] = 1
        I_tcal_p = ma(I_tcal_p,mask)
        I_tcal_errs_p = ma(I_tcal_errs_p,mask)
        time_axis_p = ma(time_axis_p,mask)
        state_dict['time_axis_scattering'] = time_axis_p
        state_dict['I_tcal_scattering'] = I_tcal_p
        state_dict['I_tcal_err_scattering'] = I_tcal_errs_p

        #plot initial guess
        p0_full = []
        for comp in scattermenu.value:
            k = int(comp[-1])
            
            p0_full += [x0_guess_comps[k].value, amp_guess_comps[k].value, sigma_guess_comps[k].value, tau_guess_comps[k].value]

    scaler = 1
    if calc_scat_button.clicked:
       
        if scatfitmenu.value == 'Nested Sampling':
            ncomps = len(x0_guess_comps)
            

            timeseries_for_fit = state_dict['I_tcal_scattering'].data[~state_dict['I_tcal_scattering'].mask]/scaler
            timeaxis_for_fit = state_dict['time_axis_scattering'].data[~state_dict['time_axis_scattering'].mask]/1e3/scaler
            if scatterweights.value:
                #weights_for_fit = state_dict['weights'][~state_dict['time_axis_scattering'].mask] 
                #weights_for_fit *= (state_dict['I_tcal_scattering'].data[state_dict['peak']]/state_dict['weights'][state_dict['peak']])
                sigma_for_fit = state_dict['I_tcal_err_scattering']
                #sigma_for_fit = 1/weights_for_fit
                #sigma_for_fit[weights_for_fit==0] = np.nanmax(sigma_for_fit)*100
            else:
                sigma_for_fit = np.nanmean(state_dict['I_tcal_err_scattering'])#np.nanstd(state_dict['I_tcal_scattering'][:int(NOFFDEF/state_dict['n_t'])])*np.ones(len(timeaxis_for_fit))

            #get number of live points
            nlive = scatter_nlive.value#len(timeaxis_for_fit)#int(len(state_dict['I_tcal_scattering']) - np.sum(state_dict['I_tcal_scattering'].mask))
            
            #init bounds
            low_bounds = []
            upp_bounds = []
            p0 = []
            for i in range(ncomps):
                p0 += [x0_guess_comps[i].value/scaler,amp_guess_comps[i].value/scaler,sigma_guess_comps[i].value/scaler,tau_guess_comps[i].value/scaler]
                #low_bounds += [np.nanmin(timeaxis_for_fit),amp_guess_comps[i].value/10/1000,state_dict['tsamp'],state_dict['tsamp']]
                #upp_bounds += [np.nanmax(timeaxis_for_fit),amp_guess_comps[i].value*10/1000,sigma_guess_comps[i].value*10/1000,tau_guess_comps[i].value*10/1000]
                #low_bounds += [x0_guess_comps[i].value/2/1000,amp_guess_comps[i].value/2/1000,sigma_guess_comps[i].value/2/1000,tau_guess_comps[i].value/2/1000]
                #upp_bounds += [3*x0_guess_comps[i].value/2/1000,3*amp_guess_comps[i].value/2/1000,3*sigma_guess_comps[i].value/2/1000,3*tau_guess_comps[i].value/2/1000]

                low_bounds += [x0_range_sliders[i].value[0]/scaler,amp_range_sliders[i].value[0]/scaler,sigma_range_sliders[i].value[0]/scaler,tau_range_sliders[i].value[0]/scaler]
                upp_bounds += [x0_range_sliders[i].value[1]/scaler,amp_range_sliders[i].value[1]/scaler,sigma_range_sliders[i].value[1]/scaler,tau_range_sliders[i].value[1]/scaler]

            #run nested sampling
            if scatterbackground.value:
                state_dict['scatter_bilby_dname_result'],state_dict['scatter_bilby_dname_result_params'],state_dict['scatter_bilby_dname_BIC'], state_dict['scatter_bilby_dname'] = scatscint.run_nested_sampling(timeseries_for_fit,
                                                                outdir=state_dict['datadir'], label=state_dict['ids'] + "_" + state_dict['nickname'],
                                                                p0=p0, comp_num=len(x0_guess_comps), nlive=nlive, time_resolution=state_dict['tsamp']/1e3/scaler,
                                                                timeaxis_for_fit=timeaxis_for_fit,
                                                                low_bounds=low_bounds,upp_bounds=upp_bounds,sigma_for_fit=sigma_for_fit,background=True,
                                                                resume=scatterresume.value)
            else:

                state_dict['scatter_results'],state_dict['scatter_params_best'],state_dict['scatter_params_best_upperr'],state_dict['scatter_params_best_lowerr'],state_dict['scatter_BIC'] = scatscint.run_nested_sampling(timeseries_for_fit,
                                                                outdir=state_dict['datadir'], label=state_dict['ids'] + "_" + state_dict['nickname'],
                                                                p0=p0, comp_num=len(x0_guess_comps), nlive=nlive, time_resolution=state_dict['tsamp']/1e3/scaler,
                                                                timeaxis_for_fit=timeaxis_for_fit,
                                                                low_bounds=low_bounds,upp_bounds=upp_bounds,sigma_for_fit=sigma_for_fit,background=False,
                                                                resume=scatterresume.value)


                df_scat.loc[",".join(scattermenu.value)] = [",".join([str(np.around(state_dict['scatter_params_best'][4*i],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_upperr'][4*i],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_lowerr'][4*i],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best'][4*i + 1],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_upperr'][4*i + 1],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_lowerr'][4*i + 1],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best'][4*i + 2],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_upperr'][4*i + 2],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_lowerr'][4*i + 2],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best'][4*i + 3],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_upperr'][4*i + 3],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_lowerr'][4*i + 3],2)) for i in range(len(scattermenu.value))]),
                                                    "","","",
                                                    str(np.around(state_dict['scatter_BIC'],2))]

                """
                
                                                    [np.around(state_dict['scatter_params_best_lowerr'][4*i]) for i in range(len(scattermenu.value))],
                                                    [np.around(state_dict['scatter_params_best'][4*i + 1]) for i in range(len(scattermenu.value))],
                                                    [np.around(state_dict['scatter_params_best_upperr'][4*i + 1]) for i in range(len(scattermenu.value))],
                                                    [np.around(state_dict['scatter_params_best_lowerr'][4*i + 1]) for i in range(len(scattermenu.value))],
                                                    [np.around(state_dict['scatter_params_best'][4*i + 2]) for i in range(len(scattermenu.value))],
                                                    [np.around(state_dict['scatter_params_best'][4*i + 3]) for i in range(len(scattermenu.value))],
                                                    [np.around(state_dict['scatter_params_best_upperr'][4*i + 3]) for i in range(len(scattermenu.value))],
                                                    [np.around(state_dict['scatter_params_best_lowerr'][4*i + 3]) for i in range(len(scattermenu.value))]])
                """
        elif scatfitmenu.value == 'EMCEE Markov-Chain Monte Carlo':
            ncomps = len(x0_guess_comps)


            timeseries_for_fit = state_dict['I_tcal_scattering'].data[~state_dict['I_tcal_scattering'].mask]/scaler
            timeaxis_for_fit = state_dict['time_axis_scattering'].data[~state_dict['time_axis_scattering'].mask]/1e3/scaler
            if scatterweights.value:
                #weights_for_fit = state_dict['weights'][~state_dict['time_axis_scattering'].mask]
                #weights_for_fit *= (state_dict['I_tcal_scattering'].data[state_dict['peak']]/state_dict['weights'][state_dict['peak']])
                sigma_for_fit = state_dict['I_tcal_err_scattering']
                #sigma_for_fit = 1/weights_for_fit
                #sigma_for_fit[weights_for_fit==0] = np.nanmax(sigma_for_fit)*100
            else:
                sigma_for_fit = np.nanmean(state_dict['I_tcal_err_scattering'])#np.nanstd(state_dict['I_tcal_scattering'][:int(NOFFDEF/state_dict['n_t'])])*np.ones(len(timeaxis_for_fit))

            #init bounds
            low_bounds = []
            upp_bounds = []
            p0 = []
            for i in range(ncomps):
                p0 += [x0_guess_comps[i].value/scaler,amp_guess_comps[i].value/scaler,sigma_guess_comps[i].value/scaler,tau_guess_comps[i].value/scaler]
                #low_bounds += [np.nanmin(timeaxis_for_fit),amp_guess_comps[i].value/10/1000,state_dict['tsamp'],state_dict['tsamp']]
                #upp_bounds += [np.nanmax(timeaxis_for_fit),amp_guess_comps[i].value*10/1000,sigma_guess_comps[i].value*10/1000,tau_guess_comps[i].value*10/1000]
                #low_bounds += [x0_guess_comps[i].value/2/1000,amp_guess_comps[i].value/2/1000,sigma_guess_comps[i].value/2/1000,tau_guess_comps[i].value/2/1000]
                #upp_bounds += [3*x0_guess_comps[i].value/2/1000,3*amp_guess_comps[i].value/2/1000,3*sigma_guess_comps[i].value/2/1000,3*tau_guess_comps[i].value/2/1000]

                low_bounds += [x0_range_sliders[i].value[0]/scaler,amp_range_sliders[i].value[0]/scaler,sigma_range_sliders[i].value[0]/scaler,tau_range_sliders[i].value[0]/scaler]
                upp_bounds += [x0_range_sliders[i].value[1]/scaler,amp_range_sliders[i].value[1]/scaler,sigma_range_sliders[i].value[1]/scaler,tau_range_sliders[i].value[1]/scaler]

            if scatterbackground.value:
                state_dict['scatter_MCMC_dname_samples'],state_dict['scatter_MCMC_dname_result'],state_dict['scatter_MCMC_dname_BIC'],state_dict['scatter_MCMC_dname'] = scatscint.run_MCMC_sampling(timeseries_for_fit,
                                                                outdir=state_dict['datadir'], label=state_dict['ids'] + "_" + state_dict['nickname'],
                                                                p0=p0, timeaxis_for_fit=timeaxis_for_fit,
                                                                low_bounds=low_bounds,upp_bounds=upp_bounds,sigma_for_fit=sigma_for_fit,background=True,
                                                                nwalkers=int(scatter_nwalkers.value),nsteps=int(scatter_nsteps.value),discard=int(scatter_nburn.value),thin=int(scatter_nthin.value))
            else:

                state_dict['scatter_MCMC_samples'],state_dict['scatter_params_best'],state_dict['scatter_params_best_upperr'],state_dict['scatter_params_best_lowerr'],state_dict['scatter_BIC'] = scatscint.run_MCMC_sampling(timeseries_for_fit,
                                                                outdir=state_dict['datadir'], label=state_dict['ids'] + "_" + state_dict['nickname'],
                                                                p0=p0, timeaxis_for_fit=timeaxis_for_fit,
                                                                low_bounds=low_bounds,upp_bounds=upp_bounds,sigma_for_fit=sigma_for_fit,background=False,
                                                                nwalkers=int(scatter_nwalkers.value),nsteps=int(scatter_nsteps.value),discard=int(scatter_nburn.value),thin=int(scatter_nthin.value))


                state_dict['scatter_MCMC_samples_f'] = state_dict['scatter_MCMC_samples'][:,-1]*scaler
                state_dict['scatter_params_best_f'] = state_dict['scatter_params_best'][-1]*scaler
                state_dict['scatter_params_best_upperr_f'] = state_dict['scatter_params_best_upperr'][-1]*scaler
                state_dict['scatter_params_best_lowerr_f'] = state_dict['scatter_params_best_lowerr'][-1]*scaler


                state_dict['scatter_MCMC_samples'] = state_dict['scatter_MCMC_samples'][:,:-1]*scaler
                state_dict['scatter_params_best'] = state_dict['scatter_params_best'][:-1]*scaler
                state_dict['scatter_params_best_upperr'] = state_dict['scatter_params_best_upperr'][:-1]*scaler
                state_dict['scatter_params_best_lowerr'] = state_dict['scatter_params_best_lowerr'][:-1]*scaler

                df_scat.loc[",".join(scattermenu.value)] = [",".join([str(np.around(state_dict['scatter_params_best'][4*i],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_upperr'][4*i],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_lowerr'][4*i],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best'][4*i + 1],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_upperr'][4*i + 1],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_lowerr'][4*i + 1],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best'][4*i + 2],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_upperr'][4*i + 2],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_lowerr'][4*i + 2],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best'][4*i + 3],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_upperr'][4*i + 3],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_lowerr'][4*i + 3],2)) for i in range(len(scattermenu.value))]),
                                                    str(np.around(state_dict['scatter_params_best_f'],2)),
                                                    str(np.around(state_dict['scatter_params_best_upperr_f'],2)),
                                                    str(np.around(state_dict['scatter_params_best_lowerr_f'],2)),
                                                    str(np.around(state_dict['scatter_BIC'],2))]




        elif scatfitmenu.value == 'LMFIT Non-Linear Least Squares':
            ncomps = len(x0_guess_comps)

            if ncomps == 1:
                fit_func = lambda x, x00, amp0, sigma0, tau0: scat.exp_gauss_n(x,x00,amp0,sigma0,tau0)
            elif ncomps == 2:
                fit_func = lambda x, x00, amp0, sigma0, tau0, x01, amp1, sigma1, tau1: scat.exp_gauss_n(x,x00,amp0,sigma0,tau0,x01, amp1, sigma1, tau1)
            elif ncomps == 3:
                fit_func = lambda x, x00, amp0, sigma0, tau0, x01, amp1, sigma1, tau1, x02, amp2, sigma2, tau2: scat.exp_gauss_n(x,x00,amp0,sigma0,tau0,x01, amp1, sigma1, tau1,x02, amp2, sigma2, tau2)
            elif ncomps == 4:
                fit_func = lambda x, x00, amp0, sigma0, tau0, x01, amp1, sigma1, tau1, x02, amp2, sigma2, tau2,x03, amp3, sigma3, tau3: scat.exp_gauss_n(x,x00,amp0,sigma0,tau0,x01, amp1, sigma1, tau1,x02, amp2, sigma2, tau2, x03, amp3, sigma3, tau3)
            elif ncomps == 5:
                fit_func = lambda x, x00, amp0, sigma0, tau0, x01, amp1, sigma1, tau1, x02, amp2, sigma2, tau2,x03, amp3, sigma3, tau3,x04, amp4, sigma4, tau4: scat.exp_gauss_n(x,x00,amp0,sigma0,tau0,x01, amp1, sigma1, tau1,x02, amp2, sigma2, tau2, x03, amp3, sigma3, tau3,x04, amp4, sigma4, tau4)
            
            gmodel = Model(fit_func,independent_vars=['x'])

            timeseries_for_fit = state_dict['I_tcal_scattering'].data[~state_dict['I_tcal_scattering'].mask]/scaler
            timeaxis_for_fit = state_dict['time_axis_scattering'].data[~state_dict['time_axis_scattering'].mask]/1e3/scaler
            if scatterweights.value:
                weights_for_fit = state_dict['weights'][~state_dict['time_axis_scattering'].mask]/np.sum(state_dict['weights'][~state_dict['time_axis_scattering'].mask])
            else:
                weights_for_fit = None


            params = Parameters()
            for i in range(ncomps):
                
                params.add('x0' + str(i),value=x0_guess_comps[i].value/scaler,min=x0_range_sliders[i].value[0]/scaler,max=x0_range_sliders[i].value[1]/scaler,vary=True)#x0_guess_comps[i].value/10/1000,max=x0_guess_comps[i].value*10/1000,vary=True)
                params.add('amp' + str(i),value=amp_guess_comps[i].value/scaler,min=amp_range_sliders[i].value[0]/scaler,max=amp_range_sliders[i].value[1]/scaler,vary=True)
                params.add('sigma' + str(i),value=sigma_guess_comps[i].value/scaler,min=sigma_range_sliders[i].value[0]/scaler,max=sigma_range_sliders[i].value[1]/scaler,vary=True)
                params.add('tau' + str(i),value=tau_guess_comps[i].value/scaler,min=tau_range_sliders[i].value[0]/scaler,max=tau_range_sliders[i].value[1]/scaler,vary=True)


            result = gmodel.fit(timeseries_for_fit,params,x=timeaxis_for_fit,weights=weights_for_fit)
            state_dict['scatter_params_best'] = np.array([])
            state_dict['scatter_params_best_upperr'] = np.array([])
            state_dict['scatter_params_best_lowerr'] = np.array([])

            for i in range(ncomps):
                state_dict['scatter_params_best'] = np.concatenate([state_dict['scatter_params_best'],[result.params['x0' + str(i)].value*scaler,result.params['amp' + str(i)].value*scaler,result.params['sigma' + str(i)].value*scaler,result.params['tau' + str(i)].value*scaler]])
                state_dict['scatter_params_best_upperr'] = np.concatenate([state_dict['scatter_params_best_upperr'],[result.params['x0' + str(i)].stderr*scaler,result.params['amp' + str(i)].stderr*scaler,result.params['sigma' + str(i)].stderr*scaler,result.params['tau' + str(i)].stderr*scaler]])
                state_dict['scatter_params_best_lowerr'] = np.concatenate([state_dict['scatter_params_best_lowerr'],[result.params['x0' + str(i)].stderr*scaler,result.params['amp' + str(i)].stderr*scaler,result.params['sigma' + str(i)].stderr*scaler,result.params['tau' + str(i)].stderr*scaler]])


            # Print or output the fit report for display
            df_scat.loc[",".join(scattermenu.value)] = [",".join([str(np.around(state_dict['scatter_params_best'][4*i],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_upperr'][4*i],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_lowerr'][4*i],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best'][4*i + 1],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_upperr'][4*i + 1],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_lowerr'][4*i + 1],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best'][4*i + 2],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_upperr'][4*i + 2],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_lowerr'][4*i + 2],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best'][4*i + 3],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_upperr'][4*i + 3],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_lowerr'][4*i + 3],2)) for i in range(len(scattermenu.value))]),
                                                    "","","",""]

            """
    
            df_scat.loc[",".join(scattermenu.value)] = np.concatenate([[np.around(state_dict['scatter_params_best'][4*i]) for i in range(ncomps)],
                                                    [np.around(state_dict['scatter_params_best_upperr'][4*i]) for i in range(ncomps)],
                                                    [np.around(state_dict['scatter_params_best_lowerr'][4*i]) for i in range(ncomps)],
                                                    [np.around(state_dict['scatter_params_best'][4*i + 1]) for i in range(ncomps)],
                                                    [np.around(state_dict['scatter_params_best_upperr'][4*i + 1]) for i in range(ncomps)],
                                                    [np.around(state_dict['scatter_params_best_lowerr'][4*i + 1]) for i in range(ncomps)],
                                                    [np.around(state_dict['scatter_params_best'][4*i + 2]) for i in range(ncomps)],
                                                    [np.around(state_dict['scatter_params_best_upperr'][4*i + 2]) for i in range(ncomps)],
                                                    [np.around(state_dict['scatter_params_best_lowerr'][4*i + 2]) for i in range(ncomps)],
                                                    [np.around(state_dict['scatter_params_best'][4*i + 3]) for i in range(ncomps)],
                                                    [np.around(state_dict['scatter_params_best_upperr'][4*i + 3]) for i in range(ncomps)],
                                                    [np.around(state_dict['scatter_params_best_lowerr'][4*i + 3]) for i in range(ncomps)]])
            """

        else:
            ncomps = len(x0_guess_comps)

            if ncomps == 1:
                fit_func = lambda x, x00, amp0, sigma0, tau0: scat.exp_gauss_n(x,x00,amp0,sigma0,tau0)
            elif ncomps == 2:
                fit_func = lambda x, x00, amp0, sigma0, tau0, x01, amp1, sigma1, tau1: scat.exp_gauss_n(x,x00,amp0,sigma0,tau0,x01, amp1, sigma1, tau1)
            elif ncomps == 3:
                fit_func = lambda x, x00, amp0, sigma0, tau0, x01, amp1, sigma1, tau1, x02, amp2, sigma2, tau2: scat.exp_gauss_n(x,x00,amp0,sigma0,tau0,x01, amp1, sigma1, tau1,x02, amp2, sigma2, tau2)
            elif ncomps == 4:
                fit_func = lambda x, x00, amp0, sigma0, tau0, x01, amp1, sigma1, tau1, x02, amp2, sigma2, tau2,x03, amp3, sigma3, tau3: scat.exp_gauss_n(x,x00,amp0,sigma0,tau0,x01, amp1, sigma1, tau1,x02, amp2, sigma2, tau2, x03, amp3, sigma3, tau3)
            elif ncomps == 5:
                fit_func = lambda x, x00, amp0, sigma0, tau0, x01, amp1, sigma1, tau1, x02, amp2, sigma2, tau2,x03, amp3, sigma3, tau3,x04, amp4, sigma4, tau4: scat.exp_gauss_n(x,x00,amp0,sigma0,tau0,x01, amp1, sigma1, tau1,x02, amp2, sigma2, tau2, x03, amp3, sigma3, tau3,x04, amp4, sigma4, tau4)


            timeseries_for_fit = state_dict['I_tcal_scattering'].data[~state_dict['I_tcal_scattering'].mask]/scaler
            timeaxis_for_fit = state_dict['time_axis_scattering'].data[~state_dict['time_axis_scattering'].mask]/1e3/scaler
            if scatterweights.value:
                #weights_for_fit = state_dict['weights'][~state_dict['time_axis_scattering'].mask]/np.sum(state_dict['weights'][~state_dict['time_axis_scattering'].mask])
                #sigma_for_fit = 1/weights_for_fit
                #sigma_for_fit[weights_for_fit==0] = np.nanmax(sigma_for_fit)*100
                sigma_for_fit = state_dict['I_tcal_err_scattering']
            else:
                sigma_for_fit = None

            p0 = []
            low_bounds = []
            upp_bounds = []
            for i in range(ncomps):
                p0 += [x0_guess_comps[i].value/scaler,amp_guess_comps[i].value/scaler,sigma_guess_comps[i].value/scaler,tau_guess_comps[i].value/scaler]
                low_bounds += [x0_range_sliders[i].value[0]/scaler,amp_range_sliders[i].value[0]/scaler,sigma_range_sliders[i].value[0]/scaler,tau_range_sliders[i].value[0]/scaler]
                upp_bounds += [x0_range_sliders[i].value[1]/scaler,amp_range_sliders[i].value[1]/scaler,sigma_range_sliders[i].value[1]/scaler,tau_range_sliders[i].value[1]/scaler]
            
            state_dict['scatter_params_best'],pcov = curve_fit(fit_func,timeseries_for_fit,timeaxis_for_fit,p0=p0,bounds=(low_bounds,upp_bounds),sigma=sigma_for_fit)
            state_dict['scatter_params_best_upperr'] = []
            state_dict['scatter_params_best_lowerr'] = []
            for i in range(ncomps):
                state_dict['scatter_params_best_upperr'] = np.concatenate([state_dict['scatter_params_best_upperr'],[pcov[4*i,4*i],pcov[4*i + 1,4*i + 1], pcov[4*i + 2,4*i + 2],pcov[4*i + 3,4*i + 3]]])
                state_dict['scatter_params_best_lowerr'] = np.concatenate([state_dict['scatter_params_best_lowerr'],[pcov[4*i,4*i],pcov[4*i + 1,4*i + 1], pcov[4*i + 2,4*i + 2],pcov[4*i + 3,4*i + 3]]])
            state_dict['scatter_params_best'] *= scaler
            state_dict['scatter_params_best_upperr'] = np.sqrt(state_dict['scatter_params_best_upperr'])*scaler
            state_dict['scatter_params_best_lowerr'] = np.sqrt(state_dict['scatter_params_best_upperr'])*scaler

            df_scat.loc[",".join(scattermenu.value)] = [",".join([str(np.around(state_dict['scatter_params_best'][4*i],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_upperr'][4*i],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_lowerr'][4*i],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best'][4*i + 1],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_upperr'][4*i + 1],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_lowerr'][4*i + 1],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best'][4*i + 2],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_upperr'][4*i + 2],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_lowerr'][4*i + 2],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best'][4*i + 3],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_upperr'][4*i + 3],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_lowerr'][4*i + 3],2)) for i in range(len(scattermenu.value))]),
                                                    "","","",""]



            """
            df_scat.loc[",".join(scattermenu.value)] = np.concatenate([[np.around(state_dict['scatter_params_best'][4*i]) for i in range(ncomps)],
                                                    [np.around(state_dict['scatter_params_best_upperr'][4*i]) for i in range(ncomps)],
                                                    [np.around(state_dict['scatter_params_best_lowerr'][4*i]) for i in range(ncomps)],
                                                    [np.around(state_dict['scatter_params_best'][4*i + 1]) for i in range(ncomps)],
                                                    [np.around(state_dict['scatter_params_best_upperr'][4*i + 1]) for i in range(ncomps)],
                                                    [np.around(state_dict['scatter_params_best_lowerr'][4*i + 1]) for i in range(ncomps)],
                                                    [np.around(state_dict['scatter_params_best'][4*i + 2]) for i in range(ncomps)],
                                                    [np.around(state_dict['scatter_params_best_upperr'][4*i + 2]) for i in range(ncomps)],
                                                    [np.around(state_dict['scatter_params_best_lowerr'][4*i + 2]) for i in range(ncomps)],
                                                    [np.around(state_dict['scatter_params_best'][4*i + 3]) for i in range(ncomps)],
                                                    [np.around(state_dict['scatter_params_best_upperr'][4*i + 3]) for i in range(ncomps)],
                                                    [np.around(state_dict['scatter_params_best_lowerr'][4*i + 3]) for i in range(ncomps)]])

            """

    #check if nested sampling/MCMC is done
    if refresh_button.clicked:
        print("Refreshing...")
        if 'scatter_bilby_dname_result' in state_dict.keys():
            l = glob.glob(dirs['logs'] + "scat_files/" + state_dict['scatter_bilby_dname_result'])
            if len(l) > 0:
                state_dict['scatter_results'] = bcresult.read_in_result(filename=dirs['logs'] + "scat_files/" + state_dict['scatter_bilby_dname_result'])
                tmp = np.load(dirs['logs'] + "scat_files/" + state_dict['scatter_bilby_dname_result_params'])*scaler
                state_dict['scatter_params_best'] = tmp[0,:]
                state_dict['scatter_params_best_upperr'] = tmp[1,:]
                state_dict['scatter_params_best_lowerr'] = tmp[2,:]
            
                state_dict['scatter_BIC'] = np.load(dirs['logs'] + "scat_files/" + state_dict['scatter_bilby_dname_BIC'])

                df_scat.loc[",".join(scattermenu.value)] = [",".join([str(np.around(state_dict['scatter_params_best'][4*i],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_upperr'][4*i],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_lowerr'][4*i],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best'][4*i + 1],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_upperr'][4*i + 1],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_lowerr'][4*i + 1],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best'][4*i + 2],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_upperr'][4*i + 2],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_lowerr'][4*i + 2],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best'][4*i + 3],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_upperr'][4*i + 3],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_lowerr'][4*i + 3],2)) for i in range(len(scattermenu.value))]),
                                                    "","","",
                                                    str(np.around(state_dict['scatter_BIC'],2))]
                
                """
                df_scat.loc[",".join(scattermenu.value)] = np.concatenate([[np.around(state_dict['scatter_params_best'][4*i]) for i in range(len(scattermenu.value))],
                                                    [np.around(state_dict['scatter_params_best_upperr'][4*i]) for i in range(len(scattermenu.value))],
                                                    [np.around(state_dict['scatter_params_best_lowerr'][4*i]) for i in range(len(scattermenu.value))],
                                                    [np.around(state_dict['scatter_params_best'][4*i + 1]) for i in range(len(scattermenu.value))],
                                                    [np.around(state_dict['scatter_params_best_upperr'][4*i + 1]) for i in range(len(scattermenu.value))],
                                                    [np.around(state_dict['scatter_params_best_lowerr'][4*i + 1]) for i in range(len(scattermenu.value))],
                                                    [np.around(state_dict['scatter_params_best'][4*i + 2]) for i in range(len(scattermenu.value))],
                                                    [np.around(state_dict['scatter_params_best_upperr'][4*i + 2]) for i in range(len(scattermenu.value))],
                                                    [np.around(state_dict['scatter_params_best_lowerr'][4*i + 2]) for i in range(len(scattermenu.value))],
                                                    [np.around(state_dict['scatter_params_best'][4*i + 3]) for i in range(len(scattermenu.value))],
                                                    [np.around(state_dict['scatter_params_best_upperr'][4*i + 3]) for i in range(len(scattermenu.value))],
                                                    [np.around(state_dict['scatter_params_best_lowerr'][4*i + 3]) for i in range(len(scattermenu.value))]])

                """
        elif 'scatter_MCMC_dname' in state_dict.keys():
            l = glob.glob(dirs['logs'] + "scat_files/" + state_dict['scatter_MCMC_dname_result'])
            if len(l) > 0:
                state_dict['scatter_MCMC_samples'] = np.load(dirs['logs'] + "scat_files/" + state_dict['scatter_MCMC_dname_samples'])[:,:-1]*scaler
                tmp = np.load(dirs['logs'] + "scat_files/" + state_dict['scatter_MCMC_dname_result'])*scaler
                state_dict['scatter_params_best'] = tmp[0,:-1]
                state_dict['scatter_params_best_upperr'] = tmp[1,:-1]
                state_dict['scatter_params_best_lowerr'] = tmp[2,:-1]

                state_dict['scatter_MCMC_samples_f'] = np.load(dirs['logs'] + "scat_files/" + state_dict['scatter_MCMC_dname_samples'])[:,-1]*scaler
                state_dict['scatter_params_best_f'] = tmp[0,-1]
                state_dict['scatter_params_best_upperr_f'] = tmp[1,-1]
                state_dict['scatter_params_best_lowerr_f'] = tmp[2,-1]

                state_dict['scatter_BIC'] = np.load(dirs['logs'] + "scat_files/" + state_dict['scatter_MCMC_dname_BIC'])

                df_scat.loc[",".join(scattermenu.value)] = [",".join([str(np.around(state_dict['scatter_params_best'][4*i],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_upperr'][4*i],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_lowerr'][4*i],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best'][4*i + 1],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_upperr'][4*i + 1],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_lowerr'][4*i + 1],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best'][4*i + 2],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_upperr'][4*i + 2],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_lowerr'][4*i + 2],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best'][4*i + 3],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_upperr'][4*i + 3],2)) for i in range(len(scattermenu.value))]),
                                                    ",".join([str(np.around(state_dict['scatter_params_best_lowerr'][4*i + 3],2)) for i in range(len(scattermenu.value))]),
                                                    str(np.around(state_dict['scatter_params_best_f'],2)),
                                                    str(np.around(state_dict['scatter_params_best_upperr_f'],2)),
                                                    str(np.around(state_dict['scatter_params_best_lowerr_f'],2)),
                                                    str(np.around(state_dict['scatter_BIC'],2))]


                
    #begin plotting

    #plot components and initial fits
    g2 = plt.GridSpec(2,1,hspace=0,height_ratios=[2,1],top=0.7)
    fig = plt.figure(figsize=(18,12))
    ax1 = fig.add_subplot(g2[0,:])

    #plot the full component
    if ~np.all(np.isnan(state_dict['I_tcal'])):
        c = ax1.step(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3,
            state_dict['I_tcal'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],label='Timeseries',where='post',alpha=0.25)

        #plot the selected components
        ax1.step(time_axis_p[state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3,
            I_tcal_p[state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],color=c[0].get_color(),where='post',alpha=1)

        #plot initial guess
        #initial_fit_full = np.zeros(state_dict['I_tcal'].shape)
        for comp in scattermenu.value:
            k = int(comp[-1])

            ax1.plot(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3,
                scat.exp_gauss(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3, x0_guess_comps[k].value, amp_guess_comps[k].value, sigma_guess_comps[k].value, tau_guess_comps[k].value),color='purple',alpha=0.25)

            #initial_fit_full += scat.exp_gauss_1(state_dict['time_axis']*1e-3,x0_guess_comps[k].value, amp_guess_comps[k].value, sigma_guess_comps[k].value, tau_guess_comps[k].value)
            #indicate proposed values
            ax1.axvline(state_dict['comps'][k]['peak']*32.7e-3,color='purple',linestyle='--')
            ax1.axvspan(state_dict['comps'][k]['intL']*32.7e-3,state_dict['comps'][k]['intR']*32.7e-3,color='purple',alpha=0.1)
            ax1.text((state_dict['comps'][k]['peak'] + 5)*32.7e-3,I_tcal_p[state_dict['comps'][k]['peak']]+5,
                'Proposed Guess:\n$x_0={a:.2f}$ ms\n$\\sigma = {b:.2f}$ ms\n$\\tau = {c:.2f}$ ms\n amp = {d:.2f}'.format(a=state_dict['comps'][k]['peak']*32.7e-3,
                                                                                                        b=state_dict['comps'][k]['FWHM']*32.7e-3,
                                                                                                        c=(state_dict['comps'][k]['intR'] - state_dict['comps'][k]['peak'])*32.7e-3,
                                                                                                        d=I_tcal_p[state_dict['comps'][k]['peak']]),
                backgroundcolor='thistle',fontsize=18)
        ax1.plot(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3,
                scat.exp_gauss_n(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3, *p0_full),
                color='purple',label='Initial Guess')



        ax1.set_ylim(-1,np.nanmax(state_dict['I_tcal']) + 10)
        ax1.set_xlim(32.7*state_dict['n_t']*state_dict['timestart']*1e-3 - state_dict['window']*32.7*state_dict['n_t']*1e-3,
            32.7*state_dict['n_t']*state_dict['timestop']*1e-3 + state_dict['window']*32.7*state_dict['n_t']*1e-3)
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("S/N")
    ax1.set_xticks([])
    ax1.legend(loc='upper right')

    #plot residuals

    ax3 = fig.add_subplot(g2[1,:])#,sharex=ax1)

    if ~np.all(np.isnan(state_dict['I_tcal'])):

        for comp in scattermenu.value:
            k = int(comp[-1])
            #k in range(state_dict['n_comps']):
            ax3.plot(time_axis_p[state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3,
                    scat.exp_gauss(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3, x0_guess_comps[k].value, amp_guess_comps[k].value, sigma_guess_comps[k].value, tau_guess_comps[k].value)-I_tcal_p[state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],color='purple',alpha=0.25)
        ax3.plot(time_axis_p[state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3,
                (scat.exp_gauss_n(state_dict['time_axis']*1e-3, *p0_full)-I_tcal_p)[state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],color='purple',label='Initial Residuals',alpha=1)
        ax3.set_xlim(32.7*state_dict['n_t']*state_dict['timestart']*1e-3 - state_dict['window']*32.7*state_dict['n_t']*1e-3,
            32.7*state_dict['n_t']*state_dict['timestop']*1e-3 + state_dict['window']*32.7*state_dict['n_t']*1e-3)



        ##for Nested sampling/MCMC, plot subset of samples
        if (scatfitmenu.value == 'Nested Sampling') and ('scatter_results' in state_dict.keys()) and scattersamps.value:
            nsamps,npars = state_dict['scatter_results'].samples.shape
            samps = np.random.choice(np.arange(nsamps),size=nsamps//10,replace=False)
            for i in samps:#range(nsamps):
                ax1.plot(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3,
                        scat.exp_gauss_n(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3, *list(scaler*state_dict['scatter_results'].samples[i,:])),color='red',alpha=0.05)
                ax3.plot(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3,
                        scat.exp_gauss_n(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3, *list(scaler*state_dict['scatter_results'].samples[i,:]))-I_tcal_p[state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],color='red',alpha=0.05)
        
        if (scatfitmenu.value == 'EMCEE Markov-Chain Monte Carlo') and ('scatter_MCMC_samples' in state_dict.keys()) and scattersamps.value:
            nsamps,npars = state_dict['scatter_MCMC_samples'].shape
            samps = np.random.choice(np.arange(nsamps),size=nsamps//10,replace=False)
            for i in samps:
                ax1.plot(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3,
                        scat.exp_gauss_n(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3, *list(scaler*state_dict['scatter_MCMC_samples'][i,:])),color='red',alpha=0.05)
                ax3.plot(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3,
                        scat.exp_gauss_n(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3, *list(scaler*state_dict['scatter_MCMC_samples'][i,:]))-I_tcal_p[state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],color='red',alpha=0.05)

        #plot best fit
        if 'scatter_params_best' in state_dict.keys():
            ax1.plot(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3,
                        scat.exp_gauss_n(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3, *list(state_dict['scatter_params_best'])),color='red',alpha=1,linewidth=4,label='Best Fit')        
            ax3.plot(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3,
                        scat.exp_gauss_n(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3, *list(state_dict['scatter_params_best']))-I_tcal_p[state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],color='red',alpha=1,linewidth=4,label='Best Fit')

        




    ax3.set_xlabel("Time (ms)")
    ax3.set_ylabel(r"$\Delta$")
    ax3.legend(loc='upper right')


    if save_scat_button.clicked:
        plt.savefig(state_dict['datadir'] + "/" + state_dict['ids'] + "_" + state_dict['nickname'] + "_scatter.pdf")
        df_scat.to_csv(state_dict['datadir'] + "/" + state_dict['ids'] + "_" + state_dict['nickname'] + "_scatter_params.csv")



    plt.show()


    #update wdict
    update_wdict([scattermenu,scatfitmenu,scatterbackground,scatterweights,scatterresume,scatter_nlive,scatter_sliderrange,scattersamps,scatter_nwalkers,scatter_nsteps,scatter_nburn,scatter_nthin],
            ['scattermenu','scatfitmenu','scatterbackground','scatterweights','scatterresume','scatter_nlive','scatter_sliderrange','scattersamps','scatter_nwalkers','scatter_nsteps','scatter_nburn','scatter_nthin'])
    for i in range(len(x0_guess_comps)):
        update_wdict([x0_guess_comps[i],amp_guess_comps[i],sigma_guess_comps[i],tau_guess_comps[i],
                      x0_range_sliders[i],amp_range_sliders[i],sigma_range_sliders[i],tau_range_sliders[i]],
                ['x0_guess_' + str(i), 'amp_guess_' + str(i), 'sigma_guess_' + str(i), 'tau_guess_' + str(i),
                 'x0_range_' + str(i), 'amp_range_' + str(i), 'sigma_range_' + str(i), 'tau_range_' + str(i)])
    if 'scatter_results' in state_dict.keys() and (scatfitmenu.value == 'Nested Sampling') :
        return state_dict['scatter_results']
    elif (scatfitmenu.value == 'EMCEE Markov-Chain Monte Carlo') and ('scatter_MCMC_samples' in state_dict.keys()):
        return state_dict['scatter_MCMC_samples']
    else: 
        return None


def scint_screen(scintfitmenu,calc_bw_button,gamma_guess,m_guess,c_guess,scintmenu,scint_fit_range,save_scint_button):
    # Define frequency resolution (Hz)
    f_res = state_dict['n_f']*state_dict['base_df']/1E6 #MHz


    if ~np.all(np.isnan(state_dict['I_fcal'])):
        #compute autocorrelation function
        state_dict['autocorr_I'] = scint.autocorr(state_dict['I_fcal'])
        state_dict['scint_lags'] = np.arange(len(state_dict['autocorr_I'])) + 1
    
        # Create array of lag values
        state_dict['autocorr_I'] = state_dict['autocorr_I'][1:]
        state_dict['scint_lags'] = state_dict['scint_lags'][1:]

        # Create symmetric ACF and lags for fitting
        state_dict['autocorr_I'] = np.concatenate((state_dict['autocorr_I'][::-1],state_dict['autocorr_I']))
        state_dict['scint_lags'] = np.concatenate((-1 * state_dict['scint_lags'][::-1], state_dict['scint_lags'])) * f_res

        state_dict['autocorr_I'] = (state_dict['autocorr_I'] - np.nanmean(state_dict['autocorr_I']))/np.nanmax(state_dict['autocorr_I'])
    
    else:
        state_dict['autocorr_I'] = np.nan*np.ones(len(state_dict['I_fcal']))
        state_dict['scint_lags'] = np.nan*np.ones(len(state_dict['I_fcal']))

    #update autocorr spectra for each component
    if state_dict['n_comps'] > 1:
        for i in range(state_dict['n_comps']):
            if ~np.all(np.isnan(state_dict['comps'][i]['I_fcal'])):
                #compute autocorrelation function
                state_dict['comps'][i]['autocorr_I'] = scint.autocorr(state_dict['comps'][i]['I_fcal'])
                state_dict['comps'][i]['scint_lags'] = np.arange(len(state_dict['comps'][i]['I_fcal'])) + 1

                # Create array of lag values
                state_dict['comps'][i]['autocorr_I'] = state_dict['comps'][i]['autocorr_I'][1:]
                state_dict['comps'][i]['scint_lags'] = state_dict['comps'][i]['scint_lags'][1:]

                # Create symmetric ACF and lags for fitting
                state_dict['comps'][i]['autocorr_I'] = np.concatenate((state_dict['comps'][i]['autocorr_I'][::-1],state_dict['comps'][i]['autocorr_I']))
                state_dict['comps'][i]['scint_lags'] = np.concatenate((-1 * state_dict['comps'][i]['scint_lags'][::-1], state_dict['comps'][i]['scint_lags'])) * f_res

                state_dict['comps'][i]['autocorr_I'] = (state_dict['comps'][i]['autocorr_I'] - np.nanmean(state_dict['comps'][i]['autocorr_I']))/np.nanmax(state_dict['comps'][i]['autocorr_I'])
            else:
                state_dict['comps'][i]['autocorr_I'] = np.nan*np.ones(len(state_dict['comps'][i]['I_fcal']))
                state_dict['comps'][i]['scint_lags'] = np.nan*np.ones(len(state_dict['comps'][i]['I_fcal']))


    
    #plot frequency spectrum and initial guess fit autocorrelation spectrum
    #plt.figure(figsize=(18,12))
    #plt.subplot(311)

    g1 = plt.GridSpec(3,1,height_ratios=[2,2,1])
    g2 = plt.GridSpec(3,1,hspace=0,height_ratios=[2,2,1],top=0.7)
    fig = plt.figure(figsize=(18,12))
    ax1 = fig.add_subplot(g1[0,:])
    if scintmenu.value == 'All' and ~np.all(np.isnan(state_dict['I_fcal'])):
        ax1.plot(state_dict['freq_test'][0],state_dict['I_fcal'],label='Spectrum')
        ax1.set_title("All Components")
        ax1.set_xlim(np.min(state_dict['freq_test'][0]),np.max(state_dict['freq_test'][0]))
    elif scintmenu.value != 'All' and state_dict['n_comps'] > 1:
        i = int(scintmenu.value[-1])
        ax1.plot(state_dict['freq_test'][0],state_dict['comps'][i]['I_fcal'],label='Spectrum')
        ax1.set_title("Component " + str(i))
        ax1.set_xlim(np.min(state_dict['freq_test'][0]),np.max(state_dict['freq_test'][0]))
    ax1.set_xlabel("Frequency (MHz)")
    ax1.set_ylabel("S/N")
    ax1.legend(loc='upper right',fontsize=18)

    #plt.subplot(312)
    ax2 = fig.add_subplot(g2[1,:])
    if scintmenu.value == 'All' and ~np.all(np.isnan(state_dict['I_fcal'])):
        ax2.plot(state_dict['scint_lags'],state_dict['autocorr_I'],label='ACF')
        ax2.plot(state_dict['scint_lags'],scint.lorentz(state_dict['scint_lags'],gamma_guess.value,m_guess.value,c_guess.value),label='Initial Guess',color='purple')
        ax2.set_xlim(np.min(state_dict['scint_lags']),np.max(state_dict['scint_lags']))
    elif scintmenu.value != 'All' and state_dict['n_comps'] > 1:
        i = int(scintmenu.value[-1])
        ax2.plot(state_dict['comps'][i]['scint_lags'],state_dict['comps'][i]['autocorr_I'],label='ACF')
        ax2.plot(state_dict['comps'][i]['scint_lags'],scint.lorentz(state_dict['comps'][i]['scint_lags'],gamma_guess.value,m_guess.value,c_guess.value),label='Initial Guess',color='purple')
        ax2.set_xlim(np.min(state_dict['scint_lags']),np.max(state_dict['scint_lags']))
    #plt.xlabel("Lag (MHz)")
    ax2.set_ylabel("Intensity")
    ax2.set_title("ACF")

    if calc_bw_button.clicked:
        """
        # Define frequency resolution (Hz) 
        f_res = state_dict['n_f']*state_dict['base_df'] #Hz

        # Define a lag range for fitting the ACF in frequency lag space
        lagrange_for_fit = 100

        # Load profile data as numpy array or convert data to numpy array
        spec = state_dict['I_fcal']#np.load('./37888771_sb1.npy')

        # Compute the autocorrelation function (ACF)
        acf = state_dict['autocorr_I']#scint.autocorr(spec)

        # Create array of lag values
        lags = np.arange(len(acf)) + 1
        acf = acf[1:]
        lags = lags[1:]

        # Create symmetric ACF and lags for fitting
        #acf = np.concatenate((acf[::-1], acf))
        #lags = np.concatenate((-1 * lags[::-1], lags)) * f_res
        """

        
        # Define a lag range for fitting the ACF in frequency lag space
        lagrange_for_fit = scint_fit_range.value# MHz #*1e6 #100E6

        badflag = 0
        if scintmenu.value == 'All' and ~np.all(np.isnan(state_dict['I_fcal'])):
            acf = state_dict['autocorr_I']
            lags = state_dict['scint_lags']
        elif scintmenu.value != 'All' and state_dict['n_comps'] > 1 and ~np.all(np.isnan(state_dict['comps'][int(str(scintmenu.value[-1]))]['I_fcal'])):
            i = int(scintmenu.value[-1])
            acf = state_dict['comps'][i]['autocorr_I']
            lags = state_dict['comps'][i]['scint_lags']
        else: 
            badflag = 1
            print("Previous Stages Required")

        if badflag == 0:
            acf = acf[~np.isnan(acf)]
            lags = lags[~np.isnan(lags)]
            # Set up the fit of a Lorentzian function to the ACF
            #gmodel = Model(lambda x, gamma1,m1,c: np.nan_to_num(scint.lorentz(x,gamma1,m1,c)),independent_vars=['x'],nan_policy='omit')
            
            if scintfitmenu.value == 'LMFIT Non-Linear Least Squares':
            
                gmodel = Model(scint.lorentz,independent_vars=['x'],nan_policy='omit')
                acf_for_fit = acf[int(len(acf) / 2.) - int(lagrange_for_fit / f_res) : int(len(acf) / 2.) + int(lagrange_for_fit / f_res)]
                lags_for_fit = lags[int(len(acf) / 2.) - int(lagrange_for_fit / f_res) : int(len(acf) / 2.) + int(lagrange_for_fit / f_res)]
                
                """f = open("tmp.csv","w")
                cf = csv.writer(f,delimiter=',')
                cf.writerow(lags_for_fit)
                cf.writerow(acf_for_fit)
                f.close()"""

                # Execute the fit
                result = gmodel.fit(acf_for_fit[~np.isnan(acf_for_fit)], x = lags_for_fit[~np.isnan(acf_for_fit)], gamma1=gamma_guess.value,m1=m_guess.value,c=c_guess.value)
                #                gamma1=Parameter(gamma_guess.value,min=1/1e3,max=50/1e3),
                #                m1=Parameter(m_guess.value,min=0,max=1),
                #                c1=Parameter(c_guess.value,min=-1,max=1))#gamma1 = 10, m1 = 1, c = 0)

                # Print or output the fit report for display
                #print(result.fit_report())

                df_scint.loc[str(scintmenu.value)] = [result.params['gamma1'].value,
                               result.params['gamma1'].stderr,
                               result.params['m1'].value,
                               result.params['m1'].stderr,
                               result.params['c'].value,
                               result.params['c'].stderr]
                if str(scintmenu.value) == 'All':
                    state_dict['gamma_best'] = [result.params['gamma1'].value,result.params['gamma1'].stderr]
                    state_dict['m_best'] = [result.params['m1'].value,result.params['m1'].stderr]
                    state_dict['c_best'] = [result.params['c'].value,result.params['c'].stderr]
                    state_dict['scint_residuals'] = state_dict['autocorr_I'] - scint.lorentz(state_dict['scint_lags'],state_dict['gamma_best'][0],state_dict['m_best'][0],state_dict['c_best'][0])
                else:
                    i = int(scintmenu.value[-1])
                    state_dict['comps'][i]['gamma_best'] = [result.params['comps'][i]['gamma1'].value,result.params['comps'][i]['gamma1'].stderr]
                    state_dict['comps'][i]['m_best'] = [result.params['m1'].value,result.params['m1'].stderr]
                    state_dict['comps'][i]['c_best'] = [result.params['c'].value,result.params['c'].stderr]
                    state_dict['comps'][i]['scint_residuals'] = state_dict['comps'][i]['autocorr_I'] - scint.lorentz(state_dict['comps'][i]['scint_lags'],state_dict['comps'][i]['gamma_best'][0],state_dict['comps'][i]['m_best'][0],state_dict['comps'][i]['c_best'][0])


            else:

                acf_for_fit = acf[int(len(acf) / 2.) - int(lagrange_for_fit / f_res) : int(len(acf) / 2.) + int(lagrange_for_fit / f_res)]
                lags_for_fit = lags[int(len(acf) / 2.) - int(lagrange_for_fit / f_res) : int(len(acf) / 2.) + int(lagrange_for_fit / f_res)]

                popt,pcov = curve_fit(scint.lorentz,lags_for_fit[~np.isnan(acf_for_fit)],acf_for_fit[~np.isnan(acf_for_fit)],p0=[gamma_guess.value,m_guess.value,c_guess.value])

                df_scint.loc[str(scintmenu.value)] = [popt[0],
                               np.sqrt(pcov[0,0]),
                               popt[1],
                               np.sqrt(pcov[1,1]),
                               popt[2],
                               np.sqrt(pcov[2,2])]


                if str(scintmenu.value) == 'All':
                    state_dict['gamma_best'] = [popt[0],np.sqrt(pcov[0,0])]
                    state_dict['m_best'] = [popt[1],np.sqrt(pcov[1,1])]
                    state_dict['c_best'] = [popt[2],np.sqrt(pcov[2,2])]
                    state_dict['scint_residuals'] = state_dict['autocorr_I'] - scint.lorentz(state_dict['scint_lags'],state_dict['gamma_best'][0],state_dict['m_best'][0],state_dict['c_best'][0])
                else:
                    i = int(scintmenu.value[-1])
                    state_dict['comps'][i]['gamma_best'] = [popt[0],np.sqrt(pcov[0,0])]
                    state_dict['comps'][i]['m_best'] = [popt[1],np.sqrt(pcov[1,1])]
                    state_dict['comps'][i]['c_best'] = [popt[2],np.sqrt(pcov[2,2])]                    
                    state_dict['comps'][i]['scint_residuals'] = state_dict['comps'][i]['autocorr_I'] - scint.lorentz(state_dict['comps'][i]['scint_lags'],state_dict['comps'][i]['gamma_best'][0],state_dict['comps'][i]['m_best'][0],state_dict['comps'][i]['c_best'][0])

    #plot the resulting fit
    if scintmenu.value == 'All' and 'gamma_best' in state_dict.keys():
        ax2.plot(state_dict['scint_lags'],scint.lorentz(state_dict['scint_lags'],state_dict['gamma_best'][0],state_dict['m_best'][0],state_dict['c_best'][0]),label='Least-Squares Fit',color='red')
    elif scintmenu.value != 'All' and state_dict['n_comps'] > 1:
        i = int(scintmenu.value[-1])
        if 'gamma_best' in state_dict['comps'][i].keys():
            ax2.plot(state_dict['comps'][i]['scint_lags'],scint.lorentz(state_dict['comps'][i]['scint_lags'],state_dict['comps'][i]['gamma_best'][0],state_dict['comps'][i]['m_best'][0],state_dict['comps'][i]['c_best'][0]),label='Least-Squares Fit',color='red')
    ax2.legend(loc='upper right',fontsize=18)
    ax2.axvspan(-scint_fit_range.value,scint_fit_range.value,color='red',alpha=0.2)
    ax2.set_xticks([])

    #plot the residuals
    #plt.subplot(313)
    ax3 = fig.add_subplot(g2[2,:])#,sharex=ax2)
    if scintmenu.value == 'All' and ~np.all(np.isnan(state_dict['I_fcal'])):
        ax3.plot(state_dict['scint_lags'],state_dict['autocorr_I'] - scint.lorentz(state_dict['scint_lags'],gamma_guess.value,m_guess.value,c_guess.value),label='Initial Residuals',color='purple')
        if 'gamma_best' in state_dict.keys():
            ax3.plot(state_dict['scint_lags'],state_dict['scint_residuals'],label='LSF Residuals',color='red')
        ax3.set_xlim(np.min(state_dict['scint_lags']),np.max(state_dict['scint_lags']))
    elif scintmenu.value != 'All' and state_dict['n_comps'] > 1:
        i = int(scintmenu.value[-1])
        if ~np.all(np.isnan(state_dict['comps'][i]['I_fcal'])) and state_dict['n_comps'] > 1:
            ax3.plot(state_dict['comps'][i]['scint_lags'],state_dict['comps'][i]['autocorr_I'] - scint.lorentz(state_dict['comps'][i]['scint_lags'],gamma_guess.value,m_guess.value,c_guess.value),label='Initial Residuals',color='purple')
            if 'gamma_best' in state_dict['comps'][i].keys():
                ax3.plot(state_dict['comps'][i]['scint_lags'],state_dict['comps'][i]['scint_residuals'],label='LSF Residuals',color='red')
            ax3.set_xlim(np.min(state_dict['scint_lags']),np.max(state_dict['scint_lags']))
    ax3.axvspan(-scint_fit_range.value,scint_fit_range.value,color='red',alpha=0.2)
    ax3.axhline(0,linestyle='--',color='black')
    ax3.set_xlabel("Lag (MHz)")
    ax3.set_ylabel(r"$\Delta$")
    ax3.legend(loc='upper right',fontsize=18)


    if save_scint_button.clicked:
        plt.savefig(state_dict['datadir'] + "/" + state_dict['ids'] + "_" + state_dict['nickname'] + "_scintillation.pdf")
        df_scint.to_csv(state_dict['datadir'] + "/" + state_dict['ids'] + "_" + state_dict['nickname'] + "_scintillation_params.csv")

    plt.show()


    #update wdict
    update_wdict([gamma_guess,m_guess,c_guess,scintmenu,scint_fit_range,scintfitmenu],
            ['gamma_guess','m_guess','c_guess','scintmenu','scint_fit_range','scintfitmenu'])

    return

"""
Spectral Index Fitting Screen
"""
def specidx_screen(specidxfitmenu,calc_specidx_button,specidx_guess,F0_guess,specidxmenu,save_specidx_button):


    #plot frequency spectrum and initial guess fit 

    g2 = plt.GridSpec(2,1,hspace=0,height_ratios=[2,1],top=0.7)
    fig = plt.figure(figsize=(18,12))
    ax1 = fig.add_subplot(g2[0,:])
    if specidxmenu.value == 'All' and ~np.all(np.isnan(state_dict['I_fcal'])):
        ax1.plot(state_dict['freq_test'][0],state_dict['I_fcal'],label='Spectrum')
        ax1.set_title("All Components")
        ax1.set_xlim(np.min(state_dict['freq_test'][0]),np.max(state_dict['freq_test'][0]))
    elif specidxmenu.value != 'All' and state_dict['n_comps'] > 1:
        i = int(specidxmenu.value[-1])
        ax1.plot(state_dict['freq_test'][0],state_dict['comps'][i]['I_fcal'],label='Spectrum')
        ax1.set_title("Component " + str(i))
        ax1.set_xlim(np.min(state_dict['freq_test'][0]),np.max(state_dict['freq_test'][0]))
    ax1.plot(state_dict['freq_test'][0],scatscint.specidx_fit_fn(state_dict['freq_test'][0],specidx_guess.value,F0_guess.value),label='Initial Guess',color='purple')
    ax1.set_xlabel("Frequency (MHz)")
    ax1.set_ylabel("S/N")

    if calc_specidx_button.clicked and ((specidxmenu.value == 'All' and ~np.all(np.isnan(state_dict['I_fcal']))) or (specidxmenu.value != 'All' and state_dict['n_comps'] > 1 and ~np.all(np.isnan(state_dict['comps'][i]['I_fcal'])))):

        if (specidxmenu.value == 'All' and ~np.all(np.isnan(state_dict['I_fcal']))):
            spec_for_fit = state_dict['I_fcal'][~np.isnan(state_dict['I_fcal'])]
            freq_for_fit = state_dict['freq_test'][0][~np.isnan(state_dict['I_fcal'])]
        else:
            i = int(specidxmenu.value[-1])
            spec_for_fit = state_dict['comps'][i]['I_fcal'][~np.isnan(state_dict['comps'][i]['I_fcal'])]
            freq_for_fit = state_dict['freq_test'][0][~np.isnan(state_dict['comps'][i]['I_fcal'])]

        #weight the spectrum by the intensity
        weights_for_fit = np.clip(spec_for_fit,a_min=0,a_max=np.inf)
        sigma_for_fit = 1/np.clip(spec_for_fit,a_min=0,a_max=np.inf)
        spec_for_fit = spec_for_fit[~np.isinf(sigma_for_fit)]
        freq_for_fit = freq_for_fit[~np.isinf(sigma_for_fit)]
        weights_for_fit = weights_for_fit[~np.isinf(sigma_for_fit)]
        sigma_for_fit = sigma_for_fit[~np.isinf(sigma_for_fit)]
        

        if specidxfitmenu.value == 'LMFIT Non-Linear Least Squares':

            #LMFIT least squares fit
            gmodel = Model(scatscint.specidx_fit_fn,independent_vars=['x'],nan_policy='omit')

            # Execute the fit
            result = gmodel.fit(spec_for_fit, x = freq_for_fit, Gamma=specidx_guess.value,F0=F0_guess.value,weights=weights_for_fit)

            df_specidx.loc[str(specidxmenu.value)] = [result.params['Gamma'].value,
                               result.params['Gamma'].stderr,
                               result.params['F0'].value,
                               result.params['F0'].stderr]
                               
            if str(specidxmenu.value) == 'All':
                state_dict['specidx_best'] = [result.params['Gamma'].value,result.params['Gamma'].stderr]
                state_dict['F0_best'] = [result.params['F0'].value,result.params['F0'].stderr]
                state_dict['specidx_residuals'] = state_dict['I_fcal'] - scatscint.specidx_fit_fn(state_dict['freq_test'][0],state_dict['specidx_best'][0],state_dict['F0_best'][0])  
            else:
                i = int(specidxmenu.value[-1])
                state_dict['comps'][i]['specidx_best'] = [result.params['Gamma'].value,result.params['Gamma'].stderr]
                state_dict['comps'][i]['F0_best'] = [result.params['F0'].value,result.params['F0'].stderr]
                state_dict['comps'][i]['specidx_residuals'] = state_dict['comps'][i]['I_fcal'] - scatscint.specidx_fit_fn(state_dict['freq_test'][0],state_dict['comps'][i]['specidx_best'][0],state_dict['comps'][i]['F0_best'][0])

        else: #least squares fitting
            

            popt,pcov = curve_fit(scatscint.specidx_fit_fn,freq_for_fit,spec_for_fit,p0=[specidx_guess.value,F0_guess.value],sigma=sigma_for_fit)

            df_specidx.loc[str(specidxmenu.value)] = [popt[0],
                               np.sqrt(pcov[0,0]),
                               popt[1],
                               np.sqrt(pcov[1,1])]
                               

            if str(specidxmenu.value) == 'All':
                state_dict['specidx_best'] = [popt[0],np.sqrt(pcov[0,0])]
                state_dict['F0_best'] = [popt[1],np.sqrt(pcov[1,1])]
                state_dict['specidx_residuals'] = state_dict['I_fcal'] - scatscint.specidx_fit_fn(state_dict['freq_test'][0],state_dict['specidx_best'][0],state_dict['F0_best'][0])
            else:
                i = int(specidxmenu.value[-1])
                state_dict['comps'][i]['specidx_best'] =[popt[0],np.sqrt(pcov[0,0])]
                state_dict['comps'][i]['F0_best'] =[popt[1],np.sqrt(pcov[1,1])]
                state_dict['comps'][i]['specidx_residuals'] = state_dict['comps'][i]['I_fcal'] - scatscint.specidx_fit_fn(state_dict['freq_test'][0],state_dict['comps'][i]['specidx_best'][0],state_dict['comps'][i]['F0_best'][0])


    #plot residuals
    ax3 = fig.add_subplot(g2[1,:])#,sharex=ax2)
    if specidxmenu.value == 'All' and ~np.all(np.isnan(state_dict['I_fcal'])):
        ax3.plot(state_dict['freq_test'][0],state_dict['I_fcal'] - scatscint.specidx_fit_fn(state_dict['freq_test'][0],specidx_guess.value,F0_guess.value),label='Initial Residuals',color='purple')
        if 'specidx_best' in state_dict.keys():
            #best fit
            ax1.plot(state_dict['freq_test'][0],scatscint.specidx_fit_fn(state_dict['freq_test'][0],state_dict['specidx_best'][0],state_dict['F0_best'][0]),label='Best Fit',color='red',linewidth=3)
            ax3.plot(state_dict['freq_test'][0],state_dict['specidx_residuals'],label='LSF Residuals',color='red')
        ax3.set_xlim(np.min(state_dict['freq_test'][0]),np.max(state_dict['freq_test'][0]))
    elif specidxmenu.value != 'All' and state_dict['n_comps'] > 1:
        i = int(specidxmenu.value[-1])
        if ~np.all(np.isnan(state_dict['comps'][i]['I_fcal'])) and state_dict['n_comps'] > 1:
            ax3.plot(state_dict['freq_test'][0],state_dict['comps'][i]['I_fcal'] - scatscint.specidx_fit_fn(state_dict['freq_test'][0],specidx_guess.value,F0_guess.value),label='Initial Residuals',color='purple')
            if 'specidx_best' in state_dict['comps'][i].keys():
                #best fit
                ax1.plot(state_dict['freq_test'][0],scatscint.specidx_fit_fn(state_dict['freq_test'][0],state_dict['comps'][i]['specidx_best'][0],state_dict['comps'][i]['F0_best'][0]),label='Best Fit',color='red',linewidth=3)
                ax3.plot(state_dict['freq_test'][0],state_dict['comps'][i]['specidx_residuals'],label='LSF Residuals',color='red')
        ax3.set_xlim(np.min(state_dict['freq_test'][0]),np.max(state_dict['freq_test'][0]))
    ax3.set_xlabel("Frequency (MHz)")
    ax3.set_ylabel(r"$\Delta$")
    ax3.legend(loc='upper right',fontsize=18)
    ax1.legend(loc='upper right',fontsize=18)

    if save_specidx_button.clicked:
        plt.savefig(state_dict['datadir'] + "/" + state_dict['ids'] + "_" + state_dict['nickname'] + "_specidx.pdf")
        df_specidx.to_csv(state_dict['datadir'] + "/" + state_dict['ids'] + "_" + state_dict['nickname'] + "_specidx_params.csv")

    plt.show()


    #update wdict
    update_wdict([specidx_guess,F0_guess],
            ['specidx_guess','F0_guess'])

    return
"""
RM Synthesis State
"""
def RM_screen(useRMTools,maxRM_num_tools,dRM_tools,useRMsynth,nRM_num,minRM_num,
                 maxRM_num,getRMbutton,useRM2D,nRM_num_zoom,RM_window_zoom,dRM_tools_zoom,
                 getRMbutton_zoom,RM_gal_display,RM_galerr_display,RM_ion_display,RM_ionerr_display,
                 getRMgal_button,getRMion_button,rmcomp_menu,RMsynthbackground,refresh_button):
    #signal.signal(signal.SIGUSR1,handler)    
    #update component options
    update_wdict([rmcomp_menu],['rmcomp_menu'],
                param='value')

    #update RM displays
    if state_dict['RM_galRA'] != state_dict['RA'] or state_dict['RM_galDEC'] != state_dict['DEC'] or getRMgal_button.clicked:
        state_dict['RM_gal'],state_dict['RM_galerr'] = get_rm(radec=(state_dict['RA'],state_dict['DEC']),filename=repo_path + "/data/faraday2020v2.hdf5")
        state_dict['RM_gal'] = np.around(state_dict['RM_gal'],2)
        state_dict['RM_galerr'] = np.around(state_dict['RM_galerr'],2)
        state_dict['RM_galRA'] = state_dict['RA']
        state_dict['RM_galDEC'] = state_dict['DEC']
    RM_gal_display.data = np.around(state_dict['RM_gal'],2)
    RM_galerr_display.data = np.around(state_dict['RM_galerr'],2)
    
    if state_dict['RM_ionRA'] != state_dict['RA'] or state_dict['RM_ionDEC'] != state_dict['DEC'] or state_dict['RM_ionmjd'] != state_dict['mjd'] or getRMion_button.clicked:
        state_dict['RM_ion'],state_dict['RM_ionerr'] = RMcal.get_rm_ion(state_dict['RA'],state_dict['DEC'],state_dict['mjd'])
        state_dict['RM_ion'] = np.around(state_dict['RM_ion'],2)
        state_dict['RM_ionerr'] = np.around(state_dict['RM_ionerr'],2)
        state_dict['RM_ionRA'] = state_dict['RA']
        state_dict['RM_ionDEC'] = state_dict['DEC']
        state_dict['RM_ionmjd'] = state_dict['mjd']
    RM_ion_display.data = np.around(state_dict['RM_ion'],2)
    RM_ionerr_display.data = np.around(state_dict['RM_ionerr'],2)



    #if run button is clicked, run RM synthesis for selected component 
    if getRMbutton.clicked:
        #make dynamic spectra w/ high frequency resolution
        Ip_full = dsapol.avg_time(state_dict['base_Ical'],state_dict['rel_n_t'])
        Qp_full = dsapol.avg_time(state_dict['base_Qcal'],state_dict['rel_n_t'])
        Up_full = dsapol.avg_time(state_dict['base_Ucal'],state_dict['rel_n_t'])
        Vp_full = dsapol.avg_time(state_dict['base_Vcal'],state_dict['rel_n_t'])

        if rmcomp_menu.value != 'All' and rmcomp_menu.value != '':
            

            if state_dict['n_comps'] > 1:                
                #loop through each component
                #for i in range(state_dict['n_comps']):
                
                #select component
                i = int(rmcomp_menu.value)
                    
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
                If,Qf,Uf,Vf = dsapol.get_stokes_vs_freq(Ip,Qp,Up,Vp,state_dict['comps'][i]['width_native'],state_dict['tsamp'],state_dict['base_n_f'],state_dict['n_t'],state_dict['base_freq_test'],n_off=int(NOFFDEF/state_dict['n_t']),plot=False,show=False,normalize=True,buff=state_dict['comps'][i]['buff'],weighted=True,n_t_weight=state_dict['comps'][i]['avger_w'],timeaxis=state_dict['time_axis'],fobj=FilReader(state_dict['datadir']+state_dict['ids'] + state_dict['suff'] + "_0.fil"),sf_window_weights=state_dict['comps'][i]['sf_window_weights'],input_weights=state_dict['comps'][i]['weights'])
                
                #STAGE 1: RM-TOOLS
                if useRMTools.value: #STAGE 1: RM-TOOLS
                    n_off = int(NOFFDEF/state_dict['n_t'])

                    RM,RMerr,state_dict['comps'][i]['RMcalibrated']['RM_tools_snrs'],state_dict['comps'][i]['RMcalibrated']['trial_RM_tools'] = RMcal.get_RM_tools(If,Qf,Uf,Vf,Ip,Qp,Up,Vp,state_dict['base_freq_test'],state_dict['n_t'],maxRM_num_tools=maxRM_num_tools.value,dRM_tools=dRM_tools.value,n_off=int(NOFFDEF/state_dict['n_t']))
                    state_dict['comps'][i]['RMcalibrated']['RM_tools'] = [RM,RMerr]

                    #update table
                    RMdf.loc[str(i), 'RM-Tools'] = RM
                    RMdf.loc[str(i), 'RM-Tools Error'] = RMerr


                #STAGE 2: 1D RM synthesis
                if useRMsynth.value:
                    n_off = int(NOFFDEF/state_dict['n_t'])

                    if RMsynthbackground.value:

                        state_dict['comps'][i]['RMcalibrated']['dname_1D'] = RMcal.get_RM_1D(If,Qf,Uf,Vf,Ip,Qp,Up,Vp,state_dict['comps'][i]['timestart'],state_dict['comps'][i]['timestop'],state_dict['base_freq_test'],nRM_num=nRM_num.value,minRM_num=minRM_num.value,maxRM_num=maxRM_num.value,n_off=n_off,fit=False,weights=state_dict['comps'][i]['weights'],background=True)

                        # disregard return values because we'll read from file to see when ready
                    
                    else:
                        RM,RMerr,state_dict['comps'][i]['RMcalibrated']['RMsnrs1'],state_dict['comps'][i]['RMcalibrated']['trial_RM1'] = RMcal.get_RM_1D(If,Qf,Uf,Vf,Ip,Qp,Up,Vp,state_dict['comps'][i]['timestart'],state_dict['comps'][i]['timestop'],state_dict['base_freq_test'],nRM_num=nRM_num.value,minRM_num=minRM_num.value,maxRM_num=maxRM_num.value,n_off=n_off,fit=False,weights=state_dict['comps'][i]['weights'])
                        state_dict['comps'][i]['RMcalibrated']["RM1"] = [RM,RMerr]

                        #update table
                        RMdf.loc[str(i), '1D-Synth'] = RM
                        RMdf.loc[str(i), '1D-Synth Error'] = RMerr

        elif rmcomp_menu.value == 'All':
            If,Qf,Uf,Vf = dsapol.get_stokes_vs_freq(Ip_full,Qp_full,Up_full,Vp_full,state_dict['width_native'],state_dict['tsamp'],state_dict['base_n_f'],state_dict['n_t'],state_dict['base_freq_test'],n_off=int(NOFFDEF/state_dict['n_t']),plot=False,show=False,normalize=True,buff=state_dict['buff'],weighted=True,timeaxis=state_dict['time_axis'],fobj=FilReader(state_dict['datadir']+state_dict['ids'] + state_dict['suff'] + "_0.fil"),input_weights=state_dict['weights'])
        
            #STAGE 1: RM-TOOLS (Full burst)
            if useRMTools.value: 
                RM,RMerr,state_dict["RMcalibrated"]['RM_tools_snrs'],state_dict["RMcalibrated"]['trial_RM_tools'] = RMcal.get_RM_tools(If,Qf,Uf,Vf,Ip_full,Qp_full,Up_full,Vp_full,state_dict['base_freq_test'],state_dict['n_t'],maxRM_num_tools=maxRM_num_tools.value,dRM_tools=dRM_tools.value,n_off=int(NOFFDEF/state_dict['n_t']))
                state_dict["RMcalibrated"]['RM_tools'] = [RM,RMerr]


                #update table
                RMdf.loc['All', 'RM-Tools'] = RM
                RMdf.loc['All', 'RM-Tools Error'] = RMerr
        
            #STAGE 2: 1D RM synthesis
            if useRMsynth.value:
                if RMsynthbackground.value:
                    state_dict['RMcalibrated']['dname_1D'] = RMcal.get_RM_1D(If,Qf,Uf,Vf,Ip_full,Qp_full,Up_full,Vp_full,state_dict['timestart'],state_dict['timestop'],state_dict['base_freq_test'],nRM_num=nRM_num.value,minRM_num=minRM_num.value,maxRM_num=maxRM_num.value,n_off=int(NOFFDEF/state_dict['n_t']),fit=False,weights=state_dict['weights'],background=True)

                else:
                    
                    RM,RMerr,state_dict["RMcalibrated"]['RMsnrs1'],state_dict["RMcalibrated"]['trial_RM1'] = RMcal.get_RM_1D(If,Qf,Uf,Vf,Ip_full,Qp_full,Up_full,Vp_full,state_dict['timestart'],state_dict['timestop'],state_dict['base_freq_test'],nRM_num=nRM_num.value,minRM_num=minRM_num.value,maxRM_num=maxRM_num.value,n_off=int(NOFFDEF/state_dict['n_t']),fit=False,weights=state_dict['weights'])
                    state_dict["RMcalibrated"]["RM1"] = [RM,RMerr]

                    #update table
                    RMdf.loc['All', '1D-Synth'] = RM
                    RMdf.loc['All', '1D-Synth Error'] = RMerr

    #if run button 2 is clicked, run RM synthesis for all selected component, zoom around initial estimate
    if (getRMbutton_zoom.clicked and 
        ((rmcomp_menu.value == 'All' and (~np.isnan(state_dict["RMcalibrated"]["RM2"][0]) or 
                                          ~np.isnan(state_dict["RMcalibrated"]["RM1"][0]) or 
                                          ~np.isnan(state_dict["RMcalibrated"]["RM_tools"][0]))) or 
        (rmcomp_menu.value != 'All' and rmcomp_menu.value != '' and (~np.isnan(state_dict['comps'][int(rmcomp_menu.value)]["RMcalibrated"]["RM2"][0]) or 
                                                                     ~np.isnan(state_dict['comps'][int(rmcomp_menu.value)]["RMcalibrated"]["RM1"][0]) or 
                                                                     ~np.isnan(state_dict['comps'][int(rmcomp_menu.value)]["RMcalibrated"]["RM_tools"][0]))))):

        #make dynamic spectra w/ high frequency resolution
        Ip_full = dsapol.avg_time(state_dict['base_Ical'],state_dict['rel_n_t'])
        Qp_full = dsapol.avg_time(state_dict['base_Qcal'],state_dict['rel_n_t'])
        Up_full = dsapol.avg_time(state_dict['base_Ucal'],state_dict['rel_n_t'])
        Vp_full = dsapol.avg_time(state_dict['base_Vcal'],state_dict['rel_n_t'])


        if rmcomp_menu.value != 'All' and rmcomp_menu.value != '':
            i = int(rmcomp_menu.value)

            #get center RM
            if ~np.isnan(state_dict['comps'][i]["RMcalibrated"]["RM2"][0]): RMcenter = state_dict['comps'][i]["RMcalibrated"]["RM2"][0]
            elif ~np.isnan(state_dict['comps'][i]["RMcalibrated"]["RM1"][0]): RMcenter = state_dict['comps'][i]["RMcalibrated"]["RM1"][0]
            else: RMcenter = state_dict['comps'][i]["RMcalibrated"]["RM_tools"][0]

            if state_dict['n_comps'] > 1:
        
                #loop through each component
                #for i in range(state_dict['n_comps']):
                
                #select component
                i = int(rmcomp_menu.value)
                    
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
                If,Qf,Uf,Vf = dsapol.get_stokes_vs_freq(Ip,Qp,Up,Vp,state_dict['comps'][i]['width_native'],state_dict['tsamp'],state_dict['base_n_f'],state_dict['n_t'],state_dict['base_freq_test'],n_off=int(NOFFDEF/state_dict['n_t']),plot=False,show=False,normalize=True,buff=state_dict['comps'][i]['buff'],weighted=True,n_t_weight=state_dict['comps'][i]['avger_w'],timeaxis=state_dict['time_axis'],fobj=FilReader(state_dict['datadir']+state_dict['ids'] + state_dict['suff'] + "_0.fil"),sf_window_weights=state_dict['comps'][i]['sf_window_weights'],input_weights=state_dict['comps'][i]['weights'])

                #STAGE 1: RM-TOOLS
                if useRMTools.value: #STAGE 1: RM-TOOLS
                    n_off = int(NOFFDEF/state_dict['n_t'])

                    RM,RMerr,state_dict['comps'][i]['RMcalibrated']['RM_tools_snrszoom'],state_dict['comps'][i]['RMcalibrated']['trial_RM_toolszoom'] = RMcal.get_RM_tools(If,Qf,Uf,Vf,Ip,Qp,Up,Vp,state_dict['base_freq_test'],state_dict['n_t'],maxRM_num_tools=np.abs(RMcenter)+RM_window_zoom.value,dRM_tools=dRM_tools_zoom.value,n_off=int(NOFFDEF/state_dict['n_t']))
                    state_dict['comps'][i]['RMcalibrated']['RM_toolszoom'] = [RM,RMerr]
                    
                    #update table
                    RMdf.loc[str(i), 'RM-Tools'] = RM
                    RMdf.loc[str(i), 'RM-Tools Error'] = RMerr

                #STAGE 2: 1D RM synthesis
                if useRMsynth.value:
                    n_off = int(NOFFDEF/state_dict['n_t'])
                    
                    if RMsynthbackground.value:
                        state_dict['comps'][i]["RMcalibrated"]["dname_1D_zoom"] = RMcal.get_RM_1D(If,Qf,Uf,Vf,Ip,Qp,Up,Vp,state_dict['comps'][i]['timestart'],state_dict['comps'][i]['timestop'],state_dict['base_freq_test'],nRM_num=nRM_num_zoom.value,minRM_num=RMcenter-RM_window_zoom.value,maxRM_num=RMcenter+RM_window_zoom.value,n_off=n_off,fit=True,weights=state_dict['comps'][i]['weights'],background=True)
                    else:
                        RM,RMerr,state_dict['comps'][i]['RMcalibrated']['RMsnrs1zoom'],state_dict['comps'][i]['RMcalibrated']['trial_RM1zoom'] = RMcal.get_RM_1D(If,Qf,Uf,Vf,Ip,Qp,Up,Vp,state_dict['comps'][i]['timestart'],state_dict['comps'][i]['timestop'],state_dict['base_freq_test'],nRM_num=nRM_num_zoom.value,minRM_num=RMcenter-RM_window_zoom.value,maxRM_num=RMcenter+RM_window_zoom.value,n_off=n_off,fit=True,weights=state_dict['comps'][i]['weights'])
                        state_dict['comps'][i]['RMcalibrated']["RM1zoom"] = [RM,RMerr]
                        state_dict['comps'][i]["RMcalibrated"]['trial_RM2'] = copy.deepcopy(state_dict['comps'][i]["RMcalibrated"]['trial_RM1zoom'])
                        if np.all(np.isnan(state_dict['comps'][i]['RMcalibrated']['RMsnrs2'])): state_dict['comps'][i]['RMcalibrated']['RMsnrs2'] = np.nan*np.ones(len(state_dict['comps'][i]["RMcalibrated"]['trial_RM2']))

                        #update table
                        RMdf.loc[str(i), '1D-Synth'] = RM
                        RMdf.loc[str(i), '1D-Synth Error'] = RMerr

                #STAGE 3: 2D RM synthesis
                if useRM2D.value:
                    n_off = int(NOFFDEF/state_dict['n_t'])

                    if RMsynthbackground.value:
                        state_dict['comps'][i]["RMcalibrated"]['dname_2D'] = RMcal.get_RM_2D(Ip,Qp,Up,Vp,state_dict['comps'][i]['timestart'],state_dict['comps'][i]['timestop'],state_dict['comps'][i]['width_native'],state_dict['tsamp'],state_dict['comps'][i]['buff'],1,state_dict['n_t'],state_dict['base_freq_test'],state_dict['time_axis'],nRM_num=nRM_num_zoom.value,minRM_num=RMcenter-RM_window_zoom.value,maxRM_num=RMcenter+RM_window_zoom.value,n_off=n_off,fit=True,weights=state_dict['comps'][i]['weights'],background=True)
                    else:
                        RM2,RMerr2,upp,low,state_dict['comps'][i]['RMcalibrated']['RMsnrs2'],state_dict['comps'][i]['RMcalibrated']['SNRs_full'],state_dict['comps'][i]['RMcalibrated']['trial_RM2'] = RMcal.get_RM_2D(Ip,Qp,Up,Vp,state_dict['comps'][i]['timestart'],state_dict['comps'][i]['timestop'],state_dict['comps'][i]['width_native'],state_dict['tsamp'],state_dict['comps'][i]['buff'],1,state_dict['n_t'],state_dict['base_freq_test'],state_dict['time_axis'],nRM_num=nRM_num_zoom.value,minRM_num=RMcenter-RM_window_zoom.value,maxRM_num=RMcenter+RM_window_zoom.value,n_off=n_off,fit=True,weights=state_dict['comps'][i]['weights'])
                        state_dict['comps'][i]['RMcalibrated']['RM2'] = [RM2,RMerr2,upp,low]
                        state_dict['comps'][i]['RMcalibrated']['RMerrfit'] = RMerr2
                        state_dict['comps'][i]['RMcalibrated']['RMFWHM'] = upp-low
                        #update table
                        RMdf.loc[str(i), '2D-Synth'] = RM2
                        RMdf.loc[str(i), '2D-Synth Error'] = RMerr2

        elif rmcomp_menu.value == 'All':
            
            #get center RM
            if ~np.isnan(state_dict["RMcalibrated"]["RM2"][0]): RMcenter = state_dict["RMcalibrated"]["RM2"][0]
            elif ~np.isnan(state_dict["RMcalibrated"]["RM1"][0]): RMcenter = state_dict["RMcalibrated"]["RM1"][0]
            else: RMcenter = state_dict["RMcalibrated"]["RM_tools"][0]
            
            If,Qf,Uf,Vf = dsapol.get_stokes_vs_freq(Ip_full,Qp_full,Up_full,Vp_full,state_dict['width_native'],state_dict['tsamp'],state_dict['base_n_f'],state_dict['n_t'],state_dict['base_freq_test'],n_off=int(NOFFDEF/state_dict['n_t']),plot=False,show=False,normalize=True,buff=state_dict['buff'],weighted=True,timeaxis=state_dict['time_axis'],fobj=FilReader(state_dict['datadir']+state_dict['ids'] + state_dict['suff'] + "_0.fil"),input_weights=state_dict['weights'])

            #STAGE 1: RM-TOOLS (Full burst)
            if useRMTools.value:
                RM,RMerr,state_dict["RMcalibrated"]['RM_tools_snrszoom'],state_dict["RMcalibrated"]['trial_RM_toolszoom'] = RMcal.get_RM_tools(If,Qf,Uf,Vf,Ip_full,Qp_full,Up_full,Vp_full,state_dict['base_freq_test'],state_dict['n_t'],maxRM_num_tools=np.abs(RMcenter)+RM_window_zoom.value,dRM_tools=dRM_tools_zoom.value,n_off=int(NOFFDEF/state_dict['n_t']))
                state_dict["RMcalibrated"]['RM_toolszoom'] = [RM,RMerr]

                #update table
                RMdf.loc['All', 'RM-Tools'] = RM
                RMdf.loc['All', 'RM-Tools Error'] = RMerr

            #STAGE 2: 1D RM synthesis
            if useRMsynth.value:
                if RMsynthbackground.value:
                    state_dict["RMcalibrated"]["dname_1D_zoom"] = RMcal.get_RM_1D(If,Qf,Uf,Vf,Ip_full,Qp_full,Up_full,Vp_full,state_dict['timestart'],state_dict['timestop'],state_dict['base_freq_test'],nRM_num=nRM_num_zoom.value,minRM_num=RMcenter-RM_window_zoom.value,maxRM_num=RMcenter+RM_window_zoom.value,n_off=int(NOFFDEF/state_dict['n_t']),fit=True,weights=state_dict['weights'],background=True)
                else:
                    RM,RMerr,state_dict["RMcalibrated"]['RMsnrs1zoom'],state_dict["RMcalibrated"]['trial_RM1zoom'] = RMcal.get_RM_1D(If,Qf,Uf,Vf,Ip_full,Qp_full,Up_full,Vp_full,state_dict['timestart'],state_dict['timestop'],state_dict['base_freq_test'],nRM_num=nRM_num_zoom.value,minRM_num=RMcenter-RM_window_zoom.value,maxRM_num=RMcenter+RM_window_zoom.value,n_off=int(NOFFDEF/state_dict['n_t']),fit=True,weights=state_dict['weights'])
                    state_dict["RMcalibrated"]["RM1zoom"] = [RM,RMerr]
                    state_dict["RMcalibrated"]['trial_RM2'] = copy.deepcopy(state_dict["RMcalibrated"]['trial_RM1zoom'])
                    if np.all(np.isnan(state_dict['RMcalibrated']['RMsnrs2'])): state_dict['RMcalibrated']['RMsnrs2'] = np.nan*np.ones(len(state_dict["RMcalibrated"]['trial_RM2']))

                    #update table
                    RMdf.loc['All', '1D-Synth'] = RM
                    RMdf.loc['All', '1D-Synth Error'] = RMerr


            #STAGE 3: 2D RM synthesis
            if useRM2D.value:
                n_off = int(NOFFDEF/state_dict['n_t'])

                if RMsynthbackground.value:
                    state_dict["RMcalibrated"]['dname_2D'] = RMcal.get_RM_2D(Ip_full,Qp_full,Up_full,Vp_full,state_dict['timestart'],state_dict['timestop'],state_dict['width_native'],state_dict['tsamp'],state_dict['buff'],1,state_dict['n_t'],state_dict['base_freq_test'],state_dict['time_axis'],nRM_num=nRM_num_zoom.value,minRM_num=RMcenter-RM_window_zoom.value,maxRM_num=RMcenter+RM_window_zoom.value,n_off=n_off,fit=True,weights=state_dict['weights'],background=True)
                else:
                    RM2,RMerr2,upp,low,state_dict['RMcalibrated']['RMsnrs2'],state_dict['RMcalibrated']['SNRs_full'],state_dict['RMcalibrated']['trial_RM2'] = RMcal.get_RM_2D(Ip_full,Qp_full,Up_full,Vp_full,state_dict['timestart'],state_dict['timestop'],state_dict['width_native'],state_dict['tsamp'],state_dict['buff'],1,state_dict['n_t'],state_dict['base_freq_test'],state_dict['time_axis'],nRM_num=nRM_num_zoom.value,minRM_num=RMcenter-RM_window_zoom.value,maxRM_num=RMcenter+RM_window_zoom.value,n_off=n_off,fit=True,weights=state_dict['weights'])
                    state_dict['RMcalibrated']['RM2'] = [RM2,RMerr2,upp,low]
                    state_dict['RMcalibrated']['RMerrfit'] = RMerr2
                    state_dict['RMcalibrated']['RMFWHM'] = upp-low

                    #update table
                    RMdf.loc['All', '2D-Synth'] = RM2
                    RMdf.loc['All', '2D-Synth Error'] = RMerr2
            
    elif getRMbutton_zoom.clicked:
        print("Run Initial RM synthesis first")




    if refresh_button.clicked:
        print("Refreshing...")
        if 'dname_1D' in state_dict["RMcalibrated"].keys():
            
            res = np.load(dirs['logs'] + "RM_files/" + state_dict["RMcalibrated"]['dname_1D'] + "result.npy")
            RM = res[0]
            RMerr = res[1]
            state_dict["RMcalibrated"]["RM1"] = [RM,RMerr]
            RMdf.loc['All', '1D-Synth'] = RM
            RMdf.loc['All', '1D-Synth Error'] = RMerr

            state_dict["RMcalibrated"]['RMsnrs1'] = np.load(dirs['logs'] + "RM_files/" + state_dict["RMcalibrated"]['dname_1D'] + "SNRs.npy")
            state_dict["RMcalibrated"]['trial_RM1'] = np.load(dirs['logs'] + "RM_files/" + state_dict["RMcalibrated"]['dname_1D'] + "trialRM.npy")

            order = state_dict["RMcalibrated"]['trial_RM1'].argsort()
            state_dict["RMcalibrated"]['RMsnrs1'] = state_dict["RMcalibrated"]['RMsnrs1'][order]
            state_dict["RMcalibrated"]['trial_RM1'] = state_dict["RMcalibrated"]['trial_RM1'][order]

        if 'dname_1D_zoom' in state_dict["RMcalibrated"].keys():
            res = np.load(dirs['logs'] + "RM_files/" + state_dict["RMcalibrated"]['dname_1D_zoom'] + "result.npy")
            RM = res[0]
            RMerr = res[1]
            state_dict["RMcalibrated"]["RM1zoom"] = [RM,RMerr]
            RMdf.loc['All', '1D-Synth'] = RM
            RMdf.loc['All', '1D-Synth Error'] = RMerr

            state_dict["RMcalibrated"]['RMsnrs1zoom'] = np.load(dirs['logs'] + "RM_files/" + state_dict["RMcalibrated"]['dname_1D_zoom'] + "SNRs.npy")
            state_dict["RMcalibrated"]['trial_RM1zoom'] = np.load(dirs['logs'] + "RM_files/" + state_dict["RMcalibrated"]['dname_1D_zoom'] + "trialRM.npy")

            order = state_dict["RMcalibrated"]['trial_RM1zoom'].argsort()
            state_dict["RMcalibrated"]['RMsnrs1zoom'] = state_dict["RMcalibrated"]['RMsnrs1zoom'][order]
            state_dict["RMcalibrated"]['trial_RM1zoom'] = state_dict["RMcalibrated"]['trial_RM1zoom'][order]

            state_dict["RMcalibrated"]['trial_RM2'] = copy.deepcopy(state_dict["RMcalibrated"]['trial_RM1zoom'])
            if np.all(np.isnan(state_dict['RMcalibrated']['RMsnrs2'])): state_dict['RMcalibrated']['RMsnrs2'] = np.nan*np.ones(len(state_dict["RMcalibrated"]['trial_RM2']))


        if 'dname_2D' in state_dict["RMcalibrated"].keys():
            res = np.load(dirs['logs'] + "RM_files/" + state_dict["RMcalibrated"]['dname_2D'] + "result.npy") 
            RM2,RMerr2,upp,low = res

            state_dict['RMcalibrated']['RM2'] = [RM2,RMerr2,upp,low]
            state_dict['RMcalibrated']['RMerrfit'] = RMerr2
            state_dict['RMcalibrated']['RMFWHM'] = upp-low

            state_dict['RMcalibrated']['RMsnrs2'] = np.load(dirs['logs'] + "RM_files/" + state_dict["RMcalibrated"]['dname_2D'] + "SNRs.npy")
            state_dict['RMcalibrated']['SNRs_full'] = np.load(dirs['logs'] + "RM_files/" + state_dict["RMcalibrated"]['dname_2D'] + "SNRs_full.npy")
            state_dict['RMcalibrated']['trial_RM2'] = np.load(dirs['logs'] + "RM_files/" + state_dict["RMcalibrated"]['dname_2D'] + "trialRM.npy")

            order = state_dict["RMcalibrated"]['trial_RM2'].argsort()
            state_dict["RMcalibrated"]['RMsnrs2'] = state_dict["RMcalibrated"]['RMsnrs2'][order]
            state_dict["RMcalibrated"]['trial_RM2'] = state_dict["RMcalibrated"]['trial_RM2'][order]
            state_dict["RMcalibrated"]['SNRs_full'] = state_dict["RMcalibrated"]['SNRs_full'][order,:]


            #update table
            RMdf.loc['All', '2D-Synth'] = RM2
            RMdf.loc['All', '2D-Synth Error'] = RMerr2

        if state_dict['n_comps'] > 1:
            for i in range(state_dict['n_comps']):
                if 'dname_1D' in state_dict['comps'][i]['RMcalibrated'].keys():
                    res = np.load(dirs['logs'] + "RM_files/" + state_dict['comps'][i]['RMcalibrated']['dname_1D'] + "result.npy")
                    RM = res[0]
                    RMerr = res[1]
                    state_dict['comps'][i]['RMcalibrated']["RMcalibrated"]["RM1"] = [RM,RMerr]
                    RMdf.loc[str(i), '1D-Synth'] = RM
                    RMdf.loc[str(i), '1D-Synth Error'] = RMerr

                    state_dict['comps'][i]["RMcalibrated"]['RMsnrs1'] = np.load(dirs['logs'] + "RM_files/" + state_dict['comps'][i]['RMcalibrated']['dname_1D'] + "SNRs.npy")
                    state_dict['comps'][i]["RMcalibrated"]['trial_RM1'] = np.load(dirs['logs'] + "RM_files/" + state_dict['comps'][i]['RMcalibrated']['dname_1D'] + "trialRM.npy")

                    order = state_dict['comps'][i]["RMcalibrated"]['trial_RM1'].argsort()
                    state_dict['comps'][i]["RMcalibrated"]['RMsnrs1'] = state_dict['comps'][i]["RMcalibrated"]['RMsnrs1'][order]
                    state_dict['comps'][i]["RMcalibrated"]['trial_RM1'] = state_dict['comps'][i]["RMcalibrated"]['trial_RM1'][order]

                if 'dname_1D_zoom' in state_dict['comps'][i]["RMcalibrated"].keys():
                    res = np.load(dirs['logs'] + "RM_files/" + state_dict['comps'][i]["RMcalibrated"]['dname_1D_zoom'] + "result.npy")
                    RM = res[0]
                    RMerr = res[1]
                    state_dict['comps'][i]["RMcalibrated"]["RM1zoom"] = [RM,RMerr]
                    RMdf.loc[str(i), '1D-Synth'] = RM
                    RMdf.loc[str(i), '1D-Synth Error'] = RMerr

                    state_dict['comps'][i]["RMcalibrated"]['RMsnrs1zoom'] = np.load(dirs['logs'] + "RM_files/" + state_dict['comps'][i]["RMcalibrated"]['dname_1D_zoom'] + "SNRs.npy")
                    state_dict['comps'][i]["RMcalibrated"]['trial_RM1zoom'] = np.load(dirs['logs'] + "RM_files/" + state_dict['comps'][i]["RMcalibrated"]['dname_1D_zoom'] + "trialRM.npy")

                    order = state_dict['comps'][i]["RMcalibrated"]['trial_RM1zoom'].argsort()
                    state_dict['comps'][i]["RMcalibrated"]['RMsnrs1zoom'] = state_dict['comps'][i]["RMcalibrated"]['RMsnrs1zoom'][order]
                    state_dict['comps'][i]["RMcalibrated"]['trial_RM1zoom'] = state_dict['comps'][i]["RMcalibrated"]['trial_RM1zoom'][order]

                    state_dict['comps'][i]["RMcalibrated"]['trial_RM2'] = copy.deepcopy(state_dict['comps'][i]["RMcalibrated"]['trial_RM1zoom'])
                    if np.all(np.isnan(state_dict['comps'][i]['RMcalibrated']['RMsnrs2'])): state_dict['comps'][i]['RMcalibrated']['RMsnrs2'] = np.nan*np.ones(len(state_dict['comps'][i]["RMcalibrated"]['trial_RM2']))




                if 'dname_2D' in state_dict['comps'][i]['RMcalibrated'].keys():
                    res = np.load(dirs['logs'] + "RM_files/" + state_dict['comps'][i]['RMcalibrated']['dname_2D'] + "result.npy")
                    RM2,RMerr2,upp,low = res

                    state_dict['comps'][i]['RMcalibrated']['RM2'] = [RM2,RMerr2,upp,low]
                    state_dict['comps'][i]['RMcalibrated']['RMerrfit'] = RMerr2
                    state_dict['comps'][i]['RMcalibrated']['RMFWHM'] = upp-low

                    state_dict['comps'][i]['RMcalibrated']['RMsnrs2'] = np.load(dirs['logs'] + "RM_files/" + state_dict['comps'][i]["RMcalibrated"]['dname_2D'] + "SNRs.npy")
                    state_dict['comps'][i]['RMcalibrated']['SNRs_full'] = np.load(dirs['logs'] + "RM_files/" + state_dict['comps'][i]["RMcalibrated"]['dname_2D'] + "SNRs_full.npy")
                    state_dict['comps'][i]['RMcalibrated']['trial_RM2'] = np.load(dirs['logs'] + "RM_files/" + state_dict['comps'][i]["RMcalibrated"]['dname_2D'] + "trialRM.npy")

                    order = state_dict['comps'][i]["RMcalibrated"]['trial_RM2'].argsort()
                    state_dict['comps'][i]["RMcalibrated"]['RMsnrs2'] = state_dict['comps'][i]["RMcalibrated"]['RMsnrs2'][order]
                    state_dict['comps'][i]["RMcalibrated"]['trial_RM2'] = state_dict['comps'][i]["RMcalibrated"]['trial_RM2'][order]
                    state_dict['comps'][i]["RMcalibrated"]['SNRs_full'] = state_dict['comps'][i]["RMcalibrated"]['SNRs_full'][order,:]





                    #update table
                    RMdf.loc[str(i), '2D-Synth'] = RM2
                    RMdf.loc[str(i), '2D-Synth Error'] = RMerr2

                    

    #1D plots

    if rmcomp_menu.value == 'All':
        dsapol.RM_summary_plot(state_dict['ids'],state_dict['nickname'],[state_dict['RMcalibrated']['RMsnrs1'],state_dict['RMcalibrated']['RM_tools_snrs']],[state_dict['RMcalibrated']['RMsnrs1zoom'],state_dict['RMcalibrated']['RM_tools_snrszoom'],state_dict['RMcalibrated']['RMsnrs2']],state_dict['RMcalibrated']["RM2"][0],state_dict["RMcalibrated"]["RMerrfit"],state_dict["RMcalibrated"]["trial_RM1"],state_dict["RMcalibrated"]["trial_RM2"],state_dict["RMcalibrated"]["trial_RM_tools"],state_dict["RMcalibrated"]["trial_RM_toolszoom"],threshold=9,suffix="_FORMAT_UPDATE_PARSEC",show=True,title='All Components',figsize=(38,24),datadir=state_dict['datadir'])
    
    elif rmcomp_menu.value != '':
        i= int(rmcomp_menu.value)
        dsapol.RM_summary_plot(state_dict['ids'],state_dict['nickname'],[state_dict['comps'][i]['RMcalibrated']['RMsnrs1'],state_dict['comps'][i]['RMcalibrated']['RM_tools_snrs']],[state_dict['comps'][i]['RMcalibrated']['RMsnrs1zoom'],state_dict['comps'][i]['RMcalibrated']['RM_tools_snrszoom'],state_dict['comps'][i]['RMcalibrated']['RMsnrs2']],state_dict['comps'][i]['RMcalibrated']["RM2"][0],state_dict['comps'][i]["RMcalibrated"]["RMerrfit"],state_dict['comps'][i]["RMcalibrated"]["trial_RM1"],state_dict['comps'][i]["RMcalibrated"]["trial_RM2"],state_dict['comps'][i]["RMcalibrated"]["trial_RM_tools"],state_dict['comps'][i]["RMcalibrated"]["trial_RM_toolszoom"],threshold=9,suffix="_FORMAT_UPDATE_PARSEC",show=True,title='Component ' + rmcomp_menu.value,figsize=(38,24),datadir=state_dict['datadir'])

    #update widget dict
    update_wdict([maxRM_num_tools,dRM_tools,nRM_num,minRM_num,maxRM_num,nRM_num_zoom,RM_window_zoom,dRM_tools_zoom,useRMTools,useRMsynth,useRM2D,rmcomp_menu,RMsynthbackground],
                ['maxRM_num_tools','dRM_tools','nRM_num','minRM_num','maxRM_num','nRM_num_zoom','RM_window_zoom','dRM_tools_zoom','useRMTools','useRMsynth','useRM2D','rmcomp_menu','RMsynthbackground'],param='value')
    
    update_wdict([RM_gal_display,RM_galerr_display,RM_ion_display,RM_ionerr_display],['RM_gal_display','RM_galerr_display','RM_ion_display','RM_ionerr_display'],param='data')
        
    
    
    
    return



def RM_screen_plot(rmcal_menu,RMcalibratebutton,RMdisplay,RMerrdisplay,rmcal_input):
    
    #update rm cal menu
    update_wdict([rmcal_menu],['rmcal_menu'],param='value')
    
    #if RMcalibrate is clicked, calibrate to peak RM
    if RMcalibratebutton.clicked and (rmcal_menu.value != "") and (rmcal_menu.value != "No RM Calibration"):
        
        #convert menu option to RM value in dict
        RM,err = RM_from_menu(rmcal_menu.value,rmcal_input)
        RMdisplay.data,RMerrdisplay.data = np.around(RM),np.around(err)
    
        #add to state dict
        state_dict['RMcalibrated']['RMcal'] = RM
        state_dict['RMcalibrated']['RMcalerr'] = err
        state_dict['RMcalibrated']['RMcalstring']= rmcal_menu.value
    
        #rm calibrate
        state_dict['IcalRM'],state_dict['QcalRM'],state_dict['UcalRM'],state_dict['VcalRM'] = dsapol.calibrate_RM(state_dict['Ical'],
                                                                                                        state_dict['Qcal'],
                                                                                                        state_dict['Ucal'],
                                                                                                        state_dict['Vcal'],
                                                                                                        RM,0,state_dict['freq_test'],
                                                                                                        stokes=True) #total derotation

        (state_dict['I_tcalRM'],state_dict['Q_tcalRM'],state_dict['U_tcalRM'],state_dict['V_tcalRM'],state_dict['I_tcalRM_err'],state_dict['Q_tcalRM_err'],state_dict['U_tcalRM_err'],state_dict['V_tcalRM_err']) = dsapol.get_stokes_vs_time(state_dict['IcalRM'],state_dict['QcalRM'],state_dict['UcalRM'],state_dict['VcalRM'],state_dict['width_native'],state_dict['tsamp'],state_dict['n_t'],n_off=int(NOFFDEF//state_dict['n_t']),plot=False,show=False,normalize=True,buff=state_dict['buff'],window=30,error=True,badchans=state_dict['badchans'])



    elif RMcalibratebutton.clicked and (rmcal_menu.value == "No RM Calibration"):

        
        #Don't calibrate

        state_dict['IcalRM'] = copy.deepcopy(state_dict['Ical'])
        state_dict['QcalRM'] = copy.deepcopy(state_dict['Qcal'])
        state_dict['UcalRM'] = copy.deepcopy(state_dict['Ucal'])
        state_dict['VcalRM'] = copy.deepcopy(state_dict['Vcal'])
            
        state_dict['I_tcalRM'] = copy.deepcopy(state_dict['I_tcal'])
        state_dict['Q_tcalRM'] = copy.deepcopy(state_dict['Q_tcal'])
        state_dict['U_tcalRM'] = copy.deepcopy(state_dict['U_tcal'])
        state_dict['V_tcalRM'] = copy.deepcopy(state_dict['V_tcal'])
        
        state_dict['RMcalibrated']['RMcal'] = np.nan
        state_dict['RMcalibrated']['RMcalerr'] = np.nan
        state_dict['RMcalibrated']['RMcalstring'] = "No RM Calibration"

        RMdisplay.data,RMerrdisplay.data = np.nan,np.nan

    #plot RM spectrum
    if ~np.isnan(state_dict['RMcalibrated']['RM2'][0]):

        show_calibrated = ~np.isnan(state_dict['RMcalibrated']['RMcal'])

        (state_dict['I_tcalRM'],state_dict['Q_tcalRM'],state_dict['U_tcalRM'],state_dict['V_tcalRM'],state_dict['I_tcalRM_err'],state_dict['Q_tcalRM_err'],state_dict['U_tcalRM_err'],state_dict['V_tcalRM_err']) = dsapol.get_stokes_vs_time(state_dict['IcalRM'],state_dict['QcalRM'],state_dict['UcalRM'],state_dict['VcalRM'],state_dict['width_native'],state_dict['tsamp'],state_dict['n_t'],n_off=int(NOFFDEF//state_dict['n_t']),plot=False,show=False,normalize=True,buff=state_dict['buff'],window=30,error=True,badchans=state_dict['badchans'])

        RMcal.plot_RM_2D(state_dict['I_tcal'],state_dict['Q_tcal'],state_dict['U_tcal'],state_dict['V_tcal'],int(NOFFDEF/state_dict['n_t']),state_dict['n_t'],state_dict['time_axis'],state_dict['timestart'],state_dict['timestop'],state_dict['RMcalibrated']['RM2'][0],state_dict['RMcalibrated']['RM2'][1],state_dict['RMcalibrated']['trial_RM2'],state_dict['RMcalibrated']['SNRs_full'],Qnoise=np.std(state_dict['Qcal'].mean(0)[:int(NOFFDEF/state_dict['n_t'])]),show_calibrated=show_calibrated,RMcal=state_dict['RMcalibrated']['RMcal'],RMcalerr=state_dict['RMcalibrated']['RMcalerr'],I_tcal_trm=state_dict['I_tcalRM'],Q_tcal_trm=state_dict['Q_tcalRM'],U_tcal_trm=state_dict['U_tcalRM'],V_tcal_trm=state_dict['V_tcalRM'],wind=state_dict['window']*32.7*state_dict['n_t']*1e-3)#,rmbuff=500,cmapname='viridis',wind=5)

    update_wdict([RMdisplay,RMerrdisplay],['RMdisplay','RMerrdisplay'],param='data')
    update_wdict([rmcal_menu,rmcal_input],['rmcal_menu','rmcal_input'],param='value')    
    return


#sub-screen to query SIMBAD catalog for intervening galaxies
def galquery_Budget_screen(catalog_selection,galtype_selection,queryradius,runquery,limitzrange,zrange,trialz,cosmo_selection):

    #run query
    if runquery.clicked:
        state_dict['qdat'] = budget.get_SIMBAD_gals(state_dict['RA'],state_dict['DEC'],queryradius.value,
                                                    catalogs=[] if 'All' in catalog_selection.value else catalog_selection.value,
                                                    types=[] if 'All' in galtype_selection.value else galtype_selection.value,cosmology=cosmo_selection.value,
                                                    redshift=trialz.value,redshift_range=zrange.value if limitzrange.value else None)


        #update galaxy options
        if state_dict['qdat'] is not None:
            wdict['galaxy_selection_choices'] = list(state_dict['qdat']['MAIN_ID'])
            wdict['galaxy_selection'] = []
        

    #plotting
    #budget.plot_galaxies(state_dict['RA'],state_dict['DEC'],queryradius.value,cosmology=cosmo_selection.value,redshift=trialz.value,qdat=state_dict['qdat'])
    
    update_wdict([catalog_selection,galtype_selection,queryradius,limitzrange,zrange,trialz,cosmo_selection],
            ['catalog_selection','galtype_selection','queryradius','limitzrange','zrange','trialz','cosmo_selection'],param='value')

    return state_dict['qdat']


def galplot_Budget_screen(catalog_selection,galtype_selection,queryradius,runquery,limitzrange,zrange,trialz,cosmo_selection,galaxy_masses,galaxy_selection,mass_type,galaxy_bfields):

    #get DM estimates for interveners
    state_dict['intervener_DMs'] = []
    state_dict['intervener_DM_errs'] = []
    state_dict['intervener_zs'] = []
    state_dict['intervener_names_DM'] = []
    if state_dict['qdat'] is not None and len(galaxy_selection.value)>0 and len(galaxy_masses.value)>0:
        galaxy_masses_list = np.array(galaxy_masses.value.split(','),dtype=str)
        for i in range(len(galaxy_masses_list)):
            if galaxy_masses_list[i] == '' or galaxy_masses_list[i] == ' ': continue
            if '--' in galaxy_masses_list[i]:
                mass_low,mass_high = np.array(galaxy_masses_list[i].split('--'),dtype=float)
            else:
                mass_low = 0.9* float(galaxy_masses_list[i])
                mass_high = 1.1*float(galaxy_masses_list[i])
            int_DM,int_DM_err,int_rvir,int_rvir_err = budget.DM_int_vals(state_dict['qdat'][list(state_dict['qdat']['MAIN_ID']).index(galaxy_selection.value[i])],
                                                        mass_low,mass_high,cosmo_selection.value,mass_type.value)
            state_dict['intervener_DMs'].append(int_DM)
            state_dict['intervener_DM_errs'].append(int_DM_err)
            state_dict['intervener_zs'].append(state_dict['qdat']['RVZ_RADVEL'][list(state_dict['qdat']['MAIN_ID']).index(galaxy_selection.value[i])])
            state_dict['intervener_names_DM'].append(state_dict['qdat']['MAIN_ID'][list(state_dict['qdat']['MAIN_ID']).index(galaxy_selection.value[i])])
            state_dict['qdat']['M_INPUT'][list(state_dict['qdat']['MAIN_ID']).index(galaxy_selection.value[i])] = (mass_low+mass_high)/2
            state_dict['qdat']['M_INPUT_ERROR'][list(state_dict['qdat']['MAIN_ID']).index(galaxy_selection.value[i])] = (mass_high-mass_low)/2 
            state_dict['qdat']['R_EST'][list(state_dict['qdat']['MAIN_ID']).index(galaxy_selection.value[i])] = int_rvir
            state_dict['qdat']['R_EST_ERROR'][list(state_dict['qdat']['MAIN_ID']).index(galaxy_selection.value[i])] = int_rvir_err
            state_dict['qdat']['R_ANGLE_EST'][list(state_dict['qdat']['MAIN_ID']).index(galaxy_selection.value[i])] = (int_rvir/state_dict['qdat']['COMOVING_DIST_EST'][i])*(180*60/np.pi)
            state_dict['qdat']['R_ANGLE_EST_ERROR'][list(state_dict['qdat']['MAIN_ID']).index(galaxy_selection.value[i])] = (int_rvir_err/state_dict['qdat']['COMOVING_DIST_EST'][i])*(180*60/np.pi)
            state_dict['qdat']['DM_EST'][list(state_dict['qdat']['MAIN_ID']).index(galaxy_selection.value[i])] = int_DM
            state_dict['qdat']['DM_EST_ERROR'][list(state_dict['qdat']['MAIN_ID']).index(galaxy_selection.value[i])] = int_DM_err
    state_dict['intervener_RMs'] = []
    state_dict['intervener_RM_errs'] = []
    state_dict['intervener_RMzs'] = []
    state_dict['intervener_names_RM'] = []
    if state_dict['qdat'] is not None and len(galaxy_selection.value)>0 and len(galaxy_bfields.value)>0 and len(galaxy_masses.value)>0:
        galaxy_bfields_list = np.array(galaxy_bfields.value.split(','),dtype=str)
        for i in range(len(galaxy_bfields_list)):
            bfield_types = []
            if galaxy_bfields_list[i] == '' or galaxy_bfields_list[i] == ' ': continue
            if '--' in galaxy_bfields_list[i]:
                tmp = galaxy_bfields_list[i].split('--')
                if 'x' not in tmp[0] and 'uG' not in tmp[0] and 'x' not in tmp[1] and 'uG' not in tmp[1]: continue


                if 'uG' in tmp[0]:
                    bfield_low = float(tmp[0][:tmp[0].index('uG')])
                    bfield_types.append('uG')
                else:
                    bfield_low = float(tmp[0][:tmp[0].index('x')]) #budget.MWlike_Bfield(float(tmp[0][:tmp[0].index('x')]),state_dict['qdat']['IMPACT'][i])
                    bfield_types.append('x')

                if 'uG' in tmp[0]:
                    bfield_high = float(tmp[1][:tmp[1].index('uG')])
                    bfield_types.append('uG')
                else:
                    bfield_high = float(tmp[1][:tmp[1].index('x')]) #budget.MWlike_Bfield(float(tmp[1][:tmp[1].index('x')]),state_dict['qdat']['IMPACT'][i])
                    bfield_types.append('x')
            else:
                if 'uG' in galaxy_bfields_list[i]:
                    bfield_low = 0.9*float(galaxy_bfields_list[i][:galaxy_bfields_list[i].index('uG')])
                    bfield_high = 1.1*float(galaxy_bfields_list[i][:galaxy_bfields_list[i].index('uG')])
                    bfield_types = ['uG','uG']
                else:
                    bfield_low = 0.9*float(galaxy_bfields_list[i][:galaxy_bfields_list[i].index('x')]) #budget.MWlike_Bfield(float(galaxy_bfields_list[i][:galaxy_bfields_list[i].index('x')]),state_dict['qdat']['IMPACT'][i])
                    bfield_high = 1.1*float(galaxy_bfields_list[i][:galaxy_bfields_list[i].index('x')]) #budget.MWlike_Bfield(float(galaxy_bfields_list[i][:galaxy_bfields_list[i].index('x')]),state_dict['qdat']['IMPACT'][i])
                    bfield_types = ['x','x']

            int_RM,int_RM_err,int_B,int_B_err,int_B0,int_B0_err = budget.RM_int_vals(state_dict['qdat'][list(state_dict['qdat']['MAIN_ID']).index(galaxy_selection.value[i])],bfield_low,bfield_high,cosmo_selection.value,mass_type.value,bfield_types)
            state_dict['intervener_RMs'].append(int_RM)
            state_dict['intervener_RM_errs'].append(int_RM_err)
            state_dict['intervener_RMzs'].append(state_dict['qdat']['RVZ_RADVEL'][list(state_dict['qdat']['MAIN_ID']).index(galaxy_selection.value[i])])
            state_dict['intervener_names_RM'].append(state_dict['qdat']['MAIN_ID'][list(state_dict['qdat']['MAIN_ID']).index(galaxy_selection.value[i])])
            state_dict['qdat']['RM_EST'][list(state_dict['qdat']['MAIN_ID']).index(galaxy_selection.value[i])] = int_RM
            state_dict['qdat']['RM_EST_ERROR'][list(state_dict['qdat']['MAIN_ID']).index(galaxy_selection.value[i])] = int_RM_err

            state_dict['qdat']['B0_LOS_EST'][list(state_dict['qdat']['MAIN_ID']).index(galaxy_selection.value[i])] = int_B0
            state_dict['qdat']['B0_LOS_EST_ERROR'][list(state_dict['qdat']['MAIN_ID']).index(galaxy_selection.value[i])] = int_B0_err

            state_dict['qdat']['B_LOS_EST'][list(state_dict['qdat']['MAIN_ID']).index(galaxy_selection.value[i])] = int_B
            state_dict['qdat']['B_LOS_EST_ERROR'][list(state_dict['qdat']['MAIN_ID']).index(galaxy_selection.value[i])] = int_B_err


    #plotting
    budget.plot_galaxies(state_dict['RA'],state_dict['DEC'],queryradius.value,cosmology=cosmo_selection.value,redshift=trialz.value,qdat=state_dict['qdat'],save=True,savedir=state_dict['datadir'])

    update_wdict([catalog_selection,galtype_selection,queryradius,limitzrange,zrange,trialz,cosmo_selection,galaxy_masses,galaxy_selection,mass_type],
            ['catalog_selection','galtype_selection','queryradius','limitzrange','zrange','trialz','cosmo_selection','galaxy_masses','galaxy_selection','mass_type'],param='value')
    return state_dict['qdat']


#sub-screen to compute RM, DM, B field budget
def DM_Budget_screen(trialz):
        
    #if redshift is known, compute DM host
    if ~np.isnan(state_dict['z']) and ~np.isnan(state_dict['DM']) and trialz.value < 0:
        ctest = SkyCoord(ra=state_dict['RA']*u.deg,dec=state_dict['DEC']*u.deg,frame='icrs') #SkyCoord('22h34m46.93s+70d32m18.40s',frame='icrs',unit=(u.hourangle,u.deg))
        state_dict['DMhost'],state_dict['DMhost_lower_limit'],state_dict['DMhost_upper_limit'],state_dict["dmbudgetdict"] = budget.DM_host_limits(state_dict['DM'],state_dict['z'],ctest.galactic.l.value,ctest.galactic.b.value,plot=False,intervener_DMs=state_dict['intervener_DMs'],intervener_DM_errs=state_dict['intervener_DM_errs'],intervener_zs=state_dict['intervener_zs'])
        state_dict['dmdist'],state_dict['DMaxis'] = budget.DM_host_dist(state_dict['DM'],state_dict['z'],ctest.galactic.l.value,ctest.galactic.b.value,plot=True,intervener_DMs=state_dict['intervener_DMs'],intervener_DM_errs=state_dict['intervener_DM_errs'],intervener_zs=state_dict['intervener_zs'],save=True,savedir=state_dict['datadir'])
    elif ~np.isnan(state_dict['DM']) and trialz.value >= 0:
        ctest = SkyCoord(ra=state_dict['RA']*u.deg,dec=state_dict['DEC']*u.deg,frame='icrs') #SkyCoord('22h34m46.93s+70d32m18.40s',frame='icrs',unit=(u.hourangle,u.deg))
        state_dict['DMhost'],state_dict['DMhost_lower_limit'],state_dict['DMhost_upper_limit'],state_dict["dmbudgetdict"] = budget.DM_host_limits(state_dict['DM'],trialz.value,ctest.galactic.l.value,ctest.galactic.b.value,plot=False,intervener_DMs=state_dict['intervener_DMs'],intervener_DM_errs=state_dict['intervener_DM_errs'],intervener_zs=state_dict['intervener_zs'])
        state_dict['dmdist'],state_dict['DMaxis'] = budget.DM_host_dist(state_dict['DM'],trialz.value,ctest.galactic.l.value,ctest.galactic.b.value,plot=True,intervener_DMs=state_dict['intervener_DMs'],intervener_DM_errs=state_dict['intervener_DM_errs'],intervener_zs=state_dict['intervener_zs'],save=True,savedir=state_dict['datadir'])
    
    update_wdict([trialz],
            ['trialz'],param='value')


    #update table
    df_DM_budget.loc['Budget'] = [r'${a:.2f}\pm{b:.2f}$'.format(a=np.around(state_dict['DM'],2),b=0.1),
                                  r'${a:.2f}\pm{b:.2f}$'.format(a=np.around(state_dict['dmbudgetdict']['MW'],2),b=np.around(state_dict['dmbudgetdict']['MWerr'],2)),
                                  r'${a:.2f}\pm{b:.2f}$'.format(a=np.around(state_dict['dmbudgetdict']['halo'],2),b=np.around(state_dict['dmbudgetdict']['haloerr'],2)),
                                  r'${a:.2f}^{{+{b:.2f}}}_{{-{c:.2f}}}$'.format(a=np.around(state_dict['dmbudgetdict']["IGM"],2),b=np.around(state_dict['dmbudgetdict']["IGMupperr"],2),c=np.around(state_dict['dmbudgetdict']["IGMlowerr"],2)),
                                  "\n".join([r'{NAME}:${a:.2f}\pm{b:.2f}$'.format(NAME=state_dict['intervener_names_DM'][n],a=np.around(state_dict['intervener_DMs'][n],2),b=np.around(state_dict['intervener_DM_errs'][n] if np.isnan(state_dict['intervener_DM_errs'][n]) else 0.4,2)) for n in range(len(state_dict['intervener_names_DM']))]),
                                  r'${a:.2f}^{{+{b:.2f}}}_{{-{c:.2f}}}$'.format(a=np.around(state_dict['DMhost'],2),b=np.around((state_dict['DMhost_upper_limit']-state_dict['DMhost']),2),c=np.around((state_dict['DMhost']-state_dict['DMhost_lower_limit']),2)),
                                  r'${a:.2f}^{{+{b:.2f}}}_{{-{c:.2f}}}$'.format(a=np.around(state_dict['DMhost']/(1+trialz.value),2),b=np.around((state_dict['DMhost_upper_limit']-state_dict['DMhost'])/(1+trialz.value),2),c=np.around((state_dict['DMhost']-state_dict['DMhost_lower_limit'])/(1+trialz.value),2))
                                 ]

    return df_DM_budget
           


def RM_Budget_screen(trialz):

    #if redshift is known, compute RM host
    if ~np.isnan(state_dict['z']) and ~np.isnan(state_dict['RMcalibrated']['RMcal']) and ~np.isnan(state_dict['RM_gal']) and ~np.isnan(state_dict['RM_ion']) and trialz.value < 0:
        state_dict['RMhost'],state_dict['RMhost_lower_limit'],state_dict['RMhost_upper_limit'] = budget.RM_host_limits(RMobs=state_dict['RMcalibrated']['RMcal'],RMobserr=state_dict['RMcalibrated']['RMcalerr'],
                   RMmw=state_dict['RM_gal'],RMmwerr=state_dict['RM_galerr'],
                   RMion=state_dict['RM_ion'],RMionerr=state_dict['RM_ionerr'],ztest=state_dict['z'],plot=False,intervener_RMs=state_dict['intervener_RMs'],intervener_RM_errs=state_dict['intervener_RM_errs'],intervener_zs=state_dict['intervener_RMzs'])
        state_dict['rmdist'],state_dict['RMaxis'] = budget.RM_host_dist(RMobs=state_dict['RMcalibrated']['RMcal'],RMobserr=state_dict['RMcalibrated']['RMcalerr'],
                   RMmw=state_dict['RM_gal'],RMmwerr=state_dict['RM_galerr'],
                   RMion=state_dict['RM_ion'],RMionerr=state_dict['RM_ionerr'],ztest=state_dict['z'],plot=True,intervener_RMs=state_dict['intervener_RMs'],intervener_RM_errs=state_dict['intervener_RM_errs'],intervener_zs=state_dict['intervener_RMzs'],save=True,savedir=state_dict['datadir'])
        rmobs = state_dict['RMcalibrated']['RMcal']
        rmobserr = state_dict['RMcalibrated']['RMcalerr']
    elif ~np.isnan(state_dict['z']) and np.isnan(state_dict['RMcalibrated']['RMcal']) and ~np.isnan(state_dict['RMinput']) and ~np.isnan(state_dict['RM_gal']) and ~np.isnan(state_dict['RM_ion']) and trialz.value < 0:
        state_dict['RMhost'],state_dict['RMhost_lower_limit'],state_dict['RMhost_upper_limit'] = budget.RM_host_limits(RMobs=state_dict['RMinput'],RMobserr=state_dict['RMinputerr'],
                   RMmw=state_dict['RM_gal'],RMmwerr=state_dict['RM_galerr'],
                   RMion=state_dict['RM_ion'],RMionerr=state_dict['RM_ionerr'],ztest=state_dict['z'],plot=False,intervener_RMs=state_dict['intervener_RMs'],intervener_RM_errs=state_dict['intervener_RM_errs'],intervener_zs=state_dict['intervener_RMzs'])
        state_dict['rmdist'],state_dict['RMaxis']  = budget.RM_host_dist(RMobs=state_dict['RMinput'],RMobserr=state_dict['RMinputerr'],
                   RMmw=state_dict['RM_gal'],RMmwerr=state_dict['RM_galerr'],
                   RMion=state_dict['RM_ion'],RMionerr=state_dict['RM_ionerr'],ztest=state_dict['z'],plot=True,intervener_RMs=state_dict['intervener_RMs'],intervener_RM_errs=state_dict['intervener_RM_errs'],intervener_zs=state_dict['intervener_RMzs'],save=True,savedir=state_dict['datadir'])
        rmobs = state_dict['RMinput']
        rmobserr = state_dict['RMinputerr'] 
    elif ~np.isnan(state_dict['RMcalibrated']['RMcal']) and ~np.isnan(state_dict['RM_gal']) and ~np.isnan(state_dict['RM_ion']) and trialz.value >= 0:
        state_dict['RMhost'],state_dict['RMhost_lower_limit'],state_dict['RMhost_upper_limit'] = budget.RM_host_limits(RMobs=state_dict['RMcalibrated']['RMcal'],RMobserr=state_dict['RMcalibrated']['RMcalerr'],
                   RMmw=state_dict['RM_gal'],RMmwerr=state_dict['RM_galerr'],
                   RMion=state_dict['RM_ion'],RMionerr=state_dict['RM_ionerr'],ztest=trialz.value,plot=False,intervener_RMs=state_dict['intervener_RMs'],intervener_RM_errs=state_dict['intervener_RM_errs'],intervener_zs=state_dict['intervener_RMzs'])
        state_dict['rmdist'],state_dict['RMaxis']  = budget.RM_host_dist(RMobs=state_dict['RMcalibrated']['RMcal'],RMobserr=state_dict['RMcalibrated']['RMcalerr'],
                   RMmw=state_dict['RM_gal'],RMmwerr=state_dict['RM_galerr'],
                   RMion=state_dict['RM_ion'],RMionerr=state_dict['RM_ionerr'],ztest=trialz.value,plot=True,intervener_RMs=state_dict['intervener_RMs'],intervener_RM_errs=state_dict['intervener_RM_errs'],intervener_zs=state_dict['intervener_RMzs'],save=True,savedir=state_dict['datadir'])
        rmobs = state_dict['RMcalibrated']['RMcal']
        rmobserr = state_dict['RMcalibrated']['RMcalerr']
    elif np.isnan(state_dict['RMcalibrated']['RMcal']) and ~np.isnan(state_dict['RMinput']) and ~np.isnan(state_dict['RM_gal']) and ~np.isnan(state_dict['RM_ion']) and trialz.value >= 0:
        state_dict['RMhost'],state_dict['RMhost_lower_limit'],state_dict['RMhost_upper_limit'] = budget.RM_host_limits(RMobs=state_dict['RMinput'],RMobserr=state_dict['RMinputerr'],
                   RMmw=state_dict['RM_gal'],RMmwerr=state_dict['RM_galerr'],
                   RMion=state_dict['RM_ion'],RMionerr=state_dict['RM_ionerr'],ztest=trialz.value,plot=False,intervener_RMs=state_dict['intervener_RMs'],intervener_RM_errs=state_dict['intervener_RM_errs'],intervener_zs=state_dict['intervener_RMzs'])
        state_dict['rmdist'],state_dict['RMaxis'] = budget.RM_host_dist(RMobs=state_dict['RMinput'],RMobserr=state_dict['RMinputerr'],
                   RMmw=state_dict['RM_gal'],RMmwerr=state_dict['RM_galerr'],
                   RMion=state_dict['RM_ion'],RMionerr=state_dict['RM_ionerr'],ztest=trialz.value,plot=True,intervener_RMs=state_dict['intervener_RMs'],intervener_RM_errs=state_dict['intervener_RM_errs'],intervener_zs=state_dict['intervener_RMzs'],save=True,savedir=state_dict['datadir'])
        rmobs = state_dict['RMinput']
        rmobserr = state_dict['RMinputerr']

    update_wdict([trialz],['trialz'],param='value')


    #update table
    df_RM_budget.loc['Budget'] = [r'${a:.2f}\pm{b:.2f}$'.format(a=np.around(rmobs,2),b=np.around(rmobserr,2)),
                                  r'${a:.2f}\pm{b:.2f}$'.format(a=np.around(state_dict['RM_gal'],2),b=np.around(state_dict['RM_galerr'],2)),
                                  r'${a:.2f}\pm{b:.2f}$'.format(a=np.around(state_dict['RM_ion'],2),b=np.around(state_dict['RM_ionerr'],2)),
                                  "\n".join([r'{NAME}:${a:.2f}\pm{b:.2f}$'.format(NAME=state_dict['intervener_names_RM'][n],a=np.around(state_dict['intervener_RMs'][n],2),b=np.around(state_dict['intervener_RM_errs'][n] if np.isnan(state_dict['intervener_RM_errs'][n]) else 1,2)) for n in range(len(state_dict['intervener_names_RM']))]),
                                  r'${a:.2f}\pm{b:.2f}$'.format(a=np.around(state_dict['RMhost'],2),b=np.around(state_dict['RMhost_upper_limit']-state_dict['RMhost'],2)),
                                  r'${a:.2f}\pm{b:.2f}$'.format(a=np.around(state_dict['RMhost']/((1+trialz.value)**2),2),b=np.around((state_dict['RMhost_upper_limit']-state_dict['RMhost'])/((1+trialz.value)**2),2))
                                 ]

    return df_RM_budget




def Bfield_Budget_screen(getBfieldbutton,Bfield_range,Bfield_res):

    #if redshfit, DM, RM are known, compute RM host
    if getBfieldbutton.clicked:
        if ~np.isnan(state_dict['DMhost']) and ~np.isnan(state_dict['RMhost']):
            state_dict['Bdist'],state_dict['Bhost'],state_dict['Bhost_lower_limit'],state_dict['Bhost_upper_limit'],state_dict['Baxis'] = budget.Bhost_dist(DMhost=state_dict['DMhost'],
                                                                                                                                                    dmdist=state_dict['dmdist'],DMaxis=state_dict['DMaxis'],
                                                                                                                                                    RMhost=state_dict['RMhost'],
                                                                                                                                                    RMhosterr=(state_dict['RMhost_upper_limit']
                                                                                                                                                            -state_dict['RMhost']),
                                                                                                                                                    res2=int(Bfield_range.value//Bfield_res.value),plot=True,buff=Bfield_range.value,save=True,savedir=state_dict['datadir'])
            wdict['Bfield_display'] = np.around(state_dict['Bhost'],2)
            wdict['Bfield_pos_err_display'] = np.around(state_dict['Bhost_upper_limit']-state_dict['Bhost'],2)
            wdict['Bfield_neg_err_display'] = np.around(state_dict['Bhost']-state_dict['Bhost_lower_limit'],2)

    update_wdict([Bfield_range,Bfield_res],['Bfield_range','Bfield_res'],param='value')
    #update_wdict([Bfield_display,Bfield_pos_err_display,Bfield_neg_err_display],
    #        ["Bfield_display","Bfield_pos_err_display","Bfield_neg_err_display"],param='data')
    return

def polanalysis_screen(showghostPA,intLbuffer_slider,intRbuffer_slider,polcomp_menu):

    #check if RM calibrated
    if state_dict['RMcalibrated']['RMcalstring'] != "No RM Calibration":
        I_use = copy.deepcopy(state_dict['IcalRM'])
        Q_use = copy.deepcopy(state_dict['QcalRM'])
        U_use = copy.deepcopy(state_dict['UcalRM'])
        V_use = copy.deepcopy(state_dict['VcalRM'])

        I_tuse = state_dict['I_tcalRM']
        Q_tuse = state_dict['Q_tcalRM']
        U_tuse = state_dict['U_tcalRM']
        V_tuse = state_dict['V_tcalRM']

    elif "Ical" in state_dict.keys():
        I_use = copy.deepcopy(state_dict['Ical'])
        Q_use = copy.deepcopy(state_dict['Qcal'])
        U_use = copy.deepcopy(state_dict['Ucal'])
        V_use = copy.deepcopy(state_dict['Vcal'])

        I_tuse = state_dict['I_tcal']
        Q_tuse = state_dict['Q_tcal']
        U_tuse = state_dict['U_tcal']
        V_tuse = state_dict['V_tcal']

    elif 'I' in state_dict.keys(): 
        I_use = copy.deepcopy(state_dict['I'])
        Q_use = copy.deepcopy(state_dict['Q'])
        U_use = copy.deepcopy(state_dict['U'])
        V_use = copy.deepcopy(state_dict['V'])

        I_tuse = state_dict['I_t']
        Q_tuse = state_dict['Q_t']
        U_tuse = state_dict['U_t']
        V_tuse = state_dict['V_t']
    else:
        raise KeyError


    #get intL and intR from slider
    n_off = int(NOFFDEF/state_dict['n_t'])
    if 'intL' not in state_dict.keys():
        FWHM,heights,intL,intR = peak_widths(I_tuse,[np.argmax(I_tuse)])
        state_dict['intL'] = int(intL) #- intLbuffer_slider.value
        state_dict['intR'] = int(intR) #+ intRbuffer_slider.value

    if polcomp_menu.value == "All":
        state_dict['intLbuffer'] = intLbuffer_slider.value
        state_dict['intRbuffer'] = intRbuffer_slider.value
    

    #compute PA
    sigflag = state_dict['RMcalibrated']['RMcalstring'] != "No RM Calibration"
    weighted = 'weights' in state_dict.keys()
    if not weighted:
        weights_use = np.nan*np.ones(len(I_tuse))
    else:
        weights_use = state_dict['weights']
    if state_dict['n_comps'] > 1:
        for i in range(state_dict['n_comps']):
            
            #get weights
            weighted = 'weights' in state_dict['comps'][i].keys()
            if not weighted:
                weights_use = np.nan*np.ones(len(I_tuse))
            else:
                weights_use = state_dict['comps'][i]['weights']

            #check if buffers are for this component
            if polcomp_menu.value != "All" and int(polcomp_menu.value[-1]) == i:
                state_dict['comps'][i]['intLbuffer'] = intLbuffer_slider.value
                state_dict['comps'][i]['intRbuffer'] = intRbuffer_slider.value

            #get PA for subcomponent
            (PA_f,
             PA_t,
             PA_f_errs,
             PA_t_errs,
             state_dict['comps'][i]['avg_PA'],
             state_dict['comps'][i]['PA_err']) = dsapol.get_pol_angle(I_use,Q_use,U_use,V_use,state_dict['comps'][i]['width_native'],state_dict['tsamp'],
                                                                    state_dict['n_t'],state_dict['n_f'],state_dict['freq_test'],n_off=int(NOFFDEF/state_dict['n_t']),
                                                                    normalize=True,buff=state_dict['comps'][i]['buff'],weighted=weighted,input_weights=weights_use,
                                                                    fobj=FilReader(state_dict['datadir']+state_dict['ids'] + state_dict['suff'] + "_0.fil"),
                                                                    intL=state_dict['comps'][i]['intL']-state_dict['comps'][i]['intLbuffer'],
                                                                    intR=state_dict['comps'][i]['intR']+state_dict['comps'][i]['intRbuffer'])


            #get pol fractions for subcomponent
            [(pol_f,pol_t,avg,sigma_frac,snr_frac),
            (L_f,L_t,avg_L,sigma_L,snr_L),
            (C_f_unbiased,C_t_unbiased,avg_C_abs,sigma_C_abs,snr_C),
            (C_f,C_t,avg_C,sigma_C,snrC),snr]= dsapol.get_pol_fraction(I_use,Q_use,U_use,V_use,state_dict['comps'][i]['width_native'],state_dict['tsamp'],
                                                                    state_dict['n_t'],state_dict['n_f'],state_dict['freq_test'],
                                                                    n_off=int(NOFFDEF/state_dict['n_t']),plot=False,normalize=True,
                                                                    buff=state_dict['comps'][i]['buff'],full=False,weighted=weighted,input_weights=weights_use,
                                                                    fobj=FilReader(state_dict['datadir']+state_dict['ids'] + state_dict['suff'] + "_0.fil"),
                                                                    intL=state_dict['comps'][i]['intL']-state_dict['comps'][i]['intLbuffer'],
                                                                    intR=state_dict['comps'][i]['intR']+state_dict['comps'][i]['intRbuffer'])

            state_dict['comps'][i]['Tpol'] = avg
            state_dict['comps'][i]['Tpol_err'] = sigma_frac
            state_dict['comps'][i]['Tsnr'] = snr_frac
            state_dict['comps'][i]['snr'] = snr
            state_dict['comps'][i]['Lpol'] = avg_L
            state_dict['comps'][i]['Lpol_err'] = sigma_L
            state_dict['comps'][i]['Lsnr'] = snr_L
            state_dict['comps'][i]['absVpol'] = avg_C_abs
            state_dict['comps'][i]['absVpol_err'] = sigma_C_abs
            state_dict['comps'][i]['Vpol'] = avg_C
            state_dict['comps'][i]['Vpol_err'] = sigma_C
            state_dict['comps'][i]['Vsnr'] = snr_C


            #classify
            if np.abs(state_dict['comps'][i]['Tpol']/state_dict['comps'][i]['Tpol_err']) < 3 and np.abs(1-state_dict['comps'][i]['Tpol'])/state_dict['comps'][i]['Tpol_err'] > 3:
                state_dict['comps'][i]['Tclass'] = "consistent with 0%"
            elif np.abs(state_dict['comps'][i]['Tpol']/state_dict['comps'][i]['Tpol_err']) > 3 and np.abs(1-state_dict['comps'][i]['Tpol'])/state_dict['comps'][i]['Tpol_err'] > 3:
                state_dict['comps'][i]['Tclass'] = "intermediate"
            elif np.abs(state_dict['comps'][i]['Tpol']/state_dict['comps'][i]['Tpol_err']) > 3 and np.abs(1-state_dict['comps'][i]['Tpol'])/state_dict['comps'][i]['Tpol_err'] < 3:
                state_dict['comps'][i]['Tclass'] = "consistent with 100%"
            else:
                state_dict['comps'][i]['Tclass'] = "unconstrained"

            if np.abs(state_dict['comps'][i]['Lpol']/state_dict['comps'][i]['Lpol_err']) < 3 and np.abs(1-state_dict['comps'][i]['Lpol'])/state_dict['comps'][i]['Lpol_err'] > 3:
                state_dict['comps'][i]['Lclass'] = "consistent with 0%"
            elif np.abs(state_dict['comps'][i]['Lpol']/state_dict['comps'][i]['Lpol_err']) > 3 and np.abs(1-state_dict['comps'][i]['Lpol'])/state_dict['comps'][i]['Lpol_err'] > 3:
                state_dict['comps'][i]['Lclass'] = "intermediate"
            elif np.abs(state_dict['comps'][i]['Lpol']/state_dict['comps'][i]['Lpol_err']) > 3 and np.abs(1-state_dict['comps'][i]['Lpol'])/state_dict['comps'][i]['Lpol_err'] < 3:
                state_dict['comps'][i]['Lclass'] = "consistent with 100%"
            else:
                state_dict['comps'][i]['Lclass'] = "unconstrained"

            if np.abs(state_dict['comps'][i]['absVpol']/state_dict['comps'][i]['absVpol_err']) < 3 and np.abs(1-state_dict['comps'][i]['absVpol'])/state_dict['comps'][i]['absVpol_err'] > 3:
                state_dict['comps'][i]['Vclass'] = "consistent with 0%"
            elif np.abs(state_dict['comps'][i]['absVpol']/state_dict['comps'][i]['absVpol_err']) > 3 and np.abs(1-state_dict['comps'][i]['absVpol'])/state_dict['comps'][i]['absVpol_err'] > 3:
                state_dict['comps'][i]['Vclass'] = "intermediate"
            elif np.abs(state_dict['comps'][i]['absVpol']/state_dict['comps'][i]['absVpol_err']) > 3 and np.abs(1-state_dict['comps'][i]['absVpol'])/state_dict['comps'][i]['absVpol_err'] < 3:
                state_dict['comps'][i]['Vclass'] = "consistent with 100%"
            else:
                state_dict['comps'][i]['Vclass'] = "unconstrained"

            #upate dataframe
            poldf.loc['Component ' + str(i)] = [np.around(state_dict['comps'][i]['snr'],2),
                        np.around(100*state_dict['comps'][i]['Tpol'],2),
                        np.around(100*state_dict['comps'][i]['Tpol_err'],2),
                        np.around(state_dict['comps'][i]['Tsnr'],2),
                        state_dict['comps'][i]['Tclass'],
                        np.around(100*state_dict['comps'][i]['Lpol'],2),
                        np.around(100*state_dict['comps'][i]['Lpol_err'],2),
                        np.around(state_dict['comps'][i]['Lsnr'],2),
                        state_dict['comps'][i]['Lclass'],
                        np.around(100*state_dict['comps'][i]['Vpol'],2),
                        np.around(100*state_dict['comps'][i]['Vpol_err'],2),
                        np.around(100*state_dict['comps'][i]['absVpol'],2),
                        np.around(100*state_dict['comps'][i]['absVpol_err'],2),
                        np.around(state_dict['comps'][i]['Vsnr'],2),
                        state_dict['comps'][i]['Vclass'],
                        np.around((180/np.pi)*state_dict['comps'][i]['avg_PA'],2),
                        np.around((180/np.pi)*state_dict['comps'][i]['PA_err'],2)]
        
    
    #compute PA for full burst
    state_dict['PA_f'],state_dict['PA_t'],state_dict['PA_f_errs'],state_dict['PA_t_errs'],state_dict['avg_PA'],state_dict['PA_err'] = dsapol.get_pol_angle(I_use,Q_use,U_use,V_use,state_dict['width_native'],state_dict['tsamp'],state_dict['n_t'],state_dict['n_f'],state_dict['freq_test'],n_off=int(NOFFDEF/state_dict['n_t']),normalize=True,buff=state_dict['buff'],weighted=weighted,input_weights=weights_use,fobj=FilReader(state_dict['datadir']+state_dict['ids'] + state_dict['suff'] + "_0.fil"),intL=state_dict['intL']-state_dict['intLbuffer'],intR=state_dict['intR']+state_dict['intRbuffer'])



    #compute frequency spectrum
    (I_fuse,Q_fuse,U_fuse,V_fuse) = dsapol.get_stokes_vs_freq(I_use,Q_use,U_use,V_use,state_dict['width_native'],state_dict['tsamp'],
                                                        state_dict['n_f'],state_dict['n_t'],state_dict['freq_test'],n_off=n_off,plot=False,
                                                        normalize=True,buff=state_dict['buff'],weighted=weighted,input_weights=weights_use,
                                                        fobj=FilReader(state_dict['datadir']+state_dict['ids'] + state_dict['suff'] + "_0.fil"))
    state_dict['I_f'],state_dict['Q_f'],state_dict['U_f'],state_dict['V_f'] = (I_fuse,Q_fuse,U_fuse,V_fuse)
    

    #compute pol fractions
    [(pol_f,pol_t,avg,sigma_frac,snr_frac),
            (L_f,L_t,avg_L,sigma_L,snr_L),
            (C_f_unbiased,C_t_unbiased,avg_C_abs,sigma_C_abs,snr_C),
            (C_f,C_t,avg_C,sigma_C,snrC),snr]= dsapol.get_pol_fraction(I_use,Q_use,U_use,V_use,state_dict['width_native'],state_dict['tsamp'],
                                                                    state_dict['n_t'],state_dict['n_f'],state_dict['freq_test'],
                                                                    n_off=int(NOFFDEF/state_dict['n_t']),plot=False,normalize=True,
                                                                    buff=state_dict['buff'],full=False,weighted=weighted,input_weights=weights_use,
                                                                    fobj=FilReader(state_dict['datadir']+state_dict['ids'] + state_dict['suff'] + "_0.fil"),
                                                                    intL=state_dict['intL']-state_dict['intLbuffer'],intR=state_dict['intR']+state_dict['intRbuffer'])


    #update state dict
    state_dict['Tpol'] = avg
    state_dict['Tpol_err'] = sigma_frac
    state_dict['Tsnr'] = snr_frac
    state_dict['snr'] = snr
    state_dict['Lpol'] = avg_L
    state_dict['Lpol_err'] = sigma_L
    state_dict['Lsnr'] = snr_L
    state_dict['absVpol'] = avg_C_abs
    state_dict['absVpol_err'] = sigma_C_abs
    state_dict['Vpol'] = avg_C
    state_dict['Vpol_err'] = sigma_C
    state_dict['Vsnr'] = snr_C

    #classify
    if np.abs(state_dict['Tpol']/state_dict['Tpol_err']) < 3 and np.abs(1-state_dict['Tpol'])/state_dict['Tpol_err'] > 3:
        state_dict['Tclass'] = "consistent with 0%"
    elif np.abs(state_dict['Tpol']/state_dict['Tpol_err']) > 3 and np.abs(1-state_dict['Tpol'])/state_dict['Tpol_err'] > 3:
        state_dict['Tclass'] = "intermediate"
    elif np.abs(state_dict['Tpol']/state_dict['Tpol_err']) > 3 and np.abs(1-state_dict['Tpol'])/state_dict['Tpol_err'] < 3:
        state_dict['Tclass'] = "consistent with 100%"
    else:
        state_dict['Tclass'] = "unconstrained"

    if np.abs(state_dict['Lpol']/state_dict['Lpol_err']) < 3 and np.abs(1-state_dict['Lpol'])/state_dict['Lpol_err'] > 3:
        state_dict['Lclass'] = "consistent with 0%"
    elif np.abs(state_dict['Lpol']/state_dict['Lpol_err']) > 3 and np.abs(1-state_dict['Lpol'])/state_dict['Lpol_err'] > 3:
        state_dict['Lclass'] = "intermediate"
    elif np.abs(state_dict['Lpol']/state_dict['Lpol_err']) > 3 and np.abs(1-state_dict['Lpol'])/state_dict['Lpol_err'] < 3:
        state_dict['Lclass'] = "consistent with 100%"
    else:
        state_dict['Lclass'] = "unconstrained"

    if np.abs(state_dict['absVpol']/state_dict['absVpol_err']) < 3 and np.abs(1-state_dict['absVpol'])/state_dict['absVpol_err'] > 3:
        state_dict['Vclass'] = "consistent with 0%"
    elif np.abs(state_dict['absVpol']/state_dict['absVpol_err']) > 3 and np.abs(1-state_dict['absVpol'])/state_dict['absVpol_err'] > 3:
        state_dict['Vclass'] = "intermediate"
    elif np.abs(state_dict['absVpol']/state_dict['absVpol_err']) > 3 and np.abs(1-state_dict['absVpol'])/state_dict['absVpol_err'] < 3:
        state_dict['Vclass'] = "consistent with 100%"
    else:
        state_dict['Vclass'] = "unconstrained"


    unbias_factor = 1

    L_t = L_t*I_tuse
    L_f = L_f*I_fuse

    C_t = C_t*I_tuse
    C_f = C_f*I_fuse

    C_t_unbiased = C_t_unbiased*I_tuse
    C_f_unbiased = C_f_unbiased*I_fuse


    L_t = np.sqrt(Q_tuse**2 + U_tuse**2)
    L_t[L_t**2 <= (unbias_factor*np.std(I_tuse[:n_off]))**2] = np.std(I_tuse[:n_off])
    L_t = np.sqrt(L_t**2 - np.std(I_tuse[:n_off])**2)
    L_f = np.sqrt(Q_fuse**2 + U_fuse**2)
    L_f[L_f**2 <= (unbias_factor*np.std(I_tuse[:n_off]))**2] = np.std(I_tuse[:n_off])
    L_f = np.sqrt(L_f**2 - np.std(I_tuse[:n_off])**2)


    C_t = V_tuse
    C_f = V_fuse



    #upate dataframe
    poldf.loc['All'] = [np.around(state_dict['snr'],2),
                        np.around(100*state_dict['Tpol'],2),
                        np.around(100*state_dict['Tpol_err'],2),
                        np.around(state_dict['Tsnr'],2),
                        state_dict['Tclass'],
                        np.around(100*state_dict['Lpol'],2),
                        np.around(100*state_dict['Lpol_err'],2),
                        np.around(state_dict['Lsnr'],2),
                        state_dict['Lclass'],
                        np.around(100*state_dict['Vpol'],2),
                        np.around(100*state_dict['Vpol_err'],2),
                        np.around(100*state_dict['absVpol'],2),
                        np.around(100*state_dict['absVpol_err'],2),
                        np.around(state_dict['Vsnr'],2),
                        state_dict['Vclass'],
                        np.around((180/np.pi)*state_dict['avg_PA'],2),
                        np.around((180/np.pi)*state_dict['PA_err'],2)]


    #plot --> we're creating 3 separate plots: full summary, time domain, frequency domain
    intL=state_dict['intL']-state_dict['intLbuffer']
    intR=state_dict['intR']+state_dict['intRbuffer']


    #(Full Summary Plot)
    fig= plt.figure(figsize=(18,12))
    #ax0 = plt.subplot2grid(shape=(7, 7), loc=(0, 0), colspan=4)
    #ax1 = plt.subplot2grid(shape=(7, 7), loc=(1, 0), colspan=4,rowspan=2,sharex=ax0)
    #ax2 = plt.subplot2grid(shape=(7, 7), loc=(3, 0), colspan=4, rowspan=4)
    #ax3 = plt.subplot2grid(shape=(7, 7), loc=(3, 4), rowspan=4,colspan=2)
    #ax6 = plt.subplot2grid(shape=(7, 7), loc=(3,6), rowspan=4,colspan=1)
    
    ax0 = plt.subplot2grid(shape=(8, 8), loc=(0, 0), colspan=4)
    ax0_f = plt.subplot2grid(shape=(8, 8), loc=(1, 0), colspan=4,sharex=ax0)
    ax1 = plt.subplot2grid(shape=(8, 8), loc=(2, 0), colspan=4,rowspan=2,sharex=ax0)
    ax2 = plt.subplot2grid(shape=(8, 8), loc=(4, 0), colspan=4, rowspan=4)
    ax3 = plt.subplot2grid(shape=(8, 8), loc=(4, 4), rowspan=4,colspan=2)
    ax6 = plt.subplot2grid(shape=(8, 8), loc=(4,7), rowspan=4,colspan=1)
    ax6_f = plt.subplot2grid(shape=(8, 8), loc=(4,6), rowspan=4,colspan=1)

    SNRCUT = 3
    if showghostPA.value:
        ax0.errorbar(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3,
                (180/np.pi)*state_dict['PA_t'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],
                yerr=(180/np.pi)*state_dict['PA_t_errs'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],
                fmt='o',color="blue",markersize=5,linewidth=2,alpha=0.15)
        ax6.errorbar((180/np.pi)*state_dict['PA_f'],state_dict['freq_test'][0],xerr=(180/np.pi)*state_dict['PA_f_errs'],fmt='o',color="blue",markersize=10,linewidth=2,alpha=0.15)

    ax0.errorbar(state_dict['time_axis'][intL:intR][L_t[intL:intR]>=SNRCUT],(180/np.pi)*state_dict['PA_t'][intL:intR][L_t[intL:intR]>=SNRCUT],yerr=(180/np.pi)*state_dict['PA_t_errs'][intL:intR][L_t[intL:intR]>=SNRCUT],fmt='o',color="blue",markersize=10,linewidth=2)
    ax6.errorbar((180/np.pi)*state_dict['PA_f'][L_f >=SNRCUT],state_dict['freq_test'][0][L_f >=SNRCUT],xerr=(180/np.pi)*state_dict['PA_f_errs'][L_f >=SNRCUT],fmt='o',color="blue",markersize=5,linewidth=2)

    ax0.set_xlim(32.7*state_dict['n_t']*state_dict['timestart']*1e-3 - state_dict['window']*32.7*state_dict['n_t']*1e-3,
            32.7*state_dict['n_t']*state_dict['timestop']*1e-3 + state_dict['window']*32.7*state_dict['n_t']*1e-3)
    ax6.set_ylim(np.min(state_dict['freq_test'][0]),np.max(state_dict['freq_test'][0]))
    if sigflag:
        ax0.set_ylabel(r'PPA ($^\circ$)')
        ax6.set_xlabel(r'PPA ($^\circ$)')
    else:
        ax0.set_ylabel(r'PA ($^\circ$)')
        ax6.set_xlabel(r'PA ($^\circ$)')
    ax0.set_ylim(-1.4*95,1.1*95)
    ax6.set_xlim(-1.4*95,1.1*95)
    ax0.set_xticks([])
    ax6.set_yticks([])

    ax1.step(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3,
            I_tuse[state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],label='I',color="black",linewidth=3,where='post')
    ax1.step(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3,
            L_t[state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],label='L',color="blue",linewidth=2.5,where='post')
    ax1.step(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3,
            V_tuse[state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],label='V',color="orange",linewidth=2,where='post')

    ax1.set_xlim(32.7*state_dict['n_t']*state_dict['timestart']*1e-3 - state_dict['window']*32.7*state_dict['n_t']*1e-3,
            32.7*state_dict['n_t']*state_dict['timestop']*1e-3 + state_dict['window']*32.7*state_dict['n_t']*1e-3)
    ax1.set_xticks([])
    ax1.legend(loc="upper right",fontsize=16)
    ax1.set_ylabel(r'S/N')

    ax0_f.step(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3,
            100*pol_t[state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],color="green",linewidth=3,alpha=0.15,where='post')
    ax0_f.step(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3,
            100*(L_t/I_tuse)[state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],color="blue",linewidth=2.5,alpha=0.15,where='post')
    ax0_f.step(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3,
            100*(V_tuse/I_tuse)[state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],color="orange",linewidth=2,alpha=0.15,where='post')
    ax0_f.step(state_dict['time_axis'][intL:intR]*1e-3,
            100*pol_t[intL:intR],label='T/I',color="green",linewidth=3,where='post')
    ax0_f.step(state_dict['time_axis'][intL:intR]*1e-3,
            100*(L_t/I_tuse)[intL:intR],label='L/I',color="blue",linewidth=2.5,where='post')
    ax0_f.step(state_dict['time_axis'][intL:intR]*1e-3,
            100*(V_tuse/I_tuse)[intL:intR],label='V/I',color="orange",linewidth=2,where='post')
    
    ax0_f.set_xticks([])
    ax0_f.set_ylabel(r'%')
    ax0_f.set_ylim(-120,120)
    ax0_f.legend(loc="upper right",fontsize=16)


    ax3.step(I_fuse,state_dict['freq_test'][0],color="black",linewidth=3,where='post')
    ax3.step(L_f,state_dict['freq_test'][0],color="blue",linewidth=2.5,where='post')
    ax3.step(C_f,state_dict['freq_test'][0],color="orange",linewidth=2,where='post')
    ax3.set_ylim(np.min(state_dict['freq_test'][0]),np.max(state_dict['freq_test'][0]))
    ax3.set_xlabel(r'S/N') 
    ax3.set_yticks([])
    
    ax6_f.step(100*pol_f,state_dict['freq_test'][0],color="green",linewidth=3,where='post')
    ax6_f.step(100*(L_f/I_fuse),state_dict['freq_test'][0],color="blue",linewidth=2.5,where='post')
    ax6_f.step(100*(C_f/I_fuse),state_dict['freq_test'][0],color="orange",linewidth=2,where='post')
    ax6_f.set_ylim(np.min(state_dict['freq_test'][0]),np.max(state_dict['freq_test'][0]))
    ax6_f.set_xlabel(r'%') 
    ax6_f.set_xlim(-120,120)
    ax6_f.set_yticks([])
    
    ax2.imshow(I_use[:,state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],aspect='auto',
            extent=[32.7*state_dict['n_t']*state_dict['timestart']*1e-3 - state_dict['window']*32.7*state_dict['n_t']*1e-3,
                32.7*state_dict['n_t']*state_dict['timestop']*1e-3 + state_dict['window']*32.7*state_dict['n_t']*1e-3,
                np.min(state_dict['freq_test'][0]),np.max(state_dict['freq_test'][0])])
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Frequency (MHz)")
    

    if polcomp_menu.value == "All":
        ax0.axvline(state_dict['time_axis'][intL]*1e-3,color='red',linestyle='-')
        ax0.axvline(state_dict['time_axis'][intR]*1e-3,color='red',linestyle='-')
        ax1.axvline(state_dict['time_axis'][intL]*1e-3,color='red',linestyle='-')
        ax1.axvline(state_dict['time_axis'][intR]*1e-3,color='red',linestyle='-')
        ax0_f.axvline(state_dict['time_axis'][intL]*1e-3,color='red',linestyle='-')
        ax0_f.axvline(state_dict['time_axis'][intR]*1e-3,color='red',linestyle='-')
        ax2.axvline(state_dict['time_axis'][intL]*1e-3,color='red',linestyle='-')
        ax2.axvline(state_dict['time_axis'][intR]*1e-3,color='red',linestyle='-')
    else:
        ax0.axvline(state_dict['time_axis'][intL]*1e-3,color='purple',linestyle='--')
        ax0.axvline(state_dict['time_axis'][intR]*1e-3,color='purple',linestyle='--')
        ax1.axvline(state_dict['time_axis'][intL]*1e-3,color='purple',linestyle='--')
        ax1.axvline(state_dict['time_axis'][intR]*1e-3,color='purple',linestyle='--')
        ax0_f.axvline(state_dict['time_axis'][intL]*1e-3,color='purple',linestyle='--')
        ax0_f.axvline(state_dict['time_axis'][intR]*1e-3,color='purple',linestyle='--')
        ax2.axvline(state_dict['time_axis'][intL]*1e-3,color='purple',linestyle='--')
        ax2.axvline(state_dict['time_axis'][intR]*1e-3,color='purple',linestyle='--')

    if state_dict['n_comps'] > 1:
        for i in range(state_dict['n_comps']):
            if polcomp_menu.value != "All" and polcomp_menu.value[-1] == str(i):
                ax0.axvline(state_dict['time_axis'][int(state_dict['comps'][i]['intL'] - state_dict['comps'][i]['intLbuffer'])]*1e-3,color='red',linestyle='-')
                ax0.axvline(state_dict['time_axis'][int(state_dict['comps'][i]['intR'] - state_dict['comps'][i]['intRbuffer'])]*1e-3,color='red',linestyle='-')
                ax1.axvline(state_dict['time_axis'][int(state_dict['comps'][i]['intL'] - state_dict['comps'][i]['intLbuffer'])]*1e-3,color='red',linestyle='-')
                ax1.axvline(state_dict['time_axis'][int(state_dict['comps'][i]['intR'] - state_dict['comps'][i]['intRbuffer'])]*1e-3,color='red',linestyle='-')
                ax0_f.axvline(state_dict['time_axis'][int(state_dict['comps'][i]['intL'] - state_dict['comps'][i]['intLbuffer'])]*1e-3,color='red',linestyle='-')
                ax0_f.axvline(state_dict['time_axis'][int(state_dict['comps'][i]['intR'] - state_dict['comps'][i]['intRbuffer'])]*1e-3,color='red',linestyle='-')
                ax2.axvline(state_dict['time_axis'][int(state_dict['comps'][i]['intL'] - state_dict['comps'][i]['intLbuffer'])]*1e-3,color='red',linestyle='-')
                ax2.axvline(state_dict['time_axis'][int(state_dict['comps'][i]['intR'] - state_dict['comps'][i]['intRbuffer'])]*1e-3,color='red',linestyle='-')
            else:
                ax0.axvline(state_dict['time_axis'][int(state_dict['comps'][i]['intL'] - state_dict['comps'][i]['intLbuffer'])]*1e-3,color='purple',linestyle='--')
                ax0.axvline(state_dict['time_axis'][int(state_dict['comps'][i]['intR'] - state_dict['comps'][i]['intRbuffer'])]*1e-3,color='purple',linestyle='--')
                ax1.axvline(state_dict['time_axis'][int(state_dict['comps'][i]['intL'] - state_dict['comps'][i]['intLbuffer'])]*1e-3,color='purple',linestyle='--')
                ax1.axvline(state_dict['time_axis'][int(state_dict['comps'][i]['intR'] - state_dict['comps'][i]['intRbuffer'])]*1e-3,color='purple',linestyle='--')
                ax0_f.axvline(state_dict['time_axis'][int(state_dict['comps'][i]['intL'] - state_dict['comps'][i]['intLbuffer'])]*1e-3,color='purple',linestyle='--')
                ax0_f.axvline(state_dict['time_axis'][int(state_dict['comps'][i]['intR'] - state_dict['comps'][i]['intRbuffer'])]*1e-3,color='purple',linestyle='--')
                ax2.axvline(state_dict['time_axis'][int(state_dict['comps'][i]['intL'] - state_dict['comps'][i]['intLbuffer'])]*1e-3,color='purple',linestyle='--')
                ax2.axvline(state_dict['time_axis'][int(state_dict['comps'][i]['intR'] - state_dict['comps'][i]['intRbuffer'])]*1e-3,color='purple',linestyle='--')

    plt.subplots_adjust(hspace=0)
    plt.subplots_adjust(wspace=0)
    #plt.show(fig)
    plt.close() 

    #(Time Domain)
    fig_time= plt.figure(figsize=(18,12))
    #ax0 = plt.subplot2grid(shape=(7, 7), loc=(0, 0), colspan=4)
    #ax1 = plt.subplot2grid(shape=(7, 7), loc=(1, 0), colspan=4,rowspan=2,sharex=ax0)
    #ax2 = plt.subplot2grid(shape=(7, 7), loc=(3, 0), colspan=4, rowspan=4)
    #ax3 = plt.subplot2grid(shape=(7, 7), loc=(3, 4), rowspan=4,colspan=2)
    #ax6 = plt.subplot2grid(shape=(7, 7), loc=(3,6), rowspan=4,colspan=1)

    ax0 = plt.subplot2grid(shape=(4, 4), loc=(0, 0), colspan=4)
    ax0_f = plt.subplot2grid(shape=(4, 4), loc=(1, 0), colspan=4)#,sharex=ax0)
    ax1 = plt.subplot2grid(shape=(4, 4), loc=(2, 0), colspan=4,rowspan=2)#,sharex=ax0)

    SNRCUT = 3
    if showghostPA.value:
        ax0.errorbar(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3,
                (180/np.pi)*state_dict['PA_t'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],
                yerr=(180/np.pi)*state_dict['PA_t_errs'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],
                fmt='o',color="blue",markersize=5,linewidth=2,alpha=0.15)

    ax0.errorbar(state_dict['time_axis'][intL:intR][L_t[intL:intR]>=SNRCUT],(180/np.pi)*state_dict['PA_t'][intL:intR][L_t[intL:intR]>=SNRCUT],yerr=(180/np.pi)*state_dict['PA_t_errs'][intL:intR][L_t[intL:intR]>=SNRCUT],fmt='o',color="blue",markersize=10,linewidth=2)

    ax0.set_xlim(32.7*state_dict['n_t']*state_dict['timestart']*1e-3 - state_dict['window']*32.7*state_dict['n_t']*1e-3,
            32.7*state_dict['n_t']*state_dict['timestop']*1e-3 + state_dict['window']*32.7*state_dict['n_t']*1e-3)
    if sigflag:
        ax0.set_ylabel(r'PPA ($^\circ$)')
    else:
        ax0.set_ylabel(r'PA ($^\circ$)')
    ax0.set_ylim(-1.4*95,1.1*95)
    ax0.set_xticks([])

    ax1.step(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3,
            I_tuse[state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],label='I',color="black",linewidth=3,where='post')
    ax1.step(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3,
            L_t[state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],label='L',color="blue",linewidth=2.5,where='post')
    ax1.step(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3,
            V_tuse[state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],label='V',color="orange",linewidth=2,where='post')

    ax1.set_xlim(32.7*state_dict['n_t']*state_dict['timestart']*1e-3 - state_dict['window']*32.7*state_dict['n_t']*1e-3,
            32.7*state_dict['n_t']*state_dict['timestop']*1e-3 + state_dict['window']*32.7*state_dict['n_t']*1e-3)
    ax1.legend(loc="upper right",fontsize=16)
    ax1.set_ylabel(r'S/N')
    ax1.set_xlabel("Time (ms)")

    ax0_f.step(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3,
            100*pol_t[state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],color="green",linewidth=3,alpha=0.15,where='post')
    ax0_f.step(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3,
            100*(L_t/I_tuse)[state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],color="blue",linewidth=2.5,alpha=0.15,where='post')
    ax0_f.step(state_dict['time_axis'][state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']]*1e-3,
            100*(V_tuse/I_tuse)[state_dict['timestart']-state_dict['window']:state_dict['timestop']+state_dict['window']],color="orange",linewidth=2,alpha=0.15,where='post')
    ax0_f.step(state_dict['time_axis'][intL:intR]*1e-3,
            100*pol_t[intL:intR],label='T/I',color="green",linewidth=3,where='post')
    ax0_f.step(state_dict['time_axis'][intL:intR]*1e-3,
            100*(L_t/I_tuse)[intL:intR],label='L/I',color="blue",linewidth=2.5,where='post')
    ax0_f.step(state_dict['time_axis'][intL:intR]*1e-3,
            100*(V_tuse/I_tuse)[intL:intR],label='V/I',color="orange",linewidth=2,where='post')
    ax0_f.set_xlim(32.7*state_dict['n_t']*state_dict['timestart']*1e-3 - state_dict['window']*32.7*state_dict['n_t']*1e-3,
            32.7*state_dict['n_t']*state_dict['timestop']*1e-3 + state_dict['window']*32.7*state_dict['n_t']*1e-3)
    ax0_f.set_xticks([])
    ax0_f.set_ylabel(r'%')
    ax0_f.set_ylim(-120,120)
    ax0_f.legend(loc="upper right",fontsize=16)

    if polcomp_menu.value == "All":
        ax0.axvline(state_dict['time_axis'][intL]*1e-3,color='red',linestyle='-')
        ax0.axvline(state_dict['time_axis'][intR]*1e-3,color='red',linestyle='-')
        ax1.axvline(state_dict['time_axis'][intL]*1e-3,color='red',linestyle='-')
        ax1.axvline(state_dict['time_axis'][intR]*1e-3,color='red',linestyle='-')
        ax0_f.axvline(state_dict['time_axis'][intL]*1e-3,color='red',linestyle='-')
        ax0_f.axvline(state_dict['time_axis'][intR]*1e-3,color='red',linestyle='-')
    else:
        ax0.axvline(state_dict['time_axis'][intL]*1e-3,color='purple',linestyle='--')
        ax0.axvline(state_dict['time_axis'][intR]*1e-3,color='purple',linestyle='--')
        ax1.axvline(state_dict['time_axis'][intL]*1e-3,color='purple',linestyle='--')
        ax1.axvline(state_dict['time_axis'][intR]*1e-3,color='purple',linestyle='--')
        ax0_f.axvline(state_dict['time_axis'][intL]*1e-3,color='purple',linestyle='--')
        ax0_f.axvline(state_dict['time_axis'][intR]*1e-3,color='purple',linestyle='--')


    if state_dict['n_comps'] > 1:
        for i in range(state_dict['n_comps']):
            if polcomp_menu.value != "All" and polcomp_menu.value[-1] == str(i):
                ax0.axvline(state_dict['time_axis'][int(state_dict['comps'][i]['intL'] - state_dict['comps'][i]['intLbuffer'])]*1e-3,color='red',linestyle='-')
                ax0.axvline(state_dict['time_axis'][int(state_dict['comps'][i]['intR'] - state_dict['comps'][i]['intRbuffer'])]*1e-3,color='red',linestyle='-')
                ax1.axvline(state_dict['time_axis'][int(state_dict['comps'][i]['intL'] - state_dict['comps'][i]['intLbuffer'])]*1e-3,color='red',linestyle='-')
                ax1.axvline(state_dict['time_axis'][int(state_dict['comps'][i]['intR'] - state_dict['comps'][i]['intRbuffer'])]*1e-3,color='red',linestyle='-')
                ax0_f.axvline(state_dict['time_axis'][int(state_dict['comps'][i]['intL'] - state_dict['comps'][i]['intLbuffer'])]*1e-3,color='red',linestyle='-')
                ax0_f.axvline(state_dict['time_axis'][int(state_dict['comps'][i]['intR'] - state_dict['comps'][i]['intRbuffer'])]*1e-3,color='red',linestyle='-')
            else:
                ax0.axvline(state_dict['time_axis'][int(state_dict['comps'][i]['intL'] - state_dict['comps'][i]['intLbuffer'])]*1e-3,color='purple',linestyle='--')
                ax0.axvline(state_dict['time_axis'][int(state_dict['comps'][i]['intR'] - state_dict['comps'][i]['intRbuffer'])]*1e-3,color='purple',linestyle='--')
                ax1.axvline(state_dict['time_axis'][int(state_dict['comps'][i]['intL'] - state_dict['comps'][i]['intLbuffer'])]*1e-3,color='purple',linestyle='--')
                ax1.axvline(state_dict['time_axis'][int(state_dict['comps'][i]['intR'] - state_dict['comps'][i]['intRbuffer'])]*1e-3,color='purple',linestyle='--')
                ax0_f.axvline(state_dict['time_axis'][int(state_dict['comps'][i]['intL'] - state_dict['comps'][i]['intLbuffer'])]*1e-3,color='purple',linestyle='--')
                ax0_f.axvline(state_dict['time_axis'][int(state_dict['comps'][i]['intR'] - state_dict['comps'][i]['intRbuffer'])]*1e-3,color='purple',linestyle='--')

    plt.subplots_adjust(hspace=0)
    plt.subplots_adjust(wspace=0)
    #plt.show(fig)
    plt.close()


    #(Frequency Domain)
    fig_freq= plt.figure(figsize=(18,12))
    #ax0 = plt.subplot2grid(shape=(7, 7), loc=(0, 0), colspan=4)
    #ax1 = plt.subplot2grid(shape=(7, 7), loc=(1, 0), colspan=4,rowspan=2,sharex=ax0)
    #ax2 = plt.subplot2grid(shape=(7, 7), loc=(3, 0), colspan=4, rowspan=4)
    #ax3 = plt.subplot2grid(shape=(7, 7), loc=(3, 4), rowspan=4,colspan=2)
    #ax6 = plt.subplot2grid(shape=(7, 7), loc=(3,6), rowspan=4,colspan=1)

    ax6 = plt.subplot2grid(shape=(4, 4), loc=(0, 0), colspan=4)
    ax6_f = plt.subplot2grid(shape=(4, 4), loc=(1, 0), colspan=4)#,sharex=ax6)
    ax3 = plt.subplot2grid(shape=(4, 4), loc=(2, 0), colspan=4,rowspan=2)#,sharex=ax6)

    SNRCUT = 3
    if showghostPA.value:
        ax6.errorbar(state_dict['freq_test'][0],(180/np.pi)*state_dict['PA_f'],yerr=(180/np.pi)*state_dict['PA_f_errs'],fmt='o',color="blue",markersize=10,linewidth=2,alpha=0.15)

    ax6.errorbar(state_dict['freq_test'][0][L_f >=SNRCUT], (180/np.pi)*state_dict['PA_f'][L_f >=SNRCUT],yerr=(180/np.pi)*state_dict['PA_f_errs'][L_f >=SNRCUT],fmt='o',color="blue",markersize=5,linewidth=2)

    ax6.set_xlim(np.min(state_dict['freq_test'][0]),np.max(state_dict['freq_test'][0]))
    if sigflag:
        ax6.set_ylabel(r'PPA ($^\circ$)')
    else:
        ax6.set_ylabel(r'PA ($^\circ$)')
    ax6.set_ylim(-1.4*95,1.1*95)
    ax6.set_xticks([])

    ax3.step(state_dict['freq_test'][0],I_fuse,color="black",linewidth=3,where='post')
    ax3.step(state_dict['freq_test'][0],L_f,color="blue",linewidth=2.5,where='post')
    ax3.step(state_dict['freq_test'][0],C_f,color="orange",linewidth=2,where='post')
    ax3.set_xlim(np.min(state_dict['freq_test'][0]),np.max(state_dict['freq_test'][0]))
    ax3.set_ylabel(r'S/N')
    ax3.set_xlabel("Frequency (MHz)")

    ax6_f.step(state_dict['freq_test'][0],100*pol_f,color="green",linewidth=3,where='post')
    ax6_f.step(state_dict['freq_test'][0],100*(L_f/I_fuse),color="blue",linewidth=2.5,where='post')
    ax6_f.step(state_dict['freq_test'][0],100*(C_f/I_fuse),color="orange",linewidth=2,where='post')
    ax6_f.set_xlim(np.min(state_dict['freq_test'][0]),np.max(state_dict['freq_test'][0]))
    ax6_f.set_ylabel(r'%')
    ax6_f.set_ylim(-120,120)
    ax6_f.set_xticks([])

    plt.subplots_adjust(hspace=0)
    plt.subplots_adjust(wspace=0)
    #plt.show(fig)
    plt.close()

    update_wdict([showghostPA,intLbuffer_slider,intRbuffer_slider,polcomp_menu],['showghostPA','intLbuffer_slider','intRbuffer_slider','polcomp_menu'],param='value')

    return fig,fig_time,fig_freq


def archive_screen(savebutton,archivebutton,archivepolcalbutton,spreadsheetbutton,notesinput,overwritefils):

    """
    screen for RM table and archiving data
    """

    
    RMtable_archive_df = (rmtablefuncs.make_FRB_RMTable(state_dict))#.to_pandas()#rmtable.RMTable({})
    polspec_archive_df = (rmtablefuncs.make_FRB_polSpectra(state_dict))

    #save tables and calibrated filterbanks to level 3 directory
    if savebutton.clicked:
        state_dict['Level3Dir'] = level3_path + state_dict['ids'] + "/Level3/"
        RMtable_archive_df.write(state_dict['Level3Dir'] + state_dict['ids'] + "_RMTable.fits",overwrite=True) 
        polspec_archive_df.write_FITS(state_dict['Level3Dir'] + state_dict['ids'] + "_PolSpectra.fits",overwrite=True) 
        print("Saved RMtable and PolSpectra to h24:" + state_dict['Level3Dir'])
        if 'base_Ical' in state_dict.keys():
            fs = glob.glob(state_dict['datadir'] + state_dict['ids'] + state_dict['suff'] + "_polcal*fil")
            if len(fs) > 0 and not overwritefils.value:
                os.system("cp " + state_dict['datadir'] + state_dict['ids'] + state_dict['suff'] + "_polcal*fil " + state_dict['Level3Dir'])
    
            else:
            
                #load data at base resolution
                if len(glob.glob(state_dict['datadir'] + '/badchans.npy'))>0:
                    fixchansfile = state_dict['datadir'] + '/badchans.npy'
                    fixchansfile_overwrite = False
                else:
                    fixchansfile = ""
                    fixchansfile_overwrite = True

                polcal.make_polcal_filterbanks(state_dict['datadir'],[state_dict['Level3Dir'],state_dict['datadir']],state_dict['ids'],state_dict['polcalfile'],state_dict['suff'],state_dict['suff']+"_polcal",20480,int(NOFFDEF)+12800,True,maxProcesses=polcal_dict['maxProcesses'],fixchans=True,fixchansfile=fixchansfile,fixchansfile_overwrite=fixchansfile_overwrite,verbose=False,background=True)
            
            #dsapol.put_stokes_2D(state_dict['base_Ical'].astype(np.float32),state_dict['base_Qcal'].astype(np.float32),state_dict['base_Ucal'].astype(np.float32),state_dict['base_Vcal'].astype(np.float32),FilReader(state_dict['datadir']+state_dict['ids'] + state_dict['suff'] + "_0.fil"),state_dict['Level3Dir'],state_dict['ids'],suffix="dev_polcal",alpha=True)
    
            print("Saving Pol Calibrated Filterbanks to h24:" + state_dict['Level3Dir'])
    
    #move pol cal voltages to dsastorage
    if archivepolcalbutton.clicked:
        #find available polcal dirs
        polcaldirs = [x for x in next(os.walk(polcal.voltage_copy_path))][1]
        
        for polcaldir in polcaldirs:
            #make directories
            os.system("ssh user@dsa-storage.ovro.pvt mkdir " + dirs["dsastorageCALDir"][dirs["dsastorageCALDir"].index(":")+1:] + "3C48_" + polcaldir)
            os.system("ssh user@dsa-storage.ovro.pvt mkdir " + dirs["dsastorageCALDir"][dirs["dsastorageCALDir"].index(":")+1:] + "3C286_" + polcaldir)
            
            #get corr dirs
            corrdirs_3C48 = [x for x in next(os.walk(polcal.voltage_copy_path + polcaldir + "/3C48/"))][1]
            corrdirs_3C286 = [x for x in next(os.walk(polcal.voltage_copy_path + polcaldir + "/3C286/"))][1]
            
            #copy
            for corr in corrdirs_3C48:
                os.system("scp -r " + polcal.voltage_copy_path + polcaldir + "/3C48/" + corr + " " + dirs["dsastorageCALDir"] + "3C48_" + polcaldir)
            print("Archived polcal files from " + polcal.voltage_copy_path + polcaldir + "/3C48 to " + dirs["dsastorageCALDir"] + "3C48_" + polcaldir)

            for corr in corrdirs_3C286:
                os.system("scp -r " + polcal.voltage_copy_path + polcaldir + "/3C286/" + corr + " " + dirs["dsastorageCALDir"] + "3C286_" + polcaldir)
            print("Archived polcal files from " + polcal.voltage_copy_path + polcaldir + "/3C48 to " + dirs["dsastorageCALDir"] + "3C48_" + polcaldir)

    #move T3 files to dsastorage
    if archivebutton.clicked:
        state_dict['dsastorageDir'] = dsastorageFRBDir + state_dict['ids'] + "/"
        state_dict['Level3Dir'] = level3_path + state_dict['ids'] + "/Level3/"
        os.system("scp " + state_dict['Level3Dir'] + "*fil " + state_dict['dsastorageDir'] + state_dict['ids'] + "/Level3/ > " + rmtablefuncs.logfile)
        os.system("scp " + state_dict['Level3Dir'] + "*fits " + state_dict['dsastorageDir'] + state_dict['ids'] + "/Level3/ > " + rmtablefuncs.logfile)
        print("Archived filterbanks and RMtable to dsastorage:" + state_dict['dsastorageDir'] + state_dict['ids'] + "/Level3/")

    #save as a csv for dsa spreadsheet
    if spreadsheetbutton.clicked:
        rows = []
        #first make a row for full burst
        mPA = ""
        mPA_err = ""
        iPA = ""
        iPA_err = ""
        if "RMcal" in state_dict["RMcalibrated"].keys():
            iPA = state_dict['avg_PA']
            iPA_err = state_dict['PA_err']
        else:
            mPA = state_dict['avg_PA']
            mPA_err = state_dict['PA_err']

        rows.append([state_dict["nickname"],
                     state_dict["DM"],
                     "", #DM_exgal
                     "", #z
                     "", #DM_host
                     "", #DM_host + err
                     "", #DM_host -err
                     state_dict["n_t"],
                     state_dict["n_f"],
                     state_dict['polcalfile'],
                     'RMcal' in state_dict['RMcalibrated'].keys(),
                     state_dict["RMcalibrated"]["RM1"][0],
                     state_dict["RMcalibrated"]["RM1"][1],
                     state_dict["RMcalibrated"]['RM_tools'][0],
                     state_dict["RMcalibrated"]['RM_tools'][1],
                     state_dict['RMcalibrated']['RM2'][0],
                     state_dict['RMcalibrated']['RM2'][1],
                     state_dict['RMcalibrated']['RMFWHM'],
                     np.max(state_dict['RMcalibrated']['RMsnrs2']),
                     state_dict['RM_gal'],
                     state_dict['RM_galerr'],
                     state_dict['RM_ion'],
                     state_dict['RM_ionerr'],
                     "", #RM opt
                     "", #RM exgal
                     "", #RM exgal error
                     "", #RM source
                     "", #RM host
                     "", #RM host err
                     "", #B field
                     "", #B field + err
                     "", #B field - err
                     state_dict['Tpol'],
                     state_dict['Tpol_err'],
                     state_dict['Tclass'],
                     "", #upper limit
                     state_dict['Lpol'],
                     state_dict['Lpol_err'],
                     state_dict['Lclass'],
                     "", #upper limit
                     state_dict['absVpol'],
                     state_dict['absVpol_err'],
                     state_dict['absVclass'],
                     "", #upper limit
                     state_dict['Vpol'],
                     state_dict['Vpol_err'],
                     "",
                     mPA,
                     mPA_err,
                     iPA,
                     iPA_err,
                     "",
                     (state_dict['intR']-state_dict['intL'])*(32.7e-3)*state_dict['n_t'],
                     state_dict['snr'],
                     state_dict['Tsnr'],
                     state_dict['Lsnr'],
                     state_dict['Vsnr'],
                     notesinput.value]
                     )
       
        if state_dict['n_comps'] > 1:
            for i in range(state_dict['n_comps']):
                #first make a row for full burst
                mPA = ""
                mPA_err = ""
                iPA = ""
                iPA_err = ""
                if "RMcal" in state_dict['comps'][i]["RMcalibrated"].keys():
                    iPA = state_dict['comps'][i]['avg_PA']
                    iPA_err = state_dict['comps'][i]['PA_err']
                else:
                    mPA = state_dict['comps'][i]['avg_PA']
                    mPA_err = state_dict['comps'][i]['PA_err']

                rows.append(["--" + state_dict["nickname"] + " (peak " + str(i) + ")",
                     state_dict["DM"],
                     "", #DM_exgal
                     "", #z
                     "", #DM_host
                     "", #DM_host + err
                     "", #DM_host -err
                     state_dict["n_t"],
                     state_dict["n_f"],
                     state_dict['polcalfile'],
                     'RMcal' in state_dict['comps'][i]['RMcalibrated'].keys(),
                     state_dict['comps'][i]["RMcalibrated"]["RM1"][0],
                     state_dict['comps'][i]["RMcalibrated"]["RM1"][1],
                     state_dict['comps'][i]["RMcalibrated"]['RM_tools'][0],
                     state_dict['comps'][i]["RMcalibrated"]['RM_tools'][1],
                     state_dict['comps'][i]['RMcalibrated']['RM2'][0],
                     state_dict['comps'][i]['RMcalibrated']['RM2'][1],
                     state_dict['comps'][i]['RMcalibrated']['RMFWHM'],
                     np.max(state_dict['comps'][i]['RMcalibrated']['RMsnrs2']),
                     state_dict['RM_gal'],
                     state_dict['RM_galerr'],
                     state_dict['RM_ion'],
                     state_dict['RM_ionerr'],
                     "", #RM opt
                     "", #RM exgal
                     "", #RM exgal error
                     "", #RM source
                     "", #RM host
                     "", #RM host err
                     "", #B field
                     "", #B field + err
                     "", #B field - err
                     state_dict['comps'][i]['Tpol'],
                     state_dict['comps'][i]['Tpol_err'],
                     state_dict['comps'][i]['Tclass'],
                     "", #upper limit
                     state_dict['comps'][i]['Lpol'],
                     state_dict['comps'][i]['Lpol_err'],
                     state_dict['comps'][i]['Lclass'],
                     "", #upper limit
                     state_dict['comps'][i]['absVpol'],
                     state_dict['comps'][i]['absVpol_err'],
                     state_dict['comps'][i]['absVclass'],
                     "", #upper limit
                     state_dict['comps'][i]['Vpol'],
                     state_dict['comps'][i]['Vpol_err'],
                     "",
                     mPA,
                     mPA_err,
                     iPA,
                     iPA_err,
                     "",
                     (state_dict['comps'][i]['intR']-state_dict['comps'][i]['intL'])*(32.7e-3)*state_dict['n_t'],
                     state_dict['comps'][i]['snr'],
                     state_dict['comps'][i]['Tsnr'],
                     state_dict['comps'][i]['Lsnr'],
                     state_dict['comps'][i]['Vsnr'],
                     ""]
                     )

        with open(state_dict['datadir'] + '/' + state_dict['ids'] + '_spreadsheet.csv','w',newline='') as csvfile:
            wr = csv.writer(csvfile,delimiter=',')
            for r in rows:
                wr.writerow(r)


    update_wdict([notesinput,overwritefils],["notesinput","overwritefils"],param="value")
    return


def savestate(tsave):
    f = open(dirs['cwd'] + '/interface/.current_state/cache_time.pkl','wb')
    pkl.dump({"cache_time":tsave.isot,"frb":state_dict['ids']+"_"+state_dict['nickname']},f)
    f.close()

    f = open(dirs['cwd'] + '/interface/.current_state/state_dict.pkl','wb')
    pkl.dump(state_dict,f)
    f.close()

    f = open(dirs['cwd'] + '/interface/.current_state/wdict.pkl','wb')
    pkl.dump(wdict,f)
    f.close()

    f = open(dirs['cwd'] + '/interface/.current_state/polcal_dict.pkl','wb')
    pkl.dump(polcal_dict,f)
    f.close()

    f = open(dirs['cwd'] + '/interface/.current_state/RMcaldict.pkl','wb')
    pkl.dump(RMcaldict,f)
    f.close()

    f = open(dirs['cwd'] + '/interface/.current_state/df.pkl','wb')
    df.to_pickle(f)
    f.close()

    f = open(dirs['cwd'] + '/interface/.current_state/df_polcal.pkl','wb')
    df_polcal.to_pickle(f)
    f.close()

    f = open(dirs['cwd'] + '/interface/.current_state/df_beams.pkl','wb')
    df_beams.to_pickle(f)
    f.close()

    f = open(dirs['cwd'] + '/interface/.current_state/df_scint.pkl','wb')
    df_scint.to_pickle(f)
    f.close()

    f = open(dirs['cwd'] + '/interface/.current_state/df_scat.pkl','wb')
    df_scat.to_pickle(f)
    f.close()

    f = open(dirs['cwd'] + '/interface/.current_state/RMdf.pkl','wb')
    RMdf.to_pickle(f)
    f.close()

    f = open(dirs['cwd'] + '/interface/.current_state/poldf.pkl','wb')
    poldf.to_pickle(f)
    f.close()


    RMtable_archive_df.write_tsv(dirs['cwd'] + '/interface/.current_state/RMtable_archive_df.tsv',overwrite=True)

    polspec_archive_df.write_FITS(dirs['cwd'] + '/interface/.current_state/polspec_archive_df.fits',overwrite=True)

    return


def restorestate():
    global state_dict
    global df
    global df_polcal
    global polcal_dict
    global df_beams
    global df_scint
    global df_scat
    global wdict
    global RMcaldict
    global RMdf
    global poldf
    global RMtable_archive_df
    global polspec_archive_df
    
    
    f = open(dirs['cwd'] + '/interface/.current_state/cache_time.pkl','rb')
    cache = pkl.load(f)
    print("Restoring session for FRB " + str(cache['frb']) + " from " + str(cache['cache_time']))
    f.close()



    f = open(dirs['cwd'] + '/interface/.current_state/state_dict.pkl','rb')
    state_dict = pkl.load(f)
    f.close()

    f = open(dirs['cwd'] + '/interface/.current_state/wdict.pkl','rb')
    wdict = pkl.load(f)
    f.close()

    f = open(dirs['cwd'] + '/interface/.current_state/polcal_dict.pkl','rb')
    polcal_dict = pkl.load(f)
    f.close()
    
    f = open(dirs['cwd'] + '/interface/.current_state/RMcaldict.pkl','rb')
    RMcaldict = pkl.load(f)
    f.close()
    
    f = open(dirs['cwd'] + '/interface/.current_state/df.pkl','rb')
    df = pd.read_pickle(f)
    f.close()
    
    f = open(dirs['cwd'] + '/interface/.current_state/df_polcal.pkl','rb')
    df_polcal = pd.read_pickle(f)
    f.close()
    
    f = open(dirs['cwd'] + '/interface/.current_state/df_beams.pkl','rb')
    df_beams = pd.read_pickle(f)
    f.close()
    
    f = open(dirs['cwd'] + '/interface/.current_state/df_scint.pkl','rb')
    df_scint = pd.read_pickle(f)
    f.close()
    
    f = open(dirs['cwd'] + '/interface/.current_state/df_scat.pkl','rb')
    df_scat = pd.read_pickle(f)
    f.close()

    f = open(dirs['cwd'] + '/interface/.current_state/RMdf.pkl','rb')
    RMdf = pd.read_pickle(f)
    f.close()

    f = open(dirs['cwd'] + '/interface/.current_state/poldf.pkl','rb')
    poldf = pd.read_pickle(f)
    f.close()

    RMtable_archive_df = rmtable.RMTable.read_tsv(dirs['cwd'] + '/interface/.current_state/RMtable_archive_df.tsv')
    polspec_archive_df.read_FITS(dirs['cwd'] + '/interface/.current_state/polspec_archive_df.fits')

    return
