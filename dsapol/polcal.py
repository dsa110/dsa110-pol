from dsapol import dsapol
from dsapol import polbeamform
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
from scipy.ndimage import convolve1d
from RMtools_1D.do_RMsynth_1D import run_rmsynth
from RMtools_1D.do_RMclean_1D import run_rmclean
from RMtools_1D.do_QUfit_1D_mnest import run_qufit
from astropy.coordinates import EarthLocation
import astropy.units as u
from concurrent.futures import ProcessPoolExecutor
"""
This file contains functions related to creating polarization calibration Jones matrix parameters. 
This includes wrappers around Vikram's bash scripts for generating voltage files using the 
most recent calibrator observations, forming beams when known, and smoothing/averaging the solution
with previous solutions.
"""
import json
f = open(os.environ['DSAPOLDIR'] + "directories.json","r")
dirs = json.load(f)
f.close()

data_path = dirs["T3"]#"/dataz/dsa110/T3/"
voltage_copy_path = dirs["polcal_voltages"] #"/media/ubuntu/ssd/sherman/polcal_voltages/"
logfile = dirs["logs"] + "polcal_logfile.txt" #"/media/ubuntu/ssd/sherman/code/dsapol_logfiles/polcal_logfile.txt"
output_path = dirs["data"]#"/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/"
bfweights_path = dirs["gen_bfweights"] #"/dataz/dsa110/operations/beamformer_weights/generated/"
bfweights_output_path = dirs["polcal"] + "polcal_bfweights/" #"/media/ubuntu/ssd/sherman/code/pol_self_calibs_FORPAPER/"
lastcalfile = dirs["cwd"] + "interface/last_cal_metadata.txt" #"/media/ubuntu/ssd/sherman/code/dsa110-pol/interface/last_cal_metadata.txt"
default_path = dirs["polcal"] #"/media/ubuntu/ssd/sherman/code/"
repo_path = dirs["cwd"]
"""
print(data_path)
print(voltage_copy_path)
print(logfile)
print(output_path)
print(bfweights_path)
print(bfweights_output_path)
print(lastcalfile)
print(default_path)
"""
middle_beam = 125

#### Functions for copying voltages from T3 to working dir for given observing date ####
def get_voltages(calname,timespan=timedelta(days=365*2),path=data_path):
    """
    Get pol cal voltage files within the past month
    """
    
    #get list of voltage files
    vfiles = glob.glob(path + "/corr*/*" + calname + "*")
    
    #get dates for those within 1 year
    vtimes = []
    vgoodfiles = []
    currtime = Time.now().to_datetime()
    for v in vfiles:    
        d = time.gmtime(os.path.getctime(v))
        dt = datetime(year=d.tm_year,
                     month=d.tm_mon,
                     day=d.tm_mday,
                     hour=d.tm_hour,
                     minute=d.tm_min,
                     second=d.tm_sec)
        if (np.abs(currtime-dt) < timespan):
            #vtimes.append(time.ctime(os.path.getctime(v))[:10] + time.ctime(os.path.getctime(v))[-5:])
            vtimes.append(dt.isoformat()[:10])
            vgoodfiles.append(v)
    return vtimes,vgoodfiles

def iso_voltages(vtimes,vgoodfiles):
    """
    Pulls the voltage files corresponding to each unique timestamp
    """

    #get list of unique timestamps
    vtimes_unique = np.unique(vtimes)
    mapping = dict()
    for vtime in vtimes_unique:
        mapping[vtime] = []

    #get all corr03 files with these timestamps
    for i in range(len(vtimes)):
        if 'corr03' in vgoodfiles[i]:
            mapping[vtimes[i]].append(vgoodfiles[i][len(vgoodfiles[i]) - vgoodfiles[i][::-1].index("/"):])
    return mapping


def copy_voltages(filenames,caldate,calname,path=data_path,new_path=voltage_copy_path):
    """
    Copies voltages from T3 to the scratch directory specified by new_path.
    Default new_path is h23:/media/ubuntu/ssd/sherman/code/dsa110-pol/polcal_voltages/; we 
    don't want to save polcal voltage files to the git repo because they're 
    really large.
    """
    polbeamform.clear_logfile(logfile)

    #loop through and copy each file from the corr nodes
    for f in filenames:
        #print(f)
        if 'header' not in f:
            os.system("nohup " + repo_path + "offline_beamforming/move_cal_voltages.bash " + calname + " " + f[len(calname):len(calname)+3] + " " + caldate + " " + new_path + " " + path + " > " + logfile + " 2>&1 &")
        #print("/media/ubuntu/ssd/sherman/code/dsa110-pol/offline_beamforming/move_cal_voltages.bash " + calname + " " + f[len(calname):len(calname)+3] + " " + caldate + " " + new_path + " " + path + " 2>&1 > " + logfile + " &")

    #return output directory
    return new_path + caldate + "/" + calname + "/"     


#### Functions to find and copy beamformer weights for given observation ####
VLANAME_DICT = {"3C48":"0137+331","3C286":"1331+305"}
def get_bfweights(filenames,calname,path=bfweights_path,voltage_path=data_path,xtend='corr03'):
    """
    This function finds the beamformer weights most recent before the 
    observation.
    """
    
    #get list of relevant bf weights
    allbfweightfiles = glob.glob(path + "*beamformer_weights_sb00*" + VLANAME_DICT[calname] + "*")
    
    #make a list of their isot times
    allbfweightdates = np.array([Time(allbfweightfiles[i][-23:-4],format='isot') for i in range(len(allbfweightfiles))])
    #print(allbfweightfiles)
    #get mjd from one of the json file (b/c dict only saves the day, not time)
    mjd = np.nan
    #print(filenames)
    for fname in filenames:
        if 'json' in fname:
            f = open(voltage_path + xtend + "/" + fname,'r')
            mjd = json.load(f)[fname[:len(calname)+3]]['mjds']
            f.close()
            break
    if np.isnan(mjd):
        #print(voltage_path + "corr03/" + fname,mjd)
        #raise ValueError("No json with mjd found")
        return []
    obstime = Time(mjd,format='mjd').datetime

    #find bf weights with smallest date diff
    tdeltas = np.array([(obstime - allbfweightdates[i].datetime).total_seconds() for i in range(len(allbfweightdates))])

    #ignore those from after the observation
    tdeltas[tdeltas < 0] = np.inf
    bestidx = np.argmin(tdeltas)
    #print(path + "*beamformer_weights_sb*" + VLANAME_DICT[calname] + "*" + allbfweightdates[bestidx].isot + ".dat")

    #now get all files
    bestbfweights = glob.glob(path + "*beamformer_weights_sb*" + VLANAME_DICT[calname] + "*" + allbfweightdates[bestidx].isot[:19] + ".dat")
    bestbfweights = np.array([bestbfweights[i][len(path):] for i in range(len(bestbfweights))]) 
    
    return bestbfweights


def get_bfweight_isot(calname,mjd,path=bfweights_output_path):
    """
    Given the mjd, this returns the most recent self-calibrated beamformer weights
    in the provided directory
    """
    #get list of relevant bf weights
    allbfweightfiles = glob.glob(path + "*beamformer_weights_sb00*" + VLANAME_DICT[calname] + "*")

    #make a list of their isot times
    allbfweightdates = np.array([Time(allbfweightfiles[i][-23:-4],format='isot') for i in range(len(allbfweightfiles))])

    #convert mjd to time
    obstime = Time(mjd,format='mjd').datetime

    #find bf weights with smallest date diff
    tdeltas = np.array([(obstime - allbfweightdates[i].datetime).total_seconds() for i in range(len(allbfweightdates))])

    #ignore those from after the observation
    tdeltas[tdeltas < 0] = np.inf
    bestidx = np.argmin(tdeltas)

    return allbfweightdates[bestidx].isot

def copy_bfweights(bfweights,path=bfweights_path,new_path=bfweights_output_path):
    """
    After getting a list of beamformer weight files, this function copies them to 
    the pol_self_cal_FOR_PAPER directory
    """
    for fname in bfweights:
        print("copying " + path + fname + " to " + new_path)
        os.system("cp " + path + fname + " " + new_path)
    return


#functions to calibrate voltage data
def get_avail_caldates(path=voltage_copy_path):
    """
    This function gets and returns lists of dates available and their corresponding
    beamformer weights.
    """
    caldates = [f.path[-10:] for f in os.scandir(path) if f.is_dir()]
    
    return caldates

def get_all_calfiles(caldate,calname,path=voltage_copy_path,bfpath=bfweights_output_path):
    """
    This function gets and returns all the voltage files and beamformer files for a given cal date.
    """

    #first get paths for all the voltage files
    vfiles = glob.glob(path + caldate + "/*/corr*/")
    vfilejson = ""
    for v in vfiles:
        if 'json' in v: vfilejson = v
    if vfilejson == "": 
        print("No JSON files")
        return [],[]


    #then get all bf weight files
    bffiles = get_bfweights([v],calname,path=bfpath,voltage_path=path)
    return vfiles,bffiles

def beamform_polcal(vfiles,bffiles,calname,caldate,path=output_path):
    """
    This function forms beams for the polarization calibrators
    """
    if len(vfiles) == 0 or len(bffiles) == 0: return
    #collect the observation names
    cal_id_dict = dict()
    for f in vfiles:
        if f[:len(calname)+3] in cal_id_dict.keys():
            cal_id_dict[f[:len(calname)+3]] = dict()
            cal_id_dict[f[:len(calname)+3]]['files'] = []
        cal_id_dict[f[:len(calname)+3]].append(f)

        #find mjd
        if ('json' in f) and ('mjd' not in cal_id_dict[f[:len(calname)+3]].keys()):
            fobj = open(f,'r')
            cal_id_dict[f[:len(calname)+3]]['mjd'] = json.load(f)[f[:len(calname)+3]]['mjds']
            fobj.close()

    #find beamformer timestamp
    bftstamp = bffiles[0][bffiles[0].index(VLANAME_DICT[calname]):bffiles[0].index('.')]

    #clear log files
    polbeamform.clear_logfile(logfile)

    #for each observation, make voltage files
    for k in cal_id_dict.keys():
        print("./offline_beamforming/run_beamformer_offline_bfweightsupdate_cals_sb.bash " + str(k) + " " + str(cal_id_dict[f[:len(calname)+3]]['mjd']) + " " + calname + " " +  bftstamp + " " + str(caldate) + " > " + logfile + " 2>&1 &")
        os.system("nohup ./offline_beamforming/run_beamformer_offline_bfweightsupdate_cals_sb.bash " + str(k) + " " + str(cal_id_dict[f[:len(calname)+3]]['mjd']) + " " + calname + " " +  bftstamp + " " + str(caldate) + " > " + logfile + " 2>&1 &")

    #make a copy of jsons so we have access to the mjd
    for f in vfiles:
        if ('json' in f) and ('corr03' in f):
            os.system("cp " + f + " " + path + calname + "_" + caldate + "/")
    return


#functions to get beam numbers from existing files
def get_beamfinding_files(path=output_path):
    """
    This function gets all the folder dates in scratch dir that have correlator files
    """
    all_files = [f.path[-10:] for f in os.scandir(path) if (f.is_dir() and (('3C286' in f.path) or ('3C48' in f.path)))]
    return np.unique(all_files)

def get_source_beams(caldate,calname,path=output_path):
    """
    This function calls find_beam() for each of the voltage files copied to
    get the beam number for each observation. 
    """
    #get observation names
    obs_files = glob.glob(path + calname + "_" + caldate + "/corr03_" + calname + "*.out")
    obs_ids = [f[-len(calname)-3-4:-4] for f in obs_files]
    print(obs_files)
    print(obs_ids)

    #loop through files and get beam numbers
    beam_dict = dict()
    for obs_id in obs_ids:#[0:1]:
        print("Finding beam for " + obs_id + "...")
        bout = dsapol.find_beam("_" + obs_id,shape=(16,7680,256),path=path + calname + "_" + caldate + "/",plot=False)
        beam_dict[obs_id] = dict()
        beam_dict[obs_id]['beam'] = bout[0]
        beam_dict[obs_id]['beamspectrum'] = bout[1].mean(0).mean(0)
        print("Done, beam = " + str(beam_dict[obs_id]['beam']))

    return beam_dict

def get_cal_mjd(calname,caldate,calid,path=output_path):
    """
    Pulls mjd for a given cal observation from json file
    """
    f = path + calname + "_" + caldate + "/" + calname + calid + "_header.json.sav"
    fobj = open(f,'r')
    mjd = json.load(fobj)[calname+calid]['mjds']
    fobj.close()
    return mjd

#null zcj 3C48 0137+331_2024-01-30T00:57:05 72 60339.0366192841 0
def make_cal_filterbanks(calname,caldate,calid,bfweights,ibeam,mjd,path=output_path):
    """
    This function takes FRB parameters and the beamformer weights to run the
    offline beamforming script. Outputs polarized filterbanks to the path provided.
    Output is redirected to a logfile at /media/ubuntu/ssd/sherman/code/dsa110-pol/offline_beamforming/polcal_logfile.txt,
    and the command is run in the background. Returns 0 on success, 1 on failure
    """

    clear_logfile()
    return os.system("nohup /media/ubuntu/ssd/sherman/code/dsa110-pol/offline_beamforming/run_beamformer_visibs_bfweightsupdate_cals_sb.bash NA "
            + str(calid) + " " + str(calname) + " " + str(bfweights) + " " + str(ibeam) + " " + str(mjd) + " 0 " + str(caldate) +
            " > " + logfile + " 2>&1 &")



def get_best_beam(beam_dict):
    """
    Returns the obs id and beam closest to middle_beam (125 for DSA-64)
    """

    names = []
    beams = []
    for k in beam_dict.keys():
        names.append(str(k))
        beams.append(beam_dict[k]['beam'])
    idx = np.argmin(np.abs(np.array(beams) - middle_beam))
    return names[idx],beams[idx]

def get_calfil_files(calname,caldate,obsid,path=output_path):
    """
    This function returns the filterbank files available for a given calibrator and 
    observation date
    """

    obs_files = glob.glob(path + calname + "_" + caldate + "/" + obsid + "*.fil")
    obs_ids = [f[-len(calname)-3-10:-10] for f in obs_files]

    return obs_files,obs_ids

### Functions to compute new solutions

#predicted flux from Perley-Butler 2017
coeffs_3C286 = [1.2481,-0.4507,-0.1798,0.0357]
coeffs_3C48 = [ 1.3253, -0.7553, -0.1914, 0.0498]
RA_3C48 = ((1 + 37/60 + 41.1/3600)*360/24)
DEC_3C48 = (33 + 9/60 + 32/3600)
RM_3C48 = -68 #Perley-Butler 2017
p_3C48 = 0.005 #polarization fraction
chi_3C48= 25*np.pi/180 #position angle
def PB_flux(coeffs,nu_GHz):
    logS = np.zeros(len(nu_GHz))
    for i in range(len(coeffs)):
        logS += coeffs[i]*(np.log10(nu_GHz)**(i))
        
    return 10**logS

def clean_peaks(I_init,peakheight=2,padwidth=10):
    """
    This function cleans a spectrum for spurious peaks that will
    be smoothed for a calibrator solution.
    """
    #normalize
    I_new = copy.deepcopy(I_init)
    I_norm = (I_new - np.nanmean(I_new))/np.nanstd(I_new)

    #find peaks of specified height and width
    pks = find_peaks(np.abs(np.pad(I_norm,pad_width=padwidth,mode='constant')),height=peakheight)[0]
    pks = np.array(pks)-padwidth
    wds = peak_widths(np.abs(np.pad(I_norm,pad_width=padwidth,mode='constant')),pks)[0]

    #set each peak to the mean intensity 
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
        #print((low,hi))
        I_new[low:hi] = np.mean(I_init)
    return I_new

def piecewise_polyfit(GY_fullres,freq_test_fullres,edgefreq=1418,breakfreq=1418,deg=5):
    """
    This function fits a piecewise polynomial to
    a cleaned spectrum
    """

    #fit the lower part of the spectrum
    edgeidx=np.argmin(np.abs(freq_test_fullres[0]-edgefreq))
    popt = np.polyfit(freq_test_fullres[0][edgeidx:],GY_fullres[edgeidx:],deg)
    GY_fit1 = np.zeros(len(GY_fullres))
    for i in range(len(popt)):
        GY_fit1 += popt[i]*(freq_test_fullres[0]**(len(popt)-i-1))

    #fit the upper part of the spectrum
    popt = np.polyfit(freq_test_fullres[0][:edgeidx],GY_fullres[:edgeidx],5)
    GY_fit2 = np.zeros(len(GY_fullres))
    for i in range(len(popt)):
        GY_fit2 += popt[i]*(freq_test_fullres[0]**(len(popt)-i-1))

    #stitch together
    breakidx=np.argmin(np.abs(freq_test_fullres[0]-breakfreq))

    GY_fit = np.zeros(len(GY_fit2))
    GY_fit[breakidx:] = GY_fit1[breakidx:]
    GY_fit[:breakidx] = GY_fit2[:breakidx]
   
    return GY_fit,GY_fit1,GY_fit2

def abs_gyy_solution(last_cal_soln,last_cal_num,caldate,obsid,ibeam,n_t=1,n_f=1,nsamps=5,n_t_down=32,p=p_3C48,chi=chi_3C48,RM=RM_3C48,RA=RA_3C48,DEC=DEC_3C48,path=output_path,edgefreq=1418,breakfreq=1418,sf_window_weights=255,sf_order=5,peakheight=2,padwidth=10,deg=5,sfflag=False,polyfitflag=False):
    """
    This function uses the 3C48 calibrator observation given to compute the absolute gain in the y feed.
    """

    #read data
    gain_dir = path + '3C48_' + caldate + '/' 
    (Igainuc,Qgainuc,Ugainuc,Vgainuc,fobj,timeaxis,freq_test,wav_test,badchans) = dsapol.get_stokes_2D(gain_dir,obsid + "_dev",nsamps,n_t=n_t,n_f=n_t_down,n_off=int(12000//n_t),sub_offpulse_mean=False,verbose=False)

    #PA calibration
    Ical,Qcal,Ucal,Vcal,ParA = dsapol.calibrate_angle(Igainuc,Qgainuc,Ugainuc,Vgainuc,fobj,ibeam,RA,DEC)

    #compute simulated I,Q,U and XX,YY (https://science.nrao.edu/facilities/vla/docs/manuals/obsguide/modes/pol)
    I_sim = PB_flux(coeffs_3C48,freq_test[0]*1e-3) #Jy
    Q_sim = I_sim*(p*np.cos(chi)*np.cos(2*ParA) + p*np.sin(chi)*np.sin(2*ParA))
    U_sim = I_sim*(-p*np.cos(chi)*np.sin(2*ParA) + p*np.sin(chi)*np.cos(2*ParA))
    V_sim = np.zeros(I_sim.shape)

    #apply the measured RM to predict what signal should look like
    I_sim,Q_sim,U_sim,V_sim = dsapol.calibrate_RM(I_sim,Q_sim,U_sim,V_sim,-RM,0,freq_test,stokes=True)

    #compute the expected X and Y feed voltages
    XX_sim = 0.5*(I_sim + Q_sim)
    YY_sim = 0.5*(I_sim - Q_sim)

    #clean and eliminate spurious peaks
    I_new = clean_peaks(Igainuc.mean(1),peakheight=peakheight,padwidth=padwidth) 
    Q_new = clean_peaks(Qgainuc.mean(1),peakheight=peakheight,padwidth=padwidth) 

    #compute the X and Y feed voltages
    XX = 0.5*(I_new + Q_new)
    YY = 0.5*(I_new - Q_new)
    
    #compare to the simulated voltages to get the gain
    GX = np.sqrt(XX/XX_sim)
    GY = np.sqrt(YY/YY_sim)

    GX = ma.masked_invalid(GX, copy=True)
    GY = ma.masked_invalid(GY, copy=True)

    #interpolate to high resolution
    (Igainuc,Qgainuc,Ugainuc,Vgainuc,fobj,timeaxis,freq_test_fullres,wav_test,badchans) = dsapol.get_stokes_2D(gain_dir,obsid + "_dev",nsamps,n_t=n_t,n_f=1,n_off=int(12000//n_t),sub_offpulse_mean=False,verbose=False)
    idx = np.isfinite(GX)
    f_GX= interp1d(freq_test[0][idx],GX[idx],kind="linear",fill_value="extrapolate")
    GX_fullres_i = f_GX(freq_test_fullres[0])

    idx = np.isfinite(GY)
    f_GY= interp1d(freq_test[0][idx],GY[idx],kind="linear",fill_value="extrapolate")
    GY_fullres_i = f_GY(freq_test_fullres[0])

    #average with past solution
    GY_fullres = average_cal_solution(GY_fullres_i,last_cal_soln,last_cal_num)

    #piecewise cubic spline fit
    GY_fit,GY_fit1,GY_fit2 = piecewise_polyfit(GY_fullres,freq_test_fullres,edgefreq=edgefreq,breakfreq=breakfreq,deg=deg) 
    
    if sf_window_weights >= 5 and sf_window_weights > sf_order:# and sfflag:
        #savgol filter
        GY_fit_sf = sf(GY_fit,sf_window_weights,sf_order)
        GY_fullres_sf = sf(GY_fullres,sf_window_weights,sf_order)
    return GY_fit, GY_fit_sf, GY_fullres, GY_fullres_sf, GY_fullres_i, freq_test_fullres


def gain_solution(last_cal_soln,last_cal_num,caldate,obsid,ibeam,n_t=1,n_f=1,nsamps=5,n_t_down=32,path=output_path,edgefreq=1360,breakfreq=1360,sf_window_weights=255,sf_order=5,peakheight=3,padwidth=10,deg=5,sfflag=False,polyfitflag=False):
    """
    This function uses the 3C48 calibrator observation given to compute the ratio of gains in x and y feeds.
    """
    
    #read data
    gain_dir = path + '3C48_' + caldate + '/'
    (Igainuc,Qgainuc,Ugainuc,Vgainuc,fobj,timeaxis,freq_test,wav_test,badchans) = dsapol.get_stokes_2D(gain_dir,obsid + "_dev",nsamps,n_t=n_t,n_f=n_f,n_off=int(12000//n_t),sub_offpulse_mean=False,verbose=False)

    #gain calibration solution
    ratio_2,ratio_params_2,ratio_sf_2 = dsapol.gaincal_full(gain_dir,'3C48',[obsid[-3:]],n_t=n_t,n_f=n_f,nsamps=nsamps,deg=deg,suffix="_dev",average=True,plot=False,show=False,sfwindow=-1,clean=True,padwidth=padwidth,peakheight=peakheight,n_t_down=n_t_down)

    
    #interpolate to high resolution
    (Igainuc,Qgainuc,Ugainuc,Vgainuc,fobj,timeaxis,freq_test_fullres,wav_test,badchans) = dsapol.get_stokes_2D(gain_dir,obsid + "_dev",nsamps,n_t=n_t,n_f=1,n_off=int(12000//n_t),sub_offpulse_mean=False,verbose=False)
    idx = np.isfinite(ratio_2)
    f_ratio = interp1d(freq_test[0][idx],ratio_2[idx],kind="linear",fill_value="extrapolate")
    ratio_fullres_i = f_ratio(freq_test_fullres[0])

    #average with past solution
    ratio_fullres = average_cal_solution(ratio_fullres_i,last_cal_soln,last_cal_num)

    #piecewise fit
    ratio_fit,ratio_fit1,ratio_fit2 = piecewise_polyfit(ratio_fullres,freq_test_fullres,edgefreq=edgefreq,breakfreq=breakfreq,deg=deg)

    if sf_window_weights >= 5 and sf_window_weights > sf_order:# and sfflag:
        #savgol filter
        ratio_fit_sf = sf(ratio_fit,sf_window_weights,sf_order)
        ratio_fullres_sf = sf(ratio_fullres,sf_window_weights,sf_order)


    return ratio_fit, ratio_fit_sf, ratio_fullres, ratio_fullres_sf, ratio_fullres_i, freq_test_fullres



def phase_solution(last_cal_soln,last_cal_num,caldate,obsid,ibeam,n_t=1,n_f=1,nsamps=5,n_t_down=32,path=output_path,sf_window_weights=255,sf_order=5,peakheight=3,padwidth=10,deg=5,sfflag=False,polyfitflag=False):
    """
    This function uses the 3C286 calibrator observation given to compute the phase difference in x and y feeds.
    """

    #read data
    phase_dir = path + '3C286_' + caldate + '/'
    (Igainuc,Qgainuc,Ugainuc,Vgainuc,fobj,timeaxis,freq_test,wav_test,badchans) = dsapol.get_stokes_2D(phase_dir,obsid + "_dev",nsamps,n_t=n_t,n_f=n_f,n_off=int(12000//n_t),sub_offpulse_mean=False,verbose=False)

    #gain calibration solution
    phase_2,phase_params_2,phase_sf_2 = dsapol.phasecal_full(phase_dir,'3C286',[obsid[-3:]],n_t=n_t,n_f=n_f,nsamps=nsamps,deg=deg,suffix="_dev",average=True,plot=False,show=False,sfwindow=-1,clean=True,padwidth=padwidth,peakheight=peakheight,n_t_down=n_t_down)


    #interpolate to high resolution
    (Igainuc,Qgainuc,Ugainuc,Vgainuc,fobj,timeaxis,freq_test_fullres,wav_test,badchans) = dsapol.get_stokes_2D(phase_dir,obsid + "_dev",nsamps,n_t=n_t,n_f=1,n_off=int(12000//n_t),sub_offpulse_mean=False,verbose=False)
    idx = np.isfinite(phase_2)
    f_phase = interp1d(freq_test[0][idx],phase_2[idx],kind="linear",fill_value="extrapolate")
    phase_fullres_i = f_phase(freq_test_fullres[0])

    #average with past solution
    phase_fullres = average_cal_solution(phase_fullres_i,last_cal_soln,last_cal_num)

    # fit
    phase_fit = np.zeros(len(phase_2))
    for i in range(len(phase_params_2)):
        phase_fit += phase_params_2[i]*(freq_test_fullres[0]**(len(phase_params_2)-i-1))

    if sf_window_weights >= 5 and sf_window_weights > sf_order:# and sfflag:
        #savgol filter
        phase_fit_sf = sf(phase_fit,sf_window_weights,sf_order)
        phase_fullres_sf = sf(phase_fullres,sf_window_weights,sf_order)

    return phase_fit, phase_fit_sf, phase_fullres, phase_fullres_sf, phase_fullres_i, freq_test_fullres

### functions to merge with previous cal solution

def get_last_calmeta(filename=lastcalfile):
    """
    This function pulls metadata for the previous calibrator observation needed to 
    properly average with subsequent observations.
    """

    dat = []
    with open(filename,"r") as f:
        rdr = csv.reader(f,delimiter=',')
        for row in rdr:
            dat.append(row[0])
    return dat

def update_last_calmeta(caldate,calid1,calid2,lastnumobs,filename=lastcalfile):
    """
    This function updates the metadata with new calibrator observation. The caldate should be
    given in the format YY-MM-DD
    """

    with open(filename,"w") as f:
        wr = csv.writer(f,delimiter=',')
        wr.writerow([caldate])
        wr.writerow([calid1])
        wr.writerow([calid2])
        wr.writerow([lastnumobs+1])
    return lastnumobs+1
    
def average_cal_solution(cal_new,cal_old,cal_numobs):
    """
    This function averages together the new calibration solution with the previous solution
    """
    
    return (np.array(cal_new) + np.array(cal_old)*cal_numobs)/(cal_numobs + 1)

def write_polcal_solution(calid1,calid2,lastnumobs,ratio_fullavg,ratio_fit,ratio_sf,ratio_fit_sf,
                                                   phase_fullavg,phase_fit,phase_sf,phase_fit_sf,
                                                   GY_fullavg,GY_fit,GY_sf,GY_fit_sf,gxx_final,gyy_final,freq_axis,metafilename=lastcalfile):
    """
    This function writes the new calibrator solution to a file labelled 'POLCAL_PARAMETERS' with today's date, and updates metadata
    in the lastcalfile
    """

    #get today's date
    today = Time.now().to_datetime()
    new_date = str(today.year)[-2:] + "-" + f"{today.month:02}" + "-" + f"{today.day:02}"
    new_filename = "POLCAL_PARAMETERS_" + new_date + ".csv"
    
    #write solution file
    with open("/media/ubuntu/ssd/sherman/code/" + new_filename,'w',newline='') as csvfile:
        writer = csv.writer(csvfile,delimiter=',')
        writer.writerow(np.concatenate([["|gxx|/|gyy|"],ratio_fullavg]))
        writer.writerow(np.concatenate([["|gxx|/|gyy| fit"],ratio_fit]))
        writer.writerow(np.concatenate([["|gxx|/|gyy| sf"],ratio_sf]))
        writer.writerow(np.concatenate([["|gxx|/|gyy| fit sf"],ratio_fit_sf]))
        writer.writerow(np.concatenate([["phixx-phiyy"],phase_fullavg]))
        writer.writerow(np.concatenate([["phixx-phiyy fit"],phase_fit]))
        writer.writerow(np.concatenate([["phixx-phiyy sf"],phase_sf]))
        writer.writerow(np.concatenate([["phixx-phiyy fit sf"],phase_fit_sf]))
        writer.writerow(np.concatenate([["|gyy|"],GY_fullavg]))
        writer.writerow(np.concatenate([["|gyy| fit"],GY_fit]))
        writer.writerow(np.concatenate([["|gyy| sf"],GY_sf]))
        writer.writerow(np.concatenate([["|gyy| fit sf"],GY_fit_sf]))
        writer.writerow(np.concatenate([["gxx"],gxx_final]))
        writer.writerow(np.concatenate([["gyy"],gyy_final]))
        writer.writerow(np.concatenate([["freq_axis"],freq_axis]))

    #update metadata
    update_last_calmeta(caldate=new_date,calid1=calid1,calid2=calid2,lastnumobs=lastnumobs,filename=metafilename)

    return new_filename

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


def make_polcal_filterbanks(datadir,outputdirs,ids,ibeam,RA,DEC,polcalfile,init_suffix,new_suffix,nsamps,n_off,sub_offpulse_mean,maxProcesses,fixchans,fixchansfile,fixchansfile_overwrite,verbose,background):
    """
    This function reads a filterbank file, applies pol calbration, and re-saves it, all at native resolution.
    This is needed because I cannot save filterbanks at low resolution
    """

    #create executor
    if background:

        executor = ProcessPoolExecutor(5)
        executor.submit(make_polcal_filterbanks,datadir,outputdirs,ids,polcalfile,init_suffix,new_suffix,nsamps,n_off,sub_offpulse_mean,maxProcesses,fixchans,fixchansfile,fixchansfile_overwrite,verbose,False)

    else:
        #read data
        (I,Q,U,V,fobj,timeaxis,freq_test,wav_test,badchans) = dsapol.get_stokes_2D(datadir,ids + init_suffix,nsamps,start=0,n_t=1,n_f=1,n_off=n_off,sub_offpulse_mean=sub_offpulse_mean,fixchans=fixchans,verbose=verbose,fixchansfile=fixchansfile,fixchansfile_overwrite=fixchansfile_overwrite)

        #calibrate, note we have to unmask before cal
        gxx,gyy,cal_freq_axis = read_polcal(polcalfile)
        Ical,Qcal,Ucal,Vcal = dsapol.calibrate(I.data,Q.data,U.data,V.data,(gxx,gyy),stokes=True,multithread=True,maxProcesses=maxProcesses,bad_chans=badchans,verbose=verbose)
        Ical,Qcal,Ucal,Vcal = dsapol.calibrate_angle(I,Q,U,V,FilReader(datadir+ids+init_suffix+"_0.fil"),ibeam,RA,DEC)
                


        #save
        if type(outputdirs) == str:
            dsapol.put_stokes_2D(Ical.data.astype(np.float32),Qcal.data.astype(np.float32),Ucal.data.astype(np.float32),Vcal.data.astype(np.float32),FilReader(datadir+ids + init_suffix + "_0.fil"),outputdirs,ids,suffix=new_suffix,alpha=True,verbose=verbose)
        else:
            dsapol.put_stokes_2D(Ical.data.astype(np.float32),Qcal.data.astype(np.float32),Ucal.data.astype(np.float32),Vcal.data.astype(np.float32),FilReader(datadir+ids + init_suffix + "_0.fil"),outputdirs[0],ids,suffix=new_suffix,alpha=True,verbose=verbose)
            for i in range(1,len(outputdirs)):
                os.system("cp " + outputdirs[0] + "*" + ids + new_suffix + "*.fil " + outputdirs[i])

        

    return




