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
"""
This file contains functions related to creating polarization calibration Jones matrix parameters. 
This includes wrappers around Vikram's bash scripts for generating voltage files using the 
most recent calibrator observations, forming beams when known, and smoothing/averaging the solution
with previous solutions.
"""


data_path = "/dataz/dsa110/T3/"
voltage_copy_path = "/media/ubuntu/ssd/sherman/polcal_voltages/"
logfile = "/media/ubuntu/ssd/sherman/code/dsa110-pol/offline_beamforming/polcal_logfile.txt"
output_path = "/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/"
bfweights_path = "/dataz/dsa110/operations/beamformer_weights/generated/"
bfweights_output_path = "/media/ubuntu/ssd/sherman/code/pol_self_calibs_FORPAPER/"
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
            os.system("/media/ubuntu/ssd/sherman/code/dsa110-pol/offline_beamforming/move_cal_voltages.bash " + calname + " " + f[len(calname):len(calname)+3] + " " + caldate + " " + new_path + " " + path + " 2>&1 > " + logfile + " &")
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
        #os.system("cp " + path + fname + " " + new_path)
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
        print("./offline_beamforming/run_beamformer_offline_bfweightsupdate_cals_sb.bash " + str(k) + " " + str(cal_id_dict[f[:len(calname)+3]]['mjd']) + " " + calname + " " +  bftstamp + " " + str(caldate) + " 2>&1 > " + logfile + " &")
        #os.system("./offline_beamforming/run_beamformer_visibs_bfweightsupdate_cals_sb.bash " + str(k) + " " + str(cal_id_dict[f[:len(calname)+3]]['mjd']) + " " + calname + " " +  bftstamp + " 2>&1 > " + logfile + " &")

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
    return os.system("/media/ubuntu/ssd/sherman/code/dsa110-pol/offline_beamforming/run_beamformer_visibs_bfweightsupdate_cals_sb.bash NA "
            + str(calid) + " " + str(calname) + " " + str(bfweights) + " " + str(ibeam) + " " + str(mjd) + " 0 " + str(caldate) +
            " 2>&1 > " + logfile + " &")



### Functions to compute, filter, and combine solutions ###

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

def get_calfil_files(calname,caldate,path=output_path):
    """
    This function returns the filterbank files available for a given calibrator and 
    observation date
    """

    obs_files = glob.glob(path + calname + "_" + caldate + "/" + calname + "*0.fil")
    obs_ids = [f[-len(calname)-3-10:-10] for f in obs_files]

    return obs_files,obs_ids
