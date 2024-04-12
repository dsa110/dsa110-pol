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
output_path = "/media/ubuntu/ssd/sherman/code/scratch_weights_update_2022-06-03_32-7us/"
bfweights_path = "/dataz/dsa110/operations/beamformer_weights/generated/"
bfweights_output_path = "/media/ubuntu/ssd/sherman/code/pol_self_calibs_FOR_PAPER/"


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
def get_bfweights(filenames,calname,path=bfweights_path,voltage_path=data_path):
    """
    This function finds the beamformer weights most recent before the 
    observation.
    """
    
    #get list of relevant bf weights
    allbfweightfiles = glob.glob(path + "*beamformer_weights_sb00*" + VLANAME_DICT[calname] + "*")
    
    #make a list of their isot times
    allbfweightdates = np.array([Time(allbfweightfiles[i][-23:-4],format='isot') for i in range(len(allbfweightfiles))])

    #get mjd from one of the json file (b/c dict only saves the day, not time)
    mjd = np.nan
    print(filenames)
    for fname in filenames:
        if 'json' in fname:
            f = open(voltage_path + "corr03/" + fname,'r')
            mjd = json.load(f)[fname[:len(calname)+3]]['mjds']
            f.close()
            break
    if np.isnan(mjd):
        #raise ValueError("No json with mjd found")
        return []
    obstime = Time(mjd,format='mjd').datetime

    #find bf weights with smallest date diff
    tdeltas = np.array([(obstime - allbfweightdates[i].datetime).total_seconds() for i in range(len(allbfweightdates))])

    #ignore those from after the observation
    tdeltas[tdeltas < 0] = np.inf
    bestidx = np.argmin(tdeltas)

    #now get all files
    bestbfweights = glob.glob(path + "*beamformer_weights_sb*" + VLANAME_DICT[calname] + "*" + allbfweightdates[bestidx].isot + ".dat")
    bestbfweights = np.array([bestbfweights[i][len(path):] for i in range(len(bestbfweights))]) 
    return bestbfweights

def copy_bfweights(bfweights,path=bfweights_path,new_path=bfweights_output_path):
    """
    After getting a list of beamformer weight files, this function copies them to 
    the pol_self_cal_FOR_PAPER directory
    """
    for fname in bfweights:
        print("copying " + path + fname + " to " + new_path)
        #os.system("cp " + path + fname + " " + new_path)
    return

def get_source_beams(filenames,caldate,calname,path=output_path):
    """
    This function calls find_beam() for each of the voltage files copied to
    get the beam number for each observation. 
    """

    #loop through files and get beam numbers
    beam_dict = dict()
    for f in filenames:
        beam_dict[f] = dsapol.find_beam("_" + f[:len(calname)+3],shape=(16,7680,256),path=path + calname + "_" + caldate + "/",plot=False)[0]

    return beam_dict


