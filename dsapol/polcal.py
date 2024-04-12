from dsapol import dsapol
from dsapol import polbeamform
import numpy as np
import glob
from datetime import datetime
from datetime import timedelta
from astropy.time import Time
import time
import os
"""
This file contains functions related to creating polarization calibration Jones matrix parameters. 
This includes wrappers around Vikram's bash scripts for generating voltage files using the 
most recent calibrator observations, forming beams when known, and smoothing/averaging the solution
with previous solutions.
"""


data_path = "/dataz/dsa110/T3/"
voltage_copy_path = "/media/ubuntu/ssd/sherman/polcal_voltages/"
logfile = "/media/ubuntu/ssd/sherman/code/dsa110-pol/offline_beamforming/polcal_logfile.txt"

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
            vtimes.append(time.ctime(os.path.getctime(v))[:10] + time.ctime(os.path.getctime(v))[-5:])
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
        if 'header' not in f:
            os.system("/media/ubuntu/ssd/sherman/code/dsa110-pol/offline_beamforming/move_cal_voltages.bash " + calname + " " + f[len(calname):len(calname)+3] + " " + caldate.replace(" ","_") + " " + new_path + " " + path + " 2>&1 > " + logfile + " &")
    
    #return output directory
    return new_path + caldate.replace(" ","_") + "/" + calname + "/"     

