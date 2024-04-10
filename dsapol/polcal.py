from dsapol import dsapol
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

def get_voltages(calname,timespan=timedelta(days=365*2),path=data_path):
    """
    Get pol cal voltage files within the past month
    """
    
    #get list of voltage files
    vfiles = glob.glob(data_path + "/corr*/*" + calname + "*")
    
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
            vtimes.append(time.ctime(os.path.getctime(v))[:10])
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

