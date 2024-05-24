import os
import numpy as np
import glob


"""
This file contains helper functions to create polarized filterbanks using Vikram's offline
beamforming code. Functions act as python wrappers around shell scripts that will run on h23.
"""

#default directory to store filterbanks
import json
f = open("directories.json","r")
dirs = json.load(f)
f.close()
output_dir = dirs["data"]#"/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/"
cand_dir = dirs["candidates"] #"/dataz/dsa110/candidates/"#$2/Level2/voltages/"
logfile = dirs["logs"] + "beamform_logfile.txt" #"/media/ubuntu/ssd/sherman/code/dsapol_logfiles/beamform_logfile.txt"#"/media/ubuntu/ssd/sherman/code/dsa110-pol/offline_beamforming/beamforming_logfile.txt"

def clear_logfile(logfile=logfile):
    """
    This function clears the contents of the logfile
    """
    os.system("> " + logfile)
    return

#${datestrings[$i]} ${candnames[$i]} ${nicknames[$i]} ${dates[$i]} ${bms[$i]} ${mjds[$i]} ${dms[$i]}

def get_bfweights(ids,path=cand_dir):
    """
    This function takes the FRB candname and returns
    the beamformer weights applied
    """
    weightsfile = glob.glob(path + ids + "/Level2/calibration/*yaml")#[0]
    try:
        weightsfile = weightsfile[0]
        bfweights = weightsfile[weightsfile.index("beamformer_weights_")+19:-5]
    except Exception as ex:
        bfweights = None
    return bfweights

def get_fils(ids,nickname,path=output_dir):
    """
    This function takes the FRB candname and nickname and
    returns fil files from the data directory
    """
    return glob.glob(path + ids + "_" + nickname + "/*.fil")

def make_filterbanks(ids,nickname,bfweights,ibeam,mjd,DM,path=output_dir):
    """
    This function takes FRB parameters and the beamformer weights to run the 
    offline beamforming script. Outputs polarized filterbanks to the path provided.
    Output is redirected to a logfile at /media/ubuntu/ssd/sherman/code/dsa110-pol/offline_beamforming/beamforming_logfile.txt,
    and the command is run in the background. Returns 0 on success, 1 on failure
    """
    #return os.system("/media/ubuntu/ssd/sherman/code/dsa110-pol/offline_beamforming/for_testing.bash 2>&1 > /media/ubuntu/ssd/sherman/code/dsa110-pol/offline_beamforming/beamforming_logfile.txt &")
    clear_logfile()
    return os.system(dirs["cwd"] + "offline_beamforming/run_beamformer_visibs_bfweightsupdate_sb.bash NA "
            + str(ids) + " " + str(nickname) + " " + str(bfweights) + " " + str(ibeam) + " " + str(mjd) + " " + str(DM) + 
            " 2>&1 > " + logfile + " &")


