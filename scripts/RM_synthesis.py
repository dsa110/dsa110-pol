import numpy as np
import signal
import sys
import json
f = open("directories.json","r")
dirs = json.load(f)
f.close()

sys.path.append(dirs['cwd'])
sys.path.append(dirs['cwd'] + "/dsapol/")
from dsapol import dsapol
import argparse
from dsapol import parsec
import time
"""
Runs RM synthesis pulling data from file
"""


RMdir = dirs["logs"] + "RM_files/"
"""
def handler(signum,frame,process_dir):
    print("Background RM synthesis completed")
    #get results from file
    
    res = np.load(process_dir + "output_values.npy")#parsec.dirs['logs'] + "RM_files/" + parsec.state_dict['dname'] + "output_values.npy")
    RM = res[0]
    RMerr = res[2]
    parsec.state_dict["RMcalibrated"]["RM1"] = [RM,RMerr]
    parsec.RMdf.loc['All', '1D-Synth'] = RM
    parsec.RMdf.loc['All', '1D-Synth Error'] = RMerr
    res = np.load(process_dir + "output_spectrum.npy")
    parsec.state_dict["RMcalibrated"]['RMsnrs1'],parsec.state_dict["RMcalibrated"]['trial_RM1'] =res[1,:],res[0,:]
    
    RMdisplay.value = RM
    RMerrdisplay.value = RMerrdisplay
    print("handler done")
    return
"""
#signal.signal(signal.SIGUSR1,handler)




def main(args):

    farr = np.load(args.process + "/input_spectrum.npy")
    freq_test = [farr[0,:]]*4
    Q_fcal = farr[1,:]
    U_fcal = farr[2,:]
    print("input data:",farr)    
    trial_RM = np.load(args.process + "/trial_rm.npy")
    print("trial RM:",trial_RM)
    RM1,phi1,RMsnrs1,RMerr1,tmp = dsapol.faradaycal(Q_fcal,Q_fcal,U_fcal,U_fcal,freq_test,trial_RM,[0],plot=False,show=False,fit_window=100,err=False,matrixmethod=False,multithread=True,maxProcesses=10,numbatch=2,sendtofile = args.process + "/output_spectrum.npy")
    print("RM synthesis DONE!")

    np.save(args.process + "output_values.npy",np.array([RM1,phi1,RMerr1]))
    
    print("output saved")
    #signal.raise_signal(signal.SIGUSR1)
    #print("signal raised")
    
    return 0

if __name__=="__main__":
    
    #argument parsing
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--process',type=str,help='Directory for process files',default=RMdir)

    args = parser.parse_args()
    
    #signal.signal(signal.SIGUSR1,lambda signum, frame: handler(signum,frame,args.process))

    main(args)
