import json
import numpy as np
from scattering import scat
from scintillation import scint
from astropy.time import Time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed, wait
import contextlib

f = open("directories.json","r")
dirs = json.load(f)
f.close()


logfile = dirs["logs"] + "scatscint_logfile.txt" #"/media/ubuntu/ssd/sherman/code/dsapol_logfiles/archive_logfile.txt"

def future_callback_rns(future,dname,dname_result,outdir):#,fit,weights,trial_RM,fit_window,Qcal,Ucal,timestart,timestop,dname):
    print("Nested Sampling Complete")
    scatter_result = future.result()
    os.system("cp " + dirs['logs'] + "scat_files/" + dname + "* " + outdir)
    #os.system("cp " + dirs['logs'] + "scat_files/" + dname_result + " " + outdir)
    #scatter_result.save_to_file("result.json",overwrite=True,outdir=dirs['logs'] + "scat_files/" + dname,extension='json')
    print("Saved to " + dirs['logs'] + "scat_files/" + dname_result)
    return


def run_nested_sampling(timeseries_for_fit, outdir, label, p0, comp_num, nlive, time_resolution, timeaxis_for_fit, low_bounds, upp_bounds, sigma_for_fit, background,resume=False,verbose=False):

    if background:
        print("Running Nested Sampling in the background...")


        #make directory for output
        dname = "proc_scat_" + Time.now().isot + "/"
        dname_result = dname + label + "_result.json"
        os.mkdir(dirs['logs'] + "scat_files/" + dname)

        #create executor
        executor = ProcessPoolExecutor(5)
        t = executor.submit(run_nested_sampling,timeseries_for_fit, dirs['logs'] + "scat_files/" + dname, label, p0, comp_num, nlive, time_resolution, timeaxis_for_fit, low_bounds, upp_bounds, sigma_for_fit,False,resume,verbose)
        t.add_done_callback(lambda future: future_callback_rns(future,dname,dname_result,outdir))

        return dname_result, dname

    if verbose:

        scatter_result = scat.nested_sampling(timeseries_for_fit,
                                        outdir=outdir, label=label,
                                        p0=p0, comp_num=comp_num, nlive=nlive, time_resolution=time_resolution,
                                        debug=False, time=timeaxis_for_fit,
                                        lower_bounds=low_bounds,upper_bounds=upp_bounds,sigma=sigma_for_fit,resume=resume)
    else:
        with open(logfile,"w") as f:
            with contextlib.redirect_stdout(f):
                with contextlib.redirect_stderr(f):     
                    scatter_result = scat.nested_sampling(timeseries_for_fit,
                                        outdir=outdir, label=label,
                                        p0=p0, comp_num=comp_num, nlive=nlive, time_resolution=time_resolution,
                                        debug=False, time=timeaxis_for_fit,
                                        lower_bounds=low_bounds,upper_bounds=upp_bounds,sigma=sigma_for_fit,resume=resume)
        f.close()
    return scatter_result


