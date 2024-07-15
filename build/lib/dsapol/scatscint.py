import json
import numpy as np
from scattering import scat
from scintillation import scint
from astropy.time import Time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed, wait
import contextlib
import emcee

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


def run_nested_sampling(timeseries_for_fit, outdir, label, p0, comp_num, nlive, time_resolution, timeaxis_for_fit, low_bounds, upp_bounds, sigma_for_fit, background,resume=False,verbose=False,dlogz=0.1):

    if background:
        print("Running Nested Sampling in the background...")


        #make directory for output
        dname = "proc_scat_" + Time.now().isot + "/"
        dname_result = dname + label + "_result.json"
        os.mkdir(dirs['logs'] + "scat_files/" + dname)

        #create executor
        executor = ProcessPoolExecutor(5)
        t = executor.submit(run_nested_sampling,timeseries_for_fit, dirs['logs'] + "scat_files/" + dname, label, p0, comp_num, nlive, time_resolution, timeaxis_for_fit, low_bounds, upp_bounds, sigma_for_fit,False,resume,verbose,dlogz)
        t.add_done_callback(lambda future: future_callback_rns(future,dname,dname_result,outdir))

        return dname_result, dname

    if verbose:

        scatter_result = scat.nested_sampling(timeseries_for_fit,
                                        outdir=outdir, label=label,
                                        p0=p0, comp_num=comp_num, nlive=nlive, time_resolution=time_resolution,
                                        debug=False, time=timeaxis_for_fit,
                                        lower_bounds=low_bounds,upper_bounds=upp_bounds,sigma=sigma_for_fit,resume=resume,dlogz=dlogz)
    else:
        with open(logfile,"w") as f:
            with contextlib.redirect_stdout(f):
                with contextlib.redirect_stderr(f):     
                    scatter_result = scat.nested_sampling(timeseries_for_fit,
                                        outdir=outdir, label=label,
                                        p0=p0, comp_num=comp_num, nlive=nlive, time_resolution=time_resolution,
                                        debug=False, time=timeaxis_for_fit,
                                        lower_bounds=low_bounds,upper_bounds=upp_bounds,sigma=sigma_for_fit,resume=resume,dlogz=dlogz)
        f.close()
    return scatter_result




#MCMC sampling with emcee
def run_MCMC_sampling(timeseries_for_fit, outdir, label, p0, timeaxis_for_fit, low_bounds, upp_bounds, sigma_for_fit, background,verbose=False,nwalkers=32,nsteps=5000,discard=100,thin=15):
    """
    This function fits for scattering parameters using MCMC
    """
   
    if background:
        print("Running Nested Sampling in the background...")


        #make directory for output
        dname = "proc_scat_" + Time.now().isot + "/"
        dname_result = dname + label + "_MCMC_samples.npy"
        os.mkdir(dirs['logs'] + "scat_files/" + dname)

        #create executor
        executor = ProcessPoolExecutor(5)
        t = executor.submit(run_MCMC_sampling,timeseries_for_fit, dirs['logs'] + "scat_files/" + dname, label, p0, timeaxis_for_fit, low_bounds, upp_bounds, sigma_for_fit,False,verbose,nwalkers,nsteps,discard,thin)
        t.add_done_callback(lambda future: future_callback_rns(future,dname,dname_result,outdir))

        return dname_result, dname

    
    #define log likelihood function
    def log_likelihood(theta, x, y, yerr):
        f = theta[-1]
        model = scat.exp_gauss_n(x,*list(theta))
        yerr_f = np.sqrt(yerr**2 + (f*model)**2)
        return -0.5 * (np.sum(((y - model) ** 2 )/ (yerr_f**2) + np.log(yerr_f)))
    
    #define log prior
    def log_prior(theta):
        f = theta[-1]
        if np.all(np.array(theta[:-1]) >= np.array(low_bounds)) and np.all(np.array(theta[:-1]) <= np.array(upp_bounds)) and 0 <= f <= 10:
            return 0.0
        return -np.inf

    #define log posterior
    def log_probability(theta, x, y, yerr):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta, x, y, yerr)
    
    print("Process ID: " + str(os.getpid()))
    #define start values
    initvals = np.concatenate([p0,[0.5]])
    pos = np.array(initvals) + 1e-4 * np.random.randn(nwalkers,len(initvals))
    nwalkers,ndim = pos.shape

    #define sampler
    sampler = emcee.EnsembleSampler(nwalkers,ndim,log_probability, args=(timeaxis_for_fit,timeseries_for_fit,sigma_for_fit))

    #run sampler
    sampler.run_mcmc(pos,nsteps,progress=False)

    #flatten and get best fit params
    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)[:,:-1]
    p_median = np.nanmedian(flat_samples,axis=0)
    p_upper = np.nanpercentile(flat_samples,84,axis=0) - p_median
    p_lower = p_median - np.nanpercentile(flat_samples,16,axis=0)

    np.save(outdir + label + "_MCMC_samples.npy",flat_samples)

    return flat_samples,p_median,p_upper,p_lower
