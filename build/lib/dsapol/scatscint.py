import json
import numpy as np
from scattering import scat
from scintillation import scint
from astropy.time import Time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed, wait
import contextlib
import emcee
import bilby

f = open("directories.json","r")
dirs = json.load(f)
f.close()


logfile = dirs["logs"] + "scatscint_logfile.txt" #"/media/ubuntu/ssd/sherman/code/dsapol_logfiles/archive_logfile.txt"

def future_callback_rns(future,dname,dname_result,outdir):#,fit,weights,trial_RM,fit_window,Qcal,Ucal,timestart,timestop,dname):
    print("Nested Sampling Complete")
    #scatter_result = future.result()
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
        dname_result = dname + label + "_bilby_result.json"
        dname_result_params = dname + label + "_bilby_results.npy"
        dname_BIC = dname + label + "_bilby_BIC.npy"
        os.mkdir(dirs['logs'] + "scat_files/" + dname)

        #create executor
        executor = ProcessPoolExecutor(5)
        t = executor.submit(run_nested_sampling,timeseries_for_fit, dirs['logs'] + "scat_files/" + dname, label, p0, comp_num, nlive, time_resolution, timeaxis_for_fit, low_bounds, upp_bounds, sigma_for_fit,False,resume,verbose,dlogz)
        t.add_done_callback(lambda future: future_callback_rns(future,dname,dname_result,outdir))

        return dname_result, dname_result_params, dname_BIC, dname

    if verbose:
        #run nested sampling
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

    #get median parameters
    p_median = np.nanmedian(scatter_result.samples,axis=0)
    p_upper = np.nanpercentile(scatter_result.samples,84,axis=0) - p_median
    p_lower = p_median - np.nanpercentile(scatter_result.samples,16,axis=0)
    ndim = len(p_median)

    #calculate BIC
    l = bilby.core.likelihood.GaussianLikelihood(timeaxis_for_fit, timeseries_for_fit, lambda x, params : scat.exp_gauss_n(x,*params), sigma=sigma_for_fit, params=list(p_median))
    BIC = ndim*np.log(len(timeaxis_for_fit)) - 2*l.log_likelihood()   

    np.save(outdir + label + "_bilby_results.npy",np.array([p_median,p_upper,p_lower]))
    np.save(outdir + label + "_bilby_BIC.npy",np.array([BIC]))

    return scatter_result, p_median, p_upper, p_lower, BIC




#MCMC sampling with emcee
def run_MCMC_sampling(timeseries_for_fit, outdir, label, p0, timeaxis_for_fit, low_bounds, upp_bounds, sigma_for_fit, background,verbose=False,nwalkers=32,nsteps=5000,discard=100,thin=15):
    """
    This function fits for scattering parameters using MCMC
    """
   
    if background:
        print("Running Markov-Chain Monte Carlo (MCMC) in the background...")


        #make directory for output
        dname = "proc_scat_" + Time.now().isot + "/"
        dname_samples = dname + label + "_MCMC_samples.npy"
        dname_result = dname + label + "_MCMC_results.npy"
        dname_BIC = dname + label + "_MCMC_BIC.npy"
        os.mkdir(dirs['logs'] + "scat_files/" + dname)

        #create executor
        executor = ProcessPoolExecutor(5)
        t = executor.submit(run_MCMC_sampling,timeseries_for_fit, dirs['logs'] + "scat_files/" + dname, label, p0, timeaxis_for_fit, low_bounds, upp_bounds, sigma_for_fit,False,verbose,nwalkers,nsteps,discard,thin)
        t.add_done_callback(lambda future: future_callback_rns(future,dname,dname_result,outdir))

        return dname_samples,dname_result, dname_BIC,dname

    
    #define log likelihood function
    def log_likelihood(theta, x, y, yerr):
        f = theta[-1]
        model = scat.exp_gauss_n(x,*list(theta[:-1]))
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
    pos = np.array(initvals) + 1e-2 * np.random.randn(nwalkers,len(initvals))
    nwalkers,ndim = pos.shape

    #define sampler
    sampler = emcee.EnsembleSampler(nwalkers,ndim,log_probability, args=(timeaxis_for_fit,timeseries_for_fit,sigma_for_fit))

    #run sampler
    sampler.run_mcmc(pos,nsteps,progress=False)

    #flatten and get best fit params
    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    p_median = np.nanmedian(flat_samples,axis=0)
    p_upper = np.nanpercentile(flat_samples,84,axis=0) - p_median
    p_lower = p_median - np.nanpercentile(flat_samples,16,axis=0)

    #compute beyesian information criterion (BIC)
    BIC = ndim*np.log(len(timeaxis_for_fit)) - 2*log_likelihood(p_median,timeaxis_for_fit,timeseries_for_fit,sigma_for_fit)

    np.save(outdir + label + "_MCMC_samples.npy",flat_samples)
    np.save(outdir + label + "_MCMC_results.npy",np.array([p_median,p_upper,p_lower]))
    np.save(outdir + label + "_MCMC_BIC.npy",np.array([BIC]))

    return flat_samples,p_median,p_upper,p_lower,BIC


def specidx_fit_fn(x,Gamma,F0,nu0=1.4e3):
    """
    This function returns the flux assuming spectral index Gamma and flux F0 at 1.4 GHz. x should be given in MHz
    """

    return F0*((x/nu0)**(-Gamma))
