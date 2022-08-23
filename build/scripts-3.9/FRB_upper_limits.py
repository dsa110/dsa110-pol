from dsapol import dsapol
import json
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit

#fitting function
def gaussian(x,amp,mean,sigma):
    return amp*np.exp(-0.5*((x-mean)**2)/sigma**2)

ids=["220121aaat", "220204aaai", "220207aabh", "220208aaaa", "220307aaae", "220310aaam", "220319aaeb", "220330aaan", "220418aaai", "220424aabq", "220506aabd", "220426aaaw", "220726aabn", "220801aabd"]
nicknames=["clare", "fen", "zach", "ishita" ,"alex", "whitney" ,"mark", "erdos", "quincy" ,"davina", "oran", "jackie", "gertrude","augustine"]
widths=[8,4,2,16,2,4,1,32,4,2,2,2,4,4]
plot=True

#Get gain and phase cal errors and generate sample |gxx|/|gyy| and phi_xx-phi_yy

trials = 1000
conf = 0.95
extra = "_NEWPADDING"

#gain (default 3C48ane)
datadir_gain = '/home/ubuntu/sherman/scratch_weights_update_2022-06-03/3C48/'
source_name = '3C48'
obs_name = 'ane'
IMAGELABEL_GAIN=source_name + obs_name + "TEST"
suffix='_dev'
n_t=1
n_f=32
nsamps=20480
deg=10

label = source_name + obs_name + suffix
sdir = datadir_gain + label
(I,Q,U,V,fobj,timeaxis,freq_test,wav_test) = dsapol.get_stokes_2D(datadir_gain,label,nsamps,n_t=n_t,n_f=n_f,n_off=-1,sub_offpulse_mean=False)
(I_f,Q_f,U_f,V_f) = dsapol.get_stokes_vs_freq(I,Q,U,V,-1,fobj.header.tsamp,n_f,n_t,freq_test,datadir=datadir_gain,label=label,plot=False,show=False,normalize=False)

#errors in stokes parameters
I_err_f = I.std(1)/np.sqrt(I.shape[1])
Q_err_f = Q.std(1)/np.sqrt(Q.shape[1])
U_err_f = U.std(1)/np.sqrt(U.shape[1])
V_err_f = V.std(1)/np.sqrt(V.shape[1])

#generate random IQUV assuming Gaussian distribution
I_samps = []
Q_samps = []
U_samps = []
V_samps = []

for i in range(len(I_f)):
    I_samps.append(np.random.normal(I_f[i],I_err_f[i],trials))
    Q_samps.append(np.random.normal(Q_f[i],Q_err_f[i],trials))
    U_samps.append(np.random.normal(U_f[i],U_err_f[i],trials))
    V_samps.append(np.random.normal(V_f[i],V_err_f[i],trials))   
I_samps = np.array(I_samps)
Q_samps = np.array(Q_samps)
U_samps = np.array(U_samps)
V_samps = np.array(V_samps)

#generate random |gxx|/|gyy| (fit and no fit)
#Get samples of ratio
ratio_samps = []
ratio_samps_fit = []
for i in range(trials):
    ratio,ratio_fit_params = dsapol.gaincal(I_samps[:,i],Q_samps[:,i],U_samps[:,i],V_samps[:,i],freq_test,stokes=True,deg=deg,datadir=datadir_gain,label=label,plot=False,show=False)
    ratio_samps.append(ratio)
    ratio_fit = np.zeros(np.shape(freq_test[0]))
    for i in range(deg+1):
        ratio_fit += ratio_fit_params[i]*(freq_test[0]**(deg-i))
    ratio_use = ratio_fit
    
    ratio_samps_fit.append(ratio_use)
ratio_samps = np.array(ratio_samps)
ratio_samps_fit = np.array(ratio_samps_fit)


#phase
datadir_phase = '/home/ubuntu/sherman/scratch_weights_update_2022-06-03/3C286/'
source_name = '3C286'
obs_name = 'jqc'
IMAGELABEL_PHASE=source_name + obs_name + "TEST"
suffix='_dev'
n_t=1
n_f=32
nsamps=20480
deg=10

label = source_name + obs_name + suffix
sdir = datadir_phase + label
(I,Q,U,V,fobj,timeaxis,freq_test,wav_test) = dsapol.get_stokes_2D(datadir_phase,label,nsamps,n_t=n_t,n_f=n_f,n_off=-1,sub_offpulse_mean=False)
(I_f,Q_f,U_f,V_f) = dsapol.get_stokes_vs_freq(I,Q,U,V,-1,fobj.header.tsamp,n_f,n_t,freq_test,datadir=datadir_phase,label=label,plot=False,show=False,normalize=False)

#errors in stokes parameters
I_err_f = I.std(1)/np.sqrt(I.shape[1])
Q_err_f = Q.std(1)/np.sqrt(Q.shape[1])
U_err_f = U.std(1)/np.sqrt(U.shape[1])
V_err_f = V.std(1)/np.sqrt(V.shape[1])


#generate random IQUV assuming Gaussian distribution
I_samps = []
Q_samps = []
U_samps = []
V_samps = []

for i in range(len(I_f)):
    I_samps.append(np.random.normal(I_f[i],I_err_f[i],trials))
    Q_samps.append(np.random.normal(Q_f[i],Q_err_f[i],trials))
    U_samps.append(np.random.normal(U_f[i],U_err_f[i],trials))
    V_samps.append(np.random.normal(V_f[i],V_err_f[i],trials))
I_samps = np.array(I_samps)
Q_samps = np.array(Q_samps)
U_samps = np.array(U_samps)
V_samps = np.array(V_samps)

#generate random phase diff (fit and no fit)
#Get samples of phase
phase_samps = []
phase_samps_fit = []
for i in range(trials):
    phase,phase_fit_params = dsapol.phasecal(I_samps[:,i],Q_samps[:,i],U_samps[:,i],V_samps[:,i],freq_test,stokes=True,deg=deg,datadir=datadir_phase,label=label,plot=False,show=False)
    phase_samps.append(phase)
    phase_fit = np.zeros(np.shape(freq_test[0]))
    for i in range(deg+1):
        phase_fit += phase_fit_params[i]*(freq_test[0]**(deg-i))
    phase_use = phase_fit
    phase_samps_fit.append(phase_use)
phase_samps = np.array(phase_samps)
phase_samps_fit = np.array(phase_samps_fit)


if plot:
    f= plt.figure()

    for i in range(trials):
        if i ==0:
            plt.plot(freq_test[0],ratio_samps[i,:],'--',color='gray',linewidth=0.5,alpha=0.5,label="Trial gxx/gyy (no fit)")
            plt.plot(freq_test[0],ratio_samps_fit[i,:],'--',color='blue',linewidth=0.5,alpha=0.5,label="Trial gxx/gyy (fit)")
        else:
            plt.plot(freq_test[0],ratio_samps[i,:],'--',color='gray',linewidth=0.5,alpha=0.5)
            plt.plot(freq_test[0],ratio_samps_fit[i,:],'--',color='blue',linewidth=0.5,alpha=0.5)
    plt.plot(freq_test[0],ratio_samps.mean(0),color="purple",label="Trial Average gxx/gyy (no fit)")
    plt.plot(freq_test[0],ratio_samps.mean(0),color="orange",label="Trial Average gxx/gyy (fit)")
    #plt.plot(freq_test[0],ratio_samps,color='red',label="True Fraction")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel(r'$g_{xx}/g_{yy}$')
    #plt.xlim(3812,3816)
    plt.ylim(0,2)
    plt.legend()
    plt.savefig(datadir_gain + "/UPPERLIMIT_TRIALS_GAIN_" + IMAGELABEL_GAIN + extra + ".pdf")
    plt.close(f)

    f=plt.figure()

    for i in range(trials):
        if i ==0:
            plt.plot(freq_test[0],phase_samps[i,:],'--',color='gray',linewidth=0.5,alpha=0.5,label="Trial phase(gxx)-phase(gyy) (no fit)")
            plt.plot(freq_test[0],phase_samps_fit[i,:],'--',color='blue',linewidth=0.5,alpha=0.5,label="Trial phase(gxx)-phase(gyy) (fit)")
        else:
            plt.plot(freq_test[0],phase_samps[i,:],'--',color='gray',linewidth=0.5,alpha=0.5)
            plt.plot(freq_test[0],phase_samps_fit[i,:],'--',color='blue',linewidth=0.5,alpha=0.5)
    plt.plot(freq_test[0],phase_samps.mean(0),color="purple",label="Trial Average phase(gxx)-phase(gyy) (no fit)")
    plt.plot(freq_test[0],phase_samps.mean(0),color="orange",label="Trial Average phase(gxx)-phase(gyy) (fit)")
    #plt.plot(freq_test[0],ratio_samps,color='red',label="True Fraction")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel(r'$\phi_{xx}-\phi{yy}$')
    #plt.xlim(3812,3816)
    plt.ylim(-2*np.pi,2*np.pi)
    plt.legend()
    plt.savefig(datadir_phase + "/UPPERLIMIT_TRIALS_PHASE_" + IMAGELABEL_PHASE + extra + "--2.pdf")
    plt.close(f)


#Estimate errors in Jones parameters with gaussian fit
ratio_means = np.mean(ratio_samps,axis=0)
ratio_means_fit = np.mean(ratio_samps_fit,axis=0)
(hist,b) = np.histogram(ratio_means,np.linspace(0.5,1.5,50))
(hist_fit,b_fit) = np.histogram(ratio_means_fit,np.linspace(0.5,1.5,50))

popt,pcov = curve_fit(gaussian,b[:-1] + (b[2]-b[1])/2, hist,p0=[25,1.0,0.1])

print("Gain ratio estimate (no fit): "  + str(popt[1]))
print("Error estimate (no fit): "  + str(popt[2]))
popt_fit,pcov_fit = curve_fit(gaussian,b_fit[:-1] + (b_fit[2]-b_fit[1])/2, hist_fit,p0=[25,1.0,0.1])
print("Gain ratio estimate (fit): " + str(popt[1]))
print("Error estimate (fit): " + str(popt[2]))

MEANGAIN = popt[1]
ERRGAIN = popt[2]
MEANGAIN_FIT = popt_fit[1]
ERRGAIN_FIT = popt_fit[2]

if plot:
    f =plt.figure()
    p0=plt.plot(np.linspace(0.5,1.5,100), gaussian(np.linspace(0.5,1.5,100),popt[0],popt[1],popt[2]),label="No Fit Gaussian")
    p1=plt.plot(np.linspace(0.5,1.5,100), gaussian(np.linspace(0.5,1.5,100),popt_fit[0],popt_fit[1],popt_fit[2]),label="Fit Gaussian")

    plt.hist(ratio_means,np.linspace(0.5,1.5,50),color=p0[0].get_color(),alpha=0.5,label="No Fit")
    plt.hist(ratio_means_fit,np.linspace(0.5,1.5,50),color=p1[0].get_color(),alpha=0.5,label="Fit")

    #plt.axvline(Upperlimit,ls='--',color=p0[0].get_color(),label=r'$\sigma = $' + str(np.around(popt[2],2)) + ' (no fit)')
    #plt.axvline(Upperlimit_fit,ls='--',color=p1[0].get_color(),label=r'$\sigma = $' + str(np.around(popt_fit[2],2)) + ' (no fit)')
    #plt.plot(b[:-1] + (b[2]-b[1])/2,hist,label)
    plt.xlabel(r'$|g_{xx}|/|g_{yy}|$')
    plt.legend()
    plt.savefig(datadir_gain + "/UPPERLIMIT_GAIN_" + IMAGELABEL_GAIN + extra + ".pdf")
    plt.close(f)


phase_means = np.mean(phase_samps,axis=0)
phase_means_fit = np.mean(phase_samps_fit,axis=0)
(hist,b) = np.histogram(phase_means,np.linspace(0,np.pi,20))
(hist_fit,b_fit) = np.histogram(phase_means_fit,np.linspace(0,np.pi,20))

popt,pcov = curve_fit(gaussian,b[:-1] + (b[2]-b[1])/2, hist,p0=[80,2,0.1])
print("Phase difference estimate (no fit): "  + str(popt[1]) + " rad")
print("Error estimate (no fit): "  + str(popt[2]) + " rad")

popt_fit,pcov_fit = curve_fit(gaussian,b_fit[:-1] + (b_fit[2]-b_fit[1])/2, hist_fit,p0=[80,2,0.1])
print("Phase difference estimate (fit): " + str(popt[1]) + " rad")
print("Error estimate (fit): " + str(popt[2]) + " rad")

MEANPHASE = popt[1]
ERRPHASE = popt[2]
MEANPHASE_FIT = popt_fit[1]
ERRPHASE_FIT = popt_fit[2]

if plot:
    f=plt.figure()
    p0=plt.plot(np.linspace(0,np.pi,100), gaussian(np.linspace(0,np.pi,100),popt[0],popt[1],popt[2]),label="No Fit Gaussian")
    p1=plt.plot(np.linspace(0,np.pi,100), gaussian(np.linspace(0,np.pi,100),popt_fit[0],popt_fit[1],popt_fit[2]),label="Fit Gaussian")

    plt.hist(phase_means,np.linspace(0,np.pi,20),color=p0[0].get_color(),alpha=0.5,label="No Fit")
    plt.hist(phase_means_fit,np.linspace(0,np.pi,20),color=p1[0].get_color(),alpha=0.5,label="Fit")

    #plt.axvline(Upperlimit,ls='--',color=p0[0].get_color(),label=r'$\sigma = $' + str(popt[2]) + ' (no fit)')
    #plt.axvline(Upperlimit_fit,ls='--',color=p1[0].get_color(),label=r'$\sigma = $' + str(popt_fit[2]) + ' (no fit)')
    #plt.plot(b[:-1] + (b[2]-b[1])/2,hist,label)
    plt.xlabel(r'$\phi_{xx} - \phi_{yy}$')
    plt.legend()
    plt.savefig(datadir_phase + "/UPPERLIMIT_PHASE_" + IMAGELABEL_PHASE + ".pdf")
    #plt.show()
    plt.close(f)


#Loop through FRBs
for i in [6]:#[0,1,3,5,6,7,8,10,13]:#range(len(nicknames)):
    outdict = dict() #for saving data to json file
    outdict["trials"] = trials
    outdict["gain_avg"] = MEANGAIN
    outdict["gain_err"] = ERRGAIN
    outdict["gain_avg_fit"] = MEANGAIN_FIT
    outdict["gain_err_fit"] = ERRGAIN_FIT


    outdict["phase_avg"] = MEANPHASE
    outdict["phase_err"] = ERRPHASE
    outdict["phase_avg_fit"] = MEANPHASE_FIT
    outdict["phase_err_fit"] = ERRPHASE_FIT

    datadir = '/home/ubuntu/sherman/scratch_weights_update_2022-06-03/' + ids[i] + "_" + nicknames[i] + "/"
    label=ids[i] + "_" + nicknames[i]#"220319aaeb_mark"
    IMAGELABEL=label + "TEST"
    suffix='_dev'
    n_t=1
    n_f=32
    nsamps=20480
    deg=10  
    buff = 0
    width = widths[i]

    #Calculate polarization fraction estimates
    (I,Q,U,V,fobj,timeaxis,freq_test,wav_test) = dsapol.get_stokes_2D(datadir,ids[i] + '_dev',nsamps,n_t=n_t,n_f=n_f,n_off=12000,sub_offpulse_mean=True)#("/home/ubuntu/sherman/scratch_weights_update_2022-06-03/220319aaeb_mark/","220319aaeb_dev",5120,n_f=32)
    (I_cal_f_sim,Q_cal_f_sim,V_cal_f_sim,U_cal_f_sim) = dsapol.get_stokes_vs_freq(I,Q,U,V,width,fobj.header.tsamp,n_f,n_t,freq_test,datadir=datadir,label=label,plot=False,show=False,buff=buff,normalize=True,weighted=True,n_t_weight=2,timeaxis=timeaxis,fobj=fobj,n_off=12000)
    peak,timestart,timestop=dsapol.find_peak(I, width, fobj.header.tsamp,n_t,buff=buff)

    p_all = []
    p_t_trials = []
    L_all = []
    L_t_trials = []
    C_all = []
    C_t_trials = []

    p_all_fit = []
    p_t_trials_fit = []
    L_all_fit = []
    L_t_trials_fit = []
    C_all_fit = []
    C_t_trials_fit = []

    for i in range(trials):
        (gxx,gyy) = dsapol.get_calmatrix_from_ratio_phasediff(ratio_samps[i,:],phase_samps[i,:]%(2*np.pi))
        (I_cal_sim,Q_cal_sim,V_cal_sim,U_cal_sim) = dsapol.calibrate(I,Q,U,V,(gxx,gyy),stokes=True)
        (I_cal_f_sim,Q_cal_f_sim,U_cal_f_sim,V_cal_f_sim) = dsapol.get_stokes_vs_freq(I_cal_sim,Q_cal_sim,U_cal_sim,V_cal_sim,width,fobj.header.tsamp,n_f,n_t,freq_test,datadir=datadir,label=label,plot=False,show=False,buff=buff,normalize=True,weighted=True,n_t_weight=2,timeaxis=timeaxis,fobj=fobj,n_off=12000)
        (I_cal_t_sim,Q_cal_t_sim,U_cal_t_sim,V_cal_t_sim) = dsapol.get_stokes_vs_time(I_cal_sim,Q_cal_sim,U_cal_sim,V_cal_sim,width,fobj.header.tsamp,n_t,datadir=datadir,label=label,plot=False,show=False,buff=buff,normalize=True,n_off=12000)

        [(p_f,p_t,avg,sigma_frac),(L_f,L_t,avg_L,sigma_L),(C_f,C_t,avg_C,sigma_C)] = dsapol.get_pol_fraction(I_cal_sim,Q_cal_sim,U_cal_sim,V_cal_sim,width,fobj.header.tsamp,1,32,freq_test,pre_calc_tf=False,show=False,buff=buff,normalize=True,weighted=True,n_t_weight=2,timeaxis=timeaxis,fobj=fobj,n_off=12000)
        #p_t = np.sqrt(Q_cal_t_sim**2 + U_cal_t_sim**2)/I_cal_t_sim
        #avg = np.mean(p_t[timestart:timestop])

        p_all.append(avg)
        p_t_trials.append(p_t)
        L_all.append(avg_L)
        L_t_trials.append(L_t)
        C_all.append(avg_C)
        C_t_trials.append(C_t)

        (gxx,gyy) = dsapol.get_calmatrix_from_ratio_phasediff(ratio_samps_fit[i,:],phase_samps_fit[i,:]%(2*np.pi))
        (I_cal_sim,Q_cal_sim,V_cal_sim,U_cal_sim) = dsapol.calibrate(I,Q,U,V,(gxx,gyy),stokes=True)
        (I_cal_f_sim,Q_cal_f_sim,U_cal_f_sim,V_cal_f_sim) = dsapol.get_stokes_vs_freq(I_cal_sim,Q_cal_sim,U_cal_sim,V_cal_sim,width,fobj.header.tsamp,n_f,n_t,freq_test,datadir=datadir,label=label,plot=False,show=False,buff=buff,normalize=True,weighted=True,n_t_weight=2,timeaxis=timeaxis,fobj=fobj,n_off=12000)
        (I_cal_t_sim,Q_cal_t_sim,U_cal_t_sim,V_cal_t_sim) = dsapol.get_stokes_vs_time(I_cal_sim,Q_cal_sim,U_cal_sim,V_cal_sim,width,fobj.header.tsamp,n_t,datadir=datadir,label=label,plot=False,show=False,buff=buff,normalize=True,n_off=12000)

        [(p_f,p_t,avg,sigma_frac),(L_f,L_t,avg_L,sigma_L),(C_f,C_t,avg_C,sigma_C)] = dsapol.get_pol_fraction(I_cal_sim,Q_cal_sim,U_cal_sim,V_cal_sim,width,fobj.header.tsamp,1,32,freq_test,pre_calc_tf=False,show=False,buff=buff,normalize=True,weighted=True,n_t_weight=2,timeaxis=timeaxis,fobj=fobj,n_off=12000)
        #p_t = np.sqrt(Q_cal_t_sim**2 + U_cal_t_sim**2)/I_cal_t_sim
        #avg = np.mean(p_t[timestart:timestop])

        p_all_fit.append(avg)
        p_t_trials_fit.append(p_t)
        L_all_fit.append(avg_L)
        L_t_trials_fit.append(L_t)
        C_all_fit.append(avg_C)
        C_t_trials_fit.append(C_t)


    p_t_trials = np.array(p_t_trials)
    p_t_trials_fit = np.array(p_t_trials_fit)
    L_t_trials = np.array(L_t_trials)
    L_t_trials_fit = np.array(L_t_trials_fit)
    C_t_trials = np.array(C_t_trials)
    C_t_trials_fit = np.array(C_t_trials_fit)


    if plot:
        f = plt.figure()
        for i in range(trials):
            if i ==0:
                plt.plot(p_t_trials[i,:],'--',color='gray',linewidth=0.5,alpha=0.5,label="Trial polarization fractions (no fit)")
                plt.plot(p_t_trials_fit[i,:],'--',color='blue',linewidth=0.5,alpha=0.5,label="Trial polarization fractions (fit)")
            else:
                plt.plot(p_t_trials[i,:],'--',color='gray',linewidth=0.5,alpha=0.5)
                plt.plot(p_t_trials_fit[i,:],'--',color='blue',linewidth=0.5,alpha=0.5)
        plt.plot(p_t_trials.mean(0),color="purple",label="Trial Average Fraction (no fit)")
        plt.plot(p_t_trials_fit.mean(0),color="orange",label="Trial Average Fraction (fit)")
        plt.plot(p_t,color='red',label="True Fraction")
        plt.xlabel("Time sample")
        plt.ylabel("Polarization Fraction")
        plt.xlim(timestart,timestop)
        plt.ylim(0,1.5)
        plt.legend()
        plt.savefig(datadir + "/UPPERLIMIT_TRIALS_" + IMAGELABEL + extra + ".pdf")
        plt.close(f)


    if plot:
        f = plt.figure()
        for i in range(trials):
            if i ==0:
                plt.plot(L_t_trials[i,:],'--',color='gray',linewidth=0.5,alpha=0.5,label="Trial polarization fractions (no fit)")
                plt.plot(L_t_trials_fit[i,:],'--',color='blue',linewidth=0.5,alpha=0.5,label="Trial polarization fractions (fit)")
            else:
                plt.plot(L_t_trials[i,:],'--',color='gray',linewidth=0.5,alpha=0.5)
                plt.plot(L_t_trials_fit[i,:],'--',color='blue',linewidth=0.5,alpha=0.5)
        plt.plot(L_t_trials.mean(0),color="purple",label="Trial Average Fraction (no fit)")
        plt.plot(L_t_trials_fit.mean(0),color="orange",label="Trial Average Fraction (fit)")
        plt.plot(L_t,color='red',label="True Fraction")
        plt.xlabel("Time sample")
        plt.ylabel("Linear Polarization Fraction")
        plt.xlim(timestart,timestop)
        plt.ylim(0,1.5)
        plt.legend()
        plt.savefig(datadir + "/UPPERLIMIT_LIN_TRIALS_" + IMAGELABEL + extra + ".pdf")
        plt.close(f)

    if plot:
        f = plt.figure()
        for i in range(trials):
            if i ==0:
                plt.plot(C_t_trials[i,:],'--',color='gray',linewidth=0.5,alpha=0.5,label="Trial polarization fractions (no fit)")
                plt.plot(C_t_trials_fit[i,:],'--',color='blue',linewidth=0.5,alpha=0.5,label="Trial polarization fractions (fit)")
            else:
                plt.plot(C_t_trials[i,:],'--',color='gray',linewidth=0.5,alpha=0.5)
                plt.plot(C_t_trials_fit[i,:],'--',color='blue',linewidth=0.5,alpha=0.5)
        plt.plot(C_t_trials.mean(0),color="purple",label="Trial Average Fraction (no fit)")
        plt.plot(C_t_trials_fit.mean(0),color="orange",label="Trial Average Fraction (fit)")
        plt.plot(C_t,color='red',label="True Fraction")
        plt.xlabel("Time sample")
        plt.ylabel("Circular Polarization Fraction")
        plt.xlim(timestart,timestop)
        plt.ylim(-1.5,1.5)
        plt.legend()
        plt.savefig(datadir + "/UPPERLIMIT_CIRC_TRIALS_" + IMAGELABEL + extra + ".pdf")
        plt.close(f)


    #Gaussian fit to get error and upper limit in polarization
    minbin=0
    maxbin=1#(np.nanmax(p_all))
    print(minbin,maxbin)
    nbins=int(100*(maxbin-minbin))
    (hist,b) = np.histogram(p_all,np.linspace(minbin,maxbin,nbins))
    popt,pcov = curve_fit(gaussian,b[:-1] + (b[2]-b[1])/2, hist,p0=[trials/4,np.mean(p_all),np.std(p_all)])
    Upperlimit = (norm.isf(1-conf,popt[1],np.abs(popt[2])))
    print("Polarization estimate (no fit): " + str(popt[1]))
    print("Error estimate (no fit): " + str(popt[2]))
    print(str(conf) + "% Upper Limit estimate (no fit): " + str(Upperlimit))
    outdict["pol_avg"] = popt[1]
    outdict["pol_err"] = popt[2]
    outdict["upper_lim"] = Upperlimit

    minbin_fit=0
    maxbin_fit=0.3#(np.nanmax(p_all_fit))
    nbins_fit=int(100*(maxbin_fit-minbin_fit))
    (hist_fit,b_fit) = np.histogram(p_all_fit,np.linspace(minbin_fit,maxbin_fit,nbins_fit))
    popt_fit,pcov_fit = curve_fit(gaussian,b_fit[:-1] + (b_fit[2]-b_fit[1])/2, hist_fit,p0=[trials/4,np.mean(p_all_fit),np.std(p_all_fit)])
    Upperlimit_fit = (norm.isf(1-conf,popt_fit[1],np.abs(popt_fit[2])))
    print("Polarization estimate (fit): " + str(popt_fit[1]))
    print("Error estimate (fit): " + str(popt_fit[2]))
    print(str(conf) + "% Upper Limit estimate (fit): " + str(Upperlimit_fit))
    outdict["pol_avg_fit"] = popt_fit[1]
    outdict["pol_err_fit"] = popt_fit[2]
    outdict["upper_lim_fit"] = Upperlimit_fit

    if plot:
        f=plt.figure(figsize=(12,6))
        p0=plt.plot(np.linspace(minbin,maxbin,10*nbins), gaussian(np.linspace(minbin,maxbin,10*nbins),popt[0],popt[1],popt[2]),label="No Fit Gaussian")
        p1=plt.plot(np.linspace(minbin_fit,maxbin_fit,10*nbins_fit), gaussian(np.linspace(minbin_fit,maxbin_fit,10*nbins_fit),popt_fit[0],popt_fit[1],popt_fit[2]),label="Fit Gaussian")
        plt.hist(p_all,np.linspace(minbin,maxbin,nbins),color=p0[0].get_color(),alpha=0.5,label="No Fit")
        plt.hist(p_all_fit,np.linspace(minbin_fit,maxbin_fit,nbins_fit),color=p1[0].get_color(),alpha=0.5,label="Fit")
        plt.axvline(Upperlimit,ls='--',color=p0[0].get_color(),label="95% Confidence Level (no fit)")
        plt.axvline(Upperlimit_fit,ls='--',color=p1[0].get_color(),label="95% Confidence Level (fit)")
        plt.xlabel("Polarization Fraction")
        plt.legend()
        plt.xlim(0,0.4)
        plt.savefig(datadir + "/UPPERLIMIT_" + IMAGELABEL + extra + ".pdf")
        #plt.show()
        plt.close(f)

    #Gaussian fit to get error and upper limit in linear polarization
    minbin=0
    maxbin=1#(np.nanmax(L_all))
    nbins=int(100*(maxbin-minbin))
    (hist,b) = np.histogram(L_all,np.linspace(minbin,maxbin,nbins))
    popt,pcov = curve_fit(gaussian,b[:-1] + (b[2]-b[1])/2, hist,p0=[trials/4,np.mean(L_all),np.std(L_all)])
    Upperlimit = (norm.isf(1-conf,popt[1],np.abs(popt[2])))
    print("Linear Polarization estimate (no fit): " + str(popt[1]))
    print("Error estimate (no fit): " + str(popt[2]))
    print(str(conf) + "% Upper Limit estimate (no fit): " + str(Upperlimit))
    outdict["lin_pol_avg"] = popt[1]
    outdict["lin_pol_err"] = popt[2]
    outdict["lin_upper_lim"] = Upperlimit

    minbin_fit=0
    maxbin_fit=1#(np.nanmax(L_all_fit))
    nbins_fit=int(100*(maxbin_fit-minbin_fit))
    (hist_fit,b_fit) = np.histogram(L_all_fit,np.linspace(minbin_fit,maxbin_fit,nbins_fit))
    popt_fit,pcov_fit = curve_fit(gaussian,b_fit[:-1] + (b_fit[2]-b_fit[1])/2, hist_fit,p0=[trials/4,np.mean(L_all_fit),np.std(L_all_fit)])
    Upperlimit_fit = (norm.isf(1-conf,popt_fit[1],np.abs(popt_fit[2])))
    print("Linear Polarization estimate (fit): " + str(popt_fit[1]))
    print("Error estimate (fit): " + str(popt_fit[2]))
    print(str(conf) + "% Upper Limit estimate (fit): " + str(Upperlimit_fit))
    outdict["lin_pol_avg_fit"] = popt_fit[1]
    outdict["lin_pol_err_fit"] = popt_fit[2]
    outdict["lin_upper_lim_fit"] = Upperlimit_fit

    if plot:
        f=plt.figure(figsize=(12,6))
        p0=plt.plot(np.linspace(minbin,maxbin,10*nbins), gaussian(np.linspace(minbin,maxbin,10*nbins),popt[0],popt[1],popt[2]),label="No Fit Gaussian")
        p1=plt.plot(np.linspace(minbin_fit,maxbin_fit,10*nbins_fit), gaussian(np.linspace(minbin_fit,maxbin_fit,10*nbins_fit),popt_fit[0],popt_fit[1],popt_fit[2]),label="Fit Gaussian")
        plt.hist(L_all,np.linspace(minbin,maxbin,nbins),color=p0[0].get_color(),alpha=0.5,label="No Fit")
        plt.hist(L_all_fit,np.linspace(minbin_fit,maxbin_fit,nbins_fit),color=p1[0].get_color(),alpha=0.5,label="Fit")
        plt.axvline(Upperlimit,ls='--',color=p0[0].get_color(),label="95% Confidence Level (no fit)")
        plt.axvline(Upperlimit_fit,ls='--',color=p1[0].get_color(),label="95% Confidence Level (fit)")
        plt.xlabel("Linear Polarization Fraction")
        plt.legend()
        plt.xlim(0,0.4)
        plt.savefig(datadir + "/UPPERLIMIT_LIN_" + IMAGELABEL + extra + ".pdf")
        #plt.show()
        plt.close(f)


    #Gaussian fit to get error and upper limit in circular polarization
    minbin=-1#(np.nanmin(C_all))
    maxbin=1#(np.nanmax(C_all))
    nbins=int(100*(maxbin-minbin))
    (hist,b) = np.histogram(C_all,np.linspace(minbin,maxbin,nbins))
    popt,pcov = curve_fit(gaussian,b[:-1] + (b[2]-b[1])/2, hist,p0=[trials/4,np.mean(C_all),np.std(C_all)])
    Upperlimit = (norm.isf(1-conf,popt[1],np.abs(popt[2])))
    print("Circular Polarization estimate (no fit): " + str(popt[1]))
    print("Error estimate (no fit): " + str(popt[2]))
    print(str(conf) + "% Upper Limit estimate (no fit): " + str(Upperlimit))
    outdict["circ_pol_avg"] = popt[1]
    outdict["circ_pol_err"] = popt[2]
    outdict["circ_upper_lim"] = Upperlimit

    minbin_fit=(np.nanmin(C_all_fit))
    maxbin_fit=(np.nanmax(C_all_fit))
    nbins_fit=int(100*(maxbin_fit-minbin_fit))
    (hist_fit,b_fit) = np.histogram(C_all_fit,np.linspace(minbin_fit,maxbin_fit,nbins_fit))
    popt_fit,pcov_fit = curve_fit(gaussian,b_fit[:-1] + (b_fit[2]-b_fit[1])/2, hist_fit,p0=[trials/4,np.mean(C_all_fit),np.std(C_all_fit)])
    Upperlimit_fit = (norm.isf(1-conf,popt_fit[1],np.abs(popt_fit[2])))
    print("Circular Polarization estimate (fit): " + str(popt_fit[1]))
    print("Error estimate (fit): " + str(popt_fit[2]))
    print(str(conf) + "% Upper Limit estimate (fit): " + str(Upperlimit_fit))
    outdict["circ_pol_avg_fit"] = popt_fit[1]
    outdict["circ_pol_err_fit"] = popt_fit[2]
    outdict["circ_upper_lim_fit"] = Upperlimit_fit

    if plot:
        f=plt.figure(figsize=(12,6))
        p0=plt.plot(np.linspace(minbin,maxbin,10*nbins), gaussian(np.linspace(minbin,maxbin,10*nbins),popt[0],popt[1],popt[2]),label="No Fit Gaussian")
        p1=plt.plot(np.linspace(minbin_fit,maxbin_fit,10*nbins_fit), gaussian(np.linspace(minbin_fit,maxbin_fit,10*nbins_fit),popt_fit[0],popt_fit[1],popt_fit[2]),label="Fit Gaussian")
        plt.hist(C_all,np.linspace(minbin,maxbin,nbins),color=p0[0].get_color(),alpha=0.5,label="No Fit")
        plt.hist(C_all_fit,np.linspace(minbin_fit,maxbin_fit,nbins_fit),color=p1[0].get_color(),alpha=0.5,label="Fit")
        plt.axvline(Upperlimit,ls='--',color=p0[0].get_color(),label="95% Confidence Level (no fit)")
        plt.axvline(Upperlimit_fit,ls='--',color=p1[0].get_color(),label="95% Confidence Level (fit)")
        plt.xlabel("Circular Polarization Fraction")
        plt.legend()
        plt.xlim(-0.4,0.4)
        plt.savefig(datadir + "/UPPERLIMIT_CIRC_" + IMAGELABEL + extra + ".pdf")
        #plt.show()
        plt.close(f)






    #save to json
    fname_json = datadir + label + "_upperlimit_out" + extra + ".json"
    print("Writing output data to " + fname_json)
    with open(fname_json, "w") as outfile:
        json.dump(outdict, outfile)
