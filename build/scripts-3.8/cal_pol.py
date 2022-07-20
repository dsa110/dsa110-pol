import dsapol
import numpy as np
from matplotlib import pyplot as plt
import pylab
from sigpyproc import FilReader
import os
import sys
import pickle
import argparse


def main(Calvars,wdir,n_t,n_f,n_off,nsamps,RMvars,cal_opt,deg):
    #Script for calibrating and plotting FRB Polarization data
    ext = dsapol.ext
    #Read polarization data
    i_d =Calvars[0]#sys.argv[1] # e.g. "220319aaeb"
    name = Calvars[1]#sys.argv[2] # e.g. "mark"
    #calibrate = #int(sys.argv[3])
    #use_fit = int(sys.argv[4])

    label = i_d + "_" + name
    datadir = wdir + i_d + '/'
    sdir = datadir + i_d + name + '_dev'
    (I,Q,U,V,fobj,timeaxis,freq_test,wav_test) = dsapol.get_stokes_2D(datadir,i_d + '_dev',nsamps,n_t=n_t,n_f=n_f,n_off=n_off,sub_offpulse_mean=False)

    outdata = dict()

    timestart = 0
    timestop = nsamps//n_t
    print("Min and Max Samples: (" + str(timestart) + "," + str(timestop))

    outdata["max sample"] = timestop
    outdata["min sample"] = timestart
    outdata["peak width (samples)"] = timestop - timestart

    (I_t,Q_t,U_t,V_t) = get_stokes_vs_time(I,Q,U,V)
    (I_f,Q_f,U_f,V_f) = get_stokes_vs_freq(I,Q,U,V,width_native,t_samp)

    f=plt.figure(figsize=(12,6))
    plt.plot(I_t,label="I")
    plt.plot(Q_t,label="Q")
    plt.plot(U_t,label="U")
    plt.plot(V_t,label="V")
    plt.grid()
    plt.legend()
    plt.xlim(timestart,timestop)
    plt.xlabel("Time Sample (" + str(fobj.header.tsamp*n_t) + " s sampling time)")
    plt.title(label)
    plt.savefig(datadir +label + "_time_" + str(n_t) + "_binned" + ext)
    #plt.show()
    plt.close(f)

    f=plt.figure(figsize=(12,6))
    plt.plot(freq_test[0],I_f,label="I")
    plt.plot(freq_test[1],Q_f,label="Q")
    plt.plot(freq_test[2],U_f,label="U")
    plt.plot(freq_test[3],V_f,label="V")
    plt.grid()
    plt.xlabel("frequency (MHz)")
    #plt.xlim(1365,1375)
    plt.legend()
    plt.title(label)
    #plt.title(label)
    plt.savefig(datadir + label + "_frequency_" + str(n_f) + "_binned" + ext)
    #plt.show()
    plt.close(f)

    #Dynamic Spectra 
    lim = 500
    f=plt.figure(figsize=(25,15))
    #plt.title(label)
    pylab.subplot(2,2,1)
    plt.imshow(I - np.mean(I,1,keepdims=True),aspect="auto",vmin=-lim,vmax=lim)
    plt.xlim(timestart-10,timestop+10)
    plt.title(label + " I")
    plt.colorbar()
    plt.xlabel("Time Sample (" + str(fobj.header.tsamp*n_t) + " s sampling time)")
    plt.ylabel("frequency sample")
    #plt.yticks(np.around(freq_test[0][::25]),1)
    #plt.xticks(ticks=np.arange(0,n_ints,50),labels=np.around(ra_deg[::50],1))
    #plt.show()

    #plt.figure(figsize=(12,6))
    pylab.subplot(2,2,2)
    plt.imshow(Q - np.mean(Q,1,keepdims=True),aspect="auto",vmin=-lim,vmax=lim)
    plt.xlim(timestart-10,timestop+10)
    plt.title(label + " Q")
    plt.colorbar()
    plt.xlabel("Time Sample (" + str(fobj.header.tsamp*n_t) + " s sampling time)")
    plt.ylabel("frequency sample")
    #plt.show()

    #plt.figure(figsize=(12,6))
    pylab.subplot(2,2,3)
    plt.imshow(U - np.mean(U,1,keepdims=True),aspect="auto",vmin=-lim,vmax=lim)
    plt.xlim(timestart-10,timestop+10)
    plt.title(label + " U")
    plt.colorbar()
    plt.xlabel("Time Sample (" + str(fobj.header.tsamp*n_t) + " s sampling time)")
    plt.ylabel("frequency sample")
    #plt.show()

    #plt.figure(figsize=(12,6))
    pylab.subplot(2,2,4)
    plt.imshow(V - np.mean(V,1,keepdims=True),aspect="auto",vmin=-lim,vmax=lim)
    plt.xlim(timestart-10,timestop+10)
    plt.title(label + " V")
    plt.colorbar()
    plt.xlabel("Time Sample (" + str(fobj.header.tsamp*n_t) + " s sampling time)")
    plt.ylabel("frequency sample")
    #plt.show()

    outdata["Dynamic Spectra"] = dict()
    outdata["Dynamic Spectra"]["I"] = I - np.mean(I,1,keepdims=True)
    outdata["Dynamic Spectra"]["Q"] = Q - np.mean(Q,1,keepdims=True)
    outdata["Dynamic Spectra"]["U"] = U - np.mean(U,1,keepdims=True)
    outdata["Dynamic Spectra"]["V"] = V - np.mean(V,1,keepdims=True)

    plt.savefig(datadir + label + "_freq-time_" + str(n_f) + "_binned" + ext)
    #plt.show()
    plt.close(f) 

    #Polarization Fraction and Polarization Angle
    f_f = np.sqrt((np.array(Q_f)**2 + np.array(U_f)**2 + np.array(V_f)**2)/(np.array(I_f)**2))
    f=plt.figure(figsize=(12,6))
    plt.plot(freq_test[0],f_f)
    plt.grid()
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Polarization Fraction")
    plt.ylim(-1,1)
    plt.title(label)
    plt.savefig(datadir + label + "_polfraction_frequency_"    + str(n_f) + "_binned" + ext)
    #plt.show()
    plt.close(f)

    f_t = np.sqrt((np.array(Q_t)**2 + np.array(U_t)**2 + np.array(V_t)**2)/(np.array(I_t)**2))
    f=plt.figure(figsize=(12,6))
    plt.plot(np.arange(timestart,timestop),f_t[timestart:timestop])
    plt.grid()
    plt.xlabel("Time Sample (" + str(fobj.header.tsamp*n_t) + " s sampling time)")
    plt.ylabel("Polarization Fraction")
    plt.ylim(-1,1)
    plt.title(label)
    plt.savefig(datadir + label + "_polfraction_time_"  + str(n_f) + "_binned" + ext)
    #plt.xlim(timestart,timestop)
    #plt.show()
    plt.close(f)

    PA_f = np.angle(Q_f +1j*U_f)
    f=plt.figure(figsize=(12,6))
    plt.plot(freq_test[0],PA_f)
    plt.grid()
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Polarization Angle (rad)")
    #plt.ylim(-1,1)
    plt.title(label)
    plt.savefig(datadir + label + "_polangle_frequency_"  + str(n_f) + "_binned" + ext)
    #plt.show()
    plt.close(f)

    PA_t = np.angle(Q_t +1j*U_t)
    f=plt.figure(figsize=(12,6))
    plt.plot(np.arange(timestart,timestop),PA_t[timestart:timestop])
    plt.grid()
    plt.xlabel("Time Sample (" + str(fobj.header.tsamp*n_t) + " s sampling time)")
    plt.ylabel("Polarization Angle (rad)")
    #plt.ylim(-1,1)
    plt.title(label)
    plt.savefig(datadir + label + "_polangle_time_"  + str(n_f) + "_binned" + ext)
    #plt.xlim(timestart,timestop)
    #plt.show()
    plt.close(f)

    #Calculate Average polarization and PA
    avg_pol = (np.mean(f_t[timestart:timestop][f_t[timestart:timestop]<1]))
    avg_PA = np.mean(PA_t[timestart:timestop][f_t[timestart:timestop]<1])

    print("Average polarization (%) " + str(avg_pol*100))
    print("Average polarization angle (rad) " +str(avg_PA))

    outdata["Average Polarization (%)"] = avg_pol*100 
    outdata["Average Polarization Angle (rad)"] = avg_PA 


    #Gain or Phase Cal
    if cal_opt == "gain":
        print("Calculating Gain Calibration ratio...")
        ratio = dsapol.gaincal(I_f,Q_f,U_f,V_f)
        print("Fitting to " + str(deg) + " order polynomial...")
        popt = np.polyfit(freq_test[0],np.nan_to_num(ratio,nan=np.nanmedian(ratio)),deg=deg)
        print(popt)
        ratio_fit = np.zeros(np.shape(freq_test[0]))
        for i in range(deg+1):
            ratio_fit += popt[i]*(freq_test[0]**(deg-i))
        outdata["ratio"] = ratio
        outdata["ratio median"] = np.nanmedian(ratio)
        print("Median: " + str(np.nanmedian(ratio)))
        outdata["ratio fit params"] = popt
        
        f= plt.figure()
        plt.title(r'Gain Ratio ($g_{xx}/g_{yy}$) ' + label)
        plt.plot(freq_test[0],ratio,label="Calculated")
        plt.plot(freq_test[0],ratio_fit,label="Fit")
        plt.axhline(np.nanmedian(ratio),color="red",label="median")
        plt.legend()
        plt.xlabel("Frequency (MHz)")
        plt.grid()
        plt.savefig(datadir + "C348_calibrator_gain_ratio_" + label + ext)


    elif cal_opt == "phase":
        print("Calculating Phase Calibration difference...")
        phidiff = dsapol.phasecal(I_f,Q_f,U_f,V_f)
        print("Fitting to " + str(deg) + " order polynomial...")
        popt = np.polyfit(freq_test[0],np.nan_to_num(phidiff,nan=np.nanmedian(phidiff)),deg=deg)
        print(popt)
        phidiff_fit = np.zeros(np.shape(freq_test[0]))
        for i in range(deg+1):
            phidiff_fit += popt[i]*(freq_test[0]**(deg-i))
        outdata["phidiff"] = phidiff
        outdata["phidiff median"] = np.nanmedian(phidiff)
        print("Median: " + str(np.nanmedian(phidiff)) + " rad")
        outdata["phidiff fit params"] = popt

        f=plt.figure()
        plt.title(r'Phase Difference ($\phi_{xx} - \phi_{yy}$) ' + label )
        plt.plot(freq_test[0],phidiff,label="Calculated")
        plt.plot(freq_test[0],phidiff_fit,label="Fit")
        plt.axhline(np.nanmedian(phidiff),color="red",label="median")
        plt.legend()
        plt.xlabel("Frequency (MHz)")
        plt.grid()
        plt.savefig(datadir + "C3286_calibrator_phase_ratio_" + label + ext)
        plt.close(f)
    else:
        print("Invalid Calibration Option, must be gain or phase")
        return -1


    if RMvars != None:
        #Estimate Faraday RM
        #trial_RM = np.linspace(-2000,2000,10000)
        trial_RM = np.linspace(RMvars[0],RMvars[1],int(RMvars[2]))
        trial_phi = np.zeros(1)
        (RM,phi,SNRs) = dsapol.faradaycal(np.array(I_f),np.array(Q_f),np.array(U_f),np.array(V_f),np.array(wav_test[0]),trial_RM,trial_phi,plot=False,datadir=datadir,calstr="",label=label,n_f=n_f,n_t=n_t)

        f=plt.figure()
        plt.grid()
        plt.plot(trial_RM,SNRs)
        plt.xlabel("Trial RM")
        plt.ylabel("SNR")
        plt.title(label)
        plt.savefig(datadir + label + "_faraday1D_" + str(n_f) + "_binned" + ext)
        plt.close()

        print("RM Estimate: " + str(RM) + "rad/m^2")
        print("Phase Difference Estimate: "  + str(phi) + "rad")
        outdata["RM"] = RM
        outdata["RM Spectrum"] = (trial_RM,SNRs)

    #save to file
    f = open(datadir + label + "_outdata.pkl","wb")
    pickle.dump(outdata,f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot Polarization Data for a given FRB candidate')

    #Candidate ID and name
    parser.add_argument('-obs',  dest="Calvars",action='store',nargs=2,required=True,type=str,help="Calibration source name and observation, e.g. 3C286 jqc")

    #Gain or Phase cal
    parser.add_argument('-cal_opt',  dest="cal_opt",action='store',nargs=1,required=True,type=str,help="Calibration type (gain or phase)",choices=["gain","phase"])

    #Working directory
    parser.add_argument('-wdir', dest="wdir",action='store',nargs=1,default=['/home/ubuntu/sherman/scratch_weights_update_2022-06-03/'],type=str,help="Working directory where FRB data folders are located")

    #Time binning
    parser.add_argument('-n_t', dest="n_t",action='store',nargs=1,default=[1],type=int,help="Number of time samples to average over")

    #Freq binning
    parser.add_argument('-n_f', dest="n_f",action='store',nargs=1,default=[1],type=int,help="Number of frequency samples to average over")

    #Last off pulse sample
    parser.add_argument('-n_off', dest="n_off",action='store',nargs=1,default=[3000],type=int,help="Last off-pulse sample to calculate off-pulse mean and standard deviation (not needded for cal)")

    #Number of samples to read in
    parser.add_argument('-nsamps', dest="nsamps",action='store',nargs=1,default=[5120],type=int,help="Number of time samples to read in")

    #Trial RM
    parser.add_argument('-RM', dest="RM",action='store',nargs=3,default=None,type=int,help="Minimum RM, Maximum RM, Number of RM trials (None if don't want to calculate RM)")

    #Degree of polynomial fit
    parser.add_argument('-deg', dest="deg",action='store',nargs=1,default=[10],type=int,help="Degree of Polynomial fit")


    args = parser.parse_args()
    print(args.n_t,args.n_f)

    main(args.Calvars,args.wdir[0],args.n_t[0],args.n_f[0],args.n_off[0],args.nsamps[0],args.RM,args.cal_opt[0],args.deg[0])
