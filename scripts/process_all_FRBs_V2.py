from dsapol import dsapol
import numpy as np
import sys
from matplotlib import pyplot as plt

import json
ids=["220121aaat", "220204aaai", "220207aabh", "220208aaaa", "220307aaae", "220310aaam", "220330aaan", "220418aaai", "220424aabq", "220506aabd", "220426aaaw","220319aaeb", "220726aabn", "220801aabd", "220411aabk", "220825aaad", "220831aaaj", "220914aabz"]
nicknames=["clare", "fen", "zach", "ishita" ,"alex", "whitney" , "erdos", "quincy" ,"davina", "oran", "jackie","mark","gertrude", "augustine", "maya", "ansel", "ada", "elektra"]
widths=[8,4,2,16,2,4,32,4,2,2,2,1,4,4,32,4,8,2]
DMs=[313.5,612.2,262.3,437.0,499.15,462.15,468.1,623.45,863.35,396.93,269.5,110.95,686.55,413.0,150,651.2,1146.25,631.05]
n_ts = [2,1,1,2,1,1,4,1,1,1,1,1,1,1,1,1,1,1]
RMs = [-1305.55086886348,2.79916683201975,127.199370848151,761.776036230955,-958.800918992897,15.2004530880001,-227.211015168586,64.0126035205116,178.00873672044,-46.800006992007,-140.399224816135,-575098.97219299,463.1961241282585 ,245387.033404836,0,741.199188688634,809.5977443846867,0]
RM_gals = [-68.17985557759044,-7.173808637692105,3.4538780715571686,-6.750317532361379,-14.968536640281991,-14.920449670882448,-15.576725577586794,4.84899789477938,-39.1415848090991,1.0584624922027939,7.736806347620126,-3.2924384810295937,-50.258010868238365,5.662006457588828,-0.013026056420469034,10.047789988844018,-9.21574742050912,0]
#ids = ids[2:]
#nicknames = nicknames[2:]
#widths = widths[2:]
#DMs = DMs[2:]


#defaults
#zoom_window = 1000
#n_trial_RM_zoom = 5000
if len(sys.argv) > 1:
    idx = nicknames.index(sys.argv[1])
    ids = ids[idx:idx+1]
    nicknames = nicknames[idx:idx+1]
    widths = widths[idx:idx+1]

print(nicknames)
for i in range(len(nicknames)):

    print("BEFORE RM CAL")
    out_before = dsapol.FRB_plot_all(datadir="/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/" + str(ids[i]) +"_" + str(nicknames[i]) + "/",prefix=ids[i] + "_dev",nickname=nicknames[i],nsamps=20480,n_t=n_ts[i],n_f=32,n_off=int(12000//n_ts[i]),width_native=widths[i],cal=True,gain_dir='/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/3C48/',gain_source_name="3C48",gain_obs_names=["ane"],phase_dir='/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/3C286/',phase_source_name="3C286",phase_obs_names=["jqc"],deg=10,suffix="_dev",use_fit=True,get_RM=False,RM_cal=False,trial_RM=np.linspace(-1e6,1e6,int(1e6)),trial_phi=[0],n_trial_RM_zoom=5000,zoom_window=1000,fit_window=100,cal_2D=True,sub_offpulse_mean=True,window=10,buff=1,lim=3,DM=DMs[i],weighted=True,n_t_weight=1,use_sf=True,sfwindow=19,extra="_finalv2_before")
    print(out_before[1:])
    
    #RM cal
    out_after = dict()
    gain_dir = '/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/3C48/'
    gain_source_name = "3C48"
    gain_obs_names = ["ane"]
    n_t = n_ts[i]#4
    n_f = 1
    nsamps = 20480
    deg = 10
    suffix = "_dev"
    sfwindow=19

    phase_dir ="/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/3C286/"
    phase_source_name = "3C286"
    phase_obs_names = ["jqc"]
    ratio,ratio_fit_params,ratio_sf = dsapol.gaincal_full(datadir=gain_dir,source_name=gain_source_name,obs_names=gain_obs_names,n_t=n_t,n_f=n_f,nsamps=nsamps,deg=deg,suffix=suffix,average=True,plot=True,sfwindow=sfwindow)
    ratio_use = ratio_sf

    phase_diff,phase_fit_params,phase_sf = dsapol.phasecal_full(datadir=phase_dir,source_name=phase_source_name,obs_names=phase_obs_names,n_t=n_t,n_f=n_f,nsamps=nsamps,deg=deg,suffix=suffix,average=True,plot=True,sfwindow=sfwindow)
    phase_use = phase_sf

    (gxx,gyy) = dsapol.get_calmatrix_from_ratio_phasediff(ratio_use,phase_use)

    (I,Q,U,V,fobj,timeaxis,freq_test,wav_test) = dsapol.get_stokes_2D("/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/"+ids[i] + "_" + nicknames[i] + "/",ids[i] + "_dev",20480,n_t=n_ts[i],n_f=1,n_off=int(12000//n_ts[i]),sub_offpulse_mean=True)
    (peak,timestart,timestop) = dsapol.find_peak(I,widths[i],fobj.header.tsamp,n_t=n_ts[i],peak_range=None,pre_calc_tf=False)
    (I_cal,Q_cal,U_cal,V_cal) = dsapol.calibrate(I,Q,U,V,(gxx,gyy),stokes=True)
    (I_calf,Q_calf,U_calf,V_calf) = dsapol.get_stokes_vs_freq(I_cal,Q_cal,U_cal,V_cal,widths[i],fobj.header.tsamp,1,n_ts[i],freq_test,n_off=int(12000//n_ts[i]),plot=True,datadir="/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/"+ids[i] + "_" + nicknames[i] + "/",label=str(ids[i]) + "_dev_" + str(nicknames[i]),calstr="_highres6144chan_finalv2_",show=False,normalize=True,buff=1,weighted=True,n_t_weight=1,timeaxis=timeaxis,fobj=fobj)

    (RM,phi,SNRs,RMerr,p_val,B) = dsapol.faradaycal_full(I_cal,Q_cal,U_cal,V_cal,freq_test,np.linspace(-1e6,1e6,int(1e6)),[0],widths[i],fobj.header.tsamp,n_trial_RM_zoom=5000,zoom_window=1000,plot=True,datadir="/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/"+ids[i] + "_" + nicknames[i] + "/",calstr="_highres6144chan_finalv2_",label=str(ids[i]) + "_dev_" + str(nicknames[i]),n_f=1,n_t=n_ts[i],n_off=int(12000/n_ts[i]),show=False,fit_window=100,buff=1,normalize=True,DM=DMs[i],weighted=True,n_t_weight=1,timeaxis=timeaxis,fobj=fobj,RM_tools=True,trial_RM_tools=np.linspace(-1e6,1e6,int(1e4)))
    #(RM,phi,SNRs,RMerr,upper,lower,significance) = dsapol.faradaycal_SNR(I_cal,Q_cal,U_cal,V_cal,freq_test,np.linspace(RMs[i]-1000,RMs[i]+1000,5000),[0],widths[i],fobj.header.tsamp,plot=True,datadir="/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/"+ids[i] + "_" + nicknames[i] + "/",calstr="_highres6144chan_finalv2_",label=str(ids[i]) + "_dev_" + str(nicknames[i]),n_f=1,n_t=n_ts[i],show=False,err=True,buff=1)
    #RM =RM[i]
    print("RM: " + str(RM) + " pm " + str(RMerr))
    print("RM gal: " + str(RM_gals[i]))
    RM = RM - RM_gals[i]
    out_after["RM"] = RM
    out_after["RMerr"] = RMerr
    out_after["RMgal"] = RM_gals[i]
    n_off = int(12000//n_ts[i])
    sigma_q = np.mean(np.std(Q_cal[:,:n_off],axis=1))
    sigma_u = np.mean(np.std(U_cal[:,:n_off],axis=1))
    sigma = (sigma_q + sigma_u)/2

    wav2 = ((3e8)/(freq_test[0]*(1e6)))**2
    waverr = (np.max(wav2)-np.min(wav2))/(2*np.sqrt(3))#np.sqrt(np.sum(wavs**2)/(len(wav2)-1))
    chierr = 0.5*sigma/np.abs(np.mean((Q_calf + 1j*U_calf)*np.exp(-2*1j*RM*wav2)))
    RMerr2 = chierr/(waverr*np.sqrt(len(wav2)-1))
    print("Brentjens/deBruyn Error: " + str(RMerr2) + "rad/m^2")
    print("Peak SNR: " + str(np.abs(np.mean((Q_calf + 1j*U_calf)*np.exp(-2*1j*RM*wav2)))))

    print("AFTER RM CAL")
    out_after2 = dsapol.FRB_plot_all(datadir="/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/" + str(ids[i]) +"_" + str(nicknames[i]) + "/",prefix=ids[i] + "_dev",nickname=nicknames[i],nsamps=20480,n_t=n_ts[i],n_f=32,n_off=int(12000//n_ts[i]),width_native=widths[i],cal=True,gain_dir='/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/3C48/',gain_source_name="3C48",gain_obs_names=["ane"],phase_dir='/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/3C286/',phase_source_name="3C286",phase_obs_names=["jqc"],deg=10,suffix="_dev",use_fit=True,get_RM=False,RM_cal=True,trial_RM=np.linspace(-1e6,1e6,int(1e6)),trial_phi=[0],n_trial_RM_zoom=5000,zoom_window=1000,fit_window=100,cal_2D=True,sub_offpulse_mean=True,window=10,buff=1,lim=3,DM=DMs[i],weighted=True,n_t_weight=1,use_sf=True,sfwindow=19,extra="_finalv2_after",RM_in=RM)
    print(out_after2[1:])

    """

    #get 32 binned data
    gain_dir = '/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/3C48/'
    gain_source_name = "3C48"
    gain_obs_names = ["ane"]
    n_t = n_ts[i]#4
    n_f = 32
    nsamps = 20480
    deg = 10
    suffix = "_dev"
    sfwindow=19

    phase_dir ="/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/3C286/"
    phase_source_name = "3C286"
    phase_obs_names = ["jqc"]
    ratio,ratio_fit_params,ratio_sf = dsapol.gaincal_full(datadir=gain_dir,source_name=gain_source_name,obs_names=gain_obs_names,n_t=n_t,n_f=n_f,nsamps=nsamps,deg=deg,suffix=suffix,average=True,plot=True,sfwindow=sfwindow)
    ratio_use = ratio_sf

    phase_diff,phase_fit_params,phase_sf = dsapol.phasecal_full(datadir=phase_dir,source_name=phase_source_name,obs_names=phase_obs_names,n_t=n_t,n_f=n_f,nsamps=nsamps,deg=deg,suffix=suffix,average=True,plot=True,sfwindow=sfwindow)
    phase_use = phase_sf

    (gxx,gyy) = dsapol.get_calmatrix_from_ratio_phasediff(ratio_use,phase_use)

    (I,Q,U,V,fobj,timeaxis,freq_test,wav_test) = dsapol.get_stokes_2D("/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/"+ids[i] + "_" + nicknames[i] + "/",ids[i] + "_dev",20480,n_t=n_ts[i],n_f=32,n_off=int(12000//n_ts[i]),sub_offpulse_mean=True)
    (peak,timestart,timestop) = dsapol.find_peak(I,widths[i],fobj.header.tsamp,n_t=n_ts[i],peak_range=None,pre_calc_tf=False)
    (I_cal,Q_cal,U_cal,V_cal) = dsapol.calibrate(I,Q,U,V,(gxx,gyy),stokes=True)
    (I_cal,Q_cal,U_cal,V_cal) = dsapol.calibrate_RM(I_cal,Q_cal,U_cal,V_cal,RM,phi,freq_test,stokes=True) 

    #get new PA, fractions
    (PA_f,PA_t,avg_PA) = dsapol.get_pol_angle(I_cal,Q_cal,U_cal,V_cal,widths[i],fobj.header.tsamp,n_ts[i],32,freq_test,n_off=int(12000//n_ts[i]),plot=False,datadir="/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/"+ids[i] + "_" + nicknames[i] + "/",calstr="_highres6144chan_final_after",label=str(ids[i]) + "_dev_" + str(nicknames[i]),show=False,weighted=True,n_t_weight=1,timeaxis=timeaxis,fobj=fobj,buff=1)
    [(pol_f,pol_t,avg_frac,sigma_frac,snr_frac),(L_f,L_t,avg_L,sigma_L,snr_L),(C_f,C_t,avg_C,sigma_C,snr_C)]=dsapol.get_pol_fraction(I_cal,Q_cal,U_cal,V_cal,widths[i],fobj.header.tsamp,n_ts[i],32,freq_test,n_off=int(12000//n_ts[i]),plot=False,datadir="/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/"+ids[i] + "_" + nicknames[i] + "/",label=str(ids[i]) + "_dev_" + str(nicknames[i]),calstr="_highres6144chan_final_after_",show=False,normalize=True,buff=1,full=False,weighted=True,n_t_weight=1,timeaxis=timeaxis,fobj=fobj)
    """
    
    print("perpendicular B field: " + str(np.nanmean(((np.array(out_after2[0]["PA"]["time"])+ np.pi/2)%np.pi)[timestart:timestop])) + " pm " + str(np.nanstd(((np.array(out_after2[0]["PA"]["time"])+ np.pi/2)%np.pi)[timestart:timestop])))
    """
    print("AFTER RM CAL:")
    print("PA: " + str(avg_PA) + " pm " + str(np.nanstd(PA_t)))
    print("Polarization:")
    print((avg_frac,sigma_frac))
    print((avg_L,sigma_L))
    print((avg_C,sigma_C))

    out_after["PA"] = avg_PA
    out_after["PAerr"] = np.nanstd(PA_t)
    out_after["pol"] = avg_frac
    out_after["polerr"] = sigma_frac
    out_after["lin"] = avg_L
    out_after["linerr"] = sigma_L
    """
    #Plot
    f=plt.figure(figsize=(12,6))
    plt.plot(out_before[0]["freq_test"],out_before[0]["PA"]["frequency"],label="Observed PA")
    plt.plot(out_before[0]["freq_test"],out_after2[0]["PA"]["frequency"],label="Intrinsic PPA")
    plt.plot(out_after2[0]["freq_test"],(np.array(out_after2[0]["PA"]["frequency"]) + np.pi/2)%np.pi,label="B Field Orientation")
    plt.grid()
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Polarization Angle (rad)")
    #plt.ylim(-1,1)
    plt.title(ids[i] + "_" + nicknames[i])
    plt.savefig("/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/"+ids[i] + "_" + nicknames[i] + "/" + ids[i] + "_" + nicknames[i] + "_polangle_frequency_" + "_highres6144chan_finalv2_after_"+ str(n_f) + "_binned.pdf")
    plt.legend()
    plt.close(f)

    f=plt.figure(figsize=(12,6))
    plt.plot(np.arange(timestart,timestop),out_before[0]["PA"]["time"][timestart:timestop],label="Observed PA")
    plt.plot(np.arange(timestart,timestop),out_after2[0]["PA"]["time"][timestart:timestop],label="Intrinsic PPA")
    plt.plot(np.arange(timestart,timestop),((np.array(out_after2[0]["PA"]["time"]) + np.pi/2)%np.pi)[timestart:timestop],label="B Field Orientation")
    plt.grid()
    plt.xlabel("Time Sample (" + str(fobj.header.tsamp*n_ts[i]) + " s sampling time)")
    plt.legend()
    plt.ylabel("Polarization Angle (rad)")
    #plt.ylim(-1,1)
    plt.title(ids[i] + "_" + nicknames[i])
    plt.savefig("/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/"+ids[i] + "_" + nicknames[i] + "/" + ids[i] + "_" + nicknames[i] + "_polangle_time_" + "_highres6144chan_finalv2_after_" + str(n_ts[i]) + "_binned.pdf")
    #plt.xlim(timestart,timestop)dd
    plt.close(f)

    f=plt.figure(figsize=(12,6))
    c=plt.plot(out_before[0]["freq_test"],out_before[0]["polarization"]["frequency"],label="Observed Total Polarization")
    plt.plot(out_after2[0]["freq_test"],out_after2[0]["polarization"]["frequency"],"--",color=c[0].get_color(),label="Intrinsic Total Polarization")
    c=plt.plot(out_before[0]["freq_test"],out_before[0]["linear polarization"]["frequency"],label="Observed Linear Polarization")
    plt.plot(out_after2[0]["freq_test"],out_after2[0]["linear polarization"]["frequency"],"--",color=c[0].get_color(),label="Intrinsic Linear Polarization")
    c=plt.plot(out_before[0]["freq_test"],out_before[0]["circular polarization"]["frequency"],label="Observed Circular Polarization")
    plt.plot(out_after2[0]["freq_test"],out_after2[0]["circular polarization"]["frequency"],"--",color=c[0].get_color(),label="Intrinsic Circular Polarization")
    plt.grid()
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Polarization Fraction")
    plt.ylim(-1.1,1.1)
    plt.title(ids[i] + "_" + nicknames[i])
    plt.savefig("/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/"+ids[i] + "_" + nicknames[i] + "/" + ids[i] + "_" + nicknames[i] + "_polfrac_frequency_" + "_highres6144chan_finalv2_after_"+ str(n_f) + "_binned.pdf")
    plt.legend()
    plt.close(f)

    f=plt.figure(figsize=(12,6))
    c=plt.plot(np.arange(timestart,timestop),out_before[0]["polarization"]["time"][timestart:timestop],label="Observed Total Polarization")
    plt.plot(np.arange(timestart,timestop),out_after2[0]["polarization"]["time"][timestart:timestop],"--",color=c[0].get_color(),label="Intrinsic Total Polarization")
    c=plt.plot(np.arange(timestart,timestop),out_before[0]["linear polarization"]["time"][timestart:timestop],label="Observed Linear Polarization")
    plt.plot(np.arange(timestart,timestop),out_after2[0]["linear polarization"]["time"][timestart:timestop],"--",color=c[0].get_color(),label="Intrinsic Linear Polarization")
    c=plt.plot(np.arange(timestart,timestop),out_before[0]["circular polarization"]["time"][timestart:timestop],label="Observed Circular Polarization")
    plt.plot(np.arange(timestart,timestop),out_after2[0]["circular polarization"]["time"][timestart:timestop],"--",color=c[0].get_color(),label="Intrinsic Circular Polarization")
    plt.grid()
    plt.xlabel("Time Sample (" + str(fobj.header.tsamp*n_ts[i]) + " s sampling time)")
    plt.legend()
    plt.ylabel("Polarization Fraction")
    plt.ylim(-1.1,1.1)
    plt.title(ids[i] + "_" + nicknames[i])
    plt.savefig("/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/"+ids[i] + "_" + nicknames[i] + "/" + ids[i] + "_" + nicknames[i] + "_polfrac_time_" + "_highres6144chan_finalv2_after_" + str(n_ts[i]) + "_binned.pdf")
    #plt.xlim(timestart,timestop)dd
    plt.close(f)


    fname_json = "/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/"+ids[i] + "_" + nicknames[i] + "/"+"out_RM6144_finalv2_after.json"
    print("Writing output data to " + fname_json)
    with open(fname_json, "w") as outfile:
        json.dump(out_after, outfile)
