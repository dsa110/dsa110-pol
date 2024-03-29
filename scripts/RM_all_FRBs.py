from dsapol import dsapol
import numpy as np
import sys
import json
from matplotlib import pyplot as plt

#ids = ids[2:]
#nicknames = nicknames[2:]
#widths = widths[2:]
#DMs = DMs[2:]
ids=["220121aaat", "220204aaai", "220207aabh", "220208aaaa", "220307aaae", "220310aaam", "220330aaan", "220418aaai", "220424aabq", "220506aabd", "220426aaaw","220319aaeb", "220726aabn", "220801aabd", "220411aabk", "220825aaad", "220831aaaj"]
nicknames=["clare", "fen", "zach", "ishita" ,"alex", "whitney" , "erdos", "quincy" ,"davina", "oran", "jackie","mark","gertrude", "augustine", "maya", "ansel", "ada"]
widths=[8,4,2,16,2,4,32,4,2,2,2,1,4,4,32,4,8]
DMs=[313.5,612.2,262.3,437.0,499.15,462.15,468.1,623.45,863.35,396.93,269.5,110.95,686.55,413.0,150,651.2,1146.25]
#RM_pre = [-1605.99404020451,2.86466865418861,126.864792654317,-107.345983661745,-958.625019151275,14.8797122481028,-227.511451195641,63.7819558872256,177.481338007615,-46.6241071503845,-140.864806654308,-575098.913459439,462.820033872613,245387.471011155,-1,759.602869424782,817.599344704751]
RM_pre = [-1559.00155900151,7.0000070000533,131.000131000182,-119.000119000091,-959.000959000899,13.0000129999825,-173.000173000153,1.00000100000761,135.000135000096,-47.0000470000086,-145.000145000172,-575113.575113575,485.000485000433,245447.245447245,0,749.000749000697,825.00082500081]
n_ts = [4,1,1,2,1,1,4,1,1,1,1,1,1,1,1,1,1]

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


    (I_32,Q_32,U_32,V_32,fobj_32,timeaxis_32,freq_test32,wav_test32) = dsapol.get_stokes_2D("/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/"+ids[i] + "_" + nicknames[i] + "/",ids[i] + "_dev",20480,n_t=n_ts[i],n_f=32,n_off=int(12000//n_ts[i]),sub_offpulse_mean=True)
    (peak,timestart,timestop) = dsapol.find_peak(I_32,widths[i],fobj_32.header.tsamp,n_t=n_ts[i],peak_range=None,pre_calc_tf=False)
    print(peak,timestart,timestop)
    (I_cal32,Q_cal32,U_cal32,V_cal32) = dsapol.calibrate(I_32,Q_32,U_32,V_32,(gxx,gyy),stokes=True)
    (peak,timestart,timestop) = dsapol.find_peak(I_cal32,widths[i],fobj_32.header.tsamp,n_t=n_ts[i],peak_range=None,pre_calc_tf=False)
    print(peak,timestart,timestop)


    PA_f,PA_t,avg = dsapol.get_pol_angle(I_cal32,Q_cal32,U_cal32,V_cal32,widths[i],fobj_32.header.tsamp,n_ts[i],32,freq_test32,n_off=int(12000//n_ts[i]),plot=True,datadir="/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/"+ids[i] + "_" + nicknames[i] + "/",calstr="_highres6144chan_final_before_",label=str(ids[i]) + "_dev_" + str(nicknames[i]),show=False,weighted=True,n_t_weight=1,timeaxis=timeaxis_32,fobj=fobj_32,buff=0)

    
    [(pol_f,pol_t,avg_frac,sigma_frac,snr_frac),(L_f,L_t,avg_L,sigma_L,snr_L),(C_f,C_t,avg_C,sigma_C,snr_C)]=dsapol.get_pol_fraction(I_cal32,Q_cal32,U_cal32,V_cal32,widths[i],fobj_32.header.tsamp,n_ts[i],32,freq_test32,n_off=int(12000//n_ts[i]),plot=True,datadir="/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/"+ids[i] + "_" + nicknames[i] + "/",label=str(ids[i]) + "_dev_" + str(nicknames[i]),calstr="_highres6144chan_final_before_",pre_calc_tf=False,show=False,normalize=True,buff=0,full=False,weighted=True,n_t_weight=1,timeaxis=timeaxis_32,fobj=fobj_32)



    print("BEFORE RM CORRECTION")

    print("Polarization:")

    print(avg_frac,sigma_frac)
    print(avg_L,sigma_L)
    print(avg_C,sigma_C)

    print("PA: " + str(avg) + " pm " + str(np.nanstd(PA_t)))









    """
    #(I,Q,U,V,fobj,timeaxis,freq_test,wav_test) = dsapol.get_stokes_2D("/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/3C48/","3C48ane_dev",20480,n_f=1,n_t=n_ts[i],sub_offpulse_mean=False)#test
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

    print(peak,timestart,timestop)


    #get 32 binned data
    #(I,Q,U,V,fobj,timeaxis,freq_test,wav_test) = dsapol.get_stokes_2D("/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/3C48/","3C48ane_dev",20480,n_f=32,n_t=n_ts[i],sub_offpulse_mean=False)#test
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


    (I_32,Q_32,U_32,V_32,fobj_32,timeaxis_32,freq_test32,wav_test32) = dsapol.get_stokes_2D("/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/"+ids[i] + "_" + nicknames[i] + "/",ids[i] + "_dev",20480,n_t=n_ts[i],n_f=32,n_off=int(12000//n_ts[i]),sub_offpulse_mean=True)
    (peak,timestart,timestop) = dsapol.find_peak(I_32,widths[i],fobj_32.header.tsamp,n_t=n_ts[i],peak_range=None,pre_calc_tf=False)
    (I_cal32,Q_cal32,U_cal32,V_cal32) = dsapol.calibrate(I_32,Q_32,U_32,V_32,(gxx,gyy),stokes=True)
    print(peak,timestart,timestop)

    #I_cal32 = dsapol.avg_freq(I_cal,32)
    #Q_cal32 = dsapol.avg_freq(Q_cal,32)
    #U_cal32 = dsapol.avg_freq(U_cal,32)
    #V_cal32 = dsapol.avg_freq(V_cal,32)

    #freq_test32 = [np.linspace(freq_test[0][0],freq_test[0][-1],int(len(freq_test[0])/32))]

    PA_f,PA_t,avg = dsapol.get_pol_angle(I_cal32,Q_cal32,U_cal32,V_cal32,widths[i],fobj_32.header.tsamp,n_ts[i],32,freq_test32,n_off=int(12000//n_ts[i]),plot=True,datadir="/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/"+ids[i] + "_" + nicknames[i] + "/",calstr="_highres6144chan_final_before_",label=str(ids[i]) + "_dev_" + str(nicknames[i]),show=False,weighted=True,n_t_weight=1,timeaxis=timeaxis_32,fobj=fobj_32)

    
    [(pol_f,pol_t,avg_frac,sigma_frac,snr_frac),(L_f,L_t,avg_L,sigma_L,snr_L),(C_f,C_t,avg_C,sigma_C,snr_C)]=dsapol.get_pol_fraction(I_cal32,Q_cal32,U_cal32,V_cal32,widths[i],fobj_32.header.tsamp,n_ts[i],32,freq_test32,n_off=int(12000//n_ts[i]),plot=True,datadir="/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/"+ids[i] + "_" + nicknames[i] + "/",label=str(ids[i]) + "_dev_" + str(nicknames[i]),calstr="_highres6144chan_final_before_",pre_calc_tf=False,show=False,normalize=True,buff=1,full=False,weighted=True,n_t_weight=1,timeaxis=timeaxis_32,fobj=fobj_32)

    print("BEFORE RM CORRECTION")

    print("Polarization:")

    print(avg_frac,sigma_frac)
    print(avg_L,sigma_L)
    print(avg_C,sigma_C)

    print("PA: " + str(avg) + " pm " + str(np.nanstd(PA_t)))

    (RM,phi,SNRs,RMerr,p_val,B) = dsapol.faradaycal_full(I_cal,Q_cal,U_cal,V_cal,freq_test,np.linspace(-1e6,1e6,int(1e6)),[0],widths[i],fobj.header.tsamp,n_trial_RM_zoom=5000,zoom_window=1000,plot=True,datadir="/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/"+ids[i] + "_" + nicknames[i] + "/",calstr="_highres6144chan_final_",label=str(ids[i]) + "_dev_" + str(nicknames[i]),n_f=1,n_t=n_ts[i],n_off=int(12000/n_ts[i]),show=False,fit_window=100,buff=1,normalize=True,DM=DMs[i],weighted=True,n_t_weight=1,timeaxis=timeaxis,fobj=fobj,RM_tools=True,trial_RM_tools=np.linspace(-1e6,1e6,int(1e4))) 
    outdata = dict()
    outdata["RM"] = RM
    outdata["phi"] = phi
    outdata["RMerr"] = RMerr
    outdata["p_val"] = p_val
    outdata["B"] = B

    #Calibrate and plot PA
    (I_cal2,Q_cal2,U_cal2,V_cal2) = dsapol.calibrate_RM(I_cal,Q_cal,U_cal,V_cal,RM,phi,freq_test,stokes=True)
    I_cal2 = dsapol.avg_freq(I_cal2,32)
    Q_cal2 = dsapol.avg_freq(Q_cal2,32)
    U_cal2 = dsapol.avg_freq(U_cal2,32)
    V_cal2 = dsapol.avg_freq(V_cal2,32)

    PA_f2,PA_t2,avg2 = dsapol.get_pol_angle(I_cal2,Q_cal2,U_cal2,V_cal2,widths[i],fobj_32.header.tsamp,n_ts[i],32,freq_test32,n_off=int(12000//n_ts[i]),plot=True,datadir="/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/"+ids[i] + "_" + nicknames[i] + "/",calstr="_highres6144chan_final_after",label=str(ids[i]) + "_dev_" + str(nicknames[i]),show=False,weighted=True,n_t_weight=1,timeaxis=timeaxis_32,fobj=fobj_32)

    [(pol_f,pol_t,avg_frac,sigma_frac,snr_frac),(L_f,L_t,avg_L,sigma_L,snr_L),(C_f,C_t,avg_C,sigma_C,snr_C)]=dsapol.get_pol_fraction(I_cal2,Q_cal2,U_cal2,V_cal2,widths[i],fobj_32.header.tsamp,n_ts[i],32,freq_test32,n_off=int(12000//n_ts[i]),plot=True,datadir="/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/"+ids[i] + "_" + nicknames[i] + "/",label=str(ids[i]) + "_dev_" + str(nicknames[i]),calstr="_highres6144chan_final_after_",show=False,normalize=True,buff=1,full=False,weighted=True,n_t_weight=1,timeaxis=timeaxis_32,fobj=fobj_32)

    print("AFTER RM CORRECTION")

    print("Polarization:")

    print(avg_frac,sigma_frac)
    print(avg_L,sigma_L)
    print(avg_C,sigma_C)

    print("PA: " + str(avg2) + " pm " + str(np.nanstd(PA_t2)))
    print("B_perp: "  + str(np.mean(((PA_t2 + np.pi/2)%np.pi)[timestart:timestop])))
    print("linear SNR: " + str((np.max(SNRs),snr_L)) )

    f=plt.figure(figsize=(12,6))
    plt.plot(freq_test32[0],PA_f,label="Observed PA")
    plt.plot(freq_test32[0],PA_f2,label="Intrinsic PPA")
    plt.plot(freq_test32[0],(PA_f2 + np.pi/2)%np.pi,label="B Field Orientation")
    plt.grid()
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Polarization Angle (rad)")
    #plt.ylim(-1,1)
    plt.title(ids[i] + "_" + nicknames[i])
    plt.savefig("/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/"+ids[i] + "_" + nicknames[i] + "/" + ids[i] + "_" + nicknames[i] + "_polangle_frequency_" + "_highres6144chan_final_after_"+ str(n_f) + "_binned.pdf")
    plt.legend()
    plt.close(f)

    f=plt.figure(figsize=(12,6))
    plt.plot(np.arange(timestart,timestop),PA_t[timestart:timestop],label="Observed PA")
    plt.plot(np.arange(timestart,timestop),PA_t2[timestart:timestop],label="Intrinsic PPA")
    plt.plot(np.arange(timestart,timestop),((PA_t2 + np.pi/2)%np.pi)[timestart:timestop],label="B Field Orientation")
    plt.grid()
    plt.xlabel("Time Sample (" + str(fobj.header.tsamp*n_ts[i]) + " s sampling time)")
    plt.legend()
    plt.ylabel("Polarization Angle (rad)")
    #plt.ylim(-1,1)
    plt.title(ids[i] + "_" + nicknames[i])
    plt.savefig("/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/"+ids[i] + "_" + nicknames[i] + "/" + ids[i] + "_" + nicknames[i] + "_polangle_time_" + "_highres6144chan_final_after_" + str(n_ts[i]) + "_binned.pdf")
    #plt.xlim(timestart,timestop)dd
    plt.close(f)

    fname_json = "/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/"+ids[i] + "_" + nicknames[i] + "/"+"out_RM6144_final_after.json"
    print("Writing output data to " + fname_json)
    with open(fname_json, "w") as outfile:
        json.dump(outdata, outfile)

    
    print("B field: " + str(B))

"""
