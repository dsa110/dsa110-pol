from dsapol import dsapol
import numpy as np


ids=["220121aaat", "220204aaai", "220207aabh", "220208aaaa", "220307aaae", "220310aaam", "220319aaeb", "220330aaan", "220418aaai", "220424aabq", "220506aabd", "220426aaaw"]
nicknames=["clare", "fen", "zach", "ishita" ,"alex", "whitney" ,"mark", "erdos", "quincy" ,"davina", "oran", "jackie"]
widths=[8,4,2,16,2,4,1,32,4,2,2,2]

for i in range(len(nicknames)):

    out = dsapol.FRB_plot_all(datadir="/home/ubuntu/sherman/scratch_weights_update_2022-06-03/" + str(ids[i]) +"_" + str(nicknames[i]) + "/",prefix=ids[i] + "_dev",nickname=nicknames[i],nsamps=5120,n_t=1,n_f=32,n_off=3000,width_native=widths[i],cal=True,gain_dir='/home/ubuntu/sherman/scratch_weights_update_2022-06-03/3C48_test/',gain_source_name="3C48",gain_obs_names=["ane"],phase_dir='/home/ubuntu/sherman/scratch_weights_update_2022-06-03/3C286_test/',phase_source_name="3C286",phase_obs_names=["jqc"],deg=10,suffix="_dev",use_fit=True,get_RM=True,RM_cal=False,trial_RM=np.linspace(-1e6,1e6,int(1e6)),trial_phi=[0],n_trial_RM_zoom=2500,zoom_window=200,fit_window=200,cal_2D=True,sub_offpulse_mean=True,window=10,buff=1,lim=10)
    print(out[1:])
