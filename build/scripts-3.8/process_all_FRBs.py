from dsapol import dsapol
import numpy as np
import sys



ids=["220121aaat", "220204aaai", "220207aabh", "220208aaaa", "220307aaae", "220310aaam", "220330aaan", "220418aaai", "220424aabq", "220506aabd", "220426aaaw","220319aaeb", "220726aabn", "220801aabd"]
nicknames=["clare", "fen", "zach", "ishita" ,"alex", "whitney" , "erdos", "quincy" ,"davina", "oran", "jackie","mark","gertrude", "augustine"]
widths=[8,4,2,16,2,4,32,4,2,2,2,1,4,4]
DMs=[313.5,612.2,262.3,437.0,499.15,462.15,468.1,623.45,863.35,396.93,269.5,110.95,686.55,413.0]

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

    out = dsapol.FRB_plot_all(datadir="/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/" + str(ids[i]) +"_" + str(nicknames[i]) + "/",prefix=ids[i] + "_dev",nickname=nicknames[i],nsamps=20480,n_t=1,n_f=32,n_off=12000,width_native=widths[i],cal=True,gain_dir='/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/3C48/',gain_source_name="3C48",gain_obs_names=["ane"],phase_dir='/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/3C286/',phase_source_name="3C286",phase_obs_names=["jqc"],deg=10,suffix="_dev",use_fit=True,get_RM=True,RM_cal=False,trial_RM=np.linspace(-1e6,1e6,int(1e6)),trial_phi=[0],n_trial_RM_zoom=5000,zoom_window=1000,fit_window=100,cal_2D=True,sub_offpulse_mean=True,window=10,buff=2,lim=10,DM=DMs[i],weighted=False,n_t_weight=2,use_sf=True,sfwindow=19)
    print(out[1:])
