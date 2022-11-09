from dsapol import dsapol
import numpy as np
import sys
from matplotlib import pyplot as plt

import json
ids=["220121aaat", "220204aaai", "220207aabh", "220208aaaa", "220307aaae", "220310aaam", "220330aaan", "220418aaai", "220424aabq", "220506aabd", "220426aaaw","220319aaeb", "220726aabn", "220801aabd", "220411aabk", "220825aaad", "220831aaaj", "220914aabz","220920aacl","220926aaeu","221002aaab"]
nicknames=["clare", "fen", "zach", "ishita" ,"alex", "whitney" , "erdos", "quincy" ,"davina", "oran", "jackie","mark","gertrude", "augustine", "maya", "ansel", "ada", "elektra","etienne","celeste", "arni"]
widths=[8,4,2,16,2,4,32,4,2,2,2,1,4,4,32,4,8,2,2,2,16]
DMs=[313.5,612.2,262.3,437.0,499.15,462.15,468.1,623.45,863.35,396.93,269.5,110.95,686.55,413.0,150,651.2,1146.25,631.05,315.1,441.53,322.7]
n_ts = [2,1,1,2,1,1,4,1,1,1,1,1,1,1,1,1,1,1,1,1,16]
RMs = [-1305.55086886348,2.79916683201975,127.199370848151,761.776036230955,-958.800918992897,15.2004530880001,-227.211015168586,64.0126035205116,178.00873672044,-46.800006992007,-140.399224816135,-575098.97219299,463.1961241282585 ,245387.033404836,0,741.199188688634,809.5977443846867,0,0,0,0]
RM_gals = [-68.17985557759044,-7.173808637692105,3.4538780715571686,-6.750317532361379,-14.968536640281991,-14.920449670882448,-15.576725577586794,4.84899789477938,-39.1415848090991,1.0584624922027939,7.736806347620126,-3.2924384810295937,-50.258010868238365,5.662006457588828,-0.013026056420469034,10.047789988844018,-9.21574742050912,0,0,0,0]
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
    DMs = DMs[idx:idx+1]
    n_ts = n_ts[idx:idx+1]
    RMs = RMs[idx:idx+1]
    RM_gals = RM_gals[idx:idx+1]

print(nicknames)
for i in range(len(nicknames)):

    out = dsapol.FRB_plot_all(datadir="/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/" + str(ids[i]) +"_" + str(nicknames[i]) + "/",prefix=ids[i] + "_dev",nickname=nicknames[i],nsamps=20480,n_t=n_ts[i],n_f=32,n_off=int(12000//n_ts[i]),width_native=widths[i],cal=True,gain_dir='/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/3C48_22-09-14/',gain_source_name="3C48",gain_obs_names=["lck"],phase_dir='/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/3C286_22-09-14/',phase_source_name="3C286",phase_obs_names=["mph"],deg=10,suffix="_dev",use_fit=True,get_RM=True,RM_cal=False,trial_RM=np.linspace(-1e6,1e6,int(2e6)),trial_phi=[0],n_trial_RM_zoom=5000,zoom_window=1000,fit_window=100,cal_2D=True,sub_offpulse_mean=True,window=10,buff=1,lim=3,DM=DMs[i],weighted=False,n_t_weight=8,use_sf=True,sfwindow=0,extra="_initial_",clean=True,padwidth=10,peakheight=2,n_t_down=8,sf_window_weights=45)
