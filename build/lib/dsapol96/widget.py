import numpy as np
from dsapol import parsec
from dsapol import polcal
from dsapol import polbeamform
import copy
import glob


"""
This file contains the initial values for each widget, and updates whenever a screen function is run so that when the page refreshes, widgets maintain their current values instead of being reset.
"""
frbfiles = parsec.get_frbfiles()
ids = frbfiles[0][:frbfiles[0].index('_')]
RA = parsec.FRB_RA[parsec.FRB_IDS.index(ids)]
DEC = parsec.FRB_DEC[parsec.FRB_IDS.index(ids)]
ibeam = int(parsec.FRB_BEAM[parsec.FRB_IDS.index(ids)])
mjd = parsec.FRB_mjd[parsec.FRB_IDS.index(ids)]
DMinit = parsec.FRB_DM[parsec.FRB_IDS.index(ids)]
polcalfiles_findbeams = polcal.get_beamfinding_files()
polcaldates = []
for k in parsec.polcal_dict.keys():
    if 'polcal' not in str(k):
        polcaldates.append(str(k))
polcalfiles_bf = polcal.get_avail_caldates()
polcalfiles_findbeams = polcal.get_beamfinding_files()

obs_files_3C48,obs_ids_3C48 = polcal.get_calfil_files('3C48',polcalfiles_findbeams[0],'3C48*0')
obs_files_3C286,obs_ids_3C286 = polcal.get_calfil_files('3C286',polcalfiles_findbeams[0],'3C286*0')

polcalfiles = glob.glob(parsec.default_path + 'POLCAL_PARAMETERS_*csv')
polcalfiles = [polcalfiles[i][polcalfiles[i].index('POLCAL'):] for i in range(len(polcalfiles))]


wdict = {'toggle_menu':'(0) Load Data', ############### (0) Load Data ##################
         'frbfiles_menu':frbfiles[0],
         'base_n_t_slider':1,
         'base_logn_f_slider':0,
         'logibox_slider_init':0,
         'buff_L_slider_init':1,
         'buff_R_slider_init':1,
         'RA_display':RA,
         'DEC_display':DEC,
         'ibeam_display':ibeam,
         'mjd_display':mjd,
         'DM_init_display':DMinit,
         'showlog':True,
         
         'n_t_slider':1, ############### (1) Dedispersion ##################
         'logn_f_slider':5,
         'logwindow_slider':5,
         'ddm_num':0,
         'DM_input_display':np.nan,
         'DM_new_display':np.nan,

         'polcaldate_create_menu':"", ############### (3) Calibration ##################
         'polcaldate_bf_menu':"",
         'polcaldate_findbeams_menu':polcalfiles_findbeams[0],
         'obsid3C48_menu':"",
         'obsid3C286_menu':"",
         'ParA_display':np.nan,
         'peakheight_slider':2,
         'peakwidth_slider':10,
         'sfflag':False,
         'sf_window_weight_cals':255,
         'sf_order_cals':5,
         'polyfitflag':False,
         'edgefreq_slider':1370,
         'breakfreq_slider':1370,
         'ratio_peakheight_slider':3,
         'ratio_peakwidth_slider':10,
         'ratio_sfflag':False,
         'ratio_sf_window_weight_cals':257,
         'ratio_sf_order_cals':5,
         'ratio_polyfitflag':False,
         'ratio_polyfitorder_slider':5,
         'ratio_edgefreq_slider':1360,
         'ratio_breakfreq_slider':1360,
         'phase_peakheight_slider':3,
         'phase_peakwidth_slider':10,
         'phase_sfflag':False,
         'phase_sf_window_weight_cals':255,
         'phase_sf_order_cals':5,
         'phase_polyfitflag':False,
         'phase_polyfitorder_slider':5,
         'polcaldate_menu':"",
         'showlogcal':False,
         
         'ncomps_num':1, ############### (4) Filter Weights ##################
         'comprange_slider':[0,1],
         'avger_w_slider':1,
         'sf_window_weights_slider':3,

         }



def update_wdict(objects,labels):
    """
    This function takes a list of widget objects and a list of their names as strings and updates the wdict with their curent values
    """

    assert(len(objects)==len(labels))
    for i in range(len(objects)):
        wdict[labels[i]] = objects[i].value
    return

