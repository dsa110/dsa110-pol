import numpy as np
from rmtable import rmtable
from astropy.coordinates import SkyCoord
import astropy.units as u
import polspectra
import json
f = open("directories.json","r")
dirs = json.load(f)
f.close()


logfile = dirs["logs"] + "archive_logfile.txt" #"/media/ubuntu/ssd/sherman/code/dsapol_logfiles/archive_logfile.txt"

def make_FRB_polSpectra(state_dict):

    """
    Uses the polspectra (Van Eck 2023) module to create a polspectra table entry for
    the current FRB, which can be output to the Level 3 directory.
    """

    coord = SkyCoord(ra=state_dict['RA']*u.deg,dec=state_dict['DEC']*u.deg,frame='icrs')

    p = polspectra.polarizationspectra()
    
    p.create_from_arrays(long_array=[coord.icrs.ra.value],
                     lat_array=[coord.icrs.dec.value],
                     freq_array=[state_dict['base_freq_test'][0]],
                     stokesI=[state_dict['base_Ical_f_unnormalized']],
                     stokesI_error=[state_dict['base_Ical_f_unnormalized_errs']],
                     stokesQ=[state_dict['base_Qcal_f_unnormalized']],
                     stokesQ_error=[state_dict['base_Qcal_f_unnormalized_errs']],
                     stokesU=[state_dict['base_Ucal_f_unnormalized']],
                     stokesU_error=[state_dict['base_Ucal_f_unnormalized_errs']],
                     source_number_array=[0],
                     beam_maj=[14/3600],
                     beam_min=[14/3600],
                     beam_pa=[90],
                     stokesV=[state_dict['base_Vcal_f_unnormalized']],
                     stokesV_error=[state_dict['base_Vcal_f_unnormalized_errs']],
                     quality=[np.array(state_dict['base_I'].mask[:,0],dtype=int)],
                     quality_meanings=["0 = good, 1 = bad"],
                     cat_id=[state_dict['ids']],
                     telescope=["DSA-110"],
                     epoch=[state_dict['mjd']],
                     integration_time=[32.7e-6],
                     interval=[state_dict['base_I'].shape[1]*32.7e-6/86400],
                     leakage=[0.02],
                     channel_width=[(np.max(state_dict['base_freq_test'][0]) - np.min(state_dict['base_freq_test'][0]))*1e6/6144],
                     flux_type=["integrated"],
                     aperture=[14/3600],
                     )
    return p#.table.to_pandas()

def make_FRB_RMTable(state_dict):


    """
    Uses the RMTable (Van Eck 2023) module to create an RM table entry for the current 
    FRB, which can be output to the Level 3 directory.
    """

    #make new RMTable
    #r = rmtable.RMTable({})
    #r.add_missing_columns() #creates default columns
    #print(r.columns)
    #add default column values
    coord = SkyCoord(ra=state_dict['RA']*u.deg,dec=state_dict['DEC']*u.deg,frame='icrs')
    #r.add_row({'ra':coord.icrs.ra.value,'dec':coord.icrs.dec.value})
    
    
    r = rmtable.RMTable({'ra':[coord.icrs.ra.value],
               'dec':[coord.icrs.dec.value],
               'l':[coord.galactic.l.value],
               'b':[coord.galactic.b.value],
               'pos_err':[np.nan],
               'rm':[state_dict['RMcalibrated']['RMcal']],
               'rm_err':[state_dict['RMcalibrated']['RMcalerr']],
               'rm_width':[state_dict['RMcalibrated']['RMFWHM']],
               'rm_width_err':[np.nan],
               'complex_flag':['N'],
               'complex_test':['None'],
               'rm_method':['RM Synthesis-Pol. Int'],
               'ionosphere':['RMextract (NOT APPLIED)'],
               'Ncomp':[state_dict['n_comps']],
               'stokesI':[state_dict['Iflux']],
               'stokesI_err':[state_dict['Iflux_err']],
               'spectral_index':[np.nan],
               'spectral_index_err':[np.nan],
               'reffreq_I':[1405e6],
               'polint':[state_dict['polint']],
               'polint_err':[state_dict['polint_err']],
               'pol_bias':['1985A&A...142..100S'],
               'flux_type':['Visibilities'],
               'aperture':[14/3600],
               'fracpol':[state_dict['Lpol']],
               'fracpol_err':[state_dict['Lpol_err']],
               'polangle':[np.nan],
               'polangle_err':[np.nan],
               'derot_polangle':[(180/np.pi)*state_dict['avg_PA']],
               'derot_polangle_err':[(180/np.pi)*state_dict['PA_err']],
               'reffreq_pol':[1405e6],
               'stokesQ':[state_dict['Qflux']],
               'stokesQ_err':[state_dict['Qflux_err']],
               'stokesU':[state_dict['Uflux']],
               'stokesU_err':[state_dict['Uflux_err']],
               'stokesV':[state_dict['Vflux']],
               'stokesV_err':[state_dict['Vflux_err']],
               'beam_maj':[14/3600],
               'beam_min':[14/3600],
               'beam_pa':[90],
               'reffreq_beam':[1405e6],
               'minfreq':[np.min(state_dict['base_freq_test'][0])*1e6],
               'maxfreq':[np.max(state_dict['base_freq_test'][0])*1e6],
               'channelwidth':[(np.max(state_dict['base_freq_test'][0]) - np.min(state_dict['base_freq_test'][0]))*1e6/6144],
               'Nchan':[6144],
               'rmsf_fwhm':[305.6],
               'noise_chan':[state_dict['noise_chan']],
               'telescope':['DSA-110'],
               'int_time':[32.7e-6],
               'epoch':[state_dict['mjd']],
               'obs_interval':[state_dict['base_I'].shape[1]*32.7e-6/86400],
               'leakage':[0.02],
               'beamdist':[np.abs(state_dict['ibeam']-125)*14/3600],
               'catalog_name':[""], #get Casey to fill in when added to the DSA-110 Event Archive and given a unique DOI
               'dataref':[""], #get Casey to fill in when added to the DSA-110 Event Archive and given a unique DOI
               'cat_id':[state_dict['ids']], #possibly replace with FRB TNS name when assigned
               'type':['FRB'],
               'notes':[""]
               })
  
    #additional columns
    r['Vfracpol'] = [state_dict['Vpol']]
    r['Vfracpol'].description = "Fractional (signed circular) polarization"
    r['Vfracpol_err'] = [state_dict['Vpol_err']]
    r['Vfracpol_err'].description = "Error in fractional (signed circular) polarization"
    r['absVfracpol'] = [state_dict['absVpol']]
    r['absVfracpol'].description = "Fractional (unsigned circular) polarization"
    r['absVfracpol_err'] = [state_dict['absVpol_err']]
    r['absVfracpol_err'].description = "Error in fractional (unsigned circular) polarization"
    r['totalfracpol'] = [state_dict['Tpol']]
    r['totalfracpol'].description = "Fractional (total) polarization"
    r['totalfracpol_err'] = [state_dict['Tpol_err']]
    r['totalfracpol_err'].description = "Error in fractional (total) polarization"
    
    r['snr'] = [state_dict['snr']]
    r['snr'].description = "Total intensity signal-to-noise ratio"
    r['polsnr'] = [state_dict['Lsnr']]
    r['polsnr'].description = "Linear polarization signal-to-noise ratio"
    r['totalpolsnr'] = [state_dict['Tsnr']]
    r['totalpolsnr'].description = "Total polarization signal-to-noise ratio"
    r['Vsnr'] = [state_dict['Vsnr']]
    r['Vsnr'].description = "Circular polarization signal-to-noise ratio"

    return r#.to_pandas()
