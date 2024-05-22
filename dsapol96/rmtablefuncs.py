import numpy as np
from rmtable import rmtable
from astropy.coordinates import SkyCoord
import astropy.units as u







def make_FRB_RMTable(state_dict):


    """
    Uses the RMTable (Van Eck 2015) module to create an RM table entry for the current 
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
               'rm_method':['RM Synthesis—Pol. Int'],
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
               'minfreq':[np.min(state_dict['freq_test'][0])*1e6],
               'maxfreq':[np.max(state_dict['freq_test'][0])*1e6],
               'channelwidth':[(np.max(state_dict['freq_test'][0]) - np.min(state_dict['freq_test'][0]))*1e6/6144],
               'Nchan':[6144],
               'rmsf_fwhm':[305.6],
               'noise_chan':[state_dict['noise_chan']],
               'telescope':['DSA-110'],
               'int_time':[32.7e-6],
               'epoch':[state_dict['mjd']],
               'obs_interval':[20480*32.7e-6/86400],
               'leakage':[0.02],
               'beamdist':[np.abs(state_dict['ibeam']-125)*14/3600],
               'catalog_name':[""], #get Casey to fill in when added to the DSA-110 Event Archive and given a unique DOI
               'dataref':[""], #get Casey to fill in when added to the DSA-110 Event Archive and given a unique DOI
               'cat_id':[state_dict['ids']], #possibly replace with FRB TNS name when assigned
               'type':['FRB'],
               'notes':[""]
               })
   
    return r.to_pandas()
