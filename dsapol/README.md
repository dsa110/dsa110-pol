# dsapol

This module defines user-facing functions for polarization and RM analysis (`dsapol`) and wrapper functions used 
by the `PARSEC` backend.

## Structure

- `dsapol`: Base functions for polarization calibration, Faraday Rotation Measure synthesis, plotting, and polarization fraction analysis        
- `parsec`: Mid-level "screen" functions for the `PARSEC` analysis interface
- `polcal`: Wrapper functions for polarization calibration for use by the `PARSEC` interface        
- `polbeamform`: Wrapper functions for offline beamforming of baseband voltage data
- `customfilplotfuncs`: Custom implementation of `dsa110-T3/dsaT3/filplot_funcs` (https://github.com/dsa110/dsa110-T3) for making candidate plots from visibility data
- `dedisp`: Incoherent dedispersion implementation for use by the `PARSEC` interface
- `filt`: Wrapper functions for computing matched filtered signal-to-noise for use by the `PARSEC` interface
- `RMcal`: Wrapper functions for Faraday Rotation Measure (RM) Synthesis for use by the `PARSEC` interface
- `scatscint`: Wrapper functions for `dsa110-scat` (https://github.com/dsa110/dsa110-scat) scattering and scintillation analysis for use by the `PARSEC` interface
- `rmtablefuncs`: Wrapper functions to read/write data to/from `RMTable` (https://github.com/CIRADA-Tools/RMTable) and `PolSpectra` (https://github.com/CIRADA-Tools/PolSpectra) formats (defined by Van Eck et al. 2023 (https://doi.org/10.3847/1538-4365/acda24)
- `budget`: Functions to identify RM and DM components and host magnetic field strengths

## Usage

We describe here basic usage of functions in `dsapol` necessary for polarization analysis. Details on more customized usage are
provided in the documentation for individual functions. Polarization data is assumed to be saved in filterbanks labelled "name\_0.fil",
"name\_1.fil", "name\_2.fil", "name\_3.fil" for Stokes parameters I,Q,U, and V respectively. In the following example, we assume each file
contains 20480 time samples in 6144 frequency channels. To read polarization data from filterbank format:

```
>>>from dsapol import dsapol
>>>n_t=1 #downsampling in time
>>>n_f=1 #downsampling in frequency
>>>(I,Q,U,V,fobj,timeaxis,freq,wav,badchans) = dsapol.get_stokes_2D("path-to-directory-with-fil-files","name",20480,n_t=n_t,n_f=n_f)
```

where I,Q,U, and V are 2D numpy arrays containing each Stokes parameter, fobj is a Filterbank object whose header contains metadata for the data (`fobj.header`), timeaxis is the time offset from the start of the data for each sample in seconds, freq is a list of four numpy arrays, each providing the channel frequencies in MHz, wav is the same for wavelength in meters, and badchans is an array of channel indices identified as corrupted by `dsapol`. Additional parameters control downsampling and normalization.

Polarization calibration parameters can be read from a POLCAL csv file and applied to the data using:

```
>>>from dsapol import polcal
>>>(gxx,gyy,calfreq) = polcal.read_polcal("POLCAL_PARAMETERS_22-12-11")
>>>(Ical,Qcal,Ucal,Vcal) = dsapol.calibrate(I,Q,U,V,(gxx,gyy))
```

where `gxx` and `gyy` are the complex gains in the X and Y antenna feeds versus frequency (`calfreq`). "POLCAL\_PARAMETERS\_22-12-11" can be replaced with the desire calibration solution. 

The frequency and time-averaged stokes parameters can be computed as:

```
>>>width = 4 #this is an estimate of the width of the FRB in samples
>>>(It,It_err,Qt,Qt_err,Ut,Ut_err,Vt,Vt_err) = dsapol.get_stokes_vs_time(Ical,Qcal,Ucal,Vcal,width,fobj.header.tsamp,n_t)
>>>(If,Qf,Uf,Vf) = dsapol.get_stokes_vs_freq(Ical,Qcal,Ucal,Vcal,width,fobj.header.tsamp,n_f,n_t,freq)
```

Set `plot=True` to display plots of the time series and freuqency spectra. 

`dsapol` contains functions to run Faraday Rotation Measure (RM) synthesis on either the 1D frequency spectra or the 2D dynamic spectra. For 1D RM synthesis:

```
>>>import numpy as np
>>>trial_RM = np.linspace(-1000,1000,2000) #trial RM axis
>>>(RM,phi,FDF,RM_err,tmp) = dsapol.faradaycal(Ical,Qcal,Ucal,Vcal,freq,[0],n_f=n_f,n_t=n_t)
```

where `FDF` is the Faraday Dispersion Function (FDF; see Brentjens & de Bruyn 2005 for details). RM and RMerr are the peak RM and error. The `phi` parameter is not used in the current implementation. For 2 RM synthesis:

```
>>>(RM,phi,FDF,RM_err,RM_upper,RM_lower,signal,noise,FDF_2D,peak_RMs,tmp) = dsapol.faradaycal_SNR(Ical,Qcal,Ucal,Vcal,freq,trial_RM,[0],width,fobj.heaer.tsamp,n_f=n_f,n_t=n_t)
```

In addition to the peak RM and FDF, `faradaycal_SNR` returns the 84th and 16th percentiles (`RM_upper`, `RM_lower`), the mean signal (`signal`) and noise (`noise`) used to get the linear S/N reported as the `FDF`, the 2D FDF (`FDF_2D`) which is the FDF at each timestep within the burst, and the peak RM at each timestep (`peak_RMs`). A known RM can be used to calibrate the dynamic spectrum as follows:


```
>>>(IcalRM,QcalRM,UcalRM,VcalRM) = dsapol.calibrate_RM(Ical,Qcal,Ucal,Vcal,RM,0,freq,stokes=True)
```

Finally, to estimate polarization fractions:

```
>>>allpolfracs = dsapol.get_pol_fraction(IcalRM,QcalRM,UcalRM,VcalRM,width,fobj.header.tsamp,n_t,n_f,freq)
>>>(pol_f,pol_t,avg_frac,sigma_frac,snr_frac) = allpolfracs[0] # total polarization fractions and total polarization S/N
>>>(L_f,L_t,avg_L,sigma_L,snr_L) = allpolfracs[1] # linear polarization fractions and linear polarization S/N
>>>(C_f,C_t,avg_C_abs,sigma_C_abs,snr_C) = allpolfracs[2] # absolute value circular polarization fractions and circular polarization S/N
>>>(C_f,C_t,avg_C,sigma_C,snr_C) = allpolfracs[3] # signed circular polarization fractions andd circular polarization S/N
>>>snr = allpolfracs[4] # total intensity S/N
```

As shown above, this returns the polarization fractions versus frequency (`_f`), versus time (`_t`), and averaged with standard deviation (`avg_`, `sigma_`), as well as the signal-to-noise of the polarized signal (`snr_`) and intensity (`snr`).
