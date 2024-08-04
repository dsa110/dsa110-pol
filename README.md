# dsa110-pol
DSA-110 polarization utilities


Authors: Myles Sherman, Liam Connor, Casey Law, Vikram Ravi, Jakob Faber, Dana Simard

Last Upated: 2024-08-04

This library contains functions for polarization analysis of Fast Radio Bursts (FRBs) and the implementation of the DSA-110
Polarization Analysis and RM Synthesis Enabled for Calibration (`PARSEC`) analysis interface. Functions herein were
designed for use with the DSA-110 file system and naming conventions, and therefore should be
used with caution. This is particularly applicable to calibration functions; functions for
interfacing with filterbank data should be portable to any system. The DSA-110 polarization pipeline
which was built with this module is describe in detail in Sherman et al. 2024a (https://doi.org/10.3847/1538-4357/ad275e).

## Requirements

The following modules are required:

- numpy
- matplotlib
- pylab
- sigpyproc
- os
- sys
- pickle
- argparse

These can be installed by running:

pip install -r dsa-110-pol/requirements.txt

Alternatively, a copy of the conda environment used for development can be activated with:

```
conda activate dsa-110-pol/casa38dsapol
```

## Quick Start

Clone `dsa110-pol` locally with:

```
git clone git@github.com:dsa110/dsa110-pol.git
```

Add the following lines to your .bashrc file to initialize environment variables:

```
export DSAPOLDIR="PATH-TO-DSA110-POL"/dsa110-pol
export DSA110DIR="PATH-TO-DSA110-CANDIDATES"/dsa-110
export DSAFRBDIR="PATH-TO-FRB-FILTERBANKS"/FRBdata/
export DSACALDIR="PATH-TO-POLARIZATION-CALIBRATOR-VOLTAGES"/polcal_voltages/
```

Replacing with the paths specific to your machine. Next install `dsa110-pol` with `pip`:

```
cd dsa110-pol
pip install .
```

This should have created the following directories within folder that encloses `dsa110-pol`:

- `dsapol_logfiles`
	- `scat_files`
	- `RM_files`
- `dsapol_tables`
- `dsapol_polcal`

along with logfiles for use by the `PARSEC` interface. `dsapol_tables` and `dsapol_polcal` must be populated with DSA-110-specific,
proprietary data in order to use the polarization calibration functions and `PARSEC` directly. Please submit an issue requesting 
these files with details on your intended use of this data.

From a Python script, `dsapol` can be imported using:

```
$python
>>>from dsapol import dsapol
```

to use the user-facing functions. The other modules (as described below) are primarily for use by the `PARSEC` interface. `PARSEC` can be accessed
from https://code.deepsynoptic.org/, or, if the required data files are present, by running the following:

```
$cd interface
$mercury run PARSEC_Interface-InteractiveMercury-V4.ipynb
```

## Modules

The following sub-modules are included:
- `dsapol`: Modules for polarization analysis with scripts and with the `PARSEC` interface.
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
- `dsapol96`: TO DO: IMPLEMENT UPDATED MODULE FOR 96-ANTENNA ARRAY
- `offline_beamforming`: `bash` scripts used to beamform baseband voltage data using the `toolkit` cuda kernel developed by Vikram Ravi et al.
	- `move_cal_voltages.bash`: Moves baseband voltage data from T3 sub-system to the local file system
	- `run_beamformer_offline_bfweightsupdate_cals_sb.bash`: Beamforms baseband voltages for calibrator observations at low resolution for 256 synthesized beams
	- `run_beamformer_visibs_bfweightsupdate_cals_sb.bash`: Beamforms baseband voltages for calibrator observations at high resolution at given synthesized beam
	- `run_beamformer_visibs_bfweightsupdate_sb.bash`: Beamforms baseband voltages for FRB candidates at high resolution at given synthesized beam
- `interface`: High-level juypter notebook for `PARSEC` interface implemented with `mercury` (https://github.com/mljar/mercury)
	- `PARSEC_Interface-InteractiveMercury-V4.ipynb`: `PARSEC` notebook containing formatting of all screens defined in `dsapol/parsec.py`
	- `kill_mercury.sh`: kills all sub-processes spawned by `mercury`
	- `last_cal_metadata`: info about averaging used to make most recent pol calibration solution
	- `IONEXdata`: atmospheric data from the NASA Earthdata database for ionospheric RM estimation with `RMExtract` (https://github.com/lofar-astron/RMextract)
- `scripts`, `tests`: DEPRECATED, TO DO: REPLACE WITH UPDATED UNIT TESTS

