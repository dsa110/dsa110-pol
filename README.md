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

replacing with the paths specific to your machine. One additional environment variable, `RMSYNTHTOKEN` is required to push `tqdm` progress bars to slack for background RM synthesis processes. This is already defined on relevant DSA-110 machines; please submit an issue in this repository if this token is needed on any other machine.

Next install `dsa110-pol` with `pip`:

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
	- `dsapol`
	- `parsec`
	- `polcal`
	- `polbeamform`
	- `customfilplotfuncs`
	- `dedisp`
	- `filt`
	- `RMcal`
	- `scatscint`
	- `rmtablefuncs`
	- `budget`
- `dsapol96`: TO DO: IMPLEMENT UPDATED MODULE FOR 96-ANTENNA ARRAY
- `offline_beamforming`: `bash` scripts used to beamform baseband voltage data using the `toolkit` cuda kernel developed by Vikram Ravi et al.
	- `move_cal_voltages.bash`
	- `run_beamformer_offline_bfweightsupdate_cals_sb.bash`
	- `run_beamformer_visibs_bfweightsupdate_cals_sb.bash`
	- `run_beamformer_visibs_bfweightsupdate_sb.bash`
- `interface`: High-level juypter notebook for `PARSEC` interface implemented with `mercury` (https://github.com/mljar/mercury)
	- `PARSEC_Interface-InteractiveMercury-V4.ipynb`
	- `kill_mercury.sh`
	- `last_cal_metadata`
	- `IONEXdata`
- `scripts`, `tests`: DEPRECATED, TO DO: REPLACE WITH UPDATED UNIT TESTS

