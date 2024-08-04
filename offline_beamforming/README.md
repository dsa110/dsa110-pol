# offline\_beamforming

This module contains scripts to beamform baseband voltage data for calibrators and Fast Radio Burst (FRB) candidates in order to form full-Stokes polarization filterbanks.

## Structure

For each script we provide the required command-line arguments; rudimentary parsing is used and therefore arguments should be provided in order and without keywords:

- `move_cal_voltages.bash`: Moves baseband voltage data from T3 sub-system to the local file system
	- `[calname]`: VLA name of the calibrator (e.g. 3C48, 3C286)
	- `[obs_id]`: 3-letter identifier assigned to voltage dump to be copied (e.g. `abc`)
	- `[date]`: date of observation in format `YYYY-MM-DD` (e.g. 2024-08-04)
	- `[outpath]`: path to target location
	- `[inpath]`: path to source location
- `run_beamformer_offline_bfweightsupdate_cals_sb.bash`: Beamforms baseband voltages for calibrator observations at low resolution for 256 synthesized beams
	- `[obs_id]`: 3-letter identifier assigned to voltage dump (e.g. `abc`)
	- `[mjd]`: Mean Julian Date of observation
	- `[calname]`: VLA name of the calibrator (e.g. 3C48, 3C286)
	- `[isot]`: Date and time of beamformer weights to be applied in ISOT format (e.g. 2024-08-04T01:04:32)
	- `[date]`: date of observation in format `YYYY-MM-DD` (e.g. 2024-08-04); note this may or may not match the date for beamformer weights
- `run_beamformer_visibs_bfweightsupdate_cals_sb.bash`: Beamforms baseband voltages for calibrator observations at high resolution at given synthesized beam
	- `[N/A]`: not used, just use `NA`
	- `[obs_id]`: 3-letter identifier assigned to voltage dump (e.g. `abc`)
	- `[calname]`: VLA name of the calibrator (e.g. 3C48, 3C286)
	- `[isot]`: Date and time of beamformer weights to be applied in ISOT format (e.g. 2024-08-04T01:04:32)
	- `[beam]`: Beam in which source was detected (between 1-256)
        - `[mjd]`: Mean Julian Date of observation
	- `[DM]`: Dispersion measure, set to 0 for continuum sources
        - `[date]`: date of observation in format `YYYY-MM-DD` (e.g. 2024-08-04)
- `run_beamformer_visibs_bfweightsupdate_sb.bash`: Beamforms baseband voltages for FRB candidates at high resolution at given synthesized beam
	- `[N/A]`: not used, just use `NA`
	- `[candname]`: 12-digit candidate name assigned to the FRB candidate (e.g. 220307aaae)
	- `[nickname]`: nickname assigned to the FRB candidate for ease of identification (e.g. alex)
	- `[isot]`: Date and time of beamformer weights to be applied in ISOT format (e.g. 2024-08-04T01:04:32)
	- `[beam]`: Beam in which source was detected (between 1-256)
        - `[mjd]`: Mean Julian Date of observation
	- `[DM]`: Dispersion measure in pc/cc

## Usage

Each script can be run in a `bash` shell, e.g. to make filterbanks for an FRB, run:

```
$./run_beamformer_visibs_bfweightsupdate_sb.bash NA alex 220307aaae 2022-03-07T01:04:32 138 59645.8456340 499.15
```


