# interface

This module contains high-level interface code used by the `PARSEC` dashboard. These are not meant for direct user interaction, but rather are used by `PARSEC` to create the GUI.

## Structure

- `PARSEC_Interface.ipynb`: high-level jupyter notebook for use by `PARSEC`
- `kill_mercury.sh`: kills all sub-processes spawned by `mercury`
- `last_cal_metadata`: info about averaging used to make most recent pol calibration solution
- `IONEXdata`: atmospheric data from the NASA Earthdata database for ionospheric RM estimation with `RMExtract` (https://github.com/lofar-astron/RMextract)

## Usage

To start the `PARSEC` GUI on a specific port (`PORT`), run the following command from terminal:

```
$mercury run PARSEC_Interface.ipynb PORT
```

The GUI is then accessible via http://localhost:PORT/ . Shutdown `PARSEC` using `Ctrl^C`. To ensure all sub-processes spawned by mercury are terminated, run:

```
$./kill_mercury.sh
```






