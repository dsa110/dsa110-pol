# dsa110-pol
DSA-110 polarization utilities


Authors: Myles Sherman, Liam Connor, Casey Law, Vikram Ravi, Dana Simard
Last Upated: 2024-08-04

This library contains functions and interfaces for polarization analysis of FRBs. Functions herein were
designed for use with the DSA-110 file system and naming conventions, and therefore should be
used with caution. This is particularly applicable to calibration functions; functions for
interfacing with filterbank data should be portable to any system.

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

Add the following lines to your .bashrc file to initialize environment variables:

```
export DSAPOLDIR="PATH-TO-DSA110-POL"/dsa110-pol
export DSA110DIR="PATH-TO-DSA110-CANDIDATES"/dsa-110
export DSAFRBDIR="PATH-TO-FRB-FILTERBANKS"/FRBdata/
export DSACALDIR="PATH-TO-POLARIZATION-CALIBRATOR-VOLTAGES"/polcal_voltages/
```

Replacing with the paths specific to your machine.


