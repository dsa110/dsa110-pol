# dsa110-pol
DSA-110 polarization utilities


Authors: Myles Sherman, Liam Connor, Casey Law, Vikram Ravi, Dana Simard

This library contains functions for polarization analysis of FRBs. Functions herein were
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

pip install -r dsa-110_pol-dev/requirements.txt
