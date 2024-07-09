from setuptools import setup
#from dsautils.version import get_git_version


#From Casey/Dana's template; uncomment when added to git
"""
try:
    version = get_git_version()
    assert version is not None
except (AttributeError, AssertionError):
    version = '1.0.0'
"""

#Note, to add later:
#url
#author
setup(name='dsa-110_pol-dev',
      version='1.0.0',
      description='DSA-110 Polarization Utilities',
      packages=['dsapol','dsapol96'],
      install_requires=[
          'numpy',
          'matplotlib',
          'sigpyproc'
          ],
      scripts = [
          'scripts/plot_pol.py',
          'scripts/cal_pol.py',
          'scripts/FRB_upper_limits.py',
          'scripts/process_all_FRBs.py']

     )

import os
import json
print("Creating logfiles")
os.system("rm -r ../dsapol_logfiles")
os.system("mkdir ../dsapol_logfiles")
os.system("touch ../dsapol_logfiles/beamform_logfile.txt")
os.system("touch ../dsapol_logfiles/dedisp_logfile.txt")
os.system("touch ../dsapol_logfiles/polcal_logfile.txt")
os.system("touch ../dsapol_logfiles/filter_logfile.txt")
os.system("touch ../dsapol_logfiles/RMcal_logfile.txt")
os.system("touch ../dsapol_logfiles/dsapol_logfile.txt")
os.system("touch ../dsapol_logfiles/archive_logfile.txt")
os.system("touch ../dsapol_logfiles/scatscint_logfile.txt")

os.system("> ../dsapol_logfiles/beamform_logfile.txt")
os.system("> ../dsapol_logfiles/dedisp_logfile.txt")
os.system("> ../dsapol_logfiles/polcal_logfile.txt")
os.system("> ../dsapol_logfiles/filter_logfile.txt")
os.system("> ../dsapol_logfiles/RMcal_logfile.txt")
os.system("> ../dsapol_logfiles/dsapol_logfile.txt")
os.system("> ../dsapol_logfiles/archive_logfile.txt")
os.system("> ../dsapol_logfiles/scatscint_logfile.txt")

os.system("mkdir ../dsapol_logfiles/RM_files")
os.system("mkdir ../dsapol_logfiles/scat_files")
import numpy as np
#np.save("../dsapol_logfiles/RM_files/input_spectrum.npy",np.zeros((0,0)))
#np.save("../dsapol_logfiles/RM_files/output_spectrum.npy",np.zeros((2,0)))
#np.save("../dsapol_logfiles/RM_files/output_values.npy",np.nan*np.ones(3))
#np.save("../dsapol_logfiles/RM_files/trial_rm.npy",np.zeros(0))

print("Finding working directories")
os.system("pwd > cwdpath.txt")
f = open("cwdpath.txt","r")
cwd = f.read()[:-1] + "/"
f.close()

dirs = {"cwd":cwd,
        "polcal":"/media/ubuntu/ssd/sherman/code/",
        "candidates":"/dataz/dsa110/candidates/",
        "T3":"/dataz/dsa110/T3/",
        "polcal_voltages":"/media/ubuntu/ssd/sherman/polcal_voltages/",
        "data":"/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/",
        "logs":cwd[:cwd.index("dsa110-pol")] + "dsapol_logfiles/",
        "gen_bfweights":"/dataz/dsa110/operations/beamformer_weights/generated/",
        "dsastorageFRBDir":"user@dsa-storage.ovro.pvt:/home/user/data/candidates/candidates/",
        "dsastorageCALDir":"user@dsa-storage.ovro.pvt:/mnt/data/sherman_oldpolcal_voltages/",
        "dsastorageFILDir":"user@dsa-storage.ovro.pvt:/mnt/data/dsa110/T1/"}

print(dirs)
f = open(dirs["polcal"] + "directories.json","w")
json.dump(dirs,f)
f.close()

f = open("interface/directories.json","w")
json.dump(dirs,f)
f.close()

f = open("scripts/directories.json","w")
json.dump(dirs,f)
f.close()
