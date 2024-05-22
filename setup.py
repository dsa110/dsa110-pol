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
