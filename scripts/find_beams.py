from dsapol import polcal
import pickle as pkl
import sys


#get comm line args
calname = sys.argv[1]
caldate = sys.argv[2]

#make beams
beam_dict = polcal.get_source_beams(caldate,calname)
#beam_dict = dict()

#also get mjds and beamformer weights
for k in beam_dict.keys():
    mjd = polcal.get_cal_mjd(calname,caldate,str(k)[len(calname):])
    bf = polcal.get_bfweight_isot(calname,mjd)
    beam_dict[k]['mjd'] = mjd
    beam_dict[k]['bf_weights'] = bf

#save to a pkl file if not already
f = open(polcal.output_path + calname + "_" + caldate + "/" + calname + "_" + caldate + "_beams.pkl",'wb')
pkl.dump(beam_dict,f)
f.close()
