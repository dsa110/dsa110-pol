import json
f = open("directories.json","r")
dirs = json.load(f)
f.close()


logfile = dirs["logs"] + "scatscint_logfile.txt" #"/media/ubuntu/ssd/sherman/code/dsapol_logfiles/archive_logfile.txt"


