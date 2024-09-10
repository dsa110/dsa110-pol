import time
import numpy as np
import os
from astropy.time import Time
import json
import glob
f = open(os.environ['DSAPOLDIR'] + "directories.json","r")
dirs = json.load(f)
f.close()

logfile = dirs['logs'] + "cache_cow_log.txt"


#run continuously, look for logs and caches that are more than a week old and delete them. At some point, want to make this backup to dsastorage?

#revised function: leave only the most recent cache file for each FRB

def main():
    while True:
        f = open(logfile,"w")
        #list caches and log directories
        allcaches = glob.glob(dirs['cache']+"*")
        alllogs = glob.glob(dirs['logs']+"*T*")


        #look for caches with the same FRB name
        print("Scanning Cache...",file=f)
        now = Time.now()
        cache_dict = dict()
        for cache in allcaches:
            print(cache,file=f)
            cachetime = Time(cache[len(dirs['cache']):len(dirs['cache'])+23],format='isot')
            cacheFRB = cache[len(dirs['cache'])+24:]
            print(cacheFRB + cachetime.isot,file=f)
            if cacheFRB not in cache_dict.keys():
                cache_dict[cacheFRB] = [cachetime,cache]
            elif cachetime > cache_dict[cacheFRB][0]:
                os.system("rm -r " + cache_dict[cacheFRB][1])
                cache_dict[cacheFRB] = [cachetime,cache]
                print(now.isot + " cache_cow deleted CACHE " + cache_dict[cacheFRB][1][len(dirs['cache']):],file=f)
            else:
                os.system("rm -r " + cache)
                print(now.isot + " cache_cow deleted CACHE " + cache[len(dirs['cache']):],file=f)
        print("Remaining Caches:",file=f)
        print(cache_dict,file=f)
        """
        #check which ones older than 1 week
        lim = 7 #days
        now = Time.now()
        print("Scanning Cache...",file=f)
        for cache in allcaches:
            print(cache,file=f)
            cachetime = Time(cache[len(dirs['cache']):len(dirs['cache'])+23],format='isot')
            if np.abs((now-cachetime).value) >= lim:
                os.system("rm -r " + cache)
                print(now.isot + " cache_cow deleted CACHE " + cache[len(dirs['cache']):],file=f)
        """
        #check which ones older than 1 week
        lim = 7 #days
        now = Time.now()
        print("Scanning Logs...",file=f)
        for log in alllogs:
            print(log,file=f)
            logtime = Time(log[len(dirs['logs']):],format='isot')
            if np.abs((now-logtime).value) >= lim:
                os.system("rm -r " + log)
                print(now.isot + " cache_cow deleted LOG " + log[len(dirs['logs']):],file=f)
        f.close()

        time.sleep(86400)
    return

if __name__=="__main__":
    main()
