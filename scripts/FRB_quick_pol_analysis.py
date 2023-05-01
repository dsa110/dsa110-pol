from dsapol import dsapol
import numpy as np
from matplotlib import pyplot as plt
import sys
import getopt

usage = "Usage: \n -h: print this message \n --candname: trigger name (e.g. 20121102abcd) \n --nickname: given name (e.g. Clare) \n --ibeam: detected beam \n --ibox: estimated width in samples from Heimdall \n --buff: buffer in samples around pulse \n --RA: RA in degrees \n --DEC: DEC in degrees \n --caldate: date of polarization calibration to use (e.g. 22-12-12) \n --n_t: factor to downsample by in time \n --n_f: factor to downsample by in frequency"

    
def main():
    args = sys.argv[1:]
    print(args)    
    #defaults
    buff = 5
    n_t = 1
    n_f = 1
    caldate = "22-12-18"

    opts = ['candname=','nickname=','ibeam=','ibox=','buff=','RA=','DEC=','caldate=','n_t=','n_f=']
    optlist, args = getopt.getopt(args, 'h',opts)
    print(optlist)
    print(args)

    for o,a in optlist:
        if o == '-h':
            print(usage)
            sys.exit(0)
        elif o == '--candname':
            ids = a
        elif o == '--nickname':
            nickname = a
        elif o == '--ibeam':
            ibeam = int(a)
        elif o == '--ibox':
            width_native = int(a)
        elif o == '--buff':
            buff = int(a)
        elif o == '--RA':
            RA = float(a)
        elif o == '--DEC':
            DEC = float(a)
        elif o == '--caldate':
            caldate = a
        elif o == '--n_t':
            n_t = int(a)
        elif o == '--n_f':
            n_f = int(a)
        else:
            print("Invalid Argument")
            print(usage)
            sys.exit(0)
    if len(optlist) <= len(opts)-4:
        print("Invalid number of arguments")
        print(usage)
        sys.exit(0)
        
    print(dsapol.FRB_quick_analysis(ids=ids, nickname=nickname, ibeam=ibeam, width_native=width_native, buff=buff, RA=RA, DEC=DEC, caldate=caldate, n_t=n_t, n_f=n_f,weighted=False,plot=True))


if __name__ == '__main__':
    main()
