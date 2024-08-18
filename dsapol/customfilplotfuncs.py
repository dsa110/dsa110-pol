import numpy as np
from dsapol import dsapol
import mercury as mr
from sigpyproc import FilReader
from sigpyproc.Filterbank import FilterbankBlock
from sigpyproc.Header import Header
import matplotlib
#matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import os

"""
Plotting parameters
"""
fsize=30#35
fsize2=30
plt.rcParams.update({
                    'font.size': fsize,
                    'font.family': 'sans-serif',
                    'axes.labelsize': fsize,
                    'axes.titlesize': fsize,
                    'xtick.labelsize': fsize,
                    'ytick.labelsize': fsize,
                    'xtick.direction': 'in',
                    'ytick.direction': 'in',
                    'xtick.top': True,
                    'ytick.right': True,
                    'lines.linewidth': 1,
                    'lines.markersize': 5,
                    'legend.fontsize': fsize2,
                    'legend.borderaxespad': 0,
                    'legend.frameon': False,
                    'legend.loc': 'lower right'})
import json
f = open(os.environ['DSAPOLDIR'] + "directories.json","r")
dirs = json.load(f)
f.close()

"""
These functions are adapted from https://github.com/dsa110/dsa110-T3/blob/main/dsaT3/filplot_funcs.py#L69 for PARSEC
"""

def custom_dm_transform(data, dm_max=20,
                 dm_min=0, dm0=None, ndm=64, 
                 freq_ref=None, downsample=16):
    """ Transform freq/time data to dm/time data.                                                                                                """

    ntime = data.shape[1]

    dms = np.linspace(dm_min, dm_max, ndm, endpoint=True)

    if dm0 is not None:
        dm_max_jj = np.argmin(abs(dms-dm0))
        dms += (dm0-dms[dm_max_jj])

    data_full = np.zeros([ndm, ntime//downsample])

    for ii, dm in enumerate(dms):
        dd = data.dedisperse(dm)
        _dts = np.mean(dd,axis=0)
        data_full[ii] = _dts[:ntime//downsample*downsample].reshape(ntime//downsample, downsample).mean(1)

    return data_full, dms


def custom_read_fil_data_dsa(fn, start=0, stop=1):
    """ Read in filterbank data
    """
    fil_obj = FilReader(fn)
    header = fil_obj.header
    delta_t = fil_obj.header['tsamp'] # delta_t in seconds
    fch1 = header['fch1']
    nchans = header['nchans']
    foff = header['foff']
    fch_f = fch1 + nchans*foff
    freq = np.linspace(fch1,fch_f,nchans)
    try:
        data = fil_obj.readBlock(start, stop)
    except(ValueError):
        data = 0

    return data, freq, delta_t, header

def custom_proc_cand_fil(fnfil, dm, ibox, snrheim=-1,
                  pre_rebin=1, nfreq_plot=64,
                  heim_raw_tres=1,
                  rficlean=False, ndm=64,
                  norm=True, freq_ref=None,start_time=0,stop_time=4):

    """ Take filterbank file path, preprocess, and
    plot trigger

    Parameters:
    ----------

    fnfil   : str
        path to .fil file
    DM      : float
        dispersion measure of trigger
    ibox    : int
        preferred boxcar width
    snrheim : float
        S/N of candidate found by Heimdall
    pre_rebin : int
        rebin in time by this factor *before* dedispersion (saves time)
    nfreq_plot : int
        number of frequency channels in output
    heim_raw_tres : 32
    """

    header = custom_read_fil_data_dsa(fnfil, 0, 1)[-1]
    # read in 4 seconds of data
    #nsamp = int(4.0/header['tsamp'])
    start = int(start_time/header['tsamp'])
    stop = int(stop_time/header['tsamp'])
    data, freq, delta_t_raw, header = custom_read_fil_data_dsa(fnfil, start=start,
                                                       stop=stop)

    nfreq0, ntime0 = data.shape

    if pre_rebin>1:
        # Ensure that you do not pre-downsample by more than the total boxcar
        pre_rebin = min(pre_rebin, ibox*heim_raw_tres)
        data = data.downsample(pre_rebin)

    datadm0 = data.copy()

    if rficlean:
        data = cleandata(data, clean_type='aladsa')

    tsdm0 = np.mean(data,axis=0)

    dm_err = ibox / 1.0 * 25.
    dm_err = 250.0
    datadm, dms = custom_dm_transform(data, dm_max=dm+dm_err,
                               dm_min=dm-dm_err, dm0=dm, ndm=int(ndm),
                               freq_ref=freq_ref,
                               downsample=int(heim_raw_tres*ibox//pre_rebin))
    data = data.dedisperse(dm)
    data = data.downsample(heim_raw_tres*ibox//pre_rebin)
    data = data.reshape(nfreq_plot, data.shape[0]//nfreq_plot,
                        data.shape[1]).mean(1)

    if norm:
        data = data-np.median(data,axis=1,keepdims=True)
        data /= np.std(data)

    return data, datadm, tsdm0, dms, datadm0

def custom_filplot(fn, dm, ibox, multibeam=None, figname=None,
             ndm=32, suptitle='', heimsnr=-1,
             ibeam=-1, rficlean=True, nfreq_plot=32,
             classify=False, heim_raw_tres=1,
             showplot=True, save_data=True, candname=None,
             fnT2clust=None, imjd=0, injected=False, fast_classify=False,pre_rebin=1,start_time=0,stop_time=4,fil_dedispersed=False):
    """
    This is a modified version of the filplot function found in https://github.com/dsa110/dsa110-T3/blob/main/dsaT3/filplot_funcs.py,
    that displays the dynamic spectrum, DM transform, and power and intensity vs time plots.
    Vizualize FRB candidates on DSA-110
    fn is filterbnak file name.
    dm is dispersion measure as float.
    ibox is timecar box width as integer.

    """

    #read data and pre-process
    #if dedispersed, use DMs around 0
    if fil_dedispersed:
        dataft, datadm, tsdm0, dms, datadm0 = custom_proc_cand_fil(fn, 0, ibox, snrheim=-1,
                                               pre_rebin=1, nfreq_plot=nfreq_plot,
                                               ndm=ndm, rficlean=rficlean,
                                               heim_raw_tres=heim_raw_tres,start_time=start_time,stop_time=stop_time)
        dms = np.array(dms) + dm
    else:
        dataft, datadm, tsdm0, dms, datadm0 = custom_proc_cand_fil(fn, dm, ibox, snrheim=-1,
                                               pre_rebin=1, nfreq_plot=nfreq_plot,
                                               ndm=ndm, rficlean=rficlean,
                                               heim_raw_tres=heim_raw_tres,start_time=start_time,stop_time=stop_time)

    #if data is read from high res filterbanks, use pre_rebin = 1 and bin the data
    if pre_rebin != 1:
        dataft = dsapol.avg_time(dataft,pre_rebin)
        datadm0 = dsapol.avg_time(datadm0,pre_rebin)
    beam_time_arr = None
    multibeam_dm0ts = None

    datats = dataft.mean(0)
    datadmt = datadm
    if pre_rebin != 1:
        datadmt = dsapol.avg_time(datadmt,pre_rebin)
    dms = [dms[0],dms[-1]]

    #plotting
    classification_dict = {'prob' : [],
                           'snr_dm0_ibeam' : [],
                           'snr_dm0_allbeam' : []}
    datats /= np.std(datats[datats!=np.max(datats)])
    nfreq, ntime = dataft.shape
    if start_time == 0 and stop_time == 4:
        xminplot,xmaxplot = 500.-300*ibox/16.,500.+300*ibox/16 # milliseconds
        if xminplot<0:
            xmaxplot=xminplot+500+300*ibox/16
            xminplot=0
    else:
        xminplot,xmaxplot = -start_time*1e3 + 500.-300*ibox/16.,-start_time*1e3 + 500.+300*ibox/16 # milliseconds
        if xminplot<0:
            xmaxplot=xminplot+500+169*ibox/32
            xminplot=0

#    xminplot,xmaxplot = 0, 1000.
    dm_min, dm_max = dms[0], dms[1]
    tmin, tmax = 0., 1e3*dataft.header['tsamp']*ntime*pre_rebin
    freqmax = dataft.header['fch1']
    freqmin = freqmax + dataft.header['nchans']*dataft.header['foff']
    freqs = np.linspace(freqmin, freqmax, nfreq)
    tarr = np.linspace(tmin, tmax, ntime)
    fig, axs = plt.subplots(2, 2, figsize=(24,12))#, constrained_layout=True)

    #dynamic spectrum
    extentft=[tmin,tmax,freqmin,freqmax]
    axs[0][0].imshow(dataft, aspect='auto',extent=extentft, interpolation='nearest')
    DM0_delays = xminplot + dm * 4.15E6 * (freqmin**-2 - freqs**-2)
    axs[0][0].plot(DM0_delays, freqs, c='r', lw='2', alpha=0.35)
    axs[0][0].set_xlim(xminplot,xmaxplot)
    axs[0][0].set_xlabel('Time (ms)')
    axs[0][0].set_ylabel('Freq (MHz)')

    #DM transform
    extentdm=[tmin, tmax, dm_min, dm_max]
    axs[0][1].imshow(datadmt[::-1], aspect='auto',extent=extentdm)
    axs[0][1].set_xlim(xminplot,xmaxplot)
    axs[0][1].set_xlabel('Time (ms)')
    axs[0][1].set_ylabel(r'DM (pc cm$^{-3}$)')

    #Power vs time
    axs[1][0].plot(tarr, datats)
    axs[1][0].grid('on', alpha=0.25)
    axs[1][0].set_xlabel('Time (ms)')
    axs[1][0].set_ylabel(r'Power ($\sigma$)')
    axs[1][0].set_xlim(xminplot,xmaxplot)
    axs[1][0].text(-start_time + 0.501*(xminplot+xmaxplot), -start_time + 0.5*(max(datats)+np.median(datats)),
            'Heimdall S/N : %0.1f\nHeimdall DM : %d\
            \nHeimdall ibox : %d\nibeam : %d' % (heimsnr,dm,ibox,ibeam),
            fontsize=24, verticalalignment='center')


    #Intensity vs time
    datadm0 -= np.median(datadm0.mean(0))
    datadm0_sigmas = datadm0.mean(0)/np.std(datadm0.mean(0)[-500:])
    snr_dm0ts_iBeam = np.max(datadm0_sigmas)
    axs[1][1].plot(np.linspace(0, tmax, len(datadm0[0])), datadm0_sigmas, c='k')
    classification_dict['snr_dm0_ibeam'] = snr_dm0ts_iBeam
    axs[1][1].set_xlabel('Time (ms)')
    axs[1][1].set_ylabel(r'Power ($\sigma$)')
    axs[1][1].legend(['DM=0 Timestream'], loc=2, fontsize=24)
    pre_suptitle = ''
    if pre_rebin > 1:
        pre_suptitle += '\n[Plots made from high resolution dedispersed Level3 filterbanks]'
    fig.suptitle(suptitle+pre_suptitle, color='C1',fontsize=24)
    if figname is not None:
        plt.savefig(figname)
    if showplot:
        plt.show()
    else:
        plt.close()
    #restore default backend
    #matplotlib.use(mplbackend)
    #from matplotlib import pyplot as plt
    return    
