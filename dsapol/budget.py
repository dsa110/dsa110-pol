import numpy as np
from astropy.table import Table
import sys
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import pyne2001
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.cosmology import WMAP1,WMAP3,WMAP5,WMAP7,WMAP9,Planck13,Planck15,Planck18,default_cosmology
import astropy.units as u
from astropy.modeling import physical_models
from astropy.constants import m_p,m_e
from astroquery import vizier
import copy
from scipy.signal import peak_widths
from astroquery.simbad import Simbad
from dsapol.RMcal import logfile
import os
import json
from dl import queryClient as qc
f = open(os.environ['DSAPOLDIR'] + "directories.json","r")
dirs = json.load(f)
f.close()

"""
This contains helper functions for computing the DM and RM budget. For examples see DMhost_Estimation_Script_2024-04-11.ipynb
"""

# DM_IGM parameters from From Illustris simulations, Zhang+2021: https://iopscience.iop.org/article/10.3847/1538-4357/abceb9/pdf
z_vals = [0.1,0.2,0.3,0.4,0.5,0.7,1,1.5,2,2.4,3,3.5,4,4.4,5,5.2,5.5,5.8,6,6.5,7,8,9]
A_vals = [0.04721,0.005693,0.003584,0.002876,0.002423,0.001880,0.001456,0.001098,0.0009672,0.0009220,0.0008968,0.0008862,0.0008826,0.0008827,0.0008834,0.0008846,0.0008863,0.0008878,0.0008881,0.0008881,0.0008881,0.0008881,0.0008881]
C0_vals = [-13.17,-1.008,0.596,1.010,1.127,1.170,1.189,1.163,1.162,1.142,1.119,1.104,1.092,1.084,1.076,1.073,1.070,1.067,1.066,1.066,1.066,1.066,1.066]
sig_vals = [2.554,1.118,0.7043,0.5158,0.4306,0.3595,0.3044,0.2609,0.2160,0.1857,0.1566,0.1385,0.1233,0.1134,0.1029,0.09918,0.09481,0.09072,0.08971,0.08960,0.08952,0.08944,0.08941]
z_vals = np.array(z_vals)
A_vals = np.array(A_vals)
C0_vals = np.array(C0_vals)
sig_vals = np.array(sig_vals)

#signficance level for upper and lower errors
DEF_SIGLEVEL = 0.68 #(1sigma)

#halo DM estimate
DMHALOEST = 10 #pc/cc


#Marquart DM from Equation 6 of Zhang 2021
def DM_IGM_MEAN(z,res=1000):
    """
    This function computes the mean DM_igm by numerically integrating
    equation 6 from Zhang 2021 given the redshift.
    """
    #redshift axis
    z_axis = np.linspace(0,z,res)

    #planck cosmology parameters
    H0 = (67.8e-6)*1e5 #cm/s/pc
    G = (4.3e-3)*((1e5)**2)/(2e30) #pc (cm/s)^2/kg
    fd = 0.7
    Y = 0.25
    rho_c = (3*(H0**2)/(8*np.pi*G))  #(kg/pc^3)

    Omega_M = 0.308
    Omega_L = 0.692
    Omega_B = 0.048
    mp = 1.67e-27 #kg
    c = (3e8)*1e2 #cm/s

    #numerical integration
    DMest = 0
    for i in range(1,len(z_axis)):
        z_1 = z_axis[i-1]
        z_2 = z_axis[i]

        #integrand for both redshifts
        rho1 = rho_c*Omega_B*((1+z_1)**3)
        ne1 = (fd*rho1*(1 + (Y/2))/mp)/((3e18)**3)
        igrand1 = c*ne1/(H0*((1+z_1)**2)*np.sqrt((Omega_M*(1+z_1)**3) + Omega_L))

        rho2 = rho_c*Omega_B*((1+z_2)**3)
        ne2 = (fd*rho2*(1 + (Y/2))/mp)/((3e18)**3)
        igrand2 = c*ne2/(H0*((1+z_2)**2)*np.sqrt((Omega_M*(1+z_2)**3) + Omega_L))

        #trapezoidal rule
        DMest += (z_2-z_1)*(igrand1 + igrand2)/2

    return DMest

#standard Gaussian curve
def gaus(x,mu,sig):
    return np.exp(-(x-mu)**2/(2*sig**2))

#p_igm pdf from Zhang 2021 eqn 10 (https://iopscience.iop.org/article/10.3847/1538-4357/abceb9/pdf)
def PIGM_zp1(delta, z ,alpha=3, beta=3):
    """
    This function takes the DM parameter and redshift and returns
    the IGM DM distribution using Zhang 2021 eqn 10
    delta: DM_IGM/<DM_IGM>
    z: redshift
    """
    #print(z)
    #print('Using values for z = ' + str(z_vals[np.argmin(np.abs(z-z_vals))]))
    A,C,sig=A_vals[np.argmin(np.abs(z-z_vals))],C0_vals[np.argmin(np.abs(z-z_vals))],sig_vals[np.argmin(np.abs(z-z_vals))]#0.04721,-13.17,2.554
    x = (delta**-alpha - C)**2 / (2*alpha**2*sig**2)
    return A * delta**-beta * np.exp(-x)



#convolution method to derive DMhost
def DM_host_limits(DMobs,frb_z,frb_gl,frb_gb,res=10000,plot=False,DMhalo=DMHALOEST,siglevel=DEF_SIGLEVEL,intervener_DMs=[],intervener_DM_errs=[],intervener_zs=[]):
    """
    This function derives the host DM given an FRB's observed DM, redshift, and
    galactic position. It convolves the distributions of DM_obs, DM_MW (from NE2001), DM_IGM,
    and DM_host, and returns the median and 1sigma upper and lower limits of DM_host.
    DMobs: observed DM
    frb_z: measured redshift
    frb_gl: Galactic longitude in degrees
    frb_gb: Galactic latitude in degrees
    res: resolution of the DM axis for each DM PDF
    plot: if True plots the DM budget and DM host CDF, default False
    DM_halo: estimate for Galactic halo DM, default 10 pc/cc
    siglevel: confidence interval for error estimates, default 68% (1sigma)
    """
    #DM axis
    DM_axis = np.linspace(-2000,2000,res)


    #measured values
    #DMobs = DSAFRBDMS[DSAFRBNICKNAMES.index(frbtest)]
    DMIGMmean = DM_IGM_MEAN(frb_z,res=1000) #DSAFRBZS[DSAFRBNICKNAMES.index(frbtest)],res=1000)
    DMmw = pyne2001.get_dm(frb_gl,frb_gb,30) #frb_glpyne2001.get_dm(DSAFRBGLS[DSAFRBNICKNAMES.index(frbtest)],DSAFRBGBS[DSAFRBNICKNAMES.index(frbtest)],30)
    ztest = frb_z #DSAFRBZS[DSAFRBNICKNAMES.index(frbtest)]

    #get probability distributions
    if plot:
        plt.figure(figsize=(12,6))
    #observed DM
    Pobs = gaus(DM_axis,DMobs,0.1)
    Pobs[DM_axis < 0] = 0
    Pobs = Pobs/np.sum(Pobs*(DM_axis[1]-DM_axis[0]))
    #MW DM
    Pmw = gaus(DM_axis,DMmw,30)
    Pmw[DM_axis<0] = 0
    Pmw[DM_axis>DMobs] = 0
    Pmw = Pmw/np.sum(Pmw*(DM_axis[1]-DM_axis[0]))
    #IGM DM
    Pigm = PIGM_zp1(DM_axis/DMIGMmean,ztest)
    Pigm[DM_axis < 0] = 0
    Pigmfull = copy.deepcopy(Pigm)
    Pigm[DM_axis>DMobs] = 0
    Pigm = Pigm/np.sum(Pigm*(DM_axis[1]-DM_axis[0]))
    Pigmfull = Pigmfull/np.sum(Pigmfull*(DM_axis[1]-DM_axis[0]))

    #Milky Way and host halo -- from Ravi 2023, assume DMhalo = 35 +- 10, uniformly distributed
    Pmwhalo = np.zeros(len(DM_axis))
    Pmwhalo[np.logical_and(DM_axis < DMhalo+10,DM_axis > DMhalo-10)]= 1
    Pmwhalo = Pmwhalo/np.sum(Pmwhalo*(DM_axis[1]-DM_axis[0]))

    #ignore host galaxy halo
    #Phalo = np.zeros(len(DM_axis))
    #Phalo[np.logical_and(DM_axis < (DMhalo+10)/(1+ztest),DM_axis > (DMhalo-10)/(1+ztest))]= 1
    #Phalo = Phalo/np.sum(Phalo*(DM_axis[1]-DM_axis[0]))



    #convolve to get DM host
    P1 = np.convolve(Pobs,Pmw[::-1],mode="same")
    P1[DM_axis < 0] = 0
    P2 = np.convolve(P1,Pmwhalo[::-1],mode="same")
    P2[DM_axis < 0] = 0
    P3 = np.convolve(P2,Pigm[::-1],mode="same")
    P3[DM_axis < 0] = 0
    #P4 = np.convolve(P3,Phalo[::-1],mode="same")
    #P4[DM_axis < 0] = 0

    #intervener DMs
    Pints = []
    for i in range(len(intervener_DMs)):
        if ~np.isnan(intervener_DM_errs[i]) and intervener_DM_errs[i] != 0:
            Pint = gaus(DM_axis*(1+intervener_zs[i]),intervener_DMs[i],intervener_DM_errs[i])
        else:
            Pint = gaus(DM_axis*(1+intervener_zs[i]),intervener_DMs[i],(DM_axis[1]-DM_axis[0]))
        Pint[DM_axis*(1+intervener_zs[i])<0] = 0
        Pint[DM_axis*(1+intervener_zs[i])>DMobs] = 0
        Pint = Pint/np.sum(Pint*(DM_axis[1]-DM_axis[0])*(1+intervener_zs[i]))
        Pints.append(Pint)
        P3 = np.convolve(P3,Pint[::-1],mode='same')
        P3[DM_axis<0] = 0
    Phost = copy.deepcopy(P3)
    Phost = Phost/np.sum(Phost*(DM_axis[1]*(1+ztest)-DM_axis[0]*(1+ztest)))

    if plot:
        plt.plot(DM_axis,Pobs/np.nanmax(Pobs),label=r'$DM_{obs}$')
        plt.plot(DM_axis,Pmw/np.nanmax(Pmw),label=r'$DM_{MW}$')
        plt.plot(DM_axis,Pigm/np.nanmax(Pigm),label=r'$DM_{IGM}$')
        for i in range(len(intervener_DMs)):
            plt.plot(DM_axis*(1+intervener_zs[i]),Pints[i]/np.nanmax(Pints[i]),label=r'$DM_{{int,{i}}}/(1+z_{{int,{i}}})$'.format(i=i))
        plt.plot(DM_axis,Pmwhalo/np.nanmax(Pmwhalo),label=r'$DM_{MW,halo}$')
        plt.plot(DM_axis,Phost/np.nanmax(Phost),label=r'$DM_{host}$',linewidth=4)
        plt.plot(DM_axis*(1+ztest),Phost/np.nanmax(Phost),label=r'$DM_{host}(1+z)$',linewidth=4)
        plt.xlim(0,DMobs*2)
        plt.legend(loc='upper right')
        plt.xlabel("DM")
        plt.ylabel("PDF")
        plt.show()
    #expected value
    Phost_exp = 0
    Pigm_exp = 0
    for i in range(len(DM_axis)-1):
        DM1 = DM_axis[i]*(1+ztest)
        DM2 = DM_axis[i+1]*(1+ztest)
        P1 = Phost[i]
        P2 = Phost[i+1]
        dDM = (DM2-DM1)#*(1+ztest)
        Phost_exp += dDM*((DM1*P1) + (DM2*P2))/2
        
        DM1 = DM_axis[i]
        DM2 = DM_axis[i+1]
        P1 = Pigm[i]
        P2 = Pigm[i+1]
        dDM = (DM2-DM1)
        Pigm_exp += dDM*((DM1*P1) + (DM2*P2))/2

    cumdisthost = np.cumsum(Phost)/np.sum(Phost)
    #print("calculating " + str(((1-siglevel)/2)) + " and"+ str((1-((1-siglevel)/2))) + " percentiles")
    low = (DM_axis*(1 + ztest))[np.argmin(abs(cumdisthost-((1-siglevel)/2)))]
    upp = (DM_axis*(1 + ztest))[np.argmin(abs(cumdisthost-(1-((1-siglevel)/2))))]

    cumdistigm = np.cumsum(Pigm)/np.sum(Pigm)
    igmlow = (DM_axis)[np.argmin(abs(cumdistigm-((1-siglevel)/2)))]
    igmupp = (DM_axis)[np.argmin(abs(cumdistigm-(1-((1-siglevel)/2))))]

    if plot:
        plt.figure(figsize=(24,12))
        plt.plot(DM_axis,cumdisthost)
        plt.plot(DM_axis,Phost)
        plt.axvline(low,color="red")
        plt.axvline(upp,color="red")
        plt.show()
    return Phost_exp,low,upp,{"obs":DMobs,"obserr":0.1,"MW":DMmw,"MWerr":30,"halo":DMhalo,"haloerr":10,
                              "IGM":Pigm_exp,"IGMlowerr":Pigm_exp-igmlow,"IGMupperr":igmupp-Pigm_exp}


#convolution method to derive DMhost distribution
def DM_host_dist(DMobs,frb_z,frb_gl,frb_gb,res=10000,plot=False,DMhalo=DMHALOEST,siglevel=DEF_SIGLEVEL,intervener_DMs=[],intervener_DM_errs=[],intervener_zs=[],save=False,savedir='./'):
    """
    This function derives the host DM PDF given an FRB's observed DM, redshift, and
    galactic position. It convolves the distributions of DM_obs, DM_MW (from NE2001), DM_IGM,
    and DM_host, and returns the PDF and DM axis
    DMobs: observed DM
    frb_z: measured redshift
    frb_gl: Galactic longitude in degrees
    frb_gb: Galactic latitude in degrees
    res: resolution of the DM axis for each DM PDF
    plot: if True plots the DM budget and DM host CDF, default False
    DM_halo: estimate for Galactic halo DM, default 10 pc/cc
    siglevel: confidence interval for error estimates, default 68% (1sigma)
    """
    DM_axis = np.linspace(-2000,2000,res)


    #measured values
    #DMobs = DSAFRBDMS[DSAFRBNICKNAMES.index(frbtest)]
    DMIGMmean = DM_IGM_MEAN(frb_z,res=1000) #DSAFRBZS[DSAFRBNICKNAMES.index(frbtest)],res=1000)
    DMmw = pyne2001.get_dm(frb_gl,frb_gb,30) #frb_glpyne2001.get_dm(DSAFRBGLS[DSAFRBNICKNAMES.index(frbtest)],DSAFRBGBS[DSAFRBNICKNAMES.index(frbtest)],30)
    ztest = frb_z #DSAFRBZS[DSAFRBNICKNAMES.index(frbtest)]

    if plot:
        plt.figure(figsize=(18,6))

    #get probability distributions
    #observed DM
    Pobs = gaus(DM_axis,DMobs,0.1)
    Pobs[DM_axis < 0] = 0
    Pobs = Pobs/np.sum(Pobs*(DM_axis[1]-DM_axis[0]))
    #MW DM
    Pmw = gaus(DM_axis,DMmw,30)
    Pmw[DM_axis<0] = 0
    Pmw[DM_axis>DMobs] = 0
    Pmw = Pmw/np.sum(Pmw*(DM_axis[1]-DM_axis[0]))
    #IGM DM
    Pigm = PIGM_zp1(DM_axis/DMIGMmean,ztest)
    Pigm[DM_axis < 0] = 0
    Pigmfull = copy.deepcopy(Pigm)
    Pigm[DM_axis>DMobs] = 0
    Pigm = Pigm/np.sum(Pigm*(DM_axis[1]-DM_axis[0]))
    Pigmfull = Pigmfull/np.sum(Pigmfull*(DM_axis[1]-DM_axis[0]))

    #Milky Way and host halo -- from Ravi 2023, assume DMhalo = 35 +- 10, uniformly distributed
    Pmwhalo = np.zeros(len(DM_axis))
    Pmwhalo[np.logical_and(DM_axis < DMhalo+10,DM_axis > DMhalo-10)]= 1
    Pmwhalo = Pmwhalo/np.sum(Pmwhalo*(DM_axis[1]-DM_axis[0]))

    #ignore host halo
    #Phalo = np.zeros(len(DM_axis))
    #Phalo[np.logical_and(DM_axis < (DMhalo+10)/(1+ztest),DM_axis > (DMhalo-10)/(1+ztest))]= 1
    #Phalo = Phalo/np.sum(Phalo*(DM_axis[1]-DM_axis[0]))






    #convolve to get DM host
    P1 = np.convolve(Pobs,Pmw[::-1],mode="same")
    P1[DM_axis < 0] = 0
    P2 = np.convolve(P1,Pmwhalo[::-1],mode="same")
    P2[DM_axis < 0] = 0
    P3 = np.convolve(P2,Pigm[::-1],mode="same")
    P3[DM_axis < 0] = 0
    #intervener DMs
    Pints = []
    for i in range(len(intervener_DMs)):
        if ~np.isnan(intervener_DM_errs[i]) and intervener_DM_errs[i] != 0:
            Pint = gaus(DM_axis*(1+intervener_zs[i]),intervener_DMs[i],intervener_DM_errs[i])
        else:
            Pint = gaus(DM_axis*(1+intervener_zs[i]),intervener_DMs[i],(DM_axis[1]-DM_axis[0]))
        Pint[DM_axis*(1+intervener_zs[i])<0] = 0
        Pint[DM_axis*(1+intervener_zs[i])>DMobs] = 0
        Pint = Pint/np.sum(Pint*(DM_axis[1]-DM_axis[0])*(1+intervener_zs[i]))
        Pints.append(Pint)
        P3 = np.convolve(P3,Pint[::-1],mode='same')
        P3[DM_axis<0] = 0
    #P4 = np.convolve(P3,Phalo[::-1],mode="same")
    #P4[DM_axis < 0] = 0
    Phost = copy.deepcopy(P3)
    Phost = Phost/np.sum(Phost*(DM_axis[1]*(1+ztest)-DM_axis[0]*(1+ztest)))

    #expected value
    Phost_exp = 0
    for i in range(len(DM_axis)-1):
        DM1 = DM_axis[i]*(1+ztest)
        DM2 = DM_axis[i+1]*(1+ztest)
        P1 = Phost[i]
        P2 = Phost[i+1]
        dDM = (DM2-DM1)#*(1+ztest)
        Phost_exp += dDM*((DM1*P1) + (DM2*P2))/2


    cumdisthost = np.cumsum(Phost)/np.sum(Phost)
    #print("calculating " + str(((1-siglevel)/2)) + " and"+ str((1-((1-siglevel)/2))) + " percentiles")
    low = (DM_axis*(1 + ztest))[np.argmin(abs(cumdisthost-((1-siglevel)/2)))]
    upp = (DM_axis*(1 + ztest))[np.argmin(abs(cumdisthost-(1-((1-siglevel)/2))))]
    """
    if plot:
        plt.figure(figsize=(24,12))
        plt.plot(DM_axis*(1+ztest),cumdisthost)
        plt.plot(DM_axis*(1+ztest),Phost)
        plt.axvline(low,color="red")
        plt.axvline(upp,color="red")
        plt.show()
    """

    if plot:
        plt.plot(DM_axis,Pobs/np.nanmax(Pobs),label=r'$DM_{obs}$')
        plt.plot(DM_axis,Pmw/np.nanmax(Pmw),label=r'$DM_{MW}$')
        plt.plot(DM_axis,Pigm/np.nanmax(Pigm),label=r'$DM_{IGM}$')
        for i in range(len(intervener_DMs)):
            plt.plot(DM_axis*(1+intervener_zs[i]),Pints[i]/np.nanmax(Pints[i]),label=r'$DM_{{int,{i}}}/(1+z_{{int,{i}}})$'.format(i=i))
        plt.plot(DM_axis,Pmwhalo/np.nanmax(Pmwhalo),label=r'$DM_{MW,halo}$')
        plt.plot(DM_axis,Phost/np.nanmax(Phost),label=r'$DM_{host}$/(1+z)',linewidth=4)
        plt.plot(DM_axis*(1+ztest),Phost/np.nanmax(Phost),label=r'$DM_{host}$',linewidth=4)
        plt.axvline(Phost_exp,color="purple")
        plt.axvspan(low,upp,color='purple',alpha=0.1)
        #plt.text(Phost_exp+10,1,'$DM_{{host}}={a:.2f}^{{+{b:.2f}}}_{{-{c:.2f}}}\\,pc/cm^3$'.format(a=Phost_exp,b=upp-Phost_exp,c=Phost_exp-low),
        #        backgroundcolor='thistle',fontsize=18)
        plt.xlim(0,DMobs*2)
        plt.legend(loc='upper right')
        plt.xlabel("DM")
        plt.ylabel("PDF")
        if save:
            plt.savefig(savedir + "/DM_budget_plot.pdf")
        plt.show()

    return Phost,DM_axis*(1+ztest)


#convolution method to derive DM_IGM
def DM_IGM_limits(DMobs,frb_z,frb_gl,frb_gb,res=10000,plot=False,DMhalo=DMHALOEST,siglevel=DEF_SIGLEVEL):
    """
    This function derives the IGM DM given an FRB's observed DM, redshift, and
    galactic position. The observed DM acts as a lower limit on the IGM DM. This
    returns the mean DM_IGM from equation 6 of Zhang 2021, and the median and 1sigma
    upper and lower limits of DM_IGM.
    DMobs: observed DM
    frb_z: measured redshift
    frb_gl: Galactic longitude in degrees
    frb_gb: Galactic latitude in degrees
    res: resolution of the DM axis for each DM PDF
    plot: if True plots the DM IGM CDF and PDF, default False
    DM_halo: estimate for Galactic halo DM, default 10 pc/cc
    siglevel: confidence interval for error estimates, default 68% (1sigma)
    """
    DM_axis = np.linspace(-2000,2000,res)


    #measured values
    #DMobs = DSAFRBDMS[DSAFRBNICKNAMES.index(frbtest)]
    DMIGMmean = DM_IGM_MEAN(frb_z,res=1000) #DSAFRBZS[DSAFRBNICKNAMES.index(frbtest)],res=1000)
    DMmw = pyne2001.get_dm(frb_gl,frb_gb,30) #frb_glpyne2001.get_dm(DSAFRBGLS[DSAFRBNICKNAMES.index(frbtest)],DSAFRBGBS[DSAFRBNICKNAMES.index(frbtest)],30)
    ztest = frb_z #DSAFRBZS[DSAFRBNICKNAMES.index(frbtest)]


    #IGM DM
    Pigm = PIGM_zp1(DM_axis/DMIGMmean,ztest)
    Pigm[DM_axis < 0] = 0
    Pigmfull = copy.deepcopy(Pigm)
    Pigm[DM_axis>DMobs] = 0
    Pigm = Pigm/np.sum(Pigm*(DM_axis[1]-DM_axis[0]))
    Pigmfull = Pigmfull/np.sum(Pigmfull*(DM_axis[1]-DM_axis[0]))

    #expected value
    Pigm_exp = 0
    for i in range(len(DM_axis)-1):
        DM1 = DM_axis[i]
        DM2 = DM_axis[i+1]
        P1 = Pigm[i]
        P2 = Pigm[i+1]
        dDM = (DM2-DM1)#*(1+ztest)
        Pigm_exp += dDM*((DM1*P1) + (DM2*P2))/2


    cumdisthost = np.cumsum(Pigm)/np.sum(Pigm)
    #print("calculating " + str(((1-siglevel)/2)) + " and"+ str((1-((1-siglevel)/2))) + " percentiles")
    low = (DM_axis)[np.argmin(abs(cumdisthost-((1-siglevel)/2)))]
    upp = (DM_axis)[np.argmin(abs(cumdisthost-(1-((1-siglevel)/2))))]

    if plot:
        plt.figure(figsize=(24,12))
        plt.plot(DM_axis,cumdisthost)
        plt.plot(DM_axis,Pigm)
        plt.axvline(low,color="red")
        plt.axvline(upp,color="red")
        plt.show()
    return DMIGMmean,Pigm_exp,low,upp



#convolution method to derive DM_IGM distribution
def DM_IGM_dist(DMobs,frb_z,frb_gl,frb_gb,res=10000,plot=False,DMhalo=DMHALOEST,siglevel=DEF_SIGLEVEL):
    """
    This function derives the IGM DM PDF given an FRB's observed DM, redshift, and
    galactic position. The observed DM acts as a lower limit on the IGM DM. This
    returns the PDF and DM axis for DM_IGM.
    DMobs: observed DM
    frb_z: measured redshift
    frb_gl: Galactic longitude in degrees
    frb_gb: Galactic latitude in degrees
    res: resolution of the DM axis for each DM PDF
    plot: if True plots the DM IGM CDF and PDF, default False
    DM_halo: estimate for Galactic halo DM, default 10 pc/cc
    siglevel: confidence interval for error estimates, default 68% (1sigma)
    """
    DM_axis = np.linspace(-2000,2000,res)


    #measured values
    #DMobs = DSAFRBDMS[DSAFRBNICKNAMES.index(frbtest)]
    DMIGMmean = DM_IGM_MEAN(frb_z,res=1000) #DSAFRBZS[DSAFRBNICKNAMES.index(frbtest)],res=1000)
    DMmw = pyne2001.get_dm(frb_gl,frb_gb,30) #frb_glpyne2001.get_dm(DSAFRBGLS[DSAFRBNICKNAMES.index(frbtest)],DSAFRBGBS[DSAFRBNICKNAMES.index(frbtest)],30)
    ztest = frb_z #DSAFRBZS[DSAFRBNICKNAMES.index(frbtest)]


    #IGM DM
    Pigm = PIGM_zp1(DM_axis/DMIGMmean,ztest)
    Pigm[DM_axis < 0] = 0
    Pigmfull = copy.deepcopy(Pigm)
    Pigm[DM_axis>DMobs] = 0
    Pigm = Pigm/np.sum(Pigm*(DM_axis[1]-DM_axis[0]))
    Pigmfull = Pigmfull/np.sum(Pigmfull*(DM_axis[1]-DM_axis[0]))

    #expected value
    Pigm_exp = 0
    for i in range(len(DM_axis)-1):
        DM1 = DM_axis[i]
        DM2 = DM_axis[i+1]
        P1 = Pigm[i]
        P2 = Pigm[i+1]
        dDM = (DM2-DM1)#*(1+ztest)
        Pigm_exp += dDM*((DM1*P1) + (DM2*P2))/2


    cumdisthost = np.cumsum(Pigm)/np.sum(Pigm)
    #print("calculating " + str(((1-siglevel)/2)) + " and"+ str((1-((1-siglevel)/2))) + " percentiles")
    low = (DM_axis)[np.argmin(abs(cumdisthost-((1-siglevel)/2)))]
    upp = (DM_axis)[np.argmin(abs(cumdisthost-(1-((1-siglevel)/2))))]

    if plot:
        plt.figure(figsize=(24,12))
        plt.plot(DM_axis,cumdisthost)
        plt.plot(DM_axis,Pigm)
        plt.axvline(low,color="red")
        plt.axvline(upp,color="red")
        plt.show()
    return Pigm,DM_axis




#convolution method to derive RMhost
def RM_host_limits(RMobs,RMobserr,
                   RMmw,RMmwerr,
                   RMion,RMionerr,ztest,res=10000,RMmin=-5000,RMmax=5000,
                  plot=False,intervener_RMs=[],intervener_RM_errs=[],intervener_zs=[]):
    """
    This function derives the host RM given an FRB's observed RM and redshift. 
    It convolves the distributions of RM_obs, RM_MW (from Hutschenreutrer+2021), 
    and RM_ION (from Sotomayor-Beltran+2013) and returns the median and 1sigma upper and lower limits 
    of RM_host.
    RMobs: observed RM
    RMmw: Milky Way RM estimate (Hutschenreuter+2021 or column V in spreadsheet on RM sheet)
    RMion: ionospheric RM estimate (Sotomayor-Beltran 2013 or column X in spreadsheet on RM sheet)
    frb_z: measured redshift
    res: resolution of the RM axis for each RM PDF
    plot: if True plots the RM budget and RM host CDF, default False
    RM_halo: estimate for Galactic halo RM, default 10 pc/cc
    siglevel: confidence interval for error estimates, default 68% (1sigma)
    """
    RM_axis = np.linspace(RMmin,RMmax,res)

    

    #print("RMobs: " + str(RMobs))
    #print("RMmw: " + str(RMmw))
    #print("RMion: " + str(RMion))
    #print("Initial RMhost estimate: " + str((RMobs-RMmw-RMion)*((1+ztest)**2)))

    #get probability distributions
    #observed RM
    if np.isnan(RMobserr):
        #use default 0.1 rad/m^2
        RMobserr = 0.1
    elif RMobserr < (RMmax-RMmin)/res:
        RMobserr = (RMmax-RMmin)/res
    
    
    Pobs = gaus(RM_axis,RMobs,RMobserr)
    Pobs = Pobs/np.sum(Pobs*(RM_axis[1]-RM_axis[0]))
    #MW RM
    Pmw = gaus(RM_axis,RMmw,RMmwerr)
    Pmw = Pmw/np.sum(Pmw*(RM_axis[1]-RM_axis[0]))
    #Ion RM
    Pion = gaus(RM_axis,RMion,RMionerr)
    Pion = Pion/np.sum(Pion*(RM_axis[1]-RM_axis[0]))


    #convolve to get RM host
    P1 = np.convolve(Pobs,Pmw[::-1],mode="same")
    P2 = np.convolve(P1,Pion[::-1],mode="same")

    #interveners
    Pints = []
    for i in range(len(intervener_RMs)):
        if ~np.isnan(intervener_RM_errs[i]) and intervener_RM_errs[i] > 2*(RM_axis[1]-RM_axis[0]):
            Pint = gaus(RM_axis*((1+intervener_zs[i])**2),intervener_RMs[i],intervener_RM_errs[i])
        else:
            Pint = gaus(RM_axis*((1+intervener_zs[i])**2),intervener_RMs[i],2*(RM_axis[1]-RM_axis[0]))
        Pint = Pint/np.sum(Pint*(RM_axis[1]-RM_axis[0])*((1+intervener_zs[i])**2))
        Pints.append(Pint)
        P2 = np.convolve(P2,Pint[::-1],mode='same')


    Phost = copy.deepcopy(P2)
    Phost_z = Phost/np.sum(Phost*(RM_axis[1]-RM_axis[0]))
    Phost = Phost/np.sum(Phost*(RM_axis[1]-RM_axis[0])*((1+ztest)**2))

    
    #plotting
    if plot:
        plt.figure(figsize=(12,6))
        plt.plot(RM_axis,Pobs/np.nanmax(Pobs),label=r'$RM_{obs}$')
        plt.plot(RM_axis,Pmw/np.nanmax(Pmw),label=r'$RM_{MW}$')
        plt.plot(RM_axis,Pion/np.nanmax(Pion),label=r'$RM_{ION}$')
        for i in range(len(intervener_RMs)):
            plt.plot(RM_axis*((1+intervener_zs[i])**2),Pints[i]/np.nanmax(Pints[i]),label=r'$RM_{{int,{i}}}/(1+z_{{int,{i}}})$'.format(i=i))
        plt.plot(RM_axis,Phost/np.nanmax(Phost),label=r'$RM_{host}$',linewidth=4)
        plt.plot(RM_axis*((1+ztest)**2),Phost/np.nanmax(Phost),label=r'$RM_{host}(1+z)^2$',linewidth=4)
        plt.xlim(-RMobs*2,RMobs*2)
        plt.legend(loc='upper left')
        plt.xlabel("RM")
        plt.ylabel("PDF")
        plt.show()
    
    Phost_exp = 0
    for i in range(len(RM_axis)-1):
        RM1 = RM_axis[i]*((1 + ztest)**2)
        RM2 = RM_axis[i+1]*((1 + ztest)**2)
        P1 = Phost[i]
        P2 = Phost[i+1]
        dRM = (RM2-RM1)
        Phost_exp += dRM*((RM1*P1) + (RM2*P2))/2
    
    wids,wheights,lef,righ = peak_widths(Phost,[np.argmax(Phost)],rel_height=0.5)
    lefRM=np.min(RM_axis*((1+ztest)**2)) + lef*(RM_axis[1]-RM_axis[0])*((1+ztest)**2)
    righRM=np.min(RM_axis*((1+ztest)**2)) + righ*(RM_axis[1]-RM_axis[0])*((1+ztest)**2)
    
    FWHM = righRM-lefRM
    sig = FWHM/(2*np.sqrt(2*np.log(2)))
    
    upp = (Phost_exp + sig)[0]
    low = (Phost_exp - sig)[0]
    
    return Phost_exp,low,upp


#convolution method to derive RMhost distribution
def RM_host_dist(RMobs,RMobserr,
                   RMmw,RMmwerr,
                   RMion,RMionerr,ztest,res=10000,RMmin=-5000,RMmax=5000,
                  plot=False,intervener_RMs=[],intervener_RM_errs=[],intervener_zs=[],save=False,savedir='./'):
    """
    This function derives the host RM given an FRB's observed RM and redshift. 
    It convolves the distributions of RM_obs, RM_MW (from Hutschenreutrer+2021), 
    and RM_ION (from Sotomayor-Beltran+2013) and returns the distribution and RM axis of the
    host RM.
    RMobs: observed RM
    RMmw: Milky Way RM estimate (Hutschenreuter+2021 or column V in spreadsheet on RM sheet)
    RMion: ionospheric RM estimate (Sotomayor-Beltran 2013 or column X in spreadsheet on RM sheet)
    frb_z: measured redshift
    res: resolution of the RM axis for each RM PDF
    plot: if True plots the RM budget and RM host CDF, default False
    RM_halo: estimate for Galactic halo RM, default 10 pc/cc
    siglevel: confidence interval for error estimates, default 68% (1sigma)
    """
    RM_axis = np.linspace(RMmin,RMmax,res)

    #print("RMobs: " + str(RMobs))
    #print("RMmw: " + str(RMmw))
    #print("RMion: " + str(RMion))
    #print("Initial RMhost estimate: " + str((RMobs-RMmw-RMion)*((1+ztest)**2)))

    #get probability distributions
    #observed RM
    if np.isnan(RMobserr):
        RMobserr = 0.1
    elif RMobserr < (RMmax-RMmin)/res:
        RMobserr = (RMmax-RMmin)/res
    Pobs = gaus(RM_axis,RMobs,RMobserr)
    Pobs = Pobs/np.sum(Pobs*(RM_axis[1]-RM_axis[0]))
    #MW RM
    Pmw = gaus(RM_axis,RMmw,RMmwerr)
    Pmw = Pmw/np.sum(Pmw*(RM_axis[1]-RM_axis[0]))
    #Ion RM
    Pion = gaus(RM_axis,RMion,RMionerr)
    Pion = Pion/np.sum(Pion*(RM_axis[1]-RM_axis[0]))


    #convolve to get RM host
    P1 = np.convolve(Pobs,Pmw[::-1],mode="same")
    P2 = np.convolve(P1,Pion[::-1],mode="same")

    #interveners
    Pints = []
    for i in range(len(intervener_RMs)):
        if ~np.isnan(intervener_RM_errs[i]) and intervener_RM_errs[i] > 2*(RM_axis[1]-RM_axis[0]):
            Pint = gaus(RM_axis*((1+intervener_zs[i])**2),intervener_RMs[i],intervener_RM_errs[i])
        else:
            Pint = gaus(RM_axis*((1+intervener_zs[i])**2),intervener_RMs[i],2*(RM_axis[1]-RM_axis[0]))
        Pint = Pint/np.sum(Pint*(RM_axis[1]-RM_axis[0])*((1+intervener_zs[i])**2))
        Pints.append(Pint)
        P2 = np.convolve(P2,Pint[::-1],mode='same')

    Phost = copy.deepcopy(P2)
    Phost_z = Phost/np.sum(Phost*(RM_axis[1]-RM_axis[0]))
    Phost = Phost/np.sum(Phost*(RM_axis[1]-RM_axis[0])*((1+ztest)**2))

    
    Phost_exp = 0
    for i in range(len(RM_axis)-1):
        RM1 = RM_axis[i]*((1 + ztest)**2)
        RM2 = RM_axis[i+1]*((1 + ztest)**2)
        P1 = Phost[i]
        P2 = Phost[i+1]
        dRM = (RM2-RM1)
        Phost_exp += dRM*((RM1*P1) + (RM2*P2))/2
    
    wids,wheights,lef,righ = peak_widths(Phost,[np.argmax(Phost)],rel_height=0.5)
    lefRM=np.min(RM_axis*((1+ztest)**2)) + lef*(RM_axis[1]-RM_axis[0])*((1+ztest)**2)
    righRM=np.min(RM_axis*((1+ztest)**2)) + righ*(RM_axis[1]-RM_axis[0])*((1+ztest)**2)
    
    FWHM = righRM-lefRM
    sig = FWHM/(2*np.sqrt(2*np.log(2)))
    
    upp = (Phost_exp + sig)[0]
    low = (Phost_exp - sig)[0]

    #plotting
    if plot:
        plt.figure(figsize=(18,6))
        plt.plot(RM_axis,Pobs/np.nanmax(Pobs),label=r'$RM_{obs}$')
        plt.plot(RM_axis,Pmw/np.nanmax(Pmw),label=r'$RM_{MW}$')
        plt.plot(RM_axis,Pion/np.nanmax(Pion),label=r'$RM_{ION}$')
        for i in range(len(intervener_RMs)):
            plt.plot(RM_axis*((1+intervener_zs[i])**2),Pints[i]/np.nanmax(Pints[i]),label=r'$RM_{{int,{i}}}/(1+z_{{int,{i}}})^2$'.format(i=i))
        plt.plot(RM_axis,Phost/np.nanmax(Phost),label=r'$RM_{host}/(1+z)^2$',linewidth=4)
        plt.plot(RM_axis*((1+ztest)**2),Phost/np.nanmax(Phost),label=r'$RM_{host}$',linewidth=4)
        plt.axvline(Phost_exp,color="purple")
        plt.axvspan(low,upp,color='purple',alpha=0.1)
        #plt.text(Phost_exp+10,1,'$RM_{{host}}={a:.2f}^{{+{b:.2f}}}_{{-{c:.2f}}}\\,rad/m^2$'.format(a=Phost_exp,b=upp-Phost_exp,c=Phost_exp-low),
        #        backgroundcolor='thistle',fontsize=18)
        plt.xlim(-RMobs*2,RMobs*2)
        plt.legend(loc='upper left')
        plt.xlabel("RM")
        plt.ylabel("PDF")
        plt.savefig(savedir+"/RM_budget_plot.pdf")
        plt.show()
    
    return Phost,RM_axis*((1+ztest)**2)



#convolution method to derive B||host (note, need to get RMhost and DM host distributions first)
def Bhost_dist(DMhost,dmdist,DMaxis,RMhost,RMhosterr,res=10000,res2=500,siglevel=DEF_SIGLEVEL,plot=False,buff=50,save=False,savedir='./'):
    
    #dmdist,DMaxis = DM_host_dist(frbtest,res=res)
    B_est = RMhost/0.81/DMhost #rough estimate
    
    DMaxisscaled = DMaxis*0.81
    dmdistscaled = dmdist/np.sum(dmdist*(DMaxisscaled[1]-DMaxisscaled[0]))
    Baxis = np.linspace(B_est-buff,B_est+buff,res2)
    #rmdist = gaus(Baxis*DMaxisscaled,DSAFRBRMHOSTS_EXP[1],DSAFRBRMHOSTERRS[1])
    
    
    #numerically integrate RM/DM to get host B distribution
    Bdist = np.zeros(len(Baxis))
    for i in range(len(Baxis)):
        B_i = Baxis[i]

        for j in range(len(DMaxisscaled)-1):
            dm1 = DMaxisscaled[j]
            dm2 = DMaxisscaled[j+1]

            Prm1 = gaus(B_i*dm1,RMhost,RMhosterr) #probability of getting RM corresponding to this B field and DM
            Prm2 = gaus(B_i*dm2,RMhost,RMhosterr)

            Pdm1 = dmdistscaled[j] #probability of getting DM host value
            Pdm2 = dmdistscaled[j+1]

            int1 = dm1*Prm1*Pdm1
            int2 = dm2*Prm2*Pdm2

            Bdist[i] += (((int1+int2)/2)*(dm2-dm1))

    Bdist = Bdist/np.sum(Bdist*(Baxis[1]-Baxis[0]))


    B_exp = 0
    for i in range(len(Baxis)-1):
        B1 = Baxis[i]
        B2 = Baxis[i+1]

        int1 = B1*Bdist[i]
        int2 = B2*Bdist[i+1]

        B_exp += ((int1+int2)/2)*(B2-B1)
    
    cumdisthost = np.cumsum(Bdist)/np.sum(Bdist)
    #print("calculating " + str(((1-siglevel)/2)) + " and"+ str((1-((1-siglevel)/2))) + " percentiles")
    low = Baxis[np.argmin(abs(cumdisthost-((1-siglevel)/2)))]
    upp = Baxis[np.argmin(abs(cumdisthost-(1-((1-siglevel)/2))))]

    if plot:
        plt.figure(figsize=(18,4))
        plt.plot(Baxis,Bdist,linewidth=4,color='tab:purple')
        plt.axvline(B_exp,color="purple")
        plt.axvspan(low,upp,color='purple',alpha=0.1)
        #plt.text(B_exp+0.5,np.nanmax(Bdist),'$B_{{||,host}}={a:.2f}^{{+{b:.2f}}}_{{-{c:.2f}}}\\,\\mu G$'.format(a=B_exp,b=upp-B_exp,c=B_exp-low),
        #        backgroundcolor='thistle',fontsize=18)
        plt.xlabel(r'$B_{||,host}$')
        plt.ylabel("PDF")
        plt.xlim(-B_est*2,B_est*2)
        if save:
            plt.savefig(savedir+"/Bfield_budget_plot.pdf")
        plt.show()

        
    return Bdist,B_exp,low,upp,Baxis


SIMBAD_CATALOG_OPTIONS = [
        'DELS', #DESI Legacy Imaging Survey = DECaLS + BASS + MzLS
        'NGC', #New General Catalog
        'GAIA', #Gaia
        'LEDA', #LEDA, HyperLEDA
        'PS1',#PANSTARRS
        'SDSS',#SLOAN
        'WISE',#WISE
        'WISEA',#ALLWISE (WISE All Sky Survey)
        'WISEU',#unWISE
        '2MASS',#2MASS
        'DES',#DARK ENERGY SURVEY
        'GEMINI',#GEMINI
        ]
VIZIER_CODES = {
        'LEDA':"VII/238",
        'NGC':"VII/118",
        'GAIA':"I/350",
        'PS1':"II/349",
        'SDSS':"V/139",
        'WISE':"II/311",
        'WISEA':"II/328",
        'WISEU':"II/363",
        'DES':"II/371"
        }
VIZIER_NAMEFIELDS = {
        'LEDA':"PGC",
        'NGC':"Name",
        'GAIA':"EDR3Name",
        'PS1':"objID",
        'SDSS':"SDSS9",
        'WISE':"WISE",
        'WISEA':"AllWISE",
        'WISEU':"objID",
        'DES':"DES"
        }
def get_VIZIER_cols(qdat,lf=sys.stdout):
    """
    Helper function to query vizier catalogs for a given source and 
    append to qdat SIMBAD table
    """
    id_lists = list(qdat['IDS'])

    vz = vizier.Vizier(columns=["**"])
    vz.ROW_LIMIT = -1
    for i in range(len(id_lists)):
        id_list_str = id_lists[i]
        id_list = id_list_str.split("|")
        for objname in id_list:
            print(objname,file=lf)
            sname = objname.split()
            catalog = sname[0]
            ID = "".join(sname[1:] if catalog != 'GAIA' else sname[2:])
            #catalog,ID = objname.split()
            
            #query for ID name
            if catalog in VIZIER_NAMEFIELDS.keys() and catalog in VIZIER_CODES.keys():
                vz_tables = vz.query_constraints(**{VIZIER_NAMEFIELDS[catalog]:ID},catalog=VIZIER_CODES[catalog])
                try:
                    for vz_table in vz_tables:
                        for col in vz_table.columns:
                            #add column if not present
                            if col not in qdat.columns:
                                if vz_table[col].dtype != str:
                                    qdat.add_column(np.array(np.nan*np.ones(len(qdat)),dtype=vz_table[col].dtype),name=col)
                                else:
                                    qdat.add_column([""]*len(qdat),name=col)
                            #update object with new column value
                            qdat[col][i] = vz_table[col][0]
                except TypeError as exc:
                    print("Empty table:",exc,file=lf)
    return qdat

PSQUERY_TABLES = {
        'WISE':["allwise.source","catwise2020.main","ls_dr10.wise","ls_dr9.wise","ls_dr8.wise","unwise_dr1.object"],
        'WISEA':["allwise.source","catwise2020.main","ls_dr10.wise","ls_dr9.wise","ls_dr8.wise","unwise_dr1.object"],
        'WISEU':["allwise.source","catwise2020.main","ls_dr10.wise","ls_dr9.wise","ls_dr8.wise","unwise_dr1.object"],
        'DES':["des_dr1.main","des_dr1.galaxies","des_dr1.im3shape","des_dr1.morph","des_dr1.photoz","des_dr2.main","des_dr2.flux"],
        'DESI':["desi_edr.photometry","desi_edr.target","sga2020.tractor","sga2020.ellipse","sparcl.main","splus_dr1.stripe82","splus_dr1.des_dr1","splus_dr2.main","splus_dr2.photoz"],
        'GAIA':["gaia_dr1.gaia_source","gaia_dr2.gaia_source","gaia_dr3.gaia_source","gaia_dr3.galaxy_candidates","gaia_dr3.astrophysical_parameters","gaia_dr3.qso_candidates"],
        'GEMINI':["gnirs_dqs.spec_measurements","gnirs_dqs.spec_measurements_supp","gogreen_dr1.clusters","gogreen_dr1.photo","gogreen_dr1.redshift"],
        'DELS':["ls_dr10.photo_z","ls_dr10.wise","ls_dr10.tractor","ls_dr9.photo_z","ls_dr9.wise","ls_dr9.tractor","ls_dr8.photo_z","ls_dr8.wise","ls_dr8.tractor","delve_dr1.objects","delve_dr2.objects","delve_dr2.photo_z"],
        'SDSS':["sdss_dr12.photoplate","sdss_dr12.emissionlinesport","sdss_dr12.dr12q","sdss_dr12q_duplicates","sdss_dr13.galspecline_dr8","sdss_dr16.dr16q","sdss_dr16.dr16q_duplicates","sdss_dr16.dr16q_superset","sdss_dr16.dr16q_superset_duplicates","sdss_dr16.elg_classifier","sdss_dr16.photoplate","sdss_dr17.apogee2_allstar","sdss_dr17.eboss_mcpm","sdss_dr17.photoplate","sparcl.main"],
        'LEDA':["sga2020.tractor","sga2020.ellipse"],
        '2MASS':["twomass.esc","ukidss_dr11plus.dxssource","ukidss_dr11plus.udssource","vhs_dr5.vhs_cat_v3"]

        }
PSQUERY_PHOTOZ_TABLES = {#'DES': {'des_dr1.photo_z': ('median_z', 'z_sigma', 'photometric')}, 
                        'DESI': {'splus_dr2.photoz': ('zml', 'zml_err', 'photometric')}, 
                        'GEMINI': {'gogreen_dr1.redshift': ('redshift', '', 'spectroscopic')}, 
                        'DELS': {'ls_dr10.photo_z': ('z_phot_median', 'z_phot_std', 'photometric'), 
                                'ls_dr9.photo_z': ('z_phot_median', 'z_phot_std', 'photometric'), 
                                'ls_dr8.photo_z': ('z_phot_median', 'z_phot_std', 'photometric'), 
                                'delve_dr2.photoz': ('z', 'zerr', 'photometric')}
                        }
PSQUERY_PHOTOZ_TABLES_NORADEC = ['ls_dr10.photo_z',
                                #'des_dr1.photo_z',
                                'ls_dr9.photo_z',
                                'ls_dr8.photo_z']
def find_redshift(qdat,lf=sys.stdout,radius=1/60):
    """
    Goes through queryClient catalogs to find photometric redshifts
    for each source given in qdat
    """

    ra_list = qdat['RA']
    dec_list = qdat['DEC']
    id_lists = list(qdat['IDS'])
    zcols = ["RVZ_RADVEL"]*len(id_lists) #default is SIMBAD redshift
    zerrcols = ["RVZ_ERROR"]*len(id_lists)
    qdat.add_column(qdat['RVZ_RADVEL'],name='BEST_Z')
    qdat.add_column(qdat['RVZ_ERROR'],name='BEST_Z_ERROR')

    for i in range(len(id_lists)):
        id_list_str = id_lists[i]
        id_list = id_list_str.split("|")
        coord = SkyCoord(ra_list[i] +" "+ dec_list[i],frame='icrs',unit=(u.hourangle,u.deg))
        ra = coord.ra.value
        dec = coord.dec.value
        
        allcatalogs = list(PSQUERY_PHOTOZ_TABLES.keys())
        """
        #check if any catalogs in SIMBAD have photozs
        for objname in id_list:
            print(objname)
            sname = objname.split()
            catalog = sname[0]
            ID = "".join(sname[1:] if catalog != 'GAIA' else sname[2:])
            
            if catalog in allcatalogs:
                allcatalogs = [catalog]
        """
        print(allcatalogs,file=lf)
        ztable = None
        for catalog in allcatalogs:
            #query within 1 arcsecond of source
            for tab in PSQUERY_PHOTOZ_TABLES[catalog]:
                ztable= None
                if tab not in PSQUERY_PHOTOZ_TABLES_NORADEC:
                    cols = PSQUERY_PHOTOZ_TABLES[catalog][tab][0]
                    cols += "" if PSQUERY_PHOTOZ_TABLES[catalog][tab][1]=="" else ","+PSQUERY_PHOTOZ_TABLES[catalog][tab][1]
                    cols += ",ra,dec"
                    ztable = Table.from_pandas(qc.query("select " + cols + " from " + tab + " where q3c_radial_query(ra, dec, {ra}, {dec}, {radius})".format(ra=ra,dec=dec,radius=radius),fmt='pandas'))
                else:
                    #use psquery to get source ids
                    itable = Table.from_pandas(qc.query("select ra,dec,ls_id from " + tab[:-7] + "tractor where q3c_radial_query(ra, dec, {ra}, {dec}, {radius})".format(ra=ra,dec=dec,radius=radius),fmt='pandas'))
                        
                    #get nearest one
                    if len(itable) > 0:
                        idx = np.argmin(SkyCoord(ra=ra*u.deg,dec=dec*u.deg,frame='icrs').separation(SkyCoord(ra=itable['ra'].value*u.deg,dec=itable['dec'].value*u.deg,frame='icrs')).value)
                        ls_id = itable['ls_id'][idx]

                        cols = PSQUERY_PHOTOZ_TABLES[catalog][tab][0]
                        cols += "" if PSQUERY_PHOTOZ_TABLES[catalog][tab][1]=="" else ","+PSQUERY_PHOTOZ_TABLES[catalog][tab][1]
                        ztable = Table.from_pandas(qc.query("select " + cols + " from " + tab + " where " + tab + ".ls_id=" + str(ls_id),fmt='pandas'))
                        
                        if len(ztable)>0:
                            ztable.add_column(itable['ra'][idx:idx+1],name='ra')
                            ztable.add_column(itable['dec'][idx:idx+1],name='dec')
                        else:
                            ztable = None
                if ztable is not None and len(ztable)>0: 
                    print(PSQUERY_PHOTOZ_TABLES[catalog][tab][0],file=lf)
                    zcols[i] = PSQUERY_PHOTOZ_TABLES[catalog][tab][0]
                    zerrcols[i] = PSQUERY_PHOTOZ_TABLES[catalog][tab][1]
                    break
            if ztable is not None and len(ztable)>0: break
        print("None" if ztable is None else ztable.columns,file=lf)
        if ztable is not None and len(ztable)>0:
            for col in ztable.columns:
                #add column if not present
                if col not in qdat.columns and col != 'ra' and col != 'dec':
                    if ztable[col].dtype != str:
                        qdat.add_column(np.array(np.nan*np.ones(len(qdat)),dtype=ztable[col].dtype),name=col)
                    else:
                        qdat.add_column([""]*len(qdat),name=col)
                if col != 'ra' and col != 'dec':
                    #update object with new column value
                    qdat[col][i] = ztable[col][np.argmin(SkyCoord(ra=ra*u.deg,dec=dec*u.deg,frame='icrs').separation(SkyCoord(ra=ztable['ra'].value*u.deg,dec=ztable['dec'].value*u.deg,frame='icrs')).value)]
            print(zcols,file=lf)
            #save to best col
            qdat['BEST_Z'][i] = qdat[zcols[i]][i]
            qdat['BEST_Z_ERROR'][i] = qdat[zerrcols[i]][i]

    return qdat


SIMBAD_GALAXY_OPTIONS = [
        '(G),Galaxy',
        '(IG),Interacting Galaxies',
        '(PaG),Pair of Galaxies',
        '(GrG),Group of Galaxies',
        '(CGG),Compact Group of Galaxies',
        '(ClG),Cluster of Galaxies',
        '(PCG),Proto Cluster of Galaxies',
        '(SCG),Supercluster of Galaxies'
        ]
COSMOLOGY_OPTIONS = [
        'Planck18',
        'Planck15',
        'Planck13',
        'WMAP9',
        'WMAP7',
        'WMAP5',
        'WMAP3',
        'WMAP1'
        ]
COSMOLOGIES = {'Planck18':Planck18,'Planck15':Planck15,'Planck13':Planck13,
                'WMAP9':WMAP9,'WMAP7':WMAP7,'WMAP5':WMAP5,'WMAP3':WMAP3,'WMAP1':WMAP1}
def get_SIMBAD_gals(ra,dec,radius,catalogs=[],types=[],cosmology="Planck18",redshift=-1,redshift_range=None):
    """
    This function uses astroquery to query SIMBAD and identify
    galaxies within a specific radius near an FRB. Options to isolate to
    the following surveys:
        DESI
        NGC
        GAIA
        LEDA
        PS1
        SDSS
        WISE
        WISEA
        WISEU
        2MASS
        DES
        GEMINI
        UKIDSS
        VHS

    """
    #round ra, dec to 2 decimal places
    ra = np.around(ra,2)
    dec = np.around(dec,2)
     
    #check catalog inputs
    if len(catalogs)==0:
        catstring = ""
    else:
        catstring = " & cat in ("
        for i in catalogs:
            catstring += "\'"+i+"\',"
        catstring = catstring[:-1] + ")"

    #check object types
    if len(types)==0:
        types =copy.deepcopy(SIMBAD_GALAXY_OPTIONS)

    types = [types[i][1:types[i].index(")")] for i in range(len(types))]
    typestring = "otypes in ("
    for i in types:
        typestring += "\'"+i+"\',"
        if i == 'GrG': typestring += "\'Gr?\',"
        elif i == 'ClG': typestring += "\'C?G\',"
        elif i == 'PCG': typestring += "\'PCG?\',"
        elif i == 'SCG': typestring += "\'SC?\',"
    typestring = typestring[:-1] + ")"

    #check redshift
    if redshift==-1:
        redstring = ""
    elif redshift_range is None:
        redstring = " & redshift<{z}".format(z=redshift)
    else:
        redstring = " & (redshift<{zup} & redshift>{zlow})".format(zup=redshift+redshift_range,zlow=redshift-redshift_range)


    #create custom simbad
    customSimbad = Simbad()
    customSimbad.get_votable_fields()
    customSimbad.add_votable_fields('dimensions')
    customSimbad.add_votable_fields('distance')
    customSimbad.add_votable_fields('otype')
    customSimbad.add_votable_fields('velocity')
    customSimbad.add_votable_fields('morphtype')
    customSimbad.add_votable_fields('ids')
    #create query that targets (1) galaxies and groups of galaxies (2) within specified radius (3) has redshift in specified range (4) in catalogs specified
    query_string = "region(CIRCLE,icrs,{ra} {s}{dec},{radius}) & rvtype=\'z\' & {ts}{cs}{zs}".format(ra=ra,s="+" if dec>0 else "-",dec=np.abs(dec),radius=radius,ts=typestring,cs=catstring,zs=redstring) 
    lf = open(logfile,"a")
    print(query_string,file=lf)

    #send query
    qdat=customSimbad.query_criteria(query_string)
    if qdat is not None:
        #add vizier columns
        qdat =get_VIZIER_cols(qdat,lf=lf)

        #look for redshifts
        qdat = find_redshift(qdat,lf=lf,radius=1/60)

        #make column for offset
        c = SkyCoord([qdat['RA'][i] +qdat['DEC'][i] for i in range(len(qdat))],frame='icrs',unit=(u.hourangle,u.deg))
        qdat.add_column(SkyCoord(ra=ra*u.deg,dec=dec*u.deg,frame='icrs').separation(c).to(u.deg).value,name='OFFSET',index=3)
        qdat['OFFSET'].unit = u.deg

        #make column for impact parameter
        """
        qdat.add_column(COSMOLOGIES[cosmology].comoving_distance(qdat['RVZ_RADVEL']).data*qdat['OFFSET'].data*np.pi/180,name='IMPACT',index=4)
        qdat['IMPACT'].unit = u.Mpc
        arr = COSMOLOGIES[cosmology].comoving_distance(qdat['RVZ_RADVEL']+qdat['RVZ_ERROR']).data*qdat['OFFSET'].data*np.pi/180 - qdat['IMPACT']
        arr[qdat['RVZ_ERROR'].mask] = np.nan
        qdat.add_column(arr,name='IMPACT_POSERR',index=5)
        arr = qdat['IMPACT'] - COSMOLOGIES[cosmology].comoving_distance(qdat['RVZ_RADVEL']-qdat['RVZ_ERROR']).data*qdat['OFFSET'].data*np.pi/180
        arr[qdat['RVZ_ERROR'].mask] = np.nan
        qdat.add_column(arr,name='IMPACT_NEGERR',index=6)
        qdat['IMPACT_POSERR'].unit = u.Mpc
        qdat['IMPACT_NEGERR'].unit = u.Mpc
        """
        qdat.add_column(COSMOLOGIES[cosmology].comoving_distance(qdat['BEST_Z']).data*qdat['OFFSET'].data*np.pi/180,name='IMPACT',index=4)
        qdat['IMPACT'].unit = u.Mpc
        arr = COSMOLOGIES[cosmology].comoving_distance(qdat['BEST_Z']+qdat['BEST_Z_ERROR']).data*qdat['OFFSET'].data*np.pi/180 - qdat['IMPACT']
        arr[qdat['BEST_Z_ERROR'].mask] = np.nan
        qdat.add_column(arr,name='IMPACT_POSERR',index=5)
        arr = qdat['IMPACT'] - COSMOLOGIES[cosmology].comoving_distance(qdat['BEST_Z']-qdat['BEST_Z_ERROR']).data*qdat['OFFSET'].data*np.pi/180
        arr[qdat['BEST_Z_ERROR'].mask] = np.nan
        qdat.add_column(arr,name='IMPACT_NEGERR',index=6)
        qdat['IMPACT_POSERR'].unit = u.Mpc
        qdat['IMPACT_NEGERR'].unit = u.Mpc

        qdat.add_column(np.nan*np.ones(len(qdat['IMPACT']),dtype=float),name='DM_EST',index=7)
        qdat['DM_EST'].unit = u.pc/u.cm**3
        qdat.add_column(np.nan*np.ones(len(qdat['IMPACT']),dtype=float),name='DM_EST_ERROR',index=8)
        qdat['DM_EST_ERROR'].unit = u.pc/u.cm**3

        
        qdat.add_column(np.nan*np.ones(len(qdat['IMPACT']),dtype=float),name='M_INPUT',index=7)
        qdat['M_INPUT'].unit = u.Msun
        qdat.add_column(np.nan*np.ones(len(qdat['IMPACT']),dtype=float),name='M_INPUT_ERROR',index=8)
        qdat['M_INPUT_ERROR'].unit = u.Msun


        qdat.add_column(np.nan*np.ones(len(qdat['IMPACT']),dtype=float),name='R_EST',index=9)
        qdat['R_EST'].unit = u.Mpc
        qdat.add_column(np.nan*np.ones(len(qdat['IMPACT']),dtype=float),name='R_EST_ERROR',index=10)
        qdat['R_EST_ERROR'].unit = u.Mpc

        qdat.add_column(np.nan*np.ones(len(qdat['IMPACT']),dtype=float),name='R_ANGLE_EST',index=11)
        qdat['R_ANGLE_EST'].unit = u.arcmin
        qdat.add_column(np.nan*np.ones(len(qdat['IMPACT']),dtype=float),name='R_ANGLE_EST_ERROR',index=12)
        qdat['R_ANGLE_EST_ERROR'].unit = u.arcmin

        """
        qdat.add_column(COSMOLOGIES[cosmology].comoving_distance(qdat['RVZ_RADVEL']).data,name='COMOVING_DIST_EST',index=13)
        qdat['COMOVING_DIST_EST'].unit = u.Mpc
        arr = COSMOLOGIES[cosmology].comoving_distance(qdat['RVZ_RADVEL']+qdat['RVZ_ERROR']).data-qdat['COMOVING_DIST_EST'].data
        arr[qdat['RVZ_ERROR'].mask] = np.nan
        qdat.add_column(arr,name='COMOVING_DIST_EST_POSERR',index=14)
        arr = qdat['COMOVING_DIST_EST'].data-COSMOLOGIES[cosmology].comoving_distance(qdat['RVZ_RADVEL']-qdat['RVZ_ERROR']).data
        arr[qdat['RVZ_ERROR'].mask] = np.nan
        qdat.add_column(arr,name='COMOVING_DIST_EST_NEGERR',index=15)
        qdat['COMOVING_DIST_EST_POSERR'].unit = u.Mpc
        qdat['COMOVING_DIST_EST_NEGERR'].unit = u.Mpc
        """
        qdat.add_column(COSMOLOGIES[cosmology].comoving_distance(qdat['BEST_Z']).data,name='COMOVING_DIST_EST',index=13)
        qdat['COMOVING_DIST_EST'].unit = u.Mpc
        arr = COSMOLOGIES[cosmology].comoving_distance(qdat['BEST_Z']+qdat['BEST_Z_ERROR']).data-qdat['COMOVING_DIST_EST'].data
        arr[qdat['BEST_Z_ERROR'].mask] = np.nan
        qdat.add_column(arr,name='COMOVING_DIST_EST_POSERR',index=14)
        arr = qdat['COMOVING_DIST_EST'].data-COSMOLOGIES[cosmology].comoving_distance(qdat['BEST_Z']-qdat['BEST_Z_ERROR']).data
        arr[qdat['BEST_Z_ERROR'].mask] = np.nan
        qdat.add_column(arr,name='COMOVING_DIST_EST_NEGERR',index=15)
        qdat['COMOVING_DIST_EST_POSERR'].unit = u.Mpc
        qdat['COMOVING_DIST_EST_NEGERR'].unit = u.Mpc

        qdat.add_column(np.nan*np.ones(len(qdat['IMPACT']),dtype=float),name='B_LOS_EST',index=16)
        qdat['B_LOS_EST'].unit = u.uG
        qdat.add_column(np.nan*np.ones(len(qdat['IMPACT']),dtype=float),name='B_LOS_EST_ERROR',index=17)
        qdat['B_LOS_EST_ERROR'].unit = u.uG

        qdat.add_column(np.nan*np.ones(len(qdat['IMPACT']),dtype=float),name='B0_LOS_EST',index=18)
        qdat['B0_LOS_EST'].unit = u.uG
        qdat.add_column(np.nan*np.ones(len(qdat['IMPACT']),dtype=float),name='B0_LOS_EST_ERROR',index=19)
        qdat['B0_LOS_EST_ERROR'].unit = u.uG

        qdat.add_column(np.nan*np.ones(len(qdat['IMPACT']),dtype=float),name='RM_EST',index=20)
        qdat['RM_EST'].unit = u.rad/u.m**2
        qdat.add_column(np.nan*np.ones(len(qdat['IMPACT']),dtype=float),name='RM_EST_ERROR',index=21)
        qdat['RM_EST_ERROR'].unit = u.rad/u.m**2


        #H1 Mass estimate
        if 'm21' in qdat.columns:
            qdat.add_column((1/0.2366)*(10**((-15.84 + qdat['m21'])/(-2.5))),name='FLUX_HI',index=22)
            qdat['FLUX_HI'].unit = u.Jy*u.km/u.s
            qdat.add_column((2.36e5)*(qdat['COMOVING_DIST_EST']**2)*qdat['FLUX_HI'],name='MASS_HI',index=23)
            qdat['MASS_HI'].unit = u.Msun

        print("Found " + str(len(qdat)) + " sources",file=lf)
    else:
        print("No sources found",file=lf)
    lf.close()
    return qdat

def plot_galaxies(ra,dec,radius,cosmology,redshift=-1,qdat=None,figsize=(18,12),save=False,savedir=''):
    """
    Takes a dataframe of galaxies and plots them around the FRB
    """
    


    f,(ax,ax3) = plt.subplots(1,2,gridspec_kw={'width_ratios':[2,1]},figsize=figsize)

    ax.plot(ra,dec,"*",markersize=10,color='red')
    if qdat is not None:
        for i in range(len(qdat)):
            c = SkyCoord(qdat['RA'][i]+" "+qdat['DEC'][i],frame='icrs',unit=(u.hourangle,u.deg))
            ax.plot(c.ra.value,c.dec.value,"x",markersize=20,color='blue')
            #print(c.ra.value,c.dec.value,qdat['RA'][i]+" "+qdat['DEC'][i],qdat['GALDIM_MAJAXIS'][i]/60,qdat['GALDIM_MINAXIS'][i]/60)
                
            if ~(qdat['GALDIM_MAJAXIS'].mask[i]) and ~(qdat['GALDIM_MAJAXIS'].mask[i]) and ~(qdat['GALDIM_ANGLE'].mask[i]):
                #make patch
                if np.isnan(qdat['R_ANGLE_EST'][i]):
                    gal = Ellipse((c.ra.value,c.dec.value),
                        width=qdat['GALDIM_MAJAXIS'][i]/60,
                        height=qdat['GALDIM_MINAXIS'][i]/60,
                        angle=qdat['GALDIM_ANGLE'][i],
                        alpha=0.5,color='blue',linewidth=4,facecolor='blue',fill=True)
                else:
                    gal = Ellipse((c.ra.value,c.dec.value),
                        width=qdat['R_ANGLE_EST'][i]/60,
                        height=qdat['R_ANGLE_EST'][i]/60,
                        angle=0,
                        alpha=0.5,color='red',linewidth=4,facecolor='red',fill=True)
                ax.add_patch(gal)
    #print(ra,dec)
    rad = Ellipse((ra,dec),width=radius*2/60,height=radius*2/60,alpha=1,color='red',fill=False,linewidth=2)
    ax.add_patch(rad)
    ax.set_xlim(ra-radius*1.5/60,ra+radius*1.5/60)
    ax.set_ylim(dec-radius*1.5/60,dec+radius*1.5/60)
    ax.set_xlabel(r'RA$(^{\circ})$')
    ax.set_ylabel(r'DEC$(^{\circ})$')


    #plot redshift vs offset
    if redshift!=-1:
        #ax2.plot(0,redshift,"*",markersize=10,color='red')
        #ax2.plot([0,-radius/60,radius/60,0],[0,redshift,redshift,0],color='red',linewidth=2)
    
        ax3.plot(0,redshift,"*",markersize=10,color='red')
        ax3.plot([0,-(radius*np.pi/180/60)*COSMOLOGIES[cosmology].comoving_distance(redshift).value,(radius*np.pi/180/60)*COSMOLOGIES[cosmology].comoving_distance(redshift).value,0],[0,redshift,redshift,0],color='red',linewidth=2)

    #ax2.set_ylim(0,1.5*np.abs(redshift))
    ax3.set_ylim(0,1.5*np.abs(redshift))
    zcol = 'BEST_Z'
    zcolerr = 'BEST_Z_ERROR'
    if qdat is not None and 'BEST_Z' not in qdat.columns:
        zcol = 'RVZ_RADVEL'
        zcolerr = 'RVZ_RADVEL_ERROR'
    if qdat is not None:
        for i in range(len(qdat)):
            if ~qdat['RA'].mask[i] and ~qdat['DEC'].mask[i] and ~qdat[zcol].mask[i]:
                c = SkyCoord(qdat['RA'][i]+qdat['DEC'][i],frame='icrs',unit=(u.hourangle,u.deg))
                #offset = np.sqrt((c.ra.value-ra)**2 + (c.dec.value-dec)**2)
                sign = ((c.ra.value-ra)/np.abs(c.ra.value-ra))*((c.dec.value-dec)/np.abs(c.dec.value-dec))
                z = qdat[zcol][i]
                #ax2.plot(sign*offset,z,'x',markersize=20,color='blue')
                #ax2.plot([0,sign*offset],[z,z],color='blue',linewidth=3)

                ax3.plot(sign*qdat['IMPACT'][i],z,'x',markersize=20,color='blue')
                ax3.plot([0,sign*qdat['IMPACT'][i]],[z,z],color='blue',linewidth=3)
    #ax2.set_xlabel(r'Offset ($^{\circ}$)')
    #ax2.set_ylabel("Redshift z")
    #ax2.set_xlim(-radius*1.5/60,radius*1.5/60)

    ax3.axvline(0,color='red',linewidth=2,linestyle='--')
    ax3.set_xlabel(r'$b_\perp$ (Mpc)')
    ax3.set_ylabel("Redshift z")
    ax3.set_xlim(-(radius*1.5/60)*(np.pi/180)*COSMOLOGIES[cosmology].comoving_distance(redshift).value,(radius*1.5/60)*(np.pi/180)*COSMOLOGIES[cosmology].comoving_distance(redshift).value)
    if save:
        plt.savefig(savedir + "/SIMBAD_query_plot.pdf")
    plt.show()


def DM_int_vals(qdat_row,mass_low,mass_high,cosmology,mass_type):
    """
    This function uses NFW profile to get DM and error from mass
    ,redshift, impact parameter, and angular size
    """

    zcol = 'BEST_Z'
    zcolerr = 'BEST_Z_ERROR'
    if qdat_row is not None and 'BEST_Z' not in qdat_row.columns:
        zcol = 'RVZ_RADVEL'
        zcolerr = 'RVZ_RADVEL_ERROR'
 
    #make NFW profile
    if ~np.isnan(qdat_row[zcolerr]) and ~np.isnan(qdat_row['IMPACT_NEGERR']):
        nfw = physical_models.NFW(mass=mass_low*u.Msun,redshift=qdat_row[zcol]-qdat_row[zcolerr],massfactor=mass_type,cosmo=COSMOLOGIES[cosmology])
        #get impact parameter
        b = (qdat_row['IMPACT']-qdat_row['IMPACT_NEGERR'])*u.Mpc
    else:
        nfw = physical_models.NFW(mass=mass_low*u.Msun,redshift=qdat_row[zcol],massfactor=mass_type,cosmo=COSMOLOGIES[cosmology])
        #get impact parameter
        b = (qdat_row['IMPACT'])*u.Mpc

    #get density 
    rho = nfw.evaluate(b,mass=nfw.mass,concentration=nfw.concentration,redshift=nfw.redshift) #Msun/kpc^3

    #convert to number density
    ne = rho.to(u.kg/u.cm**3)/(m_p.to(u.kg))#1/cm^3

    #estimate path length from virial radius and impact parameter
    l = np.sqrt(nfw.r_virial.to(u.pc)**2 + b.to(u.pc)**2)

    #DM
    dm1 = (ne*l).value
    rvir1 = nfw.r_virial.to(u.Mpc).value

    #make NFW profile
    if ~np.isnan(qdat_row[zcolerr]) and ~np.isnan(qdat_row['IMPACT_POSERR']):
        nfw = physical_models.NFW(mass=mass_high*u.Msun,redshift=qdat_row[zcol]+qdat_row[zcolerr],massfactor=mass_type,cosmo=COSMOLOGIES[cosmology])
        #get impact parameter
        b = (qdat_row['IMPACT']+qdat_row['IMPACT_POSERR'])*u.Mpc
    else:
        nfw = physical_models.NFW(mass=mass_low*u.Msun,redshift=qdat_row[zcol],massfactor=mass_type,cosmo=COSMOLOGIES[cosmology])
        #get impact parameter
        b = (qdat_row['IMPACT'])*u.Mpc

    #get density 
    rho = nfw.evaluate(b,mass=nfw.mass,concentration=nfw.concentration,redshift=nfw.redshift) #Msun/kpc^3

    #convert to number density
    ne = rho.to(u.kg/u.cm**3)/(m_p.to(u.kg))#1/cm^3

    #estimate path length from virial radius and impact parameter
    l = np.sqrt(nfw.r_virial.to(u.pc)**2 + b.to(u.pc)**2)

    #DM
    dm2 = (ne*l).value
    rvir2 = nfw.r_virial.to(u.Mpc).value

    dm = (dm1 + dm2)/2
    dm_err = np.abs(dm1 - dm2)/2

    rvir = (rvir1 + rvir2)/2
    rvir_err = np.abs(rvir1 - rvir2)/2

    if dm_err == 0: dm_err = np.nan
    if rvir_err == 0: rvir_err = np.nan
    return dm,dm_err,rvir,rvir_err

B_REF_MW = 6*u.G
R_REF_MW = 8*u.kpc
B0_MW = 48*u.G
R0_MW = 1*u.kpc
RVIR_MW = 200*u.kpc

def RM_int_vals(qdat_row,bfield_low,bfield_high,cosmology,mass_type,bfield_types):
    """
    This function uses NFW profile and B field estiamtes to get RM and error
    """
    zcol = 'BEST_Z'
    zcolerr = 'BEST_Z_ERROR'
    if qdat_row is not None and 'BEST_Z' not in qdat_row.columns:
        zcol = 'RVZ_RADVEL'
        zcolerr = 'RVZ_RADVEL_ERROR'

    #make NFW profile
    if ~np.isnan(qdat_row[zcolerr]) and ~np.isnan(qdat_row['IMPACT_NEGERR']):
        nfw = physical_models.NFW(mass=(qdat_row['M_INPUT']-qdat_row['M_INPUT_ERROR'])*u.Msun,redshift=qdat_row[zcol]-qdat_row[zcolerr],massfactor=mass_type,cosmo=COSMOLOGIES[cosmology])
        #get impact parameter
        b = (qdat_row['IMPACT']-qdat_row['IMPACT_NEGERR'])*u.Mpc
    else:
        nfw = physical_models.NFW(mass=(qdat_row['M_INPUT']-qdat_row['M_INPUT_ERROR'])*u.Msun,redshift=qdat_row[zcol],massfactor=mass_type,cosmo=COSMOLOGIES[cosmology])
        #get impact parameter
        b = (qdat_row['IMPACT'])*u.Mpc
        
    rvir = nfw.r_virial.to(u.Mpc)

    #estimate reference radius
    R0 = RVIR_MW*(R0_MW/rvir.to(u.kpc)) #kpc

    #if given scale factors, need to get B at impact param
    if bfield_types[0] == 'x':
        bf0_low = bfield_low*B0_MW #uG
        bf_low = bf0_low*R0/b.to(u.kpc) #uG
    else: #if given uG need to get B at r0
        bf_low = bfield_low*u.uG
        bf0_low = bf_low*b.to(u.kpc)/R0 #uG

    #get density 
    rho = nfw.evaluate(b,mass=nfw.mass,concentration=nfw.concentration,redshift=nfw.redshift) #Msun/kpc^3

    #convert to number density
    ne = rho.to(u.kg/u.cm**3)/(m_p.to(u.kg))#1/cm^3

    #estimate path length from virial radius and impact parameter
    l = np.sqrt(nfw.r_virial.to(u.pc)**2 + b.to(u.pc)**2)

    #convert to RM
    RM_low = (0.81*((u.rad/u.m**2)*(u.cm**3/u.pc)/u.uG)*bf_low*ne*l).value #rad/m^2

    #make NFW profile
    if ~np.isnan(qdat_row[zcolerr]) and ~np.isnan(qdat_row['IMPACT_POSERR']):
        nfw = physical_models.NFW(mass=(qdat_row['M_INPUT']+qdat_row['M_INPUT_ERROR'])*u.Msun,redshift=qdat_row[zcol]+qdat_row[zcolerr],massfactor=mass_type,cosmo=COSMOLOGIES[cosmology])
        #get impact parameter
        b = (qdat_row['IMPACT']+qdat_row['IMPACT_POSERR'])*u.Mpc
    else:
        nfw = physical_models.NFW(mass=(qdat_row['M_INPUT']+qdat_row['M_INPUT_ERROR'])*u.Msun,redshift=qdat_row[zcol],massfactor=mass_type,cosmo=COSMOLOGIES[cosmology])
        #get impact parameter
        b = (qdat_row['IMPACT'])*u.Mpc

    rvir = nfw.r_virial.to(u.Mpc)

    #estimate reference radius
    R0 = RVIR_MW*(R0_MW/rvir.to(u.kpc)) #kpc

    #if given scale factors, need to get B at impact param
    if bfield_types[1] == 'x':
        bf0_high = bfield_high*B0_MW #uG
        bf_high = bf0_high*R0/b.to(u.kpc) #uG
    else: #if given uG need to get B at r0
        bf_high = bfield_high*u.uG
        bf0_high = bf_high*b.to(u.kpc)/R0 #uG

    #get density 
    rho = nfw.evaluate(b,mass=nfw.mass,concentration=nfw.concentration,redshift=nfw.redshift) #Msun/kpc^3

    #convert to number density
    ne = rho.to(u.kg/u.cm**3)/(m_p.to(u.kg))#1/cm^3

    #estimate path length from virial radius and impact parameter
    l = np.sqrt(nfw.r_virial.to(u.pc)**2 + b.to(u.pc)**2)
    
    #convert to RM
    RM_high = (0.81*((u.rad/u.m**2)*(u.cm**3/u.pc)/u.uG)*bf_high*ne*l).value #rad/m^2

    
    bf = (bf_low.value + bf_high.value)/2
    bf_err = np.abs(bf_low.value - bf_high.value)/2

    bf0 = (bf0_low.value + bf0_high.value)/2
    bf0_err = np.abs(bf0_low.value - bf0_high.value)/2

    rm = (RM_high + RM_low)/2
    rm_err = np.abs(RM_high - RM_low)/2

    if bf_err == 0: bf_err = np.nan
    if bf0_err == 0: bf0_err = np.nan
    if rm_err == 0: rm_err = np.nan
    return rm,rm_err,bf,bf_err,bf0,bf0_err
