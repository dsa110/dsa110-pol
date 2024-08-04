import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import pyne2001
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.cosmology import WMAP1,WMAP3,WMAP5,WMAP7,WMAP9,Planck13,Planck15,Planck18,default_cosmology
import astropy.units as u
from astropy.modeling import physical_models
from astropy.constants import m_p,m_e
import copy
from scipy.signal import peak_widths
from astroquery.simbad import Simbad
from dsapol.RMcal import logfile
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
            Pint = np.zeros(len(DM_axis))
            Pint[np.argmin(np.abs(DM_axis*(1+intervener_zs[i])-intervener_DMs[i]))] = 1
        Pint[DM_axis*(1+intervener_zs[i])<0] = 0
        Pint[DM_axis*(1+intervener_zs[i])>DMobs] = 0
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

    if plot:
        plt.figure(figsize=(24,12))
        plt.plot(DM_axis,cumdisthost)
        plt.plot(DM_axis,Phost)
        plt.axvline(low,color="red")
        plt.axvline(upp,color="red")
        plt.show()
    return Phost_exp,low,upp


#convolution method to derive DMhost distribution
def DM_host_dist(DMobs,frb_z,frb_gl,frb_gb,res=10000,plot=False,DMhalo=DMHALOEST,siglevel=DEF_SIGLEVEL,intervener_DMs=[],intervener_DM_errs=[],intervener_zs=[]):
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
        plt.figure(figsize=(18,12))

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
            Pint = np.zeros(len(DM_axis))
            Pint[np.argmin(np.abs(DM_axis*(1+intervener_zs[i])-intervener_DMs[i]))] = 1
        Pint[DM_axis*(1+intervener_zs[i])<0] = 0
        Pint[DM_axis*(1+intervener_zs[i])>DMobs] = 0
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
        plt.text(Phost_exp+10,1,'$DM_{{host}}={a:.2f}^{{+{b:.2f}}}_{{-{c:.2f}}}\\,pc/cm^3$'.format(a=Phost_exp,b=upp-Phost_exp,c=Phost_exp-low),
                backgroundcolor='thistle',fontsize=18)
        plt.xlim(0,DMobs*2)
        plt.legend(loc='upper right')
        plt.xlabel("DM")
        plt.ylabel("PDF")
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
        if ~np.isnan(intervener_RM_errs[i]) and intervener_RM_errs[i] != 0:
            Pint = gaus(RM_axis*((1+intervener_zs[i])**2),intervener_RMs[i],intervener_RM_errs[i])
        else:
            Pint = np.zeros(len(RM_axis))
            Pint[np.argmin(np.abs(RM_axis*((1+intervener_zs[i])**2)-intervener_RMs[i]))] = 1
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
                  plot=False,intervener_RMs=[],intervener_RM_errs=[],intervener_zs=[]):
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
        if ~np.isnan(intervener_RM_errs[i]) and intervener_RM_errs[i] != 0:
            Pint = gaus(RM_axis*((1+intervener_zs[i])**2),intervener_RMs[i],intervener_RM_errs[i])
        else:
            Pint = np.zeros(len(RM_axis))
            Pint[np.argmin(np.abs(RM_axis*((1+intervener_zs[i])**2)-intervener_RMs[i]))] = 1
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
        plt.figure(figsize=(18,12))
        plt.plot(RM_axis,Pobs/np.nanmax(Pobs),label=r'$RM_{obs}$')
        plt.plot(RM_axis,Pmw/np.nanmax(Pmw),label=r'$RM_{MW}$')
        plt.plot(RM_axis,Pion/np.nanmax(Pion),label=r'$RM_{ION}$')
        for i in range(len(intervener_RMs)):
            plt.plot(RM_axis*((1+intervener_zs[i])**2),Pints[i]/np.nanmax(Pints[i]),label=r'$RM_{{int,{i}}}/(1+z_{{int,{i}}})^2$'.format(i=i))
        plt.plot(RM_axis,Phost/np.nanmax(Phost),label=r'$RM_{host}/(1+z)^2$',linewidth=4)
        plt.plot(RM_axis*((1+ztest)**2),Phost/np.nanmax(Phost),label=r'$RM_{host}$',linewidth=4)
        plt.axvline(Phost_exp,color="purple")
        plt.axvspan(low,upp,color='purple',alpha=0.1)
        plt.text(Phost_exp+10,1,'$RM_{{host}}={a:.2f}^{{+{b:.2f}}}_{{-{c:.2f}}}\\,rad/m^2$'.format(a=Phost_exp,b=upp-Phost_exp,c=Phost_exp-low),
                backgroundcolor='thistle',fontsize=18)
        plt.xlim(-RMobs*2,RMobs*2)
        plt.legend(loc='upper left')
        plt.xlabel("RM")
        plt.ylabel("PDF")
        plt.show()
    
    return Phost,RM_axis*((1+ztest)**2)



#convolution method to derive B||host (note, need to get RMhost and DM host distributions first)
def Bhost_dist(DMhost,dmdist,DMaxis,RMhost,RMhosterr,res=10000,res2=500,siglevel=DEF_SIGLEVEL,plot=False,buff=50):
    
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
        plt.figure(figsize=(18,12))
        plt.plot(Baxis,Bdist,linewidth=4,color='tab:purple')
        plt.axvline(B_exp,color="purple")
        plt.axvspan(low,upp,color='purple',alpha=0.1)
        plt.text(B_exp+0.5,np.nanmax(Bdist),'$B_{{||,host}}={a:.2f}^{{+{b:.2f}}}_{{-{c:.2f}}}\\,\\mu G$'.format(a=B_exp,b=upp-B_exp,c=B_exp-low),
                backgroundcolor='thistle',fontsize=18)
        plt.xlabel(r'$B_{||,host}$')
        plt.ylabel("PDF")
        plt.xlim(-B_est*2,B_est*2)
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
        '2MASS',#2MASS
        ]
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
        2MASS
    """

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
    #create query that targets (1) galaxies and groups of galaxies (2) within specified radius (3) has redshift in specified range (4) in catalogs specified
    query_string = "region(CIRCLE,icrs,{ra} {s}{dec},{radius}) & rvtype=\'z\' & {ts}{cs}{zs}".format(ra=ra,s="+" if dec>0 else "-",dec=np.abs(dec),radius=radius,ts=typestring,cs=catstring,zs=redstring) 
    lf = open(logfile,"a")
    print(query_string,file=lf)

    #send query
    qdat=customSimbad.query_criteria(query_string)
    if qdat is not None:
        #make column for offset
        c = SkyCoord([qdat['RA'][i] +qdat['DEC'][i] for i in range(len(qdat))],frame='icrs',unit=(u.hourangle,u.deg))
        qdat.add_column(SkyCoord(ra=ra*u.deg,dec=dec*u.deg,frame='icrs').separation(c).to(u.deg).value,name='OFFSET',index=3)
        qdat['OFFSET'].unit = u.deg

        #make column for impact parameter
        qdat.add_column(COSMOLOGIES[cosmology].comoving_distance(qdat['RVZ_RADVEL']).data*qdat['OFFSET'].data*np.pi/180,name='IMPACT',index=4)
        qdat['IMPACT'].unit = u.Mpc
        qdat.add_column(COSMOLOGIES[cosmology].comoving_distance(qdat['RVZ_RADVEL']+qdat['RVZ_ERROR']).data*qdat['OFFSET'].data*np.pi/180 - qdat['IMPACT'],name='IMPACT_POSERR',index=5)
        qdat['IMPACT_POSERR'].unit = u.Mpc
        qdat.add_column(qdat['IMPACT'] - COSMOLOGIES[cosmology].comoving_distance(qdat['RVZ_RADVEL']-qdat['RVZ_ERROR']).data*qdat['OFFSET'].data*np.pi/180,name='IMPACT_NEGERR',index=6)
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

        qdat.add_column(COSMOLOGIES[cosmology].comoving_distance(qdat['RVZ_RADVEL']).data,name='COMOVING_DIST_EST',index=13)
        qdat['COMOVING_DIST_EST'].unit = u.Mpc
        qdat.add_column(COSMOLOGIES[cosmology].comoving_distance(qdat['RVZ_RADVEL']+qdat['RVZ_ERROR']).data-qdat['COMOVING_DIST_EST'].data,name='COMOVING_DIST_EST_POSERR',index=14)
        qdat['COMOVING_DIST_EST_POSERR'].unit = u.Mpc
        qdat.add_column(qdat['COMOVING_DIST_EST'].data-COSMOLOGIES[cosmology].comoving_distance(qdat['RVZ_RADVEL']-qdat['RVZ_ERROR']).data,name='COMOVING_DIST_EST_NEGERR',index=15)
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


        print("Found " + str(len(qdat)) + " sources",file=lf)
    else:
        print("No sources found",file=lf)
    lf.close()
    return qdat

def plot_galaxies(ra,dec,radius,cosmology,redshift=-1,qdat=None,figsize=(18,12)):
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
    if qdat is not None:
        for i in range(len(qdat)):
            if ~qdat['RA'].mask[i] and ~qdat['DEC'].mask[i] and ~qdat['RVZ_RADVEL'].mask[i]:
                c = SkyCoord(qdat['RA'][i]+qdat['DEC'][i],frame='icrs',unit=(u.hourangle,u.deg))
                #offset = np.sqrt((c.ra.value-ra)**2 + (c.dec.value-dec)**2)
                sign = ((c.ra.value-ra)/np.abs(c.ra.value-ra))*((c.dec.value-dec)/np.abs(c.dec.value-dec))
                z = qdat['RVZ_RADVEL'][i]
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

    plt.show()


def DM_int_vals(qdat_row,mass_low,mass_high,cosmology,mass_type):
    """
    This function uses NFW profile to get DM and error from mass
    ,redshift, impact parameter, and angular size
    """

    #make NFW profile
    try:
        nfw = physical_models.NFW(mass=mass_low*u.Msun,redshift=qdat_row['RVZ_RADVEL']-qdat_row['RVZ_ERROR'],massfactor=mass_type,cosmo=COSMOLOGIES[cosmology])
    except:
        nfw = physical_models.NFW(mass=mass_low*u.Msun,redshift=qdat_row['RVZ_RADVEL'],massfactor=mass_type,cosmo=COSMOLOGIES[cosmology])
    #get impact parameter
    b = (qdat_row['IMPACT']-qdat_row['IMPACT_NEGERR'])*u.Mpc

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
    try:
        nfw = physical_models.NFW(mass=mass_high*u.Msun,redshift=qdat_row['RVZ_RADVEL']+qdat_row['RVZ_ERROR'],massfactor=mass_type,cosmo=COSMOLOGIES[cosmology])
    except:
        nfw = physical_models.NFW(mass=mass_low*u.Msun,redshift=qdat_row['RVZ_RADVEL'],massfactor=mass_type,cosmo=COSMOLOGIES[cosmology])
    #get impact parameter
    b = (qdat_row['IMPACT']+qdat_row['IMPACT_POSERR'])*u.Mpc

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
    
    #make NFW profile
    try:
        nfw = physical_models.NFW(mass=(qdat_row['M_INPUT']-qdat_row['M_INPUT_ERROR'])*u.Msun,redshift=qdat_row['RVZ_RADVEL']-qdat_row['RVZ_ERROR'],massfactor=mass_type,cosmo=COSMOLOGIES[cosmology])
    except:
        nfw = physical_models.NFW(mass=(qdat_row['M_INPUT']-qdat_row['M_INPUT_ERROR'])*u.Msun,redshift=qdat_row['RVZ_RADVEL'],massfactor=mass_type,cosmo=COSMOLOGIES[cosmology])

    #get impact parameter and virial radius
    b = (qdat_row['IMPACT']-qdat_row['IMPACT_NEGERR'])*u.Mpc
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
    try:
        nfw = physical_models.NFW(mass=(qdat_row['M_INPUT']+qdat_row['M_INPUT_ERROR'])*u.Msun,redshift=qdat_row['RVZ_RADVEL']+qdat_row['RVZ_ERROR'],massfactor=mass_type,cosmo=COSMOLOGIES[cosmology])
    except:
        nfw = physical_models.NFW(mass=(qdat_row['M_INPUT']+qdat_row['M_INPUT_ERROR'])*u.Msun,redshift=qdat_row['RVZ_RADVEL'],massfactor=mass_type,cosmo=COSMOLOGIES[cosmology])

    #get impact parameter and virial radius
    b = (qdat_row['IMPACT']+qdat_row['IMPACT_POSERR'])*u.Mpc
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

    return rm,rm_err,bf,bf_err,bf0,bf0_err
