import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import pyne2001
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.cosmology import WMAP1,WMAP3,WMAP5,WMAP7,WMAP9,Planck13,Planck15,Planck18,default_cosmology
import astropy.units as u
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
def DM_host_limits(DMobs,frb_z,frb_gl,frb_gb,res=10000,plot=False,DMhalo=DMHALOEST,siglevel=DEF_SIGLEVEL):
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
    Phost = copy.deepcopy(P3)
    Phost = Phost/np.sum(Phost*(DM_axis[1]*(1+ztest)-DM_axis[0]*(1+ztest)))

    if plot:
        plt.plot(DM_axis,Pobs/np.nanmax(Pobs),label=r'$DM_{obs}$')
        plt.plot(DM_axis,Pmw/np.nanmax(Pmw),label=r'$DM_{MW}$')
        plt.plot(DM_axis,Pigm/np.nanmax(Pigm),label=r'$DM_{IGM}$')
        plt.plot(DM_axis,Pmwhalo/np.nanmax(Pmwhalo),label=r'$DM_{MW,halo}$')
        plt.plot(DM_axis,Phost/np.nanmax(Phost),label=r'$DM_{host}$',linewidth=4)
        plt.plot(DM_axis*(1+ztest),Phost/np.nanmax(Phost),label=r'$DM_{host}(1+z)$',linewidth=4)
        plt.xlim(-DMobs*2,DMobs*2)
        plt.legend(loc='upper left')
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
def DM_host_dist(DMobs,frb_z,frb_gl,frb_gb,res=10000,plot=False,DMhalo=DMHALOEST,siglevel=DEF_SIGLEVEL):
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
        plt.plot(DM_axis,Pmwhalo/np.nanmax(Pmwhalo),label=r'$DM_{MW,halo}$')
        plt.plot(DM_axis,Phost/np.nanmax(Phost),label=r'$DM_{host}$/(1+z)',linewidth=4)
        plt.plot(DM_axis*(1+ztest),Phost/np.nanmax(Phost),label=r'$DM_{host}$',linewidth=4)
        plt.axvline(Phost_exp,color="purple")
        plt.axvspan(low,upp,color='purple',alpha=0.1)
        plt.text(Phost_exp+10,1,'$DM_{{host}}={a:.2f}^{{+{b:.2f}}}_{{-{c:.2f}}}\\,pc/cm^3$'.format(a=Phost_exp,b=upp-Phost_exp,c=Phost_exp-low),
                backgroundcolor='thistle',fontsize=18)
        plt.xlim(-DMobs*2,DMobs*2)
        plt.legend(loc='upper left')
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
                  plot=False):
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


    Phost = copy.deepcopy(P2)
    Phost_z = Phost/np.sum(Phost*(RM_axis[1]-RM_axis[0]))
    Phost = Phost/np.sum(Phost*(RM_axis[1]-RM_axis[0])*((1+ztest)**2))

    
    #plotting
    if plot:
        plt.figure(figsize=(12,6))
        plt.plot(RM_axis,Pobs/np.nanmax(Pobs),label=r'$RM_{obs}$')
        plt.plot(RM_axis,Pmw/np.nanmax(Pmw),label=r'$RM_{MW}$')
        plt.plot(RM_axis,Pion/np.nanmax(Pion),label=r'$RM_{ION}$')
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
                  plot=False):
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
        'Galaxy',
        'Interacting Galaxies',
        'Pair of Galaxies',
        'Group of Galaxies',
        'Compact Group of Galaxies',
        'Cluster of Galaxies',
        'Proto Cluster of Galaxies',
        'Supercluster of Galaxies'
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
        typestring = "otypes in (\'Galaxy\',\'IG\',\'PaG\',\'GrG\',\'PCG\',\'SCG\')"
    else:
        typestring = "otypes in ("
        for i in types:
            typestring += "\'"+i+"\',"
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


        print("Found " + str(len(qdat)) + " sources",file=lf)
    else:
        print("No sources found",file=lf)
    lf.close()
    return qdat

def plot_galaxies(ra,dec,radius,cosmology,redshift=-1,qdat=None,figsize=(18,4)):
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
                gal = Ellipse((c.ra.value,c.dec.value),
                        width=qdat['GALDIM_MAJAXIS'][i]/60,
                        height=qdat['GALDIM_MINAXIS'][i]/60,
                        angle=qdat['GALDIM_ANGLE'][i],
                        alpha=0.5,color='blue',linewidth=4,facecolor='blue',fill=True)
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



