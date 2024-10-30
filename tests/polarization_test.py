import pytest
from matplotlib import pyplot as plt
import numpy as np
from dsapol import dsapol
from scipy.stats import norm


"""
Last Updated 2024-10-28
This script contains unit tests for polarization fraction and PA estimates, updated
since the PARSEC V1 release.
"""



def test_pol_auto_weighted():
    #tolerances
    pol_tolerance = 0.05
    pa_tolerance = 5*np.pi/180 #rad
    snr_tolerance = 0.3 #fraction of snr_test
    Lpol_true = 0.5
    Vpol_true = 0.5
    Tpol_true = np.sqrt(Lpol_true**2 + Vpol_true**2)
    PA = 0

    nsamps = 5120
    nchans = 6144
    loc = int(nsamps//2)

    for width in [50]:
        width_native = int(np.ceil((32.7e-6/256e-6)*width)) #because internally converted to 32.7 microseconds
        for snr_true in [10000]:
            I = np.zeros((nchans,nsamps)) + norm.rvs(loc=0,scale=1*np.sqrt(nchans)*np.sqrt(width),size=(nchans,nsamps))
            I[:,loc:loc+width] += snr_true
            #I += (snr_true*norm.pdf(np.arange(nsamps),loc=loc,scale=width)/np.max(norm.pdf(np.arange(nsamps),loc=loc,scale=width)))
            Q = np.zeros((nchans,nsamps)) + norm.rvs(loc=0,scale=1*np.sqrt(nchans)*np.sqrt(width),size=(nchans,nsamps))
            Q[:,loc:loc+width] += snr_true*Lpol_true*np.cos(2*PA)
            #Q += (snr_true*norm.pdf(np.arange(nsamps),loc=loc,scale=width)/np.max(norm.pdf(np.arange(nsamps),loc=loc,scale=width)))*Lpol_true*np.cos(2*PA)
            U = np.zeros((nchans,nsamps)) + norm.rvs(loc=0,scale=1*np.sqrt(nchans)*np.sqrt(width),size=(nchans,nsamps))
            U[:,loc:loc+width] += snr_true*Lpol_true*np.sin(2*PA)
            #U += (snr_true*norm.pdf(np.arange(nsamps),loc=loc,scale=width)/np.max(norm.pdf(np.arange(nsamps),loc=loc,scale=width)))*Lpol_true*np.sin(2*PA)
            V = np.zeros((nchans,nsamps)) + norm.rvs(loc=0,scale=1*np.sqrt(nchans)*np.sqrt(width),size=(nchans,nsamps))
            V[:,loc:loc+width] += snr_true*Vpol_true
            #V += (snr_true*norm.pdf(np.arange(nsamps),loc=loc,scale=width)/np.max(norm.pdf(np.arange(nsamps),loc=loc,scale=width)))*Vpol_true
            tsamp = 32.7e-6
            n_t = 1
            n_f = 1
            freq_test = [np.linspace(1250,1520,6144)]*4
            NOFFDEF = 2000
            buff=0

            #weight params
            n_t_weight = 2
            sf_window_weights = 7
            timeaxis = np.arange(nsamps)

            #weighted
            [(pol_f,pol_t,avg,sigma_frac,snr_frac),
            (L_f,L_t,avg_L,sigma_L,snr_L),
            (C_f_unbiased,C_t_unbiased,avg_C_abs,sigma_C_abs,snr_C),
            (C_f,C_t,avg_C,sigma_C,tmp),snr]=dsapol.get_pol_fraction(I,Q,U,V,width_native,tsamp,n_t,n_f,freq_test,
                                                                    int(NOFFDEF/n_t),plot=False,normalize=True,buff=buff,
                                                                    full=False,weighted=True,n_t_weight=n_t_weight,
                                                                    sf_window_weights=sf_window_weights,timeaxis=timeaxis)

            print("Testing pulse width=",width,"(native=",width_native,"),S/N=",snr_true,"...")
            #peak,timestart,timestop = dsapol.find_peak(I,width_native,tsamp,n_t,pre_calc_tf=False,buff=buff)
            #print(peak,timestart,timestop)
            print(avg,avg_L,avg_C,avg_C_abs)
            print(sigma_frac,sigma_L,sigma_C,sigma_C_abs)
            print(snr,snr_frac,snr_L,snr_C)
            assert(np.abs(avg - Tpol_true)<pol_tolerance)#*sigma_frac)
            assert(np.abs(avg_L - Lpol_true)<pol_tolerance)#*sigma_L)
            assert(np.abs(avg_C - Vpol_true) <pol_tolerance)#*sigma_C)
            assert(np.abs(avg_C_abs - Vpol_true) < pol_tolerance)#*sigma_C_abs)
            assert(np.abs(snr - snr_true) < snr_tolerance*snr_true)
            assert(np.abs(snr_frac - snr_true*Tpol_true) < snr_tolerance*snr_true*Tpol_true)
            assert(np.abs(snr_L - snr_true*Lpol_true) < snr_tolerance*snr_true*Lpol_true)
            assert(np.abs(snr_C - snr_true*Vpol_true) < snr_tolerance*snr_true*Vpol_true)
            assert(len(pol_f) == len(L_f) == len(C_f) == len(C_f_unbiased) == nchans)
            assert(len(pol_t) == len(L_t) == len(C_t) == len(C_t_unbiased) == nsamps)
            assert(np.all(pol_t[loc:loc+width]>=0))
            assert(np.all(L_t[loc:loc+width]>=0))
            assert(np.all(C_t[loc:loc+width]>=0))
            assert(np.all(pol_f[loc:loc+width]>=0))
            assert(np.all(L_f[loc:loc+width]>=0))
            assert(np.all(C_f[loc:loc+width]>=0))

            PA_f,PA_t,PA_f_errs,PA_t_errs,PA_avg,PA_err = dsapol.get_pol_angle(I,Q,U,V,width_native,tsamp,n_t,n_f,freq_test,
                                                                               n_off=int(NOFFDEF/n_t),normalize=True,buff=buff,
                                                                               weighted=True,n_t_weight=n_t_weight,
                                                                    sf_window_weights=sf_window_weights,timeaxis=timeaxis)
            assert(np.abs(PA_avg - PA)<pol_tolerance)#*PA_err)
            assert(len(PA_t) == len(PA_t_errs) == nsamps)
            assert(len(PA_f) == len(PA_f_errs) == nchans)
            assert(np.all((np.abs(PA_t-PA)<pa_tolerance)[loc:loc+width]))#PA_t_errs)[loc:loc+width])) 


            
    return

def test_pol_custom_weighted():
    #tolerances
    pol_tolerance = 0.05
    pa_tolerance = 5*np.pi/180 #rad
    snr_tolerance = 0.3 #fraction of snr_test
    Lpol_true = 0.5
    Vpol_true = 0.5
    Tpol_true = np.sqrt(Lpol_true**2 + Vpol_true**2)
    PA = 0

    nsamps = 5120
    nchans = 6144
    loc = int(nsamps//2)

    for width in [50]:
        width_native = int(np.ceil((32.7e-6/256e-6)*width)) #because internally converted to 32.7 microseconds
        for snr_true in [10000]:
            I = np.zeros((nchans,nsamps)) + norm.rvs(loc=0,scale=1*np.sqrt(nchans)*np.sqrt(width),size=(nchans,nsamps))
            I[:,loc:loc+width] += snr_true
            #I += (snr_true*norm.pdf(np.arange(nsamps),loc=loc,scale=width)/np.max(norm.pdf(np.arange(nsamps),loc=loc,scale=width)))
            Q = np.zeros((nchans,nsamps)) + norm.rvs(loc=0,scale=1*np.sqrt(nchans)*np.sqrt(width),size=(nchans,nsamps))
            Q[:,loc:loc+width] += snr_true*Lpol_true*np.cos(2*PA)
            #Q += (snr_true*norm.pdf(np.arange(nsamps),loc=loc,scale=width)/np.max(norm.pdf(np.arange(nsamps),loc=loc,scale=width)))*Lpol_true*np.cos(2*PA)
            U = np.zeros((nchans,nsamps)) + norm.rvs(loc=0,scale=1*np.sqrt(nchans)*np.sqrt(width),size=(nchans,nsamps))
            U[:,loc:loc+width] += snr_true*Lpol_true*np.sin(2*PA)
            #U += (snr_true*norm.pdf(np.arange(nsamps),loc=loc,scale=width)/np.max(norm.pdf(np.arange(nsamps),loc=loc,scale=width)))*Lpol_true*np.sin(2*PA)
            V = np.zeros((nchans,nsamps)) + norm.rvs(loc=0,scale=1*np.sqrt(nchans)*np.sqrt(width),size=(nchans,nsamps))
            V[:,loc:loc+width] += snr_true*Vpol_true
            #V += (snr_true*norm.pdf(np.arange(nsamps),loc=loc,scale=width)/np.max(norm.pdf(np.arange(nsamps),loc=loc,scale=width)))*Vpol_true
            tsamp = 32.7e-6
            n_t = 1
            n_f = 1
            freq_test = [np.linspace(1250,1520,6144)]*4
            NOFFDEF = 2000
            buff=0

            #input weights
            input_weights = np.zeros(nsamps)
            input_weights[loc:loc+width] = 1
            input_weights = input_weights/np.sum(input_weights)

            #weighted
            [(pol_f,pol_t,avg,sigma_frac,snr_frac),
            (L_f,L_t,avg_L,sigma_L,snr_L),
            (C_f_unbiased,C_t_unbiased,avg_C_abs,sigma_C_abs,snr_C),
            (C_f,C_t,avg_C,sigma_C,tmp),snr]=dsapol.get_pol_fraction(I,Q,U,V,width_native,tsamp,n_t,n_f,freq_test,
                                                                    int(NOFFDEF/n_t),plot=False,normalize=True,buff=buff,
                                                                    full=False,weighted=True,input_weights=input_weights)
            print("Testing pulse width=",width,"(native=",width_native,"),S/N=",snr_true,"...")
            #peak,timestart,timestop = dsapol.find_peak(I,width_native,tsamp,n_t,pre_calc_tf=False,buff=buff)
            #print(peak,timestart,timestop)
            print(avg,avg_L,avg_C,avg_C_abs)
            print(sigma_frac,sigma_L,sigma_C,sigma_C_abs)
            print(snr,snr_frac,snr_L,snr_C)
            assert(np.abs(avg - Tpol_true)<pol_tolerance)#*sigma_frac)
            assert(np.abs(avg_L - Lpol_true)<pol_tolerance)#*sigma_L)
            assert(np.abs(avg_C - Vpol_true) <pol_tolerance)#*sigma_C)
            assert(np.abs(avg_C_abs - Vpol_true) < pol_tolerance)#*sigma_C_abs)
            assert(np.abs(snr - snr_true) < snr_tolerance*snr_true)
            assert(np.abs(snr_frac - snr_true*Tpol_true) < snr_tolerance*snr_true*Tpol_true)
            assert(np.abs(snr_L - snr_true*Lpol_true) < snr_tolerance*snr_true*Lpol_true)
            assert(np.abs(snr_C - snr_true*Vpol_true) < snr_tolerance*snr_true*Vpol_true)
            assert(len(pol_f) == len(L_f) == len(C_f) == len(C_f_unbiased) == nchans)
            assert(len(pol_t) == len(L_t) == len(C_t) == len(C_t_unbiased) == nsamps)
            assert(np.all(pol_t[loc:loc+width]>=0))
            assert(np.all(L_t[loc:loc+width]>=0))
            assert(np.all(C_t[loc:loc+width]>=0))
            assert(np.all(pol_f[loc:loc+width]>=0))
            assert(np.all(L_f[loc:loc+width]>=0))
            assert(np.all(C_f[loc:loc+width]>=0))

            PA_f,PA_t,PA_f_errs,PA_t_errs,PA_avg,PA_err = dsapol.get_pol_angle(I,Q,U,V,width_native,tsamp,n_t,n_f,freq_test,
                                                                               n_off=int(NOFFDEF/n_t),normalize=True,buff=buff,
                                                                               weighted=True,input_weights=input_weights)
            assert(np.abs(PA_avg - PA)<pol_tolerance)#*PA_err)
            assert(len(PA_t) == len(PA_t_errs) == nsamps)
            assert(len(PA_f) == len(PA_f_errs) == nchans)
            assert(np.all((np.abs(PA_t-PA)<pa_tolerance)[loc:loc+width]))#PA_t_errs)[loc:loc+width])) 




    return
def test_pol_unweighted():

    #tolerances
    pol_tolerance = 0.05
    pa_tolerance = 5*np.pi/180 #rad
    snr_tolerance = 0.3 #fraction of snr_test
    Lpol_true = 0.5
    Vpol_true = 0.5
    Tpol_true = np.sqrt(Lpol_true**2 + Vpol_true**2)
    PA = 0

    nsamps = 5120
    nchans = 6144
    loc = int(nsamps//2)
    for width in [50]:
        width_native = int(np.ceil((32.7e-6/256e-6)*width)) #because internally converted to 32.7 microseconds
        for snr_true in [10000]:
            I = np.zeros((nchans,nsamps)) + norm.rvs(loc=0,scale=1*np.sqrt(nchans)*np.sqrt(width),size=(nchans,nsamps)) 
            I[:,loc:loc+width] += snr_true
            #I += (snr_true*norm.pdf(np.arange(nsamps),loc=loc,scale=width)/np.max(norm.pdf(np.arange(nsamps),loc=loc,scale=width)))
            Q = np.zeros((nchans,nsamps)) + norm.rvs(loc=0,scale=1*np.sqrt(nchans)*np.sqrt(width),size=(nchans,nsamps))
            Q[:,loc:loc+width] += snr_true*Lpol_true*np.cos(2*PA)
            #Q += (snr_true*norm.pdf(np.arange(nsamps),loc=loc,scale=width)/np.max(norm.pdf(np.arange(nsamps),loc=loc,scale=width)))*Lpol_true*np.cos(2*PA)
            U = np.zeros((nchans,nsamps)) + norm.rvs(loc=0,scale=1*np.sqrt(nchans)*np.sqrt(width),size=(nchans,nsamps))
            U[:,loc:loc+width] += snr_true*Lpol_true*np.sin(2*PA)
            #U += (snr_true*norm.pdf(np.arange(nsamps),loc=loc,scale=width)/np.max(norm.pdf(np.arange(nsamps),loc=loc,scale=width)))*Lpol_true*np.sin(2*PA)
            V = np.zeros((nchans,nsamps)) + norm.rvs(loc=0,scale=1*np.sqrt(nchans)*np.sqrt(width),size=(nchans,nsamps))
            V[:,loc:loc+width] += snr_true*Vpol_true
            #V += (snr_true*norm.pdf(np.arange(nsamps),loc=loc,scale=width)/np.max(norm.pdf(np.arange(nsamps),loc=loc,scale=width)))*Vpol_true
            tsamp = 32.7e-6
            n_t = 1
            n_f = 1
            freq_test = [np.linspace(1250,1520,6144)]*4
            NOFFDEF = 2000
            buff=0

            #unweighted
            [(pol_f,pol_t,avg,sigma_frac,snr_frac),
            (L_f,L_t,avg_L,sigma_L,snr_L),
            (C_f_unbiased,C_t_unbiased,avg_C_abs,sigma_C_abs,snr_C),
            (C_f,C_t,avg_C,sigma_C,tmp),snr]=dsapol.get_pol_fraction(I,Q,U,V,width_native,tsamp,n_t,n_f,freq_test,
                                                                    int(NOFFDEF/n_t),plot=False,normalize=True,buff=buff,
                                                                    full=False,weighted=False)
            print("Testing pulse width=",width,"(native=",width_native,"),S/N=",snr_true,"...")
            #peak,timestart,timestop = dsapol.find_peak(I,width_native,tsamp,n_t,pre_calc_tf=False,buff=buff)
            #print(peak,timestart,timestop)
            print(avg,avg_L,avg_C,avg_C_abs)
            print(sigma_frac,sigma_L,sigma_C,sigma_C_abs)
            print(snr,snr_frac,snr_L,snr_C)
            assert(np.abs(avg - Tpol_true)<pol_tolerance)#*sigma_frac)
            assert(np.abs(avg_L - Lpol_true)<pol_tolerance)#*sigma_L)
            assert(np.abs(avg_C - Vpol_true) <pol_tolerance)#*sigma_C)
            assert(np.abs(avg_C_abs - Vpol_true) < pol_tolerance)#*sigma_C_abs)
            assert(np.abs(snr - snr_true) < snr_tolerance*snr_true)
            assert(np.abs(snr_frac - snr_true*Tpol_true) < snr_tolerance*snr_true*Tpol_true)
            assert(np.abs(snr_L - snr_true*Lpol_true) < snr_tolerance*snr_true*Lpol_true)
            assert(np.abs(snr_C - snr_true*Vpol_true) < snr_tolerance*snr_true*Vpol_true)
            assert(len(pol_f) == len(L_f) == len(C_f) == len(C_f_unbiased) == nchans)
            assert(len(pol_t) == len(L_t) == len(C_t) == len(C_t_unbiased) == nsamps)
            assert(np.all(pol_t[loc:loc+width]>=0))
            assert(np.all(L_t[loc:loc+width]>=0))
            assert(np.all(C_t[loc:loc+width]>=0))
            assert(np.all(pol_f[loc:loc+width]>=0))
            assert(np.all(L_f[loc:loc+width]>=0))
            assert(np.all(C_f[loc:loc+width]>=0))


            PA_f,PA_t,PA_f_errs,PA_t_errs,PA_avg,PA_err = dsapol.get_pol_angle(I,Q,U,V,width_native,tsamp,n_t,n_f,freq_test,
                                                                               n_off=int(NOFFDEF/n_t),normalize=True,buff=buff,
                                                                               weighted=False)
            assert(np.abs(PA_avg - PA)<pol_tolerance)#*PA_err)
            assert(len(PA_t) == len(PA_t_errs) == nsamps)
            assert(len(PA_f) == len(PA_f_errs) == nchans)
            assert(np.all((np.abs(PA_t-PA)<pa_tolerance)[loc:loc+width]))#PA_t_errs)[loc:loc+width])) 


    return


if __name__=="__main__":
    pytest.main()
