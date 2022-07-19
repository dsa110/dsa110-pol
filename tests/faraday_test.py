from dsapol import dsapol
import numpy as np
from scipy.stats import norm

#Single bin peak
def test_RM_test_1():
    print("Testing RM cal with single bin...")
    #Try with multiple noise levels, should get RM estimate within error bounds
    trial_sigmas = 1/(10**np.arange(1,5))
    print(trial_sigmas)
    for sigma_noise in trial_sigmas:
        #Create test signal
        nchans = 100
        nsamples = 200
        tsamp = 1
        RM_true = 100
        n_t = 1
        n_f = 1
        deg = 10
        freq_test = [np.linspace(1300,1500,nchans)]*4
        wav_test = []
        for f in freq_test:
            wav_test.append((3e8)/(np.array(f)*(1e6)))


        wavall = np.transpose(np.tile(((3e8)/(freq_test[0]*1e6)),(nsamples,1)))

        #estimated FWHM of RM dispersion function
        err_max = np.pi*sigma_noise/np.abs(wav_test[0][-1]**2 - wav_test[0][0]**2)
        print("Max error",err_max)



        noiseI = np.random.normal(0,sigma_noise,(nchans,nsamples))
        noiseQ = np.random.normal(0,sigma_noise,(nchans,nsamples))
        noiseU = np.random.normal(0,sigma_noise,(nchans,nsamples))
        noiseV = np.random.normal(0,sigma_noise,(nchans,nsamples))

        I = noiseI#np.random.normal(0,0.25,(nchans,nsamples))
        Q = noiseQ#np.random.normal(0,0.25,(nchans,nsamples))
        U = noiseU#np.random.normal(0,0.25,(nchans,nsamples))
        V = noiseV#np.random.normal(0,0.25,(nchans,nsamples))
        w=1
        timestart = 50-w//2
        timestop= 50+w//2 + 1
        I[:,timestart:timestop] += np.ones((nchans,1))
        Q[:,timestart:timestop] += np.ones((nchans,1))

        width_native = w*tsamp/(256e-6)

        P = Q + 1j*U
        RM_true = 100
        c = 3e8


        Q_obs = np.real(P*np.exp(1j*2*RM_true*(wavall**2)))
        U_obs = np.imag(P*np.exp(1j*2*RM_true*(wavall**2)))
        I_obs = I
        V_obs = V

        #(I_obs,Q_obs,U_obs,V_obs) =dsapol.calibrate_RM(I,Q,U,V,-RM_true,phi,freq_test,stokes=True)

        #dsapol.plot_spectra_2D(I,Q,U,V,width_native,tsamp,n_t,n_f,freq_test,show=True,lim=1)
        #dsapol.plot_spectra_2D(I_obs,Q_obs,U_obs,V_obs,width_native,tsamp,n_t,n_f,freq_test,show=True,lim=1)

        (I_obs_f,Q_obs_f,U_obs_f,V_obs_f) = dsapol.get_stokes_vs_freq(I_obs,Q_obs,U_obs,V_obs,width_native,tsamp,n_f,freq_test,plot=False)

        trial_RM = np.linspace(RM_true-250,RM_true+250,1000)#np.linspace(RM_true-50,RM_true+50,500)
        DEL_RM = np.abs(trial_RM[1] - trial_RM[0])
        trial_phi = [0]

        (RM,phi,snrs,RMerr) = dsapol.faradaycal(I_obs_f,Q_obs_f,U_obs_f,V_obs_f,freq_test,trial_RM,trial_phi,plot=False,show=False,fit_window=50,err=True)
        print("test 1: " + str((RM,RM-RM_true,DEL_RM,err_max)))

        assert((np.abs(RM - RM_true) <= DEL_RM) or (np.abs(RM - RM_true) <= err_max) )
        assert(RMerr != None)
        assert(RM == trial_RM[np.argmax(snrs)])


        #fine RM 
        trial_RM_zoom = np.linspace(RM_true-30,RM_true+30,1000)
        DEL_RM = np.abs(trial_RM_zoom[1] - trial_RM_zoom[0])
        (RM,phi,snrs,RMerr) = dsapol.faradaycal_SNR(I_obs,Q_obs,U_obs,V_obs,freq_test,trial_RM_zoom,trial_phi,width_native,tsamp,plot=False,show=False,err=True)
        print("test 2: " + str((RM,RM-RM_true,DEL_RM,err_max)))
        assert((np.abs(RM - RM_true) <= DEL_RM) or (np.abs(RM - RM_true) <= err_max) )
        assert(RMerr != None)
        assert(RM == trial_RM_zoom[np.argmax(snrs)])

        #both
        trial_RM = np.linspace(-1e3,1e3,1000)
        DEL_RM = np.abs(trial_RM[1] - trial_RM[0])
        (RM,phi,snrs,RMerr) = dsapol.faradaycal_full(I_obs,Q_obs,U_obs,V_obs,freq_test,trial_RM,trial_phi,width_native,tsamp,1000,zoom_window=30,plot=False,show=False,fit_window=75)
        print("test 3: " + str((RM,RM-RM_true,DEL_RM,err_max)))
        assert((np.abs(RM - RM_true) <= DEL_RM) or (np.abs(RM - RM_true) <= err_max) )
        assert(RMerr != None)
        #assert(RM == trial_RM[np.argmax(snrs)])


        #calibrate
        (I_cal,Q_cal,U_cal,V_cal) =dsapol.calibrate_RM(I_obs,Q_obs,U_obs,V_obs,RM,phi,freq_test,stokes=True)
        #dsapol.plot_spectra_2D(I_cal,Q_cal,U_cal,V_cal,width_native,tsamp,n_t,n_f,freq_test,show=True,lim=1)

        assert(np.all(np.abs(I_cal - I) < 10*np.sqrt(timestop-timestart)*np.sqrt(nchans)*sigma_noise))
        assert(np.all(np.abs(Q_cal - Q) < 10*np.sqrt(timestop-timestart)*np.sqrt(nchans)*sigma_noise))
        assert(np.all(np.abs(U_cal - U) < 10*np.sqrt(timestop-timestart)*np.sqrt(nchans)*sigma_noise))
        assert(np.all(np.abs(V_cal - V) < 10*np.sqrt(timestop-timestart)*np.sqrt(nchans)*sigma_noise))
        print("Single bin RM cal successful!")
        print("Done!")
    return 

#finite width flat peak
def test_RM_test_2():
    print("Testing RM cal with square pulse...")
    #width > 1
    #Try with multiple noise levels, should get RM estimate within error bounds
    trial_sigmas = 1/(10**np.arange(1,5))
    print(trial_sigmas)
    for sigma_noise in trial_sigmas:
        #Create test signal
        nchans = 100
        nsamples = 200
        tsamp=1
        RM_true = 100
        n_t = 1
        n_f = 1
        deg = 10
        freq_test = [np.linspace(1300,1500,nchans)]*4
        wav_test = []
        for f in freq_test:
            wav_test.append((3e8)/(np.array(f)*(1e6)))


        wavall = np.transpose(np.tile(((3e8)/(freq_test[0]*1e6)),(nsamples,1)))


        #estimated FWHM of RM dispersion function
        err_max = np.pi*sigma_noise/np.abs(wav_test[0][-1]**2 - wav_test[0][0]**2)
        print("Max error",err_max)



        noiseI = np.random.normal(0,sigma_noise,(nchans,nsamples))
        noiseQ = np.random.normal(0,sigma_noise,(nchans,nsamples))
        noiseU = np.random.normal(0,sigma_noise,(nchans,nsamples))
        noiseV = np.random.normal(0,sigma_noise,(nchans,nsamples))

        I = noiseI#np.random.normal(0,0.25,(nchans,nsamples))
        Q = noiseQ#np.random.normal(0,0.25,(nchans,nsamples))
        U = noiseU#np.random.normal(0,0.25,(nchans,nsamples))
        V = noiseV#np.random.normal(0,0.25,(nchans,nsamples))
        w=10
        timestart = 50-w//2
        timestop= 50+w//2 
        print(timestop-timestart)
        I[:,timestart:timestop] += np.ones((nchans,w))
        Q[:,timestart:timestop] += np.ones((nchans,w))

        width_native = w*tsamp/(256e-6)

        P = Q + 1j*U
        RM_true = 100
        c = 3e8


        Q_obs = np.real(P*np.exp(1j*2*RM_true*(wavall**2)))
        U_obs = np.imag(P*np.exp(1j*2*RM_true*(wavall**2)))
        I_obs = I
        V_obs = V

        #(I_obs,Q_obs,U_obs,V_obs) =dsapol.calibrate_RM(I,Q,U,V,-RM_true,phi,freq_test,stokes=True)

        #dsapol.plot_spectra_2D(I,Q,U,V,width_native,tsamp,n_t,n_f,freq_test,show=True,lim=1)
        #dsapol.plot_spectra_2D(I_obs,Q_obs,U_obs,V_obs,width_native,tsamp,n_t,n_f,freq_test,show=True,lim=1)

        (I_obs_f,Q_obs_f,U_obs_f,V_obs_f) = dsapol.get_stokes_vs_freq(I_obs,Q_obs,U_obs,V_obs,width_native,tsamp,n_f,freq_test,plot=False)

        trial_RM = np.linspace(RM_true-250,RM_true+250,1000)#np.linspace(RM_true-50,RM_true+50,500)
        DEL_RM = np.abs(trial_RM[1] - trial_RM[0])
        trial_phi = [0]

        (RM,phi,snrs,RMerr) = dsapol.faradaycal(I_obs_f,Q_obs_f,U_obs_f,V_obs_f,freq_test,trial_RM,trial_phi,plot=False,show=False,fit_window=50,err=True)
        print("test 1: " + str((RM,RM-RM_true,DEL_RM,err_max)))

        assert((np.abs(RM - RM_true) <= DEL_RM) or (np.abs(RM - RM_true) <= err_max) )
        assert(RMerr != None)
        assert(RM == trial_RM[np.argmax(snrs)])


        #fine RM 
        trial_RM_zoom = np.linspace(RM_true-30,RM_true+30,1000)
        DEL_RM = np.abs(trial_RM_zoom[1] - trial_RM_zoom[0])
        (RM,phi,snrs,RMerr) = dsapol.faradaycal_SNR(I_obs,Q_obs,U_obs,V_obs,freq_test,trial_RM_zoom,trial_phi,width_native,tsamp,plot=False,show=False,err=True)
        print("test 2: " + str((RM,RM-RM_true,DEL_RM,err_max)))
        assert((np.abs(RM - RM_true) <= DEL_RM) or (np.abs(RM - RM_true) <= err_max) )
        assert(RMerr != None)
        assert(RM == trial_RM_zoom[np.argmax(snrs)])

        #both
        trial_RM = np.linspace(-1e3,1e3,1000)
        DEL_RM = np.abs(trial_RM[1] - trial_RM[0])
        (RM,phi,snrs,RMerr) = dsapol.faradaycal_full(I_obs,Q_obs,U_obs,V_obs,freq_test,trial_RM,trial_phi,width_native,tsamp,1000,zoom_window=30,plot=False,show=False,fit_window=75)
        print("test 3: " + str((RM,RM-RM_true,DEL_RM,err_max)))
        assert((np.abs(RM - RM_true) <= DEL_RM) or (np.abs(RM - RM_true) <= err_max) )
        assert(RMerr != None)
        #assert(RM == trial_RM[np.argmax(snrs)])


        #calibrate
        (I_cal,Q_cal,U_cal,V_cal) =dsapol.calibrate_RM(I_obs,Q_obs,U_obs,V_obs,RM,phi,freq_test,stokes=True)
        #dsapol.plot_spectra_2D(I_cal,Q_cal,U_cal,V_cal,width_native,tsamp,n_t,n_f,freq_test,show=True,lim=1)

        assert(np.all(np.abs(I_cal - I) < 10*np.sqrt(timestop-timestart)*np.sqrt(nchans)*sigma_noise))
        assert(np.all(np.abs(Q_cal - Q) < 10*np.sqrt(timestop-timestart)*np.sqrt(nchans)*sigma_noise))
        assert(np.all(np.abs(U_cal - U) < 10*np.sqrt(timestop-timestart)*np.sqrt(nchans)*sigma_noise))
        assert(np.all(np.abs(V_cal - V) < 10*np.sqrt(timestop-timestart)*np.sqrt(nchans)*sigma_noise))
        print("Square pulse RM cal successful!")
        print("Done!")
    return


#continuous signal
def test_RM_test_3():
    print("Testing RM cal with continuous signal...")
    #continuous signal (e.g. calibrator)
    #Try with multiple noise levels, should get RM estimate within error bounds
    trial_sigmas = 1/(10**np.arange(1,5))
    print(trial_sigmas)
    for sigma_noise in trial_sigmas:
        #Create test signal
        nchans = 100
        nsamples = 200
        RM_true = 100
        tsamp = 1
        n_t = 1
        n_f = 1
        deg = 10
        freq_test = [np.linspace(1300,1500,nchans)]*4
        wav_test = []
        for f in freq_test:
            wav_test.append((3e8)/(np.array(f)*(1e6)))


        wavall = np.transpose(np.tile(((3e8)/(freq_test[0]*1e6)),(nsamples,1)))
        #print(wav_test)
        #I = np.ones((nchans,nsamples)) + np.random.normal(0,0.25,(nchans,nsamples))
        #Q = np.ones((nchans,nsamples))+ np.random.normal(0,0.25,(nchans,nsamples))
        #U = np.zeros((nchans,nsamples)) + np.random.normal(0,0.25,(nchans,nsamples))
        #V = np.zeros((nchans,nsamples)) + np.random.normal(0,0.25,(nchans,nsamples))

        #sigma_noise = 1e-2#0.25


        #estimated FWHM of RM dispersion function
        err_max = np.pi*sigma_noise/np.abs(wav_test[0][-1]**2 - wav_test[0][0]**2)
        print("Max error",err_max)



        noiseI = np.random.normal(0,sigma_noise,(nchans,nsamples))
        noiseQ = np.random.normal(0,sigma_noise,(nchans,nsamples))
        noiseU = np.random.normal(0,sigma_noise,(nchans,nsamples))
        noiseV = np.random.normal(0,sigma_noise,(nchans,nsamples))

        I = noiseI#np.random.normal(0,0.25,(nchans,nsamples))
        Q = noiseQ#np.random.normal(0,0.25,(nchans,nsamples))
        U = noiseU#np.random.normal(0,0.25,(nchans,nsamples))
        V = noiseV#np.random.normal(0,0.25,(nchans,nsamples))
        w=nsamples
        timestart = 0#50-w//2
        timestop= nsamples#50+w//2 + 1
        #I[:,timestart:timestop] += np.ones((nchans,1))
        #Q[:,timestart:timestop] += np.ones((nchans,1))
        I += np.ones((nchans,nsamples))
        Q += np.ones((nchans,nsamples))

        #I = Q = np.ones((nchans,nsamples))


        width_native = -1#1/(256e-6)

        P = Q + 1j*U
        RM_true = 100
        c = 3e8


        Q_obs = np.real(P*np.exp(1j*2*RM_true*(wavall**2)))
        U_obs = np.imag(P*np.exp(1j*2*RM_true*(wavall**2)))
        I_obs = I
        V_obs = V

        #(I_obs,Q_obs,U_obs,V_obs) =dsapol.calibrate_RM(I,Q,U,V,-RM_true,phi,freq_test,stokes=True)

        #dsapol.plot_spectra_2D(I,Q,U,V,width_native,tsamp,n_t,n_f,freq_test,show=True,lim=1)
        #dsapol.plot_spectra_2D(I_obs,Q_obs,U_obs,V_obs,width_native,tsamp,n_t,n_f,freq_test,show=True,lim=1)

        (I_obs_f,Q_obs_f,U_obs_f,V_obs_f) = dsapol.get_stokes_vs_freq(I_obs,Q_obs,U_obs,V_obs,width_native,tsamp,n_f,freq_test,plot=False,show=False)

        trial_RM = np.linspace(RM_true-250,RM_true+250,1000)#np.linspace(RM_true-50,RM_true+50,500)
        DEL_RM = np.abs(trial_RM[1] - trial_RM[0])
        trial_phi = [0]

        (RM,phi,snrs,RMerr) = dsapol.faradaycal(I_obs_f,Q_obs_f,U_obs_f,V_obs_f,freq_test,trial_RM,trial_phi,plot=False,show=False,fit_window=50,err=True)
        print("test 1: " + str((RM,RM-RM_true,DEL_RM,err_max)))

        assert((np.abs(RM - RM_true) <= DEL_RM) or (np.abs(RM - RM_true) <= err_max) )
        assert(RMerr != None)
        assert(RM == trial_RM[np.argmax(snrs)])
        #**Note: can't do faradaycal_SNR with continuous signal b/c can't get SNR estimate without finite width

        #calibrate
        (I_cal,Q_cal,U_cal,V_cal) =dsapol.calibrate_RM(I_obs,Q_obs,U_obs,V_obs,RM,phi,freq_test,stokes=True)
        #dsapol.plot_spectra_2D(I_cal,Q_cal,U_cal,V_cal,width_native,tsamp,n_t,n_f,freq_test,show=True,lim=1)

        assert(np.all(np.abs(I_cal - I) < 10*np.sqrt(timestop-timestart)*np.sqrt(nchans)*sigma_noise))
        assert(np.all(np.abs(Q_cal - Q) < 10*np.sqrt(timestop-timestart)*np.sqrt(nchans)*sigma_noise))
        assert(np.all(np.abs(U_cal - U) < 10*np.sqrt(timestop-timestart)*np.sqrt(nchans)*sigma_noise))
        assert(np.all(np.abs(V_cal - V) < 10*np.sqrt(timestop-timestart)*np.sqrt(nchans)*sigma_noise))
        print("Continuous signal RM cal successful!")
        print("Done!")
    return


#realistic gaussian signal
def test_RM_test_4():
    print("Testing RM cal with gaussian pulse...")
    #Realistic Pulse
    #width > 1
    #Try with multiple noise levels, should get RM estimate within error bounds
    trial_sigmas = 1/(10**np.arange(1,5))
    print(trial_sigmas)
    for sigma_noise in trial_sigmas:
        #Create test signal
        nchans = 6144
        nsamples = 5120
        tsamp = 0.000131072
        RM_true = 100
        n_t = 1
        n_f = 32
        deg = 10
        freq_test = [np.linspace(1300,1500,nchans)]*4
        wav_test = []
        for f in freq_test:
            wav_test.append((3e8)/(np.array(f)*(1e6)))


        wavall = np.transpose(np.tile(((3e8)/(freq_test[0]*1e6)),(nsamples,1)))
        #print(wav_test)
        #I = np.ones((nchans,nsamples)) + np.random.normal(0,0.25,(nchans,nsamples))
        #Q = np.ones((nchans,nsamples))+ np.random.normal(0,0.25,(nchans,nsamples))
        #U = np.zeros((nchans,nsamples)) + np.random.normal(0,0.25,(nchans,nsamples))
        #V = np.zeros((nchans,nsamples)) + np.random.normal(0,0.25,(nchans,nsamples))

        #sigma_noise = 1e-2#0.25


        #estimated FWHM of RM dispersion function
        #err_max = np.pi*sigma_noise/np.abs(wav_test[0][-1]**2 - wav_test[0][0]**2)
        #print("Max error",err_max)



        noiseI = np.random.normal(0,sigma_noise,(nchans,nsamples))
        noiseQ = np.random.normal(0,sigma_noise,(nchans,nsamples))
        noiseU = np.random.normal(0,sigma_noise,(nchans,nsamples))
        noiseV = np.random.normal(0,sigma_noise,(nchans,nsamples))

        I = noiseI#np.random.normal(0,0.25,(nchans,nsamples))
        Q = noiseQ#np.random.normal(0,0.25,(nchans,nsamples))
        U = noiseU#np.random.normal(0,0.25,(nchans,nsamples))
        V = noiseV#np.random.normal(0,0.25,(nchans,nsamples))
        gw = 10
        w=5*gw
        mid = 3100
        timestart = int(mid-w//2)
        timestop= int(mid+w//2 )
        print(timestop-timestart,timestart,timestop)
        pulse = norm.pdf(np.arange(nsamples),mid,gw)
        pulse = pulse/np.mean(pulse[timestart:timestop])
        I += np.array([pulse]*nchans)#np.ones((nchans,w))
        Q += np.array([pulse]*nchans)#np.ones((nchans,w))

        width_native = w*tsamp/(256e-6)

        P = Q + 1j*U
        RM_true = 100
        c = 3e8


        Q_obs = np.real(P*np.exp(1j*2*RM_true*(wavall**2)))
        U_obs = np.imag(P*np.exp(1j*2*RM_true*(wavall**2)))
        I_obs = I
        V_obs = V


        I_obs = dsapol.avg_freq(I_obs,n_f)
        Q_obs = dsapol.avg_freq(Q_obs,n_f)
        U_obs = dsapol.avg_freq(U_obs,n_f)
        V_obs = dsapol.avg_freq(V_obs,n_f)



        freq_test = [np.linspace(1300,1500,nchans//n_f)]*4
        wav_test = []
        for f in freq_test:
            wav_test.append((3e8)/(np.array(f)*(1e6)))


        wavall = np.transpose(np.tile(((3e8)/(freq_test[0]*1e6)),(nsamples,1)))

        #estimated FWHM of RM dispersion function
        err_max = np.pi*sigma_noise/np.abs(wav_test[0][-1]**2 - wav_test[0][0]**2)
        print("Max error",err_max)

        #(I_obs,Q_obs,U_obs,V_obs) =dsapol.calibrate_RM(I,Q,U,V,-RM_true,phi,freq_test,stokes=True)

        #dsapol.plot_spectra_2D(I,Q,U,V,width_native,tsamp,n_t,n_f,freq_test,show=True,lim=1)
        #dsapol.plot_spectra_2D(I_obs,Q_obs,U_obs,V_obs,width_native,tsamp,n_t,n_f,freq_test,show=True,lim=1)

        (I_obs_f,Q_obs_f,U_obs_f,V_obs_f) = dsapol.get_stokes_vs_freq(I_obs,Q_obs,U_obs,V_obs,width_native,tsamp,n_f,freq_test,plot=False)

        trial_RM = np.linspace(-1e6,1e6,int(1e6))#np.linspace(RM_true-250,RM_true+250,1000)#np.linspace(RM_true-50,RM_true+50,500)
        DEL_RM = np.abs(trial_RM[1] - trial_RM[0])
        trial_phi = [0]

        (RM,phi,snrs,RMerr) = dsapol.faradaycal(I_obs_f,Q_obs_f,U_obs_f,V_obs_f,freq_test,trial_RM,trial_phi,plot=False,show=False,fit_window=50,err=True)
        print("test 1: " + str((RM,RM-RM_true,DEL_RM,err_max)))

        assert((np.abs(RM - RM_true) <= DEL_RM) or (np.abs(RM - RM_true) <= err_max) )
        assert(RMerr != None)
        assert(RM == trial_RM[np.argmax(snrs)])


        #fine RM 
        trial_RM_zoom = np.linspace(RM_true-500,RM_true+500,1000)#np.linspace(RM_true-30,RM_true+30,1000)
        DEL_RM = np.abs(trial_RM_zoom[1] - trial_RM_zoom[0])
        (RM,phi,snrs,RMerr) = dsapol.faradaycal_SNR(I_obs,Q_obs,U_obs,V_obs,freq_test,trial_RM_zoom,trial_phi,width_native,tsamp,plot=False,show=False,err=True)
        print("test 2: " + str((RM,RM-RM_true,DEL_RM,err_max)))
        assert((np.abs(RM - RM_true) <= DEL_RM) or (np.abs(RM - RM_true) <= err_max) )
        assert(RMerr != None)
        assert(RM == trial_RM_zoom[np.argmax(snrs)])

        #both
        trial_RM = np.linspace(-1e3,1e3,1000)
        DEL_RM = np.abs(trial_RM[1] - trial_RM[0])
        (RM,phi,snrs,RMerr) = dsapol.faradaycal_full(I_obs,Q_obs,U_obs,V_obs,freq_test,trial_RM,trial_phi,width_native,tsamp,1000,zoom_window=30,plot=False,show=False,fit_window=75)
        print("test 3: " + str((RM,RM-RM_true,DEL_RM,err_max)))
        assert((np.abs(RM - RM_true) <= DEL_RM) or (np.abs(RM - RM_true) <= err_max) )
        assert(RMerr != None)
        #assert(RM == trial_RM[np.argmax(snrs)])


        #calibrate
        (I_cal,Q_cal,U_cal,V_cal) =dsapol.calibrate_RM(I_obs,Q_obs,U_obs,V_obs,RM,phi,freq_test,stokes=True)
        #dsapol.plot_spectra_2D(I_cal,Q_cal,U_cal,V_cal,width_native,tsamp,n_t,n_f,freq_test,show=True,lim=1)

        I_avg = dsapol.avg_freq(I,n_f)
        Q_avg = dsapol.avg_freq(Q,n_f)
        U_avg = dsapol.avg_freq(U,n_f)
        V_avg = dsapol.avg_freq(V,n_f)

        assert(np.all(np.abs(I_cal - I_avg) < 10*np.sqrt(timestop-timestart)*np.sqrt(nchans)*sigma_noise))
        assert(np.all(np.abs(Q_cal - Q_avg) < 10*np.sqrt(timestop-timestart)*np.sqrt(nchans)*sigma_noise))
        assert(np.all(np.abs(U_cal - U_avg) < 10*np.sqrt(timestop-timestart)*np.sqrt(nchans)*sigma_noise))
        assert(np.all(np.abs(V_cal - V_avg) < 10*np.sqrt(timestop-timestart)*np.sqrt(nchans)*sigma_noise))
        print("Gaussian pulse RM cal successful!")
        print("Done!")
    return


#realistic sampled continuous signal
def test_RM_test_5():
    print("Testing RM cal with realistic sampled continuous signal...")
    #Realistic calibrator
    #Try with multiple noise levels, should get RM estimate within error bounds
    trial_sigmas = 1/(10**np.arange(1,5))
    print(trial_sigmas)
    for sigma_noise in trial_sigmas:
        #Create test signal
        nchans = 6144
        nsamples = 5120
        tsamp = 0.000131072
        RM_true = 100
        n_t = 1
        n_f = 32
        deg = 10
        freq_test = [np.linspace(1300,1500,nchans)]*4
        wav_test = []
        for f in freq_test:
            wav_test.append((3e8)/(np.array(f)*(1e6)))


        wavall = np.transpose(np.tile(((3e8)/(freq_test[0]*1e6)),(nsamples,1)))
        #print(wav_test)
        #I = np.ones((nchans,nsamples)) + np.random.normal(0,0.25,(nchans,nsamples))
        #Q = np.ones((nchans,nsamples))+ np.random.normal(0,0.25,(nchans,nsamples))
        #U = np.zeros((nchans,nsamples)) + np.random.normal(0,0.25,(nchans,nsamples))
        #V = np.zeros((nchans,nsamples)) + np.random.normal(0,0.25,(nchans,nsamples))

        #sigma_noise = 1e-2#0.25


        #estimated FWHM of RM dispersion function
        #err_max = np.pi*sigma_noise/np.abs(wav_test[0][-1]**2 - wav_test[0][0]**2)
        #print("Max error",err_max)



        noiseI = np.random.normal(0,sigma_noise,(nchans,nsamples))
        noiseQ = np.random.normal(0,sigma_noise,(nchans,nsamples))
        noiseU = np.random.normal(0,sigma_noise,(nchans,nsamples))
        noiseV = np.random.normal(0,sigma_noise,(nchans,nsamples))

        I = noiseI#np.random.normal(0,0.25,(nchans,nsamples))
        Q = noiseQ#np.random.normal(0,0.25,(nchans,nsamples))
        U = noiseU#np.random.normal(0,0.25,(nchans,nsamples))
        V = noiseV#np.random.normal(0,0.25,(nchans,nsamples))
        w=nsamples
        timestart = 0#50-w//2
        timestop= nsamples#50+w//2 + 1
        #I[:,timestart:timestop] += np.ones((nchans,1))
        #Q[:,timestart:timestop] += np.ones((nchans,1))
        I += np.ones((nchans,nsamples))
        Q += np.ones((nchans,nsamples))

        #I = Q = np.ones((nchans,nsamples))


        width_native = -1#1/(256e-6)

        P = Q + 1j*U
        RM_true = 100
        c = 3e8


        Q_obs = np.real(P*np.exp(1j*2*RM_true*(wavall**2)))
        U_obs = np.imag(P*np.exp(1j*2*RM_true*(wavall**2)))
        I_obs = I
        V_obs = V


        I_obs = dsapol.avg_freq(I_obs,n_f)
        Q_obs = dsapol.avg_freq(Q_obs,n_f)
        U_obs = dsapol.avg_freq(U_obs,n_f)
        V_obs = dsapol.avg_freq(V_obs,n_f)



        freq_test = [np.linspace(1300,1500,nchans//n_f)]*4
        wav_test = []
        for f in freq_test:
            wav_test.append((3e8)/(np.array(f)*(1e6)))


        wavall = np.transpose(np.tile(((3e8)/(freq_test[0]*1e6)),(nsamples,1)))

        #estimated FWHM of RM dispersion function
        err_max = np.pi*sigma_noise/np.abs(wav_test[0][-1]**2 - wav_test[0][0]**2)
        print("Max error",err_max)

        #(I_obs,Q_obs,U_obs,V_obs) =dsapol.calibrate_RM(I,Q,U,V,-RM_true,phi,freq_test,stokes=True)

        #dsapol.plot_spectra_2D(I,Q,U,V,width_native,tsamp,n_t,n_f,freq_test,show=True,lim=1)
        #dsapol.plot_spectra_2D(I_obs,Q_obs,U_obs,V_obs,width_native,tsamp,n_t,n_f,freq_test,show=True,lim=1)

        (I_obs_f,Q_obs_f,U_obs_f,V_obs_f) = dsapol.get_stokes_vs_freq(I_obs,Q_obs,U_obs,V_obs,width_native,tsamp,n_f,freq_test,plot=False)

        trial_RM = np.linspace(-1e6,1e6,int(1e6))#np.linspace(RM_true-250,RM_true+250,1000)#np.linspace(RM_true-50,RM_true+50,500)
        DEL_RM = np.abs(trial_RM[1] - trial_RM[0])
        trial_phi = [0]

        (RM,phi,snrs,RMerr) = dsapol.faradaycal(I_obs_f,Q_obs_f,U_obs_f,V_obs_f,freq_test,trial_RM,trial_phi,plot=False,show=False,fit_window=50,err=True)
        print("test 1: " + str((RM,RM-RM_true,DEL_RM,err_max)))

        assert((np.abs(RM - RM_true) <= DEL_RM) or (np.abs(RM - RM_true) <= err_max) )
        assert(RMerr != None)
        assert(RM == trial_RM[np.argmax(snrs)])

        #calibrate
        (I_cal,Q_cal,U_cal,V_cal) =dsapol.calibrate_RM(I_obs,Q_obs,U_obs,V_obs,RM,phi,freq_test,stokes=True)
        #dsapol.plot_spectra_2D(I_cal,Q_cal,U_cal,V_cal,width_native,tsamp,n_t,n_f,freq_test,show=True,lim=1)

        I_avg = dsapol.avg_freq(I,n_f)
        Q_avg = dsapol.avg_freq(Q,n_f)
        U_avg = dsapol.avg_freq(U,n_f)
        V_avg = dsapol.avg_freq(V,n_f)

        assert(np.all(np.abs(I_cal - I_avg) < 10*np.sqrt(timestop-timestart)*np.sqrt(nchans)*sigma_noise))
        assert(np.all(np.abs(Q_cal - Q_avg) < 10*np.sqrt(timestop-timestart)*np.sqrt(nchans)*sigma_noise))
        assert(np.all(np.abs(U_cal - U_avg) < 10*np.sqrt(timestop-timestart)*np.sqrt(nchans)*sigma_noise))
        assert(np.all(np.abs(V_cal - V_avg) < 10*np.sqrt(timestop-timestart)*np.sqrt(nchans)*sigma_noise))
        print("Realistic continuous RM cal successful!")
        print("Done!")
    return

def main():
    print("Running Faraday RM Calibration Tests...")
    RM_test_1()
    RM_test_2()
    RM_test_3()
    RM_test_4()
    RM_test_5()
    print("All Tests Passed!")

if __name__ == "__main__":
    main()
