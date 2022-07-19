from dsapol import dsapol
import numpy as np
from scipy.stats import norm


#Ideal, constant in time
def test_freq_ideal_test_1():
    print("Testing frequency spec with ideal constant signal...")
    # Ideal, Constant in frequency, constant in time, no normalization
    nchans = 100
    nsamples = 200
    tsamp = 1
    w = 1
    width_native = -1#w*tsamp/(256e-6)
    n_t = 1
    n_f = 1
    n_off = -1
    freq_test = [np.linspace(1300,1500,nchans)]*4
    I = Q = U = V = np.ones((nchans,nsamples))


    (I_f,Q_f,U_f,V_f) = dsapol.get_stokes_vs_freq(I,Q,U,V,width_native,tsamp,n_f,freq_test,n_off=n_off,plot=False,show=False,normalize=False)

    assert(len(I_f) == len(Q_f) == len(U_f) == len(V_f) == nchans)
    assert(np.all(I_f == 1))
    assert(np.all(Q_f == 1))
    assert(np.all(U_f == 1))
    assert(np.all(V_f == 1))
    print("Frequency spectra contant signal success!")
    print("Done!")
    return

#Ideal pulse
def test_freq_ideal_test_2():
    print("Testing frequency spec with ideal pulse...")
    #Ideal, Constant in frequency, pulse in time, no normalization
    nchans = 100
    nsamples = 200
    tsamp = 1
    w = 10
    width_native = w*tsamp/(256e-6)
    n_t = 1
    n_f = 1
    n_off = 30
    freq_test = [np.linspace(1300,1500,nchans)]*4

    I = Q = U = V = np.zeros((nchans,nsamples))
    timestart = 50 - w//2
    timestop = 50 + w//2
    I[:,timestart:timestop] = np.ones((nchans,w))
    Q[:,timestart:timestop] = np.ones((nchans,w))
    U[:,timestart:timestop] = np.ones((nchans,w))
    V[:,timestart:timestop] = np.ones((nchans,w))


    (I_f,Q_f,U_f,V_f) = dsapol.get_stokes_vs_freq(I,Q,U,V,width_native,tsamp,n_f,freq_test,n_off=n_off,plot=False,show=False,normalize=False)
    assert(len(I_f) == len(Q_f) == len(U_f) == len(V_f) == nchans)
    assert(np.all(I_f == 1))
    assert(np.all(Q_f == 1))
    assert(np.all(U_f == 1))
    assert(np.all(V_f == 1))
    print("Frequency spectra ideal pulse success!")
    print("Done!")
    return

#Ideal time variable
def test_freq_ideal_test_3():
    print("Testing frequency spec with ideal time varying signal...")
    #Ideal, constant in frequency, variation in time, no normalization
    nchans = 100
    nsamples = 200
    tsamp = 1
    w = 10
    width_native= -1#w*tsamp/(256e-6)
    n_t = 1
    n_f = 1
    n_off = -1
    freq_test = [np.linspace(1300,1500,nchans)]*4

    I = Q = U = V = np.zeros((nchans,nsamples))
    I = Q = U = V = np.array([np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10))]*nchans)
    Q = 0.75*Q
    U = 0.5*U
    V = 0.25*V

    (I_f,Q_f,U_f,V_f) = dsapol.get_stokes_vs_freq(I,Q,U,V,width_native,tsamp,n_f,freq_test,n_off=n_off,plot=False,show=False,normalize=False)
    assert(len(I_f) == len(Q_f) == len(U_f) == len(V_f) == nchans)
    assert(np.all(np.abs(I_f - np.mean(np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10)))) < 1e-10))
    assert(np.all(np.abs(Q_f - np.mean(0.75*np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10)))) < 1e-10))
    assert(np.all(np.abs(U_f - np.mean(0.5*np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10)))) < 1e-10))
    assert(np.all(np.abs(V_f - np.mean(0.25*np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10)))) < 1e-10))
    print("Frequency spectra time varying signal success!")
    print("Done!")
    return

#Ideal frequency variable, constant in time
def test_freq_ideal_test_4():
    print("Testing frequency spec with ideal frequency varying signal...")
    # Ideal, variation in frequency, constant in time, no normalization
    nchans = 100
    nsamples = 200
    tsamp = 1
    w = 1
    width_native = -1#w*tsamp/(256e-6)
    n_t = 1
    n_f = 1
    n_off = -1
    freq_test = [np.linspace(1300,1500,nchans)]*4

    I = Q = U = V = np.transpose(np.array([np.arange(0,nchans)]*nsamples))#np.ones((nchans,nsamples))


    (I_f,Q_f,U_f,V_f) = dsapol.get_stokes_vs_freq(I,Q,U,V,width_native,tsamp,n_f,freq_test,n_off=n_off,plot=False,show=False,normalize=False)
    assert(len(I_f) == len(Q_f) == len(U_f) == len(V_f) == nchans)
    assert(np.all(np.abs(I_f - np.arange(0,nchans)) < 1e-10))
    assert(np.all(np.abs(Q_f - np.arange(0,nchans)) < 1e-10))
    assert(np.all(np.abs(U_f - np.arange(0,nchans)) < 1e-10))
    assert(np.all(np.abs(V_f - np.arange(0,nchans)) < 1e-10))
    print("Frequency spectra frequency varying signal success!")
    print("Done!")
    return

#Ideal frequency variable, time pulse
def test_freq_ideal_test_5():
    print("Testing frequency spec with ideal frequency varying time pulse...")
    #Ideal, variation in frequency, pulse in time, no normalization
    nchans = 100
    nsamples = 200
    tsamp = 1
    w = 10
    width_native= w*tsamp/(256e-6)
    n_t = 1
    n_f = 1
    n_off = 30
    freq_test = [np.linspace(1300,1500,nchans)]*4

    I = Q = U = V = np.zeros((nchans,nsamples))
    timestart = 50 - w//2
    timestop = 50 + w//2
    I[:,timestart:timestop] = np.transpose(np.array([np.arange(0,nchans)]*w)) #np.ones((nchans,w))
    Q[:,timestart:timestop] = np.transpose(np.array([np.arange(0,nchans)]*w)) #np.ones((nchans,w))
    U[:,timestart:timestop] = np.transpose(np.array([np.arange(0,nchans)]*w)) #np.ones((nchans,w))
    V[:,timestart:timestop] = np.transpose(np.array([np.arange(0,nchans)]*w)) #np.ones((nchans,w))


    (I_f,Q_f,U_f,V_f) = dsapol.get_stokes_vs_freq(I,Q,U,V,width_native,tsamp,n_f,freq_test,n_off=n_off,plot=False,show=False,normalize=False)
    assert(len(I_f) == len(Q_f) == len(U_f) == len(V_f) == nchans)
    assert(np.all(np.abs(I_f - np.arange(0,nchans)) < 1e-10))
    assert(np.all(np.abs(Q_f - np.arange(0,nchans)) < 1e-10))
    assert(np.all(np.abs(U_f - np.arange(0,nchans)) < 1e-10))
    assert(np.all(np.abs(V_f - np.arange(0,nchans)) < 1e-10))
    print("Frequency spectra frequency varying time pulse success!")
    print("Done!")
    return

#Ideal frequency variable, time variable
def test_freq_ideal_test_6():
    print("Testing frequency spec with ideal frequency varying time varying signal...")
    #Ideal, variation in frequency, variation in time, no normalization
    nchans = 100
    nsamples = 200
    tsamp = 1
    w = 10
    width_native = -1#w*tsamp/(256e-6)
    n_t = 1
    n_f = 1
    n_off = -1
    freq_test = [np.linspace(1300,1500,nchans)]*4

    I = Q = U = V = np.zeros((nchans,nsamples))
    I = Q = U = V = np.array([np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10))]*nchans)*np.transpose(np.array([np.arange(0,nchans)]*nsamples))
    Q = 0.75*Q
    U = 0.5*U
    V = 0.25*V

    (I_f,Q_f,U_f,V_f) = dsapol.get_stokes_vs_freq(I,Q,U,V,width_native,tsamp,n_f,freq_test,n_off=n_off,plot=False,show=False,normalize=False)

    assert(len(I_f) == len(Q_f) == len(U_f) == len(V_f) == nchans)
    assert(np.all(np.abs(I_f - np.arange(0,nchans)*np.mean(np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10)))) < 1e-10))
    assert(np.all(np.abs(Q_f - np.arange(0,nchans)*np.mean(0.75*np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10)))) < 1e-10))
    assert(np.all(np.abs(U_f - np.arange(0,nchans)*np.mean(0.5*np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10)))) < 1e-10))
    assert(np.all(np.abs(V_f - np.arange(0,nchans)*np.mean(0.25*np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10)))) < 1e-10))
    print("Frequency spectra frequency varying time varying signal success!")
    print("Done!")
    return

#Noisy constant in time
def test_freq_noisy_test_1():
    print("Testing frequency spec with noisy constant signal...")
    nchans = 100
    nsamples = 200
    trial_sigmas = 1/(10**np.arange(1,5))
    for sigma_noise in trial_sigmas:
        print("Trial SNR = " + str(1/sigma_noise))
        ## Noisy, Constant in frequency, constant in time
        #--->no normalization
        nchans = 100
        nsamples = 200
        tsamp = 1
        w = 1
        width_native = -1#w*tsamp/(256e-6)
        n_t = 1
        n_f = 1
        n_off = -1
        err_max = sigma_noise/np.sqrt(nsamples)
        freq_test = [np.linspace(1300,1500,nchans)]*4

        I = Q = U = V = np.ones((nchans,nsamples))
        I += np.random.normal(0,sigma_noise,(nchans,nsamples))
        Q += np.random.normal(0,sigma_noise,(nchans,nsamples))
        U += np.random.normal(0,sigma_noise,(nchans,nsamples))
        V += np.random.normal(0,sigma_noise,(nchans,nsamples))

        (I_f,Q_f,U_f,V_f) = dsapol.get_stokes_vs_freq(I,Q,U,V,width_native,tsamp,n_f,freq_test,n_off=n_off,plot=False,show=False,normalize=False)

        assert(len(I_f) == len(Q_f) == len(U_f) == len(V_f) == nchans)
        assert(np.all(np.abs(I_f - 1) < 10*err_max))
        assert(np.all(np.abs(Q_f - 1) < 10*err_max))
        assert(np.all(np.abs(U_f - 1) < 10*err_max))
        assert(np.all(np.abs(V_f - 1) < 10*err_max))

        #--->normalization
        nchans = 100
        nsamples = 200
        tsamp = 1
        w = 1
        width_native = -1#w*tsamp/(256e-6)
        n_t = 1
        n_f = 1
        n_off = -1
        freq_test = [np.linspace(1300,1500,nchans)]*4

        I = Q = U = V = np.ones((nchans,nsamples))
        I += np.random.normal(0,sigma_noise,(nchans,nsamples))
        Q += np.random.normal(0,sigma_noise,(nchans,nsamples))
        U += np.random.normal(0,sigma_noise,(nchans,nsamples))
        V += np.random.normal(0,sigma_noise,(nchans,nsamples))


        (I_f,Q_f,U_f,V_f) = dsapol.get_stokes_vs_freq(I,Q,U,V,width_native,tsamp,n_f,freq_test,n_off=n_off,plot=False,show=False,normalize=True)

        assert(len(I_f) == len(Q_f) == len(U_f) == len(V_f) == nchans)
        assert(np.all(np.abs(I_f - 1) < 10))
        assert(np.all(np.abs(Q_f - 1) < 10))
        assert(np.all(np.abs(U_f - 1) < 10))
        assert(np.all(np.abs(V_f - 1) < 10))
    print("Frequency spectra contant signal success!")
    print("Done!")
    return


#Noisy time pulse
def test_freq_noisy_test_2():
    print("Testing frequency spec with noisy pulse...")
    trial_sigmas = 1/(10**np.arange(1,5))
    nchans = 100
    nsamples = 200
    for sigma_noise in trial_sigmas:
        print("Trial SNR = " + str(1/sigma_noise))
        ##Ideal, Constant in frequency, pulse in time 
        #-->no normalization
        nchans = 100
        nsamples = 200
        tsamp = 1
        w = 10
        width_native = w*tsamp/(256e-6)
        n_t = 1
        n_f = 1
        n_off = 30
        err_max = sigma_noise/np.sqrt(w)
        freq_test = [np.linspace(1300,1500,nchans)]*4

        I = Q = U = V = np.zeros((nchans,nsamples))
        timestart = 50 - w//2
        timestop = 50 + w//2
        I[:,timestart:timestop] = np.ones((nchans,w))
        Q[:,timestart:timestop] = np.ones((nchans,w))
        U[:,timestart:timestop] = np.ones((nchans,w))
        V[:,timestart:timestop] = np.ones((nchans,w))
        I += np.random.normal(0,sigma_noise,(nchans,nsamples))
        Q += np.random.normal(0,sigma_noise,(nchans,nsamples))
        U += np.random.normal(0,sigma_noise,(nchans,nsamples))
        V += np.random.normal(0,sigma_noise,(nchans,nsamples))


        (I_f,Q_f,U_f,V_f) = dsapol.get_stokes_vs_freq(I,Q,U,V,width_native,tsamp,n_f,freq_test,n_off=n_off,plot=False,show=False,normalize=False)
        assert(len(I_f) == len(Q_f) == len(U_f) == len(V_f) == nchans)
        assert(np.all(np.abs(I_f - 1) < 10*err_max))
        assert(np.all(np.abs(Q_f - 1) < 10*err_max))
        assert(np.all(np.abs(U_f - 1) < 10*err_max))
        assert(np.all(np.abs(V_f - 1) < 10*err_max))

        #-->normalization
        nchans = 100
        nsamples = 200
        tsamp = 1
        w = 10
        width_native = w*tsamp/(256e-6)
        n_t = 1
        n_f = 1
        n_off = 30
        freq_test = [np.linspace(1300,1500,nchans)]*4

        I = Q = U = V = np.zeros((nchans,nsamples))
        timestart = 50 - w//2
        timestop = 50 + w//2
        err_max = sigma_noise/np.sqrt(w)

        I[:,timestart:timestop] = np.ones((nchans,w))
        Q[:,timestart:timestop] = np.ones((nchans,w))
        U[:,timestart:timestop] = np.ones((nchans,w))
        V[:,timestart:timestop] = np.ones((nchans,w))
        I += np.random.normal(0,sigma_noise,(nchans,nsamples))
        Q += np.random.normal(0,sigma_noise,(nchans,nsamples))
        U += np.random.normal(0,sigma_noise,(nchans,nsamples))
        V += np.random.normal(0,sigma_noise,(nchans,nsamples))

        (I_f,Q_f,U_f,V_f) = dsapol.get_stokes_vs_freq(I,Q,U,V,width_native,tsamp,n_f,freq_test,n_off=n_off,plot=False,show=False,normalize=True)
        assert(len(I_f) == len(Q_f) == len(U_f) == len(V_f) == nchans)
        assert(np.all(np.abs(I_f-1/np.std(np.mean(I[:,:n_off],axis=0))) < 10*np.sqrt(nchans/w)))
        assert(np.all(np.abs(Q_f-1/np.std(np.mean(Q[:,:n_off],axis=0))) < 10*np.sqrt(nchans/w)))
        assert(np.all(np.abs(U_f-1/np.std(np.mean(U[:,:n_off],axis=0))) < 10*np.sqrt(nchans/w)))
        assert(np.all(np.abs(V_f-1/np.std(np.mean(V[:,:n_off],axis=0))) < 10*np.sqrt(nchans/w)))
    print("Frequency spectra noisy pulse success!")
    print("Done!")
    return

#Noisy time variable
def test_freq_noisy_test_3():
    print("Testing frequency spec with noisy time varying signal...")
    trial_sigmas = 1/(10**np.arange(1,5))
    nchans = 100
    nsamples = 200
    for sigma_noise in trial_sigmas:
        print("Trial SNR = " + str(1/sigma_noise))
        ##Ideal, constant in frequency, variation in time, 
        #-->no normalization
        nchans = 100
        nsamples = 200
        tsamp = 1
        w = 10
        width_native = -1#w*tsamp/(256e-6)
        n_t = 1
        n_f = 1
        n_off = -1
        err_max = sigma_noise/np.sqrt(nsamples)
        freq_test = [np.linspace(1300,1500,nchans)]*4

        I = Q = U = V = np.zeros((nchans,nsamples))
        I = Q = U = V = np.array([np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10))]*nchans)
        Q = 0.75*Q
        U = 0.5*U
        V = 0.25*V

        I += np.random.normal(0,sigma_noise,(nchans,nsamples))
        Q += np.random.normal(0,sigma_noise,(nchans,nsamples))
        U += np.random.normal(0,sigma_noise,(nchans,nsamples))
        V += np.random.normal(0,sigma_noise,(nchans,nsamples))

        (I_f,Q_f,U_f,V_f) = dsapol.get_stokes_vs_freq(I,Q,U,V,width_native,tsamp,n_f,freq_test,n_off=n_off,plot=False,show=False,normalize=False)
        assert(len(I_f) == len(Q_f) == len(U_f) == len(V_f) == nchans)
        assert(np.all(np.abs(I_f - np.mean(np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10)))) < 10*err_max))
        assert(np.all(np.abs(Q_f - np.mean(0.75*np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10)))) < 10*err_max))
        assert(np.all(np.abs(U_f - np.mean(0.5*np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10)))) < 10*err_max))
        assert(np.all(np.abs(V_f - np.mean(0.25*np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10)))) < 10*err_max))

        #-->normalization
        nchans = 100
        nsamples = 200
        tsamp = 1
        w = 10
        width_native = -1#w*tsamp/(256e-6)
        n_t = 1
        n_f = 1
        n_off = -1
        err_max = sigma_noise/np.sqrt(nsamples)

        I = Q = U = V = np.zeros((nchans,nsamples))
        I = Q = U = V = np.array([np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10))]*nchans)
        Q = 0.75*Q
        U = 0.5*U
        V = 0.25*V

        I += np.random.normal(0,sigma_noise,(nchans,nsamples))
        Q += np.random.normal(0,sigma_noise,(nchans,nsamples))
        U += np.random.normal(0,sigma_noise,(nchans,nsamples))
        V += np.random.normal(0,sigma_noise,(nchans,nsamples))


        (I_f,Q_f,U_f,V_f) = dsapol.get_stokes_vs_freq(I,Q,U,V,width_native,tsamp,n_f,freq_test,n_off=n_off,plot=False,show=False,normalize=False)
        assert(len(I_f) == len(Q_f) == len(U_f) == len(V_f) == nchans)
        assert(np.all(np.abs(I_f - np.mean(np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10)))/np.std(np.mean(I[:,:n_off],axis=0))) < 10*np.sqrt(nchans/nsamples)))
        assert(np.all(np.abs(Q_f - np.mean(0.75*np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10)))/np.std(np.mean(I[:,:n_off],axis=0))) < 10*np.sqrt(nchans/nsamples)))
        assert(np.all(np.abs(U_f - np.mean(0.5*np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10)))/np.std(np.mean(I[:,:n_off],axis=0))) < 10*np.sqrt(nchans/nsamples)))
        assert(np.all(np.abs(V_f - np.mean(0.25*np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10)))/np.std(np.mean(I[:,:n_off],axis=0))) < 10*np.sqrt(nchans/nsamples)))
    print("Frequency spectra noisy time varying signal success!")
    print("Done!")
    return

#Noisy frequency variable, constant in time
def test_freq_noisy_test_4():
    print("Testing frequency spec with noisy frequency varying signal...")
    trial_sigmas = 1/(10**np.arange(1,5))
    nchans = 100
    nsamples = 200
    for sigma_noise in trial_sigmas:
        print("Trial SNR = " + str(1/sigma_noise))
        # Noisy, variation in frequency, constant in time
        #-->no normalization

        nchans = 100
        nsamples = 200
        tsamp = 1
        w = 1
        width_native = -1#w*tsamp/(256e-6)
        n_t = 1
        n_f = 1
        n_off = -1
        err_max = sigma_noise/np.sqrt(nsamples)
        freq_test = [np.linspace(1300,1500,nchans)]*4

        I = Q = U = V = np.transpose(np.array([np.arange(0,nchans)]*nsamples,dtype=float))#np.ones((nchans,nsamples))

        I += np.random.normal(0,sigma_noise,(nchans,nsamples))
        Q += np.random.normal(0,sigma_noise,(nchans,nsamples))
        U += np.random.normal(0,sigma_noise,(nchans,nsamples))
        V += np.random.normal(0,sigma_noise,(nchans,nsamples))

        (I_f,Q_f,U_f,V_f) = dsapol.get_stokes_vs_freq(I,Q,U,V,width_native,tsamp,n_f,freq_test,n_off=n_off,plot=False,show=False,normalize=False)
        assert(len(I_f) == len(Q_f) == len(U_f) == len(V_f) == nchans)
        assert(np.all(np.abs(I_f - np.arange(0,nchans)) < 10*err_max))
        assert(np.all(np.abs(Q_f - np.arange(0,nchans)) < 10*err_max))
        assert(np.all(np.abs(U_f - np.arange(0,nchans)) < 10*err_max))
        assert(np.all(np.abs(V_f - np.arange(0,nchans)) < 10*err_max))

        #-->normalization

        nchans = 100
        nsamples = 200
        tsamp = 1
        w = 1
        width_native = -1#w*tsamp/(256e-6)
        n_t = 1
        n_f = 1
        n_off = -1
        err_max = sigma_noise/np.sqrt(nsamples)

        I = Q = U = V = np.transpose(np.array([np.arange(0,nchans)]*nsamples,dtype=float))#np.ones((nchans,nsamples))

        I += np.random.normal(0,sigma_noise,(nchans,nsamples))
        Q += np.random.normal(0,sigma_noise,(nchans,nsamples))
        U += np.random.normal(0,sigma_noise,(nchans,nsamples))
        V += np.random.normal(0,sigma_noise,(nchans,nsamples))

        (I_f,Q_f,U_f,V_f) = dsapol.get_stokes_vs_freq(I,Q,U,V,width_native,tsamp,n_f,freq_test,n_off=n_off,plot=False,show=False,normalize=False)
        assert(len(I_f) == len(Q_f) == len(U_f) == len(V_f) == nchans)
        assert(np.all(np.abs(I_f - np.arange(0,nchans)) < 10*np.sqrt(nchans/nsamples)))
        assert(np.all(np.abs(Q_f - np.arange(0,nchans)) < 10*np.sqrt(nchans/nsamples)))
        assert(np.all(np.abs(U_f - np.arange(0,nchans)) < 10*np.sqrt(nchans/nsamples)))
        assert(np.all(np.abs(V_f - np.arange(0,nchans)) < 10*np.sqrt(nchans/nsamples)))
    print("Frequency spectra frequency noisy frequency varying signal success!")
    print("Done!")
    return


#Noisy frequency variable, time pulse
def test_freq_noisy_test_5():
    print("Testing frequency spec with noisy frequency varying time pulse...")
    trial_sigmas = 1/(10**np.arange(1,5))
    nchans = 100
    nsamples = 200
    for sigma_noise in trial_sigmas:
        print("Trial SNR = " + str(1/sigma_noise))
        ##Noisy, variation in frequency, pulse in time, 
        #-->no normalization
        nchans = 100
        nsamples = 200
        tsamp = 1
        w = 10
        width_native = w*tsamp/(256e-6)
        n_t = 1
        n_f = 1
        n_off = 30
        err_max = sigma_noise/np.sqrt(w)
        freq_test = [np.linspace(1300,1500,nchans)]*4

        I = Q = U = V = np.zeros((nchans,nsamples))
        timestart = 50 - w//2
        timestop = 50 + w//2
        I[:,timestart:timestop] = np.transpose(np.array([np.arange(0,nchans)]*w)) #np.ones((nchans,w))
        Q[:,timestart:timestop] = np.transpose(np.array([np.arange(0,nchans)]*w)) #np.ones((nchans,w))
        U[:,timestart:timestop] = np.transpose(np.array([np.arange(0,nchans)]*w)) #np.ones((nchans,w))
        V[:,timestart:timestop] = np.transpose(np.array([np.arange(0,nchans)]*w)) #np.ones((nchans,w))

        I += np.random.normal(0,sigma_noise,(nchans,nsamples))
        Q += np.random.normal(0,sigma_noise,(nchans,nsamples))
        U += np.random.normal(0,sigma_noise,(nchans,nsamples))
        V += np.random.normal(0,sigma_noise,(nchans,nsamples))

        (I_f,Q_f,U_f,V_f) = dsapol.get_stokes_vs_freq(I,Q,U,V,width_native,tsamp,n_f,freq_test,n_off=n_off,plot=False,show=False,normalize=False)
        assert(len(I_f) == len(Q_f) == len(U_f) == len(V_f) == nchans)
        assert(np.all(np.abs(I_f - np.arange(0,nchans)) < 10*err_max))
        assert(np.all(np.abs(Q_f - np.arange(0,nchans)) < 10*err_max))
        assert(np.all(np.abs(U_f - np.arange(0,nchans)) < 10*err_max))
        assert(np.all(np.abs(V_f - np.arange(0,nchans)) < 10*err_max))

        #-->normalization
        nchans = 100
        nsamples = 200
        tsamp = 1
        w = 10
        width_native = w*tsamp/(256e-6)
        n_t = 1
        n_f = 1
        n_off = 30
        err_max = sigma_noise/np.sqrt(w)

        I = Q = U = V = np.zeros((nchans,nsamples))
        timestart = 50 - w//2
        timestop = 50 + w//2
        I[:,timestart:timestop] = np.transpose(np.array([np.arange(0,nchans)]*w)) #np.ones((nchans,w))
        Q[:,timestart:timestop] = np.transpose(np.array([np.arange(0,nchans)]*w)) #np.ones((nchans,w))
        U[:,timestart:timestop] = np.transpose(np.array([np.arange(0,nchans)]*w)) #np.ones((nchans,w))
        V[:,timestart:timestop] = np.transpose(np.array([np.arange(0,nchans)]*w)) #np.ones((nchans,w))

        I += np.random.normal(0,sigma_noise,(nchans,nsamples))
        Q += np.random.normal(0,sigma_noise,(nchans,nsamples))
        U += np.random.normal(0,sigma_noise,(nchans,nsamples))
        V += np.random.normal(0,sigma_noise,(nchans,nsamples))

        (I_f,Q_f,U_f,V_f) = dsapol.get_stokes_vs_freq(I,Q,U,V,width_native,tsamp,n_f,freq_test,n_off=n_off,plot=False,show=False,normalize=False)
        assert(len(I_f) == len(Q_f) == len(U_f) == len(V_f) == nchans)
        assert(np.all(np.abs(I_f - np.arange(0,nchans)) < 10*np.sqrt(nchans/w)))
        assert(np.all(np.abs(Q_f - np.arange(0,nchans)) < 10*np.sqrt(nchans/w)))
        assert(np.all(np.abs(U_f - np.arange(0,nchans)) < 10*np.sqrt(nchans/w)))
        assert(np.all(np.abs(V_f - np.arange(0,nchans)) < 10*np.sqrt(nchans/w)))
    print("Frequency spectra noisy frequency varying time pulse success!")
    print("Done!")
    return

#Noisy frequency variable, time variable
def test_freq_noisy_test_6():
    print("Testing frequency spec with noisy frequency varying time varying signal...")
    trial_sigmas = 1/(10**np.arange(1,5))
    nchans = 100
    nsamples = 200
    for sigma_noise in trial_sigmas:
        print("Trial SNR = " + str(1/sigma_noise))
        #Noisy, variation in frequency, variation in time
        #-->no normalization
        nchans = 100
        nsamples = 200
        tsamp = 1
        w = 10
        width_native = -1#w*tsamp/(256e-6)
        n_t = 1
        n_f = 1
        n_off = -1
        err_max = sigma_noise/np.sqrt(nsamples)
        freq_test = [np.linspace(1300,1500,nchans)]*4

        I = Q = U = V = np.zeros((nchans,nsamples))
        I = Q = U = V = np.array([np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10))]*nchans)*np.transpose(np.array([np.arange(0,nchans)]*nsamples))
        Q = 0.75*Q
        U = 0.5*U
        V = 0.25*V

        I += np.random.normal(0,sigma_noise,(nchans,nsamples))
        Q += np.random.normal(0,sigma_noise,(nchans,nsamples))
        U += np.random.normal(0,sigma_noise,(nchans,nsamples))
        V += np.random.normal(0,sigma_noise,(nchans,nsamples))

        (I_f,Q_f,U_f,V_f) = dsapol.get_stokes_vs_freq(I,Q,U,V,width_native,tsamp,n_f,freq_test,n_off=n_off,plot=False,show=False,normalize=False)

        assert(len(I_f) == len(Q_f) == len(U_f) == len(V_f) == nchans)
        assert(np.all(np.abs(I_f - np.arange(0,nchans)*np.mean(np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10)))) < 10*err_max))
        assert(np.all(np.abs(Q_f - np.arange(0,nchans)*np.mean(0.75*np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10)))) < 10*err_max))
        assert(np.all(np.abs(U_f - np.arange(0,nchans)*np.mean(0.5*np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10)))) < 10*err_max))
        assert(np.all(np.abs(V_f - np.arange(0,nchans)*np.mean(0.25*np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10)))) < 10*err_max))

        #-->normalization
        nchans = 100
        nsamples = 200
        tsamp = 1
        w = 10
        width_native = -1#w*tsamp/(256e-6)
        n_t = 1
        n_f = 1
        n_off = -1

        I = Q = U = V = np.zeros((nchans,nsamples))
        I = Q = U = V = np.array([np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10))]*nchans)*np.transpose(np.array([np.arange(0,nchans)]*nsamples))
        Q = 0.75*Q
        U = 0.5*U
        V = 0.25*V

        I += np.random.normal(0,sigma_noise,(nchans,nsamples))
        Q += np.random.normal(0,sigma_noise,(nchans,nsamples))
        U += np.random.normal(0,sigma_noise,(nchans,nsamples))
        V += np.random.normal(0,sigma_noise,(nchans,nsamples))

        (I_f,Q_f,U_f,V_f) = dsapol.get_stokes_vs_freq(I,Q,U,V,width_native,tsamp,n_f,freq_test,n_off=n_off,plot=False,show=False,normalize=False)

        assert(len(I_f) == len(Q_f) == len(U_f) == len(V_f) == nchans)
        assert(np.all(np.abs(I_f - np.arange(0,nchans)*np.mean(np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10)))) < 10*err_max))
        assert(np.all(np.abs(Q_f - np.arange(0,nchans)*np.mean(0.75*np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10)))) < 10*err_max))
        assert(np.all(np.abs(U_f - np.arange(0,nchans)*np.mean(0.5*np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10)))) < 10*err_max))
        assert(np.all(np.abs(V_f - np.arange(0,nchans)*np.mean(0.25*np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10)))) < 10*err_max))
    print("Frequency spectra noisy frequency varying time varying signal success!")
    print("Done!")
    return

#frequency and time binning test
def test_binning_test():
    print("Testing frequency and time binning pipelines...")
    nchans = 100
    nsamples = 200
    test_img = np.ones((nchans,nsamples))
    out_img = dsapol.avg_time(test_img,10)
    assert(out_img.shape == (100,20))
    assert(np.all(out_img == np.ones((nchans,nsamples//10))))

    out_img = dsapol.avg_freq(out_img,10)
    assert(out_img.shape == (10,20))
    assert(np.all(out_img == np.ones((nchans//10,nsamples//10))))

    testaxis = []
    for i in range(1,nchans//5 + 1):
        for j in range(5):
            testaxis.append(i)
    test_img = np.array([testaxis]*nsamples).transpose()
    assert(test_img.shape == (100,200))
    out_img = dsapol.avg_time(test_img,10)
    assert(out_img.shape == (100,20))
    out_img = dsapol.avg_freq(out_img,5)
    assert(out_img.shape == (20,20))
    for i in range(20):
        assert(np.all(out_img[:,i] == np.arange(1,21)))
    print("Binning test successful!")
    print("Done!")
    return


def main():
    print("Running time averaged frequency spectra and binning tests...")
    freq_ideal_test_1()
    freq_ideal_test_2()
    freq_ideal_test_3()
    freq_ideal_test_4()
    freq_ideal_test_5()
    freq_ideal_test_6()
    freq_noisy_test_1()
    freq_noisy_test_2()
    freq_noisy_test_3()
    freq_noisy_test_4()
    freq_noisy_test_5()
    freq_noisy_test_6()
    binning_test()
    print("All Tests Passed!")

if __name__ == "__main__":
    main()
