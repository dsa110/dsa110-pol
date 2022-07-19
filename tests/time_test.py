from dsapol import dsapol
import numpy as np
from scipy.stats import norm


#Ideal, constant in time
def test_time_ideal_test_1():
    print("Testing time series with ideal constant signal...")
    # Ideal, Constant in frequency, constant in time, no normalization
    nchans = 100
    nsamples = 200
    tsamp = 1
    w = 1
    width_native = -1#w*tsamp/(256e-6)
    n_t = 1
    n_f = 1
    n_off = -1

    I = Q = U = V = np.ones((nchans,nsamples))


    (I_t,Q_t,U_t,V_t) = dsapol.get_stokes_vs_time(I,Q,U,V,width_native, tsamp, n_t, n_off=n_off,normalize=False)
    assert(len(I_t) == len(Q_t) == len(U_t) == len(V_t) == nsamples)
    assert(np.all(I_t == 1))
    assert(np.all(Q_t == 1))
    assert(np.all(U_t == 1))
    assert(np.all(V_t == 1))
    print("Time series contant signal success!")
    print("Done!")
    return

#Ideal pulse
def test_time_ideal_test_2():
    print("Testing time series with ideal pulse...")
    #Ideal, Constant in frequency, pulse in time, no normalization
    nchans = 100
    nsamples = 200
    tsamp = 1
    w = 10
    width_native = w*tsamp/(256e-6)
    n_t = 1
    n_f = 1
    n_off = -1

    I = Q = U = V = np.zeros((nchans,nsamples))
    timestart = 50 - w//2
    timestop = 50 + w//2
    I[:,timestart:timestop] = np.ones((nchans,w))
    Q[:,timestart:timestop] = np.ones((nchans,w))
    U[:,timestart:timestop] = np.ones((nchans,w))
    V[:,timestart:timestop] = np.ones((nchans,w))


    (I_t,Q_t,U_t,V_t) = dsapol.get_stokes_vs_time(I,Q,U,V,width_native, tsamp, n_t, n_off=n_off,normalize=False)
    assert(len(I_t) == len(Q_t) == len(U_t) == len(V_t) == nsamples)
    assert(np.all(I_t[timestart:timestop] == 1))
    assert(np.all(Q_t[timestart:timestop] == 1))
    assert(np.all(U_t[timestart:timestop] == 1))
    assert(np.all(V_t[timestart:timestop] == 1))
    print("Time series ideal pulse success!")
    print("Done!")
    return

#Ideal time variable
def test_time_ideal_test_3():
    print("Testing time series with time variable...")
    #Ideal, constant in frequency, variation in time, no normalization
    nchans = 100
    nsamples = 200
    tsamp = 1
    w = 10
    width_native= w*tsamp/(256e-6)
    n_t = 1
    n_f = 1
    n_off = -1

    I = Q = U = V = np.zeros((nchans,nsamples))
    I = Q = U = V = np.array([np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10))]*nchans)
    Q = 0.75*Q
    U = 0.5*U
    V = 0.25*V


    (I_t,Q_t,U_t,V_t) = dsapol.get_stokes_vs_time(I,Q,U,V,width_native, tsamp, n_t, n_off=n_off,normalize=False)

    assert(len(I_t) == len(Q_t) == len(U_t) == len(V_t) == nsamples)
    assert(np.all(np.abs(I_t - np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10))) < 1e-10))
    assert(np.all(np.abs(Q_t - 0.75*np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10))) < 1e-10))
    assert(np.all(np.abs(U_t - 0.5*np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10))) < 1e-10))
    assert(np.all(np.abs(V_t - 0.25*np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10))) < 1e-10))
    print("Time series ideal time variable success!")
    print("Done!")
    return

#Ideal frequency variable, constant in time
def test_time_ideal_test_4():
    print("Testing time series with frequency variable...")
    # Ideal, variation in frequency, constant in time, no normalization
    nchans = 100
    nsamples = 200
    tsamp = 1
    w = 1
    width_native = -1#w*tsamp/(256e-6)
    n_t = 1
    n_f = 1
    n_off = -1

    I = Q = U = V = np.transpose(np.array([np.arange(0,nchans)]*nsamples))#np.ones((nchans,nsamples))


    (I_t,Q_t,U_t,V_t) = dsapol.get_stokes_vs_time(I,Q,U,V,width_native, tsamp, n_t, n_off=n_off,normalize=False)
    assert(len(I_t) == len(Q_t) == len(U_t) == len(V_t) == nsamples)
    assert(np.all(I_t == nchans/2 - 0.5))
    assert(np.all(Q_t == nchans/2 - 0.5))
    assert(np.all(U_t == nchans/2 - 0.5))
    assert(np.all(V_t == nchans/2 - 0.5))
    print("Time series ideal frequency variable success!")
    print("Done!")
    return

#Ideal frequency variable, time pulse
def test_time_ideal_test_5():
    print("Testing time series with frequency variable time pulse...")
    #Ideal, variation in frequency, pulse in time, no normalization
    nchans = 100
    nsamples = 200
    tsamp = 1
    w = 10
    width_native= w*tsamp/(256e-6)
    n_t = 1
    n_f = 1
    n_off = -1

    I = Q = U = V = np.zeros((nchans,nsamples))
    timestart = 50 - w//2
    timestop = 50 + w//2
    I[:,timestart:timestop] = np.transpose(np.array([np.arange(0,nchans)]*w)) #np.ones((nchans,w))
    Q[:,timestart:timestop] = np.transpose(np.array([np.arange(0,nchans)]*w)) #np.ones((nchans,w))
    U[:,timestart:timestop] = np.transpose(np.array([np.arange(0,nchans)]*w)) #np.ones((nchans,w))
    V[:,timestart:timestop] = np.transpose(np.array([np.arange(0,nchans)]*w)) #np.ones((nchans,w))


    (I_t,Q_t,U_t,V_t) = dsapol.get_stokes_vs_time(I,Q,U,V,width_native, tsamp, n_t, n_off=n_off,normalize=False)
    assert(len(I_t) == len(Q_t) == len(U_t) == len(V_t) == nsamples)
    assert(np.all(I_t[timestart:timestop] == nchans/2 - 0.5))
    assert(np.all(Q_t[timestart:timestop] == nchans/2 - 0.5))
    assert(np.all(U_t[timestart:timestop] == nchans/2 - 0.5))
    assert(np.all(V_t[timestart:timestop] == nchans/2 - 0.5))
    print("Time series ideal frequency variable time pulse success!")
    print("Done!")
    return 

#Ideal frequency variable, time variable
def test_time_ideal_test_6():
    print("Testing time series with frequency variable time variable...")
    #Ideal, variation in frequency, variation in time, no normalization
    nchans = 100
    nsamples = 200
    tsamp = 1
    w = 10
    width_native = w*tsamp/(256e-6)
    n_t = 1
    n_f = 1
    n_off = -1

    I = Q = U = V = np.zeros((nchans,nsamples))
    I = Q = U = V = np.array([np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10))]*nchans)*np.transpose(np.array([np.arange(0,nchans)]*nsamples))
    Q = 0.75*Q
    U = 0.5*U
    V = 0.25*V


    (I_t,Q_t,U_t,V_t) = dsapol.get_stokes_vs_time(I,Q,U,V,width_native, tsamp, n_t, n_off=n_off,normalize=False)

    assert(len(I_t) == len(Q_t) == len(U_t) == len(V_t) == nsamples)
    assert(np.all(np.abs(I_t - (nchans/2 - 0.5)*np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10))) < 1e-10))
    assert(np.all(np.abs(Q_t - (nchans/2 - 0.5)*0.75*np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10))) < 1e-10))
    assert(np.all(np.abs(U_t - (nchans/2 - 0.5)*0.5*np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10))) < 1e-10))
    assert(np.all(np.abs(V_t - (nchans/2 - 0.5)*0.25*np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10))) < 1e-10))
    print("Time series ideal frequency variable time variable success!")
    print("Done!")
    return

#Noisy constant in time
def test_time_noisy_test_1():
    print("Testing time series with noisy constant signal...")
    nchans = 100
    nsamples = 200
    trial_sigmas = 1/(10**np.arange(1,5))
    for sigma_noise in trial_sigmas:
        print("Trial SNR = " + str(1/sigma_noise))
        err_max = sigma_noise/np.sqrt(nchans)

        ## Noisy, Constant in frequency, constant in time
        #--->no normalization
        tsamp = 1
        w = 1
        width_native = w*tsamp/(256e-6)
        n_t = 1
        n_f = 1
        n_off = -1

        I = Q = U = V = np.ones((nchans,nsamples))
        I += np.random.normal(0,sigma_noise,(nchans,nsamples))
        Q += np.random.normal(0,sigma_noise,(nchans,nsamples))
        U += np.random.normal(0,sigma_noise,(nchans,nsamples))
        V += np.random.normal(0,sigma_noise,(nchans,nsamples))


        (I_t,Q_t,U_t,V_t) = dsapol.get_stokes_vs_time(I,Q,U,V,width_native, tsamp, n_t, n_off=n_off,normalize=False)
        assert(len(I_t) == len(Q_t) == len(U_t) == len(V_t) == nsamples)

        #plt.plot(I_t)
        #plt.plot(Q_t)
        #plt.plot(U_t)
        #plt.plot(V_t)

        assert(np.all(np.abs(I_t - 1) < 10*err_max))
        assert(np.all(np.abs(Q_t - 1) < 10*err_max))
        assert(np.all(np.abs(U_t - 1) < 10*err_max))
        assert(np.all(np.abs(V_t - 1) < 10*err_max))


        #--->normalization
        nchans = 100
        nsamples = 200
        tsamp = 1
        w = 1
        width_native = w*tsamp/(256e-6)
        n_t = 1
        n_f = 1
        n_off = -1

        I = Q = U = V = np.ones((nchans,nsamples))
        I += np.random.normal(0,sigma_noise,(nchans,nsamples))
        Q += np.random.normal(0,sigma_noise,(nchans,nsamples))
        U += np.random.normal(0,sigma_noise,(nchans,nsamples))
        V += np.random.normal(0,sigma_noise,(nchans,nsamples))


        (I_t,Q_t,U_t,V_t) = dsapol.get_stokes_vs_time(I,Q,U,V,width_native, tsamp, n_t, n_off=n_off,normalize=True)
        assert(len(I_t) == len(Q_t) == len(U_t) == len(V_t) == nsamples)

        assert(np.all(np.abs(I_t) < 10))#10*err_max))
        assert(np.all(np.abs(Q_t) < 10))#10*err_max))
        assert(np.all(np.abs(U_t) < 10))#10*err_max))
        assert(np.all(np.abs(V_t) < 10))#10*err_max))
    print("Time series contant signal success!")
    print("Done!")
    return

#Noisy time pulse
def test_time_noisy_test_2():
    print("Testing time series with noisy pulse...")
    trial_sigmas = 1/(10**np.arange(1,5))
    nchans = 100
    nsamples = 200
    for sigma_noise in trial_sigmas:
        print("Trial SNR = " + str(1/sigma_noise))
        err_max = sigma_noise/np.sqrt(nchans)
        ##Ideal, Constant in frequency, pulse in time 
        #-->no normalization
        tsamp = 1
        w = 10
        width_native = w*tsamp/(256e-6)
        n_t = 1
        n_f = 1
        n_off = -1

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


        (I_t,Q_t,U_t,V_t) = dsapol.get_stokes_vs_time(I,Q,U,V,width_native, tsamp, n_t, n_off=n_off,normalize=False)
        assert(len(I_t) == len(Q_t) == len(U_t) == len(V_t) == nsamples)
        assert(np.all(np.abs(I_t[timestart:timestop] - 1) < 10*err_max))
        assert(np.all(np.abs(Q_t[timestart:timestop] - 1) < 10*err_max))
        assert(np.all(np.abs(U_t[timestart:timestop] - 1) < 10*err_max))
        assert(np.all(np.abs(V_t[timestart:timestop] - 1) < 10*err_max))

        #-->normalization
        nchans = 100
        nsamples = 200
        tsamp = 1
        w = 10
        width_native = w*tsamp/(256e-6)
        n_t = 1
        n_f = 1
        n_off = 30

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

        (I_t,Q_t,U_t,V_t) = dsapol.get_stokes_vs_time(I,Q,U,V,width_native, tsamp, n_t, n_off=n_off,normalize=True)
        assert(len(I_t) == len(Q_t) == len(U_t) == len(V_t) == nsamples)
        assert(np.all(np.abs(I_t[timestart:timestop] - 1/np.std(np.mean(I[:,:n_off],axis=0))) < 10))#10*err_max))
        assert(np.all(np.abs(Q_t[timestart:timestop] - 1/np.std(np.mean(Q[:,:n_off],axis=0))) < 10))#10*err_max))
        assert(np.all(np.abs(U_t[timestart:timestop] - 1/np.std(np.mean(U[:,:n_off],axis=0))) < 10))#10*err_max))
        assert(np.all(np.abs(V_t[timestart:timestop] - 1/np.std(np.mean(V[:,:n_off],axis=0))) < 10))#10*err_max))
    print("Time series ideal pulse success!")
    print("Done!")
    return

#Noisy time variable
def test_time_noisy_test_3():
    print("Testing time series with time variable...")
    trial_sigmas = 1/(10**np.arange(1,5))
    nchans = 100
    nsamples = 200
    for sigma_noise in trial_sigmas:
        print("Trial SNR = " + str(1/sigma_noise))
        err_max = sigma_noise/np.sqrt(nchans)
        ##Ideal, constant in frequency, variation in time, 
        #-->no normalization
        tsamp = 1
        w = 10
        width_native = w*tsamp/(256e-6)
        n_t = 1
        n_f = 1
        n_off = -1

        I = Q = U = V = np.zeros((nchans,nsamples))
        I = Q = U = V = np.array([np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10))]*nchans)
        Q = 0.75*Q
        U = 0.5*U
        V = 0.25*V

        I += np.random.normal(0,sigma_noise,(nchans,nsamples))
        Q += np.random.normal(0,sigma_noise,(nchans,nsamples))
        U += np.random.normal(0,sigma_noise,(nchans,nsamples))
        V += np.random.normal(0,sigma_noise,(nchans,nsamples))


        (I_t,Q_t,U_t,V_t) = dsapol.get_stokes_vs_time(I,Q,U,V,width_native, tsamp, n_t, n_off=n_off,normalize=False)

        assert(len(I_t) == len(Q_t) == len(U_t) == len(V_t) == nsamples)
        assert(np.all(np.abs(I_t - np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10))) < 10*err_max))
        assert(np.all(np.abs(Q_t - 0.75*np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10)))< 10*err_max))
        assert(np.all(np.abs(U_t - 0.5*np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10)))< 10*err_max))
        assert(np.all(np.abs(V_t - 0.25*np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10)))< 10*err_max))

        #-->normalization
        nchans = 100
        nsamples = 200
        tsamp = 1
        w = 10
        width_native = w*tsamp/(256e-6)
        n_t = 1
        n_f = 1
        n_off = -1

        I = Q = U = V = np.zeros((nchans,nsamples))
        I = Q = U = V = np.array([np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10))]*nchans)
        Q = 0.75*Q
        U = 0.5*U
        V = 0.25*V

        I += np.random.normal(0,sigma_noise,(nchans,nsamples))
        Q += np.random.normal(0,sigma_noise,(nchans,nsamples))
        U += np.random.normal(0,sigma_noise,(nchans,nsamples))
        V += np.random.normal(0,sigma_noise,(nchans,nsamples))


        (I_t,Q_t,U_t,V_t) = dsapol.get_stokes_vs_time(I,Q,U,V,width_native, tsamp, n_t, n_off=n_off,normalize=True)

        assert(len(I_t) == len(Q_t) == len(U_t) == len(V_t) == nsamples)
        assert(np.all(np.abs(I_t - np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10)/np.std(np.mean(I[:,:n_off],axis=0)))) < 10))#10*err_max))
        assert(np.all(np.abs(Q_t - 0.75*np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10)/np.std(np.mean(I[:,:n_off],axis=0)))) < 10))#10*err_max))
        assert(np.all(np.abs(U_t - 0.5*np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10)/np.std(np.mean(I[:,:n_off],axis=0)))) < 10))#10*err_max))
        assert(np.all(np.abs(V_t - 0.25*np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10)/np.std(np.mean(I[:,:n_off],axis=0)))) < 10))#10*err_max))
    print("Time series ideal time variable success!")
    print("Done!")
    return   

#Noisy frequency variable, constant in time
def test_time_noisy_test_4():
    print("Testing time series with noisy frequency variable...")
    trial_sigmas = 1/(10**np.arange(1,5))
    nchans = 100
    nsamples = 200
    for sigma_noise in trial_sigmas:
        print("Trial SNR = " + str(1/sigma_noise))
        err_max = sigma_noise/np.sqrt(nchans)
        # Noisy, variation in frequency, constant in time
        #-->no normalization

        tsamp = 1
        w = 1
        width_native = w*tsamp/(256e-6)
        n_t = 1
        n_f = 1
        n_off = -1

        I = Q = U = V = np.transpose(np.array([np.arange(0,nchans)]*nsamples,dtype=float))#np.ones((nchans,nsamples))

        I += np.random.normal(0,sigma_noise,(nchans,nsamples))
        Q += np.random.normal(0,sigma_noise,(nchans,nsamples))
        U += np.random.normal(0,sigma_noise,(nchans,nsamples))
        V += np.random.normal(0,sigma_noise,(nchans,nsamples))

        (I_t,Q_t,U_t,V_t) = dsapol.get_stokes_vs_time(I,Q,U,V,width_native, tsamp, n_t, n_off=n_off,normalize=False)

        assert(len(I_t) == len(Q_t) == len(U_t) == len(V_t) == nsamples)
        assert(np.all(np.abs(I_t - nchans/2 + 0.5) < 10*err_max))
        assert(np.all(np.abs(Q_t - nchans/2 + 0.5) < 10*err_max))
        assert(np.all(np.abs(U_t - nchans/2 + 0.5) < 10*err_max))
        assert(np.all(np.abs(V_t - nchans/2 + 0.5) < 10*err_max))

        #-->normalization

        nchans = 100
        nsamples = 200
        tsamp = 1
        w = 1
        width_native = w*tsamp/(256e-6)
        n_t = 1
        n_f = 1
        n_off = -1

        I = Q = U = V = np.transpose(np.array([np.arange(0,nchans)]*nsamples,dtype=float))#np.ones((nchans,nsamples))

        I += np.random.normal(0,sigma_noise,(nchans,nsamples))
        Q += np.random.normal(0,sigma_noise,(nchans,nsamples))
        U += np.random.normal(0,sigma_noise,(nchans,nsamples))
        V += np.random.normal(0,sigma_noise,(nchans,nsamples))

        (I_t,Q_t,U_t,V_t) = dsapol.get_stokes_vs_time(I,Q,U,V,width_native, tsamp, n_t, n_off=n_off,normalize=True)

        assert(len(I_t) == len(Q_t) == len(U_t) == len(V_t) == nsamples)
        assert(np.all(np.abs(I_t) < 10))#10*err_max))
        assert(np.all(np.abs(Q_t) < 10))#10*err_max))
        assert(np.all(np.abs(U_t) < 10))#10*err_max))
        assert(np.all(np.abs(V_t) < 10))#10*err_max))
    print("Time series ideal frequency variable success!")
    print("Done!")
    return

#Noisy frequency variable, time pulse
def test_time_noisy_test_5():
    print("Testing time series with noisy frequency variable time pulse...")
    trial_sigmas = 1/(10**np.arange(1,5))
    nchans = 100
    nsamples = 200
    for sigma_noise in trial_sigmas:
        print("Trial SNR = " + str(1/sigma_noise))
        err_max = sigma_noise/np.sqrt(nchans)
        ##Noisy, variation in frequency, pulse in time,
        #-->no normalization
        tsamp = 1
        w = 10
        width_native = w*tsamp/(256e-6)
        n_t = 1
        n_f = 1
        n_off = -1

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

        (I_t,Q_t,U_t,V_t) = dsapol.get_stokes_vs_time(I,Q,U,V,width_native, tsamp, n_t, n_off=n_off,normalize=False)
        assert(len(I_t) == len(Q_t) == len(U_t) == len(V_t) == nsamples)
        assert(np.all(np.abs(I_t[timestart:timestop] - nchans/2 + 0.5) < 10*err_max))
        assert(np.all(np.abs(Q_t[timestart:timestop] - nchans/2 + 0.5) < 10*err_max))
        assert(np.all(np.abs(U_t[timestart:timestop] - nchans/2 + 0.5) < 10*err_max))
        assert(np.all(np.abs(V_t[timestart:timestop] - nchans/2 + 0.5) < 10*err_max))

        #-->normalization
        nchans = 100
        nsamples = 200
        tsamp = 1
        w = 10
        width_native = w*tsamp/(256e-6)
        n_t = 1
        n_f = 1
        n_off = -1

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

        (I_t,Q_t,U_t,V_t) = dsapol.get_stokes_vs_time(I,Q,U,V,width_native, tsamp, n_t, n_off=n_off,normalize=True)
        assert(len(I_t) == len(Q_t) == len(U_t) == len(V_t) == nsamples)
        #assert(np.all(np.abs(I_t[timestart:timestop] - nchans/2 + 0.5) < 10*err_max))
        #assert(np.all(np.abs(Q_t[timestart:timestop] - nchans/2 + 0.5) < 10*err_max))
        #assert(np.all(np.abs(U_t[timestart:timestop] - nchans/2 + 0.5) < 10*err_max))
        #assert(np.all(np.abs(V_t[timestart:timestop] - nchans/2 + 0.5) < 10*err_max))

        assert(np.all(np.abs(I_t[timestart:timestop] - (nchans/2 - 0.5)/np.std(np.mean(I[:,:n_off],axis=0))) < 10))#10*err_max))
        assert(np.all(np.abs(Q_t[timestart:timestop] - (nchans/2 - 0.5)/np.std(np.mean(Q[:,:n_off],axis=0))) < 10))#10*err_max))
        assert(np.all(np.abs(U_t[timestart:timestop] - (nchans/2 - 0.5)/np.std(np.mean(U[:,:n_off],axis=0))) < 10))#10*err_max))
        assert(np.all(np.abs(V_t[timestart:timestop] - (nchans/2 - 0.5)/np.std(np.mean(V[:,:n_off],axis=0))) < 10))#10*err_max))
    print("Time series noisy frequency variable time pulse success!")
    print("Done!")
    return

#Noisy frequency variable, time variable
def test_time_noisy_test_6():
    print("Testing time series with frequency variable time variable...")
    trial_sigmas = 1/(10**np.arange(1,5))
    nchans = 100
    nsamples = 200
    for sigma_noise in trial_sigmas:
        print("Trial SNR = " + str(1/sigma_noise))
        err_max = sigma_noise/np.sqrt(nchans)
        #Noisy, variation in frequency, variation in time
        #-->no normalization
        tsamp = 1
        w = 10
        width_native = w*tsamp/(256e-6)
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

        (I_t,Q_t,U_t,V_t) = dsapol.get_stokes_vs_time(I,Q,U,V,width_native, tsamp, n_t, n_off=n_off,normalize=False)

        assert(len(I_t) == len(Q_t) == len(U_t) == len(V_t) == nsamples)
        assert(np.all(np.abs(I_t - (nchans/2 - 0.5)*np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10))) < 10*err_max))
        assert(np.all(np.abs(Q_t - (nchans/2 - 0.5)*0.75*np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10))) < 10*err_max))
        assert(np.all(np.abs(U_t - (nchans/2 - 0.5)*0.5*np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10))) < 10*err_max))
        assert(np.all(np.abs(V_t - (nchans/2 - 0.5)*0.25*np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10))) < 10*err_max))

        #-->normalization
        nchans = 100
        nsamples = 200
        tsamp = 1
        w = 10
        width_native = w*tsamp/(256e-6)
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

        (I_t,Q_t,U_t,V_t) = dsapol.get_stokes_vs_time(I,Q,U,V,width_native, tsamp, n_t, n_off=n_off,normalize=True)

        assert(len(I_t) == len(Q_t) == len(U_t) == len(V_t) == nsamples)
        assert(np.all(np.abs(I_t - (nchans/2 - 0.5)*np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10))/np.std(np.mean(I[:,:n_off],axis=0))) < 10))#10*err_max))
        assert(np.all(np.abs(Q_t - (nchans/2 - 0.5)*0.75*np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10))/np.std(np.mean(I[:,:n_off],axis=0))) < 10))#10*err_max))
        assert(np.all(np.abs(U_t - (nchans/2 - 0.5)*0.5*np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10))/np.std(np.mean(I[:,:n_off],axis=0))) < 10))#10*err_max))
        assert(np.all(np.abs(V_t - (nchans/2 - 0.5)*0.25*np.sin(2*np.pi*np.arange(nsamples)/(nsamples/10))/np.std(np.mean(I[:,:n_off],axis=0))) < 10))#10*err_max))
    print("Time series noisy frequency variable time variable success!")
    print("Done!")
    return

def main():
    print("Running frequency averaged time series...")
    time_ideal_test_1()
    time_ideal_test_2()
    time_ideal_test_3()
    time_ideal_test_4()
    time_ideal_test_5()
    time_ideal_test_6()
    time_noisy_test_1()
    time_noisy_test_2()
    time_noisy_test_3()
    time_noisy_test_4()
    time_noisy_test_5()
    time_noisy_test_6()
    print("All tests passed!")
if __name__ == "__main__":
    main()
