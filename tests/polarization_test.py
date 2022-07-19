from dsapol import dsapol
import numpy as np
from scipy.stats import norm

#ideal PF, unpolarized
def test_PF_ideal_test_1():
    print("Testing ideal unpolarized source...")
    #unpolarized, continuous in time, not normalzed

    nchans = 100
    nsamples = 200
    tsamp=1
    w = 10
    width_native = -1#w*tsamp/(256e-6)
    n_t = 1
    n_f = 1
    n_off = -1


    freq_test = [np.linspace(1300,1500,nchans)]*4

    I = np.ones((nchans,nsamples))
    Q = U = V = np.zeros((nchans,nsamples))

    (pol_f,pol_t,avg) = dsapol.get_pol_fraction(I,Q,U,V,width_native,tsamp,n_t,n_f,freq_test,normalize=False,buff=0)
    assert(np.abs(avg) < 1e-10)
    assert(np.all(np.abs(pol_f) < 1e-10))
    assert(np.all(np.abs(pol_t) < 1e-10))
    assert(len(pol_f)==nchans)
    assert(len(pol_t)==nsamples)

    #unpolarized, pulse in time, not normalzed

    nchans = 100
    nsamples = 200
    tsamp=1
    w = 10
    width_native = w*tsamp/(256e-6)
    n_t = 1
    n_f = 1
    n_off = 30
    timestart = 50 - w//2
    timestop = 50 + w//2


    freq_test = [np.linspace(1300,1500,nchans)]*4


    I = np.zeros((nchans,nsamples))
    Q = np.zeros((nchans,nsamples))
    U = np.zeros((nchans,nsamples))
    V = np.zeros((nchans,nsamples))
    I[:,timestart:timestop] = np.ones((nchans,w))

    (pol_f,pol_t,avg) = dsapol.get_pol_fraction(I,Q,U,V,width_native,tsamp,n_t,n_f,freq_test,normalize=False,buff=0)
    assert(np.abs(avg) < 1e-10)
    assert(np.all(np.abs(pol_f) < 1e-10))
    assert(np.all(np.abs(pol_t[timestart:timestop]) < 1e-10))
    assert(len(pol_f)==nchans)
    assert(len(pol_t)==nsamples)

    print("Ideal unpolarized test successful!")

    print("Done!")
    return

#ideal PF, 50% polarized
def test_PF_ideal_test_2():
    print("Testing ideal 50% polarized source...")
    #50% polarized, continuous in time, not normalized

    nchans = 100
    nsamples = 200
    tsamp=1
    w = 10
    width_native = -1#w*tsamp/(256e-6)
    n_t = 1
    n_f = 1
    n_off = -1

    true_pol = 0.5

    freq_test = [np.linspace(1300,1500,nchans)]*4

    I = np.ones((nchans,nsamples))
    Q = U = V = np.sqrt((true_pol**2)/3)*np.ones((nchans,nsamples))

    (pol_f,pol_t,avg) = dsapol.get_pol_fraction(I,Q,U,V,width_native,tsamp,n_t,n_f,freq_test,normalize=False,buff=0)
    assert(np.abs(avg - true_pol) < 1e-10)
    assert(np.all(np.abs(pol_f - true_pol) < 1e-10))
    assert(np.all(np.abs(pol_t - true_pol) < 1e-10))
    assert(len(pol_f)==nchans)
    assert(len(pol_t)==nsamples)

    #50% polarized, pulse in time, not normalized

    nchans = 100
    nsamples = 200
    tsamp=1
    w = 10
    width_native = w*tsamp/(256e-6)
    n_t = 1
    n_f = 1
    n_off = 30
    timestart = 50 - w//2
    timestop = 50 + w//2

    true_pol = 0.5

    freq_test = [np.linspace(1300,1500,nchans)]*4

    I = np.zeros((nchans,nsamples))
    Q = np.zeros((nchans,nsamples))
    U = np.zeros((nchans,nsamples))
    V = np.zeros((nchans,nsamples))
    I[:,timestart:timestop] = np.ones((nchans,w))
    Q[:,timestart:timestop] = np.sqrt((true_pol**2)/3)*np.ones((nchans,w))
    U[:,timestart:timestop] = np.sqrt((true_pol**2)/3)*np.ones((nchans,w))
    V[:,timestart:timestop] = np.sqrt((true_pol**2)/3)*np.ones((nchans,w))

    (pol_f,pol_t,avg) = dsapol.get_pol_fraction(I,Q,U,V,width_native,tsamp,n_t,n_f,freq_test,normalize=False,buff=0)
    assert(np.abs(avg - true_pol) < 1e-10)
    assert(np.all(np.abs(pol_f - true_pol) < 1e-10))
    assert(np.all(np.abs(pol_t[timestart:timestop] - true_pol) < 1e-10))
    assert(len(pol_f)==nchans)
    assert(len(pol_t)==nsamples)

    print("Ideal 50% test successful!")

    print("Done!")

    return

#ideal PF, 100% polarized
def test_PF_ideal_test_3():
    print("Testing ideal 100% polarized source...")
    #100% polarized, continuous in time, not normalized

    nchans = 100
    nsamples = 200
    tsamp=1
    w = 10
    width_native = -1#w*tsamp/(256e-6)
    n_t = 1
    n_f = 1
    n_off = -1

    true_pol = 1

    freq_test = [np.linspace(1300,1500,nchans)]*4

    I = np.ones((nchans,nsamples))
    Q = U = V = np.sqrt((true_pol**2)/3)*np.ones((nchans,nsamples))

    (pol_f,pol_t,avg) = dsapol.get_pol_fraction(I,Q,U,V,width_native,tsamp,n_t,n_f,freq_test,normalize=False,buff=0)
    assert(np.abs(avg - true_pol) < 1e-10)
    assert(np.all(np.abs(pol_f - true_pol) < 1e-10))
    assert(np.all(np.abs(pol_t - true_pol) < 1e-10))
    assert(len(pol_f)==nchans)
    assert(len(pol_t)==nsamples)

    #100% polarized, pulse in time, not normalized

    nchans = 100
    nsamples = 200
    tsamp=1
    w = 10
    width_native = w*tsamp/(256e-6)
    n_t = 1
    n_f = 1
    n_off = 30
    timestart = 50 - w//2
    timestop = 50 + w//2

    true_pol = 1

    freq_test = [np.linspace(1300,1500,nchans)]*4

    I = np.zeros((nchans,nsamples))
    Q = np.zeros((nchans,nsamples))
    U = np.zeros((nchans,nsamples))
    V = np.zeros((nchans,nsamples))
    I[:,timestart:timestop] = np.ones((nchans,w))
    Q[:,timestart:timestop] = np.sqrt((true_pol**2)/3)*np.ones((nchans,w))
    U[:,timestart:timestop] = np.sqrt((true_pol**2)/3)*np.ones((nchans,w))
    V[:,timestart:timestop] = np.sqrt((true_pol**2)/3)*np.ones((nchans,w))

    (pol_f,pol_t,avg) = dsapol.get_pol_fraction(I,Q,U,V,width_native,tsamp,n_t,n_f,freq_test,normalize=False,buff=0)
    assert(np.abs(avg - true_pol) < 1e-10)
    assert(np.all(np.abs(pol_f - true_pol) < 1e-10))
    assert(np.all(np.abs(pol_t[timestart:timestop] - true_pol) < 1e-10))
    assert(len(pol_f)==nchans)
    assert(len(pol_t)==nsamples)

    print("Ideal 100% test successful!")

    print("Done!")
    return

#noisy PF, unpolarized
def test_PF_noisy_test_1():
    print("Testing noisy unpolarized source...")
    trial_sigmas = 1/(10**np.arange(1,5))
    for sigma_noise in trial_sigmas:
        print("Trial SNR = " + str(1/sigma_noise))
        #unpolarized, pulse in time, not normalzed

        nchans = 100
        nsamples = 200
        tsamp=1
        w = 10
        width_native = w*tsamp/(256e-6)
        n_t = 1
        n_f = 1
        n_off = 30
        timestart = 50 - w//2
        timestop = 50 + w//2
        err_max = np.sqrt(2*3)*((sigma_noise/np.sqrt(nchans))**2)


        freq_test = [np.linspace(1300,1500,nchans)]*4


        I = np.zeros((nchans,nsamples))
        Q = np.zeros((nchans,nsamples))
        U = np.zeros((nchans,nsamples))
        V = np.zeros((nchans,nsamples))
        I[:,timestart:timestop] = np.ones((nchans,w))


        I += np.random.normal(0,sigma_noise,(nchans,nsamples))
        Q += np.random.normal(0,sigma_noise,(nchans,nsamples))
        U += np.random.normal(0,sigma_noise,(nchans,nsamples))
        V += np.random.normal(0,sigma_noise,(nchans,nsamples))

        (pol_f,pol_t,avg) = dsapol.get_pol_fraction(I,Q,U,V,width_native,tsamp,n_t,n_f,freq_test,n_off=n_off,normalize=False,buff=0)
        """
        plt.figure()
        plt.plot(pol_t)
        plt.xlim(timestart,timestop)
        plt.ylim(0,1)
        plt.figure()
        plt.plot(pol_f)
        plt.ylim(0,1)
        """
        assert(len(pol_f)==nchans)
        assert(len(pol_t)==nsamples)
        assert(np.all(np.abs(pol_t[timestart:timestop] - avg) < 10*np.std(pol_t[timestart:timestop])))
        assert(np.all(np.abs(pol_f - avg) < 10*np.std(pol_f)))

        #just an estimate, no closed form solution for distribution of pol fraction to compare to
        assert(np.abs(avg) < sigma_noise ) 
        assert(np.all(np.abs(pol_t[timestart:timestop]) < sigma_noise))
        assert(np.all(np.abs(pol_f) < 10*sigma_noise))

        #normalized
        (pol_f,pol_t,avg) = dsapol.get_pol_fraction(I,Q,U,V,width_native,tsamp,n_t,n_f,freq_test,n_off=n_off,normalize=True,buff=0)
        """
        plt.figure()
        plt.plot(pol_t)
        plt.xlim(timestart,timestop)
        plt.ylim(0,1)
        plt.figure()
        plt.plot(pol_f)
        plt.ylim(0,1)
        """
        assert(len(pol_f)==nchans)
        assert(len(pol_t)==nsamples)
        assert(np.all(np.abs(pol_t[timestart:timestop] - avg) < 10*np.std(pol_t[timestart:timestop])))
        assert(np.all(np.abs(pol_f - avg) < 10*np.std(pol_f)))

        #just an estimate, no closed form solution for distribution of pol fraction to compare to
        assert(np.abs(avg) < 10) 
        assert(np.all(np.abs(pol_t[timestart:timestop]) < 10))
        assert(np.all(np.abs(pol_f) < 10))

    print("Noisy unpolarized test successful!")

    print("Done!")
    return

#noisy PF, 50% polarized
def test_PF_noisy_test_2():
    print("Testing noisy 50% source...")
    trial_sigmas = 1/(10**np.arange(1,5))
    for sigma_noise in trial_sigmas:
        print("Trial SNR = " + str(1/sigma_noise))
        #50% polarized, pulse in time, not normalzed


        nchans = 100
        nsamples = 200
        tsamp=1
        w = 10
        width_native = w*tsamp/(256e-6)
        n_t = 1
        n_f = 1
        n_off = 30
        timestart = 50 - w//2
        timestop = 50 + w//2

        true_pol = 0.5

        freq_test = [np.linspace(1300,1500,nchans)]*4

        I = np.zeros((nchans,nsamples))
        Q = np.zeros((nchans,nsamples))
        U = np.zeros((nchans,nsamples))
        V = np.zeros((nchans,nsamples))
        I[:,timestart:timestop] = np.ones((nchans,w))
        Q[:,timestart:timestop] = np.sqrt((true_pol**2)/3)*np.ones((nchans,w))
        U[:,timestart:timestop] = np.sqrt((true_pol**2)/3)*np.ones((nchans,w))
        V[:,timestart:timestop] = np.sqrt((true_pol**2)/3)*np.ones((nchans,w))


        I += np.random.normal(0,sigma_noise,(nchans,nsamples))
        Q += np.random.normal(0,sigma_noise,(nchans,nsamples))
        U += np.random.normal(0,sigma_noise,(nchans,nsamples))
        V += np.random.normal(0,sigma_noise,(nchans,nsamples))

        (pol_f,pol_t,avg) = dsapol.get_pol_fraction(I,Q,U,V,width_native,tsamp,n_t,n_f,freq_test,n_off=n_off,normalize=False,buff=0)
        """
        plt.figure()
        plt.plot(pol_t)
        plt.xlim(timestart,timestop)
        plt.ylim(0,1)
        plt.figure()
        plt.plot(pol_f)
        plt.ylim(0,1)
        """
        assert(len(pol_f)==nchans)
        assert(len(pol_t)==nsamples)
        assert(np.all(np.abs(pol_t[timestart:timestop] - avg) < 10*np.std(pol_t[timestart:timestop])))
        assert(np.all(np.abs(pol_f - avg) < 10*np.std(pol_f)))

        #just an estimate, no closed form solution for distribution of pol fraction to compare to
        #print(avg-true_pol,sigma_noise)
        assert(np.abs(avg-true_pol) < sigma_noise ) 
        assert(np.all(np.abs(pol_t[timestart:timestop] - true_pol) < sigma_noise))
        assert(np.all(np.abs(pol_f - true_pol) < 10*sigma_noise))

        #normalized
        (pol_f,pol_t,avg) = dsapol.get_pol_fraction(I,Q,U,V,width_native,tsamp,n_t,n_f,freq_test,n_off=n_off,normalize=True,buff=0)
        """
        plt.figure()
        plt.plot(pol_t)
        plt.xlim(timestart,timestop)
        plt.ylim(0,1)
        plt.figure()
        plt.plot(pol_f)
        plt.ylim(0,1)
        """
        assert(len(pol_f)==nchans)
        assert(len(pol_t)==nsamples)
        assert(np.all(np.abs(pol_t[timestart:timestop] - avg) < 10*np.std(pol_t[timestart:timestop])))
        assert(np.all(np.abs(pol_f - avg) < 10*np.std(pol_f)))

        #just an estimate, no closed form solution for distribution of pol fraction to compare to
        #print(avg-true_pol,sigma_noise/(sigma_noise/np.sqrt(nchans)))
        assert(np.abs(avg-true_pol) < 10 ) 
        assert(np.all(np.abs(pol_t[timestart:timestop] - true_pol) < 10))
        assert(np.all(np.abs(pol_f - true_pol) < 10))
    print("Noisy 50% test successful!")

    print("Done!")

    return

#noisy PF, 100% polarized
def test_PF_noisy_test_3():
    print("Testing noisy 100% source...")
    trial_sigmas = 1/(10**np.arange(1,5))
    for sigma_noise in trial_sigmas:
        print("Trial SNR = " + str(1/sigma_noise))
        #100% polarized, pulse in time, not normalzed


        nchans = 100
        nsamples = 200
        tsamp=1
        w = 10
        width_native = w*tsamp/(256e-6)
        n_t = 1
        n_f = 1
        n_off = 30
        timestart = 50 - w//2
        timestop = 50 + w//2

        true_pol = 1

        freq_test = [np.linspace(1300,1500,nchans)]*4

        I = np.zeros((nchans,nsamples))
        Q = np.zeros((nchans,nsamples))
        U = np.zeros((nchans,nsamples))
        V = np.zeros((nchans,nsamples))
        I[:,timestart:timestop] = np.ones((nchans,w))
        Q[:,timestart:timestop] = np.sqrt((true_pol**2)/3)*np.ones((nchans,w))
        U[:,timestart:timestop] = np.sqrt((true_pol**2)/3)*np.ones((nchans,w))
        V[:,timestart:timestop] = np.sqrt((true_pol**2)/3)*np.ones((nchans,w))


        I += np.random.normal(0,sigma_noise,(nchans,nsamples))
        Q += np.random.normal(0,sigma_noise,(nchans,nsamples))
        U += np.random.normal(0,sigma_noise,(nchans,nsamples))
        V += np.random.normal(0,sigma_noise,(nchans,nsamples))

        (pol_f,pol_t,avg) = dsapol.get_pol_fraction(I,Q,U,V,width_native,tsamp,n_t,n_f,freq_test,n_off=n_off,normalize=False,buff=0)
        """
        plt.figure()
        plt.plot(pol_t)
        plt.xlim(timestart,timestop)
        plt.ylim(0,1)
        plt.figure()
        plt.plot(pol_f)
        plt.ylim(0,1)
        """
        assert(len(pol_f)==nchans)
        assert(len(pol_t)==nsamples)
        #print(np.abs(pol_t[timestart:timestop] - avg),10*np.std(pol_t[timestart:timestop]))
        assert(np.all(np.abs(pol_t[timestart:timestop] - avg) < 10*np.std(pol_t[timestart:timestop])))
        assert(np.all(np.abs(pol_f - avg) < 10*np.std(pol_f)))

        #just an estimate, no closed form solution for distribution of pol fraction to compare to
        #print(avg-true_pol,sigma_noise)
        assert(np.abs(avg-true_pol) < sigma_noise ) 
        assert(np.all(np.abs(pol_t[timestart:timestop] - true_pol) < sigma_noise))
        assert(np.all(np.abs(pol_f - true_pol) < 10*sigma_noise))

        #normalized
        (pol_f,pol_t,avg) = dsapol.get_pol_fraction(I,Q,U,V,width_native,tsamp,n_t,n_f,freq_test,n_off=n_off,normalize=True,buff=0)
        """
        plt.figure()
        plt.plot(pol_t)
        plt.xlim(timestart,timestop)
        plt.ylim(0,1)
        plt.figure()
        plt.plot(pol_f)
        plt.ylim(0,1)
        """
        assert(len(pol_f)==nchans)
        assert(len(pol_t)==nsamples)
        assert(np.all(np.abs(pol_t[timestart:timestop] - avg) < 10*np.std(pol_t[timestart:timestop])))
        assert(np.all(np.abs(pol_f - avg) < 10*np.std(pol_f)))

        #just an estimate, no closed form solution for distribution of pol fraction to compare to
        #print(np.abs(pol_t[timestart:timestop] - avg),10*np.std(pol_t[timestart:timestop]))
        assert(np.abs(avg-true_pol) < 10 ) 
        assert(np.all(np.abs(pol_t[timestart:timestop] - true_pol) < 10))
        assert(np.all(np.abs(pol_f - true_pol) < 10))
    print("Noisy 100% test successful!")

    print("Done!")

    return


#ideal PA=0
def test_PA_ideal_test_1():
    print("Testing ideal PA=0  source...")

    #PA=0, continuous in time, not normalzed

    nchans = 100
    nsamples = 200
    tsamp=1
    w = 10
    width_native = -1#w*tsamp/(256e-6)
    n_t = 1
    n_f = 1
    n_off = -1


    freq_test = [np.linspace(1300,1500,nchans)]*4

    I = np.ones((nchans,nsamples))
    Q = U = V = np.zeros((nchans,nsamples))

    (pol_f,pol_t,avg) = dsapol.get_pol_angle(I,Q,U,V,width_native,tsamp,n_t,n_f,freq_test,normalize=False,buff=0)
    assert(np.abs(avg) < 1e-10)
    assert(np.all(np.abs(pol_f) < 1e-10))
    assert(np.all(np.abs(pol_t) < 1e-10))
    assert(len(pol_f)==nchans)
    assert(len(pol_t)==nsamples)

    #PA=0, pulse in time, not normalzed

    nchans = 100
    nsamples = 200
    tsamp=1
    w = 10
    width_native = w*tsamp/(256e-6)
    n_t = 1
    n_f = 1
    n_off = 30
    timestart = 50 - w//2
    timestop = 50 + w//2


    freq_test = [np.linspace(1300,1500,nchans)]*4


    I = np.zeros((nchans,nsamples))
    Q = np.zeros((nchans,nsamples))
    U = np.zeros((nchans,nsamples))
    V = np.zeros((nchans,nsamples))
    I[:,timestart:timestop] = np.ones((nchans,w))

    (pol_f,pol_t,avg) = dsapol.get_pol_angle(I,Q,U,V,width_native,tsamp,n_t,n_f,freq_test,normalize=False,buff=0)
    assert(np.abs(avg) < 1e-10)
    assert(np.all(np.abs(pol_f) < 1e-10))
    assert(np.all(np.abs(pol_t[timestart:timestop]) < 1e-10))
    assert(len(pol_f)==nchans)
    assert(len(pol_t)==nsamples)
    print("Ideal PA=0 test successful!")

    print("Done!")

    return

#ideal PA=pi/4
def test_PA_ideal_test_2():
    print("Testing ideal PA=pi/4  source...")
    #PA=pi/4, continuous in time, not normalized

    nchans = 100
    nsamples = 200
    tsamp=1
    w = 10
    width_native = -1#w*tsamp/(256e-6)
    n_t = 1
    n_f = 1
    n_off = -1

    true_PA = np.pi/4
    true_pol = 1

    freq_test = [np.linspace(1300,1500,nchans)]*4

    I = np.ones((nchans,nsamples))
    Q = np.sqrt((true_pol**2)/3)*np.ones((nchans,nsamples))
    U = np.sqrt((true_pol**2)/3)*np.ones((nchans,nsamples))
    V = np.sqrt((true_pol**2)/3)*np.ones((nchans,nsamples))

    (pol_f,pol_t,avg) = dsapol.get_pol_angle(I,Q,U,V,width_native,tsamp,n_t,n_f,freq_test,normalize=False,buff=0)
    assert(np.abs(avg - true_PA) < 1e-10)
    assert(np.all(np.abs(pol_f - true_PA) < 1e-10))
    assert(np.all(np.abs(pol_t - true_PA) < 1e-10))
    assert(len(pol_f)==nchans)
    assert(len(pol_t)==nsamples)

    #PA=pi/4, pulse in time, not normalized

    nchans = 100
    nsamples = 200
    tsamp=1
    w = 10
    width_native = w*tsamp/(256e-6)
    n_t = 1
    n_f = 1
    n_off = 30
    timestart = 50 - w//2
    timestop = 50 + w//2

    true_PA = np.pi/4
    true_pol = 1

    freq_test = [np.linspace(1300,1500,nchans)]*4

    I = np.zeros((nchans,nsamples))
    Q = np.zeros((nchans,nsamples))
    U = np.zeros((nchans,nsamples))
    V = np.zeros((nchans,nsamples))
    I[:,timestart:timestop] = np.ones((nchans,w))
    Q[:,timestart:timestop] = np.sqrt((true_pol**2)/3)*np.ones((nchans,w))
    U[:,timestart:timestop] = np.sqrt((true_pol**2)/3)*np.ones((nchans,w))
    V[:,timestart:timestop] = np.sqrt((true_pol**2)/3)*np.ones((nchans,w))

    (pol_f,pol_t,avg) = dsapol.get_pol_angle(I,Q,U,V,width_native,tsamp,n_t,n_f,freq_test,normalize=False,buff=0)
    assert(np.abs(avg - true_PA) < 1e-10)
    assert(np.all(np.abs(pol_f - true_PA) < 1e-10))
    assert(np.all(np.abs(pol_t[timestart:timestop] - true_PA) < 1e-10))
    assert(len(pol_f)==nchans)
    assert(len(pol_t)==nsamples)
    print("Ideal PA=pi/4 test successful!")

    print("Done!")
    
    return

#ideal PA=pi/2
def test_PA_ideal_test_3():
    print("Testing ideal PA=pi/2  source...")
    #PA=pi/2, continuous in time, not normalized

    nchans = 100
    nsamples = 200
    tsamp=1
    w = 10
    width_native = -1#w*tsamp/(256e-6)
    n_t = 1
    n_f = 1
    n_off = -1

    true_PA = np.pi/2
    true_pol = 1

    freq_test = [np.linspace(1300,1500,nchans)]*4

    I = np.ones((nchans,nsamples))
    Q = np.zeros((nchans,nsamples))
    U = np.sqrt((true_pol**2)/2)*np.ones((nchans,nsamples))
    V = np.sqrt((true_pol**2)/2)*np.ones((nchans,nsamples))

    (pol_f,pol_t,avg) = dsapol.get_pol_angle(I,Q,U,V,width_native,tsamp,n_t,n_f,freq_test,normalize=False,buff=0)
    assert(np.abs(avg - true_PA) < 1e-10)
    assert(np.all(np.abs(pol_f - true_PA) < 1e-10))
    assert(np.all(np.abs(pol_t - true_PA) < 1e-10))
    assert(len(pol_f)==nchans)
    assert(len(pol_t)==nsamples)
    
    #PA=pi/2, pulse in time, not normalized

    nchans = 100
    nsamples = 200
    tsamp=1
    w = 10
    width_native = w*tsamp/(256e-6)
    n_t = 1
    n_f = 1
    n_off = 30
    timestart = 50 - w//2
    timestop = 50 + w//2

    true_PA = np.pi/2
    true_pol = 1

    freq_test = [np.linspace(1300,1500,nchans)]*4

    I = np.zeros((nchans,nsamples))
    Q = np.zeros((nchans,nsamples))
    U = np.zeros((nchans,nsamples))
    V = np.zeros((nchans,nsamples))
    I[:,timestart:timestop] = np.ones((nchans,w))
    Q[:,timestart:timestop] = np.zeros((nchans,w))
    U[:,timestart:timestop] = np.sqrt((true_pol**2)/2)*np.ones((nchans,w))
    V[:,timestart:timestop] = np.sqrt((true_pol**2)/2)*np.ones((nchans,w))

    (pol_f,pol_t,avg) = dsapol.get_pol_angle(I,Q,U,V,width_native,tsamp,n_t,n_f,freq_test,normalize=False,buff=0)
    assert(np.abs(avg - true_PA) < 1e-10)
    assert(np.all(np.abs(pol_f - true_PA) < 1e-10))
    assert(np.all(np.abs(pol_t[timestart:timestop] - true_PA) < 1e-10))
    assert(len(pol_f)==nchans)
    assert(len(pol_t)==nsamples)

    print("Ideal PA=pi/2 test successful!")

    print("Done!")
    return

#noisy PA=pi/4
def test_PA_noisy_test_1():
    print("Testing noisy PA=pi/4  source...")
    trial_sigmas = 1/(10**np.arange(1,5))
    for sigma_noise in trial_sigmas:
        print("Trial SNR = " + str(1/sigma_noise))
        #PA=pi/4, pulse in time, not normalzed

    
        nchans = 100
        nsamples = 200
        tsamp=1
        w = 10
        width_native = w*tsamp/(256e-6)
        n_t = 1
        n_f = 1
        n_off = 30
        timestart = 50 - w//2
        timestop = 50 + w//2

        true_PA = np.pi/4
        true_pol = 1

        freq_test = [np.linspace(1300,1500,nchans)]*4

        I = np.zeros((nchans,nsamples))
        Q = np.zeros((nchans,nsamples))
        U = np.zeros((nchans,nsamples))
        V = np.zeros((nchans,nsamples))
        I[:,timestart:timestop] = np.ones((nchans,w))
        Q[:,timestart:timestop] = np.sqrt((true_pol**2)/3)*np.ones((nchans,w))
        U[:,timestart:timestop] = np.sqrt((true_pol**2)/3)*np.ones((nchans,w))
        V[:,timestart:timestop] = np.sqrt((true_pol**2)/3)*np.ones((nchans,w))


        I += np.random.normal(0,sigma_noise,(nchans,nsamples))
        Q += np.random.normal(0,sigma_noise,(nchans,nsamples))
        U += np.random.normal(0,sigma_noise,(nchans,nsamples))
        V += np.random.normal(0,sigma_noise,(nchans,nsamples))

        (pol_f,pol_t,avg) = dsapol.get_pol_angle(I,Q,U,V,width_native,tsamp,n_t,n_f,freq_test,n_off=n_off,normalize=False,buff=0)
        """
        plt.figure()
        plt.plot(pol_t)
        plt.xlim(timestart,timestop)
        plt.ylim(0,1)
        plt.figure()
        plt.plot(pol_f)
        plt.ylim(0,1)
        """
        assert(len(pol_f)==nchans)
        assert(len(pol_t)==nsamples)
        assert(np.all(np.abs(pol_t[timestart:timestop] - avg) < 10*np.std(pol_t[timestart:timestop])))
        assert(np.all(np.abs(pol_f - avg) < 10*np.std(pol_f)))
    
        #just an estimate, no closed form solution for distribution of pol fraction to compare to
        #print(avg-true_pol,sigma_noise)
        assert(np.abs(avg-true_PA) < sigma_noise ) 
        assert(np.all(np.abs(pol_t[timestart:timestop] - true_PA) < sigma_noise))
        assert(np.all(np.abs(pol_f - true_PA) < 10*sigma_noise))
    
        #normalized
        (pol_f,pol_t,avg) = dsapol.get_pol_angle(I,Q,U,V,width_native,tsamp,n_t,n_f,freq_test,n_off=n_off,normalize=True,buff=0)
        """
        plt.figure()
        plt.plot(pol_t)
        plt.xlim(timestart,timestop)
        plt.ylim(0,1)
        plt.figure()
        plt.plot(pol_f)
        plt.ylim(0,1)
        """
        assert(len(pol_f)==nchans)
        assert(len(pol_t)==nsamples)
        assert(np.all(np.abs(pol_t[timestart:timestop] - avg) < 10*np.std(pol_t[timestart:timestop])))
        assert(np.all(np.abs(pol_f - avg) < 10*np.std(pol_f)))
    
        #just an estimate, no closed form solution for distribution of pol fraction to compare to
        #print(avg-true_pol,sigma_noise/(sigma_noise/np.sqrt(nchans)))
        assert(np.abs(avg-true_PA) < 10 ) 
        assert(np.all(np.abs(pol_t[timestart:timestop] - true_PA) < 10))
        assert(np.all(np.abs(pol_f - true_PA) < 10))
    print("Noisy PA=pi/4 test successful!")

    print("Done!")

    return

#noisy PA=pi/2
def test_PA_noisy_test_2():
    print("Testing noisy PA=pi/2  source...")
    trial_sigmas = 1/(10**np.arange(1,5))
    for sigma_noise in trial_sigmas:
        print("Trial SNR = " + str(1/sigma_noise))
        #PA=pi/2, pulse in time, not normalzed

    
        nchans = 100
        nsamples = 200
        tsamp=1
        w = 10
        width_native = w*tsamp/(256e-6)
        n_t = 1
        n_f = 1
        n_off = 30
        timestart = 50 - w//2
        timestop = 50 + w//2

        true_PA = np.pi/2
        true_pol = 1

        freq_test = [np.linspace(1300,1500,nchans)]*4

        I = np.zeros((nchans,nsamples))
        Q = np.zeros((nchans,nsamples))
        U = np.zeros((nchans,nsamples))
        V = np.zeros((nchans,nsamples))
        I[:,timestart:timestop] = np.ones((nchans,w))
        Q[:,timestart:timestop] = np.zeros((nchans,w))
        U[:,timestart:timestop] = np.sqrt((true_pol**2)/2)*np.ones((nchans,w))
        V[:,timestart:timestop] = np.sqrt((true_pol**2)/2)*np.ones((nchans,w))


        I += np.random.normal(0,sigma_noise,(nchans,nsamples))
        Q += np.random.normal(0,sigma_noise,(nchans,nsamples))
        U += np.random.normal(0,sigma_noise,(nchans,nsamples))
        V += np.random.normal(0,sigma_noise,(nchans,nsamples))

        (pol_f,pol_t,avg) = dsapol.get_pol_angle(I,Q,U,V,width_native,tsamp,n_t,n_f,freq_test,n_off=n_off,normalize=False,buff=0)
        """
        plt.figure()
        plt.plot(pol_t)
        plt.xlim(timestart,timestop)
        plt.ylim(0,1)
        plt.figure()
        plt.plot(pol_f)
        plt.ylim(0,1)
        """
        assert(len(pol_f)==nchans)
        assert(len(pol_t)==nsamples)
        assert(np.all(np.abs(pol_t[timestart:timestop] - avg) < 10*np.std(pol_t[timestart:timestop])))
        assert(np.all(np.abs(pol_f - avg) < 10*np.std(pol_f)))
    
        #just an estimate, no closed form solution for distribution of pol fraction to compare to
        #print(avg-true_pol,sigma_noise)
        assert(np.abs(avg-true_PA) < sigma_noise ) 
        assert(np.all(np.abs(pol_t[timestart:timestop] - true_PA) < sigma_noise))
        assert(np.all(np.abs(pol_f - true_PA) < 10*sigma_noise))
    
        #normalized
        (pol_f,pol_t,avg) = dsapol.get_pol_angle(I,Q,U,V,width_native,tsamp,n_t,n_f,freq_test,n_off=n_off,normalize=True,buff=0)
        """
        plt.figure()
        plt.plot(pol_t)
        plt.xlim(timestart,timestop)
        plt.ylim(0,1)
        plt.figure()
        plt.plot(pol_f)
        plt.ylim(0,1)
        """
        assert(len(pol_f)==nchans)
        assert(len(pol_t)==nsamples)
        assert(np.all(np.abs(pol_t[timestart:timestop] - avg) < 10*np.std(pol_t[timestart:timestop])))
        assert(np.all(np.abs(pol_f - avg) < 10*np.std(pol_f)))
    
        #just an estimate, no closed form solution for distribution of pol fraction to compare to
        #print(avg-true_pol,sigma_noise/(sigma_noise/np.sqrt(nchans)))
        assert(np.abs(avg-true_PA) < 10 ) 
        assert(np.all(np.abs(pol_t[timestart:timestop] - true_PA) < 10))
        assert(np.all(np.abs(pol_f - true_PA) < 10))

    print("Noisy PA=pi/2 test successful!")

    print("Done!")
    return

def main():
    print("Running polarization and PA tests...")
    PF_ideal_test_1()
    PF_ideal_test_2()
    PF_ideal_test_3()
    PF_noisy_test_1()
    PF_noisy_test_2()
    PF_noisy_test_3()

    PA_ideal_test_1()
    PA_ideal_test_2()
    PA_ideal_test_3()
    PA_noisy_test_1()
    PA_noisy_test_2()

    print("All Tests Passed!")

if __name__ == "__main__":
    main()
