from dsapol import dsapol
import numpy as np

#Gain/Phase calibration pipeline test
def test_caltest_1():
    print("Testing gain and phase cal pipeline...")
    datadir = '/home/ubuntu/sherman/scratch_weights_update_2022-06-03/3C48/'
    source_name = '3C48'
    obs_names = ['ane']
    n_t = 1
    n_f = 1
    nsamps = 5120
    deg = 10
    average = False
    fn_prefix= '3C48ane_dev'
    suffix = '_dev'
    (ratio,ratio_params) = dsapol.gaincal_full(datadir,source_name,obs_names,n_t,n_f,nsamps,deg,suffix,average)
    assert(len(ratio_params)==deg+1)
    obs_names = ['ane','bzj','kkz']
    (ratio,ratio_params) = dsapol.gaincal_full(datadir,source_name,obs_names,n_t,n_f,nsamps,deg,suffix,average)
    assert(len(ratio) == len(ratio_params) == len(obs_names))
    assert(len(ratio_params[0])==len(ratio_params[1])==len(ratio_params[2])==deg+1)
    average = True
    (ratio,ratio_params) = dsapol.gaincal_full(datadir,source_name,obs_names,n_t,n_f,nsamps,deg,suffix,average)
    assert(len(ratio_params)==deg+1)
    print("Gain cal successful!")

    datadir = '/home/ubuntu/sherman/scratch_weights_update_2022-06-03/3C286/'
    source_name = '3C286'
    obs_names = ['jqc']
    n_t = 1
    n_f = 1
    nsamps = 5120
    deg = 10
    average = False
    fn_prefix= '3C286jqc_dev'

    (phase,phase_params) = dsapol.phasecal_full(datadir,source_name,obs_names,n_t,n_f,nsamps,deg,suffix,average)
    assert(len(phase_params)==deg+1)
    obs_names = ['jqc','oxr','vwy','kus','dnz']
    (phase,phase_params) = dsapol.gaincal_full(datadir,source_name,obs_names,n_t,n_f,nsamps,deg,suffix,average)
    assert(len(phase) == len(phase_params) == len(obs_names))
    assert(len(phase_params[0])==len(phase_params[1])==len(phase_params[2])==len(phase_params[3])==len(phase_params[4])==deg+1)
    average = True
    (phase,phase_params) = dsapol.gaincal_full(datadir,source_name,obs_names,n_t,n_f,nsamps,deg,suffix,average)
    assert(len(phase_params)==deg+1)
    print("Phase cal successful!")
    print("Done.")
    return


#Test gain calibration recovery for ideal calibrator
def test_caltest_gaincal():
    print("Testing gain calibration...")
    nchans = 100
    nsamples = 200
    n_t = 1
    n_f = 1
    deg = 10
    I_true = np.ones((nchans,nsamples))
    Q_true = U_true = V_true = np.zeros((nchans,nsamples))
    suffix = '_dev'
    #constant ratio
    ratio_true = 20*np.ones(nchans)
    true_params = np.polyfit(np.arange(nchans),np.nan_to_num(ratio_true,nan=np.nanmedian(ratio_true)),deg=deg)

    gxx = ratio_true*np.ones(nchans)
    gxx_all = np.transpose(np.tile(gxx,(nsamples,1)))
    gyy = np.ones(nchans)
    gyy_all = np.transpose(np.tile(gyy,(nsamples,1)))

    I_obs = 0.5*(   (np.abs(gxx_all)**2)*(I_true + Q_true )     +      (np.abs(gyy_all)**2)*(I_true  - Q_true ))
    Q_obs = 0.5*(   (np.abs(gxx_all)**2)*(I_true + Q_true )     -      (np.abs(gyy_all)**2)*(I_true  - Q_true ))
    U_obs = np.zeros((nchans,nsamples))
    V_obs = np.zeros((nchans,nsamples))


    (I_f,Q_f,U_f,V_f) = dsapol.get_stokes_vs_freq(I_obs,Q_obs,U_obs,V_obs,-1,1,1,[np.arange(nchans)]*4)
    (ratio,ratio_params) = dsapol.gaincal(I_f,Q_f,U_f,V_f,[np.arange(nchans)]*4,stokes=True,deg=10)
    assert(np.all(ratio == ratio_true))
    assert(np.abs(ratio_params[-1] - 20.0) < 1e-10)
    assert(np.all(np.abs(ratio_params[:-1]) < 1e-10))

    calmatrix = dsapol.get_calmatrix_from_ratio_phasediff(ratio,np.zeros(nchans))
    assert(np.all(calmatrix[0] == gxx))
    assert(np.all(calmatrix[1] == gyy))
    (I_cal,Q_cal,U_cal,V_cal) = dsapol.calibrate(I_obs,Q_obs,U_obs,V_obs,calmatrix)
    assert(np.all(np.abs(I_cal - I_true) < 1e-10))
    assert(np.all(np.abs(Q_cal - Q_true) < 1e-10))
    assert(np.all(np.abs(U_cal - U_true) < 1e-10))
    assert(np.all(np.abs(V_cal - V_true) < 1e-10))
    print("Gain calibration recovery successful!")
    print("Done!")
    return

#Test phase calibration recovery for ideal calibrator
def test_caltest_phasecal():
    print("Testing phase calibration...")
    #Test for ideal calibrators
    nchans = 100
    nsamples = 200
    n_t = 1
    n_f = 1
    deg = 10
    I_true = np.ones((nchans,nsamples))
    Q_true = U_true = (1/np.sqrt(2))*np.ones((nchans,nsamples))
    V_true = np.zeros((nchans,nsamples))
    suffix = '_dev'

    #constant ratio
    phase_true = (np.pi/4)*np.ones(nchans)
    true_params = np.polyfit(np.arange(nchans),np.nan_to_num(phase_true,nan=np.nanmedian(phase_true)),deg=deg)

    gxx = np.ones(nchans)*np.exp(1j*phase_true)
    gxx_all = np.transpose(np.tile(gxx,(nsamples,1)))
    gyy = np.ones(nchans)
    gyy_all = np.transpose(np.tile(gyy,(nsamples,1)))

    I_obs = 0.5*(   (np.abs(gxx_all)**2)*(I_true + Q_true )     +      (np.abs(gyy_all)**2)*(I_true  - Q_true ))
    Q_obs = 0.5*(   (np.abs(gxx_all)**2)*(I_true + Q_true )     -      (np.abs(gyy_all)**2)*(I_true  - Q_true ))
    U_obs = np.real(gxx_all*np.conj(gyy_all)*(U_true + 1j*V_true))
    V_obs = np.imag(gxx_all*np.conj(gyy_all)*(U_true + 1j*V_true))


    (I_f,Q_f,U_f,V_f) = dsapol.get_stokes_vs_freq(I_obs,Q_obs,U_obs,V_obs,-1,1,1,[np.arange(nchans)]*4)
    (phase,phase_params) = dsapol.phasecal(I_f,Q_f,U_f,V_f,[np.arange(nchans)]*4,stokes=True,deg=10)
    assert(np.all(phase == phase_true))
    assert(np.abs(phase_params[-1] - np.pi/4) < 1e-10)
    assert(np.all(np.abs(phase_params[:-1]) < 1e-10))

    calmatrix = dsapol.get_calmatrix_from_ratio_phasediff(np.ones(nchans),phase)
    assert(np.all(calmatrix[0] == gxx))
    assert(np.all(calmatrix[1] == gyy))
    (I_cal,Q_cal,U_cal,V_cal) = dsapol.calibrate(I_obs,Q_obs,U_obs,V_obs,calmatrix)
    assert(np.all(np.abs(I_cal - I_true) < 1e-10))
    assert(np.all(np.abs(Q_cal - Q_true) < 1e-10))
    assert(np.all(np.abs(U_cal - U_true) < 1e-10))
    assert(np.all(np.abs(V_cal - V_true) < 1e-10))

    print("Phase calibration recovery successful!")
    print("Done!")

    return

#Test recovery of signal from gain and phase cal():
def test_caltest_allcal():
    print("Testing gain and phase cal recovery...")
    #gain cal
    nchans = 100
    nsamples = 200
    n_t = 1
    n_f = 1
    deg = 10
    I_true = np.ones((nchans,nsamples))
    Q_true = U_true = V_true = np.zeros((nchans,nsamples))
    suffix = '_dev'
    #constant ratio
    ratio_true = 20*np.ones(nchans)
    true_params_gain = np.polyfit(np.arange(nchans),np.nan_to_num(ratio_true,nan=np.nanmedian(ratio_true)),deg=deg)

    gxx = ratio_true*np.ones(nchans)
    gxx_all = np.transpose(np.tile(gxx,(nsamples,1)))
    gyy = np.ones(nchans)
    gyy_all = np.transpose(np.tile(gyy,(nsamples,1)))

    I_obs = 0.5*(   (np.abs(gxx_all)**2)*(I_true + Q_true )     +      (np.abs(gyy_all)**2)*(I_true  - Q_true ))
    Q_obs = 0.5*(   (np.abs(gxx_all)**2)*(I_true + Q_true )     -      (np.abs(gyy_all)**2)*(I_true  - Q_true ))
    U_obs = np.zeros((nchans,nsamples))
    V_obs = np.zeros((nchans,nsamples))


    (I_f,Q_f,U_f,V_f) = dsapol.get_stokes_vs_freq(I_obs,Q_obs,U_obs,V_obs,-1,1,1,[np.arange(nchans)]*4)
    (ratio,ratio_params) = dsapol.gaincal(I_f,Q_f,U_f,V_f,[np.arange(nchans)]*4,stokes=True,deg=10)

    #phase cal
    nchans = 100
    nsamples = 200
    n_t = 1
    n_f = 1
    deg = 10
    I_true = np.ones((nchans,nsamples))
    Q_true = U_true = (1/np.sqrt(2))*np.ones((nchans,nsamples))
    V_true = np.zeros((nchans,nsamples))

    #constant ratio
    phase_true = (np.pi/4)*np.ones(nchans)
    true_params_phase = np.polyfit(np.arange(nchans),np.nan_to_num(phase_true,nan=np.nanmedian(phase_true)),deg=deg)

    gxx = np.ones(nchans)*np.exp(1j*phase_true)
    gxx_all = np.transpose(np.tile(gxx,(nsamples,1)))
    gyy = np.ones(nchans)
    gyy_all = np.transpose(np.tile(gyy,(nsamples,1)))

    I_obs = 0.5*(   (np.abs(gxx_all)**2)*(I_true + Q_true )     +      (np.abs(gyy_all)**2)*(I_true  - Q_true ))
    Q_obs = 0.5*(   (np.abs(gxx_all)**2)*(I_true + Q_true )     -      (np.abs(gyy_all)**2)*(I_true  - Q_true ))
    U_obs = np.real(gxx_all*np.conj(gyy_all)*(U_true + 1j*V_true))
    V_obs = np.imag(gxx_all*np.conj(gyy_all)*(U_true + 1j*V_true))


    (I_f,Q_f,U_f,V_f) = dsapol.get_stokes_vs_freq(I_obs,Q_obs,U_obs,V_obs,-1,1,1,[np.arange(nchans)]*4)
    (phase,phase_params) = dsapol.phasecal(I_f,Q_f,U_f,V_f,[np.arange(nchans)]*4,stokes=True,deg=10)

    #arbitrary signal
    nchans = 100
    nsamples = 200
    n_t = 1
    n_f = 1
    deg = 10
    I_true = np.ones((nchans,nsamples))
    Q_true = U_true = V_true = (1/np.sqrt(3))*np.ones((nchans,nsamples))

    #constant ratio

    gxx = ratio_true*np.exp(1j*phase_true)
    gxx_all = np.transpose(np.tile(gxx,(nsamples,1)))
    gyy = np.ones(nchans)
    gyy_all = np.transpose(np.tile(gyy,(nsamples,1)))

    I_obs = 0.5*(   (np.abs(gxx_all)**2)*(I_true + Q_true )     +      (np.abs(gyy_all)**2)*(I_true  - Q_true ))
    Q_obs = 0.5*(   (np.abs(gxx_all)**2)*(I_true + Q_true )     -      (np.abs(gyy_all)**2)*(I_true  - Q_true ))
    U_obs = np.real(gxx_all*np.conj(gyy_all)*(U_true + 1j*V_true))
    V_obs = np.imag(gxx_all*np.conj(gyy_all)*(U_true + 1j*V_true))


    (I_f,Q_f,U_f,V_f) = dsapol.get_stokes_vs_freq(I_obs,Q_obs,U_obs,V_obs,-1,1,1,[np.arange(nchans)]*4)

    calmatrix = dsapol.get_calmatrix_from_ratio_phasediff(ratio,phase)
    assert(np.all(calmatrix[0] == gxx))
    assert(np.all(calmatrix[1] == gyy))
    (I_cal,Q_cal,U_cal,V_cal) = dsapol.calibrate(I_obs,Q_obs,U_obs,V_obs,calmatrix)
    assert(np.all(np.abs(I_cal - I_true) < 1e-10))
    assert(np.all(np.abs(Q_cal - Q_true) < 1e-10))
    assert(np.all(np.abs(U_cal - U_true) < 1e-10))
    assert(np.all(np.abs(V_cal - V_true) < 1e-10))

    print("Gain and phase calibration recovery successful!")
    print("Done!")

    return

def main():
    print("Running Calibration Tests...")
    caltest_1()
    caltest_gaincal()
    caltest_phasecal()
    caltest_allcal()
    print("All Tests Passed!")

if __name__ == "__main__":
    main()
