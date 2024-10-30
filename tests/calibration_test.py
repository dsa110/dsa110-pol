import pytest
from matplotlib import pyplot as plt
import numpy as np
from dsapol import dsapol
from scipy.stats import norm


"""
Last Updated 2024-10-28
This script contains unit tests for polarization calibrated updated
since the PARSEC V1 release.
"""

def test_calibrate():
    nchans = 6144
    err_tolerance = 1e-5
    I = np.random.random(size=(nchans,1))
    Q = np.random.random(size=(nchans,1))
    U = np.random.random(size=(nchans,1))
    V = np.random.random(size=(nchans,1))

    gxx_true = np.ones(nchans)
    gyy_true = np.ones(nchans)*10*np.exp(1j*np.pi/4)

    Iobs = np.zeros((nchans,1))
    Qobs = np.zeros((nchans,1))
    Uobs = np.zeros((nchans,1))
    Vobs = np.zeros((nchans,1))
    for i in range(nchans):
        J = np.matrix([[gxx_true[i],0],[0,gyy_true[i]]])
        C = np.matrix([[I[i,0]+Q[i,0],U[i,0]+1j*V[i,0]],[U[i,0]-1j*V[i,0],I[i,0]-Q[i,0]]])
        Cobs = np.matmul(J,np.matmul(C,np.conjugate(J.transpose())))
        Iobs[i,0] = 0.5*(Cobs[0,0] + Cobs[1,1])
        Qobs[i,0] = 0.5*(Cobs[0,0] - Cobs[1,1])
        Uobs[i,0] = 0.5*(Cobs[0,1] + Cobs[1,0])
        Vobs[i,0] = -1j*0.5*(Cobs[0,1] - Cobs[1,0])
    
    Ical,Qcal,Ucal,Vcal = dsapol.calibrate(Iobs,Qobs,Uobs,Vobs,(gxx_true,gyy_true),stokes=True,multithread=False)
    assert(np.all(np.abs(Ical-I)<err_tolerance))
    assert(np.all(np.abs(Qcal-Q)<err_tolerance))
    assert(np.all(np.abs(Ucal-U)<err_tolerance))
    assert(np.all(np.abs(Vcal-V)<err_tolerance))

    #state_dict['base_Ical'],state_dict['base_Qcal'],state_dict['base_Ucal'],state_dict['base_Vcal'] = dsapol.calibrate(state_dict['base_I'],state_dict['base_Q'],state_dict['base_U'],state_dict['base_V'],(state_dict['gxx'],state_dict['gyy']),stokes=True,multithread=True,maxProcesses=int(polcalprocs.value),bad_chans=state_dict['badchans'])


    return

def test_calibrate_multithread():
    nchans = 6144
    nsamps = 100
    err_tolerance = 1e-5
    I = np.random.random(size=(nchans,nsamps))
    Q = np.random.random(size=(nchans,nsamps))
    U = np.random.random(size=(nchans,nsamps))
    V = np.random.random(size=(nchans,nsamps))

    gxx_true = np.ones(nchans)
    gyy_true = np.ones(nchans)*10*np.exp(1j*np.pi/4)

    Iobs = np.zeros((nchans,nsamps))
    Qobs = np.zeros((nchans,nsamps))
    Uobs = np.zeros((nchans,nsamps))
    Vobs = np.zeros((nchans,nsamps))
    for i in range(nchans):
        for j in range(nsamps):
            J = np.matrix([[gxx_true[i],0],[0,gyy_true[i]]])
            C = np.matrix([[I[i,j]+Q[i,j],U[i,j]+1j*V[i,j]],[U[i,j]-1j*V[i,j],I[i,j]-Q[i,j]]])
            Cobs = np.matmul(J,np.matmul(C,np.conjugate(J.transpose())))
            Iobs[i,j] = 0.5*(Cobs[0,0] + Cobs[1,1])
            Qobs[i,j] = 0.5*(Cobs[0,0] - Cobs[1,1])
            Uobs[i,j] = 0.5*(Cobs[0,1] + Cobs[1,0])
            Vobs[i,j] = -1j*0.5*(Cobs[0,1] - Cobs[1,0])

    Ical,Qcal,Ucal,Vcal = dsapol.calibrate(Iobs,Qobs,Uobs,Vobs,(gxx_true,gyy_true),stokes=True,multithread=True,maxProcesses=10)
    assert(np.all(np.abs(Ical-I)<err_tolerance))
    assert(np.all(np.abs(Qcal-Q)<err_tolerance))
    assert(np.all(np.abs(Ucal-U)<err_tolerance))
    assert(np.all(np.abs(Vcal-V)<err_tolerance))

    return


if __name__=="__main__":
    pytest.main()
                      
