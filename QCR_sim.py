import numpy as np
from RNG import TRS
from numba import njit

N = 42 # Number of bits per message 
BITS_TYPE = [0,1] 
NOISE = 1

BIT00 = np.array([0.,1.])
BIT10 = np.array([1.,0.])
BIT01 = np.array([1., -1.])/np.sqrt(2)
BIT11 = np.array([1., 1.])/np.sqrt(2)

@njit(cache=True)
def Alice(bit, basis, noise=0):
    nvec = np.random.normal(0,noise,(bit.shape[0],2))

    psi = np.zeros_like(nvec)

    for k , (bt, bs) in enumerate(zip(bit, basis)):
        if bt == 0 and bs == 0:
            psi[k] = BIT00
        elif bt == 1 and bs == 0:
            psi[k] = BIT10
        elif bt == 0 and bs == 1:
            psi[k] = BIT01
        elif bt == 1 and bs == 1:
            psi[k] = BIT11
    
    psi = psi + nvec
    psi /= np.sqrt((psi ** 2).sum(-1))[..., np.newaxis]
    return psi

@njit(cache=True)
def Bob(psi, basis, noise=0):
    nvec = np.random.normal(0,noise,(psi.shape[0],2))

    psi = psi+nvec
    psi /= np.sqrt((psi ** 2).sum(-1))[..., np.newaxis]

    results = np.zeros(psi.shape[0])
    for k , (photon, bs) in enumerate(zip(psi, basis)):
        if bs == 0:
            p1 = (BIT10.T@photon)**2
            bit = np.random.binomial(1, p1, 1)[0]
            results[k] = bit
        elif bs == 1:
            p1 = (BIT11.T@photon)**2
            bit = np.random.binomial(1, p1, 1)[0]
            results[k] = bit
    
    return results

def MonteCarlo(nruns, noise):
    max_len = np.zeros(nruns)
    error = np.zeros(nruns)
    for i in range(nruns):
        bits = TRS(N, BITS_TYPE)
        basisAlice = TRS(N, BITS_TYPE)
        basisBob = TRS(N, BITS_TYPE)

        psi = Alice(bits, basisAlice, noise)
        bits_rec = Bob(psi, basisBob, noise)
        del psi

        mask = basisAlice == basisBob
        key_Bob = bits_rec[mask]
        key_Alice = bits[mask]
        
        max_len[i] = np.sum(mask)
        error[i] = 1 - np.sum(key_Bob == key_Alice)/max_len[i]
    return max_len, error

NOISE_RANGE = np.linspace(0, NOISE, 20)
avg_error = np.zeros_like(NOISE_RANGE)
max_error = np.zeros_like(NOISE_RANGE)
kl = []
for k, noise in enumerate(NOISE_RANGE):
    key_length, error = MonteCarlo(200, noise)
    avg_error[k] = np.average(error)
    max_error[k] = np.max(error)
    kl.append(key_length)
kl = np.array(kl).flatten()
print(avg_error)