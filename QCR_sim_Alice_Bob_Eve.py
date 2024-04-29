import numpy as np
from RNG import TRS
from numba import njit
import matplotlib.pyplot as plt
from scipy.stats import norm

N = 42  # Number of bits per message 
BITS_TYPE = [0, 1] 
NOISE = 1

BIT00 = np.array([0., 1.])
BIT10 = np.array([1., 0.])
BIT01 = np.array([1., -1.]) / np.sqrt(2)
BIT11 = np.array([1., 1.]) / np.sqrt(2)

@njit(cache=True)
def Alice(bit, basis, noise=0):
    """
    Implements Alice's operation in the BB84 protocol.

    Parameters:
    bit (numpy.ndarray): Array of bits to be transmitted.
    basis (numpy.ndarray): Array of bases used for transmission.
    noise (float): Standard deviation of the Gaussian noise.

    Returns:
    numpy.ndarray: Transmitted qubits after applying noise.
    """
    nvec = np.random.normal(0, noise, (bit.shape[0], 2))

    psi = np.zeros_like(nvec)

    for k, (bt, bs) in enumerate(zip(bit, basis)):
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
def Eve(psi, basis, noise=0):
    """
    Implements Eve's operation in the BB84 protocol.

    Parameters:
    psi (numpy.ndarray): Array of qubits intercepted by Eve.
    basis (numpy.ndarray): Array of bases used for interception.
    noise (float): Standard deviation of the Gaussian noise.

    Returns:
    tuple: Tuple containing intercepted qubits and measured bits.
    """
    nvec = np.random.normal(0, noise, (psi.shape[0], 2))

    psi = psi + nvec
    psi /= np.sqrt((psi ** 2).sum(-1))[..., np.newaxis]

    bit = np.zeros(psi.shape[0])
    for k, (photon, bs) in enumerate(zip(psi, basis)):
        if bs == 0:
            p1 = (BIT10.T @ photon) ** 2
            bitp = np.random.binomial(1, p1, 1)[0]
            bit[k] = bitp
        elif bs == 1:
            p1 = (BIT11.T @ photon) ** 2
            bitp = np.random.binomial(1, p1, 1)[0]
            bit[k] = bitp
    
    nvec = np.random.normal(0, noise, (bit.shape[0], 2))
    psi = np.zeros_like(nvec)

    for k, (bt, bs) in enumerate(zip(bit, basis)):
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
    return psi, bit

@njit(cache=True)
def Bob(psi, basis, noise=0):
    """
    Implements Bob's operation in the BB84 protocol.

    Parameters:
    psi (numpy.ndarray): Array of qubits received by Bob.
    basis (numpy.ndarray): Array of bases used for reception.
    noise (float): Standard deviation of the Gaussian noise.

    Returns:
    numpy.ndarray: Measured bits by Bob.
    """
    nvec = np.random.normal(0, noise, (psi.shape[0], 2))

    psi = psi + nvec
    psi /= np.sqrt((psi ** 2).sum(-1))[..., np.newaxis]

    results = np.zeros(psi.shape[0])
    for k, (photon, bs) in enumerate(zip(psi, basis)):
        if bs == 0:
            p1 = (BIT10.T @ photon) ** 2
            bit = np.random.binomial(1, p1, 1)[0]
            results[k] = bit
        elif bs == 1:
            p1 = (BIT11.T @ photon) ** 2
            bit = np.random.binomial(1, p1, 1)[0]
            results[k] = bit
    
    return results

def MonteCarlo(nruns, noise):
    """
    Perform Monte Carlo simulations for error analysis in BB84 protocol.

    Parameters:
    nruns (int): Number of simulation runs.
    noise (float): Standard deviation of the Gaussian noise.

    Returns:
    tuple: Tuple containing error rates for Alice-Bob, Alice-Eve, and Bob-Eve communication.
    """
    errorAlice_Bob = np.zeros(nruns)
    errorAlice_Eve = np.zeros(nruns)
    errorBob_Eve = np.zeros(nruns)
    for i in range(nruns):
        bits = TRS(N, BITS_TYPE)
        basisAlice = TRS(N, BITS_TYPE)
        basisEve = TRS(N, BITS_TYPE)
        basisBob = TRS(N, BITS_TYPE)

        psi = Alice(bits, basisAlice, noise)
        psi, bits_eve = Eve(psi, basisEve, noise)
        bits_rec = Bob(psi, basisBob, noise)
        del psi

        mask = basisAlice == basisBob
        key_Bob = bits_rec[mask]
        key_Alice = bits[mask]
        key_Eve = bits_eve[mask]
        
        errorAlice_Bob[i] = np.abs(key_Alice - key_Bob).sum() / len(key_Alice)
        errorAlice_Eve[i] = np.abs(key_Alice - key_Eve).sum() / len(key_Alice)
        errorBob_Eve[i] = np.abs(key_Bob - key_Eve).sum() / len(key_Alice)

    return errorAlice_Bob, errorAlice_Eve, errorBob_Eve

eAB, eAE, eBE = MonteCarlo(1000, 0)

fig, ax = plt.subplots()

ax.hist(eAB, color='orange', bins=42 // 2, histtype='step', density=True, label='Alice-Bob')
ax.hist(eAE, color='blue', bins=42 // 2, histtype='step', density=True, label='Alice-Eve')
ax.hist(eBE, color='green', bins=42 // 2, histtype='step', density=True, label='Bob-Eve')

data = np.array([eAB, eAE, eBE]).flatten()
(mu, sigma) = norm.fit(data)
print(f'Noise = 0, mu={mu:.4f}, sigma={sigma:.4f}')
dummyx = np.linspace(np.min(eAB), np.max(eAB), 100)
ax.plot(dummyx, norm.pdf(dummyx, loc=mu, scale=sigma), 'k--', label='Gaussian fit')

ax.set_title('Error in the key distribution, noise = 0')
ax.set(xlabel='Error', ylabel='Probability (normalized)')
ax.legend()
plt.savefig('Distribution_Alice_Bob_Eve_error_noise=0.png', dpi=400)

eAB, eAE, eBE = MonteCarlo(1000, 0.2)

fig, ax = plt.subplots()
ax.hist(eAB, color='orange', bins=42 // 2, histtype='step', density=True, label='Alice-Bob')
ax.hist(eAE, color='blue', bins=42 // 2, histtype='step', density=True, label='Alice-Eve')
ax.hist(eBE, color='green', bins=42 // 2, histtype='step', density=True, label='Bob-Eve')

data = np.array([eAB, eAE, eBE]).flatten()
(mu, sigma) = norm.fit(data)
print(f'Noise = 0.2, mu={mu:.4f}, sigma={sigma:.4f}')
dummyx = np.linspace(np.min(eAB), np.max(eAB), 100)
ax.plot(dummyx, norm.pdf(dummyx, loc=mu, scale=sigma), 'k--', label='Gaussian fit')

ax.set_title('Error in the key distribution, noise = 0.2')
ax.set(xlabel='Error', ylabel='Probability (normalized)')
ax.legend()
plt.savefig('Distribution_Alice_Bob_Eve_error_noise=0.2.png', dpi=400)
