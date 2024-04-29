import numpy as np
import secrets
import matplotlib.pyplot as plt
from scipy.stats import entropy

def TRS(N, BITS_TYPE):
    """
    Generate a random binary sequence of length N.

    Parameters:
    N (int): Length of the sequence.
    BITS_TYPE (list): List containing possible bit values.

    Returns:
    numpy.ndarray: Random binary sequence.
    """
    bits = []
    while len(bits) < N:
        bit = secrets.choice(BITS_TYPE)
        bits.append(bit)
    return np.array(bits, dtype=np.int16)

def entropy_seq(seq, BITS_TYPE):
    """
    Calculate the entropy of a binary sequence.

    Parameters:
    seq (numpy.ndarray): Binary sequence.
    BITS_TYPE (list): List containing possible bit values.

    Returns:
    float: Entropy value.
    """
    nbits = len(seq)
    pk = []
    for bit in BITS_TYPE:
        pk.append(np.count_nonzero(seq == bit) / nbits)
    return entropy(pk, base=len(BITS_TYPE))

if __name__ == '__main__': 
    N = 42
    BITS_TYPE = [0, 1] 
    
    sample = TRS(N, BITS_TYPE)
    for i in range(1000):
        return_sample = TRS(N, BITS_TYPE)
        sample = sample + return_sample
    sample = sample / (i + 1.)

    fig, ax = plt.subplots()
    ax.stairs(sample, fill=True, color='orange')
    ax.set(xlabel='Bit number', ylabel='Average bit value', title='Distribution of bit values for each individual bit')
    fig.savefig('TRS_distribution.png')

    ent = entropy_seq(return_sample, BITS_TYPE)
    print(f'Last TRS sample:\n{return_sample}, Entropy: {ent}')

    """
    Example outputs:
    BEFORE EVE
        #basis alice [0 1 0 0 0 1 0 1 1 0 0 1 1 1 0 0 0 0 1 1 0 1 1 0 0 1 1 0 0 1 0 0 1 0 0 1 0 1 1 0 1 0], Entropy: 0.9934472383802027
        #bit alice [1 0 1 0 0 0 1 1 1 1 1 1 0 1 1 1 0 0 0 0 1 0 0 1 1 0 0 1 0 1 1 1 1 1 0 1 0 1 1 1 1 0], Entropy: 0.9736680645496202
        #basis bob [1 0 1 1 1 0 0 1 0 1 1 0 0 0 0 1 1 1 0 0 0 0 1 0 1 1 1 1 1 0 1 1 1 1 0 1 1 0 1 1 0 0], Entropy: 0.9852281360342515

    WITH EVE
        #basis alice [1 0 1 1 1 0 1 0 0 1 0 1 1 1 0 1 1 1 1 1 0 1 0 1 0 1 0 0 1 0 1 0 1 0 1 0 0 1 0 0 0 0], Entropy: 0.998363672593813
        #bit alice [0 1 0 1 0 1 1 1 1 1 0 0 1 0 0 1 0 0 1 1 1 0 0 1 1 1 0 1 1 0 1 1 1 1 0 0 1 0 0 0 1 1], Entropy: 0.9852281360342515
        #basis Bob [1 0 0 0 0 0 1 0 0 1 0 1 0 1 0 1 0 1 1 0 0 0 0 0 1 1 0 1 1 1 1 1 1 0 0 0 0 0 1 1 1 1], Entropy: 0.998363672593813
        #baisi Eve [0 1 0 0 1 0 0 1 0 1 1 0 1 1 1 0 1 0 0 0 1 1 1 0 0 1 0 1 0 1 1 0 0 1 0 0 0 1 0 1 1 0], Entropy: 0.998363672593813
    """
