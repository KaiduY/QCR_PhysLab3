import numpy as np
from numba import njit, int64
from secrets import randbits
from random import randint

from functools import wraps
from time import time

def timeit(N=1000):
    """
    Decorator to measure the execution time of a function.

    Parameters:
    N (int): Number of iterations for timing measurement.

    Returns:
    function: Wrapped function with timing measurement.
    """
    def timing_decorator(f):
        @wraps(f)
        def wrap(*args, **kw):
            dt = []
            result = f(*args, **kw)
            for i in range(N):
                try:
                    ts = time()
                    result = f(*args, **kw)
                    te = time()
                    dt.append(te - ts)
                except:
                    i = i - 1
            dt = np.array(dt)
            print(f'Average time for {f.__name__} with arg {args}: {dt.mean()}, std {np.std(dt)}')
            with open("time.txt", "a") as myfile:
                myfile.write(f'{f.__name__};{args[0]};{dt.mean()};{np.std(dt)}\n')
            return result
        return wrap
    return timing_decorator

@njit(int64(int64,int64), cache=True)
def gcd(a, b):
    """
    Compute the greatest common divisor of two integers.

    Parameters:
    a (int): First integer.
    b (int): Second integer.

    Returns:
    int: Greatest common divisor of a and b.
    """
    while b != 0:
        t = b
        b = a % b
        a = t
    return a

@njit('UniTuple(int64, 3)(int64,int64)', cache=True)
def xgcd(a, b):
    """
    Extended Euclidean algorithm to compute the greatest common divisor and the Bezout coefficients.

    Parameters:
    a (int): First integer.
    b (int): Second integer.

    Returns:
    tuple: Tuple containing the greatest common divisor and Bezout coefficients (x0, x1, y0).
    """
    x0, x1, y0, y1 = 0, 1, 1, 0
    while a != 0:
        (q, a), b = divmod(b, a), a
        y0, y1 = y1, y0 - q * y1
        x0, x1 = x1, x0 - q * x1
    return b, x0, y0

@njit(int64(int64,int64), cache=True)
def mod_inverse(a, b):
    """
    Compute the modular inverse of an integer modulo another integer.

    Parameters:
    a (int): Integer.
    b (int): Modulus.

    Returns:
    int: Modular inverse of a modulo b.
    """
    g, x, _ = xgcd(a, b)
    if g != 1:
        return 0
    return x % b

@njit(int64(int64,int64), cache=True)
def lcm(a, b):
    """
    Compute the least common multiple of two integers.

    Parameters:
    a (int): First integer.
    b (int): Second integer.

    Returns:
    int: Least common multiple of a and b.
    """
    return a * b // gcd(a, b)

@njit(int64(int64), cache=True)
def FirstPrimeFactor(n):
    """
    Find the smallest prime factor of an integer.

    Parameters:
    n (int): Integer.

    Returns:
    int: Smallest prime factor of n.
    """
    if n % 2 == 0:
        return 2
    if n < 3:
        return -1
    d = 3
    while d * d <= n:
        if n % d == 0:
            return d
        d = d + 2
    return n

def PrimeOfOrder(order):
    """
    Generate a prime number of a given bit length.

    Parameters:
    order (int): Bit length of the prime number.

    Returns:
    int: Prime number of the specified bit length.
    """
    candidate = randbits(order)
    while FirstPrimeFactor(candidate) != candidate:
        candidate = randbits(order)
    return candidate

def keyPair(order):
    """
    Generate a key pair for RSA encryption.

    Parameters:
    order (int): Bit length of the key pair.

    Returns:
    tuple: Tuple containing the private exponent, public exponent, and modulus.
    """
    e = 149
    p = PrimeOfOrder(order)
    q = PrimeOfOrder(order)
    n = p * q
    ln = lcm(p - 1, q - 1)
    while ln < e or ln % e == 0:
        p = PrimeOfOrder(order)
        q = PrimeOfOrder(order)
        n = p * q
        ln = lcm(p - 1, q - 1)
    d = mod_inverse(e, ln)
    return d, e, n

@njit(cache=True)
def encrypt(public_key, n, mes_as_int):
    """
    Encrypt a message using RSA encryption.

    Parameters:
    public_key (int): Public exponent.
    n (int): Modulus.
    mes_as_int (numpy.ndarray): Message as an array of integers.

    Returns:
    numpy.ndarray: Encrypted message.
    """
    c = np.copy(mes_as_int)
    for k, char in enumerate(mes_as_int):
        for i in range(public_key - 1):
            c[k] = c[k] * mes_as_int[k]
            c[k] = c[k] % n
    return c

@njit(cache=True)
def decrypt(private_key, n, encrypted_mess):
    """
    Decrypt an encrypted message using RSA decryption.

    Parameters:
    private_key (int): Private exponent.
    n (int): Modulus.
    encrypted_mess (numpy.ndarray): Encrypted message.

    Returns:
    numpy.ndarray: Decrypted message.
    """
    m = np.copy(encrypted_mess)
    for k, char in enumerate(encrypted_mess):
        for i in range(private_key - 1):
            m[k] = m[k] * encrypted_mess[k]
            m[k] = m[k] % n
    return m

@njit(int64(int64,int64), cache=True)
def attack(e, n):
    """
    Perform a brute-force attack to obtain the private exponent in RSA encryption.

    Parameters:
    e (int): Public exponent.
    n (int): Modulus.

    Returns:
    int: Private exponent.
    """
    start = int(n ** 0.5)
    for trial in range(start, n):
        if n % trial == 0:
            p = trial
            q = n // p
            break
    ln = lcm(p - 1, q - 1)
    d = mod_inverse(e, ln)
    return d

@timeit(N=100)
def communication_model(order):
    """
    Simulate a communication model using RSA encryption and decryption.

    Parameters:
    order (int): Bit length of the key pair.
    """
    mess = 'I love Hania <3!'

    private_key, public_key, n = keyPair(order)

    mes_as_int = np.array(list(mess.encode("ascii")))

    c = encrypt(public_key, n, mes_as_int)
    d = decrypt(private_key, n, c)

    mess_trans = ''.join([chr(i) for i in d])
    if mess != mess_trans:
        print('The messages do not match!')

@timeit(N=100)
def attack_model(order):
    """
    Simulate an attack on RSA encryption by attempting to decrypt an encrypted message without the private key.

    Parameters:
    order (int): Bit length of the key pair.
    """
    mess = 'I love Hania <3!'

    _, public_key, n = keyPair(order)

    mes_as_int = np.array(list(mess.encode("ascii")))

    c = encrypt(public_key, n, mes_as_int)

    private_key = attack(public_key, n)

    d = decrypt(private_key, n, c)

    mess_trans = ''.join([chr(i) for i in d])
    if mess != mess_trans:
        print('The messages do not match!')

@timeit(N=20)
def attack_model_no_encryption(order):
    """
    Simulate an attack on RSA encryption without encryption being performed.

    Parameters:
    order (int): Bit length of the key pair.
    """
    true_private_key, public_key, n = keyPair(order)

    private_key = attack(public_key, n)

    if true_private_key != private_key:
        print('The attack failed!')

@timeit(N=100)
def key_gen_model(order):
    """
    Simulate key generation for RSA encryption.

    Parameters:
    order (int): Bit length of the key pair.
    """
    private_key, public_key, n = keyPair(order)

for i in range(5, 1028):
    attack_model_no_encryption(i)
    attack_model(i)
    communication_model(i)
    key_gen_model(i)
