import numpy as np
from numba import njit, int64
from secrets import randbits
from random import randint

from functools import wraps
from time import time

def timeit(N = 1000):
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
                    dt.append(te-ts)
                except:
                    i = i-1
            dt = np.array(dt)
            print(f'Average time for {f.__name__} with arg {args}: {dt.mean()}, std {np.std(dt)}')
            with open("time.txt", "a") as myfile:
                myfile.write(f'{f.__name__};{args[0]};{dt.mean()};{np.std(dt)}\n')
            return result
        return wrap
    return timing_decorator


@njit(int64(int64,int64), cache=True)
def gcd(a, b):
    while b != 0:
        t = b
        b = a % b
        a = t
    return a

@njit('UniTuple(int64, 3)(int64,int64)', cache=True)
def xgcd(a, b):
    x0, x1, y0, y1 = 0, 1, 1, 0
    while a != 0:
        (q, a), b = divmod(b, a), a
        y0, y1 = y1, y0 - q * y1
        x0, x1 = x1, x0 - q * x1
    return b, x0, y0

@njit(int64(int64,int64), cache=True)
def mod_inverse(a, b):
    g, x, _ = xgcd(a, b)
    if g != 1:
        return 0
    return x % b

@njit(int64(int64,int64), cache=True)
def lcm(a, b):
    return a*b//gcd(a,b)

@njit(int64(int64), cache=True)
def FirstPrimeFactor(n):
    if n % 2 == 0:
        return 2
    if n < 3:
        return -1
    d = 3
    while d * d <= n:
        if n % d == 0:
            return d
        d= d + 2
    return n

def PrimeOfOrder(order):
    candidate = randbits(order)
    while FirstPrimeFactor(candidate) != candidate:
        candidate = randbits(order)
    return candidate

def keyPair(order):
    e = 149
    p = PrimeOfOrder(order)
    q = PrimeOfOrder(order)
    n = p*q
    ln = lcm(p-1, q-1)
    while ln < e or ln % e == 0:
        p = PrimeOfOrder(order)
        q = PrimeOfOrder(order)
        n = p*q
        ln = lcm(p-1, q-1)
    d = mod_inverse(e, ln)
    return d, e, n

@njit(cache=True)
def encrypt(public_key, n, mes_as_int):
    c = np.copy(mes_as_int)
    for k , char in enumerate(mes_as_int):
        for i in range(public_key-1):
            c[k] = c[k] * mes_as_int[k]
            c[k] = c[k] % n
    return c

@njit(cache=True)
def decrypt(private_key, n, encrypted_mess):
    m = np.copy(encrypted_mess)
    for k , char in enumerate(encrypted_mess):
        for i in range(private_key-1):
            m[k] = m[k] * encrypted_mess[k]
            m[k] = m[k] % n
    return m

@njit(int64(int64,int64), cache=True)
def attack(e, n):
    start = int(n**0.5)
    for trial in range(start, n):
        if n % trial == 0:
            p = trial
            q = n // p
            break
    ln = lcm(p-1, q-1)
    d = mod_inverse(e, ln)
    return d

@timeit(N=100)
def communication_model(order):
    mess = 'I love Hania <3!'

    private_key, public_key, n = keyPair(order)
    #print(f'{private_key}, {public_key}, {n}')

    mes_as_int = np.array(list(mess.encode("ascii")))

    c = encrypt(public_key, n, mes_as_int)
    d = decrypt(private_key, n, c)

    mess_trans = ''.join([chr(i) for i in d])
    if mess != mess_trans:
        print(d)
        print('The messages do not match!')

@timeit(N=100)
def attack_model(order):
    mess = 'I love Hania <3!'

    _, public_key, n = keyPair(order)
    #print(f'{private_key}, {public_key}, {n}')

    mes_as_int = np.array(list(mess.encode("ascii")))

    c = encrypt(public_key, n, mes_as_int)

    private_key = attack(public_key, n)

    d = decrypt(private_key, n, c)

    mess_trans = ''.join([chr(i) for i in d])
    if mess != mess_trans:
        print('The messages do not match!')

@timeit(N=20)
def attack_model_no_encryption(order):
    true_private_key, public_key, n = keyPair(order)
    #print(f'{private_key}, {public_key}, {n}')
    private_key = attack(public_key, n)

    if true_private_key != private_key:
        print('The attack failed!')

@timeit(N=100)
def key_gen_model(order):
    private_key, public_key, n = keyPair(order)


for i in range(29, 1028):
    attack_model_no_encryption(i)
    #attack_model(i)
    #communication_model(i)
    #key_gen_model(i)
