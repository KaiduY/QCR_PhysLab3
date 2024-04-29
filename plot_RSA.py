import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
data = {
    'attack_model':{
        'n':[],
        'mu':[],
        'std':[]
    },
    'communication_model':{
        'n':[],
        'mu':[],
        'std':[]
    },
    'key_gen_model':{
        'n':[],
        'mu':[],
        'std':[]
    },
    'attack_model_no_encryption':{
        'n':[],
        'mu':[],
        'std':[]
    }
}
with open("time.txt", "r") as myfile:
    lines = myfile.readlines()
    for line in lines:
        line = line.rstrip('\n')
        line = line.split(';')
        data[line[0]]['n'].append(int(line[1]))
        data[line[0]]['mu'].append(float(line[2]))
        data[line[0]]['std'].append(float(line[3]))

df0 = pd.DataFrame.from_dict(data['attack_model'])
df1 = pd.DataFrame.from_dict(data['communication_model'])
df2 = pd.DataFrame.from_dict(data['key_gen_model'])
df3 = pd.DataFrame.from_dict(data['attack_model_no_encryption'])


fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(10,5))
axs = axs.flatten()
ax = axs[0]
ax.set(yscale='log', xlabel='Prime numbers maximum no. of bits', ylabel = 'Time (s)', title='Computation time')

ax.errorbar(df0['n'], df0['mu'], yerr=df0['std'], ecolor='red', fmt='--r', capsize=3)
ax.plot(df0['n'], df0['mu'], 'r*', label = 'Attack model')

ax.errorbar(df1['n'], df1['mu'], yerr=df1['std'], ecolor='green', fmt='--g', capsize=3)
ax.plot(df1['n'], df1['mu'], 'g*', label = 'Communication model')

ax.errorbar(df2['n'], df2['mu'], yerr=df2['std'], ecolor='blue', fmt='--b', capsize=3)
ax.plot(df2['n'], df2['mu'], 'b*', label = 'Key generation model')
ax.legend()

def exp_model(x, amp, tau):
    return amp*np.exp(x*tau)

popt, pcov = curve_fit(exp_model, df3['n'], df3['mu'], sigma=df3['std'], absolute_sigma=True)
print(popt)
print(f'Sigma: {np.sqrt(np.diag(pcov))}')
print(f'Condition number: {np.linalg.cond(pcov):.2e}')
ax = axs[1]
ax.set(yscale='log', xlabel='Prime numbers maximum no. of bits', ylabel = 'Time (s)', title='Computation time without\nmessage encryption/decryptyion')
ax.errorbar(df3['n'], df3['mu'], yerr=df3['std'], ecolor='orange', fmt='--', color='orange', capsize=3)
ax.plot(df3['n'], df3['mu'],'*', color='orange', label = 'Attack model (no message)')

dummyx = np.linspace(5, 35, 100)
model = exp_model(dummyx, *popt)

time = exp_model(1024, *popt)/3600/24/365
print(f'Time for a 1024 bit long key is apx: {time:.2e} yr')

ax.plot(dummyx, model, 'k:', label = 'Exponetial fit')
ax.legend()
fig.tight_layout()
plt.savefig('RSA_together.png', dpi=400)