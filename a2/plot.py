#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
from matplotlib.ticker import ScalarFormatter


# how we can read the binary files
# with open('poisson_res_10.bin', mode='rb') as f:
#     d = f.read()

# arr = np.frombuffer(d, dtype=np.float64)
# plt.imshow(arr.reshape(12,12,12)[2,:,:])
def lattice_updates_per_sec(n_iterations, N, s):
    return n_iterations * N**3 / s

def amdahls_law(n, p):
    return 1 / ((1-p) + p/n)

def memory_footprint(N):
    '''memory footprint of 3D matrix in kB'''
    return N**3 * 8 / 1024

#%% Plot 1 sequential Jacobi vs. sequential Gauss-Seidel
# convergence based on threshold
# ie how many seconds or iterations to reach a certain threshold
# for different N
files = glob.glob('Outputs/*.out')
df = pd.read_csv(files[0], sep='\s+', header=None, names=['time', 'iterations', 'iter_max', 'N', 'tolerance', 'start_T', 'exp_name'])
# df2 = pd.read_csv('output2/outfile.out', sep='\s+', header=None, names=['time', 'iterations', 'iter_max', 'N', 'tolerance', 'start_T', 'exp_name'])

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogx(3*memory_footprint(df['N']), df['iterations'], label='Jacobi', base=2)
# ax.semilogx(2*memory_footprint(df2['N']), df2['iterations'], label='Gauss-Seidel')

ax.set_xlabel('N (matrix size) (kB)')
ax.set_ylabel('iterations')
ax.set_title('Sequential Jacobi vs. Sequential Gauss-Seidel scaling over N')
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.vlines([32, 256, 30720], df['iterations'].min(), df['iterations'].max(), colors='k', linestyles='dashed', label='L1, L2, L3',alpha=0.6)
ax.legend()
plt.savefig('figures/sequential_convergence.pdf', bbox_inches='tight')
plt.show()
#%% Plot 2 compare lattice updates per second for different N
# Mlup/s
# Mlup/s = mega lattice updates per second
# df2 = pd.read_csv('output2/outfile.out', sep='\s+', header=None, names=['time', 'iterations', 'iter_max', 'N', 'tolerance', 'start_T', 'exp_name'])

mlups = lattice_updates_per_sec(df['iterations'],df['N'], df['time']) / 1e6
# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogx(3*memory_footprint(df['N']), mlups, label='Jacobi',base=2)
# ax.semilogx(2*memory_footprint(df2['N']),1e6*lattice_updates_per_sec(df2['iterations'],df2['N'], df2['time']), label='Gauss-Seidel')

ax.set_xlabel('N (matrix size) (kB)')
ax.set_ylabel('Mlup/s')
ax.set_title('Sequential Jacobi & Gauss-Seidel Mlups vs N')
ax.vlines([32, 256, 30720], mlups.min(), mlups.max(), colors='k', linestyles='dashed', label='L1, L2, L3',alpha=0.6)
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.legend()
plt.savefig('figures/sequential_Mlups.pdf', bbox_inches='tight')
plt.show()

#%% figure 3: parallel jacobi
def f_calc(T1, TP, P):
    return (1-TP/T1)/(1-(1/P))

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(1,25), label='1.00')

files = glob.glob('output3/*.out')
for file in files:
    df = pd.read_csv(file, sep='\s+', header=None, names=['time', 'iterations', 'iter_max', 'N', 'tolerance', 'start_T', 'exp_name'])

    T1 = df['time'].iloc[0]
    TP = df['time']
    P = df['N']
    # amdahls law stuff
    # at all times we have P, T(1) and T(P)
    # S(P)
    speedup = T1 / TP

    f = f_calc(T1, TP, P)

    # Plot
    ax.plot(P, speedup, label=f)

ax.set_xlabel('Processors')
ax.set_ylabel('Speed-up')
ax.set_title('Parallel Jacobi')
ax.legend(title='parallel fraction (f)')
ax.grid(axis='x')
plt.savefig('parallel_jacobi.pdf', bbox_inches='tight')
plt.show()
# %% figure 4: parallel gauss-seidel
# same kind of plot as above

#%% 
