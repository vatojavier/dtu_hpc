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
df = pd.read_csv('Outputs/g4_JASEQ_no_opt19885089.out', sep='\s+')
df2 = pd.read_csv('Outputs/g4_GSSEQ_no_opt19885309.out', sep='\s+')
df3 = pd.read_csv('Outputs/g4_GSSEQ_O319885473.out',sep='\s+')
df4 = pd.read_csv('Outputs/g4_JASEQ_SEQ_O319885473.out',sep='\s+')

# fig again but with iterations on y axis
fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogx(3*memory_footprint(df['N']+2), df['iterations'], 'o-', label='Jacobi', base=2)
ax.semilogx(2*memory_footprint(df2['N']+2), df2['iterations'], 'o-', label='Gauss-Seidel',base=2)

ax.set_xlabel('Memory footprint (kB)')
ax.set_ylabel('iterations')
ax.set_title('Sequential Jacobi vs. Gauss-Seidel iterations till convergence')
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.legend()
ax.grid(axis='y')
plt.savefig('figures/sequential_convergence_iterations.pdf', bbox_inches='tight')
plt.show()
# fig again but with iterations on y axis
fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogx(3*memory_footprint(df['N']+2), df['time'], 'o-', label='Jacobi', base=2)
ax.semilogx(2*memory_footprint(df2['N']+2), df2['time'], 'o-', label='Gauss-Seidel',base=2)
ax.semilogx(2*memory_footprint(df3['N']+2), df3['time'], 'o-', label='Gauss-Seidel (opt)',base=2)
ax.semilogx(3*memory_footprint(df4['N']+2), df4['time'], 'o-', label='Jacobi (opt)',base=2)

ax.set_xlabel('Memory footprint  (kB)')
ax.set_ylabel('time (s)')
ax.set_title('Sequential Jacobi vs. Gauss-Seidel time till convergence')
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.legend()
ax.grid(axis='y')
plt.savefig('figures/sequential_convergence_time.pdf', bbox_inches='tight')
plt.show()
#%% Plot 2 compare lattice updates per second for different N
# Mlup/s
# Mlup/s = mega lattice updates per second
dfs = [df, df2, df3, df4]
mlups = [lattice_updates_per_sec(df['iterations'],df['N'], df['time']) / 1e6 for df in dfs]
# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogx(3*memory_footprint(df['N']+2), mlups[0], 'o-', label='Jacobi', base=2)
ax.semilogx(2*memory_footprint(df2['N']+2), mlups[1], 'o-', label='Gauss-Seidel',base=2)
ax.semilogx(2*memory_footprint(df3['N']+2), mlups[2], 'o-', label='Gauss-seidel (opt)',base=2)
ax.semilogx(3*memory_footprint(df4['N']+2), mlups[3], 'o-', label='Jacobi (opt)',base=2)
ax.set_xlabel('Memory footprint (kB)')
ax.set_ylabel('Mlup/s')
ax.set_title('Sequential Jacobi & Gauss-Seidel Mlups vs N')
def flatten(l):
    return [item for sublist in l for item in sublist]
ax.vlines([32, 256, 30720], min(flatten(mlups)), max(flatten(mlups)), colors='k', linestyles='dashed', label='L1, L2, L3',alpha=0.6)
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.legend()
plt.savefig('figures/sequential_Mlups.pdf', bbox_inches='tight')
plt.show()

#%% figure 3: parallel jacobi
def f_calc(T1, TP, P):
    return (1-TP/T1)/(1-(1/P))


files = glob.glob('Outputs/jacobi_parallel_baseline/*.out')
exps = ['baseline no opt', 'baseline, O3']
for exp, file in zip(exps, files):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1,25),range(1,25), label='1.00')
    df = pd.read_csv(file, sep='\s+')
    for i, (N_size, grp) in enumerate(df.groupby(['N'])):
        if (i + 1) % 2 == 0: #just get every other, since we the plot is already crowded
            continue
        T1 = grp['time'].iloc[0]
        TP = grp['time']
        P = grp['n_threads']
        # amdahls law stuff
        # at all times we have P, T(1) and T(P)
        # S(P)
        speedup = T1 / TP

        # compare parallel fraction to the last element
        f = f_calc(T1, TP, P).iloc[-1]

        # Plot
        ax.plot(P, speedup, 'o--', label=f'N: {N_size[0]},  f: {round(f,3)}')

    ax.set_xlabel('Processors')
    ax.set_ylabel('Speed-up')
    ax.set_title(f'Parallel Jacobi {exp}')
    ax.legend(title='Grid (N), Parallel fraction (f)')
    ax.grid(axis='y')
    plt.savefig(f'figures/parallel_jacobi_{exp}.pdf', bbox_inches='tight')
    plt.show()

#%% 
files = glob.glob('Outputs/jacobi_parallel_baseline/*.out')
exps = ['no opt', 'O3']

for exp, file in zip(exps, files):
    fig, ax = plt.subplots(figsize=(10, 6))
    df = pd.read_csv(file, sep='\s+')
    for i, (N_size, grp) in enumerate(df.groupby(['n_threads'])):
        # if (i + 1) % 2 == 0: #just get every other, since we the plot is already crowded
        #     continue

        mlups = lattice_updates_per_sec(grp['iterations'],grp['N'], grp['time']) / 1e6
    # Plot
        ax.semilogx(3*memory_footprint(grp['N']+2), mlups, 'o-', label=f'{N_size[0]}', base=2)
    ax.set_xlabel('Memory footprint (kB)')
    ax.set_ylabel('Mlup/s')
    ax.set_title(f'Baseline Parallel Jacobi Mlups {exp}')

    ax.vlines([32, 256, 30720], min(mlups), max(mlups), colors='k', linestyles='dashed', label='L1, L2, L3',alpha=0.6)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.legend(title='n threads')
    plt.savefig(f'figures/jacobi_para_baseline_Mlups_{exp}.pdf', bbox_inches='tight')
    plt.show()
# %% figure 4: parallel gauss-seidel
# same kind of plot as above

#%% figure 5: parallel jacobi and gs 
# compared when using different grid sizes (memory footprint)
