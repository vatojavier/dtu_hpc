import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
# import numpy as np

# this only work when running in python interactive mode
%config InlineBackend.figure_format = 'retina'
#%%
# figure 1
# Read data
df = pd.read_csv('opt/mm_batch_19855582 with lib.out', sep='\s+', header=None, names=['Memory (kB)', 'Mflops/s','#', 'type'])
df.drop('#', axis=1, inplace=True)
# df.drop('err', axis=1, inplace=True)
# apply function to 1 column
df['type'] = df['type'].apply(lambda x: x[-3:])

# dtu_hpc/a1/02614_assign1_tools_SL7/mm_batch_19855582.out

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
for key, grp in df.groupby(['type']):
    ax.semilogx(grp['Memory (kB)'][1:], grp['Mflops/s'][1:], label=key[0], base=2, marker='o', linestyle='dashed')
ax.vlines([32, 256, 30720], df['Mflops/s'].min(), df['Mflops/s'].max(), colors='r', linestyles='dashed', label='L1, L2, L3')
ax.set_xlabel('Memory (kB)')
ax.set_ylabel('Mflops/s')
ax.set_title('Matrix multiplication performance')
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.legend()
plt.savefig('ordering_exp_lib.pdf', bbox_inches='tight')
plt.show()

#%%
# figure 2
df = pd.read_csv('opt/mm_batch_19855582 nat.out', sep='\s+', header=None, names=['Memory (kB)', 'Mflops/s','#', 'type'])
df.drop('#', axis=1, inplace=True)
# df.drop('err', axis=1, inplace=True)
# apply function to 1 column
df['type'] = df['type'].apply(lambda x: x[-3:])

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
for key, grp in df.groupby(['type']):
    ax.semilogx(grp['Memory (kB)'][1:], grp['Mflops/s'][1:], label=key[0], base=2, marker='o', linestyle='dashed')
ax.vlines([32, 256, 30720], df['Mflops/s'].min(), df['Mflops/s'].max(), colors='r', linestyles='dashed', label='L1, L2, L3')
ax.set_xlabel('Memory (kB)')
ax.set_ylabel('Mflops/s')
ax.set_title('Matrix multiplication performance (nat)')
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.legend()
plt.savefig('ordering_exp_nat.pdf', bbox_inches='tight')
plt.show()

#%% figure 3
df = pd.read_csv('opt/mm_batch_19855582.out', sep='\s+', header=None, names=['Memory (kB)', 'Mflops/s','#', 'type'])
df.drop('#', axis=1, inplace=True)
# df.drop('err', axis=1, inplace=True)
# apply function to 1 column
df['type'] = df['type'].apply(lambda x: x[-3:])

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
for key, grp in df.groupby(['type']):
    ax.semilogx(grp['Memory (kB)'][1:], grp['Mflops/s'][1:], label=key[0], base=2, marker='o', linestyle='dashed')
ax.vlines([32, 256, 30720], df['Mflops/s'].min(), df['Mflops/s'].max(), colors='r', linestyles='dashed', label='L1, L2, L3')
ax.set_xlabel('Memory (kB)')
ax.set_ylabel('Mflops/s')
ax.set_title('Matrix multiplication performance')
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.legend()
plt.savefig('ordering_exp_perms.pdf', bbox_inches='tight')
plt.show()
#%% figure 4
df = pd.read_csv('opt/opt_exps_nat.out', sep='\s+', header=None, names=['Memory (kB)', 'Mflops/s','err', '#', 'type'])
df.drop('#', axis=1, inplace=True)
df.drop('err', axis=1, inplace=True)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
for key, grp in df.groupby(['type']):
    ax.semilogx(grp['Memory (kB)'], grp['Mflops/s'], label=key[0], base=2, marker='o', linestyle='dashed')
ax.vlines([32, 256, 30720], df['Mflops/s'].min(), df['Mflops/s'].max(), colors='r', linestyles='dashed', label='L1, L2, L3')
ax.set_xlabel('Memory (kB)')
ax.set_ylabel('Mflops/s')
ax.set_title('Optimisation options on matrix multiplication performance (nat)')
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.legend()
plt.savefig('ordering_exp_opt_nat.pdf', bbox_inches='tight')
plt.show()
#%% figure 5
df = pd.read_csv('opt/opt_exps_all_orderings.out', sep='\s+', header=None, names=['Memory (kB)', 'Mflops/s','err', '#', 'type'])
df.drop('#', axis=1, inplace=True)
df.drop('err', axis=1, inplace=True)
df['type'] = df['type'].apply(lambda x: x[-3:])
# Plot
fig, ax = plt.subplots(figsize=(10, 6))
for key, grp in df.groupby(['type']):
    ax.semilogx(grp['Memory (kB)'], grp['Mflops/s'], label=key[0], base=2, marker='o', linestyle='dashed')
ax.vlines([32, 256, 30720], df['Mflops/s'].min(), df['Mflops/s'].max(), colors='r', linestyles='dashed', label='L1, L2, L3')
ax.set_xlabel('Memory (kB)')
ax.set_ylabel('Mflops/s')
ax.set_title('Optimisation options on matrix multiplication performance')
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.legend()
plt.savefig('ordering_exp_opt_all_orderings.pdf', bbox_inches='tight')
plt.show()
# %%
