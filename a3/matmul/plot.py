#%%
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

# this only work when running in python interactive mode
# %config InlineBackend.figure_format = 'retina'
#%%
# Read data
df = pd.read_csv('out/libs.out', sep='\s+', header=None, names=['Memory (kB)', 'Mflops/s','#', 'type'])
df.drop('#', axis=1, inplace=True)
# apply function to 1 column
df['type'] = df['type'].apply(lambda x: x[8:])

# dtu_hpc/a1/02614_assign1_tools_SL7/mm_batch_19855582.out

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
for key, grp in df.groupby(['type']):
    # if np.array(grp['type']).any() == 'lib':
    #     continue
    ax.semilogx(grp['Memory (kB)']/1024, grp['Mflops/s']/1000, label=key[0], base=2, marker='o', linestyle='dashed')
# ax.vlines([32, 256+32, 30720+256+32], df['Mflops/s'].min(), df['Mflops/s'].max(), colors='r', linestyles='dashed', label='L1, L2, L3')
ax.set_xlabel('Memory (mB)')
ax.set_ylabel('Gflops/s')
ax.set_title('Matrix multiplication performance')
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.legend()
plt.savefig('out/lib_exps.pdf', bbox_inches='tight')
plt.show()
#%% performance
df = pd.read_csv('out/matmult_results_2/offload.out', sep='\s+', header=None, names=['Memory (kB)', 'Mflops/s','#', 'type'])
df.drop('#', axis=1, inplace=True)
# apply function to 1 column
df['type'] = df['type'].apply(lambda x: x[8:])
df.sort_values(by=['Memory (kB)'], inplace=True)

# dtu_hpc/a1/02614_assign1_tools_SL7/mm_batch_19855582.out

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
for key, grp in df.groupby(['type']):
    if np.array(grp['type']).any() == 'lib_offload':
        continue
    ax.semilogx(grp['Memory (kB)']/1024, grp['Mflops/s']/1000, label=key[0], base=2, marker='o', linestyle='dashed')
# ax.vlines([32, 256+32, 30720+256+32], df['Mflops/s'].min(), df['Mflops/s'].max(), colors='r', linestyles='dashed', label='L1, L2, L3')
ax.set_xlabel('Memory (mB)')
ax.set_ylabel('Gflops/s')
ax.set_title('Matrix multiplication performance, GPU offload')
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.legend()
plt.savefig('out/perf.pdf', bbox_inches='tight')
plt.show()

#%% blk size
df = pd.read_csv('out/blk_size.out', sep='\t')
df.columns = ['blk size', 'Mflops/s']


# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df['blk size'], df['Mflops/s']/1000, marker='o', linestyle='dashed')
# ax.vlines([32, 256+32, 30720+256+32], df['Mflops/s'].min(), df['Mflops/s'].max(), colors='r', linestyles='dashed', label='L1, L2, L3')
ax.set_xlabel('block size')
ax.set_ylabel('Gflops/s')
ax.set_title('Effect of block size on performance')
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.legend()
plt.savefig('out/blocksize.pdf', bbox_inches='tight')
plt.show()
# %%
