import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
from matplotlib.ticker import ScalarFormatter

def lattice_updates_per_sec(n_iterations, N, s):
    return n_iterations * N**3 / s

df = pd.read_csv('g4_bench_refvsGPU_19921336.out', sep='\s+')

# Plot performance: Exercise 6
fig, ax = plt.subplots(figsize=(10, 6))
for key, grp in df.groupby('type'):
    ax.plot(grp['res'][1:], lattice_updates_per_sec(grp['iterations'][1:], grp['res'][1:], grp['seconds'][1:]), label=key, marker='o')
ax.set_title("Performance exercise 6")
ax.set_xlabel("N (Resolution)")
ax.set_ylabel("Performance [Mlu/s]")
ax.legend()
plt.savefig("plt_ex6.pdf", bbox_inches='tight')

