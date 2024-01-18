import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
from matplotlib.ticker import ScalarFormatter

def lattice_updates_per_sec(n_iterations, N, s):
    return n_iterations * N**3 / s

df = pd.read_csv('somethingsomething.out', sep='\s+')

# Plot performance: Exercise 6
fig, ax = plt.subplots(figsize=(10, 6))
for key, grp in df.groupby('type'):
    ax.plot(grp['res'], lattice_updates_per_sec(grp['iterations'], grp['res'], grp['seconds']), label=key(0), marker'o')
ax.set_title("Performance exercise 6")
ax.set_xlabel("Resolution")
ax.set_ylabel("Performance [Mlu/s]")
plt.savefig("plt_ex6.pdf", bbox_inches='tight')

