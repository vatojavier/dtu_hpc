"""
Execute like:
 python plot_perf_block.py <file_path> <out_name>
 i.e: python plot_perf_block.py data/mm_g4_bs_2048_m19860602.out "perf_block.pdf"
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter
import sys

def plot_performance(file_path, out_name):
    # Read data from the given file path
    df = pd.read_csv(
        file_path,
        sep="\s+",
        header=None,
        names=["Memory (kB)", "Mflops/s", "#", "type", "block_size"],
    )
    df.drop("#", axis=1, inplace=True)

    # Calculate block footprint
    df["block_footprint"] = [(3 * 8 * x * x)/1024 for x in df["block_size"]]

    # Plot block size vs. performance
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.semilogx(df["block_footprint"], df["Mflops/s"], base=2, marker='o', linestyle='dashed', label='blk')
    ax.vlines(
        [32, 256,  30720],
        df["Mflops/s"].min(),
        df["Mflops/s"].max(),
        colors="r",
        linestyles="dashed",
        label="L1, L2, L3",
    )

    ax.set_xlabel("Block footprint (kB)")
    ax.set_ylabel("Mflops/s")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_title("Matrix multiplication performance using blocking and no optimization")
    ax.legend()
    plt.savefig("data/" + out_name, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    # Command line arguments
    file_path = sys.argv[1]
    out_name = "perf_block.pdf" # default outname
    if len(sys.argv) > 2:
        out_name = sys.argv[2]

    plot_performance(file_path, out_name)
