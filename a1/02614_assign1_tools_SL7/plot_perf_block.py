import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter

# Plot block size vs. performance
df = pd.read_csv(
    "data/mm_g4_bs_2048_m19860602.out",
    sep="\s+",
    header=None,
    names=["Memory (kB)", "Mflops/s", "#", "type", "block_size"],
)
df.drop("#", axis=1, inplace=True)

df["block_footprint"] = [(3 * 8 * x * x)/1024 for x in df["block_size"]]

# Plot block size vs. performance
fig, ax = plt.subplots(figsize=(10, 6))

# ax.plot(df["block_footprint"], df["Mflops/s"], marker="o")
ax.semilogx(df["block_footprint"], df["Mflops/s"], base=2, marker='o', linestyle='dashed')
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
ax.set_title("Matrix multiplication performance using blocking")
plt.savefig("data/perf_block_footprint.pdf", bbox_inches="tight")
plt.show()
