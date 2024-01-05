import matplotlib.pyplot as plt
import pandas as pd

# Plot block size vs. performance
df = pd.read_csv(
    "data/mm_g4_bs_2000m19859234.out",
    sep="\s+",
    header=None,
    names=["Memory (kB)", "Mflops/s", "#", "type", "block_size"],
)
df.drop("#", axis=1, inplace=True)


# PLot block size vs. performance
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df["block_size"], df["Mflops/s"], marker="o")

ax.set_xlabel("Block size")
ax.set_ylabel("Mflops/s")
ax.set_title("Matrix multiplication performance")
plt.savefig("data/perf_block.pdf", bbox_inches="tight")
plt.show()
