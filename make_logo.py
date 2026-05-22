import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load UMAP coords
umap = pd.read_csv("/Users/zeev/CardamomOT/my_project/Data/umap_coords.csv", index_col=0)
x = umap["x"].values
y = umap["y"].values

# French flag colors by x position
xmin, xmax = x.min(), x.max()
t = (x - xmin) / (xmax - xmin)   # 0..1

colors = []
for ti in t:
    if ti < 0.33:
        colors.append("#002395")   # bleu
    elif ti < 0.66:
        colors.append("#FFFFFF")   # blanc
    else:
        colors.append("#ED2939")   # rouge

fig, ax = plt.subplots(figsize=(3.2, 3.2), facecolor="white")
ax.set_facecolor("white")
ax.scatter(x, y, c=colors, s=1.8, alpha=0.85, linewidths=0)
ax.axis("off")
plt.tight_layout(pad=0)
plt.savefig("/Users/zeev/CardamomOT/my_project/Data/logo.png",
            dpi=200, bbox_inches="tight", facecolor="white")
print("Saved logo.png")
