import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import networkx as nx
import numpy as np

# ── Palette ──────────────────────────────────────────────────────
C = {
    "blue":   "#4C72B0",
    "purple": "#7B5EA7",
    "green":  "#3A9E6E",
    "orange": "#E8A838",
    "red":    "#D45F5F",
    "gray":   "#AAAAAA",
    "bg":     "#F8F8F8",
    "white":  "#FFFFFF",
}

fig = plt.figure(figsize=(22, 12), facecolor="white")
fig.patch.set_facecolor("white")

# Two rows: top = RAG, bottom = GRN+Perturbation
ax_top = fig.add_axes([0.01, 0.52, 0.98, 0.46])
ax_bot = fig.add_axes([0.01, 0.02, 0.98, 0.46])
for ax in [ax_top, ax_bot]:
    ax.set_facecolor("white")
    ax.axis("off")
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 5)


# ── Helper: draw a mini graph inside a box ────────────────────────
def mini_graph(ax, cx, cy, w, h, G, pos, node_colors, edge_colors,
               title, title_color, subtitle=""):
    """Draw a rounded box with a small networkx graph inside."""
    rect = mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.12",
        facecolor=C["bg"], edgecolor=title_color,
        linewidth=2.2, zorder=2
    )
    ax.add_patch(rect)

    # scale pos into box
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pad = 0.18

    def tp(p):
        nx_ = (p[0]-xmin)/(xmax-xmin+1e-9)*(1-2*pad)+pad if xmax>xmin else 0.5
        ny_ = (p[1]-ymin)/(ymax-ymin+1e-9)*(1-2*pad)+pad if ymax>ymin else 0.5
        return cx - w/2 + nx_*w, cy - h/2 + 0.35 + ny_*(h-0.55)

    # edges
    for (u, v), ec in zip(G.edges(), edge_colors):
        x0, y0 = tp(pos[u])
        x1, y1 = tp(pos[v])
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=ec,
                                   lw=1.4, mutation_scale=10),
                    zorder=4)

    # nodes
    node_list = list(G.nodes())
    for i, n in enumerate(node_list):
        px, py = tp(pos[n])
        ax.plot(px, py, "o", markersize=9,
                color=node_colors[i], markeredgecolor="white",
                markeredgewidth=1.2, zorder=5)

    # title inside box
    ax.text(cx, cy + h/2 - 0.22, title,
            fontsize=11, fontweight="bold", color=title_color,
            ha="center", va="center", zorder=6)
    if subtitle:
        ax.text(cx, cy - h/2 + 0.16, subtitle,
                fontsize=8.5, color="#777777",
                ha="center", va="center", zorder=6, style="italic")


# ── Helper: draw a simple icon box (no graph) ────────────────────
def icon_box(ax, cx, cy, w, h, title, subtitle, icon, color):
    rect = mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.12",
        facecolor=C["bg"], edgecolor=color,
        linewidth=2.2, zorder=2
    )
    ax.add_patch(rect)
    ax.text(cx, cy + 0.28, icon, fontsize=22, ha="center", va="center", zorder=5)
    ax.text(cx, cy - 0.08, title, fontsize=10.5, fontweight="bold",
            color=color, ha="center", va="center", zorder=5)
    ax.text(cx, cy - 0.48, subtitle, fontsize=8.5, color="#777777",
            ha="center", va="center", zorder=5, style="italic")


# ── Helper: arrow between boxes ──────────────────────────────────
def arrow(ax, x0, x1, y, color="#888888", label=""):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=2.2, mutation_scale=16),
                zorder=1)
    if label:
        ax.text((x0+x1)/2, y+0.2, label, fontsize=8,
                color=color, ha="center", va="bottom", style="italic")


# ════════════════════════════════════════════════════════════════
# ROW 1 — RAG Gene Program Retrieval
# ════════════════════════════════════════════════════════════════
ax_top.text(11, 4.72, "RAG Gene Program Retrieval",
            fontsize=17, fontweight="bold", color="#222222",
            ha="center", va="center")

bw, bh, by = 3.2, 3.6, 2.3

# 1. Query gene — single node
G1 = nx.DiGraph(); G1.add_node("MKI67")
mini_graph(ax_top, 2.0, by, bw, bh, G1,
           {"MKI67": (0.5, 0.5)}, [C["blue"]], [],
           "Query Gene", C["blue"], "e.g. MKI67")

arrow(ax_top, 3.65, 4.45, by, C["purple"], "co-expression")

# 2. WGCNA graph — 6-node undirected
G2 = nx.Graph()
G2.add_edges_from([("A","B"),("A","C"),("B","D"),("C","D"),("D","E"),("B","E"),("E","F"),("C","F")])
pos2 = nx.spring_layout(G2, seed=7)
mini_graph(ax_top, 5.9, by, bw, bh, G2, pos2,
           [C["purple"]]*6,
           [C["purple"]]*G2.number_of_edges(),
           "WGCNA Co-expression", C["purple"], "gene–gene graph")

arrow(ax_top, 7.55, 8.35, by, C["green"], "encode")

# 3. GNN Encoder — layered network
G3 = nx.DiGraph()
layers = [["i1","i2","i3"],["h1","h2"],["o1"]]
for l in range(len(layers)-1):
    for u in layers[l]:
        for v in layers[l+1]:
            G3.add_edge(u, v)
pos3 = {}
for l, nodes in enumerate(layers):
    for k, n in enumerate(nodes):
        pos3[n] = (l, k - (len(nodes)-1)/2)
nc3 = ([C["blue"]]*3 + [C["green"]]*2 + [C["orange"]]*1)
mini_graph(ax_top, 9.8, by, bw, bh, G3, pos3,
           nc3, [C["green"]]*G3.number_of_edges(),
           "GNN Encoder", C["green"], "node → embedding")

arrow(ax_top, 11.45, 12.25, by, C["orange"], "store")

# 4. Vector DB — scatter points
G4 = nx.Graph()
nodes4 = [f"g{i}" for i in range(8)]
G4.add_nodes_from(nodes4)
np.random.seed(3)
pos4 = {n: (np.random.rand(), np.random.rand()) for n in nodes4}
nc4 = [C["orange"] if i == 0 else C["gray"] for i in range(8)]
mini_graph(ax_top, 13.7, by, bw, bh, G4, pos4,
           nc4, [],
           "Gene Embeddings", C["orange"], "vector DB · cosine sim")

arrow(ax_top, 15.35, 16.15, by, C["red"], "retrieve")

# 5. Output: program subgraph
G5 = nx.DiGraph()
G5.add_edges_from([("MKI67","BIRC5"),("MKI67","TOP2A"),("ASPM","MKI67"),
                   ("TOP2A","CEP55"),("BIRC5","NEK2")])
pos5 = nx.spring_layout(G5, seed=12)
nc5 = [C["red"] if n=="MKI67" else C["green"] for n in G5.nodes()]
ec5 = [C["green"]]*G5.number_of_edges()
mini_graph(ax_top, 17.8, by, bw*1.1, bh, G5, pos5,
           nc5, ec5,
           "Gene Program", C["green"], "similar genes + GRN")


# ════════════════════════════════════════════════════════════════
# ROW 2 — GRN Inference & Perturbation
# ════════════════════════════════════════════════════════════════
ax_bot.text(11, 4.72, "GRN Inference & Perturbation Simulation",
            fontsize=17, fontweight="bold", color="#222222",
            ha="center", va="center")

by2 = 2.3

# 1. scRNA-seq input
icon_box(ax_bot, 2.0, by2, bw, bh,
         "scRNA-seq + Time", "cells × genes × t₁…tₙ", "🧫", C["blue"])

arrow(ax_bot, 3.65, 4.45, by2, C["purple"], "infer")

# 2. CARDAMOM — ODE graph
G6 = nx.DiGraph()
G6.add_edges_from([("A","B"),("B","C"),("C","A"),("A","D"),("D","B")])
pos6 = nx.circular_layout(G6)
nc6 = [C["purple"]]*5
ec6 = [C["purple"]]*G6.number_of_edges()
mini_graph(ax_bot, 5.9, by2, bw, bh, G6, pos6,
           nc6, ec6,
           "CARDAMOM GRN", C["purple"], "ODE mechanistic model")

arrow(ax_bot, 7.55, 8.35, by2, C["green"], "structure")

# 3. Inferred GRN — directed weighted
G7 = nx.DiGraph()
G7.add_edges_from([("MKI67","BIRC5"),("MKI67","TOP2A"),
                   ("BIRC5","CEP55"),("TOP2A","NEK2"),
                   ("NEK2","BIRC5"),("CEP55","MKI67")])
pos7 = nx.spring_layout(G7, seed=99)
nc7 = [C["red"] if n=="MKI67" else C["green"] for n in G7.nodes()]
ec7 = [C["green"] if i%2==0 else "#D45F5F" for i in range(G7.number_of_edges())]
mini_graph(ax_bot, 9.8, by2, bw, bh, G7, pos7,
           nc7, ec7,
           "Inferred GRN", C["green"], "activation / repression")

arrow(ax_bot, 11.45, 12.25, by2, C["orange"], "simulate KO")

# 4. KO simulation — BIRC5 removed, remaining network
G8 = nx.DiGraph()
G8.add_edges_from([("MKI67","TOP2A"),("TOP2A","NEK2"),("CEP55","MKI67")])
pos8 = nx.spring_layout(G8, seed=5)
nc8 = [C["orange"] for _ in G8.nodes()]
ec8 = [C["orange"]]*G8.number_of_edges()
mini_graph(ax_bot, 13.7, by2, bw, bh, G8, pos8,
           nc8, ec8,
           "KO Simulation", C["orange"], "BIRC5 KO · propagate Δ")

arrow(ax_bot, 15.35, 16.15, by2, C["red"], "Δ expression")

# 5. Targets — bar-chart icon + highlighted nodes
G9 = nx.DiGraph()
G9.add_edges_from([("PPP1R12B","?"),("MAP3K21","?"),("CEP55","?")])
pos9 = {"PPP1R12B":(0,2),"MAP3K21":(0,1),"CEP55":(0,0),"?":(1,1)}
nc9 = [C["orange"],C["orange"],C["red"],C["gray"]]
ec9 = [C["orange"],C["orange"],C["red"]]
mini_graph(ax_bot, 17.8, by2, bw*1.1, bh, G9, pos9,
           nc9, ec9,
           "Therapeutic Targets", C["red"],
           "co-targets ↑ · direct target ↓")

# Phase divider
ax_bot.axvline(x=11, ymin=0.05, ymax=0.88,
               color="#cccccc", linewidth=1.5, linestyle="--", zorder=0)
ax_bot.text(5.9, 0.32, "① Learn network from temporal scRNA-seq",
            fontsize=10, color="#999999", ha="center", style="italic")
ax_bot.text(15.5, 0.32, "② Simulate perturbations → targets",
            fontsize=10, color="#999999", ha="center", style="italic")

plt.savefig("/Users/zeev/CardamomOT/my_project/Data/architecture.png",
            dpi=180, bbox_inches="tight", facecolor="white")
print("Saved architecture.png")
