import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import hf_hub_download

REPO_ID = "weizmannzeev/rms-gene-programs"
LOCAL_DIR = "/Users/zeev/CardamomOT/my_project/Data"

@st.cache_resource
def load_data(dataset="v1"):
    import os

    if dataset == "v1":
        files = [
            "gcn_gene_embeddings_clusters.csv",
            "cluster_annotations.csv",
            "cluster_summaries.csv",
            "umap_coords.csv",
            "gene_names.csv",
            "expr_matrix_f16.npy",
            "grn_matrix.npy",
            "grn_genes.csv",
        ]
    else:
        files = [
            "gcn_gene_embeddings_clusters_2.csv",
            "cluster_annotations_v2.csv",
            "cluster_summaries_v2.csv",
            "umap_coords_v2.csv",
            "gene_names_v2.csv",
            "expr_matrix_f16_v2.npy",
        ]

    paths = {}
    for f in files:
        local_path = os.path.join(LOCAL_DIR, f)
        if os.path.exists(local_path):
            paths[f] = local_path
        else:
            token = st.secrets.get("HF_TOKEN", None)
            paths[f] = hf_hub_download(repo_id=REPO_ID, filename=f, repo_type="dataset", token=token)

    emb_key = "gcn_gene_embeddings_clusters.csv" if dataset == "v1" else "gcn_gene_embeddings_clusters_2.csv"
    emb_df = pd.read_csv(paths[emb_key], index_col=0)
    genes = list(emb_df.index)
    embeddings = emb_df.drop(columns=["cluster"]).values
    clusters = emb_df["cluster"].values

    ann_key = "cluster_annotations.csv" if dataset == "v1" else "cluster_annotations_v2.csv"
    ann_df = pd.read_csv(paths[ann_key])
    label_col = "label" if "label" in ann_df.columns else "annotation"
    annotations = ann_df.set_index("cluster")[label_col].to_dict()

    sum_key = "cluster_summaries.csv" if dataset == "v1" else "cluster_summaries_v2.csv"
    summaries_df = pd.read_csv(paths[sum_key])
    sum_col = "summary" if "summary" in summaries_df.columns else summaries_df.columns[-1]
    summaries = summaries_df.set_index("cluster")[sum_col].to_dict()

    umap_key = "umap_coords.csv" if dataset == "v1" else "umap_coords_v2.csv"
    umap_df = pd.read_csv(paths[umap_key], index_col=0)

    gene_names_key = "gene_names.csv" if dataset == "v1" else "gene_names_v2.csv"
    gene_names = pd.read_csv(paths[gene_names_key])["0"].tolist()

    expr_key = "expr_matrix_f16.npy" if dataset == "v1" else "expr_matrix_f16_v2.npy"
    expr = np.load(paths[expr_key])

    grn_mat = None
    grn_genes = []
    if dataset == "v1":
        grn_mat = np.load(paths["grn_matrix.npy"])
        grn_genes = pd.read_csv(paths["grn_genes.csv"])["0"].tolist()

    return genes, embeddings, clusters, annotations, summaries, umap_df, expr, gene_names, grn_mat, grn_genes


@st.cache_resource
def load_grn(grn_key="original"):
    """Load GRN by key: 'original' (159 genes) or 'mki67' (201 genes, MKI67 program)."""
    import os
    if grn_key == "mki67":
        mat_file  = "grn_matrix_mki67.npy"
        gene_file = "grn_genes_mki67.csv"
        gene_col  = "gene"
    else:
        mat_file  = "grn_matrix.npy"
        gene_file = "grn_genes.csv"
        gene_col  = "0"

    paths = {}
    for f in [mat_file, gene_file]:
        local = os.path.join(LOCAL_DIR, f)
        if os.path.exists(local):
            paths[f] = local
        else:
            token = st.secrets.get("HF_TOKEN", None)
            paths[f] = hf_hub_download(repo_id=REPO_ID, filename=f, repo_type="dataset", token=token)

    grn_mat   = np.load(paths[mat_file])
    grn_genes = pd.read_csv(paths[gene_file])[gene_col].tolist()
    return grn_mat, grn_genes


@st.cache_resource
def load_perturbation():
    """Load BIRC5 KO perturbation data."""
    import os
    f = "birc5_ko_perturbation.csv"
    local = os.path.join(LOCAL_DIR, f)
    if os.path.exists(local):
        return pd.read_csv(local)
    token = st.secrets.get("HF_TOKEN", None)
    path = hf_hub_download(repo_id=REPO_ID, filename=f, repo_type="dataset", token=token)
    return pd.read_csv(path)


@st.cache_resource
def load_umap_expr():
    """Load per-cell WT vs KO expression on UMAP (v2: includes expr_real)."""
    import os
    f = "birc5_ko_umap_expr.csv"
    local = os.path.join(LOCAL_DIR, f)
    if os.path.exists(local):
        df = pd.read_csv(local)
        if "expr_real" in df.columns:
            return df
    token = st.secrets.get("HF_TOKEN", None)
    path = hf_hub_download(repo_id=REPO_ID, filename=f, repo_type="dataset",
                           token=token, force_download=True)
    return pd.read_csv(path)


def build_umap_perturbation(umap_expr_df, query_gene):
    """Show WT vs KO expression of query_gene on UMAP side by side."""
    import os
    # Always use v1 UMAP (MKI67 program was built on v1 data)
    umap_path = os.path.join(LOCAL_DIR, "umap_coords.csv")
    if not os.path.exists(umap_path):
        token = st.secrets.get("HF_TOKEN", None)
        umap_path = hf_hub_download(repo_id=REPO_ID, filename="umap_coords.csv",
                                     repo_type="dataset", token=token)
    umap_df = pd.read_csv(umap_path, index_col=0)

    gene_df = umap_expr_df[umap_expr_df["gene"] == query_gene].copy()
    if gene_df.empty:
        return None
    gene_df = gene_df.set_index("cell")

    # Positional alignment (same 13968 cells, same order)
    n = min(len(umap_df), len(gene_df))
    umap_sub = umap_df.iloc[:n]
    x = umap_sub["x"].values
    y = umap_sub["y"].values

    real_vals = gene_df["expr_real"].values[:n]
    wt_vals   = gene_df["expr_wt"].values[:n]
    ko_vals   = gene_df["expr_ko"].values[:n]
    all_vals = np.concatenate([real_vals, wt_vals, ko_vals])
    vmax = float(np.percentile(all_vals[all_vals > 0], 98)) if np.any(all_vals > 0) else 1.0

    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=[f"{query_gene} — Original data",
                                        f"{query_gene} — WT simulation",
                                        f"{query_gene} — BIRC5 KO simulation"])

    for col, vals in enumerate([real_vals, wt_vals, ko_vals], start=1):
        fig.add_trace(go.Scatter(
            x=x.tolist(), y=y.tolist(), mode="markers",
            marker=dict(size=2, color=vals.tolist(), colorscale="Turbo",
                        cmin=0, cmax=vmax, showscale=(col == 3),
                        colorbar=dict(thickness=8, len=0.6, x=1.01, title="")),
            showlegend=False
        ), row=1, col=col)

    fig.update_layout(
        height=380, margin=dict(l=0, r=0, t=40, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.05, x=0,
                    itemsizing="constant", font=dict(size=10))
    )
    for i in range(1, 4):
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=i)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=i)
    return fig


def build_perturbation_figures(pert_df, query_gene):
    """Build two figures: top-20 barplot + gene dynamics WT vs KO."""
    times = sorted(pert_df["time"].unique())
    last_t = times[-1]

    # ── Figure 1: Top-20 most affected genes at last timepoint ──
    targets = {"CEP55", "LPP", "PPP1R12B", "NEK2", "KIF2C", "BIRC5", "AC096732.1"}
    summary = pert_df[pert_df["time"] == last_t].copy()
    summary["abs_log2fc"] = summary["log2fc"].abs()
    top20 = summary.nlargest(20, "abs_log2fc").sort_values("log2fc")

    colors = []
    for gene, val in zip(top20["gene"], top20["log2fc"]):
        if gene in targets:
            colors.append("#FF8C00")   # orange = potential target
        elif val > 0:
            colors.append("#D45F5F")   # red = up
        else:
            colors.append("#4C72B0")   # blue = down

    # Arrow annotations for each target gene visible in top20
    annotations = []
    genes_list = top20["gene"].tolist()
    lfc_list   = top20["log2fc"].tolist()
    for gene, val in zip(genes_list, lfc_list):
        if gene in targets:
            ax_offset = 30 if val >= 0 else -30
            annotations.append(dict(
                x=val, y=gene,
                xref="x", yref="y",
                text="◀ target" if val < 0 else "target ▶",
                showarrow=True,
                arrowhead=2,
                arrowcolor="#FF8C00",
                arrowwidth=1.5,
                ax=ax_offset, ay=0,
                font=dict(color="#FF8C00", size=11),
                xanchor="left" if val >= 0 else "right",
            ))

    bar_fig = go.Figure(go.Bar(
        x=top20["log2fc"], y=top20["gene"],
        orientation="h",
        marker_color=colors,
        hovertemplate="%{y}: %{x:.3f}<extra></extra>"
    ))
    bar_fig.update_layout(
        title=f"Top 20 genes affected by BIRC5 KO (t={int(last_t)})",
        xaxis_title="log₂FC (KO / WT)",
        height=480, margin=dict(l=80, r=80, t=40, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(zeroline=True, zerolinecolor="#aaa"),
        annotations=annotations
    )

    # ── Figure 2: Dynamics of query gene WT vs KO ──
    gene_df = pert_df[pert_df["gene"] == query_gene]
    line_fig = go.Figure()
    if not gene_df.empty:
        line_fig.add_trace(go.Scatter(
            x=gene_df["time"], y=gene_df["mean_wt"],
            mode="lines+markers", name="WT",
            line=dict(color="#4C72B0", width=2)
        ))
        line_fig.add_trace(go.Scatter(
            x=gene_df["time"], y=gene_df["mean_ko"],
            mode="lines+markers", name="BIRC5 KO",
            line=dict(color="#D45F5F", width=2, dash="dash")
        ))
    line_fig.update_layout(
        title=f"{query_gene} expression: WT vs BIRC5 KO",
        xaxis_title="Time", yaxis_title="Mean expression",
        height=320, margin=dict(l=10, r=10, t=40, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=1.1)
    )
    return bar_fig, line_fig


def build_grn_from_program(grn_mat, grn_genes, gene_set):
    """Build GRN subgraph restricted to gene_set."""
    gene_set = set(gene_set) & set(grn_genes)
    G = nx.DiGraph()
    for gene in gene_set:
        idx = grn_genes.index(gene)
        row = grn_mat[idx]
        for j, w in enumerate(row):
            if abs(w) > 0 and grn_genes[j] in gene_set and grn_genes[j] != gene:
                G.add_edge(gene, grn_genes[j], weight=float(w))
    return G


def build_grn_figure(grn_mat, grn_genes, query_gene, gene_set=None, hops=1, top_n=10):
    if grn_mat is None or query_gene not in grn_genes:
        return None

    program_set = set(gene_set or [query_gene])

    # Build full GRN over ALL grn_genes (not just program)
    G_full = nx.DiGraph()
    G_full.add_node(query_gene)   # ensure node exists even with no edges
    if query_gene in grn_genes:
        idx = grn_genes.index(query_gene)
        for j, w in enumerate(grn_mat[idx]):
            if abs(w) > 0 and grn_genes[j] != query_gene:
                G_full.add_edge(query_gene, grn_genes[j], weight=float(w))
        for i, w in enumerate(grn_mat[:, idx]):
            if abs(w) > 0 and grn_genes[i] != query_gene:
                G_full.add_edge(grn_genes[i], query_gene, weight=float(w))

    # Expand hops from query through full GRN
    frontier = {query_gene}
    visited = {query_gene}
    for hop in range(hops - 1):
        next_frontier = set()
        for gene in frontier:
            if gene not in grn_genes:
                continue
            idx = grn_genes.index(gene)
            for j, w in enumerate(grn_mat[idx]):
                if abs(w) > 0 and grn_genes[j] not in visited:
                    G_full.add_edge(gene, grn_genes[j], weight=float(w))
                    next_frontier.add(grn_genes[j])
                    visited.add(grn_genes[j])
            for i, w in enumerate(grn_mat[:, idx]):
                if abs(w) > 0 and grn_genes[i] not in visited:
                    G_full.add_edge(grn_genes[i], gene, weight=float(w))
                    next_frontier.add(grn_genes[i])
                    visited.add(grn_genes[i])
        frontier = next_frontier

    # Keep only: query + intermediate nodes + program genes reachable within hops
    reachable = nx.single_source_shortest_path_length(
        G_full.to_undirected(), query_gene, cutoff=hops
    )
    # Show node if: it's the query, in program, or is intermediate on path to program gene
    nodes_to_keep = {query_gene}
    for node, dist in reachable.items():
        if node in program_set:
            # add this node and all nodes on shortest path
            try:
                path = nx.shortest_path(G_full.to_undirected(), query_gene, node)
                nodes_to_keep.update(path)
            except nx.NetworkXNoPath:
                pass

    G = G_full.subgraph(nodes_to_keep).copy()

    # Color: query=red, program genes=lightblue, intermediate=lightgray
    hop_colors = {0: "red"}
    node_colors = []
    for n in G.nodes():
        if n == query_gene:
            node_colors.append("red")
        elif n in program_set:
            node_colors.append("lightblue")
        else:
            node_colors.append("#cccccc")  # intermediate

    pos = nx.spring_layout(G, seed=42)

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_labels = list(G.nodes())

    fig = go.Figure()

    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        color = "green" if d["weight"] > 0 else "red"
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.5,
            arrowwidth=1.5, arrowcolor=color
        )

    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text",
                             text=node_labels, textposition="top center",
                             marker=dict(size=12, color=node_colors),
                             hoverinfo="text", showlegend=False))

    # Legend
    for color, label in [
        ("red",       "Query gene"),
        ("lightblue", "Program gene"),
        ("#cccccc",   "Intermediate node (not in program)"),
    ]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=12, color=color),
            name=label, showlegend=True
        ))

    fig.update_layout(
        title=f"GRN for {query_gene} — {hops}-hop ego network (green = activation, red = repression)",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left", x=0
        ),
        height=550,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    return fig


def build_grn_adjacency(grn_mat, grn_genes, gene_set, query_gene=None, hops=1):
    if grn_mat is None:
        return None
    # Use all program genes as rows/cols — even if no edges
    all_nodes = list(gene_set)
    adj = pd.DataFrame(0.0, index=all_nodes, columns=all_nodes)
    # Fill edges that exist within program
    program_set = set(gene_set) & set(grn_genes)
    for gene in program_set:
        idx = grn_genes.index(gene)
        row = grn_mat[idx]
        for j, w in enumerate(row):
            if abs(w) > 0 and grn_genes[j] in program_set and grn_genes[j] != gene:
                adj.loc[gene, grn_genes[j]] = float(w)
    return adj, all_nodes


col_title, col_badge = st.columns([6, 1])
col_title.title("Gene Program Explorer")
col_badge.markdown("<div style='padding-top:18px'>", unsafe_allow_html=True)
col_badge.badge("Beta", color="orange")
col_badge.markdown("</div>", unsafe_allow_html=True)

st.markdown("**14,581 unique gene embeddings** across 2 datasets &nbsp;·&nbsp; **359 genes with GRN** (2 models) &nbsp;·&nbsp; **1 perturbation simulation (BIRC5 KO)**", unsafe_allow_html=True)

with st.expander("About this tool"):
    st.markdown("""
This is a **RAG-based gene program retrieval system** applied to single-cell data.
Given a query gene, it retrieves co-expressed genes from a learned GNN embedding space
and maps them to LLM-annotated transcriptional programs.
It also displays the expression of the queried gene on the original cell UMAP
and its local gene regulatory network inferred by CARDAMOM.

**Available GRN models:**
- **Original** — 159 genes, inferred from full RMS scRNA-seq data
- **MKI67 program** — 201 genes (top-200 GNN neighbors of MKI67), BIRC5 KO perturbation simulated via CARDAMOM mechanistic model
    """)

    import os
    arch_path = os.path.join(LOCAL_DIR, "architecture.png")
    if os.path.exists(arch_path):
        st.image(arch_path, use_container_width=True)
    else:
        try:
            token = st.secrets.get("HF_TOKEN", None)
            arch_file = hf_hub_download(repo_id=REPO_ID, filename="architecture.png", repo_type="dataset", token=token)
            st.image(arch_file, use_container_width=True)
        except Exception:
            pass

# ================================================================
# DATASET SELECTOR
# ================================================================
dataset_choice = st.radio(
    "Dataset",
    options=["RMS original", "RMS 2"],
    horizontal=True
)
dataset_key = "v1" if dataset_choice == "RMS original" else "v2"

with st.spinner("Loading data..."):
    genes, embeddings, clusters, annotations, summaries, umap_df, expr, gene_names, grn_mat, grn_genes = load_data(dataset_key)

# ================================================================
# STATS
# ================================================================
n_genes = len(genes)
n_clusters = len(set(clusters))
n_cells = umap_df.shape[0]

col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
col1.metric("Gene embeddings", f"{n_genes:,}")
col2.metric("Gene programs", f"{n_clusters}")
col3.metric("Cells", f"{n_cells:,}")
if col4.button("🗑️ Clear history", key=f"clear_{dataset_key}"):
    st.session_state[f"messages_{dataset_key}"] = []
    st.session_state[f"last_selected_{dataset_key}"] = ""
    st.rerun()

col_search, col_slider, col_grn_slider = st.columns([3, 2, 2])

# ── GRN selector ──────────────────────────────────────────────
grn_choice = st.radio(
    "GRN model",
    options=["MKI67 program (201 genes, BIRC5 KO)", "Original (159 genes)"],
    horizontal=True,
    key=f"grn_choice_{dataset_key}"
)
grn_key = "mki67" if grn_choice.startswith("MKI67") else "original"
with st.spinner("Loading GRN..."):
    grn_mat, grn_genes = load_grn(grn_key)

grn_gene_set = set(grn_genes) if grn_genes else set()
def gene_label(g):
    return f"🔬 {g}" if g in grn_gene_set else g
gene_options = [""] + sorted(genes)
gene_labels = [""] + [gene_label(g) for g in sorted(genes)]
selected_label = col_search.selectbox(
    "Quick gene search  (🔬 = GRN available)",
    options=gene_labels,
    index=0,
    key=f"selectbox_{dataset_key}"
)
selected_gene = selected_label.replace("🔬 ", "") if selected_label else ""
program_size = col_slider.slider(
    "Program size (neighbors)",
    min_value=5, max_value=200, value=20, step=5,
    key=f"slider_{dataset_key}"
)
grn_hops = col_grn_slider.slider(
    "GRN hops (ego-network depth)",
    min_value=1, max_value=3, value=1, step=1,
    key=f"grn_slider_{dataset_key}"
)

if f"messages_{dataset_key}" not in st.session_state:
    st.session_state[f"messages_{dataset_key}"] = []

if f"last_selected_{dataset_key}" not in st.session_state:
    st.session_state[f"last_selected_{dataset_key}"] = ""

if f"default_run_{dataset_key}" not in st.session_state:
    st.session_state[f"default_run_{dataset_key}"] = True

messages = st.session_state[f"messages_{dataset_key}"]

for msg in messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "df" in msg:
            st.dataframe(msg["df"], use_container_width=True)
            if "cluster_id" in msg:
                summary = summaries.get(msg["cluster_id"], "")
                if summary:
                    with st.popover("Cluster annotation details"):
                        st.markdown(summary)
        msg_id = id(msg)
        # Expression UMAP — full width, taller
        if msg.get("fig") is not None:
            fig_expr = msg["fig"]
            fig_expr.update_layout(height=550)
            st.plotly_chart(fig_expr, use_container_width=True, key=f"{msg_id}_fig")
        # Time + Cell type UMAPs — side by side, smaller
        side_figs = [(k, msg.get(k)) for k in ["fig_time", "fig_celltype"] if msg.get(k) is not None]
        if side_figs:
            cols = st.columns(len(side_figs))
            for col, (k, f) in zip(cols, side_figs):
                f.update_layout(height=350)
                col.plotly_chart(f, use_container_width=True, key=f"{msg_id}_{k}")
        if "grn_fig" in msg and msg["grn_fig"] is not None:
            tab_pert, tab_graph, tab_matrix = st.tabs(["🧬 BIRC5 KO Perturbation", "Network graph", "Adjacency matrix"])
            with tab_graph:
                st.plotly_chart(msg["grn_fig"], use_container_width=True, key=f"{msg_id}_grn")
            with tab_matrix:
                if "grn_adj" in msg and msg["grn_adj"] is not None:
                    adj_df, genes_list = msg["grn_adj"]
                    vals = adj_df.values.flatten()
                    nonzero = vals[np.abs(vals) > 1e-4]
                    if len(nonzero) > 0:
                        vmax = float(np.percentile(np.abs(nonzero), 95))
                    else:
                        vmax = float(adj_df.abs().values.max()) or 1.0
                    adj_fig = px.imshow(
                        adj_df,
                        color_continuous_scale="RdBu_r",
                        color_continuous_midpoint=0,
                        zmin=-vmax, zmax=vmax,
                        title="Adjacency matrix (red=activation, blue=repression, clipped at 95th percentile)",
                        height=600,
                        aspect="auto"
                    )
                    adj_fig.update_layout(
                        xaxis=dict(tickfont=dict(size=9)),
                        yaxis=dict(tickfont=dict(size=9))
                    )
                    st.plotly_chart(adj_fig, use_container_width=True, key=f"{msg_id}_adj")
            with tab_pert:
                try:
                    pert_df = load_perturbation()
                    q_gene  = msg.get("query_gene", "MKI67")
                    # Dynamics + barplot
                    bar_fig, line_fig = build_perturbation_figures(pert_df, q_gene)
                    st.plotly_chart(line_fig, use_container_width=True, key=f"{msg_id}_pert_line")
                    st.plotly_chart(bar_fig,  use_container_width=True, key=f"{msg_id}_pert_bar")
                    st.caption("Simulation: CARDAMOM mechanistic model · MKI67 program (201 genes) · BIRC5 knocked out")
                    # ── Potential therapeutic targets ──────────────────────
                except Exception as e:
                    st.info(f"Perturbation data available only for MKI67 program GRN. ({e})")

if st.session_state.get(f"default_run_{dataset_key}"):
    st.session_state[f"default_run_{dataset_key}"] = False
    query_gene = "BIRC5" if "BIRC5" in genes else genes[0]
elif selected_gene and selected_gene != st.session_state[f"last_selected_{dataset_key}"]:
    st.session_state[f"last_selected_{dataset_key}"] = selected_gene
    query_gene = selected_gene
elif query_gene := st.chat_input("Enter a gene name (e.g. MKI67, BIRC5, FOXP3, MYC)", key=f"chat_{dataset_key}"):
    pass
else:
    query_gene = None

if query_gene:
    messages.append({"role": "user", "content": query_gene})
    query_gene = query_gene.strip().upper()

    if query_gene not in genes:
        response = f"Gene {query_gene} not found in the co-expression graph."
        messages.append({"role": "assistant", "content": response})
        st.rerun()
    else:
        idx = genes.index(query_gene)
        target_emb = embeddings[idx].reshape(1, -1)

        sims = cosine_similarity(target_emb, embeddings)[0]
        sorted_idx = np.argsort(sims)[::-1]
        sorted_idx = [i for i in sorted_idx if genes[i] != query_gene][:program_size]

        top_genes = [
            (genes[i], round(sims[i], 4), annotations.get(int(clusters[i]), ""))
            for i in sorted_idx
        ]
        df = pd.DataFrame(top_genes, columns=["Gene", "Similarity", "Cluster annotation"])

        query_cluster = int(clusters[idx])
        query_annotation = annotations.get(query_cluster, "")

        fig = None
        fig_time = None
        fig_celltype = None
        if query_gene in gene_names:
            gene_idx = gene_names.index(query_gene)
            expr_vals = expr[:, gene_idx].astype(float)
            plot_df = umap_df.copy()
            plot_df["expression"] = expr_vals

            fig = px.scatter(
                plot_df, x="x", y="y",
                color="expression",
                color_continuous_scale="Viridis",
                title=f"{query_gene} — Expression",
                labels={"x": "UMAP 1", "y": "UMAP 2"},
                opacity=0.6, height=450
            )
            fig.update_traces(marker=dict(size=3))
            fig.update_layout(coloraxis_colorbar=dict(title="Expression"))

        if "time" in umap_df.columns:
            fig_time = px.scatter(
                umap_df, x="x", y="y",
                color=umap_df["time"].astype(str),
                title="Time",
                labels={"x": "UMAP 1", "y": "UMAP 2", "color": "Time"},
                opacity=0.6, height=450,
                category_orders={"color": [str(t) for t in sorted(umap_df["time"].unique())]}
            )
            fig_time.update_traces(marker=dict(size=3))

        if "cell_type" in umap_df.columns:
            fig_celltype = px.scatter(
                umap_df, x="x", y="y",
                color="cell_type",
                title="Cell Type",
                labels={"x": "UMAP 1", "y": "UMAP 2", "color": "Cell Type"},
                opacity=0.6, height=450
            )
            fig_celltype.update_traces(marker=dict(size=3))

        program_genes = [query_gene] + [genes[i] for i in sorted_idx]
        grn_fig = build_grn_figure(grn_mat, grn_genes, query_gene, gene_set=program_genes, hops=grn_hops)
        grn_adj = build_grn_adjacency(grn_mat, grn_genes, gene_set=program_genes, query_gene=query_gene, hops=grn_hops)

        messages.append({
            "role": "assistant",
            "content": f"**Gene program for {query_gene}** — cluster {query_cluster}: *{query_annotation}*",
            "df": df,
            "cluster_id": query_cluster,
            "query_gene": query_gene,
            "fig": fig,
            "fig_time": fig_time,
            "fig_celltype": fig_celltype,
            "grn_fig": grn_fig,
            "grn_adj": grn_adj
        })
        st.rerun()
