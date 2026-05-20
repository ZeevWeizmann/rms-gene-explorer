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


def build_grn_figure(grn_mat, grn_genes, query_gene, hops=1, top_n=10):
    if grn_mat is None or query_gene not in grn_genes:
        return None

    # Build full directed graph (top_n edges per gene to keep it manageable)
    G_full = nx.DiGraph()
    for i, g in enumerate(grn_genes):
        row = grn_mat[i]
        top_targets = np.argsort(np.abs(row))[::-1][:top_n]
        for j in top_targets:
            if i != j and abs(row[j]) > 0:
                G_full.add_edge(g, grn_genes[j], weight=float(row[j]))

    # Ego graph with requested radius
    G = nx.ego_graph(G_full, query_gene, radius=hops, undirected=True)

    # Color by hop distance
    hop_colors = {0: "red", 1: "lightblue", 2: "#90EE90", 3: "#FFD700"}
    lengths = nx.single_source_shortest_path_length(G.to_undirected(), query_gene)
    node_colors = [hop_colors.get(lengths.get(n, 3), "#cccccc") for n in G.nodes()]

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
                             hoverinfo="text"))
    fig.update_layout(
        title=f"GRN for {query_gene} — {hops}-hop ego network (green = activation, red = repression)",
        showlegend=False, height=500,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    return fig


col_title, col_badge = st.columns([6, 1])
col_title.title("Gene Program Explorer")
col_badge.markdown("<div style='padding-top:18px'>", unsafe_allow_html=True)
col_badge.badge("Beta", color="orange")
col_badge.markdown("</div>", unsafe_allow_html=True)

st.markdown("**14,581 unique gene embeddings** across 2 datasets")

with st.expander("About this tool"):
    st.markdown("""
This is a **RAG-based gene program retrieval system** applied to single-cell data.
Given a query gene, it retrieves co-expressed genes from a learned GNN embedding space
and maps them to LLM-annotated transcriptional programs.
It also displays the expression of the queried gene on the original cell UMAP
and its local gene regulatory network inferred by CARDAMOM.
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
selected_gene = col_search.selectbox(
    "Quick gene search",
    options=[""] + sorted(genes),
    index=0,
    key=f"selectbox_{dataset_key}"
)
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
        figs = [(k, msg.get(k)) for k in ["fig", "fig_time", "fig_celltype"] if msg.get(k) is not None]
        if figs:
            cols = st.columns(len(figs))
            for col, (k, f) in zip(cols, figs):
                col.plotly_chart(f, use_container_width=True, key=f"{msg_id}_{k}")
        if "grn_fig" in msg and msg["grn_fig"] is not None:
            st.plotly_chart(msg["grn_fig"], use_container_width=True, key=f"{msg_id}_grn")

if st.session_state.get(f"default_run_{dataset_key}"):
    st.session_state[f"default_run_{dataset_key}"] = False
    query_gene = "NACA" if "NACA" in genes else genes[0]
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

        grn_fig = build_grn_figure(grn_mat, grn_genes, query_gene, hops=grn_hops)

        messages.append({
            "role": "assistant",
            "content": f"**Gene program for {query_gene}** — cluster {query_cluster}: *{query_annotation}*",
            "df": df,
            "cluster_id": query_cluster,
            "fig": fig,
            "fig_time": fig_time,
            "fig_celltype": fig_celltype,
            "grn_fig": grn_fig
        })
        st.rerun()
