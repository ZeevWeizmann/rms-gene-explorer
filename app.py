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
def load_data():
    import os
    files = [
        "gcn_gene_embeddings_clusters.csv",
        "cluster_annotations.csv",
        "umap_coords.csv",
        "gene_names.csv",
        "expr_matrix_f16.npy",
        "grn_matrix.npy",
        "grn_genes.csv",
    ]
    paths = {}
    for f in files:
        local_path = os.path.join(LOCAL_DIR, f)
        if os.path.exists(local_path):
            paths[f] = local_path
        else:
            token = st.secrets.get("HF_TOKEN", None)
            paths[f] = hf_hub_download(repo_id=REPO_ID, filename=f, repo_type="dataset", token=token)

    emb_df = pd.read_csv(paths["gcn_gene_embeddings_clusters.csv"], index_col=0)
    genes = list(emb_df.index)
    embeddings = emb_df.drop(columns=["cluster"]).values
    clusters = emb_df["cluster"].values

    ann_df = pd.read_csv(paths["cluster_annotations.csv"])
    label_col = "label" if "label" in ann_df.columns else "annotation"
    annotations = ann_df.set_index("cluster")[label_col].to_dict()

    umap_df = pd.read_csv(paths["umap_coords.csv"], index_col=0)
    gene_names = pd.read_csv(paths["gene_names.csv"])["0"].tolist()
    expr = np.load(paths["expr_matrix_f16.npy"])
    grn_mat = np.load(paths["grn_matrix.npy"])
    grn_genes = pd.read_csv(paths["grn_genes.csv"])["0"].tolist()

    return genes, embeddings, clusters, annotations, umap_df, expr, gene_names, grn_mat, grn_genes

st.title("Gene Program Explorer")
st.badge("Beta", color="orange")

with st.spinner("Loading data..."):
    genes, embeddings, clusters, annotations, umap_df, expr, gene_names, grn_mat, grn_genes = load_data()

def build_grn_figure(grn_mat, grn_genes, query_gene, top_n=10):
    if query_gene not in grn_genes:
        return None
    idx = grn_genes.index(query_gene)
    row = grn_mat[idx]
    col = grn_mat[:, idx]

    top_targets = np.argsort(np.abs(row))[::-1][:top_n]
    top_regulators = np.argsort(np.abs(col))[::-1][:top_n]

    G = nx.DiGraph()
    G.add_node(query_gene)
    for i in top_targets:
        if grn_genes[i] != query_gene:
            G.add_edge(query_gene, grn_genes[i], weight=float(row[i]))
    for i in top_regulators:
        if grn_genes[i] != query_gene:
            G.add_edge(grn_genes[i], query_gene, weight=float(col[i]))

    pos = nx.spring_layout(G, seed=42)

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_labels = list(G.nodes())
    node_colors = ["red" if n == query_gene else "lightblue" for n in G.nodes()]

    fig = go.Figure()

    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        color = "green" if d["weight"] > 0 else "red"
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.5,
            arrowwidth=1.5,
            arrowcolor=color
        )

    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text",
                             text=node_labels, textposition="top center",
                             marker=dict(size=12, color=node_colors),
                             hoverinfo="text"))
    fig.update_layout(
        title=f"GRN for {query_gene} (green = activation, red = repression)",
        showlegend=False,
        height=500,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    return fig

selected_gene = st.selectbox(
    "Quick gene search",
    options=[""] + sorted(genes),
    index=0
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_selected" not in st.session_state:
    st.session_state.last_selected = ""

if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.default_run = True

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "df" in msg:
            st.dataframe(msg["df"], use_container_width=True)
        if "fig" in msg and msg["fig"] is not None:
            st.plotly_chart(msg["fig"], use_container_width=True)
        if "grn_fig" in msg and msg["grn_fig"] is not None:
            st.plotly_chart(msg["grn_fig"], use_container_width=True)

if st.session_state.get("default_run"):
    st.session_state.default_run = False
    query_gene = "NACA"
elif selected_gene and selected_gene != st.session_state.last_selected:
    st.session_state.last_selected = selected_gene
    query_gene = selected_gene
elif query_gene := st.chat_input("Enter a gene name (e.g. MKI67, BIRC5, FOXP3, MYC)"):
    pass
else:
    query_gene = None

if query_gene:

    st.session_state.messages.append({"role": "user", "content": query_gene})
    query_gene = query_gene.strip().upper()

    if query_gene not in genes:
        response = f"Gene {query_gene} not found in the co-expression graph."
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
    else:
        idx = genes.index(query_gene)
        target_emb = embeddings[idx].reshape(1, -1)

        sims = cosine_similarity(target_emb, embeddings)[0]
        sorted_idx = np.argsort(sims)[::-1]
        sorted_idx = [i for i in sorted_idx if genes[i] != query_gene][:20]

        top_genes = [
            (genes[i], round(sims[i], 4), annotations.get(int(clusters[i]), ""))
            for i in sorted_idx
        ]
        df = pd.DataFrame(top_genes, columns=["Gene", "Similarity", "Cluster annotation"])

        query_cluster = int(clusters[idx])
        query_annotation = annotations.get(query_cluster, "")

        fig = None
        if query_gene in gene_names:
            gene_idx = gene_names.index(query_gene)
            expr_vals = expr[:, gene_idx].astype(float)
            plot_df = umap_df.copy()
            plot_df["expression"] = expr_vals
            fig = px.scatter(
                plot_df,
                x="x", y="y",
                color="expression",
                color_continuous_scale="Viridis",
                title=f"{query_gene} expression",
                labels={"x": "UMAP 1", "y": "UMAP 2"},
                opacity=0.6,
                height=500
            )
            fig.update_traces(marker=dict(size=3))
            fig.update_layout(coloraxis_colorbar=dict(title="Expression"))

        grn_fig = build_grn_figure(grn_mat, grn_genes, query_gene)

        st.session_state.messages.append({
            "role": "assistant",
            "content": f"**Gene program for {query_gene}** — cluster {query_cluster}: *{query_annotation}*",
            "df": df,
            "fig": fig,
            "grn_fig": grn_fig
        })
        st.rerun()
