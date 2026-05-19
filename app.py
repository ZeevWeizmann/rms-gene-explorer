import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import hf_hub_download

REPO_ID = "weizmannzeev/rms-gene-programs"

@st.cache_resource
def load_data():
    files = [
        "gcn_gene_embeddings_clusters.csv",
        "cluster_annotations.csv",
        "umap_coords.csv",
        "gene_names.csv",
        "expr_matrix_f16.npy",
    ]
    paths = {}
    for f in files:
        paths[f] = hf_hub_download(repo_id=REPO_ID, filename=f, repo_type="dataset")

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

    return genes, embeddings, clusters, annotations, umap_df, expr, gene_names

st.title("Gene Program Explorer")

with st.spinner("Loading data..."):
    genes, embeddings, clusters, annotations, umap_df, expr, gene_names = load_data()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "df" in msg:
            st.dataframe(msg["df"], use_container_width=True)
        if "fig" in msg and msg["fig"] is not None:
            st.plotly_chart(msg["fig"], use_container_width=True)

if query_gene := st.chat_input("Enter a gene name (e.g. MKI67, BIRC5, FOXP3, MYC)"):

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
            (genes[i], round(sims[i], 4), int(clusters[i]), annotations.get(int(clusters[i]), ""))
            for i in sorted_idx
        ]
        df = pd.DataFrame(top_genes, columns=["Gene", "Similarity", "Cluster", "Program"])

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

        st.session_state.messages.append({
            "role": "assistant",
            "content": f"**Gene program for {query_gene}** — cluster {query_cluster}: *{query_annotation}*",
            "df": df,
            "fig": fig
        })
        st.rerun()
