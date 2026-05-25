import streamlit as st
import streamlit.components.v1 as _components
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Gene Program Explorer",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Detect mobile via JS → session_state ────────────────────────
st.markdown("""
<style>
/* Stack columns on narrow screens */
@media screen and (max-width: 640px) {
    div[data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }
    div[data-testid="stChatInput"] { width: 100% !important; }
}
section.main > div { max-width: 900px; margin: auto; }
</style>
<script>
(function() {
    const w = window.innerWidth;
    const isMobile = w < 700;
    // inject into Streamlit via URL hash trick
    window.parent.postMessage({type:"streamlit:setComponentValue", value: isMobile}, "*");
})();
</script>
""", unsafe_allow_html=True)

# simple mobile detection via screen width stored in session_state
if "is_mobile" not in st.session_state:
    st.session_state["is_mobile"] = False   # default desktop, JS updates on rerun

is_mobile = st.session_state.get("is_mobile", False)
CHART_H       = 320 if is_mobile else 520
CHART_H_SMALL = 260 if is_mobile else 380
LOGO_W        = 55  if is_mobile else 90

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


@st.cache_resource(show_spinner=False)
def load_data_traj():
    """Load trajectory GNN embeddings (beta). Falls back gracefully if file missing."""
    import os
    local = os.path.join(LOCAL_DIR, "trajectory_gene_embeddings.csv")
    ann_local = os.path.join(LOCAL_DIR, "trajectory_cluster_annotations.csv")
    sum_local = os.path.join(LOCAL_DIR, "trajectory_cluster_summaries.csv")

    try:
        emb_df = pd.read_csv(local, index_col=0) if os.path.exists(local) else pd.read_csv(
            hf_hub_download(repo_id=REPO_ID, filename="trajectory_gene_embeddings.csv",
                            repo_type="dataset", token=st.secrets.get("HF_TOKEN", None)), index_col=0)
        genes = list(emb_df.index)
        embeddings = emb_df.drop(columns=["cluster"]).values
        clusters = emb_df["cluster"].values

        try:
            ann_df = pd.read_csv(ann_local if os.path.exists(ann_local) else
                hf_hub_download(repo_id=REPO_ID, filename="trajectory_cluster_annotations.csv",
                                repo_type="dataset", token=st.secrets.get("HF_TOKEN", None)))
            label_col = "label" if "label" in ann_df.columns else ann_df.columns[-1]
            annotations = ann_df.set_index("cluster")[label_col].to_dict()
        except Exception:
            annotations = {i: f"Cluster {i}" for i in range(int(clusters.max()) + 1)}

        try:
            sum_df = pd.read_csv(sum_local if os.path.exists(sum_local) else
                hf_hub_download(repo_id=REPO_ID, filename="trajectory_cluster_summaries.csv",
                                repo_type="dataset", token=st.secrets.get("HF_TOKEN", None)))
            sum_col = "summary" if "summary" in sum_df.columns else sum_df.columns[-1]
            summaries = sum_df.set_index("cluster")[sum_col].to_dict()
        except Exception:
            summaries = {i: f"Trajectory cluster {i}" for i in range(int(clusters.max()) + 1)}

        # No UMAP, expr matrix, or GRN for trajectory beta
        umap_df    = pd.DataFrame({"x": np.zeros(len(genes)), "y": np.zeros(len(genes))},
                                  index=genes)
        expr       = np.zeros((len(genes), 1), dtype=np.float16)
        gene_names = genes

        return genes, embeddings, clusters, annotations, summaries, umap_df, expr, gene_names, None, []
    except Exception as e:
        st.warning(f"Trajectory embeddings not available yet: {e}. Run trajectory_gnn.ipynb first.")
        return [], np.zeros((0, 128)), np.zeros(0), {}, {}, \
               pd.DataFrame({"x": [], "y": []}), np.zeros((0, 1), dtype=np.float16), [], None, []


@st.cache_resource
def load_grn(grn_key="original"):
    """Load GRN by key: 'original' (159 genes), 'mki67' (201 genes), 'tubb' (201 genes), 'foxm1' (198 genes), 'full' (200 genes)."""
    import os
    if grn_key == "mki67":
        mat_file  = "grn_matrix_mki67.npy"
        gene_file = "grn_genes_mki67.csv"
        gene_col  = "gene"
    elif grn_key == "tubb":
        mat_file  = "grn_matrix_tubb.npy"
        gene_file = "grn_genes_tubb.csv"
        gene_col  = "gene"
    elif grn_key == "foxm1":
        mat_file  = "grn_matrix_foxm1.npy"
        gene_file = "grn_genes_foxm1.csv"
        gene_col  = "gene"
    elif grn_key == "full":
        mat_file  = "grn_matrix_full.npy"
        gene_file = "grn_genes_full.csv"
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
def load_grn_gene_list(grn_key="original"):
    """Load only the gene list (no matrix) — cheap, used to decide which GRN options to show."""
    import os
    if grn_key == "mki67":
        gene_file, gene_col = "grn_genes_mki67.csv", "gene"
    elif grn_key == "tubb":
        gene_file, gene_col = "grn_genes_tubb.csv", "gene"
    elif grn_key == "foxm1":
        gene_file, gene_col = "grn_genes_foxm1.csv", "gene"
    elif grn_key == "full":
        gene_file, gene_col = "grn_genes_full.csv", "gene"
    else:
        gene_file, gene_col = "grn_genes.csv", "0"
    local = os.path.join(LOCAL_DIR, gene_file)
    if os.path.exists(local):
        return set(pd.read_csv(local)[gene_col].tolist())
    token = st.secrets.get("HF_TOKEN", None)
    path = hf_hub_download(repo_id=REPO_ID, filename=gene_file, repo_type="dataset", token=token)
    return set(pd.read_csv(path)[gene_col].tolist())


@st.cache_resource
def load_perturbation(grn_key="mki67"):
    """Load KO perturbation data for the given GRN model."""
    import os
    if grn_key == "tubb":
        f = "tubb_ko_perturbation.csv"
    elif grn_key == "foxm1":
        f = "foxm1_ko_perturbation.csv"
    elif grn_key == "full":
        f = "full_ko_perturbation.csv"
    else:
        f = "birc5_ko_perturbation.csv"
    local = os.path.join(LOCAL_DIR, f)
    if os.path.exists(local):
        return pd.read_csv(local)
    token = st.secrets.get("HF_TOKEN", None)
    path = hf_hub_download(repo_id=REPO_ID, filename=f, repo_type="dataset", token=token)
    return pd.read_csv(path)


@st.cache_resource
def load_real_expr_means():
    """Load pre-computed per-gene mean expression (~158KB CSV, not the full 225MB matrix)."""
    import os
    local = os.path.join(LOCAL_DIR, "expr_gene_means.csv")
    if os.path.exists(local):
        df = pd.read_csv(local)
    else:
        token = st.secrets.get("HF_TOKEN", None)
        path  = hf_hub_download(repo_id=REPO_ID, filename="expr_gene_means.csv",
                                repo_type="dataset", token=token)
        df = pd.read_csv(path)
    return dict(zip(df["gene"].tolist(), df["mean_expr"].tolist()))


@st.cache_resource
def load_foxm1_umap_populations():
    """Per-cell UMAP coords + cell type + time + FOXM1/NUSAP1/PABPN1 WT vs KO (v3)."""
    import os
    # v3 adds 'time' column from real data
    for fname in ["foxm1_umap_pop_v3.csv", "foxm1_umap_pop_v2.csv"]:
        local = os.path.join(LOCAL_DIR, fname)
        if os.path.exists(local):
            return pd.read_csv(local)
    token = st.secrets.get("HF_TOKEN", None)
    path = hf_hub_download(repo_id=REPO_ID, filename="foxm1_umap_pop_v3.csv",
                           repo_type="dataset", token=token)
    return pd.read_csv(path)


@st.cache_resource
def load_celltype_gene_means():
    """Per-gene per-cell-type mean expression WT vs KO (FOXM1 model, 45KB)."""
    import os
    f = "foxm1_celltype_gene_means.csv"
    local = os.path.join(LOCAL_DIR, f)
    if os.path.exists(local):
        return pd.read_csv(local)
    token = st.secrets.get("HF_TOKEN", None)
    path = hf_hub_download(repo_id=REPO_ID, filename=f, repo_type="dataset", token=token)
    return pd.read_csv(path)


@st.cache_resource
def load_foxm1_sim_umap_proj():
    """Simulation cells projected onto real UMAP via UMAP transform (v2, 177 KB).
    Columns: x_wt, y_wt, x_ko, y_ko, time."""
    import os
    for fname in ["foxm1_sim_umap_proj_v2.csv", "foxm1_sim_umap_proj.csv"]:
        local = os.path.join(LOCAL_DIR, fname)
        if os.path.exists(local):
            return pd.read_csv(local)
    token = st.secrets.get("HF_TOKEN", None)
    path = hf_hub_download(repo_id=REPO_ID, filename="foxm1_sim_umap_proj_v2.csv",
                           repo_type="dataset", token=token)
    return pd.read_csv(path)


@st.cache_resource
def load_mki67_sim_umap_proj():
    """Simulation cells (mki67 model) projected onto real UMAP via UMAP transform.
    Columns: x_wt, y_wt, x_ko, y_ko, time."""
    import os
    local = os.path.join(LOCAL_DIR, "mki67_sim_umap_proj_v3.csv")
    if os.path.exists(local):
        return pd.read_csv(local)
    token = st.secrets.get("HF_TOKEN", None)
    path = hf_hub_download(repo_id=REPO_ID, filename="mki67_sim_umap_proj_v3.csv",
                           repo_type="dataset", token=token)
    return pd.read_csv(path)


@st.cache_resource
def load_tubb_sim_umap_proj():
    """Simulation cells (tubb model) projected onto real UMAP via UMAP transform.
    Columns: x_wt, y_wt, x_ko, y_ko, time."""
    import os
    local = os.path.join(LOCAL_DIR, "tubb_sim_umap_proj_v4.csv")
    if os.path.exists(local):
        return pd.read_csv(local)
    token = st.secrets.get("HF_TOKEN", None)
    path = hf_hub_download(repo_id=REPO_ID, filename="tubb_sim_umap_proj_v4.csv",
                           repo_type="dataset", token=token)
    return pd.read_csv(path)


@st.cache_resource
def load_original_sim_umap_proj():
    """Original GRN (159 genes) WT simulation projected onto real UMAP.
    Columns: x, y, time."""
    import os
    local = os.path.join(LOCAL_DIR, "original_sim_umap_proj.csv")
    if os.path.exists(local):
        return pd.read_csv(local)
    token = st.secrets.get("HF_TOKEN", None)
    path = hf_hub_download(repo_id=REPO_ID, filename="original_sim_umap_proj.csv",
                           repo_type="dataset", token=token)
    return pd.read_csv(path)


@st.cache_resource
def load_foxm1_pop_real():
    """Real cells scored by proliferative/quiescent gene programs (100+100 genes).
    Columns: x, y, population, score_prolif, score_quies, cell_type, time."""
    import os
    f = "foxm1_pop_real_scored.csv"
    local = os.path.join(LOCAL_DIR, f)
    if os.path.exists(local):
        return pd.read_csv(local)
    token = st.secrets.get("HF_TOKEN", None)
    path = hf_hub_download(repo_id=REPO_ID, filename=f, repo_type="dataset", token=token)
    return pd.read_csv(path)


@st.cache_resource
def load_foxm1_pop_sim():
    """Sim cells (WT+KO) scored by proliferative/quiescent gene programs.
    Columns: x_wt, y_wt, x_ko, y_ko, pop_wt, pop_ko, time."""
    import os
    f = "foxm1_pop_sim_scored.csv"
    local = os.path.join(LOCAL_DIR, f)
    if os.path.exists(local):
        return pd.read_csv(local)
    token = st.secrets.get("HF_TOKEN", None)
    path = hf_hub_download(repo_id=REPO_ID, filename=f, repo_type="dataset", token=token)
    return pd.read_csv(path)


@st.cache_resource
def load_tubb_pop_sim():
    """TUBB sim cells (WT+KO) scored by proliferative/quiescent gene programs.
    Columns: x_wt, y_wt, x_ko, y_ko, pop_wt, pop_ko, time."""
    import os
    f = "tubb_pop_sim_scored_v3.csv"
    local = os.path.join(LOCAL_DIR, f)
    if os.path.exists(local):
        return pd.read_csv(local)
    token = st.secrets.get("HF_TOKEN", None)
    path = hf_hub_download(repo_id=REPO_ID, filename=f, repo_type="dataset", token=token)
    return pd.read_csv(path)


@st.cache_resource
def load_mki67_pop_sim():
    """MKI67 sim cells (WT+KO) scored by proliferative program (proxy populations).
    Columns: x_wt, y_wt, x_ko, y_ko, pop_wt, pop_ko, time."""
    import os
    f = "mki67_pop_sim_scored.csv"
    local = os.path.join(LOCAL_DIR, f)
    if os.path.exists(local):
        return pd.read_csv(local)
    token = st.secrets.get("HF_TOKEN", None)
    path = hf_hub_download(repo_id=REPO_ID, filename=f, repo_type="dataset", token=token)
    return pd.read_csv(path)


@st.cache_resource
def load_full_sim_umap_proj():
    """Full program (200 genes) WT+KO simulation projected onto real UMAP.
    Columns: x_wt, y_wt, x_ko, y_ko, time."""
    import os
    f = "full_sim_umap_proj.csv"
    local = os.path.join(LOCAL_DIR, f)
    if os.path.exists(local): return pd.read_csv(local)
    token = st.secrets.get("HF_TOKEN", None)
    path = hf_hub_download(repo_id=REPO_ID, filename=f, repo_type="dataset", token=token)
    return pd.read_csv(path)


@st.cache_resource
def load_full_pop_sim():
    """Full program (200 genes) sim cells scored by population (WT+KO).
    Columns: x_wt, y_wt, x_ko, y_ko, pop_wt, pop_ko, time."""
    import os
    f = "full_pop_sim_scored.csv"
    local = os.path.join(LOCAL_DIR, f)
    if os.path.exists(local): return pd.read_csv(local)
    token = st.secrets.get("HF_TOKEN", None)
    path = hf_hub_download(repo_id=REPO_ID, filename=f, repo_type="dataset", token=token)
    return pd.read_csv(path)


@st.cache_resource
def load_foxm1_real_timecourse():
    """Real data mean expression per gene per timepoint (93 KB, 1188 rows)."""
    import os
    f = "foxm1_real_timecourse.csv"
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


def build_foxm1_timecourse(pert_df, real_tc_df):
    """Compare simulation WT vs real data expression over time for key genes.

    pert_df      — foxm1_ko_perturbation.csv  (gene, time, mean_wt, mean_ko)
    real_tc_df   — foxm1_real_timecourse.csv  (gene, time, mean_expr_all)

    Shows 2 rows × 3 cols = 6 panels for key genes.
    Each panel: solid line = real data, dashed line = sim WT, dotted = sim KO.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    SHOW_GENES = ["FOXM1", "NUSAP1", "TOP2A", "PABPN1", "EIF2A", "BIRC5"]
    TITLES     = ["FOXM1<br><sup>KO target</sup>",
                  "NUSAP1<br><sup>mitotic (direct target)</sup>",
                  "TOP2A<br><sup>mitotic marker</sup>",
                  "PABPN1<br><sup>mRNA processing</sup>",
                  "EIF2A<br><sup>stress / quiescence</sup>",
                  "BIRC5<br><sup>anti-apoptosis</sup>"]

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=TITLES,
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )

    positions = [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]

    for idx, (gene, title) in enumerate(zip(SHOW_GENES, TITLES)):
        row, col = positions[idx]
        show_leg = (idx == 0)

        # Real data
        real_sub = real_tc_df[real_tc_df["gene"] == gene].sort_values("time")
        # Sim WT
        sim_sub  = pert_df[pert_df["gene"] == gene].sort_values("time")

        if not real_sub.empty:
            fig.add_trace(go.Scatter(
                x=real_sub["time"], y=real_sub["mean_expr_all"],
                mode="lines+markers",
                line=dict(color="#2196F3", width=2.5),
                marker=dict(size=5),
                name="Real data",
                legendgroup="real",
                showlegend=show_leg,
                hovertemplate=f"<b>{gene}</b> real<br>t=%{{x}}  expr=%{{y:.3f}}<extra></extra>",
            ), row=row, col=col)

        if not sim_sub.empty:
            fig.add_trace(go.Scatter(
                x=sim_sub["time"], y=sim_sub["mean_wt"],
                mode="lines+markers",
                line=dict(color="#FF7043", width=2, dash="dash"),
                marker=dict(size=5, symbol="diamond"),
                name="Simulation WT",
                legendgroup="sim_wt",
                showlegend=show_leg,
                hovertemplate=f"<b>{gene}</b> sim WT<br>t=%{{x}}  expr=%{{y:.3f}}<extra></extra>",
            ), row=row, col=col)

            fig.add_trace(go.Scatter(
                x=sim_sub["time"], y=sim_sub["mean_ko"],
                mode="lines+markers",
                line=dict(color="#9C27B0", width=1.8, dash="dot"),
                marker=dict(size=4, symbol="x"),
                name="Simulation KO",
                legendgroup="sim_ko",
                showlegend=show_leg,
                hovertemplate=f"<b>{gene}</b> sim KO<br>t=%{{x}}  expr=%{{y:.3f}}<extra></extra>",
            ), row=row, col=col)

    fig.update_layout(
        height=480,
        margin=dict(t=60, b=10, l=50, r=15),
        plot_bgcolor="white",
        paper_bgcolor="white",
        title=dict(
            text="Gene expression over time — real data vs CARDAMOM simulation",
            font=dict(size=13), x=0.5,
        ),
        legend=dict(
            orientation="h", yanchor="top", y=-0.04,
            xanchor="center", x=0.5, font=dict(size=11),
        ),
    )
    for r in [1, 2]:
        for c in [1, 2, 3]:
            fig.update_xaxes(
                title_text="Time (h)" if r == 2 else "",
                tickvals=[0, 16, 32, 48, 64, 80],
                gridcolor="#eeeeee", showgrid=True,
                row=r, col=c,
            )
            fig.update_yaxes(
                title_text="Mean expr" if c == 1 else "",
                gridcolor="#eeeeee", showgrid=True,
                zeroline=True, zerolinecolor="#cccccc",
                row=r, col=c,
            )
    return fig


def build_foxm1_population_umap(pop_df):
    """Three-panel UMAP: cell types | FOXM1 WT | FOXM1 KO (same colorscale)."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    CT_COLORS = {"quiescent": "#4C72B0", "intermediate": "#999999", "proliferative": "#D45F5F"}
    CT_ORDER   = ["quiescent", "intermediate", "proliferative"]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Cell types (DNAJB1 / MKI67-100)", "FOXM1 — WT simulation (DNAJB1 / MKI67-100)", "FOXM1 — KO simulation (DNAJB1 / MKI67-100)"],
        horizontal_spacing=0.05,
    )

    # ── Panel 1: cell type labels ─────────────────────────────────
    for ct in CT_ORDER:
        sub = pop_df[pop_df["cell_type"] == ct]
        fig.add_trace(go.Scatter(
            x=sub["x"], y=sub["y"], mode="markers",
            marker=dict(size=3, color=CT_COLORS[ct], opacity=0.7),
            name=ct.capitalize(), legendgroup=ct,
            hovertemplate=f"<b>{ct}</b><extra></extra>",
        ), row=1, col=1)

    # ── Panels 2 & 3: FOXM1 expression on same colorscale ────────
    wt = pop_df["FOXM1_wt"].values
    ko = pop_df["FOXM1_ko"].values
    vmax = float(np.percentile(wt, 97))
    vmin = 0.0

    for ci, (expr, label) in enumerate([(wt, "WT"), (ko, "KO")]):
        col = 2 + ci
        is_last = (ci == 1)
        fig.add_trace(go.Scatter(
            x=pop_df["x"], y=pop_df["y"], mode="markers",
            marker=dict(
                size=3, opacity=0.8,
                color=expr,
                colorscale="YlOrRd",
                cmin=vmin, cmax=vmax,
                showscale=is_last,
                colorbar=dict(
                    title=dict(text="FOXM1<br>expr", side="right", font=dict(size=10)),
                    thickness=12, len=0.65, tickfont=dict(size=9), x=1.01,
                ) if is_last else None,
            ),
            text=[f"{label} | {ct}<br>FOXM1: {v:.3f}"
                  for ct, v in zip(pop_df["cell_type"], expr)],
            hoverinfo="text", showlegend=False,
        ), row=1, col=col)

    fig.update_layout(
        height=360, margin=dict(t=45, b=10, l=5, r=65),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="v", x=-0.01, y=0.5, xanchor="right",
                    font=dict(size=10), itemsizing="constant"),
    )
    for col in [1, 2, 3]:
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=col)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=col)
    return fig


def build_population_proportions_figure(sim_df, ko_label="FOXM1 KO"):
    """Grouped bar chart: % of cells in each population, WT simulation vs KO.
    sim_df must have columns: pop_wt, pop_ko."""
    import plotly.graph_objects as go

    POP_ORDER  = ["proliferative", "quiescent", "intermediate"]
    POP_LABELS = {"proliferative": "Proliferative", "quiescent": "Quiescent",
                  "intermediate": "Intermediate"}
    POP_COLORS = {"proliferative": "#e63946", "quiescent": "#1a6faf",
                  "intermediate": "#999999"}

    n = len(sim_df)
    wt_counts = sim_df["pop_wt"].value_counts()
    ko_counts = sim_df["pop_ko"].value_counts()

    wt_pct = {p: wt_counts.get(p, 0) / n * 100 for p in POP_ORDER}
    ko_pct = {p: ko_counts.get(p, 0) / n * 100 for p in POP_ORDER}

    fig = go.Figure()
    for pop in POP_ORDER:
        color = POP_COLORS[pop]
        label = POP_LABELS[pop]
        fig.add_trace(go.Bar(
            name=label,
            x=["WT simulation", ko_label],
            y=[wt_pct[pop], ko_pct[pop]],
            marker_color=[color, color],
            marker_opacity=[0.9, 0.55],
            marker_line=dict(color=color, width=1.5),
            width=0.25,
            hovertemplate=f"<b>{label}</b><br>%{{x}}: %{{y:.1f}}%<extra></extra>",
        ))

    # Δ% annotations above KO bars
    annotations = []
    x_offsets = {"proliferative": -0.26, "quiescent": 0.0, "intermediate": 0.26}
    for pop in POP_ORDER:
        delta = ko_pct[pop] - wt_pct[pop]
        sign = "+" if delta >= 0 else ""
        annotations.append(dict(
            x=1 + x_offsets[pop],
            y=ko_pct[pop] + 0.8,
            text=f"{sign}{delta:.1f}%",
            showarrow=False,
            font=dict(size=11, color=POP_COLORS[pop]),
            xanchor="center",
        ))

    fig.update_layout(
        height=360,
        margin=dict(t=60, b=10, l=65, r=15),
        plot_bgcolor="white", paper_bgcolor="white",
        barmode="group", bargroupgap=0.18,
        title=dict(text=f"Cell population proportions: WT vs {ko_label} simulation (DNAJB1 / MKI67-100)",
                   font=dict(size=15), x=0.5),
        yaxis=dict(
            title=dict(text="% of cells", font=dict(size=13)),
            tickfont=dict(size=12),
            gridcolor="#eeeeee", gridwidth=1,
            zeroline=True, zerolinecolor="#cccccc",
            rangemode="tozero",
        ),
        xaxis=dict(showgrid=False, tickfont=dict(size=14)),
        legend=dict(orientation="h", yanchor="top", y=-0.06,
                    xanchor="center", x=0.5, font=dict(size=13)),
        annotations=annotations,
    )
    return fig


def build_population_delta_figure(sim_df, ko_label="FOXM1 KO"):
    """Horizontal bar chart: Δ% change in each population after KO.
    sim_df must have columns: pop_wt, pop_ko."""
    import plotly.graph_objects as go

    POP_ORDER  = ["proliferative", "quiescent", "intermediate"]
    POP_LABELS = {"proliferative": "Proliferative", "quiescent": "Quiescent",
                  "intermediate": "Intermediate"}
    POP_COLORS = {"proliferative": "#e63946", "quiescent": "#1a6faf",
                  "intermediate": "#999999"}

    n = len(sim_df)
    wt_pct = {p: (sim_df["pop_wt"] == p).sum() / n * 100 for p in POP_ORDER}
    ko_pct = {p: (sim_df["pop_ko"] == p).sum() / n * 100 for p in POP_ORDER}
    deltas = {p: ko_pct[p] - wt_pct[p] for p in POP_ORDER}

    fig = go.Figure(go.Bar(
        x=[deltas[p] for p in POP_ORDER],
        y=[POP_LABELS[p] for p in POP_ORDER],
        orientation="h",
        marker_color=[POP_COLORS[p] for p in POP_ORDER],
        text=[f"{deltas[p]:+.1f}%" for p in POP_ORDER],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Δ = %{x:+.1f}%<extra></extra>",
    ))
    fig.update_layout(
        height=260,
        margin=dict(t=60, b=20, l=130, r=70),
        plot_bgcolor="white", paper_bgcolor="white",
        title=dict(text=f"Population shift after {ko_label} (Δ%) — DNAJB1 / MKI67-100",
                   font=dict(size=15), x=0.5),
        xaxis=dict(
            title=dict(text="Δ% (KO − WT)", font=dict(size=13)),
            tickfont=dict(size=12),
            zeroline=True, zerolinecolor="#aaa", zerolinewidth=2,
            gridcolor="#eeeeee",
        ),
        yaxis=dict(showgrid=False, tickfont=dict(size=13)),
    )
    return fig


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


def build_perturbation_figures(pert_df, query_gene, ko_gene="BIRC5", real_expr_means=None):
    """Build two figures: top-20 barplot + gene dynamics WT vs KO."""
    times = sorted(pert_df["time"].unique())
    last_t = times[-1]

    # ── Model-specific target annotations ───────────────────────
    _TARGET_DEFS = {
        "BIRC5": {
            "co_targets":     {"PPP1R12B", "MAP3K21"},
            "direct_targets": {"CEP55"},
            "subtitle": (
                "🟠 co-target: goes UP after KO — compensatory escape mechanism &nbsp;|&nbsp;"
                " direct target: overexpressed in cancer, drives cytokinesis (CEP55)"
            ),
        },
        "TUBB": {
            "co_targets":     {"KIFC1"},    # goes UP — compensatory minus-end kinesin
            "direct_targets": {"MAPK4"},    # goes DOWN — suppressed by TUBB loss
            "subtitle": (
                "🟠 co-target: goes UP after KO — compensatory kinesin (KIFC1) &nbsp;|&nbsp;"
                " direct target: suppressed by TUBB loss (MAPK4) — check real expression"
            ),
        },
        "FOXM1": {
            "co_targets":     {"EIF2A", "HSPA1A"},  # go UP — quiescent/stress genes reactivated
            "direct_targets": {"NUSAP1", "PSRC1"},   # go DOWN — mitotic targets of FOXM1
            "subtitle": (
                "🟠 co-target: goes UP after KO — quiescent stress response (EIF2A, HSPA1A) &nbsp;|&nbsp;"
                " direct target: mitotic gene driven by FOXM1 (NUSAP1, PSRC1)"
            ),
        },
        "HSPA1B": {
            "co_targets":     {"HSPA1A", "DNAJB1", "HSPA6"},  # UP — compensatory heat-shock response
            "direct_targets": {"RACGAP1", "NUSAP1", "FOXM1"},  # DOWN — GRN targets / mitotic suppression
            "subtitle": (
                "🟠 co-target: goes UP after KO — compensatory heat-shock response (HSPA1A, DNAJB1, HSPA6) &nbsp;|&nbsp;"
                " direct target: suppressed by HSPA1B loss — mitotic/cytokinesis genes (RACGAP1, NUSAP1, FOXM1)"
            ),
        },
    }
    _def = _TARGET_DEFS.get(ko_gene, {"co_targets": set(), "direct_targets": set(), "subtitle": ""})
    co_targets     = _def["co_targets"]
    direct_targets = _def["direct_targets"]
    all_targets    = co_targets | direct_targets
    subtitle       = _def["subtitle"]

    # ── Figure 1: Top-20 most affected genes at last timepoint ──
    # exclude the KO gene itself (trivial — we set it to 0 by design)
    summary = pert_df[(pert_df["time"] == last_t) & (pert_df["gene"] != ko_gene)].copy()
    summary["abs_log2fc"] = summary["log2fc"].abs()
    top20 = summary.nlargest(20, "abs_log2fc").sort_values("log2fc")

    colors = []
    for gene, val in zip(top20["gene"], top20["log2fc"]):
        if gene in all_targets:
            colors.append("#FF8C00")   # orange = potential target
        elif val > 0:
            colors.append("#D45F5F")   # red = up
        else:
            colors.append("#4C72B0")   # blue = down

    # Arrow annotations — label differs by target type
    # "overexpressed" qualifier only added if real mean expression is substantial
    _OVEREXPR_THRESH = 0.3   # log-normalised mean expression threshold
    annotations = []
    for gene, val in zip(top20["gene"].tolist(), top20["log2fc"].tolist()):
        if gene in all_targets:
            if gene in direct_targets:
                _re = real_expr_means.get(gene, 0.0) if real_expr_means else 0.0
                if _re >= _OVEREXPR_THRESH:
                    label = "◀ direct target (overexpressed)"
                else:
                    label = "◀ direct target (low expr in data)"
                ax_offset = -30
            else:
                label = "co-target ▶" if val >= 0 else "◀ co-target"
                ax_offset = 30 if val >= 0 else -30
            annotations.append(dict(
                x=val, y=gene,
                xref="x", yref="y",
                text=label,
                showarrow=True,
                arrowhead=2,
                arrowcolor="#FF8C00",
                arrowwidth=1.5,
                ax=ax_offset, ay=0,
                font=dict(color="#FF8C00", size=11),
                xanchor="left" if val >= 0 else "right",
            ))

    # Real expression values for top-20 genes (if available)
    _real_vals = []
    if real_expr_means:
        for g in top20["gene"]:
            _real_vals.append(real_expr_means.get(g, 0.0))
    else:
        _real_vals = [0.0] * len(top20)
    _real_arr = np.array(_real_vals, dtype=float)
    _real_max = float(_real_arr.max()) if _real_arr.max() > 0 else 1.0

    # Hover text: log2FC + real expression
    _hover = [
        f"<b>{g}</b><br>log₂FC: {lfc:.3f}<br>Real mean expr: {re:.3f}"
        for g, lfc, re in zip(top20["gene"], top20["log2fc"], _real_arr)
    ]

    bar_fig = go.Figure()

    # ── Bar trace (log2FC, primary x-axis) ──────────────────────────
    bar_fig.add_trace(go.Bar(
        x=top20["log2fc"], y=top20["gene"],
        orientation="h",
        marker_color=colors,
        customdata=_real_arr,
        hovertext=_hover,
        hoverinfo="text",
        name="log₂FC",
        showlegend=False,
    ))

    # ── Dot overlay (real expression, secondary x-axis on top) ──────
    if real_expr_means:
        # Normalise dot size: 6–20 px
        _dot_sizes = 6 + 14 * (_real_arr / (_real_max + 1e-9))
        bar_fig.add_trace(go.Scatter(
            x=_real_arr, y=top20["gene"].tolist(),
            mode="markers",
            xaxis="x2",
            marker=dict(
                size=_dot_sizes.tolist(),
                color=_real_arr.tolist(),
                colorscale="YlOrRd",
                cmin=0, cmax=_real_max,
                showscale=True,
                colorbar=dict(
                    title=dict(text="Real expr", side="right"),
                    thickness=10, len=0.6, x=1.04,
                    tickfont=dict(size=11),
                ),
                line=dict(width=0.5, color="white"),
                opacity=0.85,
            ),
            hovertext=[
                f"<b>{g}</b><br>Real mean expr: {re:.3f}"
                for g, re in zip(top20["gene"], _real_arr)
            ],
            hoverinfo="text",
            name="Real expr",
            showlegend=False,
        ))

    bar_fig.update_layout(
        title=dict(
            text=(
                f"Top 20 genes affected by {ko_gene} KO (t={int(last_t)})<br>"
                f"<sup style='color:#FF8C00'>{subtitle}</sup>"
            ),
            font=dict(size=16),
        ),
        xaxis=dict(
            title=dict(text="log₂FC (KO / WT)", font=dict(size=13)),
            tickfont=dict(size=12),
            zeroline=True, zerolinecolor="#aaa",
        ),
        yaxis=dict(
            tickfont=dict(size=13),
        ),
        xaxis2=dict(
            title=dict(text="Mean expression (real data)", font=dict(size=12, color="#999")),
            overlaying="x", side="top",
            range=[0, _real_max * 1.25],
            showgrid=False,
            tickfont=dict(size=11, color="#999"),
        ) if real_expr_means else {},
        height=560, margin=dict(l=100, r=100, t=110, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
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
            mode="lines+markers", name=f"{ko_gene} KO",
            line=dict(color="#D45F5F", width=2, dash="dash")
        ))
    line_fig.update_layout(
        title=f"{query_gene} expression: WT vs {ko_gene} KO",
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
        return None, None

    program_set = set(gene_set or [query_gene])

    # Build full GRN over ALL grn_genes (not just program)
    G_full = nx.DiGraph()
    G_full.add_node(query_gene)
    if query_gene in grn_genes:
        idx = grn_genes.index(query_gene)
        for j, w in enumerate(grn_mat[idx]):
            if abs(w) > 0 and grn_genes[j] != query_gene:
                G_full.add_edge(query_gene, grn_genes[j], weight=float(w))
        for i, w in enumerate(grn_mat[:, idx]):
            if abs(w) > 0 and grn_genes[i] != query_gene:
                G_full.add_edge(grn_genes[i], query_gene, weight=float(w))

    frontier = {query_gene}
    visited  = {query_gene}
    for hop in range(hops - 1):
        next_frontier = set()
        for gene in frontier:
            if gene not in grn_genes:
                continue
            idx = grn_genes.index(gene)
            for j, w in enumerate(grn_mat[idx]):
                if abs(w) > 0 and grn_genes[j] not in visited:
                    G_full.add_edge(gene, grn_genes[j], weight=float(w))
                    next_frontier.add(grn_genes[j]); visited.add(grn_genes[j])
            for i, w in enumerate(grn_mat[:, idx]):
                if abs(w) > 0 and grn_genes[i] not in visited:
                    G_full.add_edge(grn_genes[i], gene, weight=float(w))
                    next_frontier.add(grn_genes[i]); visited.add(grn_genes[i])
        frontier = next_frontier

    reachable = nx.single_source_shortest_path_length(
        G_full.to_undirected(), query_gene, cutoff=hops
    )
    nodes_to_keep = {query_gene}
    for node, dist in reachable.items():
        if node in program_set:
            try:
                path = nx.shortest_path(G_full.to_undirected(), query_gene, node)
                nodes_to_keep.update(path)
            except nx.NetworkXNoPath:
                pass

    G = G_full.subgraph(nodes_to_keep).copy()

    # ── Topology analysis ────────────────────────────────────────
    # upstream / downstream relative to query
    try:
        upstream   = set(nx.ancestors(G, query_gene))
    except Exception:
        upstream   = set()
    try:
        downstream = set(nx.descendants(G, query_gene))
    except Exception:
        downstream = set()
    feedback_nodes = upstream & downstream   # in a cycle with query

    # all simple cycles in subgraph (length ≤ 5 to stay fast)
    all_cycles = list(nx.simple_cycles(G))
    cycle_nodes = {n for cyc in all_cycles for n in cyc}

    # per-node metrics
    in_deg  = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    try:
        between = nx.betweenness_centrality(G, normalized=True)
    except Exception:
        between = {n: 0.0 for n in G.nodes()}

    topo_rows = []
    for n in G.nodes():
        if n == query_gene:
            role = "query"
        elif n in feedback_nodes:
            role = "feedback loop"
        elif n in upstream:
            role = "upstream"
        elif n in downstream:
            role = "downstream"
        else:
            role = "intermediate"
        topo_rows.append({
            "Gene": n,
            "Role": role,
            "In-degree":  in_deg.get(n, 0),
            "Out-degree": out_deg.get(n, 0),
            "Total degree": in_deg.get(n, 0) + out_deg.get(n, 0),
            "Betweenness": round(between.get(n, 0), 3),
            "Feedback loop": n in cycle_nodes,
        })
    topo_df = pd.DataFrame(topo_rows).sort_values("Total degree", ascending=False)

    # ── Node visual properties ───────────────────────────────────
    ROLE_COLOR = {
        "query":         "#D62728",   # red
        "upstream":      "#FF7F0E",   # orange
        "downstream":    "#4C72B0",   # blue
        "feedback loop": "#9467BD",   # purple
        "intermediate":  "#AAAAAA",   # gray
    }
    max_deg = max((in_deg[n] + out_deg[n]) for n in G.nodes()) or 1
    node_colors = []
    node_sizes  = []
    node_symbols = []
    for n in G.nodes():
        role = topo_df.loc[topo_df["Gene"] == n, "Role"].values[0]
        node_colors.append(ROLE_COLOR[role])
        deg = in_deg[n] + out_deg[n]
        node_sizes.append(14 + 22 * (deg / max_deg))
        node_symbols.append("diamond" if n in cycle_nodes and n != query_gene else "circle")

    pos = nx.spring_layout(G, seed=42)
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_labels = list(G.nodes())

    hover_texts = []
    for n in G.nodes():
        row = topo_df[topo_df["Gene"] == n].iloc[0]
        fl = " 🔄" if row["Feedback loop"] else ""
        hover_texts.append(
            f"<b>{n}</b>{fl}<br>Role: {row['Role']}<br>"
            f"In: {row['In-degree']}  Out: {row['Out-degree']}<br>"
            f"Betweenness: {row['Betweenness']}"
        )

    fig = go.Figure()

    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        color = "#2CA02C" if d["weight"] > 0 else "#D62728"
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.5,
            arrowwidth=1.5, arrowcolor=color
        )

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=node_labels, textposition="top center",
        marker=dict(size=node_sizes, color=node_colors,
                    symbol=node_symbols,
                    line=dict(width=1.5, color="white")),
        hovertext=hover_texts, hoverinfo="text",
        showlegend=False
    ))

    for role, color in ROLE_COLOR.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=12, color=color),
            name=role.capitalize(), showlegend=True
        ))

    fig.update_layout(
        title=(f"GRN for {query_gene} — {hops}-hop ego network  "
               f"(green=activation, red=repression | size=degree | ◆=feedback loop)"),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=580,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    return fig, topo_df


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


# ── Logo + Title ─────────────────────────────────────────────────
import os as _os, base64 as _b64

_logo_local = _os.path.join(LOCAL_DIR, "logo.png")
if not _os.path.exists(_logo_local):
    try:
        _token = st.secrets.get("HF_TOKEN", None)
        _logo_local = hf_hub_download(repo_id=REPO_ID, filename="logo.png",
                                      repo_type="dataset", token=_token)
    except Exception:
        _logo_local = None

_logo_b64 = ""
if _logo_local:
    with open(_logo_local, "rb") as _f:
        _logo_b64 = _b64.b64encode(_f.read()).decode()

# ── Calanques de Marseille header ────────────────────────────────
# Streamlit's HTML sanitizer strips <svg> tags from st.markdown(), so we must
# encode SVGs as base64 <img> tags.  CSS @keyframes animations work fine in
# Streamlit unsafe_allow_html; SVG SMIL animations inside <img> do NOT.
# Strategy: static background as base64 <img>, animated boat as a second
# absolutely-positioned base64 <img> driven by CSS @keyframes.

_calanques_static_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 160">
  <defs>
    <linearGradient id="csky" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#87CEEB"/><stop offset="100%" stop-color="#E0F4FF"/>
    </linearGradient>
    <linearGradient id="csea" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#1A9DAB"/><stop offset="100%" stop-color="#0D6E78"/>
    </linearGradient>
    <linearGradient id="ccl" x1="0" y1="0" x2="0.3" y2="1">
      <stop offset="0%" stop-color="#E8E0D0"/><stop offset="100%" stop-color="#C8B89A"/>
    </linearGradient>
    <linearGradient id="ccr" x1="1" y1="0" x2="0.7" y2="1">
      <stop offset="0%" stop-color="#DDD5C5"/><stop offset="100%" stop-color="#BCA888"/>
    </linearGradient>
  </defs>
  <rect width="900" height="160" fill="url(#csky)"/>
  <path d="M0 95 Q225 80 450 90 Q675 100 900 85 L900 160 L0 160 Z" fill="url(#csea)"/>
  <path d="M80 110 Q180 106 280 112" stroke="#4DD0DC" stroke-width="1" fill="none" opacity="0.4"/>
  <path d="M550 108 Q650 104 750 110" stroke="#4DD0DC" stroke-width="1" fill="none" opacity="0.4"/>
  <path d="M0 160 L0 40 L30 20 L55 35 L80 15 L110 30 L140 10 L170 28 L200 18 L230 45 L255 35 L280 60 L300 50 L330 75 L310 160 Z" fill="url(#ccl)"/>
  <path d="M0 160 L0 60 L20 45 L50 55 L80 40 L120 55 L160 40 L200 55 L240 65 L270 75 L300 70 L310 160 Z" fill="#C8B89A" opacity="0.5"/>
  <path d="M60 60 L70 90 L65 110" stroke="#A89880" stroke-width="0.8" fill="none" opacity="0.6"/>
  <path d="M130 48 L140 78" stroke="#A89880" stroke-width="0.8" fill="none" opacity="0.6"/>
  <g opacity="0.85">
    <polygon points="85,14 80,30 90,30" fill="#2D5A27"/><polygon points="85,22 78,36 92,36" fill="#2D5A27"/><rect x="83" y="36" width="4" height="6" fill="#5C3D2E"/>
    <polygon points="145,8 139,22 151,22" fill="#2D5A27"/><polygon points="145,17 137,30 153,30" fill="#2D5A27"/><rect x="143" y="30" width="4" height="5" fill="#5C3D2E"/>
    <polygon points="200,16 195,28 205,28" fill="#336B2C"/><polygon points="200,24 193,36 207,36" fill="#336B2C"/><rect x="198" y="36" width="4" height="5" fill="#5C3D2E"/>
  </g>
  <path d="M900 160 L900 35 L870 18 L845 32 L820 12 L790 28 L760 8 L730 25 L700 15 L670 40 L645 30 L620 55 L600 48 L575 70 L590 160 Z" fill="url(#ccr)"/>
  <path d="M900 160 L900 55 L880 42 L850 52 L820 38 L780 52 L740 38 L700 52 L660 62 L630 72 L600 68 L590 160 Z" fill="#BCA888" opacity="0.5"/>
  <path d="M840 55 L830 82 L835 105" stroke="#A08868" stroke-width="0.8" fill="none" opacity="0.6"/>
  <path d="M760 42 L750 70" stroke="#A08868" stroke-width="0.8" fill="none" opacity="0.6"/>
  <g opacity="0.85">
    <polygon points="820,10 815,24 825,24" fill="#2D5A27"/><polygon points="820,18 813,32 827,32" fill="#2D5A27"/><rect x="818" y="32" width="4" height="5" fill="#5C3D2E"/>
    <polygon points="760,6 755,20 765,20" fill="#2D5A27"/><polygon points="760,15 753,28 767,28" fill="#2D5A27"/><rect x="758" y="28" width="4" height="5" fill="#5C3D2E"/>
    <polygon points="700,13 695,26 705,26" fill="#336B2C"/><polygon points="700,22 693,34 707,34" fill="#336B2C"/><rect x="698" y="34" width="4" height="5" fill="#5C3D2E"/>
  </g>
  <g stroke="#555" stroke-width="1" fill="none" opacity="0.5">
    <path d="M320 45 Q325 41 330 45"/><path d="M340 38 Q346 34 352 38"/>
    <path d="M560 42 Q565 38 570 42"/><path d="M578 50 Q583 46 588 50"/>
  </g>
  <rect width="900" height="160" fill="white" opacity="0.08"/>
</svg>"""

# Sailboat — separate SVG, animated via CSS @keyframes (SMIL won't work inside <img>)
_boat_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="-38 -20 76 40">
  <path d="M-18 10 Q0 8 18 10 L12 15 L-12 15 Z" fill="#F5F0E8"/>
  <line x1="0" y1="15" x2="0" y2="-15" stroke="#8B7355" stroke-width="1.2"/>
  <path d="M0 -14 L16 8 L0 10 Z" fill="white" opacity="0.92"/>
  <path d="M0 -8 L-10 6 L0 8 Z" fill="#F0E8D8" opacity="0.82"/>
  <path d="M-20 16 Q-30 19 -37 16" stroke="#6DD4DF" stroke-width="0.9" fill="none" opacity="0.6"/>
  <path d="M20 16 Q30 19 37 16" stroke="#6DD4DF" stroke-width="0.9" fill="none" opacity="0.6"/>
</svg>"""


_cal_b64  = _b64.b64encode(_calanques_static_svg.encode()).decode()
_boat_b64 = _b64.b64encode(_boat_svg.encode()).decode()

# Custom background: use uploaded image from session_state if available
_custom_bg  = st.session_state.get("custom_bg_b64", None)
_bg_img_src = _custom_bg if _custom_bg else f"data:image/svg+xml;base64,{_cal_b64}"
_bg_opacity = "0.88" if _custom_bg else "0.55"
_boat_html  = "" if _custom_bg else (
    f"<img src='data:image/svg+xml;base64,{_boat_b64}' "
    f"style='position:absolute;width:5.5%;min-width:36px;"
    f"animation:boatSail 10s ease-in-out infinite;pointer-events:none;'/>"
)

st.markdown(f"""
<style>
@keyframes boatSail {{
  0%   {{ left:42%; top:54%; }}
  25%  {{ left:46%; top:51%; }}
  50%  {{ left:50%; top:54%; }}
  75%  {{ left:46%; top:57%; }}
  100% {{ left:42%; top:54%; }}
}}
@keyframes betaRay {{
  0%, 100% {{ opacity:0.12; transform:scaleY(0.55); }}
  50%       {{ opacity:1;    transform:scaleY(1);    }}
}}
.bsun {{ position:absolute; top:2px; left:68%; transform:translateX(-50%); width:54px; height:54px; z-index:3; }}
.bsun-circle {{
  position:absolute; top:50%; left:50%;
  transform:translate(-50%,-50%);
  width:28px; height:28px; border-radius:50%;
  background:linear-gradient(135deg,#FDD835,#E8A020);
  box-shadow:0 0 12px 3px rgba(253,216,53,0.55);
  display:flex; align-items:center; justify-content:center;
  color:white; font-size:0.42rem; font-weight:800; letter-spacing:0.8px;
  z-index:2;
}}
.brw {{
  position:absolute; top:0; left:0; right:0; bottom:0;
  display:flex; align-items:flex-start; justify-content:center;
}}
.br {{
  display:block; width:3px; height:8px; margin-top:2px;
  background:#FDD835; border-radius:2px;
  animation:betaRay 1.8s ease-in-out infinite;
  transform-origin:center bottom;
}}
</style>
<div id='gpe-header' style='position:relative; border-radius:12px; margin-bottom:8px;'>
  <!-- inner clip wrapper keeps background + boat rounded, outer div stays overflow:visible for camera btn -->
  <div style='position:absolute;top:0;left:0;right:0;bottom:0;border-radius:12px;overflow:hidden;z-index:0;'>
    <img src='{_bg_img_src}'
         style='position:absolute;top:0;left:0;width:100%;height:100%;
                object-fit:cover;opacity:{_bg_opacity};pointer-events:none;'/>
    {_boat_html}
  </div>
  <!-- BETA sun in the sky -->
  <div class='bsun'>
    <div class='brw' style='transform:rotate(0deg)'><span   class='br' style='animation-delay:0s'></span></div>
    <div class='brw' style='transform:rotate(45deg)'><span  class='br' style='animation-delay:0.225s'></span></div>
    <div class='brw' style='transform:rotate(90deg)'><span  class='br' style='animation-delay:0.45s'></span></div>
    <div class='brw' style='transform:rotate(135deg)'><span class='br' style='animation-delay:0.675s'></span></div>
    <div class='brw' style='transform:rotate(180deg)'><span class='br' style='animation-delay:0.9s'></span></div>
    <div class='brw' style='transform:rotate(225deg)'><span class='br' style='animation-delay:1.125s'></span></div>
    <div class='brw' style='transform:rotate(270deg)'><span class='br' style='animation-delay:1.35s'></span></div>
    <div class='brw' style='transform:rotate(315deg)'><span class='br' style='animation-delay:1.575s'></span></div>
    <div class='bsun-circle'>BETA</div>
  </div>
  <div style='position:relative; z-index:1;
              display:flex; align-items:center; gap:12px;
              padding:18px 20px 14px 20px; flex-wrap:nowrap;'>
    {"<a href='/' target='_self' style='flex-shrink:0; line-height:0;'><img src='data:image/png;base64," + _logo_b64 + "' style='height:60px; width:auto; filter:drop-shadow(0 1px 3px rgba(0,0,0,0.3)); cursor:pointer;'/></a>" if _logo_b64 else ""}
    <div>
      <div style='white-space:nowrap;'>
        <span style='font-size:clamp(1.2rem,4vw,2.2rem); font-weight:800;
                     color:#002395; text-shadow:0 1px 3px rgba(255,255,255,0.8);'>Gene</span>
        <span style='font-size:clamp(1.2rem,4vw,2.2rem); font-weight:800;
                     color:#FFFFFF; text-shadow:0 1px 4px rgba(0,0,0,0.4);'> Program </span>
        <span style='font-size:clamp(1.2rem,4vw,2.2rem); font-weight:800;
                     color:#ED2939; text-shadow:0 1px 3px rgba(255,255,255,0.8);'>Explorer</span>
      </div>
      <div style='font-size:clamp(0.65rem,2vw,0.85rem); color:#333;
                  margin-top:2px; text-shadow:0 1px 2px rgba(255,255,255,0.7);'>
        <b>14,581</b> gene embeddings &nbsp;·&nbsp; <b>594</b> GRN genes &nbsp;·&nbsp; HSPA1B KO · BIRC5 KO · TUBB KO · FOXM1 KO &nbsp;·&nbsp; RMS
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Hidden file uploader (triggered by JS camera icon in header) ──
st.markdown('<div id="cam-fu-anchor" style="display:none"></div>', unsafe_allow_html=True)
_cam_upload = st.file_uploader(
    "bg", type=["png","jpg","jpeg","webp"],
    key="bg_cam_uploader", label_visibility="collapsed"
)
if _cam_upload is not None:
    _file_id = f"{_cam_upload.name}_{_cam_upload.size}"
    if st.session_state.get("_bg_last_file") != _file_id:
        st.session_state["_bg_last_file"] = _file_id
        st.session_state["custom_bg_b64"] = (
            f"data:{_cam_upload.type};base64," + _b64.b64encode(_cam_upload.read()).decode()
        )
        st.rerun()

# ── JS: inject camera icon into header; hide file uploader via DOM; proxy click ──
_components.html("""<script>
(function() {{
  function hideFU() {{
    const pd = window.parent.document;
    const anchor = pd.getElementById('cam-fu-anchor');
    if (!anchor) return;
    // Find stFileUploader that comes after the anchor in document order
    const uploaders = [...pd.querySelectorAll('[data-testid="stFileUploader"]')];
    const fu = uploaders.find(u => anchor.compareDocumentPosition(u) & 4);
    if (!fu) return;
    // opacity+height:0 keeps element interactive (.click() works), just invisible
    const hide = 'opacity:0!important;height:0!important;overflow:hidden!important;'
               + 'margin:0!important;padding:0!important;border:none!important;';
    fu.style.cssText = hide;
    if (fu.parentElement) fu.parentElement.style.cssText = hide;
  }}

  function inject() {{
    const pd = window.parent.document;
    const hdr = pd.getElementById('gpe-header');
    if (!hdr) {{ setTimeout(inject, 200); return; }}
    if (pd.getElementById('gpe-cam-btn')) {{ hideFU(); return; }}

    hideFU();

    // ── camera button ──
    const cam = pd.createElement('div');
    cam.id = 'gpe-cam-btn'; cam.title = 'Change background'; cam.textContent = '📷';
    cam.style.cssText = 'position:absolute;top:10px;right:14px;z-index:500;'
      + 'display:inline-flex;align-items:center;justify-content:center;'
      + 'width:28px;height:28px;border-radius:50%;cursor:pointer;font-size:13px;'
      + 'background:rgba(255,255,255,0.68);border:1.5px solid rgba(255,255,255,0.55);'
      + 'box-shadow:0 1px 6px rgba(0,0,0,0.25);user-select:none;';
    cam.onmouseover = () => cam.style.background = 'rgba(255,255,255,0.95)';
    cam.onmouseout  = () => cam.style.background = 'rgba(255,255,255,0.68)';
    cam.onclick = () => {{
      const anchor = pd.getElementById('cam-fu-anchor');
      const btns = [...pd.querySelectorAll('[data-testid="stFileUploaderDropzone"] button')];
      const fu = anchor ? btns.find(b => anchor.compareDocumentPosition(b) & 4) : btns[0];
      if (fu) fu.click();
    }};
    hdr.appendChild(cam);

  }}

  setInterval(() => {{
    const pd = window.parent.document;
    if (pd.getElementById('gpe-header') && !pd.getElementById('gpe-cam-btn')) inject();
    else hideFU();
  }}, 500);
  inject();
}})();
</script>""", height=0, scrolling=False)

with st.expander("About this tool"):
    st.markdown("""
This is a **RAG-based gene program retrieval system** applied to single-cell RMS (Rhabdomyosarcoma) data.

**How it works:**
1. A query gene is embedded in a GNN co-expression space (trained on WGCNA graphs from scRNA-seq)
2. Nearest neighbors in embedding space define a **transcriptional program**
3. The retrieved gene list is exported and run through **CardamomOT** (ODE mechanistic model + optimal transport) on the scRNA-seq time-course data to infer a **gene regulatory network (GRN)** for that program
4. The same CardamomOT model is then used for in silico **perturbation simulations** (e.g. BIRC5 knockout) — revealing which genes change and enabling **network-based target identification**

> 💡 **For several key programs (Full, FOXM1, MKI67, TUBB) this has already been done** — the GRN and perturbation results are precomputed and available directly in this app. You can explore the candidate therapeutic targets without running CardamomOT yourself.

**Therapeutic target logic (network perturbation approach):**
- **Direct targets** — genes overexpressed in the tumor that are essential nodes in the GRN (e.g. CEP55: drives cytokinesis, supra-expressed in RMS)
- **Co-targets** — genes that go *up* after BIRC5 KO, acting as compensatory escape mechanisms (e.g. PPP1R12B, MAP3K21); blocking them alongside BIRC5 leaves the cell no survival route
- This approach is called **network-informed synthetic lethality** — targets are chosen not in isolation but based on their role in the regulatory network under perturbation

**Upload your own data:**
Upload any `.h5ad` file to compute a UMAP and visualize gene expression in your dataset.
Genes overlapping with the RMS embedding space can be queried directly in the chat to retrieve co-expression programs and GRN context from the RMS model.

**Available GRN models:**
- **Original** — 159 genes, inferred from full RMS scRNA-seq data
- **Full program** — 200 genes (complete quiescent + proliferative gene set), HSPA1B KO perturbation simulated via CARDAMOM mechanistic model
- **FOXM1 program** — 198 genes (top-200 GNN neighbors of FOXM1), FOXM1 KO perturbation simulated via CARDAMOM mechanistic model
- **MKI67 program** — 201 genes (top-200 GNN neighbors of MKI67), BIRC5 KO perturbation simulated via CARDAMOM mechanistic model
- **TUBB program** — 201 genes (top-200 GNN neighbors of TUBB), TUBB KO perturbation simulated via CARDAMOM mechanistic model

**Population dynamics under knockout (Full program):**
When querying a gene from the **Full program** (e.g. HSPA1B), the app runs a full CARDAMOM mechanistic simulation of HSPA1B knockout and tracks how the cell population composition changes over time.
RMS tumours contain three co-existing cell states — **Proliferative**, **Quiescent**, and **Intermediate** — that are in dynamic equilibrium.
Each state is scored from the scRNA-seq data using gene signatures: **DNAJB1 z-score** (top HSPA1B neighbour, sim=0.91) for quiescence; **mean of top-100 MKI67 neighbours** for proliferation. Cells in neither top-30% → intermediate.
After KO, CARDAMOM propagates the perturbation through the GRN and re-simulates cell trajectories; the resulting shift in population fractions (Δ%) shows whether the knockout pushes cells toward or away from proliferation.
This reveals not just which genes change in expression, but **how the Waddington landscape reorganises** — a key step toward identifying interventions that durably suppress the proliferative state rather than merely reducing a single gene's expression.

**Gene program annotation (LLM):**
Each retrieved gene program is automatically annotated by **Llama 3.1-8B** (via Nebius AI Studio) — the model receives the top co-expressed genes and generates a concise biological label (e.g. *"Mitotic Cell Proliferation"*, *"Cytoskeletal remodelling"*). This enables rapid biological interpretation of each program without manual curation.

**Cell population scoring (DNAJB1 / MKI67-100):**
Three co-existing RMS cell states are defined by gene expression thresholds applied uniformly to real data and all simulations:
- **Proliferative** (red) — mean expression of top-100 MKI67 co-expression neighbours >= 70th percentile
- **Quiescent** (blue) — DNAJB1 z-score >= 70th percentile (DNAJB1 is the top HSPA1B neighbour, cosine sim = 0.91)
- **Intermediate** (grey) — all remaining cells

DNAJB1/HSPA1B anti-correlate with the FOXM1 proliferative program; their upregulation marks cells exiting the cell cycle. The same scoring rule is applied to real data, WT simulation, and KO simulation — making population shifts directly comparable.

**References:**
- CARDAMOM / CardamomOT: [github.com/eliasventre/CardamomOT](https://github.com/eliasventre/CardamomOT)
- Nebius AI Studio (Llama 3.1-8B inference): [studio.nebius.com](https://studio.nebius.com)
    """)

    _components.html(
        f"<div style='width:100%;overflow:hidden'>{_arch_svg}</div>",
        height=510, scrolling=False
    )

with st.expander("About Trajectory GNN (beta)"):
    st.markdown("""
> **Beta** — trajectory embeddings are computed from untrained model weights (random init).
> Biological interpretability will improve after supervised fine-tuning.

**What is Trajectory GNN?**

A new gene embedding method that captures how each gene's co-expression context **changes over time** across the 6 RMS timepoints (t=0, 16, 32, 48, 64, 80h).

**Architecture:**

1. **hdWGCNA graph per timepoint** — soft-thresholded Pearson co-expression graph built separately for each timepoint's cells (beta=6). Time is encoded implicitly: each graph reflects the co-expression landscape of that moment.

2. **GAT (Graph Attention Network)** — shared weights applied to each timepoint's graph. Attention mechanism learns to down-weight noisy co-expression edges. Output: gene embedding snapshot per timepoint.

3. **OT alignment (Sinkhorn)** — since GAT encodes neighborhood structure (not gene identity), genes are partially anonymous across timepoints. Optimal transport finds the optimal mapping between embedding clouds at each timepoint and t=0, without assuming gene i maps to gene i.

4. **PPGN on NEKO prior** (ProgramEncoder) — separately, for a specific gene program (~200 genes), a Provably Powerful GNN (WL-3) is run on the mechanistic NEKO interaction graph. Captures regulatory motifs: feedback loops, triangles, paths of length 2.

5. **Trajectory embedding** = MLP([delta, mean]) where:
   - `delta[i]` = aligned_tN[i] - aligned_t0[i] — how gene i's co-expression context shifted
   - `mean[i]` = average position across all timepoints — stable neighborhood

**Why OT?**

CardamomOT uses optimal transport to match cells across timepoints (unknown correspondence).
Here the same principle applies to gene embedding clouds — GAT captures structural patterns, so two genes with similar neighborhood statistics get similar embeddings regardless of which specific genes are their neighbors. OT resolves this partial anonymity.

**Files:**
- `trajectory_gene_embeddings.csv` — 3887 genes x 128 dims + cluster
- `trajectory_cluster_annotations.csv` — K-means cluster labels (LLM annotation pending)
    """)

    _arch_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -36 920 501" style="font-family:Arial,sans-serif">
<defs>
  <style>
    @keyframes arrowPulse {
      0%, 100% { opacity: 1;    stroke-width: 2.5; }
      50%       { opacity: 0.1; stroke-width: 1.5; }
    }
    @keyframes labelPulse {
      0%, 100% { opacity: 1; }
      50%       { opacity: 0.1; }
    }
    .blink-arrow path { animation: arrowPulse 1.4s ease-in-out infinite; }
    .blink-arrow rect  { animation: labelPulse 1.4s ease-in-out infinite; }
    .blink-arrow text  { animation: labelPulse 1.4s ease-in-out infinite; }
  </style>
  <marker id="ah" markerWidth="8" markerHeight="8" refX="7" refY="3.5" orient="auto">
    <path d="M0,0 L0,7 L8,3.5 z" fill="#888"/>
  </marker>
  <marker id="ahRed" markerWidth="10" markerHeight="10" refX="9" refY="4" orient="auto">
    <path d="M0,0 L0,8 L10,4 z" fill="#C0392B"/>
  </marker>
  <marker id="ahBlue" markerWidth="8" markerHeight="8" refX="7" refY="3.5" orient="auto">
    <path d="M0,0 L0,7 L8,3.5 z" fill="#5B9BD5"/>
  </marker>
</defs>

<!-- ═══ ROW 1 TITLE ═══ -->
<text x="380" y="25" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50">Data-driven Gene Program Retrieval</text>

<!-- ── Query Gene: input arrow into box 5 from above ── -->
<rect x="718" y="-34" width="154" height="24" rx="5" fill="#EBF4FC" stroke="#5B9BD5" stroke-width="1.5"/>
<text x="795" y="-17" text-anchor="middle" font-size="12" font-weight="bold" fill="#5B9BD5">🔍 Query Gene</text>
<line x1="795" y1="-10" x2="795" y2="31" stroke="#5B9BD5" stroke-width="1.8" stroke-dasharray="4,3" marker-end="url(#ahBlue)"/>

<!-- ── Box 1: scRNA-seq Data (blue) ── -->
<rect x="12" y="34" width="158" height="170" rx="8" fill="#EBF4FC" stroke="#5B9BD5" stroke-width="2"/>
<text x="91" y="56" text-anchor="middle" font-size="14" font-weight="bold" fill="#5B9BD5">scRNA-seq</text>
<text x="91" y="72" text-anchor="middle" font-size="13" font-weight="bold" fill="#5B9BD5">Data</text>
<rect x="56" y="84" width="70" height="72" rx="3" fill="none" stroke="#5B9BD5" stroke-width="1.5" opacity="0.7"/>
<line x1="56" y1="102" x2="126" y2="102" stroke="#5B9BD5" stroke-width="0.8" opacity="0.4"/>
<line x1="56" y1="120" x2="126" y2="120" stroke="#5B9BD5" stroke-width="0.8" opacity="0.4"/>
<line x1="56" y1="138" x2="126" y2="138" stroke="#5B9BD5" stroke-width="0.8" opacity="0.4"/>
<line x1="79" y1="84" x2="79" y2="156" stroke="#5B9BD5" stroke-width="0.8" opacity="0.4"/>
<line x1="102" y1="84" x2="102" y2="156" stroke="#5B9BD5" stroke-width="0.8" opacity="0.4"/>
<text x="91" y="175" text-anchor="middle" font-size="11" fill="#777">cells × genes × time</text>

<!-- ── Box 2: WGCNA Co-expression (purple) ── -->
<rect x="188" y="34" width="158" height="170" rx="8" fill="#F5EFF9" stroke="#9B59B6" stroke-width="2"/>
<text x="267" y="56" text-anchor="middle" font-size="14" font-weight="bold" fill="#9B59B6">WGCNA</text>
<text x="267" y="72" text-anchor="middle" font-size="12" fill="#9B59B6">Co-expression</text>
<line x1="267" y1="88" x2="233" y2="113" stroke="#9B59B6" stroke-width="1.5"/>
<line x1="267" y1="88" x2="301" y2="113" stroke="#9B59B6" stroke-width="1.5"/>
<line x1="233" y1="113" x2="245" y2="148" stroke="#9B59B6" stroke-width="1.5"/>
<line x1="301" y1="113" x2="289" y2="148" stroke="#9B59B6" stroke-width="1.5"/>
<line x1="245" y1="148" x2="289" y2="148" stroke="#9B59B6" stroke-width="1.5"/>
<circle cx="267" cy="88" r="6" fill="#9B59B6"/>
<circle cx="233" cy="113" r="6" fill="#9B59B6"/>
<circle cx="301" cy="113" r="6" fill="#9B59B6"/>
<circle cx="245" cy="148" r="6" fill="#9B59B6"/>
<circle cx="289" cy="148" r="6" fill="#9B59B6"/>
<text x="267" y="175" text-anchor="middle" font-size="11" fill="#777">gene-gene graph</text>

<!-- ── Box 3: GNN Encoder (green) ── -->
<rect x="364" y="34" width="158" height="170" rx="8" fill="#EAFAF1" stroke="#27AE60" stroke-width="2"/>
<text x="443" y="56" text-anchor="middle" font-size="14" font-weight="bold" fill="#27AE60">GNN Encoder</text>
<line x1="415" y1="90" x2="443" y2="125" stroke="#27AE60" stroke-width="1.5"/>
<line x1="443" y1="90" x2="443" y2="125" stroke="#27AE60" stroke-width="1.5"/>
<line x1="471" y1="90" x2="443" y2="125" stroke="#27AE60" stroke-width="1.5"/>
<line x1="443" y1="125" x2="427" y2="153" stroke="#27AE60" stroke-width="1.5"/>
<line x1="443" y1="125" x2="459" y2="153" stroke="#27AE60" stroke-width="1.5"/>
<circle cx="415" cy="90" r="6" fill="#27AE60"/>
<circle cx="443" cy="90" r="6" fill="#27AE60"/>
<circle cx="471" cy="90" r="6" fill="#27AE60"/>
<circle cx="443" cy="125" r="9" fill="#1A8A40"/>
<circle cx="427" cy="153" r="6" fill="#27AE60"/>
<circle cx="459" cy="153" r="6" fill="#27AE60"/>
<text x="443" y="175" text-anchor="middle" font-size="11" fill="#777">node → embedding</text>

<!-- ── Box 4: Gene Embeddings (orange) ── -->
<rect x="540" y="34" width="158" height="170" rx="8" fill="#FEF5E7" stroke="#E67E22" stroke-width="2"/>
<text x="619" y="56" text-anchor="middle" font-size="14" font-weight="bold" fill="#E67E22">Gene Embeddings</text>
<circle cx="597" cy="96" r="5" fill="#E67E22" opacity="0.6"/>
<circle cx="628" cy="86" r="8" fill="#E67E22" opacity="0.9"/>
<circle cx="648" cy="112" r="5" fill="#E67E22" opacity="0.5"/>
<circle cx="601" cy="126" r="5" fill="#999" opacity="0.45"/>
<circle cx="638" cy="136" r="5" fill="#999" opacity="0.4"/>
<circle cx="617" cy="110" r="9" fill="#E67E22" opacity="0.85"/>
<circle cx="607" cy="148" r="4" fill="#999" opacity="0.3"/>
<circle cx="643" cy="95" r="4" fill="#999" opacity="0.35"/>
<text x="619" y="175" text-anchor="middle" font-size="11" fill="#777">vector DB · cosine sim</text>

<!-- ── Box 5: Gene Program (teal) ── -->
<rect x="716" y="34" width="158" height="170" rx="8" fill="#E8FAF5" stroke="#16A085" stroke-width="2.5"/>
<text x="795" y="56" text-anchor="middle" font-size="14" font-weight="bold" fill="#16A085">Gene Program</text>
<!-- "LLM-annotated" badge -->
<rect x="738" y="62" width="114" height="17" rx="8" fill="#8E44AD" opacity="0.12" stroke="#8E44AD" stroke-width="1"/>
<text x="795" y="74" text-anchor="middle" font-size="9.5" font-weight="bold" fill="#8E44AD">LLM · Llama 3.1</text>
<circle cx="780" cy="112" r="7" fill="#16A085"/>
<circle cx="802" cy="99" r="6" fill="#16A085"/>
<circle cx="818" cy="120" r="7" fill="#16A085"/>
<circle cx="791" cy="133" r="6" fill="#16A085"/>
<circle cx="811" cy="139" r="5" fill="#16A085"/>
<circle cx="772" cy="129" r="5" fill="#16A085"/>
<circle cx="823" cy="103" r="5" fill="#16A085" opacity="0.7"/>
<circle cx="763" cy="107" r="4" fill="#16A085" opacity="0.6"/>
<text x="795" y="170" text-anchor="middle" font-size="11" fill="#777">co-expression neighbors</text>
<text x="795" y="185" text-anchor="middle" font-size="10" font-style="italic" fill="#16A085">(context-driven)</text>

<!-- ═══ ROW 1 ARROWS ═══ -->
<line x1="170" y1="119" x2="185" y2="119" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>
<line x1="346" y1="119" x2="361" y2="119" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>
<line x1="522" y1="119" x2="537" y2="119" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>
<line x1="698" y1="119" x2="713" y2="119" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>

<!-- ═══ SPECIAL ARROW: Gene Program (box5) → CardamomOT GRN (box7) ═══ -->
<g class="blink-arrow">
  <path d="M 795,206 C 795,250 267,240 267,274"
        stroke="#C0392B" stroke-width="2.5" fill="none" stroke-dasharray="7,4"
        marker-end="url(#ahRed)"/>
  <rect x="390" y="228" width="260" height="24" rx="5" fill="white" opacity="0.85"/>
  <text x="520" y="244" text-anchor="middle" font-size="12" font-weight="bold" fill="#C0392B">Gene Program → CardamomOT GRN</text>
</g>

<!-- ═══ ROW 2 TITLE ═══ -->
<text x="460" y="268" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50">GRN Inference &amp; Perturbation Simulation</text>

<!-- ── Box 6: scRNA-seq + Time (blue) ── -->
<rect x="12" y="276" width="158" height="170" rx="8" fill="#EBF4FC" stroke="#5B9BD5" stroke-width="2"/>
<text x="91" y="298" text-anchor="middle" font-size="14" font-weight="bold" fill="#5B9BD5">scRNA-seq</text>
<text x="91" y="314" text-anchor="middle" font-size="13" font-weight="bold" fill="#5B9BD5">+ Time</text>
<rect x="66" y="323" width="50" height="58" rx="3" fill="none" stroke="#5B9BD5" stroke-width="1.5"/>
<line x1="66" y1="338" x2="116" y2="338" stroke="#5B9BD5" stroke-width="0.8" opacity="0.5"/>
<line x1="66" y1="353" x2="116" y2="353" stroke="#5B9BD5" stroke-width="0.8" opacity="0.5"/>
<line x1="82" y1="323" x2="82" y2="381" stroke="#5B9BD5" stroke-width="0.8" opacity="0.5"/>
<line x1="98" y1="323" x2="98" y2="381" stroke="#5B9BD5" stroke-width="0.8" opacity="0.5"/>
<text x="91" y="408" text-anchor="middle" font-size="11" fill="#777">cells × genes × t</text>

<!-- ── Box 7: CARDAMOM GRN (purple, program-augmented) ── -->
<rect x="188" y="276" width="158" height="170" rx="8" fill="#F5EFF9" stroke="#9B59B6" stroke-width="2.5"/>
<!-- teal accent border top to signal gene-program input -->
<rect x="188" y="276" width="158" height="5" rx="3" fill="#16A085" opacity="0.7"/>
<text x="267" y="296" text-anchor="middle" font-size="14" font-weight="bold" fill="#9B59B6">CardamomOT GRN</text>
<!-- "program-augmented" badge -->
<rect x="210" y="302" width="114" height="17" rx="8" fill="#16A085" opacity="0.15" stroke="#16A085" stroke-width="1"/>
<text x="267" y="314" text-anchor="middle" font-size="9.5" font-weight="bold" fill="#16A085">program-augmented</text>
<line x1="267" y1="330" x2="237" y2="360" stroke="#9B59B6" stroke-width="1.5"/>
<line x1="267" y1="330" x2="297" y2="360" stroke="#9B59B6" stroke-width="1.5"/>
<line x1="237" y1="360" x2="267" y2="390" stroke="#9B59B6" stroke-width="1.5"/>
<line x1="297" y1="360" x2="267" y2="390" stroke="#9B59B6" stroke-width="1.5"/>
<line x1="237" y1="360" x2="297" y2="360" stroke="#9B59B6" stroke-width="1.5"/>
<circle cx="267" cy="330" r="6" fill="#9B59B6"/>
<circle cx="237" cy="360" r="6" fill="#9B59B6"/>
<circle cx="297" cy="360" r="6" fill="#9B59B6"/>
<circle cx="267" cy="390" r="6" fill="#9B59B6"/>
<text x="267" y="411" text-anchor="middle" font-size="11" fill="#777">ODE mechanistic model</text>
<text x="267" y="426" text-anchor="middle" font-size="10" font-style="italic" fill="#777">+ optimal transport</text>

<!-- ── Box 8: Inferred GRN (green+red) ── -->
<rect x="364" y="276" width="158" height="170" rx="8" fill="#EAFAF1" stroke="#27AE60" stroke-width="2"/>
<text x="443" y="298" text-anchor="middle" font-size="14" font-weight="bold" fill="#27AE60">Inferred GRN</text>
<line x1="443" y1="322" x2="415" y2="357" stroke="#27AE60" stroke-width="1.8"/>
<line x1="443" y1="322" x2="471" y2="357" stroke="#27AE60" stroke-width="1.8"/>
<line x1="415" y1="357" x2="443" y2="392" stroke="#E74C3C" stroke-width="1.8"/>
<line x1="471" y1="357" x2="443" y2="392" stroke="#27AE60" stroke-width="1.8"/>
<circle cx="443" cy="322" r="7" fill="#27AE60"/>
<circle cx="415" cy="357" r="6" fill="#27AE60"/>
<circle cx="471" cy="357" r="6" fill="#27AE60"/>
<circle cx="443" cy="392" r="6" fill="#E74C3C"/>
<text x="443" y="418" text-anchor="middle" font-size="11" fill="#777">activation / repression</text>

<!-- ── Box 9: KD Simulation (orange) ── -->
<rect x="540" y="276" width="158" height="170" rx="8" fill="#FEF5E7" stroke="#E67E22" stroke-width="2"/>
<text x="619" y="291" text-anchor="middle" font-size="13" font-weight="bold" fill="#E67E22">Perturbation</text>
<text x="619" y="307" text-anchor="middle" font-size="13" font-weight="bold" fill="#E67E22">Simulation</text>
<line x1="619" y1="322" x2="591" y2="357" stroke="#E67E22" stroke-width="1.8" stroke-dasharray="3,2"/>
<line x1="619" y1="322" x2="647" y2="357" stroke="#E67E22" stroke-width="1.8"/>
<line x1="591" y1="357" x2="619" y2="392" stroke="#E67E22" stroke-width="1.8"/>
<line x1="647" y1="357" x2="619" y2="392" stroke="#E67E22" stroke-width="1.8"/>
<circle cx="619" cy="322" r="8" fill="#E74C3C" opacity="0.75"/>
<line x1="612" y1="315" x2="626" y2="329" stroke="white" stroke-width="2.2"/>
<line x1="626" y1="315" x2="612" y2="329" stroke="white" stroke-width="2.2"/>
<circle cx="591" cy="357" r="6" fill="#E67E22"/>
<circle cx="647" cy="357" r="6" fill="#E67E22"/>
<circle cx="619" cy="392" r="6" fill="#E67E22"/>
<text x="619" y="418" text-anchor="middle" font-size="11" fill="#777">BIRC5 KO · propagate Δ</text>

<!-- ── Box 10: Therapeutic Targets (red) ── -->
<rect x="716" y="276" width="158" height="170" rx="8" fill="#FDEDEC" stroke="#E74C3C" stroke-width="2"/>
<text x="795" y="298" text-anchor="middle" font-size="14" font-weight="bold" fill="#E74C3C">Therapeutic</text>
<text x="795" y="314" text-anchor="middle" font-size="13" font-weight="bold" fill="#E74C3C">Targets</text>
<line x1="775" y1="338" x2="815" y2="338" stroke="#E74C3C" stroke-width="1.8"/>
<line x1="775" y1="338" x2="795" y2="375" stroke="#E67E22" stroke-width="1.8"/>
<line x1="815" y1="338" x2="795" y2="375" stroke="#E67E22" stroke-width="1.8"/>
<circle cx="775" cy="338" r="10" fill="#E74C3C"/>
<circle cx="815" cy="338" r="10" fill="#E67E22"/>
<circle cx="795" cy="375" r="8" fill="#999"/>
<text x="775" y="342" text-anchor="middle" font-size="11" fill="white" font-weight="bold">✕</text>
<text x="815" y="342" text-anchor="middle" font-size="11" fill="white" font-weight="bold">✕</text>
<text x="795" y="411" text-anchor="middle" font-size="11" fill="#777">co-targets · direct targets</text>
<text x="795" y="426" text-anchor="middle" font-size="10" font-style="italic" fill="#E74C3C">reshape Waddington landscape</text>

<!-- ═══ ROW 2 ARROWS ═══ -->
<line x1="170" y1="361" x2="185" y2="361" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>
<line x1="346" y1="361" x2="361" y2="361" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>
<line x1="522" y1="361" x2="537" y2="361" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>
<line x1="698" y1="361" x2="713" y2="361" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>

</svg>"""

# ================================================================
# DATASET SELECTOR  (must come before upload so `genes` is defined)
# ================================================================
dataset_choice = st.radio(
    "Dataset",
    options=["RMS original", "RMS 2", "Trajectory GNN (beta)"],
    horizontal=True
)
dataset_key = "v1" if dataset_choice == "RMS original" else ("traj" if dataset_choice == "Trajectory GNN (beta)" else "v2")

with st.spinner("Loading data..."):
    if dataset_key == "traj":
        genes, embeddings, clusters, annotations, summaries, umap_df, expr, gene_names, grn_mat, grn_genes = load_data_traj()
        st.info("Trajectory GNN (beta) — embeddings from GAT + WGCNA per timepoint + OT alignment. "
                "Model weights are random init; run trajectory_gnn.ipynb to regenerate after training.")
    else:
        genes, embeddings, clusters, annotations, summaries, umap_df, expr, gene_names, grn_mat, grn_genes = load_data(dataset_key)

# ================================================================
# UPLOAD YOUR OWN DATA
# ================================================================
with st.expander("📂 Upload your own .h5ad", expanded=False):
    st.caption("Your file is processed in memory only — not stored anywhere. Max recommended: ~50k cells.")
    uploaded_file = st.file_uploader("Upload .h5ad file", type=["h5ad"], key="h5ad_upload")

    if uploaded_file is not None:
        file_id = uploaded_file.name + str(uploaded_file.size)

        if st.session_state.get("_upload_file_id") != file_id:
            import io, anndata as ad, scanpy as sc, scipy.sparse as sp_sparse

            with st.spinner("Reading file..."):
                bytes_data = uploaded_file.read()
                adata = ad.read_h5ad(io.BytesIO(bytes_data))

            st.success(f"✅ Loaded: **{adata.n_obs:,} cells × {adata.n_vars:,} genes**")

            if adata.n_obs > 100_000:
                st.warning("Large dataset — UMAP may be slow or run out of memory on Streamlit Cloud.")

            with st.spinner("Normalizing → PCA → UMAP (1–2 min for ~10k cells)…"):
                sc.pp.normalize_total(adata, target_sum=1e4)
                sc.pp.log1p(adata)
                n_top = min(3000, adata.n_vars)
                sc.pp.highly_variable_genes(adata, n_top_genes=n_top)
                n_comps = min(50, adata.n_obs - 2, adata.n_vars - 1)
                sc.pp.pca(adata, n_comps=n_comps)
                sc.pp.neighbors(adata, n_neighbors=15, n_pcs=min(30, n_comps))
                sc.tl.umap(adata)

            umap_coords = pd.DataFrame(adata.obsm["X_umap"], columns=["x", "y"])
            for col in ["time", "cell_type", "cluster", "leiden", "louvain", "sample"]:
                if col in adata.obs.columns:
                    umap_coords[col] = adata.obs[col].values

            # store gene names (all)
            uploaded_var_names = list(adata.var_names)

            # overlap with RMS embeddings — store expression for those genes only
            # (will be used later when `genes` is available in scope)
            X_full = adata.X
            if sp_sparse.issparse(X_full):
                X_full = X_full.toarray()
            X_f16 = X_full.astype(np.float16)

            st.session_state["_upload_file_id"]   = file_id
            st.session_state["_upload_umap"]       = umap_coords
            st.session_state["_upload_var_names"]  = uploaded_var_names
            st.session_state["_upload_expr"]       = X_f16

        # ── Render ──────────────────────────────────────────────────────
        umap_up    = st.session_state.get("_upload_umap")
        var_names  = st.session_state.get("_upload_var_names", [])
        expr_up    = st.session_state.get("_upload_expr")

        if umap_up is not None:
            meta_cols = [c for c in umap_up.columns if c not in ["x", "y"]]

            overlap = [g for g in var_names if g in set(genes)]
            st.info(f"**{len(overlap):,}** of your {len(var_names):,} genes found in RMS embedding space. "
                    "Type any of them in the chat below to query similar genes and GRN.")

            # ── Auto-show cell_type and time side by side ────────────
            auto_cols = [c for c in ["cell_type", "time"] if c in umap_up.columns]
            if auto_cols:
                auto_figs = st.columns(len(auto_cols))
                for col_ui, meta_col in zip(auto_figs, auto_cols):
                    fig_auto = px.scatter(umap_up, x="x", y="y", color=meta_col,
                                         title=meta_col,
                                         labels={"x": "UMAP 1", "y": "UMAP 2"},
                                         render_mode="webgl", height=400)
                    fig_auto.update_traces(marker=dict(size=2.5, opacity=0.75))
                    fig_auto.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                           margin=dict(l=0, r=0, t=30, b=0))
                    col_ui.plotly_chart(fig_auto, use_container_width=True,
                                        key=f"upload_auto_{meta_col}")

            # ── Gene expression coloring ─────────────────────────────
            gene_sel_up = st.selectbox(
                "Color UMAP by gene expression",
                options=["— none —"] + sorted(var_names),
                key="upload_gene_sel"
            )
            if gene_sel_up != "— none —" and gene_sel_up in var_names:
                g_idx = var_names.index(gene_sel_up)
                plot_up = umap_up.copy()
                plot_up["expression"] = expr_up[:, g_idx].astype(float)
                fig_gene = px.scatter(plot_up, x="x", y="y", color="expression",
                                      color_continuous_scale="Viridis",
                                      title=f"{gene_sel_up}",
                                      labels={"x": "UMAP 1", "y": "UMAP 2"},
                                      render_mode="webgl", height=420)
                fig_gene.update_traces(marker=dict(size=2.5, opacity=0.8))
                fig_gene.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                       margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_gene, use_container_width=True, key="upload_gene_fig")


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

# ── GRN selector — hide only if gene is in NO model at all ──────
_orig_gene_set   = load_grn_gene_list("original")
_mki67_gene_set  = load_grn_gene_list("mki67")
_tubb_gene_set   = load_grn_gene_list("tubb")
_foxm1_gene_set  = load_grn_gene_list("foxm1")
_full_gene_set   = load_grn_gene_list("full")

# model label → (key, gene_set)
_ALL_MODELS = {
    "Full program (200 genes, HSPA1B KO)":    ("full",     _full_gene_set),
    "FOXM1 program (198 genes, FOXM1 KO)":    ("foxm1",    _foxm1_gene_set),
    "MKI67 program (201 genes, BIRC5 KO)":    ("mki67",    _mki67_gene_set),
    "TUBB program (201 genes, TUBB KO)":       ("tubb",     _tubb_gene_set),
    "Original (159 genes)":                    ("original", _orig_gene_set),
}

# use last queried gene (or most recent search) to decide visibility
_last_q      = st.session_state.get(f"last_selected_{dataset_key}", "")
_recent_list = st.session_state.get(f"recent_{dataset_key}", [])
_check_gene  = (_last_q or (_recent_list[0] if _recent_list else "")).strip().upper()

_gene_in_any_grn = (not _check_gene) or any(
    _check_gene in gs for _, gs in _ALL_MODELS.values()
)

_grn_state_key = f"grn_choice_{dataset_key}"
if not _gene_in_any_grn:
    grn_mat, grn_genes = None, []
else:
    if _check_gene:
        grn_options = [label for label, (_, gs) in _ALL_MODELS.items()
                       if _check_gene in gs]
    else:
        grn_options = list(_ALL_MODELS.keys())

    if len(grn_options) == 1:
        grn_choice = grn_options[0]
        st.caption(f"GRN model: **{grn_choice}**")
    else:
        if st.session_state.get(_grn_state_key, "") not in grn_options:
            st.session_state[_grn_state_key] = grn_options[0]
        grn_choice = st.radio(
            "GRN model", options=grn_options,
            horizontal=True, key=_grn_state_key
        )

    grn_key = _ALL_MODELS[grn_choice][0]
    with st.spinner("Loading GRN..."):
        grn_mat, grn_genes = load_grn(grn_key)

# 🔬 icon in search = gene present in ANY GRN model
grn_gene_set = _mki67_gene_set | _orig_gene_set | _tubb_gene_set | _foxm1_gene_set | _full_gene_set

# ── Recent searches ───────────────────────────────────────────────
recent_key = f"recent_{dataset_key}"
if recent_key not in st.session_state:
    st.session_state[recent_key] = []
recent = st.session_state[recent_key]

def gene_label(g, is_recent=False):
    prefix = "🕐 " if is_recent else ""
    suffix = " 🔬" if g in grn_gene_set else ""
    return f"{prefix}{g}{suffix}"

# recent первыми, потом все остальные по алфавиту
recent_labels   = [gene_label(g, is_recent=True) for g in recent]
all_labels      = [gene_label(g) for g in sorted(genes) if g not in recent]
separator       = ["─── Recent ───"] if recent else []
separator2      = ["─── All genes ───"] if recent else []
gene_labels_all = [""] + separator + recent_labels + separator2 + all_labels

selected_label = col_search.selectbox(
    "Quick gene search  (🕐 = recent · 🔬 = GRN)",
    options=gene_labels_all,
    index=0,
    key=f"selectbox_{dataset_key}"
)
# strip prefixes/suffixes to get clean gene name
selected_gene = (selected_label
    .replace("🕐 ", "").replace(" 🔬", "").replace("🔬 ", "")
    .strip()) if selected_label and not selected_label.startswith("─") else ""
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

load_real_expr_means()

if f"messages_{dataset_key}" not in st.session_state:
    st.session_state[f"messages_{dataset_key}"] = []

if f"last_selected_{dataset_key}" not in st.session_state:
    st.session_state[f"last_selected_{dataset_key}"] = ""

if f"default_run_{dataset_key}" not in st.session_state:
    st.session_state[f"default_run_{dataset_key}"] = True

messages = st.session_state[f"messages_{dataset_key}"]

def _render_msg_figures(msg, msg_id):
    """Render all figures for a single message."""
    if "df" in msg:
        st.dataframe(msg["df"], use_container_width=True)
        if "cluster_id" in msg:
            summary = summaries.get(msg["cluster_id"], "")
            if summary:
                with st.popover("Cluster annotation details"):
                    st.markdown(summary)
    # Expression UMAP — full width, taller
    if msg.get("fig") is not None:
        fig_expr = msg["fig"]
        fig_expr.update_layout(height=CHART_H)
        st.plotly_chart(fig_expr, use_container_width=True, key=f"{msg_id}_fig")
    # UMAPs — row 1: Time / Sim-time / Cell type
    _row1_keys = ["fig_time", "fig_sim_time", "fig_celltype"]
    _row1 = [(k, msg.get(k)) for k in _row1_keys if msg.get(k) is not None]
    if _row1:
        cols = st.columns(1 if is_mobile else len(_row1))
        for col, (k, f) in zip(cols, _row1):
            f.update_layout(height=CHART_H_SMALL)
            col.plotly_chart(f, use_container_width=True, key=f"{msg_id}_{k}")
    # Population panels for foxm1/tubb/mki67 are shown inside perturbation tab
    if "grn_fig" in msg and msg["grn_fig"] is not None:
            _msg_grn_model = msg.get("grn_model")
            _has_pert = _msg_grn_model in ("mki67", "tubb", "foxm1", "full")
            _ko_gene_label = {"mki67": "BIRC5", "tubb": "TUBB", "foxm1": "FOXM1", "full": "HSPA1B"}.get(_msg_grn_model, "")
            if _has_pert:
                _tab_results = st.tabs([f"🧬 {_ko_gene_label} KO Perturbation", "Network graph", "Adjacency matrix"])
                tab_pert, tab_graph, tab_matrix = _tab_results
            else:
                _tab_results = st.tabs(["Network graph", "Adjacency matrix"])
                tab_graph, tab_matrix = _tab_results
                tab_pert = None
            with tab_graph:
                st.plotly_chart(msg["grn_fig"], use_container_width=True, key=f"{msg_id}_grn")
                topo = msg.get("grn_topo")
                if topo is not None and not topo.empty:
                    with st.expander("📊 Network topology analysis", expanded=True):
                        # summary metrics
                        n_fb = int(topo["Feedback loop"].sum())
                        n_up = int((topo["Role"] == "upstream").sum())
                        n_dn = int((topo["Role"] == "downstream").sum())
                        n_fb_role = int((topo["Role"] == "feedback loop").sum())
                        top_hub = topo[topo["Role"] != "query"].nlargest(1, "Total degree")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Upstream regulators", n_up)
                        c2.metric("Downstream targets", n_dn)
                        c3.metric("Feedback loop nodes", n_fb_role)
                        c4.metric("Nodes in any cycle", n_fb)
                        if not top_hub.empty:
                            hub_name = top_hub.iloc[0]["Gene"]
                            hub_deg  = int(top_hub.iloc[0]["Total degree"])
                            st.info(f"🔵 **Top hub**: **{hub_name}** (degree {hub_deg}) — "
                                    f"most connected node in this subnetwork")
                        # table
                        show_cols = ["Gene", "Role", "In-degree", "Out-degree",
                                     "Total degree", "Betweenness", "Feedback loop"]
                        st.dataframe(
                            topo[show_cols].reset_index(drop=True),
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Betweenness": st.column_config.ProgressColumn(
                                    "Betweenness", min_value=0, max_value=1, format="%.3f"),
                                "Feedback loop": st.column_config.CheckboxColumn("🔄 Cycle"),
                            }
                        )
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
            if tab_pert is not None:
                with tab_pert:
                    try:
                        pert_df = load_perturbation(_msg_grn_model)
                        q_gene  = msg.get("query_gene", "MKI67")
                        _real_means = load_real_expr_means()
                        bar_fig, line_fig = build_perturbation_figures(
                            pert_df, q_gene, ko_gene=_ko_gene_label,
                            real_expr_means=_real_means)
                        st.plotly_chart(bar_fig, use_container_width=True, key=f"{msg_id}_pert_bar")
                        # Population proportions + delta + UMAP — foxm1, tubb, mki67
                        if _msg_grn_model in ("foxm1", "tubb", "mki67", "full"):
                            try:
                                if _msg_grn_model == "foxm1":
                                    _sim_scored = load_foxm1_pop_sim()
                                    _ko_label   = "FOXM1 KO"
                                elif _msg_grn_model == "tubb":
                                    _sim_scored = load_tubb_pop_sim()
                                    _ko_label   = "TUBB KO"
                                elif _msg_grn_model == "mki67":
                                    _sim_scored = load_mki67_pop_sim()
                                    _ko_label   = "BIRC5 KO"
                                else:
                                    _sim_scored = load_full_pop_sim()
                                    _ko_label   = "HSPA1B KO"
                                _POP_COLORS = {"proliferative": "#e63946", "quiescent": "#1a6faf",
                                               "intermediate": "#999999"}
                                _prop_fig  = build_population_proportions_figure(_sim_scored, ko_label=_ko_label)
                                _delta_fig = build_population_delta_figure(_sim_scored, ko_label=_ko_label)
                                st.plotly_chart(_prop_fig,  use_container_width=True, key=f"{msg_id}_pop_prop")
                                st.plotly_chart(_delta_fig, use_container_width=True, key=f"{msg_id}_pop_delta")
                                # UMAP: Real data | WT simulation | KO simulation
                                _umap_col1, _umap_col2, _umap_col3 = st.columns(3)
                                _pt_pop_order = ["quiescent", "proliferative", "intermediate"]  # intermediate on top
                                _pt_pop_sizes = {"intermediate": 2, "proliferative": 3, "quiescent": 4}
                                # Real data population UMAP
                                try:
                                    _pr = load_foxm1_pop_real()
                                    _pop_umap_real = px.scatter(
                                        _pr, x="x", y="y", color="population",
                                        color_discrete_map=_POP_COLORS,
                                        title="Real data (DNAJB1 / MKI67-100)",
                                        labels={"x": "UMAP 1", "y": "UMAP 2", "population": "Population"},
                                        opacity=0.6, height=420, render_mode="svg",
                                        category_orders={"population": _pt_pop_order},
                                    )
                                    for _pp, _ps in _pt_pop_sizes.items():
                                        _pop_umap_real.update_traces(marker=dict(size=_ps), selector=dict(name=_pp))
                                    _pop_umap_real.update_layout(plot_bgcolor="white", paper_bgcolor="white")
                                    _umap_col1.plotly_chart(_pop_umap_real, use_container_width=True, key=f"{msg_id}_pop_umap_real")
                                except Exception:
                                    pass
                                # WT simulation UMAP
                                _pop_umap_wt = px.scatter(
                                    _sim_scored, x="x_wt", y="y_wt", color="pop_wt",
                                    color_discrete_map=_POP_COLORS,
                                    title="Simulation WT (DNAJB1 / MKI67-100)",
                                    labels={"x_wt": "UMAP 1", "y_wt": "UMAP 2", "pop_wt": "Population"},
                                    opacity=0.6, height=420, render_mode="svg",
                                    category_orders={"pop_wt": _pt_pop_order},
                                )
                                for _pp, _ps in _pt_pop_sizes.items():
                                    _pop_umap_wt.update_traces(marker=dict(size=_ps), selector=dict(name=_pp))
                                _pop_umap_wt.update_layout(plot_bgcolor="white", paper_bgcolor="white")
                                # KO simulation UMAP
                                _pop_umap_ko = px.scatter(
                                    _sim_scored, x="x_ko", y="y_ko", color="pop_ko",
                                    color_discrete_map=_POP_COLORS,
                                    title=f"Simulation {_ko_label} (DNAJB1 / MKI67-100)",
                                    labels={"x_ko": "UMAP 1", "y_ko": "UMAP 2", "pop_ko": "Population"},
                                    opacity=0.6, height=420, render_mode="svg",
                                    category_orders={"pop_ko": _pt_pop_order},
                                )
                                for _pp, _ps in _pt_pop_sizes.items():
                                    _pop_umap_ko.update_traces(marker=dict(size=_ps), selector=dict(name=_pp))
                                _pop_umap_ko.update_layout(plot_bgcolor="white", paper_bgcolor="white")
                                _umap_col2.plotly_chart(_pop_umap_wt, use_container_width=True, key=f"{msg_id}_pop_umap_wt")
                                _umap_col3.plotly_chart(_pop_umap_ko, use_container_width=True, key=f"{msg_id}_pop_umap_ko")
                            except Exception as _e:
                                st.info(f"Population shift unavailable: {_e}")
                        _prog_map = {"tubb": "TUBB program (201 genes)", "foxm1": "FOXM1 program (198 genes)", "mki67": "MKI67 program (201 genes)", "full": "Full program (200 genes)"}
                        _prog = _prog_map.get(_msg_grn_model, "program")
                        st.caption(f"Simulation: CARDAMOM mechanistic model · {_prog} · {_ko_gene_label} knocked out")
                    except Exception as e:
                        st.info(f"Perturbation data not available. ({e})")

# ── Message rendering loop ───────────────────────────────────────
# Only the LAST assistant message is fully expanded.
# All older messages are collapsed in expanders (no Plotly rendered = fast).
_asst_indices = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
_last_asst_idx = _asst_indices[-1] if _asst_indices else -1

for _mi, msg in enumerate(messages):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "fig" in msg:
            if _mi == _last_asst_idx:
                # Latest result — render everything
                _render_msg_figures(msg, id(msg))
            else:
                # Older result — collapsed, no heavy rendering
                gene_label = msg.get("query_gene", "")
                with st.expander(f"📊 Show results for {gene_label}", expanded=False):
                    _render_msg_figures(msg, id(msg))

if st.session_state.get(f"default_run_{dataset_key}"):
    st.session_state[f"default_run_{dataset_key}"] = False
    # First: TUBB (tubb model)
    st.session_state[f"forced_grn_{dataset_key}"] = "tubb"
    st.session_state[f"default_run2_{dataset_key}"] = True
    query_gene = "TUBB" if "TUBB" in genes else genes[0]
elif st.session_state.get(f"default_run2_{dataset_key}"):
    st.session_state[f"default_run2_{dataset_key}"] = False
    # Second: BIRC5 (mki67 model)
    st.session_state[f"forced_grn_{dataset_key}"] = "mki67"
    st.session_state[f"default_run3_{dataset_key}"] = True
    query_gene = "BIRC5" if "BIRC5" in genes else None
elif st.session_state.get(f"default_run3_{dataset_key}"):
    st.session_state[f"default_run3_{dataset_key}"] = False
    # Third (last): HSPA1B (full model) — shown last so it stays on top
    st.session_state[f"forced_grn_{dataset_key}"] = "full"
    query_gene = "HSPA1B" if "HSPA1B" in genes else None
elif st.session_state.get(f"recent_clicked_{dataset_key}"):
    query_gene = st.session_state.pop(f"recent_clicked_{dataset_key}")
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

    # Save to recent searches (max 8, no duplicates)
    recent = st.session_state.get(recent_key, [])
    if query_gene in recent:
        recent.remove(query_gene)
    recent.insert(0, query_gene)
    st.session_state[recent_key] = recent[:8]

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

        # Downsample UMAP to max 4000 cells for fast SVG rendering
        _UMAP_MAX = 4000
        _n_cells = len(umap_df)
        if _n_cells > _UMAP_MAX:
            _sample_idx = np.random.default_rng(42).choice(_n_cells, _UMAP_MAX, replace=False)
            _sample_idx = np.sort(_sample_idx)
            umap_plot = umap_df.iloc[_sample_idx].copy()
        else:
            _sample_idx = np.arange(_n_cells)
            umap_plot = umap_df.copy()

        fig = None
        fig_time = None
        fig_celltype = None
        if query_gene in gene_names:
            gene_idx = gene_names.index(query_gene)
            expr_vals = expr[_sample_idx, gene_idx].astype(float)
            umap_plot["expression"] = expr_vals

            fig = px.scatter(
                umap_plot, x="x", y="y",
                color="expression",
                color_continuous_scale="Viridis",
                title=f"{query_gene} — Expression ({len(umap_plot):,} cells shown)",
                labels={"x": "UMAP 1", "y": "UMAP 2"},
                opacity=0.6, height=450,
                render_mode="svg"
            )
            fig.update_traces(marker=dict(size=3))
            fig.update_layout(coloraxis_colorbar=dict(title="Expression"),
                              plot_bgcolor="white", paper_bgcolor="white")

        if "time" in umap_df.columns:
            fig_time = px.scatter(
                umap_plot, x="x", y="y",
                color=umap_plot["time"].astype(str),
                title="Time",
                labels={"x": "UMAP 1", "y": "UMAP 2", "color": "Time"},
                opacity=0.6, height=450,
                render_mode="svg",
                category_orders={"color": [str(t) for t in sorted(umap_df["time"].unique())]}
            )
            fig_time.update_traces(marker=dict(size=3))
            fig_time.update_layout(plot_bgcolor="white", paper_bgcolor="white")

        if "cell_type" in umap_df.columns:
            fig_celltype = px.scatter(
                umap_plot, x="x", y="y",
                color="cell_type",
                title="Cell Type",
                labels={"x": "UMAP 1", "y": "UMAP 2", "color": "Cell Type"},
                opacity=0.6, height=450,
                render_mode="svg"
            )
            fig_celltype.update_traces(marker=dict(size=3))
            fig_celltype.update_layout(plot_bgcolor="white", paper_bgcolor="white")

        program_genes = [query_gene] + [genes[i] for i in sorted_idx]

        # Always pick the right GRN model for THIS query gene (not the selector,
        # which lags one step behind because it's rendered before query processing)
        _q = query_gene.upper()
        _q_models = [k for k, (key, gs) in _ALL_MODELS.items() if _q in gs]
        if len(_q_models) == 1:
            # unambiguous — use the only matching model
            _grn_model_label = _q_models[0]
            _grn_mat_q, _grn_genes_q = load_grn(_ALL_MODELS[_grn_model_label][0])
            _grn_model_q = _ALL_MODELS[_grn_model_label][0]
        elif len(_q_models) > 1:
            # gene in multiple models → check forced flag first, else respect selector
            _forced_key = st.session_state.pop(f"forced_grn_{dataset_key}", None)
            _forced_valid = _forced_key and any(
                _ALL_MODELS[k][0] == _forced_key for k in _q_models)
            if _forced_valid:
                _grn_model_q = _forced_key
                _grn_mat_q, _grn_genes_q = load_grn(_forced_key)
                # Note: cannot set _grn_state_key here — widget already rendered this rerun.
                # The correct model is stored in the message; selector syncs on next rerun.
            else:
                _grn_mat_q, _grn_genes_q = grn_mat, grn_genes
                _grn_model_q = grn_key if grn_mat is not None else None
        else:
            # gene in no model
            _grn_mat_q, _grn_genes_q = None, []
            _grn_model_q = None

        grn_fig, grn_topo = build_grn_figure(_grn_mat_q, _grn_genes_q, query_gene, gene_set=program_genes, hops=grn_hops)
        grn_adj = build_grn_adjacency(_grn_mat_q, _grn_genes_q, gene_set=program_genes, query_gene=query_gene, hops=grn_hops)

        # Simulation time UMAP — all GRN models
        # Sim cells projected via UMAP transform (fit on real data, embedding
        # replaced with original Scanpy UMAP, then transform applied to sim cells)
        _sim_proj_loaders = {
            "foxm1": load_foxm1_sim_umap_proj,
            "mki67": load_mki67_sim_umap_proj,
            "tubb":  load_tubb_sim_umap_proj,
            "full":  load_full_sim_umap_proj,
            # original GRN: 159 differentiation genes don't overlap with
            # cell-cycle UMAP space → projection collapses to one point, omitted
        }
        fig_sim_time = None
        if _grn_model_q in _sim_proj_loaders:
            try:
                _proj = _sim_proj_loaders[_grn_model_q]()
                _xcol = "x_wt" if "x_wt" in _proj.columns else "x"
                _ycol = "y_wt" if "y_wt" in _proj.columns else "y"
                _tcol = "time"  if "time"  in _proj.columns else "time_sim"
                _t_order = [str(int(t)) for t in sorted(_proj[_tcol].unique())]
                fig_sim_time = px.scatter(
                    _proj, x=_xcol, y=_ycol,
                    color=_proj[_tcol].astype(int).astype(str),
                    title="Time (simulation WT)",
                    labels={_xcol: "UMAP 1", _ycol: "UMAP 2", "color": "Time"},
                    opacity=0.6, height=450,
                    render_mode="svg",
                    category_orders={"color": _t_order},
                )
                fig_sim_time.update_traces(marker=dict(size=3))
                fig_sim_time.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            except Exception:
                pass

        # Population panels — real data (panel 4) and simulation (panel 5)
        # Real data: shown for foxm1 + original models
        # Simulation: foxm1 only (only model with scored sim CSV)
        # Quiescent = DNAJB1 z-score (top HSPA1B neighbour); Proliferative = mean(top-100 MKI67 neighbours)
        _POP_COLORS = {"proliferative": "#e63946", "quiescent": "#1a6faf", "intermediate": "#999999"}
        _POP_ORDER  = ["quiescent", "proliferative", "intermediate"]  # intermediate drawn on top (largest group)
        _POP_SIZES  = {"intermediate": 2, "proliferative": 3, "quiescent": 4}
        fig_pop_real = None
        fig_pop_sim  = None
        if _grn_model_q in ("foxm1", "tubb", "mki67", "full"):
            try:
                _pr = load_foxm1_pop_real()
                fig_pop_real = px.scatter(
                    _pr, x="x", y="y",
                    color="population",
                    color_discrete_map=_POP_COLORS,
                    title="Population — real data (DNAJB1 / MKI67-100)",
                    labels={"x": "UMAP 1", "y": "UMAP 2", "population": "Population"},
                    opacity=0.6, height=450, render_mode="svg",
                    category_orders={"population": _POP_ORDER},
                )
                for _pop, _sz in _POP_SIZES.items():
                    fig_pop_real.update_traces(marker=dict(size=_sz), selector=dict(name=_pop))
                fig_pop_real.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            except Exception:
                pass
        # Population sim — original GRN has no cell-cycle genes → skip
        if _grn_model_q in ("foxm1", "tubb", "mki67", "full"):
            try:
                if _grn_model_q == "foxm1":
                    _ps = load_foxm1_pop_sim()
                elif _grn_model_q == "tubb":
                    _ps = load_tubb_pop_sim()
                elif _grn_model_q == "mki67":
                    _ps = load_mki67_pop_sim()
                else:
                    _ps = load_full_pop_sim()
                fig_pop_sim = px.scatter(
                    _ps, x="x_wt", y="y_wt",
                    color="pop_wt",
                    color_discrete_map=_POP_COLORS,
                    title="Population — simulation WT (DNAJB1 / MKI67-100)",
                    labels={"x_wt": "UMAP 1", "y_wt": "UMAP 2", "pop_wt": "Population"},
                    opacity=0.6, height=450, render_mode="svg",
                    category_orders={"pop_wt": _POP_ORDER},
                )
                for _pop, _sz in _POP_SIZES.items():
                    fig_pop_sim.update_traces(marker=dict(size=_sz), selector=dict(name=_pop))
                fig_pop_sim.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            except Exception:
                pass

        messages.append({
            "role": "assistant",
            "content": f"**Gene program for {query_gene}** — cluster {query_cluster}: *{query_annotation}*",
            "df": df,
            "cluster_id": query_cluster,
            "query_gene": query_gene,
            "fig": fig,
            "fig_time": fig_time,
            "fig_sim_time": fig_sim_time,
            "fig_celltype": fig_celltype,
            "fig_pop_real": fig_pop_real,
            "fig_pop_sim":  fig_pop_sim,
            "grn_fig": grn_fig,
            "grn_topo": grn_topo,
            "grn_adj": grn_adj,
            "grn_model": _grn_model_q,
        })
        st.rerun()
