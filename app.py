import streamlit as st
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
    # co-targets: compensatory survival genes that go UP after BIRC5 KO
    co_targets     = {"PPP1R12B", "MAP3K21"}
    # direct targets: overexpressed in cancer, essential for cytokinesis
    direct_targets = {"CEP55"}
    all_targets    = co_targets | direct_targets

    summary = pert_df[pert_df["time"] == last_t].copy()
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
    annotations = []
    for gene, val in zip(top20["gene"].tolist(), top20["log2fc"].tolist()):
        if gene in all_targets:
            if gene in direct_targets:
                label = "◀ direct target (overexpressed)"
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

    bar_fig = go.Figure(go.Bar(
        x=top20["log2fc"], y=top20["gene"],
        orientation="h",
        marker_color=colors,
        hovertemplate="%{y}: %{x:.3f}<extra></extra>"
    ))
    bar_fig.update_layout(
        title=dict(
            text=(
                f"Top 20 genes affected by BIRC5 KO (t={int(last_t)})<br>"
                "<sup style='color:#FF8C00'>🟠 co-target: goes UP after KO — compensatory escape mechanism &nbsp;|&nbsp;"
                " direct target: overexpressed in cancer, drives cytokinesis (CEP55)</sup>"
            ),
            font=dict(size=14),
        ),
        xaxis_title="log₂FC (KO / WT)",
        height=520, margin=dict(l=80, r=80, t=80, b=40),
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
import os as _os
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
    import base64
    with open(_logo_local, "rb") as _f:
        _logo_b64 = base64.b64encode(_f.read()).decode()

st.markdown(f"""
<div style='display:flex; align-items:center; gap:12px; flex-wrap:nowrap; margin-bottom:4px;'>
  {"<img src='data:image/png;base64," + _logo_b64 + "' style='height:60px; width:auto; flex-shrink:0;'/>" if _logo_b64 else ""}
  <div>
    <div style='white-space:nowrap;'>
      <span style='font-size:clamp(1.2rem,4vw,2.2rem); font-weight:800; color:#002395;'>Gene</span>
      <span style='font-size:clamp(1.2rem,4vw,2.2rem); font-weight:800; color:#555555;'> Program </span>
      <span style='font-size:clamp(1.2rem,4vw,2.2rem); font-weight:800; color:#ED2939;'>Explorer</span>
      <span style='background:#E8A838; color:white; font-size:0.6rem; font-weight:700;
        padding:2px 7px; border-radius:10px; margin-left:8px;
        vertical-align:middle;'>BETA</span>
    </div>
    <div style='font-size:clamp(0.65rem,2vw,0.85rem); color:#888; margin-top:2px;'>
      <b>14,581</b> embeddings &nbsp;·&nbsp; <b>359</b> GRN genes &nbsp;·&nbsp; BIRC5 KO &nbsp;·&nbsp; RMS
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

with st.expander("About this tool"):
    st.markdown("""
This is a **RAG-based gene program retrieval system** applied to single-cell RMS (Rhabdomyosarcoma) data.

**How it works:**
1. A query gene is embedded in a GNN co-expression space (trained on WGCNA graphs from scRNA-seq)
2. Nearest neighbors in embedding space define a **transcriptional program**
3. The program is displayed on a cell UMAP and mapped to its **gene regulatory network (GRN)** inferred by CARDAMOM
4. In silico **perturbation simulations** (e.g. BIRC5 knockout) reveal which genes change — enabling **network-based target identification**

**Therapeutic target logic (network perturbation approach):**
- **Direct targets** — genes overexpressed in the tumor that are essential nodes in the GRN (e.g. CEP55: drives cytokinesis, supra-expressed in RMS)
- **Co-targets** — genes that go *up* after BIRC5 KO, acting as compensatory escape mechanisms (e.g. PPP1R12B, MAP3K21); blocking them alongside BIRC5 leaves the cell no survival route
- This approach is called **network-informed synthetic lethality** — targets are chosen not in isolation but based on their role in the regulatory network under perturbation

**Upload your own data:**
Upload any `.h5ad` file to compute a UMAP and visualize gene expression in your dataset.
Genes overlapping with the RMS embedding space can be queried directly in the chat to retrieve co-expression programs and GRN context from the RMS model.

**Available GRN models:**
- **Original** — 159 genes, inferred from full RMS scRNA-seq data
- **MKI67 program** — 201 genes (top-200 GNN neighbors of MKI67), BIRC5 KO perturbation simulated via CARDAMOM mechanistic model

**References:**
- CARDAMOM / CardamomOT: [github.com/eliasventre/CardamomOT](https://github.com/eliasventre/CardamomOT)
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
# DATASET SELECTOR  (must come before upload so `genes` is defined)
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
            fig_expr.update_layout(height=CHART_H)
            st.plotly_chart(fig_expr, use_container_width=True, key=f"{msg_id}_fig")
        # Time + Cell type UMAPs — side by side, smaller
        side_figs = [(k, msg.get(k)) for k in ["fig_time", "fig_celltype"] if msg.get(k) is not None]
        if side_figs:
            cols = st.columns(1 if is_mobile else len(side_figs))
            for col, (k, f) in zip(cols, side_figs):
                f.update_layout(height=CHART_H_SMALL)
                col.plotly_chart(f, use_container_width=True, key=f"{msg_id}_{k}")
        if "grn_fig" in msg and msg["grn_fig"] is not None:
            tab_pert, tab_graph, tab_matrix = st.tabs(["🧬 BIRC5 KO Perturbation", "Network graph", "Adjacency matrix"])
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
        grn_fig, grn_topo = build_grn_figure(grn_mat, grn_genes, query_gene, gene_set=program_genes, hops=grn_hops)
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
            "grn_topo": grn_topo,
            "grn_adj": grn_adj
        })
        st.rerun()
