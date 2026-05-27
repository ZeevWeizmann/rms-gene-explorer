import streamlit as st
import os

st.set_page_config(
    page_title="About — Gene Program Explorer",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("# 🧬 About Gene Program Explorer")

st.markdown("<a href='/' target='_self' style='font-size:0.9rem;color:#555;text-decoration:none;'>← Back to Explorer</a>", unsafe_allow_html=True)

st.markdown("---")

# ── About this tool ──────────────────────────────────────────────
st.markdown("## About this tool")

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

**Available GRN models:**
- **Original** — 159 genes, inferred from full RMS scRNA-seq data
- **Full program** — 200 genes (complete quiescent + proliferative gene set), HSPA1B KO
- **FOXM1 program** — 198 genes (top-200 GNN neighbors of FOXM1), FOXM1 KO
- **MKI67 program** — 201 genes (top-200 GNN neighbors of MKI67), BIRC5 KO
- **TUBB program** — 201 genes (top-200 GNN neighbors of TUBB), TUBB KO

**Population dynamics under knockout (Full program):**
RMS tumours contain three co-existing cell states — **Proliferative**, **Quiescent**, and **Intermediate** — in dynamic equilibrium.
Each state is scored from scRNA-seq data: **DNAJB1 z-score** for quiescence; **mean of top-100 MKI67 neighbours** for proliferation.
After KO, CARDAMOM propagates the perturbation through the GRN and re-simulates cell trajectories — revealing how the Waddington landscape reorganises.

**Gene program annotation (LLM):**
Each retrieved gene program is automatically annotated by **Llama 3.1-8B** (via Nebius AI Studio) — generating concise biological labels (e.g. *"Mitotic Cell Proliferation"*, *"Cytoskeletal remodelling"*).

**Cell population scoring (DNAJB1 / MKI67-100):**
- **Proliferative** (red) — mean expression of top-100 MKI67 neighbours ≥ 70th percentile
- **Quiescent** (blue) — DNAJB1 z-score ≥ 70th percentile
- **Intermediate** (grey) — all remaining cells

**References:**
- CARDAMOM / CardamomOT: [github.com/eliasventre/CardamomOT](https://github.com/eliasventre/CardamomOT)
- Nebius AI Studio (Llama 3.1-8B): [studio.nebius.com](https://studio.nebius.com)
""")

st.markdown("---")

# ── Architecture diagram ─────────────────────────────────────────
LOCAL_DIR = "/Users/zeev/CardamomOT/my_project/Data"

_arch_svg_path = os.path.join(LOCAL_DIR, "architecture.png")
if os.path.exists(_arch_svg_path):
    st.image(_arch_svg_path, use_column_width=True)

st.markdown("---")

# ── About Gene Trajectory Graph Embeddings ───────────────────────
st.markdown("## Gene Trajectory Graph Embeddings")

st.markdown("""
Each gene receives a vector that encodes **how its co-expression neighbourhood changed over time**.
Two sources of information are combined:

- **Temporal trajectory** — WGCNA co-expression graphs are built per time point, encoded by a shared GAT, and aligned with optimal transport.
- **Regulatory structure** — a PPGN (WL-3) runs on the OmniPath mechanistic interaction graph (via NEKO) and captures regulatory motifs such as feedback loops and triangles.
  Added on top of the trajectory embedding for genes with known OmniPath interactions.

**References:**
- Maron et al. *Provably Powerful Graph Networks.* NeurIPS 2019 · [arXiv:1905.11136](https://arxiv.org/abs/1905.11136)
- CARDAMOM / CardamomOT: [github.com/eliasventre/CardamomOT](https://github.com/eliasventre/CardamomOT)
- OmniPath / NEKO: [omnipathdb.org](https://omnipathdb.org)
""")

# ── Trajectory architecture SVG ─────────────────────────────────
_ot_img = os.path.join(LOCAL_DIR, "ot_explanation.png")
if os.path.exists(_ot_img):
    st.image(_ot_img, caption="Optimal transport alignment of gene embeddings across timepoints",
             use_column_width=True)
