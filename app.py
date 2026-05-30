import streamlit as st
import streamlit.components.v1 as _components
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import hf_hub_download
import json
import os as _os

# ── Recent searches (session + optional disk persistence) ─────────
_RECENT_FILE = _os.path.join(_os.path.dirname(__file__), "recent_searches.json")
_MAX_RECENT  = 5
_RECENT_SS_KEY = "__recent_searches__"   # key inside st.session_state

def _load_recent(dataset_key: str) -> list[str]:
    """Load per-dataset recent searches: session_state first, then disk fallback."""
    # 1. session_state (works on cloud, survives reruns)
    ss = st.session_state.get(_RECENT_SS_KEY, {})
    if dataset_key in ss:
        return list(ss[dataset_key])
    # 2. disk (works locally, skipped silently if unavailable)
    try:
        with open(_RECENT_FILE, "r") as f:
            data = json.load(f)
        recent = data.get(dataset_key, [])
        # seed session_state from disk so subsequent calls are fast
        if _RECENT_SS_KEY not in st.session_state:
            st.session_state[_RECENT_SS_KEY] = {}
        st.session_state[_RECENT_SS_KEY][dataset_key] = recent
        return recent
    except Exception:
        return []

def _save_recent(dataset_key: str, gene: str) -> list[str]:
    """Prepend *gene* to the recent list, deduplicate, cap at _MAX_RECENT, persist."""
    # Update session_state
    if _RECENT_SS_KEY not in st.session_state:
        st.session_state[_RECENT_SS_KEY] = {}
    recent = list(st.session_state[_RECENT_SS_KEY].get(dataset_key, []))
    recent = [g for g in recent if g != gene]
    recent.insert(0, gene)
    recent = recent[:_MAX_RECENT]
    st.session_state[_RECENT_SS_KEY][dataset_key] = recent
    # Also persist to disk (best-effort, silent on cloud)
    try:
        try:
            with open(_RECENT_FILE, "r") as f:
                data = json.load(f)
        except Exception:
            data = {}
        data[dataset_key] = recent
        with open(_RECENT_FILE, "w") as f:
            json.dump(data, f)
    except Exception:
        pass
    return recent

# ── Translations ─────────────────────────────────────────────────
_TRANSLATIONS = {
    'en': {
        # Search
        'search_placeholder': '🔍 Select a gene to explore its program',
        # Main sections
        'settings': 'Settings',
        'genes_from_data': 'Genes from your data',
        'featured_targets': 'Featured drug targets',
        'about': 'About Gene Program Explorer',
        # Settings labels
        'vector_db': 'Vector database',
        'program_size': 'Program size',
        'grn_hops': 'GRN hops',
        'grn_model': 'GRN model',
        # Data upload
        'upload_note': 'Processed in memory only — not stored anywhere. Max ~50k cells.',
        'upload_label': 'Upload .h5ad file',
        'loading': 'Loading...',
        'reading_file': 'Reading file...',
        'normalizing': 'Normalizing → PCA → UMAP (1–2 min for ~10k cells)…',
        'large_dataset': 'Large dataset — UMAP may be slow or run out of memory on Streamlit Cloud.',
        'color_umap': 'Color UMAP by gene expression',
        # Population labels
        'quiescent': 'quiescent',
        'proliferative': 'proliferative',
        'intermediate': 'intermediate',
        'real_data': 'Real data (populations *)',
        'sim_wt': 'Simulation WT (populations *)',
        'population': 'Population',
        'pop_shift': 'Population shift after {gene} KO (Δ%)',
        'delta_pct': 'Δ% (KO – WT)',
        # Result tabs
        'gene_program': 'Gene Program',
        # Network tabs
        'ko_perturbation': 'KO Simulation',
        'network_graph': 'Gene Network',
        'adjacency_matrix': 'Network Weights',
        'network_topology': '📊 Network topology analysis',
        'upstream': 'Upstream regulators',
        'downstream': 'Downstream targets',
        'feedback': 'Feedback loop nodes',
        'cycle_nodes': 'Nodes in any cycle',
        # Plot labels
        'umap1': 'UMAP 1',
        'umap2': 'UMAP 2',
        'expression': 'Expression',
        'time': 'Time',
        'cell_type': 'Cell Type',
        'time_sim': 'Time (simulation WT)',
        'knocked_out': 'knocked out',
        # About section (short)
        'about_intro': 'This is a **RAG-based gene program retrieval system** applied to single-cell RMS (Rhabdomyosarcoma) data.',
        'how_it_works': '**How it works**',
        'show_results': '📊 Show results for',
        'gene_program_for': 'Gene program for',
        'cluster': 'cluster',
        'top_20': 'Top 20 genes affected by',
        'top_hub': '🔵 **Top hub**',
        'most_connected': 'most connected node in this subnetwork',
        'sim_caption': 'Simulation: CARDAMOM mechanistic model',
        'pop_unavail': 'Population shift unavailable',
        'pert_unavail': 'Perturbation data not available',
        'loading_data': 'Loading data...',
        'loading_grn': 'Loading GRN...',
        # Misc UI
        'select_gene_option': '— Select a gene —',
        'genes_found': '**{n_overlap:,}** of your {n_total:,} genes found in the Explorer. Select any of them in the 🔍 search above to explore their gene program and GRN.',
        'loaded_success': '✅ Loaded: **{n_obs:,} cells × {n_vars:,} genes**',
        'cluster_details': 'Cluster annotation details',
        'how_it_works_body': """
**How it works:**
1. A query gene is embedded in a GNN co-expression space (trained on WGCNA graphs from scRNA-seq)
2. Nearest neighbors in embedding space define a **transcriptional program**
3. The retrieved gene list is exported and run through **CardamomOT** (ODE mechanistic model + optimal transport) on the scRNA-seq time-course data to infer a **gene regulatory network (GRN)** for that program
4. The same CardamomOT model is then used for in silico **perturbation simulations** (e.g. BIRC5 knockout) — revealing which genes change and enabling **network-based target identification**

> **For several key programs (Full, FOXM1, MKI67, TUBB) this has already been done** — the GRN and perturbation results are precomputed and available directly in this app. You can explore the candidate therapeutic targets without running CardamomOT yourself.

**Vector databases:**

| Database | Embeddings | Cells | Expression genes | Notes |
|---|---|---|---|---|
| **RMS original** | 8,442 | 13,968 | 8,442 | Primary RMS scRNA-seq · GCN embeddings |
| **RMS 2** | 8,836 | 4,706 | 8,836 | Second RMS cohort · GCN embeddings |
| **Trajectory (beta)** | 3,887 | 13,968 | 8,442 | Temporal trajectory GNN embeddings · cells & expression from RMS original |

**Therapeutic target logic (network perturbation approach):**
- **Direct targets** — genes overexpressed in the tumor that are essential nodes in the GRN (e.g. CEP55: drives cytokinesis, supra-expressed in RMS)
- **Co-targets** — genes that go *up* after BIRC5 KO, acting as compensatory escape mechanisms (e.g. PPP1R12B, MAP3K21); blocking them alongside BIRC5 leaves the cell no survival route
- This approach is called **network-informed synthetic lethality** — targets are chosen not in isolation but based on their role in the regulatory network under perturbation

**Upload your own data:**
Upload any `.h5ad` file for query of interest.

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

---

**Gene Trajectory Graph Embeddings**

Each gene receives a vector that encodes **how its co-expression neighbourhood changed over time**. Two sources of information are combined:

- **Temporal trajectory** — WGCNA co-expression graphs are built per time point, encoded by a shared GAT, and aligned with optimal transport.
- **Regulatory structure** — a PPGN (WL-3) runs on the OmniPath mechanistic interaction graph (accessed via NEKO) and captures regulatory motifs such as feedback loops and triangles. Added on top of the trajectory embedding for genes with known OmniPath interactions.
""",
    },
    'fr': {
        # Search
        'search_placeholder': '🔍 Sélectionnez un gène pour explorer son programme',
        # Main sections
        'settings': 'Paramètres',
        'genes_from_data': 'Gènes depuis vos données',
        'featured_targets': 'Cibles thérapeutiques',
        'about': "À propos du Gene Program Explorer",
        # Settings labels
        'vector_db': 'Base vectorielle',
        'program_size': 'Taille du programme',
        'grn_hops': 'Sauts GRN',
        'grn_model': 'Modèle GRN',
        # Data upload
        'upload_note': 'Traité en mémoire uniquement — aucune donnée stockée. Max ~50k cellules.',
        'upload_label': 'Importer un fichier .h5ad',
        'loading': 'Chargement...',
        'reading_file': 'Lecture du fichier...',
        'normalizing': 'Normalisation → ACP → UMAP (1–2 min pour ~10k cellules)…',
        'large_dataset': "Grand jeu de données — l'UMAP peut être lent ou manquer de mémoire.",
        'color_umap': "Colorier l'UMAP par expression génique",
        # Population labels
        'quiescent': 'quiescentes',
        'proliferative': 'prolifératives',
        'intermediate': 'intermédiaires',
        'real_data': 'Données réelles (populations *)',
        'sim_wt': 'Simulation WT (populations *)',
        'population': 'Population',
        'pop_shift': 'Changement de population après KO de {gene} (Δ%)',
        'delta_pct': 'Δ% (KO – WT)',
        # Result tabs
        'gene_program': 'Programme Génique',
        # Network tabs
        'ko_perturbation': 'Simulation KO',
        'network_graph': 'Réseau Génique',
        'adjacency_matrix': "Poids du Réseau",
        'network_topology': '📊 Analyse topologique du réseau',
        'upstream': 'Régulateurs en amont',
        'downstream': 'Cibles en aval',
        'feedback': 'Nœuds en boucle de rétroaction',
        'cycle_nodes': 'Nœuds dans un cycle',
        # Plot labels
        'umap1': 'UMAP 1',
        'umap2': 'UMAP 2',
        'expression': 'Expression',
        'time': 'Temps',
        'cell_type': 'Type cellulaire',
        'time_sim': 'Temps (simulation WT)',
        'knocked_out': 'invalidé',
        # About section (short)
        'about_intro': 'Système de récupération de programmes géniques basé sur **RAG**, appliqué à des données scRNA-seq de RMS (Rhabdomyosarcome).',
        'how_it_works': '**Comment ça fonctionne**',
        'show_results': '📊 Résultats pour',
        'gene_program_for': 'Programme génique pour',
        'cluster': 'cluster',
        'top_20': '20 gènes les plus affectés par',
        'top_hub': '🔵 **Nœud central**',
        'most_connected': 'nœud le plus connecté dans ce sous-réseau',
        'sim_caption': 'Simulation : modèle mécaniste CARDAMOM',
        'pop_unavail': 'Changement de population non disponible',
        'pert_unavail': 'Données de perturbation non disponibles',
        'loading_data': 'Chargement des données...',
        'loading_grn': 'Chargement du GRN...',
        # Misc UI
        'select_gene_option': '— Sélectionner un gène —',
        'genes_found': '**{n_overlap:,}** de vos {n_total:,} gènes trouvés dans l\'Explorateur. Sélectionnez-en un dans la 🔍 recherche ci-dessus pour explorer son programme génique et son GRN.',
        'loaded_success': '✅ Chargé : **{n_obs:,} cellules × {n_vars:,} gènes**',
        'cluster_details': 'Détails d\'annotation du cluster',
        'how_it_works_body': """
**Comment ça fonctionne :**
1. Un gène requête est représenté dans un espace GNN de co-expression (entraîné sur des graphes WGCNA issus de scRNA-seq)
2. Les voisins les plus proches dans cet espace définissent un **programme transcriptionnel**
3. La liste de gènes récupérée est analysée par **CardamomOT** (modèle ODE mécaniste + transport optimal) sur des données scRNA-seq temporelles pour inférer un **réseau de régulation génique (GRN)**
4. Le même modèle CardamomOT est utilisé pour des **simulations de perturbation in silico** (ex : knockout de BIRC5), révélant quels gènes changent et permettant l'**identification de cibles par le réseau**

> **Pour plusieurs programmes clés (Full, FOXM1, MKI67, TUBB) cela a déjà été réalisé** — les résultats GRN et perturbation sont précalculés et directement disponibles dans cette application. Vous pouvez explorer les cibles thérapeutiques candidates sans exécuter CardamomOT vous-même.

**Bases de données vectorielles :**

| Base de données | Embeddings | Cellules | Gènes d'expression | Notes |
|---|---|---|---|---|
| **RMS original** | 8 442 | 13 968 | 8 442 | scRNA-seq RMS primaire · embeddings GCN |
| **RMS 2** | 8 836 | 4 706 | 8 836 | Deuxième cohorte RMS · embeddings GCN |
| **Trajectoire (bêta)** | 3 887 | 13 968 | 8 442 | Embeddings GNN de trajectoire temporelle |

**Logique de cible thérapeutique (approche par perturbation de réseau) :**
- **Cibles directes** — gènes surexprimés dans la tumeur qui sont des nœuds essentiels du GRN (ex : CEP55 : conduit la cytocinèse, supra-exprimé dans le RMS)
- **Co-cibles** — gènes qui *augmentent* après KO de BIRC5, agissant comme mécanismes de compensation (ex : PPP1R12B, MAP3K21) ; les bloquer simultanément ne laisse aucune voie de survie à la cellule
- Cette approche est appelée **létalité synthétique guidée par le réseau** — les cibles sont choisies non isolément mais selon leur rôle dans le réseau régulateur sous perturbation

**Importer vos propres données :**
Importez n'importe quel fichier `.h5ad` pour votre gène d'intérêt.

**Modèles GRN disponibles :**
- **Original** — 159 gènes, inféré sur les données complètes scRNA-seq RMS
- **Programme complet** — 200 gènes (ensemble quiescent + prolifératif), perturbation KO HSPA1B simulée via CARDAMOM
- **Programme FOXM1** — 198 gènes (top-200 voisins GNN de FOXM1), perturbation KO FOXM1 simulée via CARDAMOM
- **Programme MKI67** — 201 gènes (top-200 voisins GNN de MKI67), perturbation KO BIRC5 simulée via CARDAMOM
- **Programme TUBB** — 201 gènes (top-200 voisins GNN de TUBB), perturbation KO TUBB simulée via CARDAMOM

**Dynamique de population sous knockout (programme complet) :**
Lors de l'interrogation d'un gène du **programme complet** (ex : HSPA1B), l'application exécute une simulation CARDAMOM du knockout de HSPA1B et suit l'évolution de la composition de la population cellulaire.
Les tumeurs RMS contiennent trois états cellulaires coexistants — **Prolifératif**, **Quiescent** et **Intermédiaire** — en équilibre dynamique.
Chaque état est scoré depuis les données scRNA-seq : **z-score DNAJB1** (top voisin HSPA1B, sim=0.91) pour la quiescence ; **moyenne des top-100 voisins MKI67** pour la prolifération.
Après KO, CARDAMOM propage la perturbation à travers le GRN et resimule les trajectoires cellulaires ; le décalage résultant (Δ%) montre si le knockout pousse les cellules vers ou hors de la prolifération.

**Annotation des programmes géniques (LLM) :**
Chaque programme génique est automatiquement annoté par **Llama 3.1-8B** (via Nebius AI Studio) — le modèle reçoit les gènes les plus co-exprimés et génère un label biologique concis (ex : *"Prolifération cellulaire mitotique"*, *"Remodelage du cytosquelette"*).

**Scoring de population cellulaire (DNAJB1 / MKI67-100) :**
Trois états cellulaires RMS coexistants sont définis par des seuils d'expression appliqués uniformément aux données réelles et à toutes les simulations :
- **Prolifératif** (rouge) — expression moyenne des top-100 voisins MKI67 >= 70e percentile
- **Quiescent** (bleu) — z-score DNAJB1 >= 70e percentile
- **Intermédiaire** (gris) — toutes les cellules restantes

---

**Embeddings de graphes de trajectoire génique**

Chaque gène reçoit un vecteur encodant **comment son voisinage de co-expression a évolué dans le temps**. Deux sources d'information sont combinées :

- **Trajectoire temporelle** — des graphes de co-expression WGCNA sont construits par point temporel, encodés par un GAT partagé et alignés par transport optimal.
- **Structure régulatrice** — un PPGN (WL-3) analyse le graphe d'interaction mécaniste OmniPath (via NEKO) et capture les motifs régulateurs tels que boucles de rétroaction et triangles.
""",
    }
}

st.set_page_config(
    page_title="Gene Program Explorer",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Reset app state when ?reset=1 is in URL ──────────────────────
if st.query_params.get("reset"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.query_params.clear()
    st.rerun()

# ── Language detection (early — needed for CSS placeholder) ──────
if 'lang' in st.query_params:
    st.session_state['_lang'] = st.query_params['lang']
_cur_lang = st.session_state.get('_lang', 'fr')
T = _TRANSLATIONS[_cur_lang]

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
}
section.main > div { max-width: 960px; margin: auto; }

/* ── Fake placeholder for main search selectbox only ────────────
   _PLACEHOLDER="" so field is truly empty on click.
   ::before shows hint text when not focused; hides on click.
   Excluded for selectboxes inside expanders/columns. */
div[data-testid="stSelectbox"] > div {
    position: relative;
}
div[data-testid="stSelectbox"] > div::before {
    content: "🔍 Select a gene to explore its program";
    position: absolute;
    top: 50%; left: 16px;
    transform: translateY(-50%);
    color: #9aa0a6;
    font-size: 1rem;
    pointer-events: none;
    white-space: nowrap;
    z-index: 5;
}
div[data-testid="stSelectbox"]:focus-within > div::before {
    display: none !important;
}
/* Hide ::before inside expanders (Settings etc) */
div[data-testid="stExpander"] div[data-testid="stSelectbox"] > div::before {
    content: none !important;
}

/* ── Google-style search selectbox ────────────────────────────── */
div[data-testid="stSelectbox"] > label {
    font-size: 0.78rem !important;
    color: #666 !important;
    margin-bottom: 2px !important;
}
/* Search bar — pill style */
div[data-testid="stSelectbox"] > div > div {
    border-radius: 24px !important;
    border: 1.5px solid #dfe1e5 !important;
    box-shadow: 0 1px 6px rgba(32,33,36,0.1) !important;
    font-size: 1rem !important;
    transition: box-shadow 0.2s;
    min-height: 46px !important;
}
div[data-testid="stSelectbox"] > div > div:hover,
div[data-testid="stSelectbox"] > div > div:focus-within {
    box-shadow: 0 2px 12px rgba(32,33,36,0.2) !important;
    border-color: #bdc1c6 !important;
}
/* Vector database selector — borderless (inside a column) */
div[data-testid="column"] div[data-testid="stSelectbox"] > div > div,
div[data-testid="column"] div[data-testid="stSelectbox"] > div > div * {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    min-height: unset !important;
    border-radius: 6px !important;
}
div[data-testid="column"] div[data-testid="stSelectbox"] > div > div:hover {
    background: #f5f5f5 !important;
}

/* ── Google-style spacing: search is the hero element ──────────── */
div[data-testid="stVerticalBlock"] > div:has(div[data-testid="stSelectbox"]:first-child) {
    margin-top: 8px !important;
}


/* Compact sliders */
div[data-testid="stSlider"] label { font-size: 0.78rem !important; color: #666 !important; }
div[data-testid="stSlider"] { padding-top: 4px !important; }

/* Compact radio (dataset chooser) */
div[data-testid="stRadio"] label { font-size: 0.82rem !important; }
div[data-testid="stRadio"] > div { gap: 6px !important; }

/* Clear history — small subtle button */
div[data-testid="stButton"] > button {
    font-size: 0.78rem !important;
    padding: 2px 10px !important;
    border-radius: 12px !important;
    color: #888 !important;
    border-color: #ddd !important;
    background: transparent !important;
}
div[data-testid="stButton"] > button:hover {
    color: #333 !important;
    border-color: #aaa !important;
    background: #f5f5f5 !important;
}

/* Reduce gaps between sections */
div[data-testid="stCaptionContainer"] { margin-top: -4px !important; }
div.block-container { padding-top: 0.5rem !important; }

/* Borderless expanders */
div[data-testid="stExpander"],
div[data-testid="stExpander"] details,
div[data-testid="stExpander"] details summary {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    margin-bottom: 2px !important;
}
details[data-testid],
details {
    border: none !important;
    box-shadow: none !important;
}
details > summary:hover {
    color: #000 !important;
    background: transparent !important;
}

/* Disable Streamlit header title link — clicking it would reload the page */
header[data-testid="stHeader"] a,
header[data-testid="stHeader"] [data-testid="stAppViewBlockContainer"] {
    pointer-events: none !important;
    cursor: default !important;
}
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

# ── Dynamic CSS: translated placeholder for search bar ───────────
st.markdown(f"""
<style>
div[data-testid="stSelectbox"] > div::before {{
    content: "{T['search_placeholder']}";
}}
</style>
""", unsafe_allow_html=True)

# ── Sticky search bar via components.html (runs real JS in parent frame) ──
_components.html("""
<script>
(function() {
    function setup() {
        try {
            var doc = window.parent.document;
            var scrollEl = doc.querySelector('[data-testid="stMain"]');
            if (!scrollEl) { setTimeout(setup, 400); return; }

            // Find first stSelectbox not in expander/column
            var searchBlock = null;
            var boxes = doc.querySelectorAll('[data-testid="stSelectbox"]');
            for (var i = 0; i < boxes.length; i++) {
                var box = boxes[i];
                if (box.closest('[data-testid="stExpander"]') ||
                    box.closest('[data-testid="column"]') ||
                    box.closest('[data-testid="stColumn"]')) continue;
                var el = box;
                while (el && el.getAttribute && el.getAttribute('data-testid') !== 'stVerticalBlock') {
                    el = el.parentElement;
                }
                if (el) { searchBlock = el; break; }
            }
            if (!searchBlock) { setTimeout(setup, 400); return; }

            // Height of Streamlit's own fixed header toolbar
            var headerEl = doc.querySelector('[data-testid="stHeader"]');
            var headerH = headerEl ? headerEl.offsetHeight : 0;

            // Spacer to fill gap when element goes fixed
            var spacer = doc.createElement('div');
            spacer.style.display = 'none';
            searchBlock.parentElement.insertBefore(spacer, searchBlock);

            // Original distance from top of scroll container
            var originalTop = searchBlock.offsetTop;

            scrollEl.addEventListener('scroll', function() {
                if (scrollEl.scrollTop > originalTop - headerH - 10) {
                    var ref = (searchBlock.closest('.block-container') || scrollEl).getBoundingClientRect();
                    spacer.style.height  = searchBlock.offsetHeight + 'px';
                    spacer.style.display = 'block';
                    searchBlock.style.position   = 'fixed';
                    searchBlock.style.top        = headerH + 'px';
                    searchBlock.style.left       = ref.left + 'px';
                    searchBlock.style.width      = ref.width + 'px';
                    searchBlock.style.zIndex     = '999980';
                    searchBlock.style.background = 'transparent';
                    searchBlock.style.padding    = '6px 0 8px';
                    searchBlock.style.boxShadow  = 'none';
                } else {
                    spacer.style.display = 'none';
                    searchBlock.style.cssText = '';
                }
            }, { passive: true });

        } catch(e) { console.error('[sticky]', e); }
    }
    setTimeout(setup, 800);
    setTimeout(setup, 2000);
})();
</script>
""", height=1)

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


@st.cache_data(show_spinner=False)
def load_data_traj():
    """Load trajectory GNN embeddings. Falls back gracefully if file missing."""
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

        # Load v1 cell-level data for Expression / Time / Cell Type UMAPs
        v1_files = [
            "umap_coords.csv",
            "gene_names.csv",
            "expr_matrix_f16.npy",
        ]
        v1_paths = {}
        for f in v1_files:
            local_path = os.path.join(LOCAL_DIR, f)
            if os.path.exists(local_path):
                v1_paths[f] = local_path
            else:
                token = st.secrets.get("HF_TOKEN", None)
                v1_paths[f] = hf_hub_download(repo_id=REPO_ID, filename=f,
                                               repo_type="dataset", token=token)
        umap_df    = pd.read_csv(v1_paths["umap_coords.csv"], index_col=0)
        gene_names = pd.read_csv(v1_paths["gene_names.csv"])["0"].tolist()
        expr       = np.load(v1_paths["expr_matrix_f16.npy"])

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
    elif grn_key in ("full", "full_foxm1"):
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
    elif grn_key in ("full", "full_foxm1"):
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
def load_perturbation(grn_key="mki67", version="v20260530"):
    """Load KO perturbation data for the given GRN model."""
    import os
    if grn_key == "tubb":
        f = "tubb_ko_perturbation.csv"
    elif grn_key == "foxm1":
        f = "foxm1_ko_perturbation.csv"
    elif grn_key == "full":
        f = "full_ko_perturbation.csv"
    elif grn_key == "full_foxm1":
        f = "full_foxm1_ko_perturbation.csv"
    elif grn_key == "full_aurkb":
        f = "full_aurkb_ko_perturbation.csv"
    else:
        f = "birc5_ko_perturbation.csv"
    local = os.path.join(LOCAL_DIR, f)
    if os.path.exists(local):
        return pd.read_csv(local)
    token = st.secrets.get("HF_TOKEN", None)
    path = hf_hub_download(repo_id=REPO_ID, filename=f, repo_type="dataset", token=token,
                           force_download=True)
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
    """Full program (200 genes) sim cells scored by population (WT + HSPA1B KO).
    Columns: x_wt, y_wt, x_ko, y_ko, pop_wt, pop_ko, time."""
    import os
    f = "full_pop_sim_scored.csv"
    local = os.path.join(LOCAL_DIR, f)
    if os.path.exists(local): return pd.read_csv(local)
    token = st.secrets.get("HF_TOKEN", None)
    path = hf_hub_download(repo_id=REPO_ID, filename=f, repo_type="dataset", token=token)
    return pd.read_csv(path)


@st.cache_resource
def load_full_foxm1_pop_sim():
    """Full program (200 genes) sim cells scored by population (WT + FOXM1 KO).
    Columns: x_wt, y_wt, x_ko, y_ko, pop_wt, pop_ko, time."""
    import os
    f = "full_foxm1_pop_sim_scored.csv"
    local = os.path.join(LOCAL_DIR, f)
    if os.path.exists(local): return pd.read_csv(local)
    token = st.secrets.get("HF_TOKEN", None)
    path = hf_hub_download(repo_id=REPO_ID, filename=f, repo_type="dataset", token=token)
    return pd.read_csv(path)


@st.cache_resource
def load_full_aurkb_pop_sim():
    """Full program (200 genes) sim cells scored by population (WT + AURKB KO).
    Columns: x_wt, y_wt, x_ko, y_ko, pop_wt, pop_ko, time."""
    import os
    f = "full_aurkb_pop_sim_scored.csv"
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


@st.cache_resource
def load_gene_umap2d(dataset="v1", _version="v2"):
    """Load 2D UMAP projection of gene embeddings (word2vec-style map)."""
    import os
    f = "gcn_gene_umap2d.csv" if dataset == "v1" else "trajectory_gene_umap2d.csv"
    local = os.path.join(LOCAL_DIR, f)
    if os.path.exists(local):
        return pd.read_csv(local)
    token = st.secrets.get("HF_TOKEN", None)
    path = hf_hub_download(repo_id=REPO_ID, filename=f, repo_type="dataset",
                           token=token, force_download=True)
    return pd.read_csv(path)


def build_gene_embedding_map(gene_umap_df, query_gene, program_genes, annotations,
                             embeddings_arr=None, genes_list=None, sim_scores=None):
    """Beautiful PCA neighborhood map — gradient by similarity, connection lines."""
    if embeddings_arr is None or genes_list is None:
        return None
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize as _normalize

    valid = [g for g in program_genes if g in genes_list]
    if len(valid) < 3:
        return None

    idxs   = [genes_list.index(g) for g in valid]
    X      = _normalize(embeddings_arr[idxs], norm="l2")
    coords = PCA(n_components=2, random_state=42).fit_transform(X)

    lut = gene_umap_df.set_index("gene")["cluster"].to_dict()
    rows = []
    for i, g in enumerate(valid):
        sim = sim_scores.get(g, 0.95) if sim_scores else 0.95
        rows.append({"gene": g, "x": float(coords[i,0]), "y": float(coords[i,1]),
                     "sim": sim, "is_query": g == query_gene})
    df = pd.DataFrame(rows)

    qrow = df[df["is_query"]].iloc[0]
    prog = df[~df["is_query"]].sort_values("sim", ascending=False)

    # Similarity → dot size + color intensity
    s_min, s_max = prog["sim"].min(), prog["sim"].max()
    def _norm(s):
        return (s - s_min) / (s_max - s_min + 1e-9)

    fig = go.Figure()

    # Thin connection lines from query to each neighbor
    for _, row in prog.iterrows():
        alpha = 0.15 + 0.35 * _norm(row["sim"])
        fig.add_trace(go.Scatter(
            x=[qrow["x"], row["x"]], y=[qrow["y"], row["y"]],
            mode="lines",
            line=dict(color=f"rgba(180,180,180,{alpha:.2f})", width=1),
            hoverinfo="skip", showlegend=False
        ))

    # Neighbor dots — color gradient deep blue → light blue by similarity
    colors = [f"rgba({int(60+_norm(s)*160)},{int(100+_norm(s)*80)},{int(200-_norm(s)*80)},0.9)"
              for s in prog["sim"]]
    sizes  = [8 + 6 * _norm(s) for s in prog["sim"]]

    fig.add_trace(go.Scatter(
        x=prog["x"], y=prog["y"], mode="markers+text",
        marker=dict(size=sizes, color=colors, line=dict(width=1, color="white")),
        text=prog["gene"], textposition="top center",
        textfont=dict(size=8, color="#555"),
        hovertext=prog["gene"] + " · " + prog["sim"].round(4).astype(str),
        hoverinfo="text", showlegend=False
    ))

    # Query gene — 3D ball effect: large circle + small white highlight
    fig.add_trace(go.Scatter(
        x=[qrow["x"]], y=[qrow["y"]], mode="markers+text",
        marker=dict(size=28, color="#1a6faf", symbol="circle",
                    line=dict(width=3, color="#0d4a7a"),
                    gradient=dict(type="radial", color="rgba(130,190,255,0.9)")),
        text=[query_gene], textposition="top center",
        textfont=dict(size=12, color="#1a6faf", family="Arial Black"),
        hoverinfo="text", hovertext=[query_gene],
        showlegend=False
    ))
    # White highlight dot for 3D effect
    fig.add_trace(go.Scatter(
        x=[qrow["x"] - 0.015 * (coords[:,0].max() - coords[:,0].min())],
        y=[qrow["y"] + 0.02  * (coords[:,1].max() - coords[:,1].min())],
        mode="markers",
        marker=dict(size=8, color="rgba(255,255,255,0.7)", symbol="circle"),
        hoverinfo="skip", showlegend=False
    ))

    fig.update_layout(
        title=dict(text=f"{query_gene} · nearest neighbors in embedding space",
                   font=dict(size=12, color="#444")),
        height=440,
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(visible=False, zeroline=False),
        yaxis=dict(visible=False, zeroline=False),
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="closest",
        showlegend=False
    )
    return fig


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
        title=dict(text=f"Cell population proportions: WT vs {ko_label} simulation *",
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
        title=dict(text=f"Population shift after {ko_label} (Δ%) *",
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
            "co_targets":     {"PBX3", "FAM83G", "HSPA1A", "AC148476.1", "PABPN1", "HBA2",
                               "DNAJB1", "MAPK11", "KLF2", "HSPA6", "FBXO5", "OTUD1", "FAM122B"},
            "direct_targets": {"RACGAP1", "NUSAP1", "FOXM1", "TRIM59", "SPC24", "NEK2", "GTSE1", "EIF2A"},  # DOWN
            "subtitle": (
                "🟠 co-target: goes UP after KO — derepressed or compensatory response &nbsp;|&nbsp;"
                " direct target: suppressed by HSPA1B loss — mitotic/cytokinesis genes (FOXM1, NUSAP1)"
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

    # Keep all direct GRN neighbours (1-hop ego network).
    # The program_set filter was removed because GNN program genes and GRN genes
    # are independent sets and rarely overlap — filtering by program_set produced
    # an empty graph for almost every gene.
    # Limit to top_n neighbours by edge weight to avoid overloaded graphs.
    direct_neighbors = set(G_full.successors(query_gene)) | set(G_full.predecessors(query_gene))
    if len(direct_neighbors) > top_n:
        # Keep the top_n edges by absolute weight
        all_edges = (
            [(query_gene, n, abs(G_full[query_gene][n]['weight'])) for n in G_full.successors(query_gene)]
            + [(n, query_gene, abs(G_full[n][query_gene]['weight'])) for n in G_full.predecessors(query_gene)]
        )
        all_edges.sort(key=lambda x: x[2], reverse=True)
        top_neighbors = set()
        for src, dst, _ in all_edges[:top_n]:
            top_neighbors.add(src); top_neighbors.add(dst)
        direct_neighbors = top_neighbors - {query_gene}
    nodes_to_keep = {query_gene} | direct_neighbors

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
            f"In: {row['In-degree']}  Out: {row['Out-degree']}"
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
        title=(f"GRN for {query_gene} — "
               f"({'top hubs in program' if len(nodes_to_keep) > 1 and G.degree(query_gene) == 0 else f'{hops}-hop ego network'})  "
               f"(green=activation, red=repression | size=degree | ◆=feedback loop)"),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=580,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    return fig, topo_df


def build_grn_adjacency(grn_mat, grn_genes, gene_set, query_gene=None, hops=1, top_n=20):
    if grn_mat is None or query_gene is None or query_gene not in grn_genes:
        return None
    # Build the same direct-neighbour node set as build_grn_figure uses.
    # gene_set (GNN program) and grn_genes (CardamomOT) rarely overlap, so we
    # ignore gene_set here and use direct GRN neighbours of query_gene instead.
    q_idx = grn_genes.index(query_gene)
    direct = set()
    for j, w in enumerate(grn_mat[q_idx]):
        if abs(w) > 0 and grn_genes[j] != query_gene:
            direct.add(grn_genes[j])
    for i, w in enumerate(grn_mat[:, q_idx]):
        if abs(w) > 0 and grn_genes[i] != query_gene:
            direct.add(grn_genes[i])
    # Limit to top_n by max(|outgoing|, |incoming|) weight
    if len(direct) > top_n:
        scored = []
        for g in direct:
            gi = grn_genes.index(g)
            w_out = abs(float(grn_mat[q_idx, gi]))
            w_in  = abs(float(grn_mat[gi, q_idx]))
            scored.append((g, max(w_out, w_in)))
        scored.sort(key=lambda x: x[1], reverse=True)
        direct = {g for g, _ in scored[:top_n]}
    all_nodes = [query_gene] + sorted(direct)
    node_set  = set(all_nodes)
    adj = pd.DataFrame(0.0, index=all_nodes, columns=all_nodes)
    for gene in all_nodes:
        gi = grn_genes.index(gene)
        for j, w in enumerate(grn_mat[gi]):
            if abs(w) > 0 and grn_genes[j] in node_set and grn_genes[j] != gene:
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

# ── Google-homepage style: centered title above, search below ─────

# Language toggle — HTML flag links (query param based, lang already detected above)
_en_op = '1.0' if _cur_lang == 'en' else '0.35'
_fr_op = '1.0' if _cur_lang == 'fr' else '0.35'
st.markdown(f"""
<div style="position:fixed; top:62px; right:20px; z-index:9999; display:flex; gap:8px; align-items:center;">
  <a href="?lang=en" style="text-decoration:none; opacity:{_en_op}; transition:opacity .2s;">
    <div style="position:relative; width:30px; height:30px; border-radius:50%;
                box-shadow: 0 4px 12px rgba(0,0,0,0.35), 0 1px 3px rgba(0,0,0,0.2), inset 0 1px 2px rgba(255,255,255,0.5);">
      <img src="https://flagcdn.com/w40/gb.png"
           style="width:30px; height:30px; border-radius:50%; object-fit:cover; display:block;">
      <div style="position:absolute; top:0; left:0; right:0; bottom:0; border-radius:50%;
                  background: radial-gradient(circle at 38% 28%, rgba(255,255,255,0.55) 0%, rgba(255,255,255,0.0) 60%);
                  pointer-events:none;"></div>
    </div>
  </a>
  <a href="?lang=fr" style="text-decoration:none; opacity:{_fr_op}; transition:opacity .2s;">
    <div style="position:relative; width:30px; height:30px; border-radius:50%;
                box-shadow: 0 4px 12px rgba(0,0,0,0.35), 0 1px 3px rgba(0,0,0,0.2), inset 0 1px 2px rgba(255,255,255,0.5);">
      <img src="https://flagcdn.com/w40/fr.png"
           style="width:30px; height:30px; border-radius:50%; object-fit:cover; display:block;">
      <div style="position:absolute; top:0; left:0; right:0; bottom:0; border-radius:50%;
                  background: radial-gradient(circle at 38% 28%, rgba(255,255,255,0.55) 0%, rgba(255,255,255,0.0) 60%);
                  pointer-events:none;"></div>
    </div>
  </a>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# Logo + title — centered, vertical stack
st.markdown(f"""
<div style='text-align:center; margin-bottom:18px;'>
  <a href='https://rmsexplorer.com' target='_self' style='text-decoration:none; cursor:pointer;'>
    <span style='font-size:clamp(1.8rem,5vw,2.6rem);font-weight:800;letter-spacing:-0.5px;color:#002395;'>Gene</span><span
          style='font-size:clamp(1.8rem,5vw,2.6rem);font-weight:800;letter-spacing:-0.5px;color:#444;'>&nbsp;Program&nbsp;</span><span
          style='font-size:clamp(1.8rem,5vw,2.6rem);font-weight:800;letter-spacing:-0.5px;color:#ED2939;'>Explorer</span>
  </a>
</div>
""", unsafe_allow_html=True)

# Search bar placeholder — filled after data loads (sits directly below title)
_search_container = st.container()


# ================================================================
# DATASET SELECTOR  — read from session_state so widget can live
# inside _search_container (rendered visually under the banner).
# ================================================================
_ds_options = [
    "8,442 genes · 13,968 cells · GCN",
    "8,836 genes · 4,706 cells · GCN",
    "3,887 embeddings · 13,968 cells · GNN+OT temporal (beta)",
]
dataset_choice = st.session_state.get("dataset_select", _ds_options[0])
if dataset_choice not in _ds_options:
    dataset_choice = _ds_options[0]
dataset_key = "v1" if dataset_choice == _ds_options[0] else ("traj" if dataset_choice == _ds_options[2] else "v2")

with st.spinner(T['loading_data']):
    if dataset_key == "traj":
        genes, embeddings, clusters, annotations, summaries, umap_df, expr, gene_names, grn_mat, grn_genes = load_data_traj()
    else:
        genes, embeddings, clusters, annotations, summaries, umap_df, expr, gene_names, grn_mat, grn_genes = load_data(dataset_key)

# ================================================================
# UPLOAD YOUR OWN DATA — rendered AFTER results (placeholder processed here)
# ================================================================
# The actual expander is rendered below the query results.
# Pre-process the upload if a file is already in session state.
uploaded_file = None  # will be set when the expander is rendered later


# ================================================================
# STATS (compact inline)
# ================================================================
n_genes = len(genes)
n_clusters = len(set(clusters))
n_cells = umap_df.shape[0]

# ── GRN selector — hide only if gene is in NO model at all ──────
_orig_gene_set   = load_grn_gene_list("original")
_mki67_gene_set  = load_grn_gene_list("mki67")
_tubb_gene_set   = load_grn_gene_list("tubb")
_foxm1_gene_set  = load_grn_gene_list("foxm1")
_full_gene_set   = load_grn_gene_list("full")

# model label → (key, gene_set)
_ALL_MODELS = {
    "Full program (200 genes)":               ("full",     _full_gene_set),
    "FOXM1 program (198 genes, FOXM1 KO)":    ("foxm1",    _foxm1_gene_set),
    "MKI67 program (201 genes, BIRC5 KO)":    ("mki67",    _mki67_gene_set),
    "TUBB program (201 genes, TUBB KO)":       ("tubb",     _tubb_gene_set),
    "Original (159 genes)":                    ("original", _orig_gene_set),
}

# use last queried gene (or most recent search) to decide visibility
_last_q      = st.session_state.get(f"last_selected_{dataset_key}", "")
_check_gene  = _last_q.strip().upper()

_gene_in_any_grn = bool(_check_gene) and any(
    _check_gene in gs for _, gs in _ALL_MODELS.values()
)

# ── Load GRN data (no UI rendering yet) ──────────────────────────
_grn_state_key = f"grn_choice_{dataset_key}"
grn_hops = st.session_state.get(f"grn_slider_{dataset_key}", 1)
if not _gene_in_any_grn:
    grn_mat, grn_genes = None, []
    grn_options = []
else:
    if _check_gene:
        grn_options = [label for label, (_, gs) in _ALL_MODELS.items()
                       if _check_gene in gs]
    else:
        grn_options = list(_ALL_MODELS.keys())
    if st.session_state.get(_grn_state_key, "") not in grn_options:
        st.session_state[_grn_state_key] = grn_options[0]
    grn_choice = st.session_state[_grn_state_key]
    grn_key = _ALL_MODELS[grn_choice][0]
    with st.spinner(T['loading_grn']):
        grn_mat, grn_genes = load_grn(grn_key)

# 🔬 icon = gene present in ANY GRN model
grn_gene_set = _mki67_gene_set | _orig_gene_set | _tubb_gene_set | _foxm1_gene_set | _full_gene_set

def gene_label(g):
    suffix = " 🔬" if g in grn_gene_set else ""
    return f"{g}{suffix}"

_PLACEHOLDER    = ""
_RECENT_SEP     = "── Recent ──────────────────────"

# ── Build dropdown options with recent searches at top ────────────
_recent_genes = [g for g in _load_recent(dataset_key) if g in genes]

def _recent_label(g):
    """Label for a recently-searched gene: clock icon + optional 🔬."""
    suffix = " 🔬" if g in grn_gene_set else ""
    return f"🕐 {g}{suffix}"

_recent_options = [_recent_label(g) for g in _recent_genes]

if _recent_options:
    gene_labels_all = (
        [_PLACEHOLDER]
        + _recent_options
        + [gene_label(g) for g in sorted(genes)]
    )
else:
    gene_labels_all = [_PLACEHOLDER] + [gene_label(g) for g in sorted(genes)]

# ── Fill: search bar → right column of header ────────────────────
_sel_ver_key = f"sel_version_{dataset_key}"
if _sel_ver_key not in st.session_state:
    st.session_state[_sel_ver_key] = 0

_last_q_gene  = st.session_state.get(f"last_selected_{dataset_key}", "")
# Always start at placeholder so the field is empty when user clicks —
# recent searches at the top of the list serve as quick-access history
_default_idx = 0

selected_label = _search_container.selectbox(
    "🔍 Search gene",
    options=gene_labels_all,
    index=_default_idx,
    label_visibility="collapsed",
    key=f"selectbox_{dataset_key}_v{st.session_state[_sel_ver_key]}"
)
# Separator is not a valid gene — treat as placeholder
_is_placeholder = (
    (not selected_label)
    or selected_label == _PLACEHOLDER
    or selected_label == _RECENT_SEP
)
# Strip both 🕐 prefix (recent) and 🔬 suffix to get raw gene name
def _strip_label(label: str) -> str:
    if label.startswith("🕐 "):
        label = label[len("🕐 "):]
    return label.replace(" 🔬", "").strip()

selected_gene = _strip_label(selected_label) if not _is_placeholder else ""


# ── Read control values from session state (widgets rendered after results) ──
program_size = st.session_state.get(f"slider_{dataset_key}", 20)

load_real_expr_means()

if f"messages_{dataset_key}" not in st.session_state:
    st.session_state[f"messages_{dataset_key}"] = []

if f"last_selected_{dataset_key}" not in st.session_state:
    st.session_state[f"last_selected_{dataset_key}"] = ""

if f"default_run_{dataset_key}" not in st.session_state:
    st.session_state[f"default_run_{dataset_key}"] = False

messages = st.session_state[f"messages_{dataset_key}"]

def _render_msg_figures(msg, msg_id):
    """Render all figures for a single message using tabs."""
    # ── Determine what content is available ──────────────────────────────────
    _has_gene_prog = "df" in msg
    _has_expression = msg.get("fig") is not None or any(
        msg.get(k) is not None for k in ["fig_time", "fig_sim_time", "fig_celltype"]
    )
    _msg_grn_model = msg.get("grn_model")
    _has_grn = msg.get("grn_fig") is not None
    _has_adj = _has_grn and msg.get("grn_adj") is not None
    _has_pert = _msg_grn_model in ("mki67", "tubb", "foxm1", "full")
    _ko_gene_label = {"mki67": "BIRC5", "tubb": "TUBB", "foxm1": "FOXM1", "full": "HSPA1B", "full_foxm1": "FOXM1"}.get(_msg_grn_model, "")

    # ── Build tab list dynamically ────────────────────────────────────────────
    _tab_defs = []  # list of (key, title) for tabs to show
    if _has_gene_prog:
        _tab_defs.append(("gene_prog", T['gene_program']))
    if _has_expression:
        _tab_defs.append(("expression", T['expression']))
    if _has_grn:
        _tab_defs.append(("network", T['network_graph']))
    if _has_adj:
        _tab_defs.append(("adjacency", T['adjacency_matrix']))
    if _has_pert:
        _pert_tab_title = T['ko_perturbation'] if _msg_grn_model == "full" else f"{_ko_gene_label} {T['ko_perturbation']}"
        _tab_defs.append(("perturbation", _pert_tab_title))

    if not _tab_defs:
        return

    _tab_keys   = [d[0] for d in _tab_defs]
    _tab_titles = [d[1] for d in _tab_defs]
    _tabs = st.tabs(_tab_titles)
    _tab_map = dict(zip(_tab_keys, _tabs))

    # ── Tab: Gene Program ─────────────────────────────────────────────────────
    if "gene_prog" in _tab_map:
        with _tab_map["gene_prog"]:
            if msg.get("fig_gene_map") is not None:
                _mc, _tc = st.columns([3, 2])
                with _mc:
                    st.plotly_chart(msg["fig_gene_map"], use_container_width=True,
                                    key=f"{msg_id}_gene_map")
                with _tc:
                    st.dataframe(msg["df"], use_container_width=True, height=440)
                    if "cluster_id" in msg:
                        summary = summaries.get(msg["cluster_id"], "")
                        if summary:
                            with st.popover(T['cluster_details']):
                                st.markdown(summary)
            else:
                st.dataframe(msg["df"], use_container_width=True)
                if "cluster_id" in msg:
                    summary = summaries.get(msg["cluster_id"], "")
                    if summary:
                        with st.popover(T['cluster_details']):
                            st.markdown(summary)

    # ── Tab: Expression ───────────────────────────────────────────────────────
    if "expression" in _tab_map:
        with _tab_map["expression"]:
            if msg.get("fig") is not None:
                fig_expr = msg["fig"]
                fig_expr.update_layout(height=CHART_H)
                st.plotly_chart(fig_expr, use_container_width=True, key=f"{msg_id}_fig")
            _row1_keys = ["fig_time", "fig_sim_time", "fig_celltype"]
            _row1 = [(k, msg.get(k)) for k in _row1_keys if msg.get(k) is not None]
            if _row1:
                cols = st.columns(1 if is_mobile else len(_row1))
                for col, (k, f) in zip(cols, _row1):
                    f.update_layout(height=CHART_H_SMALL)
                    col.plotly_chart(f, use_container_width=True, key=f"{msg_id}_{k}")

    # ── Tab: KO Perturbation ──────────────────────────────────────────────────
    if "perturbation" in _tab_map:
        with _tab_map["perturbation"]:
            try:
                # KO selector for Full program (same GRN, three KO choices)
                if _msg_grn_model == "full":
                    _full_ko_key = f"{msg_id}_full_ko_choice"
                    _full_ko_choice = st.radio(
                        "KO gene", ["HSPA1B", "FOXM1", "AURKB"],
                        horizontal=True, key=_full_ko_key
                    )
                    _effective_grn_model = {
                        "FOXM1": "full_foxm1",
                        "AURKB": "full_aurkb",
                    }.get(_full_ko_choice, "full")
                    _ko_gene_label = _full_ko_choice
                else:
                    _effective_grn_model = _msg_grn_model
                pert_df = load_perturbation(_effective_grn_model)
                q_gene  = msg.get("query_gene", "MKI67")
                _real_means = load_real_expr_means()
                bar_fig, line_fig = build_perturbation_figures(
                    pert_df, q_gene, ko_gene=_ko_gene_label,
                    real_expr_means=_real_means)
                st.plotly_chart(bar_fig, use_container_width=True, key=f"{msg_id}_pert_bar")
                st.plotly_chart(line_fig, use_container_width=True, key=f"{msg_id}_pert_line")
                # Population proportions + delta + UMAP
                if _effective_grn_model in ("foxm1", "tubb", "mki67", "full", "full_foxm1", "full_aurkb"):
                    try:
                        if _effective_grn_model == "foxm1":
                            _sim_scored = load_foxm1_pop_sim()
                            _ko_label   = "FOXM1 KO"
                        elif _effective_grn_model == "tubb":
                            _sim_scored = load_tubb_pop_sim()
                            _ko_label   = "TUBB KO"
                        elif _effective_grn_model == "mki67":
                            _sim_scored = load_mki67_pop_sim()
                            _ko_label   = "BIRC5 KO"
                        elif _effective_grn_model == "full_foxm1":
                            _sim_scored = load_full_foxm1_pop_sim()
                            _ko_label   = "FOXM1 KO"
                        elif _effective_grn_model == "full_aurkb":
                            _sim_scored = load_full_aurkb_pop_sim()
                            _ko_label   = "AURKB KO"
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
                        _pt_pop_order = ["quiescent", "proliferative", "intermediate"]
                        _pt_pop_sizes = {"intermediate": 2, "proliferative": 3, "quiescent": 4}
                        try:
                            _pr = load_foxm1_pop_real()
                            _pop_umap_real = px.scatter(
                                _pr, x="x", y="y", color="population",
                                color_discrete_map=_POP_COLORS,
                                title=T['real_data'],
                                labels={"x": T['umap1'], "y": T['umap2'], "population": T['population']},
                                opacity=0.6, height=420, render_mode="svg",
                                category_orders={"population": _pt_pop_order},
                            )
                            for _pp, _ps in _pt_pop_sizes.items():
                                _pop_umap_real.update_traces(marker=dict(size=_ps), selector=dict(name=_pp))
                            _pop_umap_real.update_layout(plot_bgcolor="white", paper_bgcolor="white")
                            _umap_col1.plotly_chart(_pop_umap_real, use_container_width=True, key=f"{msg_id}_pop_umap_real")
                        except Exception:
                            pass
                        _needs_flip = _effective_grn_model in ("full_aurkb", "full_foxm1")
                        if _needs_flip:
                            _sim_scored = _sim_scored.copy()
                            _xm = _sim_scored["x_wt"].max()
                            _ym = _sim_scored["y_wt"].max()
                            _sim_scored["x_wt"] = _xm - _sim_scored["x_wt"]
                            _sim_scored["y_wt"] = _ym - _sim_scored["y_wt"]
                            _sim_scored["x_ko"] = _xm - _sim_scored["x_ko"]
                            _sim_scored["y_ko"] = _ym - _sim_scored["y_ko"]
                        _pop_umap_wt = px.scatter(
                            _sim_scored, x="x_wt", y="y_wt", color="pop_wt",
                            color_discrete_map=_POP_COLORS,
                            title=T['sim_wt'],
                            labels={"x_wt": T['umap1'], "y_wt": T['umap2'], "pop_wt": T['population']},
                            opacity=0.6, height=420, render_mode="svg",
                            category_orders={"pop_wt": _pt_pop_order},
                        )
                        for _pp, _ps in _pt_pop_sizes.items():
                            _pop_umap_wt.update_traces(marker=dict(size=_ps), selector=dict(name=_pp))
                        _pop_umap_wt.update_layout(plot_bgcolor="white", paper_bgcolor="white")
                        _pop_umap_ko = px.scatter(
                            _sim_scored, x="x_ko", y="y_ko", color="pop_ko",
                            color_discrete_map=_POP_COLORS,
                            title=f"Simulation {_ko_label} (populations *)",
                            labels={"x_ko": T['umap1'], "y_ko": T['umap2'], "pop_ko": T['population']},
                            opacity=0.6, height=420, render_mode="svg",
                            category_orders={"pop_ko": _pt_pop_order},
                        )
                        for _pp, _ps in _pt_pop_sizes.items():
                            _pop_umap_ko.update_traces(marker=dict(size=_ps), selector=dict(name=_pp))
                        _pop_umap_ko.update_layout(plot_bgcolor="white", paper_bgcolor="white")
                        _umap_col2.plotly_chart(_pop_umap_wt, use_container_width=True, key=f"{msg_id}_pop_umap_wt")
                        _umap_col3.plotly_chart(_pop_umap_ko, use_container_width=True, key=f"{msg_id}_pop_umap_ko")
                        st.caption(
                            "\\* **Populations scored by gene signatures:** "
                            "**Proliferative** — mean expression of top-100 MKI67 co-expression neighbours ≥ 70th percentile · "
                            "**Quiescent** — DNAJB1 z-score ≥ 70th percentile (DNAJB1 is the top HSPA1B neighbour, cosine sim = 0.91) · "
                            "**Intermediate** — all remaining cells"
                        )
                    except Exception as _e:
                        st.info(f"{T['pop_unavail']}: {_e}")
                _prog_map = {"tubb": "TUBB program (201 genes)", "foxm1": "FOXM1 program (198 genes)", "mki67": "MKI67 program (201 genes)", "full": "Full program (200 genes)", "full_foxm1": "Full program (200 genes)", "full_aurkb": "Full program (200 genes)"}
                _prog = _prog_map.get(_effective_grn_model, "program")
                st.caption(f"{T['sim_caption']} · {_prog} · {_ko_gene_label} {T['knocked_out']}")
            except Exception as e:
                st.info(f"{T['pert_unavail']}. ({e})")

    # ── Tab: Network graph GNN ────────────────────────────────────────────────
    if "network" in _tab_map:
        with _tab_map["network"]:
            st.plotly_chart(msg["grn_fig"], use_container_width=True, key=f"{msg_id}_grn")
            topo = msg.get("grn_topo")
            if topo is not None and not topo.empty:
                with st.expander(T['network_topology'], expanded=True):
                    n_fb = int(topo["Feedback loop"].sum())
                    n_up = int((topo["Role"] == "upstream").sum())
                    n_dn = int((topo["Role"] == "downstream").sum())
                    n_fb_role = int((topo["Role"] == "feedback loop").sum())
                    top_hub = topo[topo["Role"] != "query"].nlargest(1, "Total degree")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric(T['upstream'], n_up)
                    c2.metric(T['downstream'], n_dn)
                    c3.metric(T['feedback'], n_fb_role)
                    c4.metric(T['cycle_nodes'], n_fb)
                    if not top_hub.empty:
                        hub_name = top_hub.iloc[0]["Gene"]
                        hub_deg  = int(top_hub.iloc[0]["Total degree"])
                        st.info(f"{T['top_hub']}: **{hub_name}** (degree {hub_deg}) — "
                                f"{T['most_connected']}")
                    show_cols = ["Gene", "Role", "In-degree", "Out-degree",
                                 "Total degree", "Feedback loop"]
                    st.dataframe(
                        topo[show_cols].reset_index(drop=True),
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Feedback loop": st.column_config.CheckboxColumn("🔄 Cycle"),
                        }
                    )

    # ── Tab: Adjacency matrix GNN ─────────────────────────────────────────────
    if "adjacency" in _tab_map:
        with _tab_map["adjacency"]:
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

# ── Message rendering loop ───────────────────────────────────────
# Only the LAST assistant message is fully expanded.
# All older messages are collapsed in expanders (no Plotly rendered = fast).
_asst_indices = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
_last_asst_idx = _asst_indices[-1] if _asst_indices else -1

for _mi, msg in enumerate(messages):
    if msg["role"] == "user":
        continue  # gene name is already shown in the assistant message header — no duplication
    # assistant message — plain container, no chat bubble icons
    with st.container():
        if "fig" in msg:
            if _mi == _last_asst_idx:
                # Latest result — render everything
                _render_msg_figures(msg, id(msg))
            else:
                # Older result — collapsed
                _gene_lbl = msg.get("query_gene", "")
                with st.expander(f"{T['show_results']} {_gene_lbl}", expanded=False):
                    _render_msg_figures(msg, id(msg))
        else:
            st.write(msg["content"])

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
    _save_recent(dataset_key, query_gene)
    # Clear last_selected so the search bar resets to placeholder after a Featured gene run,
    # allowing the user to pick any gene (including the same one) without hitting the same-gene guard
    st.session_state[f"last_selected_{dataset_key}"] = ""
elif selected_gene and selected_gene != _last_q_gene:
    # Only fire a new query when the user picks a *different* gene
    # (the bar now keeps the current gene, so same-gene reruns must be ignored)
    st.session_state[f"last_selected_{dataset_key}"] = selected_gene
    _save_recent(dataset_key, selected_gene)
    query_gene = selected_gene
else:
    query_gene = None

if query_gene:
    # Clear previous results — no history, each search is fresh
    st.session_state[f"messages_{dataset_key}"] = []
    messages = st.session_state[f"messages_{dataset_key}"]
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

            # Clip colorscale at 99th percentile so sparse high-expressors
            # don't crush the rest of the scale into dark purple
            _p99 = float(np.percentile(expr_vals[expr_vals > 0], 99)) if (expr_vals > 0).any() else 1.0
            _cmax = max(_p99, float(expr_vals.max()) * 0.5)  # never below half the max

            fig = px.scatter(
                umap_plot, x="x", y="y",
                color="expression",
                color_continuous_scale="Viridis",
                range_color=[0, _cmax],
                title=f"{query_gene} — {T['expression']} ({len(umap_plot):,} cells shown)",
                labels={"x": T['umap1'], "y": T['umap2']},
                opacity=0.6, height=450,
                render_mode="svg"
            )
            fig.update_traces(marker=dict(size=3))
            fig.update_layout(coloraxis_colorbar=dict(title=T['expression']),
                              plot_bgcolor="white", paper_bgcolor="white")

        if "time" in umap_df.columns:
            fig_time = px.scatter(
                umap_plot, x="x", y="y",
                color=umap_plot["time"].astype(str),
                title=T['time'],
                labels={"x": T['umap1'], "y": T['umap2'], "color": T['time']},
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
                title=T['cell_type'],
                labels={"x": T['umap1'], "y": T['umap2'], "color": T['cell_type']},
                opacity=0.6, height=450,
                render_mode="svg"
            )
            fig_celltype.update_traces(marker=dict(size=3))
            fig_celltype.update_layout(plot_bgcolor="white", paper_bgcolor="white")

        program_genes = [query_gene] + [genes[i] for i in sorted_idx]

        # For the gene map always use top-200 neighbors so dots are visible
        _map_idx = np.argsort(sims)[::-1]
        _map_idx = [i for i in _map_idx if genes[i] != query_gene][:200]
        map_genes = [query_gene] + [genes[i] for i in _map_idx]

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
            elif grn_mat is not None:
                # Selector already has a valid model loaded — use it
                _grn_mat_q, _grn_genes_q = grn_mat, grn_genes
                _grn_model_q = grn_key
            else:
                # grn_mat not yet loaded (e.g. last_selected was empty, _gene_in_any_grn=False).
                # Fall back to the first model that contains this gene so GRN is never silently lost.
                _fallback_label = _q_models[0]
                _grn_model_q = _ALL_MODELS[_fallback_label][0]
                _grn_mat_q, _grn_genes_q = load_grn(_grn_model_q)
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
                    title=T['time_sim'],
                    labels={_xcol: T['umap1'], _ycol: T['umap2'], "color": T['time']},
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
                    title=f"{T['population']} — real data (DNAJB1 / MKI67-100)",
                    labels={"x": T['umap1'], "y": T['umap2'], "population": T['population']},
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
                    title=f"{T['population']} — simulation WT (DNAJB1 / MKI67-100)",
                    labels={"x_wt": T['umap1'], "y_wt": T['umap2'], "pop_wt": T['population']},
                    opacity=0.6, height=450, render_mode="svg",
                    category_orders={"pop_wt": _POP_ORDER},
                )
                for _pop, _sz in _POP_SIZES.items():
                    fig_pop_sim.update_traces(marker=dict(size=_sz), selector=dict(name=_pop))
                fig_pop_sim.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            except Exception:
                pass

        # Gene embedding map (PCA on neighborhood vectors)
        fig_gene_map = None
        try:
            _gene_umap_df = load_gene_umap2d(dataset_key)
            # similarity scores for program_genes (skip query itself)
            _prog_sims = {genes[i]: float(sims[i]) for i in sorted_idx}
            fig_gene_map = build_gene_embedding_map(
                _gene_umap_df, query_gene, program_genes, annotations,
                embeddings_arr=embeddings, genes_list=genes,
                sim_scores=_prog_sims
            )
        except Exception:
            pass

        _DRUG_MAP = {
            "TUBB":   "💊 **Vincristine** (FDA-approved, RMS standard of care)",
            "BIRC5":  "💊 **YM155** / Sepantronium (clinical trials)",
            "HSPA1B": "🎯 **Novel target** — HSP70 inhibitor class, no approved RMS drug",
        }
        _drug_line = _DRUG_MAP.get(query_gene.upper(), "")
        _content = f"**{T['gene_program_for']} {query_gene}** — {T['cluster']} {query_cluster}: *{query_annotation}*"
        if _drug_line:
            _content += f"\n\n{_drug_line}"

        messages.append({
            "role": "assistant",
            "content": _content,
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
            "fig_gene_map": fig_gene_map,
        })
        # Reset selectbox to placeholder (allows re-selecting the same gene next time)
        st.session_state[f"sel_version_{dataset_key}"] = st.session_state.get(f"sel_version_{dataset_key}", 0) + 1
        st.rerun()

# ── Settings expander — after results ────────────────────────────
with st.expander(T['settings'], expanded=False):
    if _gene_in_any_grn:
        _c1, _c2, _c3 = st.columns([3, 2, 2])
    else:
        _c1, _c2 = st.columns([3, 3])
    dataset_choice = _c1.selectbox(
        T['vector_db'],
        options=_ds_options,
        index=_ds_options.index(dataset_choice),
        key="dataset_select",
    )
    _c2.slider(
        T['program_size'],
        min_value=5, max_value=200, value=20, step=5,
        key=f"slider_{dataset_key}"
    )
    if _gene_in_any_grn:
        _c3.slider(
            T['grn_hops'],
            min_value=1, max_value=3, value=1, step=1,
            key=f"grn_slider_{dataset_key}"
        )
    if _gene_in_any_grn:
        if len(grn_options) == 1:
            st.caption(f"{T['grn_model']}: **{grn_options[0]}**")
        else:
            st.radio(
                T['grn_model'], options=grn_options,
                horizontal=True, key=_grn_state_key
            )

# ── Genes from your data expander — after results ─────────────────
with st.expander(T['genes_from_data'], expanded=False):
    st.caption(T['upload_note'])
    uploaded_file = st.file_uploader(T['upload_label'], type=["h5ad"], key="h5ad_upload")

    if uploaded_file is not None:
        file_id = uploaded_file.name + str(uploaded_file.size)

        if st.session_state.get("_upload_file_id") != file_id:
            import io, anndata as ad, scanpy as sc, scipy.sparse as sp_sparse

            with st.spinner(T['reading_file']):
                bytes_data = uploaded_file.read()
                adata = ad.read_h5ad(io.BytesIO(bytes_data))

            st.success(T['loaded_success'].format(n_obs=adata.n_obs, n_vars=adata.n_vars))

            if adata.n_obs > 100_000:
                st.warning(T['large_dataset'])

            with st.spinner(T['normalizing']):
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

            uploaded_var_names = list(adata.var_names)
            X_full = adata.X
            if sp_sparse.issparse(X_full):
                X_full = X_full.toarray()
            X_f16 = X_full.astype(np.float16)

            st.session_state["_upload_file_id"]   = file_id
            st.session_state["_upload_umap"]       = umap_coords
            st.session_state["_upload_var_names"]  = uploaded_var_names
            st.session_state["_upload_expr"]       = X_f16

        # ── Render ──────────────────────────────────────────────────────
        umap_up   = st.session_state.get("_upload_umap")
        var_names = st.session_state.get("_upload_var_names", [])
        expr_up   = st.session_state.get("_upload_expr")

        if umap_up is not None:
            overlap = [g for g in var_names if g in set(genes)]
            st.info(T['genes_found'].format(n_overlap=len(overlap), n_total=len(var_names)))

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

            gene_sel_up = st.selectbox(
                T['color_umap'],
                options=[T['select_gene_option']] + sorted(var_names),
                key="upload_gene_sel"
            )
            if gene_sel_up != T['select_gene_option'] and gene_sel_up in var_names:
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

# ── Featured genes — after results, before About ─────────────────
_FEATURED = [
    ("TUBB",   "tubb",  "💊 Vincristine · FDA-approved"),
    ("BIRC5",  "mki67", "💊 YM155 · clinical trials"),
    ("HSPA1B", "full",  "🎯 Novel target · HSP70 class"),
]
with st.expander(T['featured_targets'], expanded=False):
    _fcols = st.columns(len(_FEATURED))
    for _fc, (_fg, _fgrn, _fdrug) in zip(_fcols, _FEATURED):
        with _fc:
            st.caption(_fdrug)
            if st.button(f"**{_fg}**", key=f"feat_{_fg}_{dataset_key}", use_container_width=True):
                st.session_state[f"forced_grn_{dataset_key}"] = _fgrn
                st.session_state[f"recent_clicked_{dataset_key}"] = _fg
                st.rerun()

# ── About sections (bottom of page) ────────────────────────────
_about_expander = st.expander(T['about'], expanded=False)
with _about_expander:
    st.markdown(T['about_intro'])
    _arch_diagram_slot = st.container()   # arch diagram rendered here (defined later)
    st.markdown(T['how_it_works_body'])

    # ── SVG text translations (applied via .replace() to avoid f-string brace issues) ──
    _svg_fr = {
        # Trajectory diagram
        "Temporal Co-expression Trajectory":  "Trajectoire de co-expression temporelle",
        "cells × genes":                      "cellules × gènes",
        "× 6 time points":                    "× 6 points temporels",
        "per timepoint":                      "par point temporel",
        "gene-gene graph":                    "graphe gène-gène",
        "soft-threshold β=6":                 "seuillage doux β=6",
        "shared weights":                     "poids partagés",
        "align snapshots":                    "aligner les captures",
        "to common frame":                    "sur cadre commun",
        "gene shift + stable":                "décalage + stable",
        "Regulatory Structure  (OmniPath + PPGN)": "Structure régulatrice  (OmniPath + PPGN)",
        "expressed genes only":               "gènes exprimés uniquement",
        "feedback loops · triangles":         "boucles rétro · triangles",
        "regulatory motif":                   "motif régulateur",
        "RAG retrieval":                      "récupération RAG",
        # Arch diagram
        "Data-driven Gene Program Retrieval": "Récupération du programme génique",
        "🔍 Query Gene":                      "🔍 Gène requête",
        "cells × genes × time":              "cellules × gènes × temps",
        "cells × genes × t":                 "cellules × gènes × t",
        "node → embedding":                   "nœud → plongement",
        "vector DB · cosine sim":             "BD vectorielle · sim. cosinus",
        "co-expression neighbors":            "voisins co-expression",
        "(context-driven)":                   "(guidé par contexte)",
        "Gene Program":                       "Programme génique",
        "GRN Inference &amp; Perturbation Simulation": "Inférence GRN &amp; Simulation de perturbation",
        "ODE mechanistic model":              "modèle mécaniste ODE",
        "+ optimal transport":               "+ transport optimal",
        "activation / repression":            "activation / répression",
        "Perturbation":                       "Perturbation",
        "Simulation":                         "Simulation",
        "BIRC5 KO · propagate Δ":            "BIRC5 KO · propager Δ",
        "program-augmented":                  "guidé par programme",
        "Inferred GRN":                       "GRN inféré",
        "Therapeutic":                        "Thérapeutique",
        "Targets":                            "Cibles",
        "co-targets · direct targets":        "co-cibles · cibles directes",
        "reshape Waddington landscape":       "remodeler le paysage de Waddington",
    }
    def _svg_translate(svg):
        if _cur_lang == 'fr':
            for src, tgt in _svg_fr.items():
                svg = svg.replace(src, tgt)
        return svg

    _traj_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -36 920 501" style="font-family:Arial,sans-serif">
<defs>
  <style>
    @keyframes tPulse {
      0%, 100% { opacity: 1;    stroke-width: 2.5; }
      50%       { opacity: 0.1; stroke-width: 1.5; }
    }
    @keyframes tLabelPulse {
      0%, 100% { opacity: 1; }
      50%       { opacity: 0.1; }
    }
    .tblink path { animation: tPulse 1.4s ease-in-out infinite; }
    .tblink rect  { animation: tLabelPulse 1.4s ease-in-out infinite; }
    .tblink text  { animation: tLabelPulse 1.4s ease-in-out infinite; }
  </style>
  <marker id="th" markerWidth="8" markerHeight="8" refX="7" refY="3.5" orient="auto">
    <path d="M0,0 L0,7 L8,3.5 z" fill="#888"/>
  </marker>
  <marker id="thG" markerWidth="8" markerHeight="8" refX="7" refY="3.5" orient="auto">
    <path d="M0,0 L0,7 L8,3.5 z" fill="#16A085"/>
  </marker>
</defs>

<!-- ═══ ROW 1 TITLE ═══ -->
<text x="380" y="25" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50">Temporal Co-expression Trajectory</text>

<!-- ── Box 1: scRNA-seq × T (blue) ── -->
<rect x="12" y="34" width="158" height="170" rx="8" fill="#EBF4FC" stroke="#5B9BD5" stroke-width="2"/>
<text x="91" y="56" text-anchor="middle" font-size="14" font-weight="bold" fill="#5B9BD5">scRNA-seq</text>
<text x="91" y="72" text-anchor="middle" font-size="13" font-weight="bold" fill="#5B9BD5">× T</text>
<rect x="38" y="84" width="106" height="78" rx="3" fill="none" stroke="#5B9BD5" stroke-width="1.5" opacity="0.7"/>
<line x1="38" y1="104" x2="144" y2="104" stroke="#5B9BD5" stroke-width="0.8" opacity="0.4"/>
<line x1="38" y1="124" x2="144" y2="124" stroke="#5B9BD5" stroke-width="0.8" opacity="0.4"/>
<line x1="38" y1="144" x2="144" y2="144" stroke="#5B9BD5" stroke-width="0.8" opacity="0.4"/>
<line x1="73" y1="84" x2="73" y2="162" stroke="#5B9BD5" stroke-width="0.8" opacity="0.4"/>
<line x1="108" y1="84" x2="108" y2="162" stroke="#5B9BD5" stroke-width="0.8" opacity="0.4"/>
<text x="91" y="178" text-anchor="middle" font-size="11" fill="#777">cells × genes</text>
<text x="91" y="191" text-anchor="middle" font-size="11" fill="#777">× 6 time points</text>

<!-- ── Box 2: WGCNA × T (purple) ── -->
<rect x="188" y="34" width="158" height="170" rx="8" fill="#F5EFF9" stroke="#9B59B6" stroke-width="2"/>
<text x="267" y="56" text-anchor="middle" font-size="14" font-weight="bold" fill="#9B59B6">WGCNA</text>
<text x="267" y="72" text-anchor="middle" font-size="12" fill="#9B59B6">per timepoint</text>
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
<rect x="291" y="83" width="30" height="14" rx="4" fill="#9B59B6" opacity="0.15" stroke="#9B59B6" stroke-width="0.8"/>
<text x="306" y="93" text-anchor="middle" font-size="9" font-weight="bold" fill="#9B59B6">× 6</text>
<text x="267" y="175" text-anchor="middle" font-size="11" fill="#777">gene-gene graph</text>
<text x="267" y="188" text-anchor="middle" font-size="11" fill="#777">soft-threshold β=6</text>

<!-- ── Box 3: GAT Encoder (green) ── -->
<rect x="364" y="34" width="158" height="170" rx="8" fill="#EAFAF1" stroke="#27AE60" stroke-width="2"/>
<text x="443" y="56" text-anchor="middle" font-size="14" font-weight="bold" fill="#27AE60">GAT Encoder</text>
<text x="443" y="72" text-anchor="middle" font-size="11" fill="#27AE60">shared weights</text>
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
<text x="443" y="175" text-anchor="middle" font-size="11" fill="#777">snapshot emb_t</text>
<text x="443" y="188" text-anchor="middle" font-size="11" fill="#777">per timepoint</text>

<!-- ── Box 4: OT Alignment (orange) ── -->
<rect x="540" y="34" width="158" height="170" rx="8" fill="#FEF5E7" stroke="#E67E22" stroke-width="2"/>
<text x="619" y="56" text-anchor="middle" font-size="13" font-weight="bold" fill="#E67E22">OT Alignment</text>
<text x="619" y="72" text-anchor="middle" font-size="11" fill="#E67E22">Sinkhorn</text>
<circle cx="563" cy="100" r="5" fill="#E67E22" opacity="0.7"/>
<circle cx="563" cy="118" r="5" fill="#E67E22" opacity="0.6"/>
<circle cx="563" cy="136" r="5" fill="#E67E22" opacity="0.8"/>
<circle cx="563" cy="154" r="5" fill="#E67E22" opacity="0.5"/>
<line x1="570" y1="100" x2="666" y2="97" stroke="#E67E22" stroke-width="1" stroke-dasharray="3,2" opacity="0.55"/>
<line x1="570" y1="118" x2="666" y2="121" stroke="#E67E22" stroke-width="1" stroke-dasharray="3,2" opacity="0.55"/>
<line x1="570" y1="136" x2="666" y2="130" stroke="#E67E22" stroke-width="1" stroke-dasharray="3,2" opacity="0.55"/>
<line x1="570" y1="154" x2="666" y2="151" stroke="#E67E22" stroke-width="1" stroke-dasharray="3,2" opacity="0.45"/>
<circle cx="671" cy="97" r="5" fill="#E67E22"/>
<circle cx="671" cy="121" r="5" fill="#E67E22"/>
<circle cx="671" cy="130" r="5" fill="#E67E22"/>
<circle cx="671" cy="151" r="5" fill="#E67E22" opacity="0.75"/>
<text x="619" y="175" text-anchor="middle" font-size="11" fill="#777">align snapshots</text>
<text x="619" y="188" text-anchor="middle" font-size="11" fill="#777">to common frame</text>

<!-- ── Box 5: trajectory_emb (gold) ── -->
<rect x="716" y="34" width="158" height="170" rx="8" fill="#FEFDE7" stroke="#D4AC0D" stroke-width="2.5"/>
<text x="795" y="56" text-anchor="middle" font-size="13" font-weight="bold" fill="#B8860B">trajectory_emb</text>
<text x="795" y="72" text-anchor="middle" font-size="10" fill="#B8860B">δ + μ → MLP</text>
<circle cx="767" cy="100" r="5" fill="#D4AC0D" opacity="0.8"/>
<circle cx="787" cy="88" r="8" fill="#D4AC0D"/>
<circle cx="812" cy="103" r="5" fill="#D4AC0D" opacity="0.7"/>
<circle cx="773" cy="120" r="4" fill="#888" opacity="0.4"/>
<circle cx="820" cy="114" r="5" fill="#888" opacity="0.4"/>
<circle cx="800" cy="130" r="8" fill="#D4AC0D" opacity="0.9"/>
<circle cx="760" cy="134" r="4" fill="#888" opacity="0.3"/>
<circle cx="827" cy="92" r="4" fill="#888" opacity="0.3"/>
<circle cx="777" cy="148" r="5" fill="#D4AC0D" opacity="0.65"/>
<circle cx="813" cy="145" r="4" fill="#888" opacity="0.35"/>
<text x="795" y="175" text-anchor="middle" font-size="11" fill="#777">gene shift + stable</text>
<text x="795" y="188" text-anchor="middle" font-size="10" font-style="italic" fill="#B8860B">[N, 128]</text>

<!-- ═══ ROW 1 ARROWS ═══ -->
<line x1="170" y1="119" x2="185" y2="119" stroke="#888" stroke-width="1.5" marker-end="url(#th)"/>
<line x1="346" y1="119" x2="361" y2="119" stroke="#888" stroke-width="1.5" marker-end="url(#th)"/>
<line x1="522" y1="119" x2="537" y2="119" stroke="#888" stroke-width="1.5" marker-end="url(#th)"/>
<line x1="698" y1="119" x2="713" y2="119" stroke="#888" stroke-width="1.5" marker-end="url(#th)"/>

<!-- ═══ SPECIAL BLINK ARROW: trajectory_emb → final_emb ═══ -->
<g class="tblink">
  <path d="M 795,206 L 795,274"
        stroke="#16A085" stroke-width="2.5" fill="none" stroke-dasharray="7,4"
        marker-end="url(#thG)"/>
  <rect x="702" y="228" width="186" height="24" rx="5" fill="white" opacity="0.88"/>
  <text x="795" y="244" text-anchor="middle" font-size="12" font-weight="bold" fill="#16A085">+ OmniPath structure</text>
</g>

<!-- ═══ ROW 2 TITLE ═══ -->
<text x="460" y="268" text-anchor="middle" font-size="16" font-weight="bold" fill="#2c3e50">Regulatory Structure  (OmniPath + PPGN)</text>

<!-- ── Box 6: OmniPath (deep purple) ── -->
<rect x="12" y="276" width="158" height="170" rx="8" fill="#F0EBFF" stroke="#7D3C98" stroke-width="2"/>
<text x="91" y="298" text-anchor="middle" font-size="14" font-weight="bold" fill="#7D3C98">OmniPath</text>
<circle cx="55" cy="332" r="5" fill="#7D3C98"/>
<circle cx="91" cy="320" r="5" fill="#7D3C98"/>
<circle cx="127" cy="332" r="5" fill="#7D3C98"/>
<circle cx="68" cy="358" r="5" fill="#7D3C98"/>
<circle cx="114" cy="358" r="5" fill="#7D3C98"/>
<circle cx="91" cy="378" r="5" fill="#7D3C98"/>
<line x1="55" y1="332" x2="91" y2="320" stroke="#7D3C98" stroke-width="1" opacity="0.6"/>
<line x1="91" y1="320" x2="127" y2="332" stroke="#7D3C98" stroke-width="1" opacity="0.6"/>
<line x1="55" y1="332" x2="68" y2="358" stroke="#7D3C98" stroke-width="1" opacity="0.6"/>
<line x1="127" y1="332" x2="114" y2="358" stroke="#7D3C98" stroke-width="1" opacity="0.6"/>
<line x1="68" y1="358" x2="91" y2="378" stroke="#7D3C98" stroke-width="1" opacity="0.6"/>
<line x1="114" y1="358" x2="91" y2="378" stroke="#7D3C98" stroke-width="1" opacity="0.6"/>
<line x1="55" y1="332" x2="127" y2="332" stroke="#7D3C98" stroke-width="1" opacity="0.35"/>
<line x1="91" y1="320" x2="68" y2="358" stroke="#7D3C98" stroke-width="1" opacity="0.25"/>
<line x1="91" y1="320" x2="114" y2="358" stroke="#7D3C98" stroke-width="1" opacity="0.25"/>
<text x="91" y="408" text-anchor="middle" font-size="11" fill="#777">139K interactions</text>
<text x="91" y="421" text-anchor="middle" font-size="10" font-style="italic" fill="#777">PPI · signalling · regulatory</text>

<!-- ── Box 7: NEKO filter (teal) ── -->
<rect x="188" y="276" width="158" height="170" rx="8" fill="#E8FAF5" stroke="#16A085" stroke-width="2"/>
<text x="267" y="298" text-anchor="middle" font-size="14" font-weight="bold" fill="#16A085">NEKO</text>
<text x="267" y="314" text-anchor="middle" font-size="12" fill="#16A085">filter</text>
<circle cx="216" cy="332" r="3.5" fill="#7D3C98" opacity="0.5"/>
<circle cx="216" cy="347" r="3.5" fill="#7D3C98" opacity="0.5"/>
<circle cx="216" cy="362" r="3.5" fill="#7D3C98" opacity="0.5"/>
<circle cx="216" cy="377" r="3.5" fill="#888" opacity="0.3"/>
<circle cx="216" cy="392" r="3.5" fill="#888" opacity="0.3"/>
<line x1="221" y1="332" x2="304" y2="338" stroke="#16A085" stroke-width="1" opacity="0.6"/>
<line x1="221" y1="347" x2="304" y2="351" stroke="#16A085" stroke-width="1" opacity="0.6"/>
<line x1="221" y1="362" x2="304" y2="364" stroke="#16A085" stroke-width="1" opacity="0.5"/>
<circle cx="309" cy="338" r="5" fill="#16A085"/>
<circle cx="309" cy="351" r="5" fill="#16A085"/>
<circle cx="309" cy="364" r="5" fill="#16A085"/>
<text x="267" y="408" text-anchor="middle" font-size="11" fill="#777">expressed genes only</text>
<text x="267" y="421" text-anchor="middle" font-size="10" font-style="italic" fill="#777">gene symbol → UniProt</text>

<!-- ── Box 8: PPGN WL-3 (purple) ── -->
<rect x="364" y="276" width="158" height="170" rx="8" fill="#F5EFF9" stroke="#9B59B6" stroke-width="2"/>
<text x="443" y="298" text-anchor="middle" font-size="14" font-weight="bold" fill="#9B59B6">PPGN</text>
<rect x="390" y="304" width="106" height="17" rx="8" fill="#8E44AD" opacity="0.12" stroke="#8E44AD" stroke-width="1"/>
<text x="443" y="316" text-anchor="middle" font-size="9.5" font-weight="bold" fill="#8E44AD">WL-3 · Maron et al. 2019</text>
<rect x="399" y="328" width="88" height="88" rx="2" fill="none" stroke="#9B59B6" stroke-width="1.2" opacity="0.5"/>
<rect x="399" y="328" width="22" height="22" rx="1" fill="#9B59B6" opacity="0.7"/>
<rect x="443" y="328" width="22" height="22" rx="1" fill="#9B59B6" opacity="0.3"/>
<rect x="421" y="350" width="22" height="22" rx="1" fill="#9B59B6" opacity="0.55"/>
<rect x="465" y="350" width="22" height="22" rx="1" fill="#9B59B6" opacity="0.7"/>
<rect x="399" y="372" width="22" height="22" rx="1" fill="#9B59B6" opacity="0.3"/>
<rect x="443" y="372" width="22" height="22" rx="1" fill="#9B59B6" opacity="0.65"/>
<rect x="421" y="394" width="22" height="22" rx="1" fill="#9B59B6" opacity="0.7"/>
<rect x="465" y="394" width="22" height="22" rx="1" fill="#9B59B6" opacity="0.35"/>
<line x1="399" y1="350" x2="487" y2="350" stroke="#9B59B6" stroke-width="0.7" opacity="0.35"/>
<line x1="399" y1="372" x2="487" y2="372" stroke="#9B59B6" stroke-width="0.7" opacity="0.35"/>
<line x1="399" y1="394" x2="487" y2="394" stroke="#9B59B6" stroke-width="0.7" opacity="0.35"/>
<line x1="421" y1="328" x2="421" y2="416" stroke="#9B59B6" stroke-width="0.7" opacity="0.35"/>
<line x1="443" y1="328" x2="443" y2="416" stroke="#9B59B6" stroke-width="0.7" opacity="0.35"/>
<line x1="465" y1="328" x2="465" y2="416" stroke="#9B59B6" stroke-width="0.7" opacity="0.35"/>
<text x="443" y="432" text-anchor="middle" font-size="11" fill="#777">feedback loops · triangles</text>

<!-- ── Box 9: structural_emb (red) ── -->
<rect x="540" y="276" width="158" height="170" rx="8" fill="#FDEDEC" stroke="#E74C3C" stroke-width="2"/>
<text x="619" y="291" text-anchor="middle" font-size="13" font-weight="bold" fill="#E74C3C">structural_emb</text>
<circle cx="591" cy="325" r="5" fill="#E74C3C" opacity="0.7"/>
<circle cx="614" cy="314" r="7" fill="#E74C3C"/>
<circle cx="643" cy="328" r="5" fill="#E74C3C" opacity="0.6"/>
<circle cx="598" cy="348" r="4" fill="#888" opacity="0.4"/>
<circle cx="637" cy="354" r="5" fill="#888" opacity="0.4"/>
<circle cx="621" cy="367" r="8" fill="#E74C3C" opacity="0.85"/>
<circle cx="594" cy="378" r="4" fill="#888" opacity="0.3"/>
<circle cx="647" cy="343" r="4" fill="#888" opacity="0.35"/>
<circle cx="607" cy="388" r="5" fill="#E74C3C" opacity="0.6"/>
<circle cx="641" cy="380" r="4" fill="#888" opacity="0.3"/>
<text x="619" y="411" text-anchor="middle" font-size="11" fill="#777">regulatory motif</text>
<text x="619" y="424" text-anchor="middle" font-size="11" fill="#777">fingerprint  [N_omni, 128]</text>

<!-- ── Box 10: final_emb (teal) ── -->
<rect x="716" y="276" width="158" height="170" rx="8" fill="#E8FAF5" stroke="#16A085" stroke-width="2.5"/>
<text x="795" y="298" text-anchor="middle" font-size="14" font-weight="bold" fill="#16A085">final_emb</text>
<rect x="737" y="304" width="116" height="17" rx="8" fill="#16A085" opacity="0.1" stroke="#16A085" stroke-width="1"/>
<text x="795" y="316" text-anchor="middle" font-size="9.5" font-weight="bold" fill="#16A085">traj + structural</text>
<circle cx="767" cy="345" r="6" fill="#D4AC0D" opacity="0.85"/>
<circle cx="788" cy="332" r="7" fill="#D4AC0D"/>
<circle cx="813" cy="346" r="5" fill="#D4AC0D" opacity="0.7"/>
<circle cx="775" cy="364" r="5" fill="#E74C3C" opacity="0.7"/>
<circle cx="810" cy="357" r="7" fill="#E74C3C"/>
<circle cx="795" cy="377" r="8" fill="#16A085" opacity="0.9"/>
<circle cx="760" cy="358" r="4" fill="#16A085" opacity="0.45"/>
<circle cx="826" cy="336" r="4" fill="#16A085" opacity="0.4"/>
<text x="795" y="411" text-anchor="middle" font-size="11" fill="#777">RAG retrieval</text>
<text x="795" y="424" text-anchor="middle" font-size="10" font-style="italic" fill="#16A085">[N, 128]</text>

<!-- ═══ ROW 2 ARROWS ═══ -->
<line x1="170" y1="361" x2="185" y2="361" stroke="#888" stroke-width="1.5" marker-end="url(#th)"/>
<line x1="346" y1="361" x2="361" y2="361" stroke="#888" stroke-width="1.5" marker-end="url(#th)"/>
<line x1="522" y1="361" x2="537" y2="361" stroke="#888" stroke-width="1.5" marker-end="url(#th)"/>
<line x1="698" y1="361" x2="713" y2="361" stroke="#888" stroke-width="1.5" marker-end="url(#th)"/>

</svg>"""
    _traj_svg = _svg_translate(_traj_svg)
    _components.html(
        f"""<style>html,body{{margin:0;padding:0;overflow:hidden}}</style>
        <div style='width:100%'>{_traj_svg}</div>
        <script>
        var s = document.querySelector('svg');
        function _sh() {{
            var h = s.getBoundingClientRect().height;
            window.parent.postMessage({{type:'streamlit:setFrameHeight', height: Math.ceil(h) + 4}}, '*');
        }}
        window.addEventListener('load', _sh);
        window.addEventListener('resize', _sh);
        setTimeout(_sh, 200);
        </script>""",
        height=440, scrolling=False
    )

    _ot_img_local = _os.path.join(LOCAL_DIR, "ot_explanation.png")
    if not _os.path.exists(_ot_img_local):
        try:
            _token = st.secrets.get("HF_TOKEN", None)
            _ot_img_local = hf_hub_download(repo_id=REPO_ID, filename="ot_explanation.png",
                                             repo_type="dataset", token=_token)
        except Exception:
            _ot_img_local = None
    if _ot_img_local:
        st.image(
            _ot_img_local,
            caption="Why optimal transport is needed before computing the trajectory delta: OT normalises the embedding distributions across timepoints so that the delta reflects real context shift rather than a scaling artefact of the WGCNA graph density.",
            use_container_width=True,
        )

    st.markdown("""
**References:**
- CARDAMOM / CardamomOT: [github.com/eliasventre/CardamomOT](https://github.com/eliasventre/CardamomOT)
- Nebius AI Studio (Llama 3.1-8B inference): [studio.nebius.com](https://studio.nebius.com)
- Maron et al. *Provably Powerful Graph Networks.* NeurIPS 2019 · [arXiv:1905.11136](https://arxiv.org/abs/1905.11136)
- OmniPath / NEKO: [omnipathdb.org](https://omnipathdb.org)
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
  <text x="520" y="244" text-anchor="middle" font-size="12" font-weight="bold" fill="#C0392B">Gene Program</text>
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

# Arch diagram goes into the slot at the TOP of "About this tool"
_arch_svg = _svg_translate(_arch_svg)
with _arch_diagram_slot:
    _components.html(
        f"""<style>html,body{{margin:0;padding:0;overflow:hidden}}</style>
        <div style='width:100%'>{_arch_svg}</div>
        <script>
        var s = document.querySelector('svg');
        function _sh() {{
            var h = s.getBoundingClientRect().height;
            window.parent.postMessage({{type:'streamlit:setFrameHeight', height: Math.ceil(h) + 4}}, '*');
        }}
        window.addEventListener('load', _sh);
        window.addEventListener('resize', _sh);
        setTimeout(_sh, 200);
        </script>""",
        height=440, scrolling=False
    )

# ── Override styles — injected LAST so they beat emotion CSS ─────
st.markdown("""
<style>
/* Placeholder text grey */
div[data-testid="stSelectbox"] [class*="placeholder"] { color: #aaa !important; }

/* Borderless expanders — covers Streamlit 1.3x–1.5x structures */
div[data-testid="stExpander"] details,
div[data-testid="stExpander"] > details,
section.main details {
    border: none !important;
    border-top: none !important;
    border-bottom: none !important;
    box-shadow: none !important;
    outline: none !important;
    background: transparent !important;
}
div[data-testid="stExpander"] details > summary,
section.main details > summary {
    background: transparent !important;
    padding-left: 0.5rem !important;
}
div[data-testid="stExpander"] details > summary:hover,
section.main details > summary:hover {
    background: rgba(0,0,0,0.03) !important;
}

/* Vector database selectbox — borderless (stColumn with capital C, Streamlit ≥1.3) */
div[data-testid="stColumn"] div[data-testid="stSelectbox"] > div > div,
div[data-testid="column"]   div[data-testid="stSelectbox"] > div > div {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    min-height: unset !important;
    border-radius: 6px !important;
}
div[data-testid="stColumn"] div[data-testid="stSelectbox"] > div > div:hover,
div[data-testid="column"]   div[data-testid="stSelectbox"] > div > div:hover {
    background: #f5f5f5 !important;
}
</style>
""", unsafe_allow_html=True)


