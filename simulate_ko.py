"""
Minimal standalone KO simulation — no ot/seaborn/CardamomOT import needed.

Functions are adapted directly from CardamomOT/inference/trajectory.py and
simulations.py with heavy-dependency imports removed.

The simulation uses the BurstyPDMP (stochastic) model by default, matching
the pre-computed results.  For the last timepoint it switches to deterministic
ODE (finish_by_determinist = 1), also matching the CardamomOT default.

Public API
----------
simulate_ko(gene_name, gene_names, grn_params)
    Returns (X_wt, X_ko) — RNA count matrices, shape (n_cells, n_genes).
"""

import numpy as np
from numba import njit, prange
from joblib import Parallel, delayed

# ─────────────────────────────────────────────────────────────────────────────
# Numba kernels (identical to CardamomOT/inference/simulations.py)
# ─────────────────────────────────────────────────────────────────────────────

@njit
def _base_kon_vector(theta_basal, theta_inter, y_prot):
    n_cells, G = y_prot.shape
    Gm1, n_net = theta_basal.shape[0], theta_basal.shape[1]
    Z = np.zeros((n_cells, Gm1, n_net))
    result = np.zeros((n_cells, Gm1, n_net + 1))
    for i in range(n_cells):
        for j in range(Gm1):
            z_max = -1e300
            for k in range(n_net):
                Z[i, j, k] = theta_basal[j, k]
                for g in range(G):
                    Z[i, j, k] += y_prot[i, g] * theta_inter[g, j, k]
                if Z[i, j, k] > z_max:
                    z_max = Z[i, j, k]
            denom = np.exp(-z_max)
            result[i, j, 0] = np.exp(-z_max)
            for k in range(n_net):
                val = np.exp(Z[i, j, k] - z_max)
                denom += val
                result[i, j, k + 1] = val
            result[i, j] /= denom
    return result


@njit(fastmath=True, parallel=True)
def _kon_ref_vector(y_prot, kz, theta_inter, theta_basal):
    sigma = _base_kon_vector(theta_basal, theta_inter, y_prot)
    out = np.zeros(sigma.shape[:2])
    for i in prange(sigma.shape[0]):
        for j in prange(sigma.shape[1]):
            for k in prange(sigma.shape[2]):
                out[i, j] += kz[j, k] * sigma[i, j, k]
    return out


@njit
def _kon_ref(y_prot, kz, theta_inter, theta_basal):
    """Single-cell kon_ref used in BurstyPDMP.step."""
    result = _base_kon_vector(theta_basal, theta_inter, y_prot)
    res = np.zeros(theta_basal.shape[0])
    for j in range(theta_basal.shape[0]):
        for k in range(result.shape[2]):
            res[j] += kz[j, k] * result[-1, j, k]
    return res


@njit
def _kon_ref_single(y_prot_row, kz, theta_inter, theta_basal):
    """Single-cell version used in ODE step."""
    y = y_prot_row.reshape(1, -1)
    sigma = _base_kon_vector(theta_basal, theta_inter, y)
    out = np.zeros(sigma.shape[1])
    for j in range(sigma.shape[1]):
        for k in range(sigma.shape[2]):
            out[j] += kz[j, k] * sigma[0, j, k]
    return out


@njit
def _flow(time, d1, P):
    """Exponential decay of protein levels."""
    Pnew = P * np.exp(-time * d1)
    Pnew[0] = P[0]   # freeze stimulus
    return Pnew


@njit
def _step_ode(d1, ks, inter, basal, dt, scale, P_row):
    """Single Euler step for deterministic ODE."""
    a = _kon_ref_single(P_row, ks, inter, basal)
    Pnew = scale * a + (P_row - scale * a) * np.exp(-d1 * dt)
    Pnew[0] = P_row[0]
    return Pnew


# ─────────────────────────────────────────────────────────────────────────────
# BurstyPDMP — stochastic piecewise-deterministic Markov process
# (copied from CardamomOT/inference/simulations.py)
# ─────────────────────────────────────────────────────────────────────────────

class _BurstyPDMP:
    def __init__(self, ks, basal, inter):
        G = basal.shape[0]
        self.basal  = basal
        self.inter  = inter
        type_list = [('P', 'float')]
        self.state  = np.array([(0,) for _ in range(G)], dtype=type_list)
        self.thin_cst = np.sum(np.max(ks[1:, :], axis=1))

    def step(self, d1, ks, c, scale):
        tau = self.thin_cst
        U   = np.random.exponential(scale=1.0 / tau)
        P   = _flow(U, d1, self.state['P'])
        self.state['P'] = P

        v = _kon_ref(P.reshape((1, -1)), ks, self.inter, self.basal) / tau
        v[0] = 1.0 - np.sum(v[1:])
        i    = np.searchsorted(np.cumsum(v), np.random.random(), side='right')
        jump = i > 0
        if jump:
            r = (c / scale)[i]
            self.state['P'][i] += np.random.exponential(1.0 / r)
        return U, jump

    def simulation(self, d1, ks, c, timepoints, scale):
        G    = self.basal.shape[0]
        sim  = []
        T    = 0.0
        Told, state_old = T, self.state.copy()
        for t in timepoints:
            while T < t:
                Told, state_old = T, self.state.copy()
                U, _ = self.step(d1, ks, c, scale)
                T   += U
            P = _flow(t - Told, d1, state_old['P'])
            sim.append(np.array([(P[i],) for i in range(1, G)],
                                 dtype=[('P', 'float64')]))
        self.state['P'] = P
        return np.array(sim)


class _Simulation:
    def __init__(self, t, p):
        self.t = t
        self.p = p


def _simulate_next_prot_pdmp(d, a, c, basal, inter, t, scale, P0=None):
    """Stochastic PDMP simulation (BurstyPDMP) for one cell over delta_t."""
    if np.ndim(t) == 0:
        t = np.array([float(t)])
    network = _BurstyPDMP(a, basal, inter)
    if P0 is not None:
        network.state['P'][1:] = P0[1:]
    network.state['P'][0] = 1.0
    sim = network.simulation(d, a, c, t, scale)
    p = sim['P']
    return _Simulation(t, p)


def _simulate_next_prot_ode(d, ks, basal, inter, t, scale, P0=None):
    """Deterministic ODE simulation for one cell over delta_t."""
    if np.ndim(t) == 0:
        t = np.array([float(t)])
    G = d.size
    euler_step = 1e-2 / max(np.max(d), 1e-9)
    dt = min(euler_step, np.min(t[t > 0]) if np.any(t > 0) else euler_step)
    state = np.zeros(G)
    if P0 is not None:
        state[1:] = P0[1:]
    state[0] = 1.0
    sim = []
    T   = 0.0
    for tp in t:
        while T < tp:
            step = min(dt, tp - T)
            state = _step_ode(d, ks, inter, basal, step, scale, state)
            T    += step
        sim.append(state[1:].copy())
    return _Simulation(t, np.array(sim))


# ─────────────────────────────────────────────────────────────────────────────
# simulate_trajectories_unitary  (standalone, PDMP by default)
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_trajectories_unitary(
    prot_init,        # (N, G+1)
    kon_beta_init,    # (N, G+1)
    inter,            # (G+1, G+1, n_nets)  non-temporal
    basal,            # (G+1, n_nets)
    inter_t,          # (T-1, G+1, G+1, n_nets)
    basal_t,          # (T-1, G+1, n_nets)
    d_t,              # (T-1, 2, G+1)
    ks,               # (G+1, n_nets+1)  normalised kz
    rescale,          # (G+1,)  max kz per gene (for PDMP burst scaling)
    times,            # sorted 1-D array of simulation timepoints
    times_train,      # sorted unique training timepoints
    scale=1.0,
    stochastic=True,  # True = PDMP, False = ODE
    finish_by_det=True,  # switch to ODE for last time interval
):
    times = np.sort(times)
    N  = prot_init.shape[0]
    G1 = prot_init.shape[1]

    prot_modified = np.ones((N * len(times), G1))
    kon_vector    = np.ones((N * len(times), G1))

    prot_modified[:N, :] = prot_init
    kon_vector[:N, :]    = kon_beta_init
    kon_vector[:N, 1:]   = _kon_ref_vector(prot_init, ks, inter, basal)[:, 1:]

    start_time = 0
    if times_train[-1] < times[1]:
        times = np.concatenate(([0, times_train[-1]], times[1:]))
        prot_modified = np.ones((N * len(times), G1))
        kon_vector    = np.ones((N * len(times), G1))
        prot_modified[:N, :] = prot_init
        kon_vector[:N, :]    = kon_beta_init
        kon_vector[:N, 1:]   = _kon_ref_vector(prot_init, ks, inter, basal)[:, 1:]
        prot_modified[N:2*N, :] = prot_init
        kon_vector[N:2*N, 1:]   = _kon_ref_vector(prot_init, ks, inter_t[-1], basal_t[-1])[:, 1:]
        start_time = 1

    T_sim = len(times)
    d_t_sim     = np.zeros((T_sim - 1, 2, G1))
    inter_t_sim = np.zeros((T_sim - 1, G1, G1, inter_t.shape[-1]))
    basal_t_sim = np.zeros((T_sim - 1, G1, basal_t.shape[-1]))

    for cnt, time in enumerate(times[:-1]):
        idx = int(np.argmin(np.abs(times_train[:-1] - time)))
        d_t_sim[cnt]     = d_t[idx]
        inter_t_sim[cnt] = inter_t[idx]
        basal_t_sim[cnt] = basal_t[idx]

    times_simulation = np.zeros(T_sim * N)
    for t_idx in range(T_sim):
        times_simulation[t_idx * N:(t_idx + 1) * N] = times[t_idx]

    # Pre-compute kz (rescaled) once: shape (G+1, n_nets+1)
    kz = ks * rescale.reshape(ks.shape[0], 1)

    # min_ratio / max_ratio defaults from CardamomOT NetworkModel.__init__
    MIN_RATIO = 0.01
    MAX_RATIO = 100.0

    for cnt in range(start_time, T_sim - 1):
        delta_t = times[cnt + 1] - times[cnt]
        deg     = d_t_sim[cnt].copy()

        # Apply the PDMP stochastic degradation modification:
        # ratios_cnt = clipped version of loaded ratios (from ratios.npy)
        # degradations[0] = d[1] * ratios * (1 + sqrt(cnt))
        # This is what CardamomOT does in simulate_trajectories_unitary.
        # We load ratios per interval (matching the same index used for d_t).
        # For simplicity we use the stored d_t[cnt, 0] / d_t[cnt, 1] as ratios
        # (since ratios.npy was used to fill d_t during inference).
        ratios_cnt = np.clip(
            d_t_sim[cnt, 0, :] / np.maximum(d_t_sim[cnt, 1, :], 1e-9),
            MIN_RATIO, MAX_RATIO,
        )
        # Override mRNA degradation with scaled version used in PDMP
        deg[0, :] = d_t_sim[cnt, 1, :] * ratios_cnt * min(1.0 + np.sqrt(cnt), MAX_RATIO)

        # BurstyPDMP params
        d1    = deg[1, :]                                     # protein deg
        a_pdm = kz * deg[0, :].reshape(kz.shape[0], 1)       # burst rates
        c_pdm = rescale * (deg[0, :] / np.maximum(deg[1, :], 1e-9))  # burst sizes

        # Switch to ODE for last interval if finish_by_det
        use_stochastic = stochastic
        if finish_by_det and cnt >= T_sim - 2:
            use_stochastic = False

        start_i = cnt * N
        end_i   = (cnt + 1) * N

        def _run_cell(n, _start=start_i, _d1=d1, _a=a_pdm, _c=c_pdm,
                      _bt=basal_t_sim[cnt], _it=inter_t_sim[cnt],
                      _dt=delta_t, _ks=ks, _use_stoch=use_stochastic):
            P0 = prot_modified[_start + n, :]
            if _use_stoch:
                return _simulate_next_prot_pdmp(
                    _d1, _a, _c, _bt, _it, np.array([_dt]), scale, P0=P0
                ).p[-1]
            else:
                return _simulate_next_prot_ode(
                    _d1, _ks, _bt, _it, np.array([_dt]), scale, P0=P0
                ).p[-1]

        results = Parallel(n_jobs=-1)(delayed(_run_cell)(n) for n in range(N))

        for n, res in enumerate(results):
            prot_modified[end_i + n, 1:] = res

        kon_vector[end_i:end_i + N, 1:] = _kon_ref_vector(
            prot_modified[end_i:end_i + N, :],
            ks, inter_t_sim[cnt], basal_t_sim[cnt],
        )[:, 1:]

    return prot_modified, kon_vector, times_simulation


# ─────────────────────────────────────────────────────────────────────────────
# High-level simulate_network wrapper
# ─────────────────────────────────────────────────────────────────────────────

def _run_simulation(p, times, ko_idx=None, stochastic=True):
    """
    Simulate the GRN network for WT (ko_idx=None) or KO (ko_idx=int).

    Parameters
    ----------
    p          : dict from load_grn_params()
    times      : sorted 1-D array, e.g. [0,16,32,48,64,80]
    ko_idx     : gene column index (0=stimulus, 1..200=genes) to knock out
    stochastic : True = PDMP (matches pre-computed results), False = ODE (faster)

    Returns
    -------
    kon_theta : np.ndarray, shape (N*len(times), 201)
    N         : int, number of cells at t=0
    """
    inter_t   = p["inter_t_simul"].copy()       # (5, 201, 201, 1)
    basal     = p["basal_simul"].copy()          # (201, 1)
    basal_t   = p["basal_t_simul"].copy()        # (5, 201, 1)
    prot      = p["data_prot_sub200"].copy()     # (1200, 201)
    kon_beta  = p["data_kon_beta_sub200"].copy() # (1200, 201)
    times_data = p["data_times_sub200"].copy()   # (1200,)
    d_t       = p["degradations_temporal"].copy()   # (5, 2, 201)
    mixture   = p["mixture_parameters"].copy()   # (3, 201)

    # non-temporal interaction matrix (inter_simul.npy, NOT inter_t[0])
    inter = p.get("inter_simul", inter_t[0]).copy()   # (201, 201, 1)

    # ─ KO perturbation ───────────────────────────────────────────────────────
    if ko_idx is not None:
        basal_t[:, ko_idx, :] = -100.0 - np.sum(inter_t[-1, :, ko_idx, :], axis=0)
        prot[:, ko_idx] = 0.0

    # ─ Build ks (normalised kz) and rescale ──────────────────────────────────
    kz_raw  = mixture[:-1, :]                            # (2, 201)
    max_kz  = np.max(kz_raw, axis=0)                     # (201,)
    max_kz  = np.where(max_kz == 0, 1.0, max_kz)
    ks      = (kz_raw / max_kz).T                        # (201, 2)

    # rescale[0] = 1 (stimulus); rescale[1:] = max kz per gene
    rescale       = np.ones(201)
    rescale[1:]   = np.max(kz_raw[:, 1:], axis=0)       # (200,)

    times_train = np.sort(np.unique(times_data))
    N = int(np.sum(times_data == times_train[0]))

    prot_init     = prot[:N, :]
    kon_beta_init = kon_beta[:N, :]

    _, kon_theta, _ = _simulate_trajectories_unitary(
        prot_init, kon_beta_init,
        inter, basal,
        inter_t, basal_t,
        d_t, ks, rescale,
        np.sort(times),
        times_train,
        scale=1.0,
        stochastic=stochastic,
        finish_by_det=True,
    )
    return kon_theta, N


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def simulate_ko(gene_name: str, gene_names: list, grn_params: dict):
    """
    Run a CardamomOT KO simulation for *gene_name*.

    Parameters
    ----------
    gene_name  : gene to knock out (must be in gene_names)
    gene_names : list of 200 gene names (length 200, 0-indexed)
    grn_params : dict from load_grn_params() — numpy arrays loaded from HF

    Returns
    -------
    X_wt, X_ko : simulated RNA count matrices, shape (n_cells, n_genes)
                 n_genes=200, n_cells = N_cells_t0 × 6 timepoints
    """
    times = np.array([0., 16., 32., 48., 64., 80.])

    kon_wt, _ = _run_simulation(grn_params, times, ko_idx=None, stochastic=True)
    ko_idx = gene_names.index(gene_name) + 1   # +1 because col 0 = stimulus
    kon_ko, _ = _run_simulation(grn_params, times, ko_idx=ko_idx, stochastic=True)

    p  = grn_params
    mp = p["mixture_parameters"]     # (3, 201)
    c        = mp[-1, :]             # (201,)
    kz       = mp[:-1, :]           # (2, 201)
    pi_zinb  = p["pi_zinb"]         # (200,)

    def _sample_rna(kon):
        """Convert kon_theta → RNA counts via NB + zero inflation.

        NB parametrization: NB(n, p) where n = max(kz)*kon, p = c/(c+1).
        Mean = n*(1-p)/p = max(kz)*kon/c  (matches check_KOV_to_sim.py).
        """
        n_nb = np.maximum(
            (np.max(kz, axis=0) * kon)[:, 1:],   # (n_cells, 200)
            1e-3,
        )
        prob = c[1:] / (c[1:] + 1.0)             # (200,)
        rna = np.random.negative_binomial(
            n_nb.astype(np.float64),
            np.broadcast_to(prob, n_nb.shape),
        ).astype(float)
        pi_z = np.minimum(pi_zinb, 0.99)
        rna[np.random.uniform(0, 1, rna.shape) < pi_z] = 0.0
        return rna

    return _sample_rna(kon_wt), _sample_rna(kon_ko)


def simulate_ko_kon(gene_name: str, gene_names: list, grn_params: dict):
    """
    Run a CardamomOT KO simulation (PDMP, matching pre-computed results) and
    return kon_theta values at the final timepoint (t=80).

    Returns
    -------
    kon_wt, kon_ko : np.ndarray, shape (N_cells_t0, 201) — last timepoint only.
                     Col 0 = stimulus, cols 1..200 = genes.

    gene_names is ignored if grn_params contains "gene_names" (preferred).
    """
    # Use the simulation-internal gene list if available (200 GRN genes, cols 1..200)
    sim_gene_names = grn_params.get("gene_names", gene_names)
    if gene_name not in sim_gene_names:
        raise ValueError(
            f"Gene '{gene_name}' not found in the simulation GRN "
            f"({len(sim_gene_names)} genes). "
            f"Available genes include: {sim_gene_names[:10]}..."
        )
    times = np.array([0., 16., 32., 48., 64., 80.])
    kon_wt_all, N = _run_simulation(grn_params, times, ko_idx=None, stochastic=True)
    ko_idx = sim_gene_names.index(gene_name) + 1   # +1 to skip stimulus col 0
    kon_ko_all, _ = _run_simulation(grn_params, times, ko_idx=ko_idx, stochastic=True)
    # Return only the final timepoint (t=80)
    return kon_wt_all[-N:, :], kon_ko_all[-N:, :]
