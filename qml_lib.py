from __future__ import annotations

import time
from collections import defaultdict
from functools import reduce
from itertools import combinations, product

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix, eye as sparse_eye, kron as sparse_kron
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset



# ─────────────────────────────────────────────────────────────────────────────
# Each section is blocked by these comments. To see their explanations,
# please check the relevant part in the report. The order of functions and
# their order in the report is same.
# ─────────────────────────────────────────────────────────────────────────────




# ─────────────────────────────────────────────────────────────────────────────
# Single-qubit building blocks
# For details, see final report: Constructing the Hamiltonians
# ─────────────────────────────────────────────────────────────────────────────

I_SP = sparse_eye(2, format="csr")
X_SP = csr_matrix([[0, 1], [1, 0]])
Y_SP = csr_matrix([[0, -1j], [1j, 0]])
Z_SP = csr_matrix([[1, 0], [0, -1]])

SPARSE_MAP = {"I": I_SP, "X": X_SP, "Y": Y_SP, "Z": Z_SP}
STATE_MAP = {0: np.array([1, 0], dtype=complex),
             1: np.array([0, 1], dtype=complex)}


def pauli_string_to_matrix(s: str):
    """Sparse 2^n x 2^n matrix for a Pauli string like 'XYZI'."""
    return reduce(sparse_kron, (SPARSE_MAP[c] for c in s)).tocsr()


# ─────────────────────────────────────────────────────────────────────────────
# Hamiltonian families: each returns a term list [(coeff, pauli_string), ...]
# For details, see final report: Hamiltonian Families
# ─────────────────────────────────────────────────────────────────────────────

def build_hamiltonian(terms, n_qubits):
    """
    Compile a term list into (sparse H, pauli_strings, coeffs).
    That is mainly used to derive labels.
    """
    pauli_strings, coeffs = [], []
    for c, s in terms:
        assert len(s) == n_qubits, (
            f"term '{s}' has length {len(s)} != n_qubits={n_qubits}")
        pauli_strings.append(s)
        coeffs.append(c)

    H_sparse = sum(c * pauli_string_to_matrix(s)
                   for c, s in zip(coeffs, pauli_strings)).tocsr()
    return H_sparse, pauli_strings, np.array(coeffs)


def heisenberg_2d_terms(n_rows, n_cols, rng):
    """2D random anti-ferromagnetic Heisenberg on an n_rows x n_cols grid."""
    n = n_rows * n_cols
    terms = []

    def idx(r, c):
        return r * n_cols + c

    for r in range(n_rows):
        for c in range(n_cols):
            if c + 1 < n_cols:
                J = rng.uniform(0, 1)
                for P in "XYZ":
                    s = ["I"] * n
                    s[idx(r, c)] = P
                    s[idx(r, c + 1)] = P
                    terms.append((J, "".join(s)))
            if r + 1 < n_rows:
                J = rng.uniform(0, 1)
                for P in "XYZ":
                    s = ["I"] * n
                    s[idx(r, c)] = P
                    s[idx(r + 1, c)] = P
                    terms.append((J, "".join(s)))
    return terms


def heisenberg_1d_terms(n, rng):
    """1D random Heisenberg chain (open boundary)."""
    terms = []
    for i in range(n - 1):
        J = rng.uniform(0, 1)
        for P in "XYZ":
            s = ["I"] * n
            s[i] = P
            s[i + 1] = P
            terms.append((J, "".join(s)))
    return terms


def tfim_terms(n, J=1.0, h_x=1.0):
    """1D transverse-field Ising chain (open boundary)."""
    terms = []
    for i in range(n - 1):
        s = ["I"] * n
        s[i] = s[i + 1] = "Z"
        terms.append((-J, "".join(s)))
    for i in range(n):
        s = ["I"] * n
        s[i] = "X"
        terms.append((-h_x, "".join(s)))
    return terms


def mixed_field_ising_terms(n, J=1.0, h_x=1.0, h_z=0.5):
    """1D mixed-field Ising chain. """
    terms = tfim_terms(n, J=J, h_x=h_x)
    for i in range(n):
        s = ["I"] * n
        s[i] = "Z"
        terms.append((-h_z, "".join(s)))
    return terms


# ─────────────────────────────────────────────────────────────────────────────
# Trotter circuit primitives 
# For details, see final report: Trotterization Implementations Details
# ─────────────────────────────────────────────────────────────────────────────

def apply_pauli_rotation(pauli_string, theta):
    """e^{-i theta P}: basis-change -> CNOT ladder -> Rz(2θ) -> uncompute"""
    support = [(q, p) for q, p in enumerate(pauli_string) if p != "I"]
    if not support:
        return          # identity  skip

    # 1. Rotate each support qubit into the Z eigenbasis
    for q, p in support:
        if p == "X":
            qml.Hadamard(wires=q)
        elif p == "Y":
            qml.RX(np.pi / 2, wires=q)

    # 2. CNOT ladder
    support_qubits = [q for q, _ in support]
    for i in range(len(support_qubits) - 1):
        qml.CNOT(wires=[support_qubits[i], support_qubits[i + 1]])

    # 3. Rz on target
    qml.RZ(2 * theta, wires=support_qubits[-1])

    # 4. Uncompute CNOT ladder
    for i in reversed(range(len(support_qubits) - 1)):
        qml.CNOT(wires=[support_qubits[i], support_qubits[i + 1]])

    # 5. Undo basis change
    for q, p in support:
        if p == "X":
            qml.Hadamard(wires=q)
        elif p == "Y":
            qml.RX(-np.pi / 2, wires=q)


def trotter_circuit(pauli_strings, coeffs, t, n_steps):
    """First-order Trotterised e^{-iHt} for H = Σ c_k P_k."""
    tau = t / n_steps
    for _ in range(n_steps):
        for s, c in zip(pauli_strings, coeffs):
            apply_pauli_rotation(s, c * tau)


def evolve_bitstring(x, pauli_strings, coeffs, t, n_steps):
    """Prepare |x> and apply Trotterised time evolution."""
    for q, bit in enumerate(x):
        if bit == 1:
            qml.PauliX(wires=q)
    trotter_circuit(pauli_strings, coeffs, t, n_steps)


# ─────────────────────────────────────────────────────────────────────────────
# Feature basis and grouping
# For details, see final report: Feature Construction and Measurement Grouping
# ─────────────────────────────────────────────────────────────────────────────

def enumerate_k_local_paulis(k, n):
    """All weight-k Pauli strings on n qubits."""
    strings = []
    for positions in combinations(range(n), k):
        for paulis in product("XYZ", repeat=k):
            s = ["I"] * n
            for pos, p in zip(positions, paulis):
                s[pos] = p
            strings.append("".join(s))
    return strings


def _basis_key(pauli_str):
    """Per-qubit measurement basis: X/Y/Z (treat I as Z)."""
    return tuple("Z" if p == "I" else p for p in pauli_str)


def group_by_basis(feature_strings):
    """basis_key -> [(pauli_index, support_qubits_tuple), ...]"""
    groups = defaultdict(list)
    for j, s in enumerate(feature_strings):
        support = tuple(q for q, p in enumerate(s) if p != "I")
        groups[_basis_key(s)].append((j, support))
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# Quantum pipeline
# For details, see final report: State Preparation and Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────

def make_qnodes(n_qubits, pauli_strings, coeffs, t, n_trotter, n_shots):
    """Build the two QNodes for state prep and basis sampling.

    Returned closures capture the Hamiltonian, so call this once per
    (Hamiltonian, t, n_trotter) configuration.
    """
    dev_evolve = qml.device("lightning.qubit", wires=n_qubits)
    dev_measure = qml.device("lightning.qubit", wires=n_qubits, shots=n_shots)

    @qml.qnode(dev_evolve)
    def evolved_state(x_bits):
        evolve_bitstring(x_bits, pauli_strings, coeffs, t, n_trotter)
        return qml.state()

    @qml.qnode(dev_measure)
    def sample_from_state(state_vec, basis_key):
        qml.StatePrep(state_vec, wires=range(n_qubits))
        for q, b in enumerate(basis_key):
            if b == "X":
                qml.Hadamard(wires=q)
            elif b == "Y":
                qml.adjoint(qml.S)(wires=q)
                qml.Hadamard(wires=q)
        return qml.sample(wires=range(n_qubits))

    return evolved_state, sample_from_state


def compute_phi_shots(x_bits, evolved_state, sample_from_state,
                       basis_groups, m):
    """Trotter once, then sample in every measurement basis."""
    psi = np.asarray(evolved_state(x_bits))
    phi = np.zeros(m)
    for basis_key, members in basis_groups.items():
        samples = np.atleast_2d(sample_from_state(psi, basis_key))
        for j, support in members:
            if not support:
                phi[j] = 1.0
                continue
            bits = samples[:, list(support)].sum(axis=1) & 1
            phi[j] = float((1 - 2 * bits).mean())
    return phi




# ─────────────────────────────────────────────────────────────────────────────
# NN baselines
# For details, see final report: ML Models and LASSO
# ─────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(128, 128, 64), out_dim=1, dropout=0.0):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(model, X_tr_t, y_tr_t, X_te_t, y_te_t, device,
              epochs=300, lr=1e-3, batch_size=64, weight_decay=1e-5,
              verbose=False):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                        batch_size=batch_size, shuffle=True)

    history = {"train_mse": [], "test_mse": []}
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            tr_mse = loss_fn(model(X_tr_t), y_tr_t).item()
            te_mse = loss_fn(model(X_te_t), y_te_t).item()
        history["train_mse"].append(tr_mse)
        history["test_mse"].append(te_mse)

        if verbose and (ep + 1) % 100 == 0:
            print(f"  ep {ep+1:4d}   train MSE = {tr_mse:.5f}   "
                  f"test MSE = {te_mse:.5f}")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end experiment driver
# ─────────────────────────────────────────────────────────────────────────────

def make_sparse_observable(feature_strings, n_obs_terms, rng):
    """Pick a random sparse alpha* over the feature basis.

    Returns (O_sparse, alpha_true, obs_idx, obs_alphas).
    """
    m = len(feature_strings)
    obs_idx = rng.choice(m, n_obs_terms, replace=False)
    obs_alphas = rng.uniform(-1, 1, n_obs_terms)
    obs_strs = [feature_strings[i] for i in obs_idx]

    O_sparse = sum(a * pauli_string_to_matrix(s)
                   for a, s in zip(obs_alphas, obs_strs)).tocsr()

    alpha_true = np.zeros(m)
    alpha_true[obs_idx] = obs_alphas
    return O_sparse, alpha_true, obs_idx, obs_alphas


def run_experiment(*,
                   H_terms,
                   n_qubits=10,
                   t=2.0,
                   n_trotter=20,
                   n_obs_terms=10,
                   k_obs=3,
                   k_observable=None,
                   feature_subset_size=None,
                   basis_subset_size=None,
                   n_samples=500,
                   n_shots=1000,
                   obs_idx=None,
                   obs_alphas=None,
                   seed=23,
                   verbose=False):
    """Run the full pipeline once and return metrics for all four models.

    The pipeline: build a Hamiltonian, draw a random k-local observable O(α*),
    generate (x, y=<x|U†OU|x>) samples by exact diagonalisation, estimate the
    feature vector φ(x) via shot-based measurement in the evolved basis, and
    fit four models — LASSO on φ, linear regression on x, MLP on x, MLP on φ.

    Parameters
    ----------
    H_terms : list
        Pauli terms defining the Hamiltonian, passed to ``build_hamiltonian``.
    n_qubits : int
        Number of qubits in the system.
    t : float
        Evolution time for U = exp(-iHt).
    n_trotter : int
        Number of Trotter steps used by the shot-based feature estimator.
    n_obs_terms : int
        Number of nonzero terms in the random observable α* (ignored when
        ``obs_idx`` / ``obs_alphas`` are supplied).
    n_samples : int
        Number of (x, y) training+test samples to generate.
    n_shots : int
        Number of measurement shots per basis when estimating φ(x).
    k_obs : int
        Locality of the feature basis used by LASSO and NN-on-φ.
    k_observable : int, optional
        Locality of the *unknown* observable O(α*). Defaults to ``k_obs``.
        Set differently from ``k_obs`` to study what happens when the feature
        basis is too narrow (``k_obs < k_observable``) or too wide
        (``k_obs > k_observable``) compared to the truth.
    feature_subset_size : int, optional
        If given, randomly subsample this many features from the full
        weight-``k_obs`` basis. Tests compressed-sensing-style recovery from a
        partial measurement budget. The subset is drawn using the same seed as
        α*, so different seeds give different subsets. (Note: a small subset
        will probably miss the true support and recover nothing — this is
        correct behaviour but a boring plot; see ``basis_subset_size`` for a
        more useful variant.)
    basis_subset_size : int, optional
        If given, keep only this many random measurement bases (out of
        ``|basis_groups|``), but all Paulis inside those bases. This models
        the realistic resource constraint: every measurement basis requires a
        separate circuit run. Mutually exclusive with ``feature_subset_size``.
    obs_idx : np.ndarray, optional
        Indices into the observable basis specifying the support of α*. Pass
        together with ``obs_alphas`` to fix the unknown observable across runs
        of a sweep — otherwise a fresh random α* is drawn each call.
    obs_alphas : np.ndarray, optional
        Coefficients of α* at the positions given by ``obs_idx``.
    seed : int
        Seed for the numpy RNG and torch.
    verbose : bool
        If True, print sample-generation progress and per-epoch NN training
        loss.

    Returns
    -------
    metrics : dict
        Nested dict of per-model metrics plus a few globals (``B_true``,
        ``gen_time``, ``m``, ``alpha_true_l0``).
    alpha_true : np.ndarray
        Length-``m`` vector with the true α* embedded into the feature basis
        (zero on features not present in the observable basis).
    obs_idx : np.ndarray
        The observable support actually used (random or echoed back).
    obs_alphas : np.ndarray
        The observable coefficients actually used.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    if k_observable is None:
        k_observable = k_obs

    # ── Hamiltonian + label-generation machinery ────────────────────────────
    H_sparse, H_pauli_strings, H_coeffs = build_hamiltonian(H_terms, n_qubits)
    eigvals, V = np.linalg.eigh(H_sparse.toarray())
    Vdag = V.conj().T
    phases = np.exp(-1j * t * eigvals)

    # ── Feature basis (used by LASSO / NN-on-phi) ───────────────────────────
    feature_strings = enumerate_k_local_paulis(k_obs, n_qubits)

    if feature_subset_size is not None and basis_subset_size is not None:
        raise ValueError(
            "feature_subset_size and basis_subset_size are mutually exclusive")

    # Optional: random subset of the feature basis (compressed-sensing test).
    if feature_subset_size is not None:
        if feature_subset_size > len(feature_strings):
            raise ValueError(
                f"feature_subset_size={feature_subset_size} > "
                f"|basis|={len(feature_strings)}")
        subset_idx = rng.choice(len(feature_strings),
                                size=feature_subset_size, replace=False)
        feature_strings = [feature_strings[i] for i in sorted(subset_idx)]

    # Optional: random subset of measurement bases.
    # All Paulis inside the kept bases are still measured.
    if basis_subset_size is not None:
        full_groups = group_by_basis(feature_strings)
        basis_keys = list(full_groups.keys())
        if basis_subset_size > len(basis_keys):
            raise ValueError(
                f"basis_subset_size={basis_subset_size} > "
                f"|basis_groups|={len(basis_keys)}")
        kept_idx = rng.choice(len(basis_keys),
                              size=basis_subset_size, replace=False)
        kept_keys = {basis_keys[i] for i in kept_idx}
        # Rebuild feature_strings to contain only Paulis whose basis is kept.
        feature_strings = [feature_strings[j]
                           for key, members in full_groups.items()
                           if key in kept_keys
                           for j, _ in members]

    m = len(feature_strings)

    observable_basis = (feature_strings if k_observable == k_obs
                        else enumerate_k_local_paulis(k_observable, n_qubits))
    m_obs = len(observable_basis)

    if obs_idx is None or obs_alphas is None:
        obs_idx = rng.choice(m_obs, n_obs_terms, replace=False)
        obs_alphas = rng.uniform(-1, 1, n_obs_terms)

    obs_strs = [observable_basis[i] for i in obs_idx]
    O_matrix = sum(a * pauli_string_to_matrix(s)
                   for a, s in zip(obs_alphas, obs_strs)).tocsr()

    feature_index = {s: j for j, s in enumerate(feature_strings)}
    alpha_true = np.zeros(m)
    alpha_true_obs_indices_in_features = []
    for obs_s, a in zip(obs_strs, obs_alphas):
        if obs_s in feature_index:
            j = feature_index[obs_s]
            alpha_true[j] = a
            alpha_true_obs_indices_in_features.append(j)
    obs_idx_in_features = np.array(alpha_true_obs_indices_in_features, dtype=int)

    B_true = float(np.sum(np.abs(obs_alphas)))

    # ── Generate samples ────────────────────────────────────────────────────
    Phi = np.zeros((n_samples, m), dtype=float)
    y = np.zeros(n_samples, dtype=float)
    X = np.zeros((n_samples, n_qubits), dtype=int)

    basis_groups = group_by_basis(feature_strings)
    evolved_state, sample_from_state = make_qnodes(
        n_qubits, H_pauli_strings, H_coeffs, t, n_trotter, n_shots)

    t0 = time.time()
    for i in range(n_samples):
        if verbose and i % 100 == 0 and i > 0:
            print(f"    sample {i}/{n_samples} ({i/(time.time()-t0):.1f} /s)")

        x = rng.integers(0, 2, size=n_qubits)
        X[i] = x

        Phi[i] = compute_phi_shots(x, evolved_state, sample_from_state,
                                   basis_groups, m)

        # Label via ED
        psi0 = reduce(np.kron, (STATE_MAP[int(b)] for b in x))
        psi_t = V @ (phases * (Vdag @ psi0))
        y[i] = np.vdot(psi_t, O_matrix @ psi_t).real

    gen_time = time.time() - t0

    # ── Train/test split shared by all models ───────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42)
    Phi_tr, Phi_te, _, _ = train_test_split(
        Phi, y, test_size=0.2, random_state=42)

    metrics = {"B_true": B_true, "gen_time": gen_time, "m": m,
               "alpha_true_l0": int((alpha_true != 0).sum())}

    # ── LASSO on φ(x) ───────────────────────────────────────────────────────
    lasso = LassoCV(cv=5, max_iter=10_000,
                    alphas=np.logspace(-4, 0, 40), n_jobs=-1)
    lasso.fit(Phi_tr, y_tr)
    yp_tr = lasso.predict(Phi_tr)
    yp_te = lasso.predict(Phi_te)
    recovered = np.where(lasso.coef_ != 0)[0]
    metrics["lasso"] = {
        "alpha_cv": float(lasso.alpha_),
        "w_l1": float(np.abs(lasso.coef_).sum()),
        "train_mse": float(mean_squared_error(y_tr, yp_tr)),
        "test_mse": float(mean_squared_error(y_te, yp_te)),
        "train_r2": float(r2_score(y_tr, yp_tr)),
        "test_r2": float(r2_score(y_te, yp_te)),
        # support_tp counts true α* features found in the LASSO support,
        # but only among those that exist in the (possibly smaller) feature basis.
        "support_tp": int(len(set(recovered) & set(obs_idx_in_features))),
        "support_size": int(len(recovered)),
        "obs_terms_in_features": int(len(obs_idx_in_features)),
        "obs_terms_total": int(len(obs_idx)),
        "coef_l2_err": float(np.linalg.norm(lasso.coef_ - alpha_true)),
    }

    # ── Linear on raw x ─────────────────────────────────────────────────────
    lin = LinearRegression().fit(X_tr, y_tr)
    yp_tr = lin.predict(X_tr)
    yp_te = lin.predict(X_te)
    metrics["linear_x"] = {
        "train_mse": float(mean_squared_error(y_tr, yp_tr)),
        "test_mse": float(mean_squared_error(y_te, yp_te)),
        "train_r2": float(r2_score(y_tr, yp_tr)),
        "test_r2": float(r2_score(y_te, yp_te)),
    }

    # ── NN on raw x ─────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tr_s = 2.0 * X_tr.astype(np.float32) - 1.0
    X_te_s = 2.0 * X_te.astype(np.float32) - 1.0

    X_tr_t = torch.tensor(X_tr_s, dtype=torch.float32, device=device)
    X_te_t = torch.tensor(X_te_s, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr.astype(np.float32),
                          dtype=torch.float32, device=device)
    y_te_t = torch.tensor(y_te.astype(np.float32),
                          dtype=torch.float32, device=device)

    nn_x = MLP(in_dim=n_qubits, hidden=(128, 128, 64))
    train_mlp(nn_x, X_tr_t, y_tr_t, X_te_t, y_te_t, device,
              epochs=300, lr=1e-3, weight_decay=1e-5, verbose=verbose)
    nn_x.eval()
    with torch.no_grad():
        yp_tr_nnx = nn_x(X_tr_t).cpu().numpy()
        yp_te_nnx = nn_x(X_te_t).cpu().numpy()
    metrics["nn_x"] = {
        "train_mse": float(mean_squared_error(y_tr, yp_tr_nnx)),
        "test_mse": float(mean_squared_error(y_te, yp_te_nnx)),
        "train_r2": float(r2_score(y_tr, yp_tr_nnx)),
        "test_r2": float(r2_score(y_te, yp_te_nnx)),
    }

    # ── NN on φ(x) ──────────────────────────────────────────────────────────
    scaler = StandardScaler().fit(Phi_tr)
    Phi_tr_s = scaler.transform(Phi_tr).astype(np.float32)
    Phi_te_s = scaler.transform(Phi_te).astype(np.float32)
    Phi_tr_t = torch.tensor(Phi_tr_s, dtype=torch.float32, device=device)
    Phi_te_t = torch.tensor(Phi_te_s, dtype=torch.float32, device=device)

    nn_phi = MLP(in_dim=m, hidden=(128, 128, 64), dropout=0.1)
    train_mlp(nn_phi, Phi_tr_t, y_tr_t, Phi_te_t, y_te_t, device,
              epochs=300, lr=1e-3, weight_decay=1e-4, verbose=verbose)
    nn_phi.eval()
    with torch.no_grad():
        yp_tr_nnp = nn_phi(Phi_tr_t).cpu().numpy()
        yp_te_nnp = nn_phi(Phi_te_t).cpu().numpy()
    metrics["nn_phi"] = {
        "train_mse": float(mean_squared_error(y_tr, yp_tr_nnp)),
        "test_mse": float(mean_squared_error(y_te, yp_te_nnp)),
        "train_r2": float(r2_score(y_tr, yp_tr_nnp)),
        "test_r2": float(r2_score(y_te, yp_te_nnp)),
    }

    return metrics, alpha_true, obs_idx, obs_alphas