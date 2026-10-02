"""Fidelity Projected Least Squares (FPLS): the algorithm of the paper.

    Fast and Sure-ious Quantum Process Tomography
    A. Afham, S. Sen, M. Tomamichel

A channel Phi: L(A) -> L(B) is estimated from single-copy local Pauli
measurements on copies of its normalised Choi state rho = C(Phi), in three
steps:

    1. least squares          frequencies  ->  rho_LS   (Hermitian, unit trace,
                                                         generally indefinite)
    2. density estimate       rho_LS       ->  rho_hat  (PSD, unit trace),
                              by thresholding the spectrum at tau = beta_N
    3. TP regularization      rho_hat      ->  C(Phi_hat) = N_A(rho_hat),
                              the partial normalization

Step 3 is the paper's subject: N_A is the *exact* projection onto the Choi
states of channels in fidelity (Theorem 1), it preserves the Choi rank, and it
is a single congruence -- one d_A x d_A eigendecomposition, no iteration.

Everything here is plain numpy; no part of the pipeline calls a solver.
"""
import hashlib

import numpy as np
import scipy.linalg as sla

__all__ = [
    "PAPER_SEED", "EPS_ZERO", "DELTA",
    "make_rng", "paper_rng",
    "partial_trace_B", "matrix_sqrt", "matrix_inv_sqrt", "choi_rank",
    "fidelity", "infidelity", "trace_distance", "fidelity_to_channels",
    "fidelity_projection", "choi_to_kraus", "kraus_from_eig", "kraus_to_choi",
    "normalise_kraus", "is_channel",
    "bernstein_radius", "threshold_density_estimate",
    "pauli_probs", "sample_frequencies", "ls_estimate", "simulate_ls_estimate",
    "qft_choi", "haar_unitary_choi", "rank_two_channel",
    "amplitude_damping_choi", "werner_holevo_choi", "depolarizing_choi",
    "random_channel_choi",
    "fpls", "lmin_density_estimate",
    "use_style", "save_figure",
]

# An eigenvalue counts as zero iff it is <= EPS_ZERO. Every rank quoted in the
# paper uses this cutoff, so the companion uses it too.
EPS_ZERO = 1e-10
# Confidence of the concentration event, fixed throughout the paper.
DELTA = 0.05
# Frozen label: the paper's instances and measurement streams were drawn from
# numpy seed sequences built on this integer. Changing it resamples every
# cached number, so it is kept verbatim rather than folded into make_rng.
PAPER_SEED = 20260817


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def make_rng(*parts):
    """Generator seeded by the sha256 of a descriptive tag.

    Python's builtin hash is salted per process, so it cannot be used: two runs
    of the same script would draw different instances. sha256 of the repr makes
    the draw a function of the tag alone, which is what "regenerates bit for
    bit" means below.
    """
    tag = "|".join(repr(p) for p in parts).encode()
    seed = int.from_bytes(hashlib.sha256(tag).digest()[:8], "big")
    return np.random.default_rng(seed)


def paper_rng(*ints):
    """The paper's own generator: np.random.default_rng([i0, i1, ...]).

    Kept because the cached sweeps were drawn from it; use it to rebuild an
    instance or a measurement stream that a cached number refers to, and
    make_rng for anything new.
    """
    return np.random.default_rng(list(ints))


# ---------------------------------------------------------------------------
# Linear algebra
# ---------------------------------------------------------------------------
def partial_trace_B(P, dA, dB):
    """Tr_B[P] for P on A (x) B, shape (dA*dB, dA*dB) -> (dA, dA)."""
    return np.trace(P.reshape(dA, dB, dA, dB), axis1=1, axis2=3)


def matrix_sqrt(P):
    """PSD square root, negative eigenvalues clipped to zero."""
    D, V = sla.eigh((P + P.conj().T) / 2)
    return (V * np.sqrt(np.maximum(D, 0.0))) @ V.conj().T


def matrix_inv_sqrt(P, tol=1e-12):
    """Inverse square root on the support: eigenvalues <= tol are dropped.

    This is the operator rho_A^{-1/2} of the partial normalization; `tol`
    implements the support convention of Remark 2 for a singular marginal.
    """
    D, V = sla.eigh(P)
    D = np.maximum(D, 0.0)
    D_inv = np.zeros_like(D)
    nz = D > tol
    D_inv[nz] = 1.0 / np.sqrt(D[nz])
    return (V * D_inv) @ V.conj().T


def choi_rank(rho, tol=EPS_ZERO):
    """Number of eigenvalues above the zero cutoff."""
    return int((np.linalg.eigvalsh(rho) > tol).sum())


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------
def fidelity(P, Q):
    """Root fidelity F(P, Q) = || sqrt(P) sqrt(Q) ||_1.

    Evaluated as the sum of the singular values of sqrt(P) sqrt(Q), not as
    Tr sqrt(sqrt(P) Q sqrt(P)): the latter forms A A^dag and loses half the
    significant digits, which at large shot counts is the difference between an
    infidelity of 1e-12 and an infidelity reported as exactly 0.
    """
    return float(np.sum(sla.svdvals(matrix_sqrt(P) @ matrix_sqrt(Q))))


def infidelity(P, Q, tol=1e-8):
    """1 - F(P, Q) for two states, clipped at 0 (F can exceed 1 by ~1e-15).

    Both arguments must have unit trace: otherwise 1 - F is not an
    infidelity, and the clip would hide the error.
    """
    for M in (P, Q):
        assert abs(np.trace(M).real - 1.0) < tol, f"infidelity: trace {np.trace(M).real:.6g} != 1"
    return float(max(0.0, 1.0 - fidelity(P, Q)))


def trace_distance(P, Q):
    """(1/2) ||P - Q||_1."""
    return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(P - Q))))


def fidelity_to_channels(rho, dA):
    """max over channels sigma of F(rho, sigma) = Tr sqrt(rho_A) / sqrt(d_A).

    The optimal value of Theorem 1: the fidelity of rho to the whole set of
    Choi states depends on the marginal alone, so it costs one d_A x d_A
    eigendecomposition rather than an optimization.
    """
    rho_A = partial_trace_B(rho, dA, rho.shape[0] // dA)
    return float(np.sum(np.sqrt(np.maximum(np.linalg.eigvalsh(rho_A), 0.0))) / np.sqrt(dA))


# ---------------------------------------------------------------------------
# The fidelity projection (Theorem 1) and its Kraus form (Theorem 5)
# ---------------------------------------------------------------------------
def fidelity_projection(rho, dA):
    """N_A(rho) = (1/d_A) (rho_A^{-1/2} otimes I_B) rho (rho_A^{-1/2} otimes I_B).

    The partial normalization, which Theorem 1 identifies as the exact fidelity
    (equivalently Bures, equivalently purified-distance) projection of rho onto
    the Choi states of channels. Being a congruence by an invertible operator,
    it preserves the rank of rho.
    """
    dB = rho.shape[0] // dA
    M = np.kron(matrix_inv_sqrt(partial_trace_B(rho, dA, dB)), np.eye(dB))
    return (M @ rho @ M.conj().T) / dA


def kraus_from_eig(lam, vecs, dA, dB, tol=EPS_ZERO):
    """Kraus operators from the eigenpairs of a normalised Choi state.

    Eigenvector |v_i> of eigenvalue lam_i, reshaped to a dA x dB array V_i,
    gives K_i = sqrt(dA lam_i) V_i^T of shape (dB, dA). Only the eigenvalues
    above `tol` are returned, so a rank-r Choi state yields r operators.
    """
    keep = np.argsort(lam)[::-1]                   # descending
    return [np.sqrt(dA * lam[i]) * vecs[:, i].reshape(dA, dB).T for i in keep if lam[i] > tol]


def choi_to_kraus(rho, dA, dB, tol=EPS_ZERO):
    """Kraus operators of the CP map with normalised Choi state rho."""
    lam, vecs = np.linalg.eigh(rho)
    return kraus_from_eig(lam, vecs, dA, dB, tol)


def kraus_to_choi(Ks, dA, dB):
    """Normalised Choi state of the CP map with Kraus operators Ks."""
    omega = np.zeros(dA * dA, dtype=complex)
    omega[:: dA + 1] = 1.0 / np.sqrt(dA)
    rho = np.zeros((dA * dB, dA * dB), dtype=complex)
    for K in Ks:
        v = np.kron(np.eye(dA), K) @ omega
        rho += np.outer(v, v.conj())
    return rho


def normalise_kraus(Ks):
    """K_i -> K_i R^{-1/2} with R = sum_i K_i^dag K_i (Theorem 5).

    The fidelity projection in Kraus form: it touches only the d_A x d_A defect
    R, so it costs O(r d_A^2 d_B + d_A^3) operations and O(r d_A d_B + d_A^2)
    memory, and never forms a d_AB x d_AB matrix. The output is trace
    preserving identically, sum_i K_i'^dag K_i' = I_A.
    """
    R = sum(K.conj().T @ K for K in Ks)
    R_ihf = matrix_inv_sqrt(R)
    return [K @ R_ihf for K in Ks]


def is_channel(rho, dA, tol=1e-9):
    """PSD, unit trace, and maximally mixed A-marginal."""
    dB = rho.shape[0] // dA
    lam = np.linalg.eigvalsh(rho)
    return bool(lam.min() > -tol
                and abs(np.trace(rho).real - 1.0) < tol
                and np.linalg.norm(partial_trace_B(rho, dA, dB) - np.eye(dA) / dA) < tol)


# ---------------------------------------------------------------------------
# The density estimate: Bernstein threshold + Algorithm 3
# ---------------------------------------------------------------------------
def bernstein_radius(d, N, delta=DELTA):
    """beta_N = sqrt(8 g(d) log(d/delta) / (3N)), g(d) = 3^n = d^(log2 3).

    The operator-norm concentration radius of the local-Pauli least-squares
    estimator: Pr[ ||rho_LS - rho||_inf >= beta_N ] <= delta. Thresholding the
    density estimate here, above the noise rather than below it, is what makes
    exact rank recovery provable past a finite onset (Corollary 8).
    """
    g = 3.0 ** np.log2(d)
    return float(np.sqrt(8.0 * g * np.log(d / delta) / (3.0 * N)))


def threshold_density_estimate(H, tau, return_eig=False):
    """Algorithm 3: nearest-unit-trace PSD matrix after thresholding at tau.

    Eigenvalues at or below tau are annihilated, survivors are boosted by tau,
    and the trace is restored: by a uniform water-filling subtraction if the
    boosted spectrum oversums, else by rescuing the raw eigenvalues top-down.
    tau = 0 is the Frobenius projection onto the density matrices.
    With return_eig=True the eigenpairs (mu, vecs) of the estimate are returned
    as well, which is all the Kraus form of the projection needs.
    """
    H = (H + H.conj().T) / 2
    lam, vecs = np.linalg.eigh(H)
    d = len(lam)
    mu = np.where(lam <= tau, 0.0, lam + tau)

    if mu.sum() >= 1.0:
        # supernormalized: subtract a common level from every survivor
        srt = np.sort(mu)[::-1]
        cssv = np.cumsum(srt)
        lev = (cssv - 1.0) / (np.arange(d) + 1)
        x0 = lev[np.nonzero(srt > lev)[0][-1]]
        mu = np.maximum(mu - x0, 0.0)
    else:
        # subnormalized: fill from the top until the trace budget is spent
        j = None
        for k in range(d):
            hi = lam[k:].sum() + (d - k) * tau
            lo = lam[k + 1:].sum() + (d - k - 1) * tau if k < d - 1 else 0.0
            if lo < 1.0 <= hi:
                j = k
                break
        if j is None:                                   # fall back to tau = 0
            raw = np.maximum(lam, 0.0)
            srt = np.sort(raw)[::-1]
            cssv = np.cumsum(srt)
            lev = (cssv - 1.0) / (np.arange(d) + 1)
            x0 = lev[np.nonzero(srt > lev)[0][-1]]
            mu = np.maximum(lam - x0, 0.0)
        else:
            mu = np.zeros(d)
            if j + 1 < d:
                mu[j + 1:] = lam[j + 1:] + tau
            mu[j] = max(1.0 - mu.sum(), 0.0)

    mu = np.maximum(mu, 0.0)
    if mu.sum() > 1e-14:
        mu = mu / mu.sum()
    rho_hat = (vecs * mu) @ vecs.conj().T
    return (rho_hat, mu, vecs) if return_eig else rho_hat


def lmin_density_estimate(H):
    """The PLS-QPT rule, tau = max(0, -lambda_min(H)).

    Surawy-Stepney et al. write tau = -lambda_min; the clamp matters only when
    H is already positive definite, where the rule read literally would give a
    negative threshold and alter an estimate that needs no correction.
    """
    H = (H + H.conj().T) / 2
    lam_min = float(np.linalg.eigh(H)[0].min())
    return threshold_density_estimate(H, max(0.0, -lam_min))


# ---------------------------------------------------------------------------
# Local Pauli measurements and the least-squares estimate
# ---------------------------------------------------------------------------
_PAULI_VECS = [np.array(v, dtype=complex) for v in
               ([1, 0], [0, 1], [1, 1], [1, -1], [1, 1j], [1, -1j])]
_PAULI_VECS = [v / np.sqrt(np.vdot(v, v).real) for v in _PAULI_VECS]
# Q[i] = vec(|v_i><v_i|)/3: the six single-qubit Pauli POVM elements, each of
# trace 1/3 -- equal traces, which is what makes the LS estimate unit trace.
_Q = np.array([np.kron(v, v.conj()) for v in _PAULI_VECS]) / 3.0
_PAULIS = np.column_stack([m.flatten() for m in
                           (np.eye(2, dtype=complex),
                            np.array([[0, 1], [1, 0]], dtype=complex),
                            np.array([[0, -1j], [1j, 0]], dtype=complex),
                            np.array([[1, 0], [0, -1]], dtype=complex))])
_W_INV = np.linalg.pinv(_Q @ _PAULIS).T


def pauli_probs(rho):
    """The 6^n outcome distribution of rho under local Pauli measurements."""
    n = int(np.log2(rho.shape[0]))
    R = np.reshape(rho, [2] * (2 * n))
    R = np.transpose(R, [i for q in range(n) for i in (q, q + n)])
    R = np.reshape(R, [4] * n)
    for q in range(n):
        R = np.moveaxis(R, q, 0)
        rest = R.shape[1:]
        R = (_Q @ R.reshape(4, -1)).reshape((6,) + rest)
        R = np.moveaxis(R, 0, q)
    p = np.maximum(np.real(R.flatten()), 0.0)
    return p / p.sum()


def sample_frequencies(p, N, rng):
    """Exactly N i.i.d. shots: a multinomial draw over the 6^n outcomes."""
    return rng.multinomial(int(N), p / p.sum()) / int(N)


def ls_estimate(freqs, n):
    """Least-squares (linear inversion) estimate from outcome frequencies.

    Closed form for the local Pauli POVM: rho_LS = sum_(o,s) c_(o,s) (x)_i
    (3 |o_i, s_i><o_i, s_i| - I). Hermitian and of unit trace by construction,
    but indefinite, which is why step 2 exists.
    """
    c = freqs.reshape((6,) * n)
    for _ in range(n):
        c = np.moveaxis(np.dot(c, _W_INV), -1, 0)
    mats = c.flatten().reshape(4 ** n, 1, 1)
    for k in range(n):
        dim = 2 ** k
        out = np.zeros((mats.shape[0] // 4, dim * 2, dim * 2), dtype=complex)
        for i in range(out.shape[0]):
            m_i, m_x, m_y, m_z = mats[4 * i], mats[4 * i + 1], mats[4 * i + 2], mats[4 * i + 3]
            out[i, :dim, :dim] = m_i + m_z
            out[i, :dim, dim:] = m_x - 1j * m_y
            out[i, dim:, :dim] = m_x + 1j * m_y
            out[i, dim:, dim:] = m_i - m_z
        mats = out
    return mats[0]


def simulate_ls_estimate(rho, N, rng):
    """N simulated shots of local Pauli measurements on rho, then step 1."""
    n = int(np.log2(rho.shape[0]))
    return ls_estimate(sample_frequencies(pauli_probs(rho), N, rng), n)


# ---------------------------------------------------------------------------
# Channels (Choi states, normalised to unit trace)
# ---------------------------------------------------------------------------
def _vec(M, dA):
    """Choi vector of a single Kraus operator M, A-first convention."""
    omega = np.zeros(dA * dA, dtype=complex)
    omega[:: dA + 1] = 1.0 / np.sqrt(dA)
    return np.kron(np.eye(dA), M) @ omega


def qft_choi(nq):
    """Choi state of the n-qubit quantum Fourier transform. Rank 1."""
    d = 2 ** nq
    idx = np.arange(d)
    U = np.exp(2j * np.pi / d) ** np.outer(idx, idx) / np.sqrt(d)
    v = _vec(U, d)
    rho = np.outer(v, v.conj())
    return rho / np.trace(rho)


def haar_unitary_choi(d, rng):
    """Choi state of a Haar-random unitary channel. Rank 1.

    Haar via QR of a Ginibre matrix with the phases of R's diagonal divided
    out (column scaling). The paper's instances are frozen with this sampler,
    so it is kept rather than replaced by the polar-factor route.
    """
    G = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    U, R = np.linalg.qr(G)
    U = U * (np.diag(R) / np.abs(np.diag(R)))
    v = U.flatten() / np.sqrt(d)
    return np.outer(v, v.conj())


def rank_two_channel(nq, rng, eps_mix=0.5):
    """The headline instance: (1-eps) QFT + eps (one Haar unitary).

    A mixed-unitary channel of Choi rank exactly 2, with both nonzero Choi
    eigenvalues near 1/2 at eps = 1/2, and no noise floor -- so "recovers the
    rank" is literal rather than a statement about a cutoff.
    """
    d = 2 ** nq
    rho = (1 - eps_mix) * qft_choi(nq) + eps_mix * haar_unitary_choi(d, rng)
    return rho / np.trace(rho).real


def amplitude_damping_choi(nq, k, rng):
    """Independent amplitude damping on k of nq qubits. Choi rank 2^k.

    Damping strengths are drawn uniformly in [0.15, 0.45]; the Choi eigenvalues
    are the products of 1 - phi_j/2 and phi_j/2 over the damped qubits. The
    only non-unital family used in the paper.
    """
    d = 2 ** nq
    phi = rng.uniform(0.15, 0.45, size=k)
    K = [[np.array([[1, 0], [0, np.sqrt(1 - p)]], complex),
          np.array([[0, np.sqrt(p)], [0, 0]], complex)] for p in phi]
    J = np.zeros((d * d, d * d), dtype=complex)
    for s in range(2 ** k):
        M = np.eye(1)
        for q in range(nq):
            M = np.kron(M, K[q][(s >> q) & 1] if q < k else np.eye(2))
        v = _vec(M, d)
        J += np.outer(v, v.conj())
    return J


def werner_holevo_choi(nq):
    """rho -> (Tr[rho] I - rho^T)/(d-1): flat spectrum of rank d(d-1)/2."""
    d = 2 ** nq
    S = np.zeros((d * d, d * d))
    for i in range(d):
        for j in range(d):
            S[i * d + j, j * d + i] = 1.0
    return (np.eye(d * d) - S).astype(complex) / (d * (d - 1))


def depolarizing_choi(nq, p, unitary=None):
    """(1-p) x a unitary channel + p x the completely depolarizing channel.

    Full Choi rank for every p > 0: one eigenvalue (1-p) + p/d^2 and d^2 - 1
    equal to p/d^2. unitary="qft" precedes the noise with the QFT.
    """
    d = 2 ** nq
    if unitary == "qft":
        J = qft_choi(nq)
    else:
        v = _vec(np.eye(d), d)
        J = np.outer(v, v.conj())
    return (1 - p) * J + p * np.eye(d * d) / (d * d)


def local_depolarizing_choi(nq, p):
    """Independent depolarizing at p on each of the nq qubits. Full Choi rank.

    Each qubit's Kraus set is {I, X, Y, Z} with weights (1 - 3p/4, p/4, p/4,
    p/4), so the Choi eigenvalues are products of those: a tiered tail.
    """
    d = 2 ** nq
    P = [np.eye(2), np.array([[0, 1], [1, 0]], complex),
         np.array([[0, -1j], [1j, 0]]), np.diag([1.0, -1.0]).astype(complex)]
    w1 = np.array([1 - 3 * p / 4, p / 4, p / 4, p / 4])
    J = np.zeros((d * d, d * d), dtype=complex)
    for sel in range(4 ** nq):
        idx = [(sel >> (2 * q)) & 3 for q in range(nq)]
        M, w = np.eye(1), 1.0
        for q in range(nq):
            M, w = np.kron(M, P[idx[q]]), w * w1[idx[q]]
        v = _vec(M, d)
        J += w * np.outer(v, v.conj())
    return J


def qft_bcsz_choi(nq, p, rng):
    """(1 - p) x the QFT + p x a random channel of full Choi rank (BCSZ).

    The random channel is the partial normalization of a normalised full-rank
    Wishart matrix, so the mixture has full Choi rank with a spread tail.
    """
    d = 2 ** nq
    G = (rng.standard_normal((d * d, d * d)) + 1j * rng.standard_normal((d * d, d * d))) / np.sqrt(2)
    W = G @ G.conj().T
    W /= np.trace(W).real
    return (1 - p) * qft_choi(nq) + p * fidelity_projection(W, d)


def random_channel_choi(dA, dB, r, rng):
    """BCSZ: a rank-r Wishart matrix normalised by its input marginal.

    The standard sampler for random channels -- itself an application of the
    partial normalization, which is one of the places the formula already
    appears in the literature.
    """
    X = rng.standard_normal((dA * dB, r)) + 1j * rng.standard_normal((dA * dB, r))
    return fidelity_projection(X @ X.conj().T, dA)


# ---------------------------------------------------------------------------
# The protocol
# ---------------------------------------------------------------------------
def fpls(rho_true, N, rng, dA=None, c=1.0, delta=DELTA, dense=True):
    """Run FPLS-QPT end to end on simulated data and score the estimate.

    c scales the threshold, tau = c beta_N: c = 1 is the rule the guarantee is
    proved for, c = 1/2 the heuristic the paper's figures use.

    The TP regularization is done on Kraus operators (Theorem 5): the
    eigenpairs of the density estimate are reshaped into Kraus operators and
    normalised, K_i -> K_i R^{-1/2}, which never forms a d_AB x d_AB matrix.
    The returned "kraus" list is the estimate in that form, one operator per
    unit of Choi rank. The dense "estimate" used for scoring is, by default,
    the dense congruence of Theorem 1 applied to the same density estimate (the
    form the cached sweeps were computed with, so the numbers match them to
    the bit); with dense=False it is assembled from the Kraus operators
    instead. The two agree to machine precision.

    Returns a dict: the channel estimate and the quantities the figures plot.
    """
    d = rho_true.shape[0]
    dA = int(np.sqrt(d)) if dA is None else dA
    dB = d // dA
    beta = bernstein_radius(d, N, delta)
    rho_ls = simulate_ls_estimate(rho_true, N, rng)
    rho_hat, mu, vecs = threshold_density_estimate(rho_ls, tau=c * beta, return_eig=True)
    Ks = normalise_kraus(kraus_from_eig(mu, vecs, dA, dB))
    est = fidelity_projection(rho_hat, dA) if dense else kraus_to_choi(Ks, dA, dB)
    return {"estimate": est, "kraus": Ks, "beta_N": beta,
            "noise": float(np.abs(np.linalg.eigvalsh(rho_ls - rho_true)).max()),
            "rank": choi_rank(est), "infidelity": infidelity(rho_true, est),
            "trace_distance": trace_distance(rho_true, est),
            "bytes": 16 * sum(K.size for K in Ks)}


# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
OURS = "#004488"      # this work
HIP = "#BB5566"       # PLS (baseline)
SDP = "#333333"       # diamond-norm SDP (charcoal, as in Fig. 2 of the paper)
GREY = "#666666"

_STYLE = {
    # The paper's figures use matplotlib's built-in ggplot style unmodified
    # (Code/experiments/plotstyle.py); only output settings and the math font
    # are overridden, so the companion's redraws match the article's look.
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "mathtext.fontset": "cm",                # Computer Modern math, as in the paper
    "pdf.fonttype": 42, "ps.fonttype": 42,   # Type 42: arXiv rejects Type 3
}



def use_style():
    """Apply the paper's matplotlib style (ggplot + _STYLE) and return pyplot."""
    import logging
    import matplotlib.pyplot as plt
    plt.style.use("ggplot")
    plt.rcParams.update(_STYLE)
    # Embedding the Computer Modern fonts as Type 42 makes fontTools warn about their
    # (harmless) 1990s file timestamps on every PDF written; silence just that warning.
    logging.getLogger("fontTools.ttLib.tables._h_e_a_d").setLevel(logging.ERROR)
    return plt


def save_figure(fig, name, outdir=None):
    """Write <name>.pdf and <name>.png; the pdf is what the paper includes."""
    from pathlib import Path
    outdir = Path(__file__).resolve().parent / "figures" if outdir is None else Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{name}.{ext}")
    return outdir / f"{name}.pdf"
