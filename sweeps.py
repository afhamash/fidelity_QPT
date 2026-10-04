"""Run the paper's sweeps from scratch, at whatever size you can afford.

Every function here simulates measurements, forms the least-squares estimate,
thresholds it and applies the fidelity projection, exactly as `fpls` does, and
returns the same dict shape as the corresponding `data/*.json` file. So

    data = sweep_headline(nq=2, shots=SHOTS, trials=3)
    fig_headline(data=data)

draws the figure from data computed in the cell, and `fig_headline()` with no
argument still draws it from the cached sweep.

Cost. One trial at one shot count costs a 4^nq-dimensional eigendecomposition
plus the sampling of N outcomes, and the least-squares stage materialises the
6^nq outcome distribution, so memory grows as 6^nq and the paper's nq = 4 needs
a few seconds per trial. The defaults in the notebook are small on purpose; the
paper's parameters are stated next to them and are what `data/*.json` holds.

The PLS baseline. PLS-QPT of Surawy-Stepney, Kahn, Kueng and Guta is run with
its authors' own code, shipped unmodified in hip/ and driven by `pls.py` at the
settings of their simulation driver: the defaults where its cost is reported
(Figs. 2 left and 4, the time column of Table 1), and a tight fixed tolerance
where its accuracy and rank are reported (Figs. 2 right, 3, 5, 6, 7). Every
sweep computes the PLS arm live, on the same least-squares estimate as ours,
and `have_hip()` reports True. If hip/ cannot be imported, that arm is read
from `data/*.json` and merged in when its dimension and shot grid match what
you asked for, and otherwise dropped, so a figure never mixes a live curve with
a cached one computed at another size.
Everything this work contributes -- the thresholded density estimate and the
fidelity projection -- is always computed live.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

from fpls import (partial_trace_B, local_depolarizing_choi, qft_bcsz_choi, fidelity, DELTA, PAPER_SEED, amplitude_damping_choi, bernstein_radius,
                  choi_rank, choi_to_kraus, depolarizing_choi, fidelity_projection,
                  fidelity_to_channels, infidelity, kraus_to_choi,
                  lmin_density_estimate, normalise_kraus, paper_rng, qft_choi,
                  rank_two_channel, random_channel_choi, simulate_ls_estimate,
                  threshold_density_estimate, trace_distance, werner_holevo_choi)

DATA = Path(__file__).resolve().parent / "data"

# The paper's shot grid: twelve points, three per decade, 1e4 to 1e10.
PAPER_SHOTS = [1e4, 1e5, 10 ** 5.5, 1e6, 10 ** 6.5, 1e7,
               10 ** 7.5, 1e8, 10 ** 8.5, 1e9, 10 ** 9.5, 1e10]

__all__ = ["PAPER_SHOTS", "HIP_URL", "HIP_FILE", "have_hip", "pls_estimate", "hip_regularize", "channel",
           "sweep_headline", "sweep_accuracy", "sweep_spectra", "sweep_thresholds",
           "sweep_robustness", "sweep_cost_ladder", "sweep_hip_iterations",
           "sweep_table", "sweep_contraction", "check_proj_TP", "check_diamond_sdp"]


# ---------------------------------------------------------------------------
# the PLS baseline: the authors' code in hip/, run by pls.py
# ---------------------------------------------------------------------------
_PLS = None
_PLS_TRIED = False

HIP_URL = "https://github.com/Hannoskaj/Hyperplane_Intersection_Projection"
HIP_FILE = Path(__file__).resolve().parent / "hip"


def have_hip():
    """True if the authors' implementation in hip/ is importable."""
    global _PLS, _PLS_TRIED
    if not _PLS_TRIED:
        _PLS_TRIED = True
        try:
            import pls
            _PLS = pls
        except Exception:                    # a broken copy is not an error here
            _PLS = None
    return _PLS is not None


def _pls():
    if not have_hip():
        raise RuntimeError(f"the PLS baseline is not importable; see {HIP_FILE}")
    return _PLS


def check_proj_TP(rng=None, verbose=True):
    """Our projection onto the TP subspace must be the authors' one.

    Theirs assumes d_A = d_B and takes the two systems in the opposite order to
    ours, so on square dimensions ours must equal theirs conjugated by the swap
    of the two systems; on rectangular ones its output must have marginal
    exactly I_A/d_A.
    """
    pls = _pls()
    theirs = pls._proj.proj_TP                 # their routine, never replaced
    rng = np.random.default_rng(0) if rng is None else rng
    swap = lambda M, d: M.reshape(d, d, d, d).transpose(1, 0, 3, 2).reshape(d * d, d * d)
    worst_sq = worst_tp = 0.0
    for d in (2, 4, 8):
        for _ in range(10):
            H = rng.standard_normal((d * d, d * d)) + 1j * rng.standard_normal((d * d, d * d))
            H = (H + H.conj().T) / 2
            H /= np.trace(H).real
            worst_sq = max(worst_sq, np.abs(swap(theirs(swap(H, d)), d) - pls.proj_TP_for(d, d)(H)).max())
    for dA, dB in ((2, 4), (4, 2), (4, 8), (2, 8)):
        for _ in range(10):
            n = dA * dB
            H = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
            H = (H + H.conj().T) / 2
            H /= np.trace(H).real
            marg = np.trace(pls.proj_TP_for(dA, dB)(H).reshape(dA, dB, dA, dB), axis1=1, axis2=3)
            worst_tp = max(worst_tp, np.abs(marg - np.eye(dA) / dA).max())
    if verbose:
        print(f"  vs the authors' proj_TP, square:  max|diff|    = {worst_sq:.2e}")
        print(f"  marginal of the output, rectangular: max|err| = {worst_tp:.2e}")
    assert worst_sq < 1e-12 and worst_tp < 1e-12
    return worst_sq, worst_tp


def pls_estimate(rho_ls, dA, dB, tight=True):
    """PLS-QPT end to end from a least-squares estimate: the authors' threshold,
    HIP and final mixing. tight=True is the paper's setting for accuracy and
    rank, tight=False the authors' default, used for cost. Returns (rho, info).
    """
    pls = _pls()
    return pls.pls_tight(rho_ls, dA, dB) if tight else pls.pls_default(rho_ls, dA, dB)


def hip_regularize(rho_hat, dA, dB, ls_least_ev=0.0, tight=False):
    """HIP and the final mixing on a given density estimate. Returns (rho, info).

    tight=False: the authors' default, with the stopping tolerance set from
    `ls_least_ev`, the least eigenvalue of the least-squares estimate (the
    paper's cost measurements); tight=True: the tight fixed tolerance.
    """
    pls = _pls()
    if tight:
        return pls.hip_tight(rho_hat, dA, dB)
    return pls.hip_default(rho_hat, ls_least_ev, dA, dB)


# ---------------------------------------------------------------------------
# channels, by the names the paper uses
# ---------------------------------------------------------------------------
def channel(name, nq, seed=PAPER_SEED):
    """One of the paper's test channels, as a Choi state of dimension 4^nq.

    "rank2"            the headline channel: an equal mixture of the quantum
                       Fourier transform with one Haar-random unitary, Choi
                       rank 2 with both eigenvalues near 1/2
    "amp_damping_rK"   independent amplitude damping on K of the nq qubits,
                       Choi rank 2^K
    "werner_holevo"    Choi rank d_A(d_A - 1)/2 with a flat spectrum
    "depolarizing_pP"  global depolarizing at p = P/100, full Choi rank
    "qft_depol_pP"     the QFT followed by global depolarizing at p = P/100
    "local_depol_pP"   independent depolarizing at p = P/100 on every qubit,
                       full Choi rank with a tiered tail
    "qft_bcsz_pP"      (1 - p) QFT + p (a random full-rank channel), p = P/100,
                       full Choi rank with a spread tail
    "logrank"          a random channel of Choi rank log2(d_AB)
    """
    rng = paper_rng(seed, 1)
    if name == "rank2":
        return rank_two_channel(nq, rng)
    # the draws and the trace normalization below are the paper's own, so that
    # the instances, and the measurement records sampled from them, are bit for
    # bit those of data/*.json
    if name.startswith("amp_damping_r"):
        k = int(name.split("_r")[1])
        rho = amplitude_damping_choi(nq, k, paper_rng(seed, 4 ** nq, 5, 2 ** k, 0))
        return rho / np.trace(rho).real
    if name == "werner_holevo":
        rho = werner_holevo_choi(nq)
        return rho / np.trace(rho).real
    if name.startswith("depolarizing_p"):
        return depolarizing_choi(nq, int(name.split("_p")[1]) / 100)
    if name.startswith("local_depol_p"):
        return local_depolarizing_choi(nq, int(name.split("_p")[1]) / 100)
    if name.startswith("qft_bcsz_p"):
        return qft_bcsz_choi(nq, int(name.split("_p")[1]) / 100, paper_rng(seed, 4 ** nq, 17))
    if name.startswith("qft_depol_p"):
        return depolarizing_choi(nq, int(name.split("_p")[1]) / 100, unitary="qft")
    if name == "logrank":
        d = 2 ** nq
        return random_channel_choi(d, d, max(1, int(np.log2(d * d))), rng)
    raise ValueError(f"unknown channel {name!r}; see channel.__doc__")


def _split_dims(spec):
    """Resolve a ladder entry to (dA, dB, d_AB).

    A pair (dA, dB) is taken as given; an integer is a Choi dimension d_AB and
    is split as evenly as the powers of two allow, so 2^(2k) gives the square
    point dA = dB = 2^k and 2^(2k+1) the rectangular one dA = 2^k, dB = 2^(k+1).
    This is the paper's ladder, whose square points carry the d_AB = 4^n curve
    and whose rectangular points interleave between them.
    """
    if isinstance(spec, (tuple, list)):
        dA, dB = int(spec[0]), int(spec[1])
        return dA, dB, dA * dB
    d_AB = int(spec)
    e = int(round(np.log2(d_AB)))
    if 2 ** e != d_AB:
        raise ValueError(f"d_AB = {d_AB} is not a power of two; pass an explicit "
                         f"(dA, dB) pair instead")
    return 2 ** (e // 2), 2 ** (e - e // 2), d_AB


# ---------------------------------------------------------------------------
# one trial of one rule
# ---------------------------------------------------------------------------
_PROGRESS_WIDTH = 78


class _Line:
    """Plain text for IPython's display, shown without quotes."""
    def __init__(self, text):
        self.text = text

    def __repr__(self):
        return self.text


_LINE = None          # (display handle, last message) of the notebook's progress line


def _progress(msg=""):
    """Overwrite the current progress line with `msg`.

    In a terminal the line is rewritten in place with a carriage return, padded
    to a fixed width so that a shorter message fully replaces a longer one, and
    a call with no argument clears it. In a notebook a carriage return would
    leave every overwritten version behind as a blank line, so the line is a
    display updated in place instead, and a call with no argument marks it done.
    """
    global _LINE
    if sys.stdout.isatty():
        print("\r" + f"  {msg}"[:_PROGRESS_WIDTH].ljust(_PROGRESS_WIDTH) + ("" if msg else "\r"),
              end="", flush=True)
        return
    try:
        from IPython import get_ipython
        from IPython.display import display
    except ImportError:
        return
    if get_ipython() is None:
        return
    if not msg:
        if _LINE is not None and not _LINE[1].endswith("done"):
            _LINE[0].update(_Line(f"  {_LINE[1]}: done"))
        _LINE = None                         # the next sweep starts a line of its own
        return
    if _LINE is None:
        _LINE = (display(_Line(f"  {msg}"), display_id=True), msg)
    else:
        _LINE[0].update(_Line(f"  {msg}"))
        _LINE = (_LINE[0], msg)


def _estimate(rho_ls, rho_true, dA, rule, beta):
    """Density estimate under one threshold rule, then the fidelity projection.

    All four rules of Fig. 8 are instances of the same thresholded estimator and
    are TP regularized the same way, so they differ only in tau.
    """
    if rule == "tau0":
        rho_hat = threshold_density_estimate(rho_ls, tau=0.0)
    elif rule == "lmin":
        rho_hat = lmin_density_estimate(rho_ls)
    elif rule == "bernstein":
        rho_hat = threshold_density_estimate(rho_ls, tau=beta)
    elif rule == "bernstein_half":
        rho_hat = threshold_density_estimate(rho_ls, tau=0.5 * beta)
    elif rule == "pls_hip":
        # the published baseline end to end, at the tight tolerance: its own
        # threshold, then its own TP regularization, so nothing of ours enters
        est, _ = pls_estimate(rho_ls, dA, rho_ls.shape[0] // dA)
        return None, est
    else:
        raise ValueError(f"unknown rule {rule!r}")
    est = fidelity_projection(rho_hat, dA)
    return rho_hat, est


def _shot_key(N):
    """The shot count's index in the paper's seeds, int(10 log10 N) of the
    integer shot count, so that 10^5.5 maps to 54, as it did there."""
    return int(np.log10(int(N)) * 10)


def _sweep(rho, shots, trials, rules, seed, fields=("rank", "inf"), progress=True, stream=4,
           label="", finish=True):
    """{rule: {field: [[trial, ...] per shot count]}} over a shot grid.

    One least-squares estimate per (shot count, trial) is shared by every rule,
    which is what makes the comparison in Fig. 8 a comparison of thresholds
    alone, and is also how the paper's sweeps were run.
    """
    d = rho.shape[0]
    dA = int(round(np.sqrt(d)))
    out = {r: {f: [[] for _ in shots] for f in fields} for r in rules}
    for i, N in enumerate(shots):
        for t in range(trials):
            # stream 4 is the one the paper's accuracy sweeps (Figs. 5-7) were drawn
            # from; Fig. 2 (right) was drawn from stream 1 (sweep_headline) and
            # Fig. 8 from stream 2 (sweep_thresholds)
            rng = paper_rng(seed, stream, _shot_key(N), t)
            rho_ls = simulate_ls_estimate(rho, int(N), rng)
            beta = bernstein_radius(d, int(N), DELTA)
            for rule in rules:
                _, est = _estimate(rho_ls, rho, dA, rule, beta)
                vals = {"rank": choi_rank(est), "inf": infidelity(rho, est),
                        "tr": trace_distance(rho, est),
                        "fro": float(np.linalg.norm(rho - est))}
                for f in fields:
                    out[rule][f][i].append(vals[f])
        if progress:
            _progress(f"{label}N = 1e{np.log10(N):g}  ({i + 1} of {len(shots)} shot counts)")
    if progress and finish:
        _progress()
    return out


def _cached(name):
    try:
        return json.load(open(DATA / f"{name}.json"))
    except FileNotFoundError:
        return None


def _merge_pls(live, cache_name, shots, d_AB, path=("pls_hip",)):
    """Add the cached PLS arm when it is the same sweep, else leave it out.

    Used only when hip/ cannot be imported (see the module docstring), in
    which case this curve can only come from the cache. It is merged only if the cached sweep
    has the same Choi dimension and shot grid, so a figure never mixes a live
    FPLS curve with a PLS curve computed at another size.
    """
    d = _cached(cache_name)
    if d is None:
        return None
    node = d
    for k in path[:-1]:
        node = node[k]
    if d.get("d_AB") != d_AB or not np.allclose(d.get("shots", []), shots, rtol=1e-6):
        return None
    return node.get(path[-1])


# ---------------------------------------------------------------------------
# the sweeps, one per figure
# ---------------------------------------------------------------------------
def sweep_headline(nq=2, shots=None, trials=3, seed=PAPER_SEED, progress=True):
    """Fig. 2 (right): recovered rank and infidelity against the shot count."""
    shots = list(PAPER_SHOTS if shots is None else shots)
    rho = channel("rank2", nq, seed)
    d_AB = rho.shape[0]
    rules = ("bernstein", "pls_hip") if have_hip() else ("bernstein",)
    res = _sweep(rho, shots, trials, rules, seed, progress=progress, stream=1)
    out = {"d_AB": d_AB, "true_rank": choi_rank(rho), "delta": DELTA,
           "shots": shots, "fpls": res["bernstein"]}
    pls = res.get("pls_hip") or _merge_pls(None, "headline", shots, d_AB)
    if pls:
        out["pls_hip"] = pls
    return out


def sweep_accuracy(nq=2, shots=None, trials=3, seed=PAPER_SEED, progress=True):
    """Fig. 5: the same estimates in infidelity, trace and Frobenius distance."""
    shots = list(PAPER_SHOTS if shots is None else shots)
    rho = channel("rank2", nq, seed)
    d_AB = rho.shape[0]
    rules = ["bernstein", "bernstein_half"] + (["pls_hip"] if have_hip() else [])
    res = _sweep(rho, shots, trials, rules, seed,
                 fields=("rank", "inf", "tr", "fro"), progress=progress)
    out = {"d_AB": d_AB, "true_rank": choi_rank(rho), "shots": shots,
           "fpls_beta": res["bernstein"], "fpls_half": res["bernstein_half"]}
    pls = res.get("pls_hip") or _merge_pls(None, "accuracy", shots, d_AB)
    if pls:
        out["pls_hip"] = pls
    return out


def sweep_thresholds(channels=(("rank2", None), ("amp_damping_r2", None)),
                     nq=2, shots=None, trials=3, seed=PAPER_SEED, progress=True):
    """Fig. 8: the four threshold rules, all regularized the same way.

    This figure needs no cached data at all: every curve in it is an instance of
    the thresholded estimator followed by the fidelity projection. A channel is
    (name, label) or (name, label, nq); the paper's two panels sit at different
    sizes, nq = 3 and nq = 4.
    """
    shots = list(PAPER_SHOTS if shots is None else shots)
    rules = ("tau0", "lmin", "bernstein", "bernstein_half")
    panels = []
    for name, label, *own in channels:      # an optional third entry overrides nq for that channel
        rho = channel(name, own[0] if own else nq, seed)
        lam = np.linalg.eigvalsh(rho)
        res = _sweep(rho, shots, trials, rules, seed, progress=progress, stream=2,
                     label=f"{name}: ", finish=False)
        panels.append({"label": label or f"{name}, $d_{{AB}} = {rho.shape[0]}$",
                       "d_AB": rho.shape[0], "true_rank": choi_rank(rho),
                       "lambda_r": float(lam[lam > 1e-10].min()),
                       "shots": shots, **res})
    if progress:
        _progress(f"{len(channels)} channels, {len(shots)} shot counts each")
        _progress()
    return {"panels": panels}


def sweep_spectra(channels=(("rank2", "rank-2 channel"),
                            ("amp_damping_r2", "amplitude damping, rank 2")),
                  nq=2, N=1e6, seed=PAPER_SEED):
    """Fig. 3: the density estimate and its fidelity projection at one N.

    Both regularizations, the fidelity projection and HIP (at the tight
    tolerance), are applied to the same density estimate at tau = beta_N/2.
    The figure draws its own instance of the rank-2 family and its own
    measurement record, as the paper's Fig. 3 does, so nq=4, N=1e8 reproduces
    data/spectra.json.
    """
    cache = _cached("spectra")
    panels = []
    for name, label in channels:
        rho = rank_two_channel(nq, paper_rng(seed, 3)) if name == "rank2" else channel(name, nq, seed)
        d_AB = rho.shape[0]
        dA = int(round(np.sqrt(d_AB)))
        rng = paper_rng(seed, 3, 1)
        rho_ls = simulate_ls_estimate(rho, int(N), rng)
        beta = bernstein_radius(d_AB, int(N), DELTA)
        rho_hat, est = _estimate(rho_ls, rho, dA, "bernstein_half", beta)
        panel = {"label": label, "d_AB": d_AB, "N": N, "true_rank": choi_rank(rho),
                 "true_spectrum": sorted(np.linalg.eigvalsh(rho), reverse=True),
                 "density": {"spectrum": sorted(np.linalg.eigvalsh(rho_hat), reverse=True),
                             "rank": choi_rank(rho_hat), "infidelity": infidelity(rho, rho_hat)},
                 "fpls": {"spectrum": sorted(np.linalg.eigvalsh(est), reverse=True),
                          "rank": choi_rank(est), "infidelity": infidelity(rho, est)}}
        if have_hip():
            # as in the paper's Fig. 3: HIP regularizes the SAME density estimate
            hip_est, _ = hip_regularize(rho_hat, dA, dA, tight=True)
            panel["pls_hip"] = {"spectrum": sorted(np.linalg.eigvalsh(hip_est), reverse=True),
                                "rank": choi_rank(hip_est),
                                "infidelity": infidelity(rho, hip_est)}
        else:
            hit = next((p for p in (cache or {}).get("panels", [])
                        if p["d_AB"] == d_AB and p["N"] == N), None)
            if hit:
                panel["pls_hip"] = hit["pls_hip"]
        panels.append(panel)
    return {"panels": panels}


def sweep_robustness(channels, nq=3, shots=None, trials=3, seed=PAPER_SEED,
                     rule="bernstein_half", spectra_at=(1e6, None), progress=True):
    """Figs. 6 and 7: one column per channel; spectra, rank and infidelity.

    `channels` is a sequence of (name, label) pairs; see `channel.__doc__`.
    `spectra_at` lists the shot counts whose estimated Choi spectra fill the
    top rows of the figure; None stands for the largest shot count of the grid
    (the paper uses 1e6 and 1e10). Each spectrum is that of trial 0 at that
    shot count, the same least-squares estimate the rank and infidelity rows
    use. Pass spectra_at=() to skip the spectra rows.
    """
    shots = list(PAPER_SHOTS if shots is None else shots)
    at = []
    for N in spectra_at:
        N = shots[-1] if N is None else N
        if N in shots and N not in at:
            at.append(N)
    cols = []
    for name, label in channels:
        rho = channel(name, nq, seed)
        d = rho.shape[0]
        dA = int(round(np.sqrt(d)))
        rules = (rule, "pls_hip") if have_hip() else (rule,)
        res = _sweep(rho, shots, trials, rules, seed, progress=progress,
                     label=f"{name}: ", finish=False)
        col = {"label": label, "d_AB": d, "true_rank": choi_rank(rho),
               "shots": shots, "fpls": res[rule]}
        if "pls_hip" in res:
            col["pls_hip"] = res["pls_hip"]
        if at:
            desc = lambda m: np.linalg.eigvalsh(m)[::-1].tolist()
            col["true_spectrum"] = desc(rho)
            col["spectra"] = {}
            for N in at:
                # the trial-0 stream of _sweep, so this is one of its estimates
                rng = paper_rng(seed, 4, _shot_key(N), 0)
                rho_ls = simulate_ls_estimate(rho, int(N), rng)
                beta = bernstein_radius(d, int(N), DELTA)
                entry = {}
                for key, r in (("fpls", rule), ("pls_hip", "pls_hip")):
                    if r in rules:
                        _, est = _estimate(rho_ls, rho, dA, r, beta)
                        entry[key] = {"spectrum": desc(est), "rank": choi_rank(est)}
                col["spectra"][f"N{N:.0e}"] = entry
        cols.append(col)
    if progress:
        _progress(f"{len(channels)} channels, {len(shots)} shot counts each")
        _progress()
    return {"columns": cols}


# ---------------------------------------------------------------------------
# the cost sweeps: everything here is a wall-clock measurement on YOUR machine
# ---------------------------------------------------------------------------
def _dist_to_channels(rho, dA):
    """Purified distance from rho to the Choi states of channels (Eq. 46)."""
    F = fidelity_to_channels(rho, dA)
    return float(np.sqrt(max(0.0, 1.0 - min(1.0, F) ** 2)))


def _calibrated_estimate(rho, target_dpu, rng_seed, dA, N=None, lo=1e2, hi=1e8,
                         iters=30, rel_tol=1e-2):
    """A density estimate sitting a fixed purified distance from the channel set.

    This is the paper's calibration. Comparing regularization cost across
    dimensions at a fixed shot count would be unfair, since the same N buys a
    better estimate at small d; instead the shot count is searched, by bisection
    in log N, until the density estimate (tau = -lambda_min) is within 1% of
    the prescribed distance from the channel set. The distance costs one
    d_A x d_A eigendecomposition (Eq. 46), not an optimization, which is what
    makes the search cheap. Every evaluation redraws the measurement record
    from the same seed `rng_seed`, so the distance is a deterministic function
    of N and the search is reproducible; the paper stores the N it found for
    every instance, and passing that N skips the search.

    Returns the density estimate, N, its distance, and the least eigenvalue of
    the least-squares estimate, which sets HIP's default stopping tolerance.
    """
    def at(N):
        rho_ls = simulate_ls_estimate(rho, int(N), paper_rng(*rng_seed))
        rho_hat = lmin_density_estimate(rho_ls)
        return _dist_to_channels(rho_hat, dA), rho_hat, rho_ls

    if N is None:
        d_lo = at(lo)[0]                       # few shots: far from the set
        while d_lo < target_dpu and lo > 8:
            lo = max(8, lo // 10)
            d_lo = at(lo)[0]
        d_hi = at(hi)[0]                       # many shots: close to it
        while d_hi > target_dpu and hi < 1e11:
            hi *= 10
            d_hi = at(hi)[0]
        for _ in range(iters):
            mid = int(np.sqrt(lo * hi))
            d = at(mid)[0]
            if d > target_dpu:
                lo = mid                       # too noisy: need more shots
            else:
                hi = mid
            if abs(d - target_dpu) < rel_tol * target_dpu:
                break
        N = int(np.sqrt(lo * hi))
    d, rho_hat, rho_ls = at(N)
    return rho_hat, int(N), d, float(np.linalg.eigvalsh(rho_ls)[0])


def _calibration_channel(fam, dA, dB, seed):
    """The random channels of Figs. 2 (left) and 4, with the paper's draws:
    Choi rank 1 (a Haar isometry), log2(d_AB) or d_AB, at any (d_A, d_B)."""
    d_AB = dA * dB
    if fam == "rank1":
        return random_channel_choi(dA, dB, 1, paper_rng(seed, d_AB, dA))
    ranks = {"logrank": int(round(np.log2(d_AB))), "full": d_AB}
    if fam not in ranks:
        raise ValueError(f"unknown family {fam!r}; use 'rank1', 'logrank' or 'full'")
    r = ranks[fam]
    return random_channel_choi(dA, dB, r, paper_rng(seed, d_AB, dA, r))


def _stored_N(cache, fam, target, d_AB):
    """The paper's calibrated shot counts for one cell, or None."""
    d = _cached(cache) or {}
    if cache == "cost_ladder":
        row = next((r for r in d.get("hip", []) if r["d_AB"] == d_AB), None)
        return row.get("N") if row and d.get("target_purified_distance") == target else None
    fam = {"rank1": "iso1"}.get(fam, fam)          # the data's name for the isometries
    panel = next((p for p in d.get("panels", []) if p["family"] == fam), None)
    lev = next((l for l in (panel or {}).get("levels", []) if l["target"] == target), None)
    if lev and d_AB in lev["d_AB"] and "N" in lev:
        return lev["N"][lev["d_AB"].index(d_AB)]
    return None


def sweep_cost_ladder(dims=(4, 16, 64), target_dpu=0.05, trials=3, seed=PAPER_SEED,
                      sdp=False, sdp_max=128, stored_N=False, progress=True):
    """Fig. 2 (left): wall-clock of ONE TP regularization against dimension.

    Times, on this machine, the same calibrated density estimate regularized
    two or three ways: the closed-form fidelity projection in Kraus form, HIP at
    its authors' default settings, and, with sdp=True, the diamond-norm
    projection semidefinite program, solved by the paper's rules (`_time_sdp`;
    needs cvxpy, clarabel and scs).

    WARNING: sdp=True is SLOW and is therefore off by default: about 20 s per
    instance at d_AB = 2^6 and 90 s at 2^7, several hours for the paper's
    ladder; it is why the paper stops that curve at d_AB = 2^7.
    `check_diamond_sdp()` verifies the SDP against three closed forms first.

    Rectangular dimensions. The authors' proj_TP reshapes into four axes of
    equal length and so assumes d_A = d_B; `pls.proj_TP_for` generalizes it, and
    `check_proj_TP()` verifies the two agree exactly where both are defined.

    HIP's time is the authors' own accounting of its computation (pls.py).

    Each trial is its own instance: a Haar-random isometry (Choi rank 1) and a
    measurement record drawn from the paper's seeds, calibrated by
    `_calibrated_estimate`. stored_N=True takes the calibrated shot counts the
    paper found from data/cost_ladder.json instead of searching for them again,
    which gives the same instances and skips most of the run time.

    `dims` entries are either a Choi dimension d_AB, a power of two split as
    evenly as possible into (d_A, d_B), or an explicit (d_A, d_B) pair. So 64
    means 8 x 8 and 128 means 8 x 16, which is how the paper's ladder reaches
    the odd powers of two between its square points; pass pairs to control the
    split yourself. The paper runs 4 to 1024 for HIP and the projection and 4 to
    128 for the SDP, which is hours; the default here is seconds. Absolute times
    are your machine's, but the SCALING is the claim, and that is what this
    reproduces.
    """
    import time
    if sdp:
        warnings.warn(_SDP_WARNING, stacklevel=2)
    out = {"target_purified_distance": target_dpu, "hip_settings": "authors' default",
           "note": "one TP regularization of one calibrated density estimate, timed live",
           "hip": [], "fpls": [], "sdp": []}
    for spec in dims:
        dA, dB, d_AB = _split_dims(spec)
        rho = _calibration_channel("rank1", dA, dB, seed)
        Ns = _stored_N("cost_ladder", "rank1", target_dpu, d_AB) if stored_N else None
        insts = [_calibrated_estimate(rho, target_dpu, (seed, d_AB, dA, t), dA,
                                      N=Ns[t] if Ns else None) for t in range(trials)]
        if progress:
            _progress(f"d_A x d_B = {dA} x {dB}: calibrated, median N = {np.median([x[1] for x in insts]):.1e}")
        # ours: the Kraus form of Theorem 5, which never forms a d_AB x d_AB matrix
        ts = []
        for rho_hat, *_ in insts:
            t0 = time.perf_counter()
            normalise_kraus(choi_to_kraus(rho_hat, dA, dB))
            ts.append(time.perf_counter() - t0)
        out["fpls"].append({"d_AB": d_AB, "t": ts})
        if have_hip():
            ts, iters = [], []
            for rho_hat, _, _, ls_least in insts:
                _, info = hip_regularize(rho_hat, dA, dB, ls_least)
                ts.append(info["t"])
                iters.append(info["iters"])
            out["hip"].append({"d_AB": d_AB, "t": ts, "iters": iters, "N": [x[1] for x in insts]})
        if sdp and d_AB <= sdp_max:     # the SDP is out of reach beyond 2^7 (hours per instance)
            res = [_time_sdp(rho_hat, dA, dB) for rho_hat, *_ in insts]
            ok = [r for r in res if r is not None]
            out["sdp"].append({"d_AB": d_AB, "t": [r["t"] for r in ok],
                               "objective": [r["objective"] for r in ok],
                               "solver": [r["solver"] for r in ok], "invalid": len(res) - len(ok)})
    if progress:
        _progress()
    for k in ("hip", "sdp"):                      # an absent arm is dropped, not empty
        if not out[k]:
            out.pop(k)
    return out


# The diamond-norm projection SDP of Fig. 2 (left), run by the paper's rules.
# SLOW: seconds per instance at d_AB = 2^4, about 20 s at 2^6 and 90 s at 2^7,
# and several hours for the ladder with 10 trials; it is therefore never run
# unless asked for (sweep_cost_ladder(sdp=True)).
SDP_SETTINGS = [                    # (solver, label, options): the defaults and one alternative each
    ("CLARABEL", "default", {}),
    ("CLARABEL", "tol1e-7", dict(tol_gap_abs=1e-7, tol_gap_rel=1e-7, tol_feas=1e-7)),
    ("SCS", "default", {}),
    ("SCS", "eps1e-9", dict(eps_abs=1e-9, eps_rel=1e-9, max_iters=200_000)),
]
SDP_MARGINAL_TOL = 1e-6             # a run counts only if its output is a channel to this accuracy
SDP_MIN_EIG_TOL = -1e-6
SDP_REPS = 3                        # timing repetitions; the fastest is reported

_SDP_WARNING = ("the diamond-norm SDP is slow: about 20 s per instance at d_AB = 2^6 and 90 s "
                "at 2^7, several hours for the paper's ladder")


def _cvxpy():
    try:
        import cvxpy as cp
    except ImportError as exc:                    # pragma: no cover
        raise RuntimeError("the SDP needs cvxpy, clarabel and scs: "
                           "pip install cvxpy clarabel scs") from exc
    return cp


def _diamond_norm_problem(Delta, dA, dB):
    """Watrous' SDP for the diamond norm of a Hermiticity-preserving map given
    its unnormalised Choi operator Delta on A (x) B (A, the input, first):
        minimize y  s.t.  Z + Delta >= 0,  Z - Delta >= 0,  Tr_B Z <= y I_A.
    """
    cp = _cvxpy()
    n = dA * dB
    Z = cp.Variable((n, n), hermitian=True)
    y = cp.Variable()
    cons = [Z + Delta >> 0, Z - Delta >> 0,
            cp.partial_trace(Z, (dA, dB), 1) << y * np.eye(dA)]
    return cp.Problem(cp.Minimize(y), cons)


def _projection_problem(rho_hat, dA, dB):
    """The diamond-norm projection of the Choi state rho_hat onto the channels:
    the SDP above applied to sigma - rho_hat, with sigma a variable constrained
    to be the Choi state of a channel. Returns (problem, sigma).

    y is deliberately not declared nonnegative, though it is: the redundant cone
    makes Clarabel stall at reduced accuracy.
    """
    cp = _cvxpy()
    n = dA * dB
    sigma = cp.Variable((n, n), hermitian=True)
    Z = cp.Variable((n, n), hermitian=True)
    y = cp.Variable()
    Delta = dA * (sigma - rho_hat)                # Choi OPERATOR of the difference map
    cons = [sigma >> 0,
            cp.partial_trace(sigma, (dA, dB), 1) == np.eye(dA) / dA,
            Z + Delta >> 0, Z - Delta >> 0,
            cp.partial_trace(Z, (dA, dB), 1) << y * np.eye(dA)]
    return cp.Problem(cp.Minimize(y), cons), sigma


def check_diamond_sdp(rng=None, verbose=True):
    """The diamond-norm SDP against three closed forms (a few seconds).

      (a) a completely positive map Psi:  ||Psi||_diamond = ||Tr_B J||_inf;
      (b) two replacer channels X -> Tr(X) rho_i:  ||Psi_0 - Psi_1||_diamond
          = ||rho_0 - rho_1||_1;
      (c) two unitary channels:  ||Psi_U - Psi_V||_diamond = 2 sqrt(1 - nu^2),
          nu the distance from 0 to the convex hull of the spectrum of U^dag V.
    """
    cp = _cvxpy()
    rng = np.random.default_rng(0) if rng is None else rng
    solve = lambda prob: prob.solve(solver="SCS", eps_abs=1e-10, eps_rel=1e-10)
    worst = {}
    err = 0.0                                                     # (a)
    for dA, dB in ((2, 2), (2, 4), (3, 2)):
        n = dA * dB
        G = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        J = G @ G.conj().T
        ref = np.linalg.eigvalsh(np.trace(J.reshape(dA, dB, dA, dB), axis1=1, axis2=3)).max()
        err = max(err, abs(solve(_diamond_norm_problem(J, dA, dB)) - ref) / ref)
    worst["completely positive map"] = err
    err = 0.0                                                     # (b)
    for d in (2, 3, 4):
        def rand_state():
            G = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
            R = G @ G.conj().T
            return R / np.trace(R).real
        r0, r1 = rand_state(), rand_state()
        ref = np.abs(np.linalg.eigvalsh(r0 - r1)).sum()
        err = max(err, abs(solve(_diamond_norm_problem(np.kron(np.eye(d), r0 - r1), d, d)) - ref) / ref)
    worst["replacer channels"] = err
    err = 0.0                                                     # (c)
    for d in (2, 3):
        def haar():
            G = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
            Q, R = np.linalg.qr(G)
            return Q * (np.diag(R) / np.abs(np.diag(R)))

        def choi_op(U):                          # unnormalised, A (x) B, A first
            J = np.zeros((d * d, d * d), complex)
            for a in range(d):
                for b in range(d):
                    E = np.zeros((d, d))
                    E[a, b] = 1.0
                    J += np.kron(E, U @ E @ U.conj().T)
            return J
        U, V = haar(), haar()
        ev = np.linalg.eigvals(U.conj().T @ V)
        P = np.vstack([ev.real, ev.imag])        # nu: a small QP over the simplex
        w = cp.Variable(d, nonneg=True)
        nu = cp.Problem(cp.Minimize(cp.norm(P @ w, 2)), [cp.sum(w) == 1]).solve(solver="CLARABEL")
        ref = 2 * np.sqrt(max(0.0, 1 - nu ** 2))
        err = max(err, abs(solve(_diamond_norm_problem(choi_op(U) - choi_op(V), d, d)) - ref) / max(ref, 1e-12))
    worst["unitary channels"] = err
    if verbose:
        for k, v in worst.items():
            print(f"  diamond-norm SDP vs closed form, {k:24s} relative error {v:.1e}")
    assert max(worst.values()) < 1e-6, worst
    return worst


def _time_sdp(rho_hat, dA, dB, reps=SDP_REPS):
    """One diamond-norm projection, timed by the paper's rules. SLOW; see above.

    The instance is solved with Clarabel and SCS, each at its defaults and at
    one alternative setting (below d_AB = 64; from there on SCS only, since
    Clarabel's interior point is already the slower one at 16 and 32). A run
    counts only if the returned sigma is a channel to 1e-6 in its marginal and
    its least eigenvalue; the time reported is the fastest valid run's own
    solver time, best of `reps`, which excludes cvxpy's modelling overhead.
    Returns {"t", "objective", "solver"}, or None if no run was valid.
    """
    best = None
    settings = SDP_SETTINGS if dA * dB < 64 else [x for x in SDP_SETTINGS if x[0] == "SCS"]
    for solver, label, kw in settings:
        t_min, value, sig = np.inf, None, None
        try:
            for _ in range(reps):
                prob, sigma = _projection_problem(rho_hat, dA, dB)
                with warnings.catch_warnings():      # inaccurate solves are caught by the validity test
                    warnings.simplefilter("ignore")
                    prob.solve(solver=solver, warm_start=False, **kw)
                st = getattr(prob.solver_stats, "solve_time", None)
                t_min = min(t_min, st if st is not None else np.inf)
                value, sig = prob.value, sigma.value
        except Exception:                          # a failed solver is one invalid run
            continue
        if sig is None or not np.isfinite(t_min):
            continue
        S = (sig + sig.conj().T) / 2
        marg = np.abs(partial_trace_B(S, dA, dB) - np.eye(dA) / dA).max()
        if marg < SDP_MARGINAL_TOL and np.linalg.eigvalsh(S).min() > SDP_MIN_EIG_TOL:
            if best is None or t_min < best["t"]:
                best = {"t": float(t_min), "objective": float(value), "solver": f"{solver}/{label}"}
    return best


def sweep_hip_iterations(dims=(4, 16, 64), targets=(0.05, 0.1, 0.2, 0.3),
                         families=(("rank1", "Choi rank 1 (isometry)"),), trials=3,
                         seed=PAPER_SEED, stored_N=False, progress=True):
    """Fig. 4: HIP iterations against dimension, at fixed distance to the set.

    Each instance is calibrated to a purified distance from CPTP first, so
    dimension rather than estimate quality is the variable. The point of the
    figure is that it is not only the cost of an iteration that grows with
    dimension but the NUMBER of them. HIP runs at its authors' default settings.

    The families are random channels of Choi rank 1 ("rank1"), log2(d_AB)
    ("logrank") and d_AB ("full"), drawn at the instance's own (d_A, d_B), so
    they exist at the rectangular dimensions too. Each trial is its own
    instance, drawn and calibrated as in the paper. stored_N=True takes the
    calibrated shot counts the paper found from data/hip_iterations.json
    instead of searching for them again: the same instances, much faster.
    """
    if not have_hip():
        raise RuntimeError(f"this figure is HIP's iteration count; the PLS baseline is not importable, see {HIP_FILE}")
    panels = []
    for fam, label in families:
        levels = []
        for target in targets:
            x, iters_all, Ns_all = [], [], []
            for spec in dims:
                dA, dB, d_AB = _split_dims(spec)
                rho = _calibration_channel(fam, dA, dB, seed)
                Ns = _stored_N("hip_iterations", fam, target, d_AB) if stored_N else None
                its, Nc = [], []
                for t in range(trials):
                    rho_hat, N, _, ls_least = _calibrated_estimate(
                        rho, target, (seed, d_AB, dA, t), dA, N=Ns[t] if Ns else None)
                    its.append(hip_regularize(rho_hat, dA, dB, ls_least)[1]["iters"])
                    Nc.append(N)
                Ns_all.append(Nc)
                x.append(d_AB)
                iters_all.append(its)
                if progress:
                    _progress(f"{fam}, d_pu = {target}: d_A x d_B = {dA} x {dB}")
            med = [float(np.median(i)) for i in iters_all]
            # a small instance already inside the tolerance needs zero iterations,
            # which has no logarithm; such points are excluded from the power law
            ok = [k for k, m in enumerate(med) if m > 0]
            fit = None
            if len(ok) >= 3:
                xf = np.log([x[k] for k in ok])
                yf = np.log([med[k] for k in ok])
                b, loga = np.polyfit(xf, yf, 1)
                ss = ((yf - yf.mean()) ** 2).sum()
                resid = yf - np.polyval([b, loga], xf)
                fit = {"a": float(np.exp(loga)), "b": float(b),
                       "r2": float(1 - (resid ** 2).sum() / ss) if ss > 0 else 1.0}
            levels.append({"target": target, "fidelity": float(np.sqrt(1 - target ** 2)),
                           "d_AB": x, "iters": iters_all, "N": Ns_all, "median": med, "fit": fit})
        panels.append({"family": fam, "label": label, "levels": levels})
    if progress:
        _progress()
    return {"panels": panels}


def sweep_table(dims=(16, 64), N=1e6, trials=3, families=("rank2", "logrank"),
                seed=PAPER_SEED, progress=True):
    """Table 1: fidelity, TP-regularization time and storage, head to head.

    Storage is counted the way the table does: a dense d_AB x d_AB matrix for
    the PLS estimate against r Kraus operators of size d_B x d_A for ours, in
    double precision, which is where the ratio comes from. PLS-QPT runs end to
    end at its authors' default settings, and its time is their own accounting
    of HIP and the final mixing (pls.py). The paper's table is dims=(64, 1024),
    N=1e8, trials=3, whose instances and measurement records this reproduces.
    """
    import time
    rows = []
    for fam, spec in ((f, s) for f in families for s in dims):
        dA, dB, d_AB = _split_dims(spec)
        if dA != dB:
            raise ValueError("the table's channels are qubit-structured, so it needs "
                             "d_A = d_B; pass Choi dimensions 4, 16, 64, ...")
        nq = int(np.log2(dA))
        if fam == "rank2":
            rho = channel("rank2", nq, seed)
        elif fam == "logrank":                 # the table's own draw of the log-rank channel
            r = int(round(np.log2(d_AB)))
            rho = random_channel_choi(dA, dB, r, paper_rng(seed, d_AB, dA, r))
        else:
            raise ValueError(f"the table's families are 'rank2' and 'logrank', not {fam!r}")
        fp, hp = [], []
        for t in range(trials):
            rng = paper_rng(seed, 9, nq, t)
            rho_ls = simulate_ls_estimate(rho, int(N), rng)
            beta = bernstein_radius(d_AB, int(N), DELTA)
            rho_hat = threshold_density_estimate(rho_ls, tau=0.5 * beta)
            t0 = time.perf_counter()
            Ks = normalise_kraus(choi_to_kraus(rho_hat, dA, dB))
            t_f = time.perf_counter() - t0
            est = kraus_to_choi(Ks, dA, dB)
            fp.append({"F": 1 - infidelity(rho, est), "t": t_f, "rank": choi_rank(est),
                       "bytes": 16 * len(Ks) * dA * dB})
            if have_hip():
                est_h, info = pls_estimate(rho_ls, dA, dB, tight=False)
                est_h = _pls().psd_guard(est_h)
                hp.append({"F": 1 - infidelity(rho, est_h), "t": info["t"],
                           "rank": choi_rank(est_h), "bytes": 16 * d_AB * d_AB,
                           "iters": info["iters"]})
        row = {"family": fam, "nq": nq, "d_AB": d_AB,
               "true_rank": choi_rank(rho), "fpls": fp}
        if hp:
            row["hip"] = hp
        rows.append(row)
        if progress:
            _progress(f"{fam}, d_AB = {d_AB}")
    if progress:
        _progress()
    return {"N": N, "trials": trials, "hip_settings": "authors' default", "rows": rows}


# ---------------------------------------------------------------------------
# Fig. 9: the lifting loss in practice
# ---------------------------------------------------------------------------
def _haar_isometry(dA, dB, rng):
    """A Haar-random isometry C^dA -> C^dB, as a dB x dA matrix."""
    G = rng.standard_normal((dB, dA)) + 1j * rng.standard_normal((dB, dA))
    Q, R = np.linalg.qr(G)
    return Q * (np.diag(R) / np.abs(np.diag(R)))


def _kraus_choi(Ks, dA, dB):
    """Normalised Choi state of a channel from its Kraus operators (dB x dA)."""
    J = np.zeros((dA * dB, dA * dB), complex)
    for M in Ks:
        v = np.concatenate([M[:, a] for a in range(dA)]) / np.sqrt(dA)
        J += np.outer(v, v.conj())
    return J


def _amp_damping_rect(dA, dB, r, rng):
    """Independent amplitude damping on log2(r) of the input qubits, identity on
    the rest, followed (when d_B = 2 d_A) by an output qubit appended in |0>.
    Choi rank r, at square and rectangular dimensions alike."""
    nA, k = int(np.log2(dA)), int(np.log2(r))
    g = rng.uniform(0.15, 0.45, size=k)
    K = [[np.array([[1, 0], [0, np.sqrt(1 - gi)]], complex),
          np.array([[0, np.sqrt(gi)], [0, 0]], complex)] for gi in g]
    anc = np.array([[1.0], [0.0]]) if dB == 2 * dA else np.eye(1)
    Ks = []
    for sel in range(r):
        M = np.eye(1)
        for q in range(nA):
            M = np.kron(M, K[q][(sel >> q) & 1] if q < k else np.eye(2))
        Ks.append(np.kron(M, anc))
    return _kraus_choi(Ks, dA, dB)


def contraction_families(dA, dB):
    """The eight families of Fig. 9 at (dA, dB), as (label, kind, parameter)."""
    d = dA * dB
    fam = [("rank 1 (isometry)", "rank", 1), ("rank 2", "rank", 2)]
    if int(np.log2(d)) != 2:                      # at d_AB = 4 it is the rank-2 family
        fam.append((r"rank $\log_2 d_{\mathsf{AB}}$", "rank", int(np.log2(d))))
    fam.append(("full rank (BCSZ)", "rank", d))
    fam += [("amplitude damping", "amp", 2 ** k) for k in range(1, int(np.log2(dA)) + 1)]
    fam += [("isometry + 1% depol.", "noisy", 1), ("isometry + 5% depol.", "noisy", 5),
            (r"flat, rank $\approx\sqrt{d_{\mathsf{AB}}}$", "flat", 2 ** (int(np.log2(d)) // 2))]
    return fam


def sweep_contraction(dims=(4, 8, 16, 32, 64), trials=10, eps=0.5, delta=DELTA,
                      seed=PAPER_SEED, progress=True):
    """Fig. 9: how much the fidelity projection loses on tomography estimates.

    For each family and dimension, `trials` random channels rho are drawn, the
    density estimate rho_hat of FPLS (tau = beta_N/2) is formed from N local
    Pauli shots, and the Bures distance of its fidelity projection to rho is
    compared with its own: Theorem 1 bounds the ratio by 2. N is the sample
    size of Eq. (GutaQST) at accuracy eps and confidence delta with the true
    Choi rank r (r = 1 for the noisy isometries, the rank of their dominant
    part): N = (32/3) r^2 3^n log(d_AB/delta) / eps^2, with n = log2 d_AB.
    The draws are the paper's, so dims 4 to 1024 with trials=10 reproduce
    data/contraction.json.
    """
    rows = []
    for spec in dims:
        dA, dB, d = _split_dims(spec)
        n = int(np.log2(d))
        for label, kind, r in contraction_families(dA, dB):
            for t in range(trials):
                key = (dA, dB, r, t, int(kind == "amp"))
                rng = paper_rng(seed, 31, *key)
                if kind == "rank":
                    rho = random_channel_choi(dA, dB, r, rng)
                elif kind == "amp":
                    rho = _amp_damping_rect(dA, dB, r, rng)
                elif kind == "noisy":            # r is the depolarizing percentage
                    p = r / 100
                    rho = (1 - p) * _kraus_choi([_haar_isometry(dA, dB, rng)], dA, dB) + p * np.eye(d) / d
                else:                            # an equal mixture of r isometries
                    rho = _kraus_choi([_haar_isometry(dA, dB, rng) / np.sqrt(r) for _ in range(r)], dA, dB)
                r_true = choi_rank(rho)
                r_N = 1 if kind == "noisy" else r_true
                N = int(np.ceil(32 / 3 * r_N ** 2 * 3 ** n * np.log(d / delta) / eps ** 2))
                rho_ls = simulate_ls_estimate(rho, N, paper_rng(seed, 32, *key))
                est = threshold_density_estimate(rho_ls, tau=0.5 * bernstein_radius(d, N, delta))
                proj = fidelity_projection(est, dA)
                f0, f1 = fidelity(rho, est), fidelity(rho, proj)
                rows.append({"family": label, "kind": kind, "d_AB": d, "rank": r_true, "trial": t, "N": N,
                             "inf_est": 1 - f0, "inf_proj": 1 - f1, "kappa": (1 - f1) / (1 - f0),
                             "bures_ratio": float(np.sqrt(max(0.0, 2 - 2 * f1)) / np.sqrt(max(1e-300, 2 - 2 * f0))),
                             "rank_est": choi_rank(est)})
        if progress:
            _progress(f"d_A x d_B = {dA} x {dB}")
    if progress:
        _progress()
    return {"eps": eps, "delta": delta, "threshold": "beta_N/2", "rows": rows}
