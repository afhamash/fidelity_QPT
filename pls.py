"""The PLS-QPT baseline, run exactly as its authors run it.

PLS-QPT is the projected least-squares estimator of Surawy-Stepney, Kahn,
Kueng and Guta: threshold the least-squares estimate to a density matrix, then
make it trace preserving by the hyperplane intersection projection (HIP). The
files in hip/ are byte-identical copies of their implementation (origin in
hip/NOTICE.md) and are executed unmodified. The pipeline and every parameter
below are those of `generate_simulations` in their experiment_generation.py,
with its defaults:

    rhoCP, ls_least = proj_CP_threshold(rho_LS, thres_least_ev=True)
    tol             = max(depo_rtol * |ls_least|, depo_tol)       (1e-1, 1e-3)
    rho             = hyperplane_intersection_projection_switch(rhoCP, **OPTIONS,
                                                                least_ev_x_dim2_tol=tol)
    rho             = final_CPTP_by_mixing(rho)

HIP stops once the least eigenvalue is within tol/d_AB of zero, and the final
step mixes with the maximally mixed state just enough to lift it to zero.

The paper reports HIP's COST (Fig. 2 left, Fig. 4, the time column of Table 1)
at these defaults, `pls_default`. It reports PLS ACCURACY and RANK (Fig. 2
right, Figs. 3, 5, 6, 7) at a tight, fixed tolerance instead, `pls_tight`: the
default floor depo_tol = 1e-3 does not shrink with the shot count and would put
a plateau on the PLS infidelity. Only the tolerance and the iteration cap
differ; the authors override the tolerance in the same way, to 1e-12, for the
method comparison of their own article.

The only departures from their code, none of which touches the algorithm, are
listed in hip/NOTICE.md.
"""
from __future__ import annotations

import importlib.util
import sys
import time
import types
import warnings
from pathlib import Path

import numpy as np

HIP_DIR = Path(__file__).resolve().parent / "hip"
HIP_URL = "https://github.com/Hannoskaj/Hyperplane_Intersection_Projection"

# generate_simulations(default_options_proj=...), verbatim
OPTIONS = {'maxiter': 300, 'HIP_to_alt_switch': 'first', 'missing_w': 3, 'min_part': .1,
           'HIP_steps': 10, 'alt_steps': 4, 'alt_to_HIP_switch': 'cos', 'min_cos': .99,
           'max_mem_w': 30, 'genarg_alt': (1, 3, 20), 'genarg_HIP': (5,)}
DEPO_TOL, DEPO_RTOL = 1e-3, 1e-1

# accuracy and rank: a tight fixed tolerance, the cap raised so that the
# tolerance, not the cap, ends the iteration; everything else at the default
TIGHT = dict(depo_tol=1e-8, depo_rtol=0.0, options={**OPTIONS, 'maxiter': 2000})

__all__ = ["OPTIONS", "DEPO_TOL", "DEPO_RTOL", "TIGHT", "HIP_URL", "HIP_DIR",
           "proj_CP_threshold", "final_CPTP_by_mixing", "proj_TP_for",
           "hip_default", "pls_default", "pls_tight", "hip_tight", "psd_guard"]

if not hasattr(np, "row_stack"):               # removed in NumPy 2.5
    np.row_stack = np.vstack

# imported by the authors' files but unused on this code path
for _name in ("tables", "pandas", "numexpr"):
    if importlib.util.find_spec(_name) is None:
        sys.modules.setdefault(_name, types.ModuleType(_name))


def _load(name, alias):
    spec = importlib.util.spec_from_file_location(alias, HIP_DIR / name)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    with warnings.catch_warnings():             # "\s" in two of their docstrings
        warnings.simplefilter("ignore", SyntaxWarning)
        spec.loader.exec_module(mod)
    return mod


# the introspection module does `from projections import *`
_saved = sys.modules.get("projections")
_proj = _load("projections.py", "projections")
_intro = _load("_projections_with_introspection.py", "pls_introspection")
if _saved is None:
    del sys.modules["projections"]
else:
    sys.modules["projections"] = _saved
_intro.print = lambda *a, **k: None            # per-iteration prints

proj_CP_threshold = _intro.proj_CP_threshold
final_CPTP_by_mixing = _intro.final_CPTP_by_mixing
_hip_switch = _intro.hyperplane_intersection_projection_switch_with_storage


def proj_TP_for(dA, dB):
    """Orthogonal projection onto {X : Tr_B X = I_A/d_A}, input system first."""
    IA, IB = np.eye(dA) / dA, np.eye(dB) / dB

    def proj_TP(rho):
        marg = np.trace(rho.reshape(dA, dB, dA, dB), axis1=1, axis2=3)
        return rho + np.einsum('de,fg->dfeg', IA - marg, IB).reshape(dA * dB, dA * dB)

    return proj_TP


class _Row(dict):
    def append(self):
        pass


class _Sink:
    def append(self, *a):
        pass


class _Group:
    """Stand-in for the HDF5 group the routine logs into; keeps nothing."""
    def __init__(self):
        self.loops = types.SimpleNamespace(row=_Row(), attrs=types.SimpleNamespace(), flush=lambda: None)
        self.xw = self.active_w = self.coeffs = self.target = self.rhoTP = _Sink()


def hip_default(rhoCP, ls_least_ev, dA, dB, true_choi=0.0, options=None,
                depo_tol=DEPO_TOL, depo_rtol=DEPO_RTOL):
    """HIP and the final mixing on a density estimate, at the authors' settings.

    `ls_least_ev` is the least eigenvalue of the least-squares estimate, which
    sets the stopping tolerance. Returns (rho, info); info["t"] is the
    authors' own accounting of the computation time (their comp_time, which
    excludes the logging, plus the last step and the final mixing).
    """
    opts = dict(OPTIONS if options is None else options)
    tol = float(np.maximum(-ls_least_ev * depo_rtol, depo_tol))
    _intro.proj_TP = proj_TP_for(dA, dB)
    w0 = time.perf_counter()
    rho, dt, comp_time, m = _hip_switch(rhoCP, _Group(), true_choi, **opts,
                                        least_ev_x_dim2_tol=tol, all_dists=False,
                                        with_evs=False, dist_L2=True, save_intermediate=False)
    t0 = time.perf_counter()
    rho, least_ev = final_CPTP_by_mixing(rho, full_output=True)
    t1 = time.perf_counter()
    return rho, dict(iters=int(m), t=float(comp_time + dt + (t1 - t0)), t_wall=float(t1 - w0),
                     tol=tol, least_ev_before_mixing=float(least_ev),
                     converged=bool(-least_ev < tol / len(rho)))


def pls_default(rho_ls, dA, dB, true_choi=0.0, **kw):
    """The authors' PLS-QPT on a least-squares estimate. Returns (rho, info)."""
    t0 = time.perf_counter()
    rhoCP, ls_least = proj_CP_threshold(rho_ls, full_output=True, thres_least_ev=True)
    t_cp = time.perf_counter() - t0
    rho, info = hip_default(rhoCP, ls_least, dA, dB, true_choi, **kw)
    info.update(t_first_cp=float(t_cp), ls_least_ev=float(ls_least))
    return rho, info


def psd_guard(rho):
    """The authors' output is Hermitian and positive semidefinite only up to
    roundoff (asymmetry and least eigenvalue of order 1e-17); clear that
    residue so that fidelity routines accept it."""
    rho = (rho + rho.conj().T) / 2
    lmin = np.linalg.eigvalsh(rho)[0]
    return rho if lmin >= 0 else (rho - lmin * np.eye(len(rho))) / (1 - lmin * len(rho))


def pls_tight(rho_ls, dA, dB):
    """The authors' PLS-QPT at the tight tolerance. Returns (rho, info)."""
    rho, info = pls_default(rho_ls, dA, dB, **TIGHT)
    return psd_guard(rho), info


def hip_tight(rho_hat, dA, dB):
    """HIP at the tight tolerance on a given density estimate (Fig. 3, where
    it regularizes the same estimate as the fidelity projection)."""
    rho, info = hip_default(rho_hat, 0.0, dA, dB, **TIGHT)
    return psd_guard(rho), info
