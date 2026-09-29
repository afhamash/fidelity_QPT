"""One fig_*() per figure of the paper, plus the head-to-head table.

Each function reads a cached sweep from data/, draws the figure into figures/
(or into `outdir`, which is how the paper's own figure directory is kept in
sync), and PRINTS the numbers its caption quotes. A run of the notebook is
therefore an audit of the captions, not just a redraw.

The sweeps themselves are far too slow for a laptop -- the headline figure
alone is 12 shot counts x 10 trials x two pipelines at d_AB = 256, with shot
counts to 1e10 -- so they ship as data. Everything the algorithm itself does is
in fpls.py and runs live; see the notebook's worked example.
"""
import json
from pathlib import Path

import numpy as np
from matplotlib.colors import to_rgb, to_rgba
from matplotlib.ticker import LogFormatterSciNotation, LogLocator

from fpls import OURS, HIP, SDP, GREY, use_style, save_figure
from sweeps import HIP_URL

plt = use_style()
DATA = Path(__file__).resolve().parent / "data"

__all__ = ["fig_headline", "fig_cost_ladder", "fig_hip_iterations",
           "fig_threshold_strategies", "fig_spectra", "fig_accuracy",
           "fig_channel_robustness", "fig_fullrank_tradeoff",
           "table_head_to_head"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _load(name, data=None):
    """The cached sweep, or the dict a `sweeps.sweep_*` call just produced.

    Passing `data=` is what makes the notebook reproduce a figure from scratch;
    with `data=None` the figure is drawn from `data/<name>.json` as shipped.
    """
    if data is not None:
        return data
    return json.load(open(DATA / f"{name}.json"))


def _rows(series, field):
    """(n_shots, n_trials) array of one field of one series."""
    return np.array(series[field], float)


def _mode(row):
    """Most frequent value, ties broken upward -- the paper's convention for
    integer-valued ranks, where a median over ten trials could be a half."""
    vals, cnts = np.unique(np.asarray(row), return_counts=True)
    return vals[cnts == cnts.max()].max()


def _geomean(rows, axis=1):
    """Geometric mean over trials, the paper's centre statistic for accuracy.

    The tau = -lambda_min rule hands the regularizer a density estimate of the
    true rank in about half the trials, so the PLS accuracies are bimodal and
    the two clusters sit a factor of ~3 apart; a median jumps between them and
    can rise with N on an estimator that is not getting worse. Figs 2, 5, 6 and
    7 of the paper therefore report geometric means. Fig 8 instead splits the
    two branches, which is the other honest way to show the same structure.
    """
    return np.exp(np.mean(np.log(np.clip(rows, 1e-14, None)), axis=axis))


def _band(ax, x, rows, colour, marker, label, alpha=0.18, centre="geomean"):
    """Opaque centre line over translucent per-trial trajectories.

    centre="mode" for ranks (integer valued), "geomean" for accuracies (see
    _geomean), "median" for wall-clock timings.
    """
    for t in range(rows.shape[1]):
        ax.plot(x, rows[:, t], marker[0], color=colour, ls="none", ms=3, alpha=alpha)
    y = (np.array([_mode(r) for r in rows]) if centre == "mode"
         else np.median(rows, axis=1) if centre == "median" else _geomean(rows))
    ax.plot(x, y, marker, color=colour, label=label, zorder=3)
    return y


def _onset(shots, rank_rows, true_rank):
    """Smallest N from which every trial returns the true rank, there and above."""
    exact = np.all(rank_rows == true_rank, axis=1)
    idx = [i for i in range(len(exact)) if exact[i:].all()]
    return shots[idx[0]] if idx else None


def _exponent(x, y, lo=None):
    """Slope of a log-log fit, optionally restricted to x >= lo."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.ones_like(x, bool) if lo is None else (x >= lo)
    m &= y > 0
    return float(np.polyfit(np.log10(x[m]), np.log10(y[m]), 1)[0])


def _guide(ax, x, y, exponent, text, colour, bump=1.0):
    """Dotted fixed-slope guide line, offset off the curve it describes."""
    x = np.asarray(x, float)
    yy = y * bump * (x / x[0]) ** exponent
    ax.plot(x, yy, ":", color=colour, lw=1.2, alpha=0.9)
    i = int(0.55 * (len(x) - 1))          # inside the axes, never at the clipped last point
    # a guide pushed BELOW its curve (bump < 1) takes its label below the guide,
    # one pushed above takes it above, so the text never lands on the data
    ax.annotate(text, (x[i], yy[i]), color=colour, fontsize=7.5, ha="center",
                va="top" if bump < 1 else "bottom")


def _shots_axis(ax, label=True):
    ax.set_xscale("log")
    if label:
        ax.set_xlabel("measurement shots $N$")


def _say(quiet, *args):
    if not quiet:
        print(*args)


def _fmt(N):
    return f"$10^{{{int(round(np.log10(N)))}}}$" if N else "-"


# ---------------------------------------------------------------------------
# Figure 2 (right): the headline claim
# ---------------------------------------------------------------------------
def fig_headline(data=None, ladder=None, outdir=None, quiet=False):
    """Fig. 2: what the closed form buys, in one figure.

    Left, full height: wall-clock of ONE TP regularization against Choi
    dimension, for the same calibrated density estimate regularized three ways.
    Right, stacked and sharing the shot axis: recovered Choi rank over the
    infidelity of the same estimates. Both right-hand pipelines post-process the
    SAME least-squares estimate; ours thresholds at tau = beta_N and applies the
    fidelity projection, PLS-QPT thresholds at tau = -lambda_min and applies HIP.
    """
    d = _load("headline", data)
    # `ladder=` accepts a live sweep_cost_ladder result; with None the cached one
    # is used, whose timings run to d_AB = 2^10 and are hours of compute
    c = _load("cost_ladder", ladder)
    shots = np.array(d["shots"], float)
    r_true = d["true_rank"]

    # the paper's gridspec: one tall panel on the left, two stacked on the right
    fig = plt.figure(figsize=(9.6, 4.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.45, 0.55], height_ratios=[0.42, 0.58])
    axA = fig.add_subplot(gs[:, 0])
    axR = fig.add_subplot(gs[0, 1])
    axF = fig.add_subplot(gs[1, 1], sharex=axR)

    # ---- (left) the cost ladder
    for key, colour, marker, label in (
            ("fpls", OURS, "o-", "Fidelity projection\n(this work)"),
            ("sdp", SDP, "D--", "Diamond-norm\nprojection (SDP)"),
            ("hip", HIP, "s--", "HIP")):
        rows = c.get(key)
        if not rows:            # a live ladder may omit the SDP or the HIP arm
            continue
        for r in rows:
            axA.plot([r["d_AB"]] * len(r["t"]), r["t"], marker[0], color=colour,
                     ls="none", ms=3, alpha=0.22)
        axA.plot([r["d_AB"] for r in rows], [np.median(r["t"]) for r in rows],
                 marker, color=colour, label=label, zorder=3)   # wall-clock: median
    axA.set(xscale="log", yscale="log")
    axA.set_xscale("log", base=2)
    axA.set_xlabel(r"Choi dimension $d_{\mathsf{AB}}$")
    axA.set_ylabel("Post-processing wall-clock (s)")
    axA.set_title("TP regularization cost")
    axA.set_ylim(top=axA.get_ylim()[1] * 12)   # headroom so the legend clears the SDP points
    axA.legend(loc="upper left")

    # ---- (top right) recovered rank
    rank_f, rank_h = _rows(d["fpls"], "rank"), _rows(d["pls_hip"], "rank")
    inf_f, inf_h = _rows(d["fpls"], "inf"), _rows(d["pls_hip"], "inf")
    onset = _onset(shots, rank_f, r_true)
    _band(axR, shots, rank_h, HIP, "s--", "PLS", centre="mode")
    _band(axR, shots, rank_f, OURS, "o-", "FPLS (this work)", centre="mode")
    axR.axhline(r_true, color="k", ls=":", lw=1.4)
    axR.text(shots[0], r_true * 1.2, f"True Choi rank = {r_true}", color="k", fontsize=7.5,
             ha="left", va="bottom")
    axR.set_yscale("log", base=2)
    axR.set_ylabel("Choi rank")
    axR.set_title("Choi ranks of estimates")
    axR.legend(loc="center right")

    # ---- (bottom right) infidelity of the same estimates
    _band(axF, shots, inf_h, HIP, "s--", "PLS")
    _band(axF, shots, inf_f, OURS, "o-", "FPLS (this work)")
    med_f, med_h = _geomean(inf_f), _geomean(inf_h)
    post = shots >= onset
    _guide(axF, shots[post], med_f[post][0], -1.0, r"$1/N$", OURS, bump=0.32)
    _guide(axF, shots, med_h[0], -0.5, r"$1/\sqrt{N}$", HIP, bump=2.2)
    axF.set_yscale("log")
    axF.set_ylabel(r"Infidelity $1 - \mathrm{F}(\mathrm{C}(\Phi), \mathrm{C}(\hat\Phi))$")
    axF.set_title("Choi infidelity of the estimates")

    for ax in (axR, axF):
        _shots_axis(ax)
        if onset:
            ax.axvline(onset, color="k", ls="--", lw=1.2, alpha=0.8)
    axR.tick_params(labelbottom=False)          # shared shot axis, labelled once
    axR.set_xlabel("")
    axF.text(shots[0] * 1.3, 0.06, f"Rank recovered\nat {_fmt(onset)}",
             transform=axF.get_xaxis_transform(), color="k", fontsize=7.5,
             va="bottom", ha="left", linespacing=1.15)
    fig.tight_layout()
    save_figure(fig, "headline", outdir)

    med = {k: {r["d_AB"]: float(np.median(r["t"])) for r in c.get(k, [])}
           for k in ("hip", "fpls", "sdp")}
    big = max(med["fpls"]) if med["fpls"] else None
    if big and big in med.get("hip", {}):
        _say(quiet, f"at d_AB = {big:<5d}  HIP {med['hip'][big]:8.3f} s   FPLS"
                    f" {med['fpls'][big] * 1e3:8.3f} ms   (paper, at 2^10: 52 s against 0.28 ms)")
    if med["sdp"]:
        at = "   ".join(f"at {k}: {v:.2f} s" for k, v in sorted(med["sdp"].items()))
        _say(quiet, f"SDP              {at}      (paper: 19 s at 2^6, 86 s at 2^7)")
    _say(quiet, f"onset of exact rank recovery      N = {onset:.0e}        (paper: 1e6)")
    _say(quiet, f"PLS-QPT rank over all N, trials   {int(rank_h.min())}-{int(rank_h.max())} of {d['d_AB']}"
                f"   (paper: 202-241 of 256)")
    _say(quiet, f"slopes past the onset             FPLS {_exponent(shots, med_f, onset):+.2f},"
                f" PLS-QPT {_exponent(shots, med_h, onset):+.2f}")
    _say(quiet, "  (the guides are the PREDICTED rates 1/N and 1/sqrt(N), not fits;"
                " Fig. 5 quotes the fitted values)")
    return fig

def fig_cost_ladder(data=None, outdir=None, quiet=False):
    """The left panel of Fig. 2 on its own, for reference.

    `fig_headline` draws this alongside the rank and infidelity panels, exactly
    as the paper's Fig. 2 does; this standalone version is kept for anyone who
    wants the cost ladder by itself.

    Wall-clock of one TP regularization against Choi dimension.

    One density estimate, calibrated to a fixed fidelity from the channel set,
    regularized three ways: the closed-form fidelity projection in Kraus form,
    HIP at its authors' default tolerance, and the diamond-norm projection SDP.
    Single-threaded, medians over ten instances.
    """
    d = _load("cost_ladder", data)
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for key, colour, marker, label in (
            ("sdp", SDP, "D--", "diamond-norm projection (SDP)"),
            ("hip", HIP, "s--", "HIP"),
            ("fpls", OURS, "o-", "fidelity projection (this work)")):
        rows = d[key]
        x = [r["d_AB"] for r in rows]
        for r in rows:
            ax.plot([r["d_AB"]] * len(r["t"]), r["t"], marker[0], color=colour,
                    ls="none", ms=3, alpha=0.22)
        ax.plot(x, [np.median(r["t"]) for r in rows], marker, color=colour, label=label, zorder=3)   # wall-clock: median, as Fig 2 (left) does
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"Choi dimension $d_{AB}$")
    ax.set_ylabel("TP-regularization time (s)")
    ax.set_title("Cost of one TP regularization")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save_figure(fig, "cost_ladder", outdir)

    med = {k: {r["d_AB"]: float(np.median(r["t"])) for r in d[k]} for k in ("hip", "fpls", "sdp")}
    _say(quiet, f"at d_AB = 2^10   HIP {med['hip'][1024]:8.1f} s   FPLS {med['fpls'][1024] * 1e3:6.2f} ms"
                f"   (paper: 52 s against 0.28 ms)")
    _say(quiet, f"SDP              at 2^6 {med['sdp'][64]:5.1f} s     at 2^7 {med['sdp'][128]:5.1f} s"
                f"      (paper: 19 s, 86 s)")
    _say(quiet, f"speed-up at 2^10 {med['hip'][1024] / med['fpls'][1024]:.2e}x")
    return fig


# ---------------------------------------------------------------------------
# Figure 4: why HIP is expensive
# ---------------------------------------------------------------------------
def fig_hip_iterations(data=None, outdir=None, quiet=False):
    """HIP iterations against dimension, at four fixed distances to the set.

    Each instance is calibrated to a common fidelity from the channel set, so
    the variable is dimension rather than estimate quality. The iteration count
    grows with dimension; every iteration is a d_AB-sized eigendecomposition.
    """
    d = _load("hip_iterations", data)
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 2.9), sharey=True)
    cmap = plt.get_cmap("viridis")
    for ax, panel in zip(axes, d["panels"]):
        for k, lev in enumerate(panel["levels"]):
            colour = cmap([0.08, 0.36, 0.62, 0.88][k])
            x = np.array(lev["d_AB"], float)
            for xi, iters in zip(x, lev["iters"]):
                ax.plot([xi] * len(iters), iters, "o", color=colour, ms=3, alpha=0.22)
            ax.plot(x, lev["median"], "o", color=colour, ms=4.5)
            if lev["fit"]:
                ax.plot(x, lev["fit"]["a"] * x ** lev["fit"]["b"], "-", color=colour,
                        lw=1.6, label=fr"$\mathrm{{F}} = {lev['fidelity']:.3f}$")
        ax.set(xscale="log", yscale="log", title=panel["label"])
        ax.set_xlabel(r"$d_{AB}$")
        ax.set_xscale("log", base=2)
    axes[0].set_ylabel("HIP iterations")
    axes[0].legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    save_figure(fig, "hip_iterations", outdir)

    for panel in d["panels"]:
        fits = "  ".join(f"{l['fit']['a']:.1f} d^{l['fit']['b']:.2f}" for l in panel["levels"] if l["fit"])
        _say(quiet, f"{panel['family']:8s} {fits}")
    _say(quiet, "paper: isometry 1.8 d^0.68, 2.3 d^0.62, 1.2 d^0.73, 0.9 d^0.76;"
                " exponents 0.48/0.48/0.62/0.75 and 0.29/0.41/0.62/0.76")
    return fig


# ---------------------------------------------------------------------------
# Figure 8: the threshold rules
# ---------------------------------------------------------------------------
def fig_threshold_strategies(data=None, outdir=None, quiet=False):
    """The four threshold rules on the same least-squares estimates.

    tau = 0 keeps every positive noise eigenvalue; tau = -lambda_min compares
    two order statistics of the same noise and recovers the rank in about half
    the trials at every N; the Bernstein thresholds sit above the noise and are
    exact past their onset.
    """
    d = _load("threshold_strategies", data)
    style = {"tau0": (GREY, "^-", r"$\tau = 0$"),
             "lmin": (HIP, "s--", r"$\tau = -\lambda_{\min}$"),
             "bernstein": (OURS, "o-", r"$\tau = \beta_N$"),
             "bernstein_half": ("#66AADD", "d-", r"$\tau = \beta_N/2$")}
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.2))
    for col, panel in enumerate(d["panels"]):
        shots = np.array(panel["shots"], float)
        for key, (colour, marker, label) in style.items():
            _band(axes[0, col], shots, _rows(panel[key], "rank"), colour, marker, label,
                  alpha=0.12, centre="mode")
            if key == "lmin":
                # Fig. 8 of the paper splits this rule instead of averaging it: the
                # trials divide into those that recovered the rank and those that did
                # not, and the two branches follow the two different rates, so one
                # centre line through them would describe neither.
                rk = _rows(panel[key], "rank")
                inf = _rows(panel[key], "inf")
                hit = np.where(rk == panel["true_rank"], inf, np.nan)
                miss = np.where(rk != panel["true_rank"], inf, np.nan)
                # a shot count where every trial fell on one side leaves the other
                # branch empty; nanmedian warns on that column, so mask it first
                med = lambda a: np.array([np.median(row[~np.isnan(row)])
                                          if np.any(~np.isnan(row)) else np.nan for row in a])
                y_hit, y_miss = med(hit), med(miss)
                axes[1, col].plot(shots, y_hit, marker, color=colour, zorder=3,
                                  label=label + ", rank recovered")
                axes[1, col].plot(shots, y_miss, marker[0] + "--", color=colour, mfc="none",
                                  zorder=3, label=label + ", rank missed")
            else:
                _band(axes[1, col], shots, _rows(panel[key], "inf"), colour, marker, label,
                      alpha=0.12)
        axes[0, col].axhline(panel["true_rank"], color="k", ls=":", lw=1.1)
        axes[0, col].set(yscale="log", title=panel["label"])
        axes[1, col].set(yscale="log")
        for ax in axes[:, col]:
            _shots_axis(ax, label=ax is axes[1, col])
        onset = _onset(shots, _rows(panel["bernstein"], "rank"), panel["true_rank"])
        if onset:
            for ax in axes[:, col]:
                ax.axvline(onset, color=OURS, ls="--", lw=1.1, alpha=0.8)
        guaranteed = (8 * 3.0 ** np.log2(panel["d_AB"]) * np.log(panel["d_AB"] / 0.05)
                      / (3 * (panel["lambda_r"] / 2) ** 2))
        mode0 = np.array([_mode(r) for r in _rows(panel["tau0"], "rank")])
        lmin_rank = _rows(panel["lmin"], "rank")
        mode_l = np.array([_mode(r) for r in lmin_rank])
        late = shots >= 1e5
        hit = (lmin_rank[late] == panel["true_rank"]).sum(axis=1)
        _say(quiet, f"{panel['label']}")
        _say(quiet, f"   lambda_r = {panel['lambda_r']:.3f};  guaranteed onset beta_N < lambda_r/2"
                    f" at N = {guaranteed:.0e}")
        _say(quiet, f"   tau = 0        modal rank climbs {int(mode0[0])} -> {int(mode0[-1])} of {panel['d_AB']}")
        _say(quiet, f"   tau = -lmin    modal rank {int(mode_l[late].min())}-{int(mode_l[late].max())},"
                    f" exact in {hit.min()}-{hit.max()} of {lmin_rank.shape[1]} trials at every"
                    f" N >= 1e5, with no trend")
        _say(quiet, f"   tau = beta_N   exact from N = {onset:.0e};"
                    f"  beta_N/2 from N = {_onset(shots, _rows(panel['bernstein_half'], 'rank'), panel['true_rank']):.0e}")
    axes[0, 0].set_ylabel("recovered Choi rank")
    axes[1, 0].set_ylabel(r"Choi infidelity $1 - \mathrm{F}$")
    axes[0, 0].legend(loc="upper left", fontsize=7, ncol=2)
    fig.tight_layout()
    save_figure(fig, "threshold_strategies", outdir)
    _say(quiet, "paper: 6->15 of 64 and 7->18 of 256 at tau = 0; modal rank 4-5 and 2-3 and exact in"
                " 3-9 of 10 trials at -lambda_min;\n       onsets 3e7 and 1e6 (halved: 1e7 and 3e5),"
                " guaranteed onsets 6e7 and 3e6")
    return fig


# ---------------------------------------------------------------------------
# Figure 3: rank preservation
# ---------------------------------------------------------------------------
def fig_spectra(data=None, outdir=None, quiet=False):
    """Fig. 3: ordered Choi spectra at N = 1e8, one row per channel.

    Three columns, all seeded by the SAME density estimate: the density estimate
    itself, then the two TP regularizations of it. The fidelity projection is a
    congruence by an invertible matrix, so it returns the rank of its input
    exactly; HIP's TP step and its final mixing with the maximally mixed state
    fill in every remaining direction.
    """
    d = _load("spectra", data)
    rows = d["panels"]
    fig, axes = plt.subplots(len(rows), 3, figsize=(9.0, 2.5 * len(rows)),
                             squeeze=False, sharex=True, sharey=True)
    COLUMNS = (("density", "tab:gray", "Density estimate"),
               ("fpls", OURS, "Fidelity projection"),
               ("pls_hip", HIP, "HIP"))
    for r, panel in enumerate(rows):
        idx = np.arange(1, panel["d_AB"] + 1)
        lam_true = np.clip(np.array(panel["true_spectrum"]), 1e-16, None)
        for ax, (key, base, title) in zip(axes[r], COLUMNS):
            e = panel[key]
            ax.step(idx, lam_true, where="mid", color="k", lw=2.2,
                    label=f"true (rank {panel['true_rank']})", zorder=10)
            # width=1.0 so adjacent bars touch: with width<1 the white gaps go
            # sub-pixel at high index and anti-alias into the fill. Each bar is
            # outlined in a darker shade of its own colour, with the outline
            # opacity ramped down as 1/index: on a log axis the number of
            # outlines overlapping within one stroke width grows as i, and n
            # strokes of opacity a composite to 1-(1-a)^n, so a ~ 1/i holds the
            # rendered darkness constant instead of letting the dense region
            # saturate and read as emphasis where the eigenvalues are smaller.
            rgb = np.array(to_rgb(base))
            alpha = 0.6 * np.minimum(1.0, 16.0 / idx)
            ax.bar(idx, np.clip(np.array(e["spectrum"]), 1e-16, None), width=1.0,
                   facecolor=to_rgba(base, 0.55),
                   edgecolor=[(*(rgb * 0.55), ai) for ai in alpha], linewidth=0.35,
                   label=f"estimate (rank {e['rank']})")
            ax.set(xscale="log", yscale="log", ylim=(1e-15, 2))   # floor at 1e-15: below is numerical noise
            ax.set_xscale("log", base=2)          # eigenvalue index in powers of two
            ax.xaxis.set_major_locator(LogLocator(base=2, numticks=6))
            ax.xaxis.set_major_formatter(LogFormatterSciNotation(base=2))
            ax.xaxis.set_minor_locator(LogLocator(base=2, subs=[]))
            if r == 0:
                ax.set_title(title)
            if r == len(rows) - 1:
                ax.set_xlabel("eigenvalue index")
            ax.text(0.97, 0.95, f"$\\mathrm{{F}} = {1 - e['infidelity']:.5f}$",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7.5, zorder=11,
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=2))
            ax.grid(True, which="both", ls="--", alpha=.3)
            ax.legend(loc="lower right", fontsize=6.5, framealpha=0.9)
        axes[r][0].set_ylabel(r"eigenvalue $\lambda_i$")
        _say(quiet, f"{panel['label']:28s} true rank {panel['true_rank']:3d}"
                    f"   FPLS {panel['fpls']['rank']:3d} (1-F {panel['fpls']['infidelity']:.2e})"
                    f"   PLS-QPT {panel['pls_hip']['rank']:3d} (1-F {panel['pls_hip']['infidelity']:.2e})")
    fig.tight_layout(w_pad=1.0, h_pad=0.4)
    fig.suptitle("True and estimated Choi spectra", y=1.02)
    save_figure(fig, "spectra", outdir)
    _say(quiet, f"paper: FPLS returns 2 and 4; PLS-QPT returns 225 and 240 of 256")
    return fig

def fig_accuracy(data=None, outdir=None, quiet=False):
    """Infidelity, trace distance and Frobenius distance on the same estimates.

    The quadratic gain from recovering the rank is a property of the FIDELITY:
    in the two norms all three pipelines converge at 1/sqrt(N) and differ only
    by a constant.
    """
    d = _load("accuracy", data)
    shots = np.array(d["shots"], float)
    series = (("fpls_beta", OURS, "o-", r"FPLS ($\tau=\beta_N$)"),
              ("fpls_half", "#66AADD", "d-", r"FPLS ($\tau=\beta_N/2$)"),
              ("pls_hip", HIP, "s--", "PLS"))
    # One panel per metric, as in Fig. 5 of the paper.
    METRICS = (("inf", "Choi infidelity", r"infidelity $1-\mathrm{F}(\rho,\hat\rho)$"),
               ("tr", "Choi trace distance", r"trace distance $\frac{1}{2}\|\rho-\hat\rho\|_1$"),
               ("fro", "Choi Frobenius distance", r"Frobenius distance $\|\rho-\hat\rho\|_2$"))
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1))
    onset_half = _onset(shots, _rows(d["fpls_half"], "rank"), d["true_rank"])
    for ax, (field, title, ylab) in zip(axes, METRICS):
        for key, colour, marker, label in series:
            _band(ax, shots, _rows(d[key], field), colour, marker, label, alpha=0.12)
        _shots_axis(ax)
        ax.set(yscale="log", ylabel=ylab, title=title)
        # the guides carry the PREDICTED exponents, displaced off the curve they
        # describe and coloured like it in the infidelity panel, black in the norms
        if field == "inf":
            _guide(ax, shots, _geomean(_rows(d["pls_hip"], "inf"))[0], -0.5,
                   r"$\propto N^{-1/2}$", HIP, bump=2.5)
            post = shots >= onset_half
            _guide(ax, shots[post], _geomean(_rows(d["fpls_half"], "inf"))[post][0], -1.0,
                   r"$\propto N^{-1}$", "#66AADD", bump=0.35)
        else:
            _guide(ax, shots, _geomean(_rows(d["pls_hip"], field))[0], -0.5,
                   r"$\propto N^{-1/2}$", "k", bump=2.2)
    # one legend for all three panels, under the figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=len(series), frameon=False,
               bbox_to_anchor=(0.5, 0.0), columnspacing=2.4)
    save_figure(fig, "accuracy", outdir)

    # Every exponent is fitted from the onset upward: below it the thresholded
    # estimate sits on a plateau, so a fit over the whole range would describe
    # the plateau rather than the rate. This is the paper's convention.
    med = {k: {f: _geomean(_rows(d[k], f)) for f in ("inf", "tr", "fro")}
           for k, *_ in series}
    onset = {k: _onset(shots, _rows(d[k], "rank"), d["true_rank"]) for k in med}
    o = onset["fpls_beta"]
    # Every line below is guarded: at a reduced NQ, TRIALS or shot grid an onset
    # can sit at the first point or fall outside the range, which leaves these
    # slices empty. The figure is still correct; only the audit has nothing to
    # compare, and says so rather than failing.
    has_pls = "pls_hip" in med
    fmt = lambda v: "n/a" if v is None else f"{v:.1f}x"
    def ratio(num, den, idx):
        if not has_pls or idx is None or not np.size(np.atleast_1d(num[idx])):
            return None
        return float(np.max(np.atleast_1d(num[idx]) / np.atleast_1d(den[idx])))
    _say(quiet, f"onsets: beta_N {onset['fpls_beta']:.0e}, beta_N/2 {onset['fpls_half']:.0e}"
                f"        (paper: 1e6 and 3e5)")
    for f, want in (("inf", "-1.00 and -0.50"), ("tr", "-0.50 and -0.49"), ("fro", "-0.50 and -0.50")):
        pls = f"{_exponent(shots, med['pls_hip'][f], o):+.2f}" if has_pls else "  n/a"
        _say(quiet, f"  exponent ({f:3s})  FPLS {_exponent(shots, med['fpls_beta'][f], o):+.2f}"
                    f"   PLS-QPT {pls}      (paper: {want})")
    if has_pls:
        for N in (1e6, 1e8, 1e10):
            if N > shots[-1] or N < shots[0]:
                continue
            i = int(np.argmin(np.abs(shots - N)))
            _say(quiet, f"  infidelity ratio at N = {N:.0e}   "
                        f"{med['pls_hip']['inf'][i] / med['fpls_beta']['inf'][i]:8.3g}x")
        _say(quiet, "  (paper: 8x, 0.9e2, 1.0e3)")
        for k in ("fpls_half", "fpls_beta"):
            j = int(np.where(shots == onset[k])[0][0]) if onset[k] in shots else None
            if j is None or j == 0:
                _say(quiet, f"  {k:10s} its onset is the first shot count here, so there is no"
                            f" pre-onset point   (paper: 3.4x and 6.0x)")
                continue
            _say(quiet, f"  {k:10s} behind PLS-QPT by {med[k]['inf'][j - 1] / med['pls_hip']['inf'][j - 1]:.1f}x"
                        f" at the last N before its onset   (paper: 3.4x and 6.0x)")
        _say(quiet, f"  at N = {shots[-1]:.0e} ahead by {med['pls_hip']['tr'][-1] / med['fpls_beta']['tr'][-1]:.1f}x in trace"
                    f" and {med['pls_hip']['fro'][-1] / med['fpls_beta']['fro'][-1]:.1f}x in Frobenius   (paper: 1.5x, 1.3x)")
        pre = shots < onset["fpls_half"]
        _say(quiet, f"  before its onset beta_N/2 is behind by up to"
                    f" {fmt(ratio(med['fpls_half']['tr'], med['pls_hip']['tr'], pre))} in trace and"
                    f" {fmt(ratio(med['fpls_half']['fro'], med['pls_hip']['fro'], pre))} in Frobenius"
                    f"   (paper: 2.0x, 3.0x)")
    else:
        _say(quiet, "  (the PLS baseline is absent, so the ratios against it are not shown;"
                    f"\n   clone it to compute them: git clone {HIP_URL})")
    return fig


# ---------------------------------------------------------------------------
# Figures 6 and 7: across channels
# ---------------------------------------------------------------------------
def _robustness(name, title, outdir, quiet, expect, data=None):
    """Figs 6-7: four columns, four rows, in the paper's layout.

    Rows one and two are the true Choi spectrum against the two estimates at
    N = 1e6 and N = 1e10; row three the recovered rank; row four the infidelity.
    """
    d = _load(name, data)
    cols = d["columns"]
    Ns = [N for N in ("N1e+06", "N1e+10") if all(N in c.get("spectra", {}) for c in cols)]
    nrow = len(Ns) + 2
    fig, axes = plt.subplots(nrow, len(cols), figsize=(2.55 * len(cols), 1.9 * nrow),
                             squeeze=False)
    for j, col in enumerate(cols):
        shots = np.array(col["shots"], float)
        # --- spectra rows
        for r, N in enumerate(Ns):
            ax = axes[r, j]
            e = col["spectra"][N]
            idx = np.arange(1, col["d_AB"] + 1)
            ax.bar(idx, np.clip(np.array(col["true_spectrum"]), 1e-16, None), width=1.0,
                   color="#44AA99", alpha=0.55, label="true spectrum", zorder=1)
            for key, colour, ls in (("fpls", OURS, "--"), ("pls_hip", HIP, "--")):
                ax.step(idx, np.clip(np.array(e[key]["spectrum"]), 1e-16, None), where="mid",
                        color=colour, ls=ls, lw=1.3, zorder=3,
                        label=f"{'FPLS' if key == 'fpls' else 'PLS'}, rank {e[key]['rank']}")
            ax.set(xscale="log", yscale="log", ylim=(1e-16, 4))
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_locator(LogLocator(base=2, numticks=4))
            ax.xaxis.set_major_formatter(LogFormatterSciNotation(base=2))
            ax.xaxis.set_minor_locator(LogLocator(base=2, subs=[]))
            ax.legend(loc="lower left", fontsize=5.5, framealpha=0.85, borderpad=0.25,
                      labelspacing=0.25, handlelength=1.4)
            if j == 0:
                ax.set_ylabel(f"$N = 10^{{{int(N[3:])}}}$", fontsize=7)
            if r == 0:
                ax.set_title(col["label"], fontsize=8)
        # --- rank row
        axr = axes[len(Ns), j]
        rank_f, rank_h = _rows(col["fpls"], "rank"), _rows(col["pls_hip"], "rank")
        _band(axr, shots, rank_h, HIP, "s--", "PLS", alpha=0.12, centre="mode")
        _band(axr, shots, rank_f, OURS, "o-", r"FPLS ($\beta_N/2$)", alpha=0.12, centre="mode")
        axr.axhline(col["true_rank"], color="k", ls=":", lw=1.1, label="true Choi rank")
        axr.set(yscale="log")
        axr.set_yscale("log", base=2)
        # --- infidelity row
        axi = axes[len(Ns) + 1, j]
        _band(axi, shots, _rows(col["pls_hip"], "inf"), HIP, "s--", "PLS", alpha=0.12)
        _band(axi, shots, _rows(col["fpls"], "inf"), OURS, "o-", "FPLS", alpha=0.12)
        axi.set(yscale="log")
        _shots_axis(axi)
        onset = _onset(shots, rank_f, col["true_rank"])
        if onset:
            for ax in (axr, axi):
                ax.axvline(onset, color="k", ls="--", lw=1.1, alpha=0.8)
        for ax in (axr, axi):
            ax.set_xscale("log")
            ax.tick_params(labelsize=7)
        _say(quiet, f"{col['label']:26s} true rank {col['true_rank']:3d}"
                    f"   onset {('%.0e' % onset) if onset else 'not reached':>11s}"
                    f"   PLS-QPT modal rank {int(min(_mode(r) for r in rank_h))}-"
                    f"{int(max(_mode(r) for r in rank_h))}"
                    f" (per trial {int(rank_h.min())}-{int(rank_h.max())}) of {col['d_AB']}"
                    f"   exponents {_exponent(shots, _geomean(_rows(col['fpls'], 'inf')), onset):+.1f}"
                    f" / {_exponent(shots, _geomean(_rows(col['pls_hip'], 'inf')), onset):+.1f}")
    axes[len(Ns), 0].set_ylabel("recovered\nChoi rank", fontsize=7)
    axes[len(Ns) + 1, 0].set_ylabel("infidelity\n" + r"$1-\mathrm{F}$", fontsize=7)
    axes[len(Ns), 0].legend(loc="lower right", fontsize=6, framealpha=0.85)
    fig.suptitle(title, y=1.005)
    fig.tight_layout(h_pad=0.5, w_pad=0.6)
    save_figure(fig, name, outdir)
    _say(quiet, f"paper: {expect}")
    return fig

def fig_channel_robustness(data=None, outdir=None, quiet=False):
    """Four channels without full Choi rank, d_AB = 2^6.

    The onset moves with the smallest nonzero Choi eigenvalue, as the analysis
    predicts: the Werner-Holevo channel has rank 28 but a flat spectrum, so it
    recovers two decades before rank-8 amplitude damping.
    """
    return _robustness("channel_robustness", "Channels without full Choi rank", outdir, quiet,
                       "onsets 3e5, 1e7, 1e9, 1e7; PLS-QPT 49-63 of 64 per trial;"
                       " exponents -1.0 against -0.5 to -0.6", data)


def fig_fullrank_tradeoff(data=None, outdir=None, quiet=False):
    """Four channels WITH full Choi rank, d_AB = 2^6: where FPLS loses.

    With no spectral gap to place the threshold in, the Bernstein rule admits a
    direction only once the data resolves it, so the returned rank climbs with
    N and never reaches the truth in the range tested; PLS-QPT, which keeps
    every direction, is the better estimator here.
    """
    return _robustness("fullrank", "Channels with full Choi rank", outdir, quiet,
                       "no FPLS onset within N <= 1e10.  The paper's \"ranks 51 to 64\" is the range of"
                       "\n       the PLS-QPT estimates shown in the spectrum panels of Fig. 7, a different"
                       "\n       set of estimates from the rank sweep summarised above", data)


# ---------------------------------------------------------------------------
# Table 1
# ---------------------------------------------------------------------------
def table_head_to_head(data=None, quiet=False):
    """PLS-QPT against FPLS on the same N = 1e8 shots: fidelity, time, storage.

    Prints Table 1 of the paper. The storage columns are what the estimate
    costs to keep: r Kraus operators for FPLS against a dense d_AB x d_AB
    matrix for PLS-QPT, whose output is of nearly full rank.
    """
    d = _load("table", data)
    med = lambda rs, f: float(np.median([r[f] for r in rs]))
    head = (f"{'d_AB':>6} {'rank':>5} | {'F (PLS)':>10} {'F (FPLS)':>10} |"
            f" {'t (PLS)':>10} {'t (FPLS)':>10} {'ratio':>9} |"
            f" {'kB (PLS)':>9} {'kB (FPLS)':>10} {'ratio':>7}")
    lines = [head, "-" * len(head)]
    for r in d["rows"]:
        if "hip" not in r:      # a live table without the PLS baseline on disk
            f_f, t_f, b_f = med(r["fpls"], "F"), med(r["fpls"], "t"), med(r["fpls"], "bytes")
            lines.append(f"{r['d_AB']:>6} {r['true_rank']:>5} | {'n/a':>10} {f_f:10.6f} |"
                         f" {'n/a':>10} {t_f * 1e3:9.2f}ms {'n/a':>9} |"
                         f" {'n/a':>9} {b_f / 1e3:10.1f} {'n/a':>7}")
            continue
        f_h, f_f = med(r["hip"], "F"), med(r["fpls"], "F")
        t_h, t_f = med(r["hip"], "t"), med(r["fpls"], "t")
        b_h, b_f = med(r["hip"], "bytes"), med(r["fpls"], "bytes")
        lines.append(f"{r['d_AB']:>6} {r['true_rank']:>5} | {f_h:10.6f} {f_f:10.6f} |"
                     f" {t_h:9.3f}s {t_f * 1e3:9.2f}ms {t_h / t_f:8.0f}x |"
                     f" {b_h / 1e3:9.1f} {b_f / 1e3:10.1f} {b_h / b_f:6.0f}x")
    _say(quiet, f"N = {d['N']:.0e}, medians over {d['trials']} trials, HIP at tolerance {d['hip_tolerance']:g}")
    _say(quiet, "\n".join(lines))
    _say(quiet, "\npaper Table 1: 0.99954 / 0.99999 at 2^6 rank 2, 16 ms against 0.03 ms, 66 kB against 3 kB;"
                "\n               0.99666 / 0.99984 at 2^10 rank 2, 39 s against 0.37 ms, 16.8 MB against 49 kB")
