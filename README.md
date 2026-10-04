# Fidelity Projected Least Squares

Companion code for **Fast and Sure-ious Quantum Process Tomography**, A. Afham, S. Sen and M. Tomamichel,
[arXiv:2609.39941](https://arxiv.org/abs/2609.39941). The code and data match version 2 of the paper (v2),
in which the PLS baseline is run with its authors' code at their default settings.
This code was written and is maintained by A. Afham.

A quantum channel is estimated by running state tomography on copies of its Choi state and then
repairing the estimate, which is a density matrix but not the Choi state of a channel. The paper
shows that the folklore repair, the **partial normalization**

    N_A(rho) = (1/d_A) (rho_A^{-1/2} (x) I_B) rho (rho_A^{-1/2} (x) I_B),

is the *exact* projection onto the Choi states of channels in fidelity. It is a single congruence,
it preserves the Choi rank, and it costs one `d_A x d_A` eigendecomposition, so any state-tomography
algorithm with a fidelity guarantee lifts to process tomography at the cost of a constant factor in
accuracy.

Open `fpls_notebook.ipynb` (committed with its outputs, so GitHub renders it without running
anything). On Colab, uncomment the clone line in the first cell.

## Files

| file | contents |
|---|---|
| `fpls.py` | the algorithm: measurements, least squares, thresholded density estimate, the fidelity projection in dense and Kraus form, the channels used as test instances |
| `fplsplots.py` | one `fig_*()` per figure of the paper, in the paper's order; each draws its figure and prints the numbers the caption quotes |
| `sweeps.py` | the sweeps behind those figures, run from scratch at a size you choose |
| `fpls_notebook.ipynb` | worked example, every figure, and a check of the implementation of Theorem 1 on a worked example |
| `data/*.json` | the paper's own sweeps, one tidy file per figure, each with a `provenance` block; what a figure falls back to when called with no data |
| `figures/` | rendered output of the `fig_*()` functions, `.pdf` and `.png` |
| `pls.py` | the PLS baseline, run with its authors' code at their default settings |
| `hip/` | the authors' HIP implementation (Kahn, BSD-3-Clause), byte-identical to their repository; see `hip/NOTICE.md` |

## What reproduces what

Every figure and the table are **recomputed in their notebook cell**, the
timing ones included. Each cell opens with two parameter blocks. The first,
`# ---- parameters (reduced: seconds)`, is active and small, so the whole
notebook runs in about a minute and a half. The second,
`# ---- paper parameters (...)`, is commented out and holds the values that
produce the paper's figure: uncomment it and comment out the first block to
reproduce that figure. Its heading states how long the run takes.

The sweeps live in `sweeps.py` and return the same dict shape as the cached
files, so `fig_headline(data=...)` draws the recomputed sweep and
`fig_headline()` draws the shipped one.

Runtimes at the paper's parameters, measured on an Apple M4 (10 cores, 16 GB) under macOS 26.6 with Python 3.14 and NumPy 2.5, single-threaded (`OMP_NUM_THREADS=1`):

<!-- paper-runtimes -->
| item | paper parameters |
|---|---|
| Fig. 2, right | about 3 min |
| Fig. 2, left, without the SDP | about 20 min; with the stored shot counts, a few minutes (estimates) |
| Fig. 2, left, with the SDP | several hours (estimate) |
| Fig. 3 | about 5 s |
| Table 1 | about 2.5 min |
| Fig. 4 | about 5 hours; with the stored shot counts, about an hour (estimates) |
| Fig. 5 | about 3 min |
| Fig. 6 | about 20 s |
| Fig. 7 | about 20 s |
| Fig. 8 | about 1 min |
| Fig. 9 | about 20 min (estimate) |
<!-- /paper-runtimes -->

Table 1 and Fig. 4 at the paper's sizes reach `d_AB = 1024` and need about 4 GB
of memory.

| paper item | recomputed by | paper's parameters |
|---|---|---|
| Fig. 2, right | `sweep_headline` | `nq=4`, `PAPER_SHOTS`, `trials=10` |
| Fig. 2, left | `sweep_cost_ladder` | `dims` 4 to 1024, SDP to 128 |
| Fig. 3 | `sweep_spectra` | `nq=4`, `N=1e8` |
| Table 1 | `sweep_table` | `dims=(64, 1024)`, `N=1e8` |
| Fig. 4 | `sweep_hip_iterations` | `dims` 4 to 1024, four targets, three families |
| Fig. 5 | `sweep_accuracy` | `nq=4`, `PAPER_SHOTS`, `trials=10` |
| Fig. 6 | `sweep_robustness` | `nq=3`, four channels of Choi rank 2, 4, 8, 28 |
| Fig. 7 | `sweep_robustness` | `nq=3`, four channels of full Choi rank |
| Fig. 8 | `sweep_thresholds` | `nq=3` and `nq=4`, shots from `1e3` |
| Fig. 9 | `sweep_contraction` | `dims` 4 to 1024, `trials=10` |
| Theorems 1 and 5 | notebook, sections 1 and 3 | run live already |

The timing figures measure the machine they run on, so their absolute seconds
will differ from the paper's. What those figures claim is the **scaling**, and
that is what reproduces. The diamond-norm SDP curve of Fig. 2 is off by default
because it is slow, several hours at the paper's sizes (about 20 s per instance
at `d_AB = 2^6` and 90 s at `2^7`), and needs `cvxpy`, `clarabel` and `scs`;
set `LADDER_SDP = True` to include it. It is solved by the paper's rules
(`sweeps._time_sdp`), and `sweeps.check_diamond_sdp()` verifies the SDP against
three closed forms: completely positive maps, differences of replacer channels
and differences of unitary channels.

The numbers each figure prints are the ones its caption quotes, each beside the
paper's value. At the small defaults they will not match; they are there so a
run at the paper's parameters can be checked line by line.

## The PLS baseline

The comparison curve is projected least squares with the hyperplane
intersection projection (HIP) of Surawy-Stepney, Kahn, Kueng and Guta,
*Projected least-squares quantum process tomography*, Quantum 6, 844 (2022).
Their implementation is BSD-3-Clause. Its two source files ship in `hip/`
byte-identical to their repository, with their licence in `hip/LICENSE` and
the upstream commit in `hip/UPSTREAM_COMMIT`, and are executed unmodified by
`pls.py`, which runs PLS-QPT as their own simulation driver does, with its
default settings. `hip/NOTICE.md` lists the few departures, none of which
touches the algorithm (chiefly, the projection onto the trace-preserving
subspace written for the paper's ordering of the two systems and for
`d_A != d_B`; `sweeps.check_proj_TP()` verifies it against theirs).

As in the paper, HIP's cost (Fig. 2 left, Fig. 4, the time column of Table 1)
is measured at those defaults, and PLS accuracy and rank (Figs. 2 right, 3, 5,
6, 7) at a tight stopping tolerance of `1e-8` with the iteration cap raised to
2000, since the default tolerance floor of `1e-3` does not shrink with the
shot count. Every sweep computes the PLS arm live, from the same least-squares
estimate as ours, and at the paper's parameters reproduces the shipped PLS
numbers exactly. `sweeps.have_hip()` reports whether `hip/` loaded.

## Requirements

`numpy`, `scipy`, `matplotlib`; `jupyter` to execute the notebook; `cvxpy`, `clarabel` and `scs` only for the slow, optional SDP curve. See `requirements.txt`.

## Citation

```bibtex
@misc{afham2026fastsureiousquantumprocess,
      title={Fast and Sure-ious Quantum Process Tomography},
      author={A. Afham and Sayantan Sen and Marco Tomamichel},
      year={2026},
      eprint={2609.39941},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2609.39941v2},
      note={Version 2}
}
```
