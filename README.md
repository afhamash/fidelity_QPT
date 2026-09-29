# Fidelity Projected Least Squares

Companion code for **Fast and Sure-ious Quantum Process Tomography**, A. Afham, S. Sen and M. Tomamichel.
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
| `fpls_notebook.ipynb` | worked example, every figure, and a from-scratch verification of Theorem 1 |
| `data/*.json` | the paper's own sweeps, one tidy file per figure, each with a `provenance` block; what a figure falls back to when called with no data |
| `figures/` | rendered output of the `fig_*()` functions, `.pdf` and `.png` |

## What reproduces what

Every figure and the table are **recomputed in their notebook cell**, the
wall-clock ones included. Each cell opens with its parameters and states, next
to them, the values that produce the paper's figure; the defaults are small so
the whole notebook runs in about fifteen seconds. The sweeps live in
`sweeps.py` and return the same dict shape as the cached files, so
`fig_headline(data=...)` draws the recomputed sweep and `fig_headline()` draws
the shipped one.

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
| Theorems 1 and 6 | notebook, section 3 | run live already |

The timing figures measure the machine they run on, so their absolute seconds
will differ from the paper's. What those figures claim is the **scaling**, and
that is what reproduces. The diamond-norm SDP curve of Fig. 2 is off by default
because it needs `cvxpy`; set `LADDER_SDP = True` to include it.

The numbers each figure prints are the ones its caption quotes, each beside the
paper's value. At the small defaults they will not match; they are there so a
run at the paper's parameters can be checked line by line.

## The PLS baseline

The comparison curve is projected least squares with the hyperplane
intersection projection of Surawy-Stepney, Kahn, Kueng and Guta. That
implementation is theirs, BSD-3-Clause, and is not vendored here. Clone it
beside this folder and every sweep computes that arm live, from the same
least-squares estimate as ours:

```
git clone https://github.com/Hannoskaj/Hyperplane_Intersection_Projection.git
```

`sweeps.have_hip()` reports whether it was found; `$HIP_REPO` overrides the
search. Without it the cells still run: the cached arm is used when it is the
same sweep, and otherwise that curve is simply omitted.

## Requirements

`numpy`, `scipy`, `matplotlib`; `jupyter` to execute the notebook. See `requirements.txt`.
