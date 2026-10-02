# Origin

`projections.py` and `_projections_with_introspection.py` are byte-identical
copies of the files of the same name in `PLSPQT_article_code/` of
<https://github.com/Hannoskaj/Hyperplane_Intersection_Projection> (Jonas Kahn,
BSD-3-Clause, see `LICENSE`; commit in `UPSTREAM_COMMIT`), the implementation
of the hyperplane intersection projection (HIP) of Surawy-Stepney, Kahn, Kueng
and Guta, *Projected least-squares quantum process tomography*, Quantum 6, 844
(2022). They are the code with which every PLS number in `data/*.json` and in
the paper was computed.

They are executed unmodified by `pls.py`, which runs PLS-QPT as the authors'
own simulation driver (`generate_simulations` in their
`experiment_generation.py`) does, with its default settings. The only
departures, none of which touches the algorithm:

- `proj_TP` is replaced, at run time, by the same orthogonal projection onto
  the trace-preserving subspace written for the paper's Choi ordering (input
  system first; theirs takes the opposite ordering) and for `d_A != d_B`
  (theirs assumes `d_A = d_B`);
- `numpy.row_stack`, removed in NumPy 2.5, is aliased to `numpy.vstack`;
- the HDF5 group into which the routine logs is replaced by a stand-in that
  discards what it is given, and the routine's per-iteration prints are
  silenced;
- the files import `tables`, `pandas` and `numexpr` without using them on this
  code path; if those packages are not installed, empty stand-ins are
  registered so that the import succeeds.
