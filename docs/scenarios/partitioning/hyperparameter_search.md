# Distributed hyperparameter search

`HyperparameterSearchEngine` creates a deterministic Cartesian grid. Parameter names are sorted; each nonempty value axis preserves caller order; `itertools.product` then assigns stable zero-based trial IDs. Claimed trial bands invoke an injected objective on a detached name/value mapping. Scores must be finite. Ranking is deterministic by objective direction and trial ID for ties.

Use `parameters_for(id)` to inspect the ID mapping, `evaluate_claim()` for explicit workers, or `run()` to evaluate the grid. Nested parameter values are cloned at every caller and objective boundary. `ranking()` and `snapshot()` expose detached results without insertion-order dependence; `audit_snapshot()` also captures private claims/events but is not restorable because it omits the grid and objective identity. Empty spaces/axes, invalid names, non-sequence axes, invalid IDs, non-callable objectives, and non-finite scores fail explicitly.

The objective is arbitrary local Python and checkpoints do not serialize it or its environment. A distributed tuner must version objective/data, persist trial definitions and scores, enforce claim fencing, define timeout/failure policy, and prevent duplicate external side effects. The local engine supplies deterministic planning and ranking only.

The example ranks a four-trial grid. The smoke executes 200 real trials and verifies every ID-to-parameter mapping against an independent Cartesian-product oracle.
