"""Option A, the team's final model: Tobias Ruff's code, ported from branch hybrid_v3_tr.

The files here are src/option_a.py and the six modules it imports, as committed at
863e751fe96f4b4350fbd4f12fc2527f68c36318, unchanged except for two path constants in
hybrid_v3.py (ROOT, DATA_DIR) so that they read this project's data. They import each other
as top-level modules (option_a.py puts this directory on sys.path), as on the branch.

Option A reads, besides the raw data, cached outputs of earlier scripts on that branch: the
sensor-assimilated core (outputs/core_assim), its cached parts and the out-of-fold
predictions of the two members (outputs/stack_v5/cv), and, for a printed comparison only,
the v3 and v7 holdout predictions. They were built by a chain of experiment scripts and are
imported as committed at 863e751 into data/external/hybrid_v3_tr/outputs/ (DVC-tracked,
not in git: they contain measured values). Run through bioreactor_decoding.modeling.option_a.
Not linted or formatted (pyproject.toml), so that the code stays as on the branch.
"""
