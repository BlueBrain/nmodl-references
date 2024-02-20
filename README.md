# Reference File for NMODL Tests
One aspect of testing NMODL is golden tests of translated MOD files. These
"tests" alert the developer that the code generated for a particular MOD file
changed. This is often desirable, just not without noticing.

This repository stores these reference values and is includes in NMODL via a
submodule.

Github as a long standing annoying bug that prevents us from fast-forward
merging PRs. This causes the commit SHA to change on merge. Meaning the SHA in
the NMODL repo points to the PR branch, not the corresponding commit on `main`.

For out particular usecase, we should require that those commits remain in the
repo.
