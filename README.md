## Reference File for NMODL Tests

One aspect of testing NMODL is golden tests of translated MOD files. These
"tests" alert the developer that the code generated for a particular MOD file
changed. This is often desirable, just not without noticing.

This repository stores these reference values and is includes in NMODL via a
submodule. Note that all of the code generated here is auto-generated from
MOD file translators from NMODL and NEURON projects.

Github as a long standing annoying bug that prevents us from fast-forward
merging PRs. This causes the commit SHA to change on merge. Meaning the SHA in
the NMODL repo points to the PR branch, not the corresponding commit on `main`.

For out particular usecase, we should require that those commits remain in the
repo.

## License

For license details, see LICENSE.txt. Note that this repository contains code
auto-generated by MOD file translators from the NMODL and NEURON projects.
Therefore, it is additionally subject to the license terms specified by those tools.

## Funding & Acknowledgment

This development is supported by funding to the Blue Brain Project, a research
center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss
government's ETH Board of the Swiss Federal Institutes of Technology.

Copyright © 2024 Blue Brain Project/EPFL

