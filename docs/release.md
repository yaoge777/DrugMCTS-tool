# Publication notes

- Publish this standalone package at `https://github.com/yaoge777/DrugMCTS-tool`. The package, installation instructions, and CI workflow are at the repository root. Keep the experimental snapshot separate.
- Package wheels contain only `drugmcts` source and the synthetic demo fixture. Source archives additionally include documentation, examples and tests. `local-data`, outputs, environments and private local config are excluded.
- The MIT license applies to this new tool's source and synthetic fixtures. Original datasets, cached abstracts, structures and third-party models retain their own terms. Provide acquisition/import instructions for those artifacts.
- Preserve the method differences and synthetic-demo labeling. Do not describe demo scores or mocked endpoint tests as reproduced research results.
- Configure and verify your chosen real model service before making scientific predictions. No live paid inference was required to build the tool.

Build with `python -m build`; test the installed wheel from outside the repository. `VALIDATION.md` records the checks performed on this delivery.
