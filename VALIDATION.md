# Delivery validation — 2026-09-22

Validated on macOS arm64, Python 3.11.8, RDKit 2026.3.6, Pydantic 2.13.5, HTTPX 0.28.1, MCP SDK 1.30.0 and pytest 9.1.1.

- **21 tests passed**. Coverage includes canonical query/alias exclusion, malformed SMILES, hybrid retrieval union with fixed embedding scores, knowledge-base references and IDs, config-relative paths, strict JSON parsing, reproducible demo ranking, UCT and backpropagation, terminal-revisit accounting, nested protein ID normalization, equal-weight rewards, invalid selection/reward rejection, request budgets, model payload/usage/error handling, import read-only behavior and output collision protection.
- **Ruff check passed** for source and tests; code was formatted with Ruff.
- **CLI and Python API** executed the synthetic demo with 12 completed rollouts and three distinct synthetic targets. CLI was also tested from a different working directory.
- **MCP stdio** was started as a real subprocess. Initialization, discovery of all three tools, info, retrieval, prediction and invalid-input handling were exercised through the official SDK client.
- **HTTP adapter** was tested through HTTPX's mock transport, including a complete prediction driven by chat-completions-shaped responses. This validates the request/response integration, not an external model service.
- **Original data import** succeeded: DrugBank contains 4,511 reference molecules / 2,277 PDB targets; KIBA contains 1,561 molecules / 128 PDB targets. These are tool knowledge-base sizes, not benchmark task counts. Original evaluation answers are not imported.
- **Real-data retrieval** returned ten DrugBank neighbors for the example query with no LLM call. Local doctor checks passed.
- **Packaging** produced a wheel and source archive. A fresh environment installed the wheel and successfully exercised the Python API and MCP prediction outside the source directory. The wheel contains package source plus synthetic fixture; neither distribution contains local scientific data, results or the original credential.
- **Original project preservation**: all 34 source/config/template/data files recorded by the prior audit retained their SHA-256 hashes. The importer reads caches as data and does not import legacy modules or write in the original directory.

Not performed: paid/live model inference, a full benchmark reproduction, a real ChemBERTa weight download/inference, tests on Linux/Windows or the CI Python versions other than this local Python 3.11. The optional hybrid implementation has unit-tested fusion; its external model loading remains an integration step for the selected environment.

The default endpoint is a configurable local service address, not a bundled running LLM. Configure a real model endpoint to use real target prediction. The demo is explicitly synthetic, and the library records that status in results.
