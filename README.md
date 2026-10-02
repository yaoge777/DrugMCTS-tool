# DrugMCTS Tool

A standalone Python, command-line and MCP tool for exploring drug-target hypotheses from SMILES with molecular retrieval and Monte Carlo tree search.

[中文说明](README.zh-CN.md) · [Method and differences from the research snapshot](docs/method.md) · [Knowledge-base format](docs/knowledge-base.md)

This package lives independently from the original experimental project. It does not import or modify that project's Python code. Predictions use a configured **OpenAI-compatible chat-completions endpoint**, so no local LLM weights or CUDA are needed for the default retrieval mode. The tool's ranking is a research hypothesis ranking, not a binding-affinity prediction or a calibrated probability. It does not require the number of known targets.

## Install and try it

Python 3.11+ is required.

```bash
git clone https://github.com/yaoge777/DrugMCTS-tool.git
cd DrugMCTS-tool
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[mcp]'
drugmcts demo --trace --output outputs/demo.json
```

The demo performs real RDKit retrieval and executes the MCTS control flow, with a **synthetic knowledge base and deterministic fake model responses**. No API key or network model call is required. Output is explicitly marked `demonstration_only: true`; it is not a scientific prediction.

## Use your experimental data

The existing DrugMCTS directory is an optional import source, not a runtime dependency. The importer only reads the raw interaction tables, mapping TSV files and scientific caches. It does not copy code, credentials, checkpoints, bytecode or evaluation answers.

```bash
drugmcts prepare-legacy --legacy-dir /absolute/path/to/original/DrugMCTS --dataset drugbank --output local-data/drugbank.json
drugmcts prepare-legacy --legacy-dir /absolute/path/to/original/DrugMCTS --dataset kiba --output local-data/kiba.json
drugmcts validate-kb local-data/drugbank.json
```

`local-data/` is ignored by Git and excluded from distribution archives. Bring your own appropriately licensed knowledge data; the package only ships synthetic demo data. Users without the original project can supply their own JSON knowledge base using the [documented format](docs/knowledge-base.md).

## Configure a real model

Edit `examples/config.toml` to point at your model endpoint and imported knowledge base. Relative paths resolve against the **config file's directory**, including when invoked from another working directory or an MCP client.

```toml
knowledge_base = "../local-data/drugbank.json"

[llm]
model = "Qwen/Qwen2.5-7B-Instruct"
base_url = "http://localhost:8000/v1"
api_key_env = ""
temperature = 0.8
```

For an authenticated endpoint, set `api_key_env = "DRUGMCTS_API_KEY"` and supply the secret through the process environment. No real key belongs in a config file. `.env.example` documents the variable; the tool intentionally does not auto-load arbitrary `.env` files. The endpoint must accept the chat-completions fields `model`, `messages`, `n`, `temperature` and `max_tokens`. JSON mode is optional and disabled by default for endpoint compatibility. Azure's deployment-specific REST routes are not automatically translated.

```bash
drugmcts doctor --config examples/config.toml
drugmcts retrieve --config examples/config.toml --smiles 'CC(=O)Oc1ccccc1C(=O)O'
drugmcts predict --config examples/config.toml --smiles 'CC(=O)Oc1ccccc1C(=O)O' --top-k 5 --trace --output outputs/prediction.json
```

`doctor` checks local configuration/data and credential availability without contacting the LLM. `retrieve` does not call the LLM; hybrid retrieval may download an embedding model. `predict` sends the query structure, molecular properties and selected knowledge-base evidence to the configured endpoint. Each prediction has a bounded request count, per-request timeout and search time budget. No fixed monetary cost is assumed.

Commands print JSON to stdout and errors/progress to stderr. Existing output files require explicit `--force`. A prediction returns fewer than `top_k` results if fewer distinct targets reached valid terminal paths; it never fills the ranking with unscored targets. `status: partial` means the rollout budget was not completed; inspect `warnings` and `search`.

## Python API

```python
from drugmcts import DrugMCTS

tool = DrugMCTS.from_config("examples/config.toml")
neighbors = tool.retrieve("CC(=O)Oc1ccccc1C(=O)O")
result = tool.predict("CC(=O)Oc1ccccc1C(=O)O", top_k=5, include_trace=True)
for target in result.targets:
    print(target.protein_id, target.uniprot_ids, target.score, target.evidence_summary)
```

`Config`, `DrugMCTS` and `Prediction` are public exports. The result is a Pydantic model; use `model_dump()` / `model_dump_json()`. Custom model adapters can implement the `Backend` protocol in `drugmcts.llm` and be supplied via `backend_factory`; each prediction receives and closes its own backend instance. Imports never load an LLM or open a network connection.

## MCP

Install the optional `mcp` extra. Register an MCP stdio process using your client's configuration format, for example:

```json
{
  "mcpServers": {
    "drugmcts": {
      "command": "/absolute/path/DrugMCTS-tool/.venv/bin/drugmcts",
      "args": ["serve", "--config", "/absolute/path/DrugMCTS-tool/examples/config.toml"]
    }
  }
}
```

To test without a model service, replace the args with `["serve", "--demo"]`.

- `drugmcts_info()` — describe the knowledge base, model and limits without model calls.
- `retrieve_similar_molecules(smiles)` — find reference molecules and associated targets.
- `predict_targets(smiles, top_k=5, include_trace=false)` — run a bounded MCTS prediction.

The server uses stdio only, exposes no public HTTP port, and accepts no client-supplied filesystem/output paths. It permits one prediction at a time. For a real endpoint, allow a sufficiently long tool timeout in the MCP host (the default search budget is 900 seconds). For authenticated model services, pass the credential environment variable to the server process using your host's secret configuration.

## Hybrid molecular retrieval

```bash
python -m pip install -e '.[mcp,hybrid]'
```

Set `retrieval.mode = "hybrid"`. This forms a stable, deduplicated union of each metric's top `top_n`: Morgan/Tanimoto and ChemBERTa cosine. Canonical self-matches are excluded in both branches. The portable default is `tanimoto`; it must not be described as the paper's full dual retrieval setup.

ChemBERTa is loaded lazily on CPU and cached in memory per tool instance, uses the first token of the last hidden state and truncates SMILES at 128 tokens as in the experimental helper. It requires an initial Hugging Face download or an explicit local model path. Pin `chemberta_revision` to a commit to freeze the embedding model for reproducible runs. Python tree randomness is seeded; external model-service sampling and hardware may still be nondeterministic.

## Output

`targets` includes the target ID, available UniProt mappings, average reward, relative/absolute components, number of unique supporting paths, tree visits, reference molecules, a concise evidence summary and source links. `retrieval`, `search`, `usage`, `provenance` and optional `traces` make the run inspectable. Token usage is taken from endpoint responses; `requests_without_token_usage` records missing accounting. No price or biological probability is inferred from token or reward counts.

Legacy targets are **PDB structure IDs** with associated UniProt IDs. The importer preserves this identity rather than pretending that different structures are unique proteins. A custom knowledge base may use UniProt IDs as its primary target identity.

## Development and publication

```bash
python -m pip install -e '.[dev,mcp]'
pytest
ruff check src tests
python -m build
```

The package is MIT-licensed. This new license covers the tool code and synthetic fixtures, not third-party scientific datasets or the original research snapshot. Review the [release notes](docs/release.md) before publishing a source archive or package. This is a usable implementation derived from the research design, **not a claim that the paper's reported benchmark numbers were reproduced**.

The MCP adapter targets the maintained [official MCP Python SDK v1](https://py.sdk.modelcontextprotocol.io/v1/) with a `<2` bound. Molecular operations use [RDKit](https://www.rdkit.org/docs/GettingStartedInPython.html); optional embeddings use [Transformers Auto classes](https://huggingface.co/docs/transformers/model_doc/auto).

Reference: [DrugMCTS, arXiv:2507.07426](https://arxiv.org/abs/2507.07426).
