"""Run after installing the package; no credentials, intentionally synthetic."""

from drugmcts import DrugMCTS

tool = DrugMCTS.demo()
result = tool.predict("CC(=O)Oc1ccccc1C(=O)O", top_k=3, include_trace=True)
print(result.model_dump_json(indent=2))

# Real use after importing knowledge data and configuring a model endpoint:
# tool = DrugMCTS.from_config("examples/config.toml")
# result = tool.predict("YOUR_SMILES", top_k=5)
