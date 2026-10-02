# Knowledge-base schema (v1)

Use a UTF-8 JSON file. Each molecule and target ID must be unique, every molecule must have a valid SMILES and at least one associated target, and each associated ID must exist in `proteins`. Invalid records fail validation with a useful error. Target IDs can be UniProt accessions or PDB IDs, but use a consistent identity policy and explain it in `provenance`.

```json
{
  "schema_version": 1,
  "name": "My curated reference set",
  "provenance": "Describe sources, dates, filtering and permissions here.",
  "demonstration_only": false,
  "molecules": [
    {"id": "REFERENCE_1", "smiles": "CCO", "protein_ids": ["TARGET_1"], "properties": {}}
  ],
  "proteins": [
    {"id": "TARGET_1", "name": "Replace with a real target", "uniprot_ids": [], "pocket": "Curated structural evidence", "literature": ["Curated evidence excerpt"], "sources": []}
  ]
}
```

This schema example is not a valid scientific association. Use the packaged demo for software tests and real curated evidence for predictions. A useful reference set needs multiple distinct molecules; the query and canonical aliases are excluded from retrieval. The software does not invent target associations or automatically fetch a complete scientific database.

`pocket`, `literature`, `sources`, `name`, `uniprot_ids` and molecular `properties` are optional fields. Missing pocket/literature information appears in result warnings. A protein with no sources does not gain citations from the model: result source links come from the knowledge base.

Legacy import reconstructs positive associations using the last PDB mapping for each UniProt and targets with existing PLIP caches. It excludes known PubMed error/empty-result cache strings, retains original source links where available and computes canonical molecules with RDKit. It does not carry over evaluation `common` labels or require the original code after export. Cache and original-structure limitations remain properties of this imported evidence.
