# Method and release semantics

This standalone tool preserves the research idea while making choices explicit where the experimental snapshot was incomplete. It does not load task `common` labels and does not offer a paper-results reproduction claim.

| Stage | Tool behavior |
|---|---|
| Retrieval | Configurable Tanimoto-only default; optional union of top-10 Tanimoto and top-10 ChemBERTa. Canonical self-matches are excluded. |
| Molecular analysis | Compute RDKit properties for arbitrary query SMILES; include any cached reference PubChem properties. Ask for multiple distinct summaries. No mandatory online PubChem request. |
| Molecule selection | Select exactly configured reference_count (default 3, capped at retrieved count). Validate ID membership and uniqueness. Malformed selections are retried, never silently replaced with random molecules. |
| Interaction analysis | One prompt per node contains all selected-reference molecular properties, associated target pockets and available literature, yielding multiple summaries. The research snapshot called the model separately per molecule-target pair. This aggregation reduces the number of model calls. |
| Protein selection | Select only from targets associated with the selected references, corresponding to the filtered reference-target pool. Accept flat protein_id and normalize legacy selected_protein responses, including nested objects. |
| Reward | Obtain exactly reward_samples valid selections and exactly that many boolean judgements. Relative = winning ID frequency / samples; absolute = yes frequency / samples; terminal reward = their equal-weight mean. |
| Search | UCT selection; configurable distinct branches; random simulation with a local RNG; reward backpropagation through ancestors. A previously scored terminal is reused without generating a new independent vote. |
| Ranking | Group unique successful terminal nodes by target ID and take their mean reward, without weighting repeated visits. Break ties by number of unique supporting paths, then target ID. Return up to user-specified top_k. This ranking rule is a documented tool policy, not a recovered original evaluation script. |

The terminal state combines protein selection and the End action. Traces include selected reference IDs, molecular and interaction summaries, per-protein votes, yes-vote counts, relative/absolute rewards and tree visit counts. We retain concise generated summaries, not hidden model reasoning.

Failure semantics: sampling retries are bounded. Insufficient distinct branches can reduce branching with a warning. Insufficient valid reward samples invalidate that path rather than changing denominators. Model/time budgets stop further calls; already successful paths may be returned with `status: partial`. If no valid terminal exists, the tool returns an error. No unscored random fallback is presented as a prediction. Configuration/authentication errors fail immediately.

Prompt size is checked in characters before a model call. The tool never silently truncates the entire prompt; an endpoint can still impose its own token/context limits. Large knowledge-base evidence may require reducing reference counts or editing excerpts. The legacy importer retains at most five interactions per PLIP type and 4,000 characters per literature excerpt; external evidence is not a guarantee of a target's interaction with a new query.

Inference data isolation: query SMILES determine neighbors; label-free query descriptors enter prediction. Known reference interactions supply candidate targets. Ground-truth target sets, ground-truth counts and benchmark splits are absent from the prediction API. Known drug identifiers or public literature may still enable model memorization; this is not a claim of leakage-free scientific validation.

Reproducibility: each run records model name, settings, RDKit version and knowledge-base digest. The local MCTS RNG is seeded per prediction. A remote LLM service does not necessarily provide deterministic sampling. Hybrid embeddings use a model/revision plus 128-token input truncation; pin a model commit for stable future comparisons.

Evaluate prediction quality independently on a held-out dataset before scientific use. The implementation does not convert consistency scores into binding affinities, experimental evidence or clinical recommendations.
