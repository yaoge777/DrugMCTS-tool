from __future__ import annotations

import threading

from .chemistry import canonicalize, fingerprint, similarities
from .config import RetrievalConfig
from .errors import ConfigurationError, InputError
from .knowledge import KnowledgeBase
from .models import Neighbor


class Retriever:
    def __init__(self, kb: KnowledgeBase, config: RetrievalConfig):
        self.kb, self.config = kb, config
        self.ids = sorted(kb.molecules)
        self._embedding_state = None
        self._lock = threading.Lock()

    def _hybrid_scores(self, smiles):
        try:
            import numpy as np
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ConfigurationError('Hybrid retrieval requires pip install ".[hybrid]".') from exc

        # Single lazy load per tool instance. CPU is portable; no implicit CUDA allocation.
        with self._lock:
            if self._embedding_state is None:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        self.config.chemberta_model,
                        revision=self.config.chemberta_revision,
                        trust_remote_code=False,
                    )
                    model = (
                        AutoModel.from_pretrained(
                            self.config.chemberta_model,
                            revision=self.config.chemberta_revision,
                            trust_remote_code=False,
                        )
                        .eval()
                        .to("cpu")
                    )

                    def embed(texts):
                        chunks = []
                        batch_size = self.config.embedding_batch_size
                        for start in range(0, len(texts), batch_size):
                            batch = tokenizer(
                                texts[start : start + batch_size],
                                padding=True,
                                truncation=True,
                                max_length=128,
                                return_tensors="pt",
                            )
                            with torch.inference_mode():
                                chunks.append(model(**batch).last_hidden_state[:, 0, :].numpy())
                        return np.concatenate(chunks, axis=0)

                    references = embed([self.kb.molecules[mid].smiles for mid in self.ids])
                    self._embedding_state = (embed, references)
                except (OSError, ValueError) as exc:
                    raise ConfigurationError(
                        "Could not load ChemBERTa; check model path/revision and network."
                    ) from exc
            embed, references = self._embedding_state
            query = embed([smiles])[0]
            denominators = np.linalg.norm(references, axis=1) * np.linalg.norm(query)
            cosine = (references @ query) / np.maximum(denominators, 1e-12)
            if not np.isfinite(cosine).all():
                raise InputError("ChemBERTa returned non-finite embeddings.")
            return dict(zip(self.ids, (float(x) for x in cosine)))

    def search(self, smiles: str) -> list[Neighbor]:
        query = canonicalize(smiles)
        scores = dict(
            zip(self.ids, similarities(fingerprint(smiles), [self.kb.fingerprints[mid] for mid in self.ids]))
        )
        # Exclude canonical aliases too, not only the query's ID.
        eligible = [mid for mid in self.ids if self.kb.canonical[mid] != query]
        tanimoto_ids = sorted(
            (m for m in eligible if scores[m] >= self.config.min_tanimoto),
            key=lambda mid: (-scores[mid], mid),
        )[: self.config.top_n]
        cosines = self._hybrid_scores(smiles) if self.config.mode == "hybrid" else {}
        cosine_ids = (
            sorted(eligible, key=lambda mid: (-cosines[mid], mid))[: self.config.top_n] if cosines else []
        )
        # Stable union: top Tanimoto, followed by new ChemBERTa neighbors.
        selected = list(dict.fromkeys(tanimoto_ids + cosine_ids))
        if not selected:
            raise InputError("No reference molecules remain after self-exclusion and similarity filtering.")
        return [
            Neighbor(
                id=mid,
                smiles=self.kb.molecules[mid].smiles,
                tanimoto=scores[mid],
                cosine=cosines.get(mid),
                protein_ids=self.kb.molecules[mid].protein_ids,
            )
            for mid in selected
        ]
