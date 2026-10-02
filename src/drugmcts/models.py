from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Protein(Record):
    id: str = Field(min_length=1)
    name: str = ""
    uniprot_ids: list[str] = Field(default_factory=list)
    pocket: str = ""
    literature: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


class Molecule(Record):
    id: str = Field(min_length=1)
    smiles: str = Field(min_length=1, max_length=10000)
    protein_ids: list[str] = Field(min_length=1)
    properties: dict[str, Any] = Field(default_factory=dict)


class KnowledgeData(Record):
    schema_version: Literal[1] = 1
    name: str
    provenance: str
    demonstration_only: bool = False
    molecules: list[Molecule] = Field(min_length=1)
    proteins: list[Protein] = Field(min_length=1)


class Neighbor(Record):
    id: str
    smiles: str
    tanimoto: float
    cosine: float | None = None
    protein_ids: list[str]


class Target(Record):
    protein_id: str
    name: str
    uniprot_ids: list[str]
    score: float = Field(ge=0, le=1)
    relative_score: float = Field(ge=0, le=1)
    absolute_score: float = Field(ge=0, le=1)
    supporting_paths: int
    visits: int
    reference_molecule_ids: list[str]
    evidence_summary: str
    sources: list[str]


class Prediction(Record):
    schema_version: Literal[1] = 1
    status: Literal["completed", "partial", "demo"]
    demonstration_only: bool
    query: dict[str, str]
    targets: list[Target]
    retrieval: dict[str, Any]
    search: dict[str, Any]
    usage: dict[str, Any]
    provenance: dict[str, Any]
    warnings: list[str]
    traces: list[dict[str, Any]] = Field(default_factory=list)
    score_interpretation: str = (
        "Mean of LLM relative selection consistency and yes-vote frequency, averaged across unique "
        "successful paths per target. This is not a calibrated binding probability or affinity."
    )
