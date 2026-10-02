from __future__ import annotations

import tomllib
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import Field, ValidationError, model_validator

from .errors import ConfigurationError
from .models import Record


class LLMConfig(Record):
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    base_url: str = "http://localhost:8000/v1"
    api_key_env: str = ""
    temperature: float = Field(default=0.8, ge=0, le=2)
    max_output_tokens: int = Field(default=1024, ge=64, le=32768)
    timeout_seconds: float = Field(default=60, gt=0, le=600)
    json_mode: bool = False

    @model_validator(mode="after")
    def valid_endpoint(self):
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("base_url must be an http(s) API base URL ending with /v1 if required")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Do not embed credentials, query parameters or fragments in base_url")
        if not self.model.strip():
            raise ValueError("model must not be empty")
        return self


class RetrievalConfig(Record):
    mode: str = "tanimoto"
    top_n: int = Field(default=10, ge=1, le=100)
    min_tanimoto: float = Field(default=0.0, ge=0, le=1)
    chemberta_model: str = "DeepChem/ChemBERTa-77M-MLM"
    chemberta_revision: str = "main"
    embedding_batch_size: int = Field(default=32, ge=1, le=256)

    @model_validator(mode="after")
    def valid_mode(self):
        if self.mode not in {"tanimoto", "hybrid"}:
            raise ValueError("retrieval.mode must be tanimoto or hybrid")
        return self


class SearchConfig(Record):
    rollouts: int = Field(default=12, ge=1, le=100)
    branches: int = Field(default=4, ge=1, le=8)
    reference_count: int = Field(default=3, ge=1, le=20)
    reward_samples: int = Field(default=4, ge=1, le=16)
    exploration_constant: float = Field(default=3.0, ge=0, le=100)
    seed: int = 42
    response_attempts: int = Field(default=3, ge=1, le=5)
    max_llm_requests: int = Field(default=120, ge=1, le=1000)
    max_seconds: float = Field(default=900, gt=0, le=7200)
    max_prompt_chars: int = Field(default=60000, ge=2000, le=1000000)


class Config(Record):
    knowledge_base: str = "local-data/drugbank.json"
    llm: LLMConfig = Field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)

    @classmethod
    def from_file(cls, path: str | Path) -> Config:
        path = Path(path).expanduser().resolve()
        try:
            with path.open("rb") as handle:
                cfg = cls.model_validate(tomllib.load(handle))
            kb = Path(cfg.knowledge_base).expanduser()
            cfg.knowledge_base = str((path.parent / kb).resolve() if not kb.is_absolute() else kb)
            return cfg
        except (OSError, ValueError, ValidationError) as exc:
            # ValidationError includes original input values; do not print config content.
            raise ConfigurationError(
                f"Cannot read/validate config {path}. Check TOML keys and value ranges."
            ) from exc
