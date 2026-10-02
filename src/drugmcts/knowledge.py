from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from pathlib import Path

from pydantic import ValidationError

from .chemistry import canonicalize, fingerprint
from .errors import ConfigurationError, InputError
from .models import KnowledgeData


class KnowledgeBase:
    def __init__(self, data: KnowledgeData, digest: str = ""):
        self.data = data
        self.digest = digest or hashlib.sha256(data.model_dump_json().encode()).hexdigest()
        self.molecules = {m.id: m for m in data.molecules}
        self.proteins = {p.id: p for p in data.proteins}
        if len(self.molecules) != len(data.molecules) or len(self.proteins) != len(data.proteins):
            raise InputError("Knowledge base molecule and protein IDs must each be unique.")
        self.canonical = {}
        self.fingerprints = {}
        for item in data.molecules:
            missing = set(item.protein_ids) - self.proteins.keys()
            if missing:
                raise InputError(f"Molecule {item.id} references missing protein IDs: {sorted(missing)}")
            if len(item.protein_ids) != len(set(item.protein_ids)):
                raise InputError(f"Molecule {item.id} has duplicate protein IDs.")
            try:
                self.canonical[item.id] = canonicalize(item.smiles)
                self.fingerprints[item.id] = fingerprint(item.smiles)
            except InputError as exc:
                raise InputError(f"Knowledge base molecule {item.id} has invalid SMILES.") from exc

    @classmethod
    def from_file(cls, path: str | Path):
        try:
            raw = Path(path).read_bytes()
            data = KnowledgeData.model_validate_json(raw)
        except (OSError, ValueError, ValidationError) as exc:
            raise ConfigurationError(f"Cannot load knowledge base {path}; run drugmcts validate-kb.") from exc
        return cls(data, hashlib.sha256(raw).hexdigest())

    @classmethod
    def demo(cls):
        raw = files("drugmcts").joinpath("resources/demo.json").read_bytes()
        return cls(KnowledgeData.model_validate_json(raw), hashlib.sha256(raw).hexdigest())

    def describe(self):
        return {
            "name": self.data.name,
            "molecules": len(self.molecules),
            "proteins": len(self.proteins),
            "demonstration_only": self.data.demonstration_only,
            "sha256": self.digest,
            "provenance": self.data.provenance,
            "proteins_without_pocket": sum(not p.pocket for p in self.proteins.values()),
            "proteins_without_literature": sum(not p.literature for p in self.proteins.values()),
        }


def write_json(path: str | Path, value: dict, force: bool = False):
    """Create a JSON artifact, refusing to clobber user data by default."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    try:
        with path.open("w" if force else "x", encoding="utf-8") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise InputError(f"Output already exists: {path}. Choose another path or pass --force.") from exc
