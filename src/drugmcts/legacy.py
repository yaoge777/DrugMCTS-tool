"""Read an experiment snapshot into a standalone knowledge base; never import its Python code."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .chemistry import canonicalize
from .errors import InputError
from .knowledge import KnowledgeBase, write_json
from .models import KnowledgeData, Molecule, Protein


def pocket_text(path: Path) -> str:
    sections = []
    name = None
    count = 0
    header = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith('"""'):
            name = line.strip(":").strip('"').lower()
            count, header = 0, []
            continue
        if not line or name is None:
            continue
        values = re.findall(r"\([^)]*\)|\S+", line)
        if not header:
            header = values
            continue
        if count >= 5 or len(values) != len(header):
            continue
        row = dict(zip(header, values))
        if name == "hydrophobic":
            sections.append(
                f"Hydrophobic contact: {row.get('RESTYPE')}-{row.get('RESCHAIN')}, "
                f"ligand chain {row.get('RESCHAIN_LIG')}, distance {row.get('DIST')} Å."
            )
        elif name == "hbond":
            sections.append(
                f"Hydrogen bond: {row.get('RESTYPE')}, {row.get('DONORTYPE')} to "
                f"{row.get('ACCEPTORTYPE')}, H-A distance {row.get('DIST_H-A')} Å, "
                f"donor angle {row.get('DON_ANGLE')} degrees."
            )
        elif name == "pistacking":
            sections.append(
                f"Pi stacking: {row.get('RESTYPE')}-{row.get('RESCHAIN')}, "
                f"centroid distance {row.get('CENTDIST')} Å, angle {row.get('ANGLE')} degrees, "
                f"offset {row.get('OFFSET')} Å."
            )
        count += 1
    return " ".join(sections)


def clean_abstract(path: Path) -> str | None:
    if not path.exists():
        return None
    text = json.loads(path.read_text()).get("abstract")
    if not isinstance(text, str) or not text.strip():
        return None
    if any(marker in text.lower() for marker in ["pubmed exception", "no good pubmed result"]):
        return None
    # Explicit bounded excerpt, retain source URL in the exported protein record.
    return text.strip()[:4000]


def prepare_legacy(legacy_dir: str | Path, dataset: str, output: str | Path, force=False) -> dict:
    root, output = Path(legacy_dir).resolve(), Path(output).resolve()
    if output == root or root in output.parents:
        raise InputError(
            "Output must be outside the legacy experiment directory; it is treated as read-only."
        )
    if dataset not in {"drugbank", "kiba"}:
        raise InputError("dataset must be drugbank or kiba")
    if output.exists() and not force:
        raise InputError(f"Output already exists: {output}. Choose another output or use --force.")
    suffix = "_KIBA" if dataset == "kiba" else ""
    raw = root / "data" / ("KIBA.txt" if suffix else "cleaned_DrugBank.txt")
    map_path = root / "data" / f"uniprot_pdb_map{suffix}.tsv"
    plip = root / "experiment" / ("plip" + suffix)
    if not raw.is_file() or not map_path.is_file() or not plip.is_dir():
        raise InputError("Missing raw dataset, UniProt-PDB mapping or PLIP cache directory.")
    mapping, uniprot_by_pdb = {}, defaultdict(set)
    with map_path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not {"From", "To"} <= set(reader.fieldnames or []):
            raise InputError("Mapping TSV must contain From and To columns.")
        for row in reader:
            mapping[row["From"]] = row["To"]
    for uni, pid in mapping.items():
        if re.fullmatch(r"[A-Za-z0-9]{4}", pid):
            uniprot_by_pdb[pid].add(uni)
    available = {p.stem for p in plip.glob("*.txt")}
    molecule_smiles, associations = {}, defaultdict(set)
    statistics = Counter()
    with raw.open() as handle:
        for line in handle:
            fields = line.split()
            if len(fields) != 5:
                statistics["malformed_rows_skipped"] += 1
                continue
            mid, uni, smiles, _, label = fields
            pid = mapping.get(uni)
            if label != "1" or pid not in available or pid not in uniprot_by_pdb:
                continue
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", mid):
                statistics["invalid_ids_skipped"] += 1
                continue
            molecule_smiles.setdefault(mid, smiles)
            associations[mid].add(pid)
    molecules = []
    for mid, pids in sorted(associations.items()):
        try:
            smiles = canonicalize(molecule_smiles[mid])
        except InputError:
            statistics["invalid_molecules_skipped"] += 1
            continue
        cache = root / "experiment" / ("mol_analysis" + suffix) / (mid + ".txt")
        properties = json.loads(cache.read_text()) if cache.exists() else {}
        if not isinstance(properties, dict):
            properties = {}
        molecules.append(Molecule(id=mid, smiles=smiles, protein_ids=sorted(pids), properties=properties))
    proteins = []
    for pid in sorted({pid for m in molecules for pid in m.protein_ids}):
        literature = []
        sources = [f"https://www.rcsb.org/structure/{pid}"]
        for name in ["pdb_desc", "pubmed"]:
            cache = root / "experiment" / (name + suffix) / (pid + ".json")
            abstract = clean_abstract(cache)
            if abstract and abstract not in literature:
                literature.append(abstract)
            else:
                statistics["missing_empty_or_duplicate_literature"] += 1
            if cache.exists():
                pmid = json.loads(cache.read_text()).get("pmid")
                if pmid is not None and str(pmid).isdigit():
                    sources.append(f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
        proteins.append(
            Protein(
                id=pid,
                name=f"PDB {pid}",
                uniprot_ids=sorted(uniprot_by_pdb[pid]),
                pocket=pocket_text(plip / (pid + ".txt")),
                literature=literature,
                sources=sources,
            )
        )
    data = KnowledgeData(
        name=f"{dataset} legacy import",
        provenance=(
            "Imported from user-supplied experiment data. Positive labels only; last UniProt-to-PDB mapping; "
            "retain targets with existing PLIP caches. PDB IDs are structural representatives, not unique proteins. "
            f"Raw dataset SHA256={hashlib.sha256(raw.read_bytes()).hexdigest()}. "
            "No evaluation ground truth is used by prediction."
        ),
        molecules=molecules,
        proteins=proteins,
    )
    kb = KnowledgeBase(data)
    write_json(output, data.model_dump(), force)
    kb.digest = hashlib.sha256(output.read_bytes()).hexdigest()
    return {"output": str(output), **kb.describe(), "import_statistics": dict(statistics)}
