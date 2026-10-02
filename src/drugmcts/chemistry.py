from __future__ import annotations

from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from .errors import InputError

GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def parse_smiles(smiles: str):
    if not isinstance(smiles, str) or not smiles.strip() or len(smiles) > 10000:
        raise InputError("Provide a non-empty SMILES string of at most 10,000 characters.")
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise InputError("Invalid SMILES. Supply a molecular structure rather than a drug name.")
    return mol


def canonicalize(smiles: str) -> str:
    return Chem.MolToSmiles(parse_smiles(smiles), isomericSmiles=True)


def fingerprint(smiles: str):
    return GENERATOR.GetFingerprint(parse_smiles(smiles))


def similarities(query_fp, references) -> list[float]:
    return list(DataStructs.BulkTanimotoSimilarity(query_fp, references))


def describe(smiles: str) -> dict:
    mol = parse_smiles(smiles)
    return {
        "source": "RDKit computed descriptors (not experimental PubChem measurements)",
        "smiles": Chem.MolToSmiles(mol),
        "molecular_weight": round(Descriptors.MolWt(mol), 3),
        "logP": round(Descriptors.MolLogP(mol), 3),
        "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 3),
        "hbond_donors": Lipinski.NumHDonors(mol),
        "hbond_acceptors": Lipinski.NumHAcceptors(mol),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "scaffold": MurckoScaffold.MurckoScaffoldSmiles(mol=mol),
        "chiral_centers": Chem.FindMolChiralCenters(mol, includeUnassigned=True),
    }
