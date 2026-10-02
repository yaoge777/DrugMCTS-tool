from __future__ import annotations

import importlib.metadata
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from .chemistry import canonicalize, describe
from .config import Config
from .errors import BudgetExceeded, ConfigurationError, InputError, ModelError
from .knowledge import KnowledgeBase
from .llm import Backend, DemoBackend, HTTPBackend, instruction, parse_object
from .models import Prediction, Target
from .retrieval import Retriever


@dataclass
class Node:
    id: int
    stage: int
    state: dict
    parent: Node | None = None
    children: list[Node] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0
    failed: bool = False

    @property
    def terminal(self):
        return self.stage == 4


def select_leaf(node: Node, c: float) -> Node:
    while node.children and not node.terminal:
        eligible = [child for child in node.children if not child.failed]
        if not eligible:
            node.failed = True
            return node
        unvisited = [child for child in eligible if child.visits == 0]
        if unvisited:
            return unvisited[0]
        node = max(
            eligible,
            key=lambda child: (
                child.total_reward / child.visits
                + c * math.sqrt(math.log(max(node.visits, 1)) / child.visits)
            ),
        )
    return node


def backpropagate(node: Node, reward: float):
    while node is not None:
        node.visits += 1
        node.total_reward += reward
        node = node.parent


def summary(value: dict) -> str:
    text = value.get("summary")
    if not isinstance(text, str) or not text.strip() or len(text) > 10000:
        raise ValueError("Expected a non-empty concise summary.")
    return text.strip()


class Sampler:
    def __init__(self, backend: Backend, config: Config):
        self.backend, self.config = backend, config
        self.started = time.monotonic()
        self.requests = 0
        self.invalid_responses = 0
        self.warnings = []

    def collect(
        self, action: str, payload: dict, count: int, validator: Callable, unique: bool = False
    ) -> list[Any]:
        if len(instruction(action, payload)) > self.config.search.max_prompt_chars:
            raise ModelError(
                f"{action}: evidence exceeds max_prompt_chars; reduce retrieval/reference counts."
            )
        accepted, seen = [], set()
        for _ in range(self.config.search.response_attempts):
            remaining = self.config.search.max_seconds - (time.monotonic() - self.started)
            if self.requests >= self.config.search.max_llm_requests or remaining <= 0:
                raise BudgetExceeded("Search model-request or time budget reached.")
            self.requests += 1
            try:
                batch = self.backend.sample(
                    action, payload, count - len(accepted), min(self.config.llm.timeout_seconds, remaining)
                )
            except ModelError as exc:
                self.warnings.append(str(exc))
                continue
            for raw in batch[: count - len(accepted)]:
                try:
                    value = raw if isinstance(raw, dict) else parse_object(raw)
                    checked = validator(value)
                    key = json.dumps(checked, sort_keys=True)
                    if unique and key in seen:
                        continue
                    seen.add(key)
                    accepted.append(checked)
                except (ValueError, TypeError, KeyError, ModelError):
                    self.invalid_responses += 1
            if len(accepted) == count:
                return accepted
        if unique and accepted:
            self.warnings.append(
                f"{action}: only {len(accepted)}/{count} distinct valid branches were available."
            )
            return accepted
        raise ModelError(f"{action}: did not obtain {count} valid samples; this path was not scored.")


class DrugMCTS:
    def __init__(
        self,
        config: Config,
        knowledge_base: KnowledgeBase | None = None,
        backend_factory: Callable[[], Backend] | None = None,
    ):
        self.config = config.model_copy(deep=True)
        self.kb = knowledge_base or KnowledgeBase.from_file(config.knowledge_base)
        self.retriever = Retriever(self.kb, self.config.retrieval)
        self.backend_factory = backend_factory or (lambda: HTTPBackend(self.config.llm))

    @classmethod
    def from_config(cls, path):
        return cls(Config.from_file(path))

    @classmethod
    def demo(cls):
        return cls(Config(), KnowledgeBase.demo(), DemoBackend)

    def retrieve(self, smiles: str) -> dict:
        return {
            "query_smiles": canonicalize(smiles),
            "mode": self.config.retrieval.mode,
            "knowledge_base": self.kb.describe(),
            "neighbors": [n.model_dump() for n in self.retriever.search(smiles)],
        }

    def _evidence(self, ids: list[str]) -> list[dict]:
        result = []
        for mid in ids:
            molecule = self.kb.molecules[mid]
            proteins = []
            for pid in molecule.protein_ids:
                protein = self.kb.proteins[pid]
                proteins.append(protein.model_dump())
            result.append(
                {
                    "molecule_id": mid,
                    "smiles": molecule.smiles,
                    "computed_properties": describe(molecule.smiles),
                    "cached_properties": molecule.properties,
                    "proteins": proteins,
                }
            )
        return result

    def predict(self, smiles: str, top_k: int = 5, include_trace: bool = False) -> Prediction:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 100:
            raise InputError("top_k must be an integer from 1 to 100.")
        query = canonicalize(smiles)
        neighbors = self.retriever.search(smiles)
        neighbor_ids = {n.id for n in neighbors}
        query_info = describe(smiles)
        references = [
            {
                **n.model_dump(),
                "computed_properties": describe(n.smiles),
                "cached_properties": self.kb.molecules[n.id].properties,
            }
            for n in neighbors
        ]
        backend = self.backend_factory()
        if isinstance(backend, DemoBackend) and not self.kb.data.demonstration_only:
            backend.close()
            raise ConfigurationError("Demo backend cannot be used with a real knowledge base.")
        sampler = Sampler(backend, self.config)
        search = self.config.search
        rng = random.Random(search.seed)
        root = Node(0, 0, {})
        nodes = [root]
        failures = []
        attempted, completed = 0, 0
        budget_stop = False

        def create(parent, states):
            for state in states:
                child = Node(len(nodes), parent.stage + 1, {**parent.state, **state}, parent)
                parent.children.append(child)
                nodes.append(child)
            rng.shuffle(parent.children)

        def expand(node):
            base = {"query": query_info}
            if node.stage == 0:
                summaries = sampler.collect("molecule_analysis", base, search.branches, summary, unique=True)
                create(node, [{"molecule_summary": s} for s in summaries])
            elif node.stage == 1:
                count = min(search.reference_count, len(neighbors))

                def validate(value):
                    mids = value.get("molecule_ids")
                    if (
                        not isinstance(mids, list)
                        or len(mids) != count
                        or any(not isinstance(x, str) for x in mids)
                    ):
                        raise ValueError("Incorrect molecule selection count/schema.")
                    if len(set(mids)) != count or not set(mids) <= neighbor_ids:
                        raise ValueError("Invalid or duplicate molecule IDs.")
                    return sorted(mids)

                selections = sampler.collect(
                    "molecule_selection",
                    {
                        **base,
                        "count": count,
                        "query_analysis": node.state["molecule_summary"],
                        "references": references,
                    },
                    search.branches,
                    validate,
                    unique=True,
                )
                create(node, [{"molecule_ids": s} for s in selections])
            elif node.stage == 2:
                analysis = sampler.collect(
                    "interaction_analysis",
                    {
                        **base,
                        "query_analysis": node.state["molecule_summary"],
                        "reference_evidence": self._evidence(node.state["molecule_ids"]),
                    },
                    search.branches,
                    summary,
                    unique=True,
                )
                create(node, [{"interaction_summary": s} for s in analysis])
            elif node.stage == 3:
                allowed = sorted(
                    {pid for mid in node.state["molecule_ids"] for pid in self.kb.molecules[mid].protein_ids}
                )

                def validate_protein(value):
                    selected = value.get("protein_id", value.get("selected_protein"))
                    if isinstance(selected, dict):
                        selected = selected.get("protein_id")
                    if not isinstance(selected, str) or selected not in allowed:
                        raise ValueError("Protein ID is outside the selected-reference target pool.")
                    reason = value.get("summary", "")
                    return {"protein_id": selected, "summary": reason if isinstance(reason, str) else ""}

                selections = sampler.collect(
                    "protein_selection",
                    {
                        **base,
                        "query_analysis": node.state["molecule_summary"],
                        "interaction_analysis": node.state["interaction_summary"],
                        "reference_evidence": self._evidence(node.state["molecule_ids"]),
                        "candidate_proteins": allowed,
                    },
                    search.reward_samples,
                    validate_protein,
                )
                votes = Counter(value["protein_id"] for value in selections)
                pid = min(votes, key=lambda p: (-votes[p], p))
                decision_summary = next(
                    (
                        value["summary"]
                        for value in selections
                        if value["protein_id"] == pid and value["summary"].strip()
                    ),
                    node.state["interaction_summary"],
                )

                def validate_judgement(value):
                    if type(value.get("interacts")) is not bool:
                        raise ValueError("interacts must be a JSON boolean.")
                    return value["interacts"]

                judgements = sampler.collect(
                    "interaction_judgement",
                    {
                        **base,
                        "query_analysis": node.state["molecule_summary"],
                        "selected_target": self.kb.proteins[pid].model_dump(),
                    },
                    search.reward_samples,
                    validate_judgement,
                )
                relative, absolute = (
                    votes[pid] / search.reward_samples,
                    sum(judgements) / search.reward_samples,
                )
                create(
                    node,
                    [
                        {
                            "protein_id": pid,
                            "relative": relative,
                            "absolute": absolute,
                            "reward": (relative + absolute) / 2,
                            "selection_votes": dict(votes),
                            "decision_summary": decision_summary,
                            "yes_votes": sum(judgements),
                        }
                    ],
                )

        try:
            for _ in range(search.rollouts):
                attempted += 1
                leaf = select_leaf(root, search.exploration_constant)
                if leaf.failed:
                    if leaf is root:
                        break
                    continue
                try:
                    while not leaf.terminal:
                        expand(leaf)
                        leaf = rng.choice(leaf.children)
                    backpropagate(leaf, leaf.state["reward"])
                    completed += 1
                except BudgetExceeded as exc:
                    failures.append(str(exc))
                    budget_stop = True
                    break
                except ModelError as exc:
                    leaf.failed = True
                    failures.append(str(exc))
        finally:
            backend.close()
        terminals = [n for n in nodes if n.terminal and n.visits > 0]
        if not terminals:
            detail = failures[-1] if failures else "No valid terminal path."
            raise ModelError(f"No target predictions could be produced. {detail}")
        grouped = defaultdict(list)
        for terminal in terminals:
            grouped[terminal.state["protein_id"]].append(terminal)
        targets = []
        for pid, paths in grouped.items():
            protein = self.kb.proteins[pid]
            best = max(paths, key=lambda n: (n.state["reward"], -n.id))
            targets.append(
                Target(
                    protein_id=pid,
                    name=protein.name,
                    uniprot_ids=protein.uniprot_ids,
                    score=sum(n.state["reward"] for n in paths) / len(paths),
                    relative_score=sum(n.state["relative"] for n in paths) / len(paths),
                    absolute_score=sum(n.state["absolute"] for n in paths) / len(paths),
                    supporting_paths=len(paths),
                    visits=sum(n.visits for n in paths),
                    reference_molecule_ids=sorted({mid for n in paths for mid in n.state["molecule_ids"]}),
                    evidence_summary=best.state["decision_summary"],
                    sources=protein.sources,
                )
            )
        targets.sort(key=lambda target: (-target.score, -target.supporting_paths, target.protein_id))
        warnings = list(dict.fromkeys(sampler.warnings + failures))
        if sampler.invalid_responses:
            warnings.append(f"Rejected {sampler.invalid_responses} malformed/out-of-pool model responses.")
        if len(targets) < top_k:
            warnings.append(
                f"Only {len(targets)} distinct targets reached valid terminal paths (requested {top_k})."
            )
        candidate_ids = {p for n in neighbors for p in n.protein_ids}
        absent = sum(
            not self.kb.proteins[p].pocket or not self.kb.proteins[p].literature for p in candidate_ids
        )
        if absent:
            warnings.append(
                f"{absent} retrieved candidate targets have incomplete pocket or literature evidence."
            )
        demo = self.kb.data.demonstration_only or isinstance(backend, DemoBackend)
        if demo:
            warnings.insert(0, "SYNTHETIC DEMO: these targets, scores and summaries are software fixtures.")
        return Prediction(
            status="demo" if demo else "partial" if completed < search.rollouts else "completed",
            demonstration_only=demo,
            query={"input_smiles": smiles, "canonical_smiles": query},
            targets=targets[:top_k],
            retrieval={
                "mode": self.config.retrieval.mode,
                "top_n_per_metric": self.config.retrieval.top_n,
                "neighbors": [n.model_dump() for n in neighbors],
                "candidate_proteins": len(candidate_ids),
                "canonical_self_matches_excluded": True,
            },
            search={
                "rollouts_requested": search.rollouts,
                "rollouts_attempted": attempted,
                "rollouts_completed": completed,
                "unique_terminal_paths": len(terminals),
                "nodes": len(nodes),
                "budget_reached": budget_stop,
                "top_k_requested": top_k,
                "ranking": "mean_reward_over_unique_terminal_paths; ties: path_count_then_protein_id",
                "elapsed_seconds": round(time.monotonic() - sampler.started, 3),
            },
            usage={**backend.usage, "model_requests_attempted": sampler.requests},
            provenance={
                "tool_version": "0.1.0",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "knowledge_base": self.kb.describe(),
                "model": "synthetic-demo" if isinstance(backend, DemoBackend) else self.config.llm.model,
                "config": self.config.model_dump(exclude={"llm": {"base_url", "api_key_env"}}),
                "rdkit_version": importlib.metadata.version("rdkit"),
                "paper_result_reproduction": False,
            },
            warnings=warnings,
            traces=[{"node_id": n.id, "visits": n.visits, **n.state} for n in terminals]
            if include_trace
            else [],
        )
