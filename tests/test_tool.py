import asyncio
import hashlib
import json
import os
import subprocess
import sys
from datetime import timedelta

import httpx
import pytest

from drugmcts import Config, DrugMCTS
from drugmcts.chemistry import canonicalize, describe
from drugmcts.config import LLMConfig, RetrievalConfig, SearchConfig
from drugmcts.engine import Node, backpropagate, select_leaf
from drugmcts.errors import ConfigurationError, InputError, ModelError
from drugmcts.knowledge import KnowledgeBase, write_json
from drugmcts.legacy import clean_abstract, prepare_legacy
from drugmcts.llm import DemoBackend, HTTPBackend, parse_object
from drugmcts.models import KnowledgeData
from drugmcts.retrieval import Retriever

SMILES = "CC(=O)Oc1ccccc1C(=O)O"


def fixture_kb(tmp_path):
    path = tmp_path / "kb.json"
    path.write_text(KnowledgeBase.demo().data.model_dump_json())
    return path


def test_canonical_self_exclusion_and_invalid_smiles():
    tool = DrugMCTS.demo()
    source = tool.kb.data.molecules[0]
    matches = tool.retrieve(canonicalize(source.smiles))["neighbors"]
    assert source.id not in {r["id"] for r in matches}
    assert all(canonicalize(r["smiles"]) != canonicalize(source.smiles) for r in matches)
    assert describe("CCO")["hbond_donors"] == 1
    for bad in ["", "not a SMILES", "C" * 10001]:
        with pytest.raises(InputError):
            tool.predict(bad)


def test_hybrid_union_self_filter_and_stable_ties(monkeypatch):
    kb = KnowledgeBase.demo()
    retriever = Retriever(kb, RetrievalConfig(mode="hybrid", top_n=2))
    # A deterministic embedding-score fixture tests fusion without a model download.
    scores = {mid: idx / 10 for idx, mid in enumerate(retriever.ids)}
    monkeypatch.setattr(retriever, "_hybrid_scores", lambda _: scores)
    result = retriever.search(kb.molecules["DEMO_M6"].smiles)
    assert len(result) <= 4
    assert len({r.id for r in result}) == len(result)
    assert "DEMO_M6" not in {r.id for r in result}
    assert "DEMO_M5" in {r.id for r in result}
    assert all(r.cosine is not None for r in result)


def test_unknown_duplicate_ids_and_canonical_aliases():
    data = KnowledgeBase.demo().data.model_dump()
    data["molecules"][0]["protein_ids"] = ["NOT_PRESENT"]
    with pytest.raises(InputError, match="missing protein"):
        KnowledgeBase(KnowledgeData.model_validate(data))
    data = KnowledgeBase.demo().data.model_dump()
    data["molecules"].append(data["molecules"][0].copy())
    with pytest.raises(InputError, match="unique"):
        KnowledgeBase(KnowledgeData.model_validate(data))
    data["molecules"][-1]["id"] = "ALIAS"
    kb = KnowledgeBase(KnowledgeData.model_validate(data))
    result = Retriever(kb, RetrievalConfig()).search(data["molecules"][0]["smiles"])
    assert not {"ALIAS", data["molecules"][0]["id"]} & {r.id for r in result}


def test_config_relative_to_file_and_strict_fields(tmp_path, monkeypatch):
    folder = tmp_path / "config"
    folder.mkdir()
    path = folder / "settings.toml"
    path.write_text('knowledge_base = "../kb.json"\n[llm]\nmodel = "configured-7b"\n')
    monkeypatch.chdir("/")
    cfg = Config.from_file(path)
    assert cfg.knowledge_base == str(tmp_path / "kb.json")
    assert cfg.llm.model == "configured-7b"
    path.write_text("[search]\nrollouts = -1\n")
    with pytest.raises(ConfigurationError):
        Config.from_file(path)
    path.write_text("accidental_typo = 123\n")
    with pytest.raises(ConfigurationError):
        Config.from_file(path)


def test_json_fences_and_errors():
    assert parse_object('```json\n{"summary":"ok"}\n```') == {"summary": "ok"}
    for raw in ["[]", "{broken", 'text {"summary": "no"}']:
        with pytest.raises(ModelError):
            parse_object(raw)


def test_demo_runs_complete_tree_with_trace_and_reproducible_ranking():
    first = DrugMCTS.demo().predict(SMILES, top_k=3, include_trace=True)
    second = DrugMCTS.demo().predict(SMILES, top_k=3, include_trace=True)
    assert first.status == "demo" and first.demonstration_only
    assert first.search["rollouts_completed"] == 12
    assert first.targets == second.targets
    assert first.traces == second.traces
    assert len(first.targets) == 3
    assert all(0 <= t.score <= 1 for t in first.targets)
    assert sum(t.supporting_paths for t in first.targets) == first.search["unique_terminal_paths"]
    assert sum(t.visits for t in first.targets) == 12
    for trace in first.traces:
        assert len(trace["molecule_ids"]) == len(set(trace["molecule_ids"])) == 3
        assert sum(trace["selection_votes"].values()) == 4
        assert trace["reward"] == (trace["relative"] + trace["absolute"]) / 2
    with pytest.raises(InputError):
        DrugMCTS.demo().predict(SMILES, top_k=0)
    with pytest.raises(InputError):
        DrugMCTS.demo().predict(SMILES, top_k=True)


def test_uct_and_backpropagate_do_not_copy_visit_statistics():
    root = Node(0, 0, {}, visits=10)
    left = Node(1, 1, {}, root, visits=5, total_reward=4)
    right = Node(2, 1, {}, root, visits=5, total_reward=1)
    root.children = [left, right]
    assert select_leaf(root, 3) is left
    right.visits = 0
    assert select_leaf(root, 3) is right
    terminal = Node(3, 4, {}, right)
    backpropagate(terminal, 0.75)
    assert terminal.visits == 1 and terminal.total_reward == 0.75
    assert right.visits == 1 and root.visits == 11


class NestedBackend(DemoBackend):
    def sample(self, action, payload, n, timeout):
        if action == "protein_selection":
            return [{"selected_protein": {"protein_id": payload["candidate_proteins"][0]}} for _ in range(n)]
        if action == "interaction_judgement":
            return [{"interacts": idx % 2 == 0} for idx in range(n)]
        return super().sample(action, payload, n, timeout)


def test_nested_protein_schema_and_equal_weight_reward():
    tool = DrugMCTS(Config(search=SearchConfig(rollouts=1)), KnowledgeBase.demo(), NestedBackend)
    result = tool.predict(SMILES, include_trace=True)
    assert result.targets[0].score == 0.75
    assert result.targets[0].relative_score == 1
    assert result.targets[0].absolute_score == 0.5


class BadBackend(DemoBackend):
    def sample(self, action, payload, n, timeout):
        if action == "molecule_selection":
            # The old two-vs-three conflict must now fail explicitly, not synthesize neighbors.
            return [{"molecule_ids": [r["id"] for r in payload["references"][:2]]}] * n
        return super().sample(action, payload, n, timeout)


def test_invalid_molecule_count_is_not_randomly_replaced():
    tool = DrugMCTS(Config(search=SearchConfig(rollouts=2)), KnowledgeBase.demo(), BadBackend)
    with pytest.raises(ModelError, match="No target predictions"):
        tool.predict(SMILES)


class BadRewardBackend(DemoBackend):
    def sample(self, action, payload, n, timeout):
        if action == "interaction_judgement":
            return [{"interacts": "sometimes"} for _ in range(n)]
        return super().sample(action, payload, n, timeout)


def test_insufficient_valid_reward_samples_never_change_denominator():
    tool = DrugMCTS(Config(search=SearchConfig(rollouts=1)), KnowledgeBase.demo(), BadRewardBackend)
    with pytest.raises(ModelError, match="did not obtain 4 valid samples"):
        tool.predict(SMILES)


def test_request_budget_and_no_unscored_filler():
    config = Config(search=SearchConfig(max_llm_requests=5, rollouts=12))
    result = DrugMCTS(config, KnowledgeBase.demo(), DemoBackend).predict(SMILES, top_k=100)
    assert result.search["budget_reached"]
    assert result.usage["model_requests_attempted"] == 5
    assert result.search["rollouts_completed"] == 1
    assert len(result.targets) == 1
    assert any("Only 1" in w for w in result.warnings)
    config.search.max_llm_requests = 1
    with pytest.raises(ModelError, match="budget"):
        DrugMCTS(config, KnowledgeBase.demo(), DemoBackend).predict(SMILES)


def test_synthetic_backend_cannot_predict_on_real_knowledge():
    data = KnowledgeBase.demo().data.model_copy(deep=True)
    data.demonstration_only = False
    with pytest.raises(ConfigurationError, match="Demo backend"):
        DrugMCTS(Config(), KnowledgeBase(data), DemoBackend).predict(SMILES)


def test_http_payload_usage_and_credential_redaction(monkeypatch):
    monkeypatch.setenv("TOOL_TEST_KEY", "private-test-value")
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        assert request.headers["Authorization"] == "Bearer private-test-value"
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"summary":"ok"}'}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 5},
            },
        )

    backend = HTTPBackend(
        LLMConfig(model="explicit-model-7b", api_key_env="TOOL_TEST_KEY"),
        transport=httpx.MockTransport(handler),
    )
    assert backend.sample("molecule_analysis", {}, 1, 10) == ['{"summary":"ok"}']
    backend.close()
    assert seen[0]["model"] == "explicit-model-7b" and seen[0]["n"] == 1
    assert backend.usage["prompt_tokens"] == 20
    backend = HTTPBackend(
        LLMConfig(), transport=httpx.MockTransport(lambda r: httpx.Response(401, text="private-test-value"))
    )
    with pytest.raises(ConfigurationError) as err:
        backend.sample("molecule_analysis", {}, 1, 10)
    backend.close()
    assert "private-test-value" not in str(err.value)
    with pytest.raises(ConfigurationError, match="environment variable"):
        HTTPBackend(LLMConfig(api_key_env="NONEXISTENT_TOOL_KEY"))


def test_http_adapter_end_to_end_prediction():
    fake = DemoBackend()

    def handler(request):
        content = json.loads(request.content)
        text = content["messages"][1]["content"]
        prefix, payload = text.split("Evidence JSON:\n", 1)
        if "Select exactly" in prefix:
            action = "molecule_selection"
        elif "Summarize the supplied reference" in prefix:
            action = "interaction_analysis"
        elif "Select ONE ID" in prefix:
            action = "protein_selection"
        elif "Independently judge" in prefix:
            action = "interaction_judgement"
        else:
            action = "molecule_analysis"
        values = fake.sample(action, json.loads(payload), content["n"], 5)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps(v)}} for v in values],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            },
        )

    config = Config(search=SearchConfig(rollouts=2))
    result = DrugMCTS(
        config, KnowledgeBase.demo(), lambda: HTTPBackend(config.llm, httpx.MockTransport(handler))
    ).predict(SMILES)
    assert result.search["rollouts_completed"] == 2
    assert result.usage["requests"] == result.usage["model_requests_attempted"]
    assert result.usage["prompt_tokens"] == 100 * result.usage["requests"]
    assert result.demonstration_only


@pytest.mark.parametrize("dataset", ["drugbank", "kiba"])
def test_legacy_import_is_read_only_and_cleans_errors(tmp_path, dataset):
    root = tmp_path / "original"
    (root / "data").mkdir(parents=True)
    suffix = "_KIBA" if dataset == "kiba" else ""
    for folder in ["plip", "pubmed", "pdb_desc"]:
        (root / "experiment" / (folder + suffix)).mkdir(parents=True)
    raw = "KIBA.txt" if suffix else "cleaned_DrugBank.txt"
    mid = "CHEMBL1" if suffix else "DB00001"
    (root / "data" / raw).write_text(f"{mid} U1 CCO AAA 1\n{mid} U2 CCO AAA 0\n")
    (root / "data" / f"uniprot_pdb_map{suffix}.tsv").write_text("From\tTo\nU1\tOLD1\nU1\t1ABC\n")
    (root / "experiment" / ("plip" + suffix) / "1ABC.txt").write_text(
        '"""hydrophobic""":\nRESTYPE RESCHAIN RESCHAIN_LIG DIST LIGCOO PROTCOO\n'
        "ALA A B 3.0 (1, 2, 3) (4, 5, 6)\n"
    )
    (root / "experiment" / ("pdb_desc" + suffix) / "1ABC.json").write_text(
        '{"abstract":"PubMed exception: retry"}'
    )
    (root / "experiment" / ("pubmed" + suffix) / "1ABC.json").write_text(
        '{"abstract":"Curated fixture", "pmid":"123"}'
    )
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
    result = prepare_legacy(root, dataset, tmp_path / "output.json")
    kb = KnowledgeBase.from_file(tmp_path / "output.json")
    assert result["molecules"] == 1
    assert result["sha256"] == hashlib.sha256((tmp_path / "output.json").read_bytes()).hexdigest()
    assert kb.proteins["1ABC"].literature == ["Curated fixture"]
    assert "3.0" in kb.proteins["1ABC"].pocket
    assert kb.molecules[mid].protein_ids == ["1ABC"]
    assert "https://pubmed.ncbi.nlm.nih.gov/123/" in kb.proteins["1ABC"].sources
    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
    assert before == after
    with pytest.raises(InputError, match="read-only"):
        prepare_legacy(root, dataset, root / "data/new.json")


def test_file_output_requires_force_and_abstract_filter(tmp_path):
    write_json(tmp_path / "out.json", {"a": 1})
    with pytest.raises(InputError, match="already exists"):
        write_json(tmp_path / "out.json", {"a": 2})
    assert json.loads((tmp_path / "out.json").read_text()) == {"a": 1}
    write_json(tmp_path / "out.json", {"a": 2}, True)
    assert clean_abstract(tmp_path / "absent.json") is None


def test_installed_cli_from_other_directory(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "drugmcts", "demo", "--top-k", "2"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["demonstration_only"]
    bad = subprocess.run(
        [sys.executable, "-m", "drugmcts", "demo", "--smiles", "invalid"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert bad.returncode == 2
    assert json.loads(bad.stderr)["error"] == "InputError"
    kb = fixture_kb(tmp_path)
    cfg = tmp_path / "config.toml"
    cfg.write_text(f'knowledge_base = "{kb.as_posix()}"\n')
    doctor = subprocess.run(
        [sys.executable, "-m", "drugmcts", "doctor", "--config", str(cfg)],
        cwd="/",
        capture_output=True,
        text=True,
        check=True,
    )
    assert not json.loads(doctor.stdout)["endpoint_contacted"]


def test_mcp_stdio_round_trip(tmp_path):
    pytest.importorskip("mcp")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def exercise():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "drugmcts", "serve", "--demo"],
            cwd=str(tmp_path),
            env=dict(os.environ),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=45)) as session:
                await session.initialize()
                listed = await session.list_tools()
                assert {t.name for t in listed.tools} == {
                    "drugmcts_info",
                    "retrieve_similar_molecules",
                    "predict_targets",
                }
                info = await session.call_tool("drugmcts_info", {})
                assert not info.isError
                assert json.loads(info.content[0].text)["knowledge_base"]["demonstration_only"]
                retrieved = await session.call_tool("retrieve_similar_molecules", {"smiles": SMILES})
                assert not retrieved.isError
                assert json.loads(retrieved.content[0].text)["neighbors"]
                prediction = await session.call_tool("predict_targets", {"smiles": SMILES, "top_k": 2})
                assert not prediction.isError
                value = json.loads(prediction.content[0].text)
                assert value["demonstration_only"] and len(value["targets"]) == 2
                invalid = await session.call_tool("predict_targets", {"smiles": "invalid"})
                assert invalid.isError

    asyncio.run(exercise())


def test_revisited_terminal_is_not_counted_as_independent_support():
    config = Config(search=SearchConfig(branches=1, rollouts=12))
    result = DrugMCTS(config, KnowledgeBase.demo(), DemoBackend).predict(SMILES, include_trace=True)
    assert result.search["unique_terminal_paths"] == 1
    assert result.targets[0].supporting_paths == 1
    assert result.targets[0].visits == 12
    assert result.usage["model_requests_attempted"] == 5


def test_output_collision_is_detected_before_loading_model_or_config(tmp_path):
    path = tmp_path / "result.json"
    path.write_text('{"preserve": true}')
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "drugmcts",
            "predict",
            "--config",
            str(tmp_path / "missing.toml"),
            "--smiles",
            SMILES,
            "--output",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "already exists" in json.loads(result.stderr)["message"]
    assert json.loads(path.read_text()) == {"preserve": True}
