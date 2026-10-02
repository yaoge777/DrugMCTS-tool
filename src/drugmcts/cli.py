from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from .config import Config
from .engine import DrugMCTS
from .errors import DrugMCTSError, InputError
from .knowledge import KnowledgeBase, write_json
from .legacy import prepare_legacy

DEMO_SMILES = "CC(=O)Oc1ccccc1C(=O)O"


def build_parser():
    parser = argparse.ArgumentParser(
        prog="drugmcts", description="Molecular retrieval and MCTS target hypotheses"
    )
    parser.add_argument("--version", action="version", version="drugmcts-tool 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [
        ("predict", "Predict research target hypotheses"),
        ("retrieve", "Retrieve reference molecules without calling an LLM"),
    ]:
        cmd = commands.add_parser(name, help=help_text)
        cmd.add_argument("--config", required=True, type=Path)
        cmd.add_argument("--smiles", required=True)
        cmd.add_argument("--output", type=Path)
        cmd.add_argument("--force", action="store_true")
        if name == "predict":
            cmd.add_argument("--top-k", type=int, default=5)
            cmd.add_argument("--trace", action="store_true")
    demo = commands.add_parser("demo", help="Offline synthetic smoke demonstration; no credentials")
    demo.add_argument("--smiles", default=DEMO_SMILES)
    demo.add_argument("--top-k", type=int, default=3)
    demo.add_argument("--trace", action="store_true")
    demo.add_argument("--output", type=Path)
    demo.add_argument("--force", action="store_true")
    validate = commands.add_parser(
        "validate-kb", help="Check knowledge-base schema, IDs and molecular structures"
    )
    validate.add_argument("path", type=Path)
    doctor = commands.add_parser(
        "doctor", help="Validate config and local knowledge data without contacting the LLM"
    )
    doctor.add_argument("--config", required=True, type=Path)
    prepare = commands.add_parser(
        "prepare-legacy", help="Import a read-only experiment snapshot into a new JSON knowledge base"
    )
    prepare.add_argument("--legacy-dir", required=True, type=Path)
    prepare.add_argument("--dataset", required=True, choices=["drugbank", "kiba"])
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--force", action="store_true")
    serve = commands.add_parser("serve", help="Serve MCP over stdio; install the mcp extra")
    source = serve.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--demo", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if getattr(args, "output", None) and args.output.exists() and not args.force:
            raise InputError(f"Output already exists: {args.output}. Choose another path or pass --force.")
        if args.command == "serve":
            from .mcp_server import serve

            serve(args.config, args.demo)
            return 0
        if args.command == "prepare-legacy":
            result = prepare_legacy(args.legacy_dir, args.dataset, args.output, args.force)
        elif args.command == "validate-kb":
            result = KnowledgeBase.from_file(args.path).describe()
        elif args.command == "doctor":
            import importlib.util
            import os

            cfg = Config.from_file(args.config)
            kb = KnowledgeBase.from_file(cfg.knowledge_base)
            checks = {
                "config_valid": True,
                "knowledge_base_valid": True,
                "model_configured": bool(cfg.llm.model),
                "credential_available": not cfg.llm.api_key_env or bool(os.environ.get(cfg.llm.api_key_env)),
                "retrieval_dependencies_available": cfg.retrieval.mode != "hybrid"
                or all(importlib.util.find_spec(name) is not None for name in ["torch", "transformers"]),
            }
            result = {
                "checks": checks,
                "knowledge_base": kb.describe(),
                "endpoint_contacted": False,
                "model": cfg.llm.model,
            }
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if all(checks.values()) else 1
        else:
            tool = DrugMCTS.demo() if args.command == "demo" else DrugMCTS.from_config(args.config)
            result = (
                tool.retrieve(args.smiles)
                if args.command == "retrieve"
                else tool.predict(args.smiles, top_k=args.top_k, include_trace=args.trace).model_dump()
            )
            if args.output:
                write_json(args.output, result, args.force)
                print(f"Saved {args.output.resolve()}", file=sys.stderr)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (DrugMCTSError, OSError, ValidationError, json.JSONDecodeError) as exc:
        message = (
            "Invalid input data; validate the knowledge base and configuration."
            if isinstance(exc, ValidationError)
            else str(exc)
        )
        print(json.dumps({"error": type(exc).__name__, "message": message}), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print('{"error": "Cancelled", "message": "Prediction cancelled."}', file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
