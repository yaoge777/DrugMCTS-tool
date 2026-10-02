from __future__ import annotations

import threading

from .engine import DrugMCTS
from .errors import ConfigurationError, InputError


def create_server(tool: DrugMCTS):
    try:
        from mcp.server.fastmcp import FastMCP
        from mcp.types import ToolAnnotations
    except ImportError as exc:
        raise ConfigurationError('Install MCP support with pip install ".[mcp]".') from exc
    server = FastMCP(
        "DrugMCTS",
        instructions=(
            "Explore research target hypotheses from SMILES. Real predictions call the configured model service "
            "and may take several minutes. Scores express LLM consistency, not experimental binding affinity. "
            "Inspect drugmcts_info first. Synthetic demo data is explicitly labeled."
        ),
    )
    prediction_lock = threading.Lock()

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
    def drugmcts_info() -> dict:
        """Describe the local knowledge base, retrieval mode, model and prediction budget; no LLM call."""
        return {
            "knowledge_base": tool.kb.describe(),
            "retrieval": tool.config.retrieval.model_dump(),
            "search_budget": tool.config.search.model_dump(),
            "model": "synthetic-demo" if tool.kb.data.demonstration_only else tool.config.llm.model,
        }

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True))
    def retrieve_similar_molecules(smiles: str) -> dict:
        """Retrieve molecular neighbors and associated targets. Hybrid mode may download ChemBERTa once."""
        return tool.retrieve(smiles)

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True))
    def predict_targets(smiles: str, top_k: int = 5, include_trace: bool = False) -> dict:
        """Run MCTS target hypothesis exploration. Sends molecular evidence to the configured LLM endpoint;
        uses the configured request/time budget. top_k is user chosen, with no ground-truth input.
        """
        if not prediction_lock.acquire(blocking=False):
            raise InputError("Another prediction is running. Retry after it completes.")
        try:
            return tool.predict(smiles, top_k=top_k, include_trace=include_trace).model_dump()
        finally:
            prediction_lock.release()

    return server


def serve(config_path=None, demo=False):
    tool = DrugMCTS.demo() if demo else DrugMCTS.from_config(config_path)
    create_server(tool).run(transport="stdio")
