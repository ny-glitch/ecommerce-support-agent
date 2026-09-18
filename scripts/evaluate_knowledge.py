#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from collections.abc import Sequence
import hashlib
from importlib import metadata, resources
from pathlib import Path
import subprocess
import sys
from typing import Any

from app.config import load_settings
from app.db.database import Database
from app.db.knowledge import KnowledgeRepository
from app.knowledge.calibration import (
    build_runtime_provenance,
    file_sha256,
    load_calibration,
)
from app.knowledge.corpus import (
    corpus_fingerprint,
    load_cases,
    load_corpus,
    validate_cases,
)
from app.knowledge.evaluation import EvaluationDependencies
from app.knowledge.evaluation_artifacts import (
    EvaluationDataError,
    build_calibration_artifact,
    execute_run,
    invalidate_run,
    read_results,
    safe_run_configuration,
    write_calibration,
)
from app.knowledge.local_models import LocalModels
from app.knowledge.milvus_store import MilvusStore
from app.knowledge.retrieval import KnowledgeRetriever
from app.model import OpenAIModelGateway
from app.resource_lifecycle import close_resources, warmup_local_models


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "data" / "knowledge" / "ch04" / "chunks.json"
STRATEGIES = ("dense", "bm25", "hybrid", "hybrid_rerank")
ANSWERABLE_TYPES = ("literal", "model", "colloquial", "synonym", "category_filter")
_DEPENDENCIES = (
    "langchain-openai",
    "langchain-core",
    "pymilvus",
    "FlagEmbedding",
    "SQLAlchemy",
    "torch",
    "transformers",
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _strategies(value: str) -> tuple[str, ...]:
    parsed = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parsed or len(parsed) != len(set(parsed)) or any(
        item not in STRATEGIES for item in parsed
    ):
        raise argparse.ArgumentTypeError(
            "strategies must be a unique comma-separated subset of "
            + ",".join(STRATEGIES)
        )
    return parsed


def _false_accept_limit(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed <= 0.1:
        raise argparse.ArgumentTypeError("value must be between 0 and 0.1")
    return parsed


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate and calibrate Chapter 4 knowledge retrieval."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibrate = subparsers.add_parser("calibrate")
    _common_arguments(calibrate)
    calibrate.add_argument("--calibration", type=Path)
    calibrate.add_argument(
        "--max-false-accept", type=_false_accept_limit, default=0.1
    )

    compare = subparsers.add_parser("compare")
    _common_arguments(compare)
    compare.add_argument("--calibration", type=Path, required=True)
    compare.add_argument("--strategies", type=_strategies, default=STRATEGIES)
    compare.add_argument("--limit", type=_positive_int)
    return parser


def validate_case_set(
    mode: str,
    cases: list[dict],
    *,
    limit: int | None,
) -> None:
    counts = Counter(case.get("query_type") for case in cases)
    answerable = sum(case.get("answerable") is True for case in cases)
    unknown = len(cases) - answerable
    if mode == "calibrate":
        expected = {**{name: 4 for name in ANSWERABLE_TYPES}, "unanswerable": 10}
        if len(cases) != 30 or answerable != 20 or unknown != 10:
            raise EvaluationDataError(
                "calibration requires exactly 30 cases: 20 answerable and 10 unknown"
            )
        if counts != expected:
            raise EvaluationDataError(
                "calibration requires four cases in each answerable bucket and ten unanswerable"
            )
        return
    expected = {**{name: 10 for name in ANSWERABLE_TYPES}, "unanswerable": 10}
    if len(cases) != 60 or answerable != 50 or unknown != 10 or counts != expected:
        raise EvaluationDataError(
            "comparison requires 60 cases: ten in each of the six main buckets"
        )
    if limit is not None and limit > len(cases):
        raise EvaluationDataError("--limit cannot exceed the case count")


async def run_command(args: argparse.Namespace) -> int:
    chunks = load_corpus(args.corpus)
    cases = load_cases(args.cases)
    validate_case_set(args.command, cases, limit=getattr(args, "limit", None))
    case_errors = validate_cases(cases, chunks)
    if case_errors:
        raise EvaluationDataError("case annotations are invalid: " + "; ".join(case_errors))

    selected_cases = cases[: args.limit] if getattr(args, "limit", None) else cases
    settings = load_settings()
    if settings.database_url is None:
        raise EvaluationDataError("DATABASE_URL is required for evaluation")

    model_gateway: Any | None = None
    owned_resources: list[Any] = []
    cancellation: asyncio.CancelledError | None = None
    try:
        model_gateway = OpenAIModelGateway(settings)
        owned_resources.append(model_gateway)
        database = Database(settings.database_url.get_secret_value())
        store = MilvusStore(settings)
        local_models = LocalModels(settings)
        owned_resources.extend((database, store, local_models))

        await database.check()
        await store.prepare_existing_collection()
        repository = KnowledgeRepository(database.sessions)
        database_chunks = await repository.list_all()
        if not database_chunks:
            raise EvaluationDataError("knowledge corpus is empty")
        file_fingerprint = corpus_fingerprint(chunks)
        starting_fingerprint = corpus_fingerprint(database_chunks)
        if starting_fingerprint != file_fingerprint:
            raise EvaluationDataError(
                "database corpus fingerprint does not match --corpus"
            )

        provenance = build_runtime_provenance(
            settings, corpus_fingerprint=starting_fingerprint
        )
        strategies: tuple[str, ...]
        threshold: float | None
        calibration_sha256: str | None = None
        calibration = None
        if args.command == "calibrate":
            strategies = ("hybrid_rerank",)
            threshold = None
        else:
            strategies = args.strategies
            calibration = load_calibration(args.calibration, expected=provenance)
            threshold = calibration.threshold
            calibration_sha256 = file_sha256(args.calibration)

        await warmup_local_models(local_models)
        knowledge_gateway = model_gateway.create_knowledge_gateway()
        dependencies = EvaluationDependencies(
            settings=settings,
            knowledge_gateway=knowledge_gateway,
            retriever=KnowledgeRetriever(repository, store, local_models),
            model_gateway=model_gateway,
        )
        configuration = safe_run_configuration(
            mode=args.command,
            strategies=strategies,
            cases_sha256=file_sha256(args.cases),
            provenance=provenance,
            settings=settings,
            judge_prompt_sha256=_judge_prompt_sha256(),
            calibration_sha256=calibration_sha256,
            threshold=threshold,
            limit=getattr(args, "limit", None),
            code_commit=_code_commit(),
            dependency_versions=_dependency_versions(),
            max_false_accept=getattr(args, "max_false_accept", 0.1),
        )
        manifest = await execute_run(
            selected_cases,
            strategies=strategies,
            calibration_threshold=threshold,
            dependencies=dependencies,
            output_dir=args.output_dir,
            configuration=configuration,
        )

        ending_fingerprint = corpus_fingerprint(await repository.list_all())
        if ending_fingerprint != starting_fingerprint:
            invalidate_run(args.output_dir, "corpus_changed_during_run")
            return 1

        calibration_path = args.output_dir / "calibration.json"
        if args.command == "calibrate":
            artifact = build_calibration_artifact(
                provenance,
                calibration_cases_sha256=file_sha256(args.cases),
                cases=cases,
                results=read_results(args.output_dir / "results.jsonl"),
                max_false_accept=args.max_false_accept,
            )
            write_calibration(calibration_path, artifact)
        else:
            assert calibration is not None
            write_calibration(calibration_path, calibration)

        return 0 if manifest.get("status") in {"complete", "smoke"} else 1
    except asyncio.CancelledError as exc:
        cancellation = exc
        raise
    finally:
        await close_resources(owned_resources, cancellation=cancellation)


def _judge_prompt_sha256() -> str:
    prompt = resources.files("app").joinpath("prompts/faithfulness_judge.txt")
    return hashlib.sha256(prompt.read_bytes()).hexdigest()


def _dependency_versions() -> dict[str, str]:
    return {name: metadata.version(name) for name in _DEPENDENCIES}


def _code_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(run_command(args))
    except (EvaluationDataError, OSError, RuntimeError, ValueError) as exc:
        print(f"evaluation setup failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
