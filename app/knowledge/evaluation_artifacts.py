from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from app.config import Settings
from app.knowledge.calibration import CalibrationArtifact, RuntimeProvenance
from app.knowledge.contracts import QueryPlan
from app.knowledge.evaluation import (
    EvaluationDependencies,
    _evaluate_strategy,
    calibrate_threshold,
    observe_normalization,
    strict_json_dumps,
)


_SAFE_CONFIGURATION_FIELDS = {
    "schema_version",
    "mode",
    "strategies",
    "corpus_fingerprint",
    "cases_sha256",
    "calibration_sha256",
    "prompt_bundle_sha256",
    "judge_prompt_sha256",
    "model_manifest_sha256",
    "embedding_model",
    "embedding_revision",
    "reranker_model",
    "reranker_revision",
    "llm_model",
    "llm_revision",
    "context_window_tokens",
    "max_output_tokens",
    "token_safety_margin",
    "knowledge_request_timeout_seconds",
    "limit",
    "max_false_accept",
    "threshold",
    "code_commit",
    "dependency_versions",
}


class EvaluationDataError(ValueError):
    pass


def safe_run_configuration(
    *,
    mode: str,
    strategies: tuple[str, ...],
    cases_sha256: str,
    provenance: RuntimeProvenance,
    settings: Settings,
    judge_prompt_sha256: str,
    calibration_sha256: str | None,
    threshold: float | None,
    limit: int | None,
    code_commit: str,
    dependency_versions: dict[str, str],
    max_false_accept: float = 0.1,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "mode": mode,
        "strategies": list(strategies),
        "corpus_fingerprint": provenance.corpus_fingerprint,
        "cases_sha256": cases_sha256,
        "calibration_sha256": calibration_sha256,
        "prompt_bundle_sha256": provenance.prompt_bundle_sha256,
        "judge_prompt_sha256": judge_prompt_sha256,
        "model_manifest_sha256": provenance.model_manifest_sha256,
        "embedding_model": provenance.embedding_model,
        "embedding_revision": provenance.embedding_revision,
        "reranker_model": provenance.reranker_model,
        "reranker_revision": provenance.reranker_revision,
        "llm_model": settings.llm_model,
        "llm_revision": "provider-configured-unpinned",
        "context_window_tokens": settings.context_window_tokens,
        "max_output_tokens": settings.max_output_tokens,
        "token_safety_margin": settings.token_safety_margin,
        "knowledge_request_timeout_seconds": (
            settings.knowledge_request_timeout_seconds
        ),
        "limit": limit,
        "max_false_accept": max_false_accept,
        "threshold": threshold,
        "code_commit": code_commit,
        "dependency_versions": dict(sorted(dependency_versions.items())),
    }


class _RunArtifacts:
    def __init__(self, output_dir: Path, configuration: dict[str, Any]) -> None:
        unknown = set(configuration) - _SAFE_CONFIGURATION_FIELDS
        if unknown:
            raise EvaluationDataError(
                "configuration contains unsafe or unknown fields: "
                + ", ".join(sorted(unknown))
            )
        self.output_dir = output_dir
        self.normalization_dir = output_dir / ".normalizations"
        self.result_dir = output_dir / ".results"
        self.configuration = configuration
        self.config_hash = hashlib.sha256(
            _canonical_json(configuration).encode("utf-8")
        ).hexdigest()
        output_dir.mkdir(parents=True, exist_ok=True)
        self.normalization_dir.mkdir(exist_ok=True)
        self.result_dir.mkdir(exist_ok=True)
        self.manifest_path = output_dir / "manifest.json"
        self.manifest = self._load_manifest()

    @property
    def complete(self) -> bool:
        return (
            self.manifest.get("status") in {"complete", "smoke"}
            and (self.output_dir / "normalizations.json").is_file()
            and (self.output_dir / "results.jsonl").is_file()
            and (self.output_dir / "report.md").is_file()
        )

    def _load_manifest(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            manifest = _read_json(self.manifest_path)
            if manifest.get("configuration_sha256") != self.config_hash:
                raise EvaluationDataError(
                    "run directory configuration does not match the requested run"
                )
            return manifest
        manifest = {
            "schema_version": 1,
            "status": "running",
            "configuration_sha256": self.config_hash,
            "configuration": self.configuration,
            "started_at": _utc_now(),
            "completed_at": None,
            "completed_results": 0,
            "technical_failure_count": 0,
            "candidate_set_mismatches": [],
        }
        _atomic_json(self.manifest_path, manifest)
        return manifest

    def load_normalization(self, query_id: str) -> dict[str, Any] | None:
        path = self.normalization_dir / f"{_key_digest(query_id)}.json"
        if not path.exists():
            return None
        value = _read_json(path)
        if value.get("query_id") != query_id:
            raise EvaluationDataError("normalization cache key mismatch")
        return value

    def save_normalization(self, value: dict[str, Any]) -> None:
        path = self.normalization_dir / f"{_key_digest(value['query_id'])}.json"
        _atomic_json(path, value)

    def load_result(self, query_id: str, strategy: str) -> dict[str, Any] | None:
        path = self.result_dir / f"{_key_digest(query_id, strategy)}.json"
        if not path.exists():
            return None
        value = _read_json(path)
        if value.get("query_id") != query_id or value.get("strategy") != strategy:
            raise EvaluationDataError("result cache key mismatch")
        return value

    def save_result(self, value: dict[str, Any]) -> None:
        path = self.result_dir / (
            f"{_key_digest(value['query_id'], value['strategy'])}.json"
        )
        _atomic_json(path, value)

    def finalize(
        self,
        normalizations: list[dict[str, Any]],
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        _atomic_json(self.output_dir / "normalizations.json", normalizations)
        jsonl = "".join(strict_json_dumps(item) + "\n" for item in results)
        _atomic_text(self.output_dir / "results.jsonl", jsonl)
        summary = summarize_results(results)
        mismatches = candidate_set_mismatches(results)
        limited = self.configuration.get("limit") is not None
        _atomic_text(
            self.output_dir / "report.md",
            render_report(summary, mismatches=mismatches, limited=limited),
        )
        failures = sum(item.get("error") is not None for item in results)
        self.manifest = {
            **self.manifest,
            "status": (
                "smoke"
                if failures == 0 and not mismatches and limited
                else "complete"
                if failures == 0 and not mismatches
                else "incomplete"
            ),
            "completed_at": _utc_now(),
            "completed_results": len(results),
            "technical_failure_count": failures,
            "candidate_set_mismatches": mismatches,
            "summary": summary,
        }
        _atomic_json(self.manifest_path, self.manifest)
        return self.manifest


async def execute_run(
    cases: list[dict[str, Any]],
    *,
    strategies: tuple[str, ...],
    calibration_threshold: float | None,
    dependencies: EvaluationDependencies,
    output_dir: Path,
    configuration: dict[str, Any],
) -> dict[str, Any]:
    store = _RunArtifacts(output_dir, configuration)
    if store.complete:
        return store.manifest

    normalizations: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for case in cases:
        normalization = store.load_normalization(case["query_id"])
        if normalization is None:
            deadline = (
                time.monotonic()
                + dependencies.settings.knowledge_request_timeout_seconds
            )
            plan, observation = await observe_normalization(
                dependencies.knowledge_gateway,
                case["query"],
                case.get("category"),
                deadline=deadline,
            )
            normalization = {
                "query_id": case["query_id"],
                "original": plan.original,
                "normalized": plan.normalized,
                "synonyms": list(plan.synonyms),
                "category": plan.category,
                "fallback": plan.fallback,
                **observation,
            }
            store.save_normalization(normalization)
        plan = QueryPlan(
            original=normalization["original"],
            normalized=normalization["normalized"],
            synonyms=tuple(normalization["synonyms"]),
            category=normalization["category"],
            fallback=bool(normalization["fallback"]),
        )
        normalizations.append(normalization)
        initial_error = None
        if not normalization["request_success"]:
            initial_error = {
                "stage": "normalization",
                "error_code": normalization["gateway_error_code"] or "gateway_error",
            }
        for strategy in strategies:
            result = store.load_result(case["query_id"], strategy)
            if result is None:
                result = await _evaluate_strategy(
                    case,
                    plan,
                    strategy,
                    calibration_threshold=(
                        calibration_threshold
                        if strategy == "hybrid_rerank"
                        else None
                    ),
                    dependencies=dependencies,
                    initial_error=initial_error,
                )
                store.save_result(result)
            results.append(result)
    return store.finalize(normalizations, results)


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_strategy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_strategy[result["strategy"]].append(result)
    strategies: dict[str, Any] = {}
    for strategy, items in sorted(by_strategy.items()):
        by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_difficulty: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            by_type[item["query_type"]].append(item)
            by_difficulty[item["difficulty"]].append(item)
        strategies[strategy] = {
            "overall": _summarize_group(items),
            "by_query_type": {
                key: _summarize_group(value)
                for key, value in sorted(by_type.items())
            },
            "by_difficulty": {
                key: _summarize_group(value)
                for key, value in sorted(by_difficulty.items())
            },
        }
    return {"strategies": strategies}


def _summarize_group(items: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [item for item in items if item["answerable"]]
    unknown = [item for item in items if not item["answerable"]]
    summary: dict[str, Any] = {
        "sample_count": len(items),
        "retrieval_denominator": len(answerable),
    }
    for name in ("recall_at_5", "recall_at_10", "recall_at_50", "mrr_at_50"):
        values = [
            item["retrieval_metrics"][name]
            for item in answerable
            if item["retrieval_metrics"][name] is not None
        ]
        summary[name] = _mean(values)
    faithfulness = [
        item["faithfulness"]
        for item in items
        if item.get("faithfulness") is not None
    ]
    summary.update(
        {
            "faithfulness": _mean(faithfulness),
            "faithfulness_count": len(faithfulness),
            "mean_retrievable_count": _mean(
                [float(item.get("retrievable_count", 0)) for item in items]
            ),
            "context_relevant_coverage": _mean(
                [
                    item["context_relevant_coverage"]
                    for item in answerable
                    if item.get("context_relevant_coverage") is not None
                ]
            ),
            "generation_sample_count": sum(bool(item["answered"]) for item in items),
            "answer_rate": _ratio(
                sum(bool(item["answered"]) for item in answerable),
                len(answerable),
            ),
            "unknown_refusal_rate": _ratio(
                sum(bool(item["refused"]) for item in unknown),
                len(unknown),
            ),
            "unknown_wrong_answer_rate": _ratio(
                sum(bool(item["answered"]) for item in unknown),
                len(unknown),
            ),
            "technical_failure_count": sum(
                item.get("error") is not None for item in items
            ),
            "mean_retrieval_ms": _mean(
                [item["timings_ms"]["retrieval"] for item in items]
            ),
            "mean_rerank_ms": _mean(
                [item["timings_ms"].get("rerank", 0.0) for item in items]
            ),
            "mean_total_ms": _mean(
                [item["timings_ms"]["total"] for item in items]
            ),
        }
    )
    return summary


def candidate_set_mismatches(results: list[dict[str, Any]]) -> list[str]:
    grouped: dict[str, dict[str, set[int]]] = defaultdict(dict)
    for item in results:
        if item["strategy"] in {"hybrid", "hybrid_rerank"}:
            grouped[item["query_id"]][item["strategy"]] = set(item["ranked_ids"])
    return sorted(
        query_id
        for query_id, values in grouped.items()
        if set(values) == {"hybrid", "hybrid_rerank"}
        and values["hybrid"] != values["hybrid_rerank"]
    )


def render_report(
    summary: dict[str, Any], *, mismatches: list[str], limited: bool = False
) -> str:
    lines = [
        "# Chapter 4 knowledge evaluation",
        "",
        "Faithfulness is produced by a separate model call and is not a human gold standard.",
        "`hybrid_rerank` applies the frozen relevance threshold; the other strategies do not.",
        "",
    ]
    if limited:
        lines.extend(
            ["SMOKE: this run used --limit and is not a full comparison.", ""]
        )
    if mismatches:
        lines.extend(
            [
                "INVALID: hybrid and hybrid_rerank candidate sets differ for: "
                + ", ".join(mismatches),
                "",
            ]
        )
    for strategy, groups in summary["strategies"].items():
        lines.extend(
            [
                f"## {strategy}",
                "",
                _markdown_table({"overall": groups["overall"]}),
                "",
                "### Main buckets",
                "",
                _markdown_table(groups["by_query_type"]),
                "",
                "### Difficulty",
                "",
                _markdown_table(groups["by_difficulty"]),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _markdown_table(groups: dict[str, dict[str, Any]]) -> str:
    rows = [
        "| group | n | retrieval n | retrievable | context coverage | R@5 | R@10 | R@50 | MRR@50 | faithfulness (n) | generated | answer rate | unknown refusal | unknown wrong answer | failures | retrieval ms | rerank ms | total ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, value in groups.items():
        rows.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(value["sample_count"]),
                    str(value["retrieval_denominator"]),
                    _format_metric(value["mean_retrievable_count"]),
                    _format_metric(value["context_relevant_coverage"]),
                    _format_metric(value["recall_at_5"]),
                    _format_metric(value["recall_at_10"]),
                    _format_metric(value["recall_at_50"]),
                    _format_metric(value["mrr_at_50"]),
                    f"{_format_metric(value['faithfulness'])} ({value['faithfulness_count']})",
                    str(value["generation_sample_count"]),
                    _format_metric(value["answer_rate"]),
                    _format_metric(value["unknown_refusal_rate"]),
                    _format_metric(value["unknown_wrong_answer_rate"]),
                    str(value["technical_failure_count"]),
                    _format_metric(value["mean_retrieval_ms"]),
                    _format_metric(value["mean_rerank_ms"]),
                    _format_metric(value["mean_total_ms"]),
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def build_calibration_artifact(
    provenance: RuntimeProvenance,
    *,
    calibration_cases_sha256: str,
    cases: list[dict[str, Any]],
    results: list[dict[str, Any]],
    max_false_accept: float = 0.1,
    smoke: bool = False,
) -> CalibrationArtifact:
    by_query = {
        item["query_id"]: item
        for item in results
        if item.get("strategy") == "hybrid_rerank"
    }
    samples = [
        (
            by_query.get(case["query_id"], {}).get("top_score"),
            bool(case["answerable"]),
        )
        for case in cases
    ]
    threshold = calibrate_threshold(samples, max_false_accept=max_false_accept)
    technical_failures = sum(
        case["query_id"] not in by_query
        or by_query[case["query_id"]].get("error") is not None
        for case in cases
    )
    answerable_count = sum(bool(case["answerable"]) for case in cases)
    unknown_count = len(cases) - answerable_count
    unknown_false_accepts = sum(
        score is not None and score >= threshold
        for score, answerable in samples
        if not answerable
    )
    if smoke:
        readiness = "smoke"
    elif technical_failures:
        readiness = "failed"
    elif len(cases) == 30 and answerable_count == 20 and unknown_count == 10:
        readiness = "production"
    else:
        readiness = "partial"
    return CalibrationArtifact(
        **provenance.model_dump(mode="python"),
        readiness=readiness,
        strategy="hybrid_rerank",
        threshold=threshold,
        calibration_cases_sha256=calibration_cases_sha256,
        sample_count=len(cases),
        answerable_count=answerable_count,
        unknown_count=unknown_count,
        technical_failure_count=technical_failures,
        unknown_false_accept_count=unknown_false_accepts,
        max_false_accept=max_false_accept,
    )


def read_results(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        values = [json.loads(line) for line in lines if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationDataError("results.jsonl is unreadable") from exc
    if any(not isinstance(value, dict) for value in values):
        raise EvaluationDataError("results.jsonl contains a non-object")
    return values


def write_calibration(path: Path, artifact: CalibrationArtifact) -> None:
    _atomic_json(path, artifact.model_dump(mode="json"))


def invalidate_run(output_dir: Path, reason: str) -> None:
    """Atomically mark an already materialized run unusable."""
    manifest_path = output_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest.update(
        {
            "status": "invalid",
            "invalid_reason": reason,
            "completed_at": _utc_now(),
        }
    )
    _atomic_json(manifest_path, manifest)
    report_path = output_dir / "report.md"
    try:
        report = report_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvaluationDataError("report.md is unreadable") from exc
    _atomic_text(report_path, f"INVALID: {reason}\n\n{report}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, strict_json_dumps(value, indent=2) + "\n")


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationDataError(f"invalid cached artifact: {path.name}") from exc
    if not isinstance(value, dict):
        raise EvaluationDataError(f"cached artifact must be an object: {path.name}")
    return value


def _key_digest(*values: str) -> str:
    return hashlib.sha256("\0".join(values).encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _format_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"
