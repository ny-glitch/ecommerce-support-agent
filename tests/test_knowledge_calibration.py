from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.knowledge.calibration import (
    CalibrationArtifact,
    build_runtime_provenance,
    file_sha256,
    load_calibration,
)
from tests.helpers import settings


MODEL_FILES = {
    "bge-m3": (
        "BAAI/bge-m3",
        "5617a9f61b028005a4858fdac845db406aefb181",
        "embed.bin",
    ),
    "bge-reranker-v2-m3": (
        "BAAI/bge-reranker-v2-m3",
        "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        "rerank.bin",
    ),
}


def write_model_manifest(root: Path) -> Path:
    models = []
    for name, (repo_id, revision, filename) in MODEL_FILES.items():
        directory = root / name
        directory.mkdir(parents=True)
        path = directory / filename
        path.write_bytes((name + "-weights").encode())
        models.append(
            {
                "name": name,
                "repo_id": repo_id,
                "revision": revision,
                "files": [
                    {
                        "path": filename,
                        "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                ],
            }
        )
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "models": models}), encoding="utf-8"
    )
    return manifest


def runtime_settings(root: Path):
    return settings(knowledge_models_dir=root)


def test_artifact_accepts_finite_all_refuse_threshold_and_round_trips(tmp_path: Path) -> None:
    manifest = write_model_manifest(tmp_path / "models")
    configuration = runtime_settings(manifest.parent)
    provenance = build_runtime_provenance(
        configuration, corpus_fingerprint="a" * 64
    )
    artifact = CalibrationArtifact(
        **provenance.model_dump(),
        readiness="production",
        strategy="hybrid_rerank",
        threshold=1.0000000000000002,
        calibration_cases_sha256="b" * 64,
        sample_count=30,
        answerable_count=20,
        unknown_count=10,
        technical_failure_count=0,
        unknown_false_accept_count=0,
        max_false_accept=0.1,
    )
    path = tmp_path / "calibration.json"
    path.write_text(artifact.model_dump_json(), encoding="utf-8")

    loaded = load_calibration(path, expected=provenance)

    assert loaded == artifact
    assert loaded.threshold > 1
    assert provenance.model_manifest_sha256 == file_sha256(manifest)


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), float("-inf")])
def test_artifact_rejects_non_finite_threshold(threshold: float) -> None:
    with pytest.raises(ValidationError):
        CalibrationArtifact(
            schema_version=1,
            corpus_fingerprint="a" * 64,
            embedding_model="BAAI/bge-m3",
            embedding_revision="embed-revision",
            reranker_model="BAAI/bge-reranker-v2-m3",
            reranker_revision="rerank-revision",
            model_manifest_sha256="c" * 64,
            prompt_bundle_sha256="d" * 64,
            readiness="production",
            strategy="hybrid_rerank",
            threshold=threshold,
            calibration_cases_sha256="b" * 64,
            sample_count=30,
            answerable_count=20,
            unknown_count=10,
            technical_failure_count=0,
            unknown_false_accept_count=0,
            max_false_accept=0.1,
        )


def test_reader_rejects_stale_corpus_or_model_files_with_diagnostic(tmp_path: Path) -> None:
    manifest = write_model_manifest(tmp_path / "models")
    configuration = runtime_settings(manifest.parent)
    provenance = build_runtime_provenance(
        configuration, corpus_fingerprint="a" * 64
    )
    artifact = CalibrationArtifact(
        **provenance.model_dump(),
        readiness="production",
        strategy="hybrid_rerank",
        threshold=0.42,
        calibration_cases_sha256="b" * 64,
        sample_count=30,
        answerable_count=20,
        unknown_count=10,
        technical_failure_count=0,
        unknown_false_accept_count=1,
        max_false_accept=0.1,
    )
    path = tmp_path / "calibration.json"
    path.write_text(artifact.model_dump_json(), encoding="utf-8")

    with pytest.raises(RuntimeError, match="corpus_fingerprint"):
        load_calibration(
            path,
            expected=provenance.model_copy(
                update={"corpus_fingerprint": "f" * 64}
            ),
        )

    model_path = manifest.parent / "bge-m3" / "embed.bin"
    model_path.write_bytes(b"x" * model_path.stat().st_size)
    with pytest.raises(RuntimeError, match="sha256"):
        build_runtime_provenance(configuration, corpus_fingerprint="a" * 64)


def test_provenance_build_does_not_require_calibration_artifact(tmp_path: Path) -> None:
    manifest = write_model_manifest(tmp_path / "models")
    configuration = runtime_settings(manifest.parent).model_copy(
        update={"knowledge_calibration_path": tmp_path / "absent.json"}
    )

    provenance = build_runtime_provenance(
        configuration, corpus_fingerprint="a" * 64
    )

    assert provenance.corpus_fingerprint == "a" * 64
    assert provenance.prompt_bundle_sha256
    assert not configuration.knowledge_calibration_path.exists()


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"readiness": "partial"}, "not production-ready"),
        ({"sample_count": 29, "answerable_count": 19}, "30 samples"),
        ({"technical_failure_count": 1}, "technical failures"),
        ({"unknown_false_accept_count": 2}, "false-accept"),
    ],
)
def test_production_reader_rejects_incomplete_or_invalid_calibration(
    tmp_path: Path, changes: dict[str, object], match: str
) -> None:
    manifest = write_model_manifest(tmp_path / "models")
    configuration = runtime_settings(manifest.parent)
    provenance = build_runtime_provenance(
        configuration, corpus_fingerprint="a" * 64
    )
    values = {
        **provenance.model_dump(),
        "readiness": "production",
        "strategy": "hybrid_rerank",
        "threshold": 0.42,
        "calibration_cases_sha256": "b" * 64,
        "sample_count": 30,
        "answerable_count": 20,
        "unknown_count": 10,
        "technical_failure_count": 0,
        "unknown_false_accept_count": 1,
        "max_false_accept": 0.1,
        **changes,
    }
    artifact = CalibrationArtifact(**values)
    path = tmp_path / "calibration.json"
    path.write_text(artifact.model_dump_json(), encoding="utf-8")

    with pytest.raises(RuntimeError, match=match):
        load_calibration(path, expected=provenance)
