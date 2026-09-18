from __future__ import annotations

import hashlib
import json
import math
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from app.config import Settings


Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_PROMPT_NAMES = (
    "query_normalization.txt",
    "evidence_assessment.txt",
    "knowledge_answer.txt",
    "tool_chat.txt",
)


class RuntimeProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    corpus_fingerprint: Sha256
    embedding_model: str
    embedding_revision: str
    reranker_model: str
    reranker_revision: str
    model_manifest_sha256: Sha256
    prompt_bundle_sha256: Sha256


class CalibrationArtifact(RuntimeProvenance):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    readiness: Literal["production", "partial", "smoke", "failed"]
    strategy: Literal["hybrid_rerank"]
    threshold: float
    calibration_cases_sha256: Sha256
    sample_count: int = Field(gt=0)
    answerable_count: int = Field(ge=0)
    unknown_count: int = Field(ge=0)
    technical_failure_count: int = Field(ge=0)
    unknown_false_accept_count: int = Field(ge=0)
    max_false_accept: float = Field(ge=0, le=0.1)

    @model_validator(mode="after")
    def validate_counts(self) -> "CalibrationArtifact":
        if self.answerable_count + self.unknown_count != self.sample_count:
            raise ValueError("answerable and unknown counts must equal sample count")
        if self.unknown_false_accept_count > self.unknown_count:
            raise ValueError("unknown false accepts cannot exceed unknown count")
        if self.technical_failure_count > self.sample_count:
            raise ValueError("technical failures cannot exceed sample count")
        return self


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise RuntimeError(f"required artifact is unreadable: {path}") from exc
    return digest.hexdigest()


def prompt_bundle_fingerprint() -> str:
    digest = hashlib.sha256()
    root = resources.files("app").joinpath("prompts")
    for name in _PROMPT_NAMES:
        try:
            payload = root.joinpath(name).read_bytes()
        except OSError as exc:
            raise RuntimeError(f"required prompt is unavailable: {name}") from exc
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest()


def model_manifest_fingerprint(settings: Settings) -> str:
    path = settings.knowledge_models_dir / "manifest.json"
    try:
        raw = path.read_bytes()
        manifest = json.loads(raw)
    except OSError as exc:
        raise RuntimeError(
            "local model manifest is missing; run scripts/prepare_knowledge_models.py"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("local model manifest is invalid JSON") from exc
    _validate_model_manifest(manifest, settings.knowledge_models_dir, settings)
    return hashlib.sha256(raw).hexdigest()


def build_runtime_provenance(
    settings: Settings,
    *,
    corpus_fingerprint: str,
) -> RuntimeProvenance:
    return RuntimeProvenance(
        corpus_fingerprint=corpus_fingerprint,
        embedding_model=settings.knowledge_embedding_model,
        embedding_revision=settings.knowledge_embedding_revision,
        reranker_model=settings.knowledge_reranker_model,
        reranker_revision=settings.knowledge_reranker_revision,
        model_manifest_sha256=model_manifest_fingerprint(settings),
        prompt_bundle_sha256=prompt_bundle_fingerprint(),
    )


def load_calibration(
    path: Path,
    *,
    expected: RuntimeProvenance,
) -> CalibrationArtifact:
    try:
        artifact = CalibrationArtifact.model_validate_json(
            path.read_text(encoding="utf-8"), strict=True
        )
    except OSError as exc:
        raise RuntimeError(
            "knowledge calibration artifact is missing; run the calibration command"
        ) from exc
    except ValidationError as exc:
        raise RuntimeError("knowledge calibration artifact has an invalid schema") from exc

    actual = artifact.model_dump(include=set(RuntimeProvenance.model_fields))
    required = expected.model_dump()
    stale = [name for name, value in required.items() if actual.get(name) != value]
    if stale:
        raise RuntimeError(
            "knowledge calibration artifact does not match runtime provenance: "
            + ", ".join(stale)
        )
    if not math.isfinite(artifact.threshold):
        raise RuntimeError("knowledge calibration threshold must be finite")
    if artifact.readiness != "production":
        raise RuntimeError("knowledge calibration artifact is not production-ready")
    if (
        artifact.sample_count != 30
        or artifact.answerable_count != 20
        or artifact.unknown_count != 10
    ):
        raise RuntimeError(
            "production calibration must contain 30 samples: 20 answerable and 10 unknown"
        )
    if artifact.technical_failure_count:
        raise RuntimeError("knowledge calibration has unresolved technical failures")
    false_accept_rate = (
        artifact.unknown_false_accept_count / artifact.unknown_count
    )
    if false_accept_rate > artifact.max_false_accept:
        raise RuntimeError("knowledge calibration exceeds the unknown false-accept limit")
    return artifact


def load_runtime_calibration(
    settings: Settings,
    *,
    corpus_fingerprint: str,
) -> CalibrationArtifact:
    expected = build_runtime_provenance(
        settings,
        corpus_fingerprint=corpus_fingerprint,
    )
    return load_calibration(settings.knowledge_calibration_path, expected=expected)


def _validate_model_manifest(
    manifest: Any,
    root: Path,
    settings: Settings,
) -> None:
    if not isinstance(manifest, dict) or set(manifest) != {"schema_version", "models"}:
        raise RuntimeError("local model manifest has an invalid schema")
    if manifest["schema_version"] != 1 or not isinstance(manifest["models"], list):
        raise RuntimeError("local model manifest has an unsupported schema version")

    expected = {
        "bge-m3": (
            settings.knowledge_embedding_model,
            settings.knowledge_embedding_revision,
        ),
        "bge-reranker-v2-m3": (
            settings.knowledge_reranker_model,
            settings.knowledge_reranker_revision,
        ),
    }
    records: dict[str, dict[str, Any]] = {}
    for item in manifest["models"]:
        if not isinstance(item, dict) or set(item) != {
            "name", "repo_id", "revision", "files"
        }:
            raise RuntimeError("local model manifest has an invalid model entry")
        name = item["name"]
        if not isinstance(name, str) or name in records:
            raise RuntimeError("local model manifest contains duplicate model names")
        records[name] = item
    if set(records) != set(expected):
        raise RuntimeError("local model manifest does not contain the required models")

    for name, (repo_id, revision) in expected.items():
        item = records[name]
        if item["repo_id"] != repo_id or item["revision"] != revision:
            raise RuntimeError(f"local model manifest revision mismatch for {name}")
        files = item["files"]
        if not isinstance(files, list) or not files:
            raise RuntimeError(f"local model manifest has no files for {name}")
        seen: set[str] = set()
        for entry in files:
            if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256"}:
                raise RuntimeError(f"local model manifest has an invalid file for {name}")
            relative = entry["path"]
            if (
                not isinstance(relative, str)
                or not relative
                or PurePosixPath(relative).is_absolute()
                or ".." in PurePosixPath(relative).parts
                or relative in seen
            ):
                raise RuntimeError(f"local model manifest has an unsafe file path for {name}")
            seen.add(relative)
            model_path = root / name / relative
            try:
                size = model_path.stat().st_size
            except OSError as exc:
                raise RuntimeError(f"local model file is missing for {name}: {relative}") from exc
            if (
                not isinstance(entry["bytes"], int)
                or isinstance(entry["bytes"], bool)
                or size != entry["bytes"]
            ):
                raise RuntimeError(f"local model file bytes mismatch for {name}: {relative}")
            digest = entry["sha256"]
            if not isinstance(digest, str) or file_sha256(model_path) != digest:
                raise RuntimeError(f"local model file sha256 mismatch for {name}: {relative}")
