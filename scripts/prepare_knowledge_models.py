from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    revision: str
    required_files: tuple[str, ...]


MODEL_SPECS = {
    "bge-m3": ModelSpec(
        repo_id="BAAI/bge-m3",
        revision="5617a9f61b028005a4858fdac845db406aefb181",
        required_files=(
            "README.md",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "sentencepiece.bpe.model",
            "pytorch_model.bin",
            "colbert_linear.pt",
            "sparse_linear.pt",
        ),
    ),
    "bge-reranker-v2-m3": ModelSpec(
        repo_id="BAAI/bge-reranker-v2-m3",
        revision="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        required_files=(
            "README.md",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "sentencepiece.bpe.model",
            "model.safetensors",
        ),
    ),
}


def prepare_models(
    models_dir: Path,
    *,
    cache_dir: Path,
    downloader: Callable[..., str] = snapshot_download,
) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_models: list[dict[str, Any]] = []
    for name, spec in MODEL_SPECS.items():
        local_dir = models_dir / name
        if not _valid_local_snapshot(local_dir, spec):
            downloader(
                repo_id=spec.repo_id,
                revision=spec.revision,
                local_dir=local_dir,
                cache_dir=cache_dir,
                allow_patterns=list(spec.required_files),
                ignore_patterns=["onnx/**", "*.onnx", "*.gguf", "*.h5"],
                token=False,
                max_workers=3,
            )
        _validate_snapshot(local_dir, spec)
        manifest_models.append(
            {
                "name": name,
                "repo_id": spec.repo_id,
                "revision": spec.revision,
                "files": [
                    _file_manifest(local_dir, filename)
                    for filename in spec.required_files
                ],
            }
        )

    manifest_path = models_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {"schema_version": 1, "models": manifest_models},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _metadata_revision(local_dir: Path, filename: str) -> str | None:
    path = local_dir / ".cache" / "huggingface" / "download" / f"{filename}.metadata"
    try:
        return path.read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError):
        return None


def _valid_local_snapshot(local_dir: Path, spec: ModelSpec) -> bool:
    return all(
        (local_dir / filename).is_file()
        and _metadata_revision(local_dir, filename) == spec.revision
        for filename in spec.required_files
    )


def _validate_snapshot(local_dir: Path, spec: ModelSpec) -> None:
    missing = [
        filename
        for filename in spec.required_files
        if not (local_dir / filename).is_file()
    ]
    if missing:
        raise RuntimeError(
            f"{spec.repo_id}@{spec.revision} is incomplete: {', '.join(missing)}"
        )
    wrong_revisions = [
        filename
        for filename in spec.required_files
        if _metadata_revision(local_dir, filename) != spec.revision
    ]
    if wrong_revisions:
        raise RuntimeError(
            f"{spec.repo_id} local metadata does not match revision {spec.revision}: "
            + ", ".join(wrong_revisions)
        )


def _file_manifest(local_dir: Path, filename: str) -> dict[str, Any]:
    path = local_dir / filename
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": filename,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare fixed local embedding and reranker snapshots."
    )
    parser.add_argument(
        "--models-dir", type=Path, default=Path(".cache/ch04/models")
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path(".cache/ch04/huggingface")
    )
    args = parser.parse_args(argv)
    print(prepare_models(args.models_dir, cache_dir=args.cache_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
