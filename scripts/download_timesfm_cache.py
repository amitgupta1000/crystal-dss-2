"""Download TimesFM weights into a local Hugging Face cache on Windows."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

_DEFAULT_REPO = "google/timesfm-2.0-500m-pytorch"
_DEFAULT_CACHE = Path(r"C:\hf_cache")


def _resolve_cache_dir(path: str | Path | None) -> Path:
    """Return an absolute cache directory path, creating it if needed."""
    target = Path(path).expanduser() if path is not None else _DEFAULT_CACHE
    target.mkdir(parents=True, exist_ok=True)
    if not target.is_dir():  # pragma: no cover - defensive
        raise RuntimeError(f"Cache path {target} is not a directory")
    return target.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prime the local cache with TimesFM weights.")
    parser.add_argument(
        "--repo",
        default=os.environ.get("TIMESFM_REPO_ID", _DEFAULT_REPO),
        help="Hugging Face repo ID for the TimesFM checkpoint.",
    )
    parser.add_argument(
        "--cache",
        default=os.environ.get("HF_HOME", str(_DEFAULT_CACHE)),
        help="Target cache directory on the local drive (default: C\\hf_cache).",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional specific git revision or tag to download.",
    )
    args = parser.parse_args()

    cache_dir = _resolve_cache_dir(args.cache)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    print(f"Downloading {args.repo} to cache: {cache_dir}")
    snapshot_path = snapshot_download(
        repo_id=args.repo,
        cache_dir=cache_dir,
        allow_patterns=["*.pt", "*.safetensors", "*.json", "*.txt"],  # Only essential files
        ignore_patterns=["*.md", "*.git*"],  # Skip docs and git files
        revision=args.revision,
        resume_download=True,
    )
    print(f"Snapshot stored under: {snapshot_path}")


if __name__ == "__main__":
    main()
