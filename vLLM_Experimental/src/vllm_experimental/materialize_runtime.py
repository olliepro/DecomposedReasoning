"""Materialize a patched vLLM runtime copy under scratch."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import shutil
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

from vllm_experimental.types import DEFAULT_SCRATCH_ROOT, VllmRuntimeSpec


def hash_paths(*, paths: tuple[Path, ...]) -> str:
    """Return a stable hash over file contents."""

    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.name).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def package_hash(*, package_root: Path) -> str:
    """Return a lightweight hash of selected vLLM source files."""

    paths = tuple(
        path for path in package_root.rglob("*.py") if "/__pycache__/" not in str(path)
    )
    return hash_paths(paths=paths[:500])


def remove_runtime_tree(*, runtime_root: Path) -> None:
    """Remove an incomplete runtime tree, retrying GPFS delayed deletes."""

    for attempt in range(5):
        try:
            shutil.rmtree(runtime_root)
            return
        except OSError:
            if attempt == 4:
                raise
            time.sleep(0.5 * (attempt + 1))


def materialize_runtime(
    *,
    runtime_parent: Path,
    patches_dir: Path,
    force: bool = False,
) -> VllmRuntimeSpec:
    """Copy installed vLLM to scratch and apply repo-tracked patches."""

    import vllm

    source_package = Path(inspect.getfile(vllm)).resolve().parent
    vllm_version = version("vllm")
    patch_files = tuple(path.resolve() for path in sorted(patches_dir.glob("*.patch")))
    patch_hash = hash_paths(paths=patch_files) if patch_files else "no_patches"
    source_digest = package_hash(package_root=source_package)
    runtime_root = runtime_parent / f"vllm_{vllm_version}_{source_digest}_{patch_hash}"
    runtime_package = runtime_root / "vllm"

    manifest_path = runtime_root / "vllm_experimental_runtime_manifest.json"
    if runtime_root.exists() and (force or not manifest_path.exists()):
        remove_runtime_tree(runtime_root=runtime_root)
    if not runtime_package.exists():
        runtime_root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_package, runtime_package)
        for patch_file in patch_files:
            subprocess.run(
                ["patch", "--dry-run", "-p0", "-i", str(patch_file)],
                cwd=runtime_root,
                check=True,
                stdout=sys.stderr,
                stderr=sys.stderr,
            )
            subprocess.run(
                ["patch", "-p0", "-i", str(patch_file)],
                cwd=runtime_root,
                check=True,
                stdout=sys.stderr,
                stderr=sys.stderr,
            )

    spec = VllmRuntimeSpec(
        source_package=source_package,
        runtime_root=runtime_root,
        vllm_version=vllm_version,
        source_hash=source_digest,
        patch_hash=patch_hash,
        patch_files=tuple(path.name for path in patch_files),
    )
    manifest_path.write_text(
        json.dumps(spec.manifest_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return spec


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for runtime materialization."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime-parent",
        type=Path,
        default=DEFAULT_SCRATCH_ROOT / "runtimes",
    )
    parser.add_argument("--patches-dir", type=Path, default=Path("patches"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    spec = materialize_runtime(
        runtime_parent=args.runtime_parent,
        patches_dir=args.patches_dir,
        force=args.force,
    )
    print(json.dumps(spec.manifest_payload(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
