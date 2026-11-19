"""Map reader controls to the pinned OpenComic pipeline and validate its weights."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.parse import quote

import requests

from api import InputId, Progress

CATALOG = json.loads(Path(__file__).with_name("catalog.json").read_text("utf-8"))
CACHE_VERSION = ("opencomic-ai-bin-1.6.1", CATALOG["revision"], 1)
ADVANCED_INPUT = InputId(14)
TTA_INPUT = InputId(16)
MODELS = CATALOG["models"]
ARTIFACT = "opencomic-ai-artifact-removal-balanced-v3-ps"
DESCREEN = "opencomic-ai-descreen-hard-fast-v4-preview"
MASK = "opencomic-ai-descreen-mask-fast-v3-256"
UPSCALE = "opencomic-ai-upscale-fast-v3"
OFF = "off"
CUSTOM = "custom"


def _step(stage: str, key: str, param: Path | None, native: int):
    if key != CUSTOM:
        if key not in MODELS[stage]:
            raise ValueError(f"Unknown OpenComic {stage} model: {key}")
        return {"model": key}, MODELS[stage][key]["scales"]
    if param is None:
        raise ValueError(f"Select a custom {stage} NCNN .param file.")
    param = param.resolve()
    weights = param.with_suffix(".bin")
    if param.suffix.lower() != ".param" or not param.is_file() or not weights.is_file():
        raise ValueError(f"Expected a matching NCNN .param and .bin beside {param}")
    identity = [
        (str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in (param, weights)
    ]
    key = (
        "local-" + hashlib.sha256(repr((stage, native, identity)).encode()).hexdigest()
    )
    return {
        "model": key,
        "custom": {"param": str(param), "scale": native, "stage": stage},
    }, [native]


def steps_for(
    width: int,
    height: int,
    artifact: str,
    artifact_param: Path | None,
    descreen: str,
    descreen_param: Path | None,
    keep: bool,
    min_size: float,
    upscale: str,
    upscale_param: Path | None,
    native: int,
    auto: bool,
    scale: int,
    max_mp: float,
    noise: int,
    tile: int,
    tta: bool,
    fp32: bool,
) -> tuple[list[dict[str, Any]], int]:
    options = {k: v for k, v in (("tileSize", tile), ("tta", tta), ("fp32", fp32)) if v}
    steps = []
    if artifact != OFF:
        step, _ = _step("artifact-removal", artifact, artifact_param, 1)
        steps.append({**step, **options})
    if descreen != OFF:
        step, _ = _step("descreen", descreen, descreen_param, 1)
        if keep:
            mask = {"model": MASK, "minSize": min_size, **options}
            if artifact == OFF:
                mask["artifactRemoval"] = {"model": ARTIFACT, **options}
            step["keepBigHalftone"] = mask
        steps.append({**step, **options})
    mp = width * height / 1_000_000
    factor = 1
    if upscale != OFF and (max_mp == 0 or mp <= max_mp):
        step, scales = _step("upscale", upscale, upscale_param, native)
        if auto:
            # OpenComic rounds positive values the JavaScript way. A disabled
            # input limit keeps the reader's default 2 MP reference for auto scale.
            scale = min(4, max(2, math.floor(2 * math.sqrt((max_mp or 2) / mp) + 0.5)))
        factor = min(scales, key=lambda s: abs(s - scale))
        steps.append({**step, **options, "scale": factor, "noise": noise})
    return steps, factor


def model_files(steps: Iterable[dict]) -> set[str]:
    files: set[str] = set()
    for step in steps:
        if not step.get("custom"):
            for models in MODELS.values():
                if step["model"] in models:
                    files.update(models[step["model"]]["files"])
                    break
        for field in ("keepBigHalftone", "artifactRemoval"):
            if field in step:
                files.update(model_files([step[field]]))
    return files


@lru_cache(maxsize=512)
def _verified(path: Path, size: int, mtime_ns: int, sha: str) -> bool:
    # These are Git blob IDs from the official immutable model tree.
    digest = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == sha


def prepare_models(root: Path, steps: list[dict], progress: Progress):
    names = sorted(model_files(steps))
    for index, name in enumerate(names):
        progress.suspend()
        path = root / name
        size, sha = CATALOG["files"][name]
        if path.is_file():
            stat = path.stat()
            if stat.st_size == size and _verified(path, size, stat.st_mtime_ns, sha):
                continue
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/ollm/opencomic-ai-models/{CATALOG['revision']}/models/{quote(name)}"
        with NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as tmp:
            partial = Path(tmp.name)
        try:
            digest = hashlib.sha1(f"blob {size}\0".encode())
            with (
                requests.get(url, stream=True, timeout=(8, 15)) as response,
                partial.open("wb") as out,
            ):
                response.raise_for_status()
                written = 0
                for block in response.iter_content(256 * 1024):
                    progress.suspend()
                    out.write(block)
                    digest.update(block)
                    written += len(block)
                    progress.set_progress(
                        0.2 * (index + min(written / size, 1)) / len(names)
                    )
            if written != size or digest.hexdigest() != sha:
                raise ValueError(
                    f"Incomplete or incorrect OpenComic model file: {name}. Run again to retry."
                )
            os.replace(partial, path)
        finally:
            partial.unlink(missing_ok=True)
