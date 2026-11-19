from __future__ import annotations

import hashlib
import os
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import psutil
import pytest

from api import Aborted, NodeContext, Progress
from nodes.impl.opencomic import pipeline as oc
from nodes.impl.opencomic.runtime import process_image


def steps(**changes):
    args = {
        "width": 1000,
        "height": 1000,
        "artifact": oc.OFF,
        "artifact_param": None,
        "descreen": oc.OFF,
        "descreen_param": None,
        "keep": True,
        "min_size": 3,
        "upscale": oc.OFF,
        "upscale_param": None,
        "native": 2,
        "auto": False,
        "scale": 2,
        "max_mp": 2,
        "noise": 0,
        "tile": 0,
        "tta": False,
        "fp32": False,
    }
    return oc.steps_for(**(args | changes))


def test_reader_order_mask_and_bypass():
    image = np.array([[0.123456]], np.float32)
    assert process_image(None, image, *steps()) is image
    passes, factor = steps(descreen=oc.DESCREEN)
    assert factor == 1
    mask = passes[0]["keepBigHalftone"]
    assert mask == {
        "model": oc.MASK,
        "minSize": 3,
        "artifactRemoval": {"model": oc.ARTIFACT},
    }
    passes, factor = steps(
        artifact=oc.ARTIFACT, descreen=oc.DESCREEN, upscale=oc.UPSCALE
    )
    assert [p["model"] for p in passes] == [oc.ARTIFACT, oc.DESCREEN, oc.UPSCALE]
    assert "artifactRemoval" not in passes[1]["keepBigHalftone"]
    assert factor == 2
    assert "keepBigHalftone" not in steps(descreen=oc.DESCREEN, keep=False)[0][0]


@pytest.mark.parametrize(
    "width,height,limit,auto,model,manual,factor",
    [
        (500, 500, 2, True, oc.UPSCALE, 2, 4),
        (1000, 1000, 2, True, oc.UPSCALE, 2, 3),
        (2000, 1000, 2, True, oc.UPSCALE, 4, 2),
        (2001, 1000, 2, False, oc.UPSCALE, 4, 1),
        (3000, 2000, 0, False, oc.UPSCALE, 3, 3),
        (500, 500, 2, False, "waifu2x-models-cunet", 8, 8),
        (1000, 1000, 2, False, "waifu2x-models-cunet", 3, 2),
    ],
)
def test_scale_gate(width, height, limit, auto, model, manual, factor):
    chain, actual = steps(
        width=width,
        height=height,
        max_mp=limit,
        auto=auto,
        upscale=model,
        scale=manual,
        descreen=oc.DESCREEN,
    )
    assert actual == factor
    assert len(chain) == (1 if factor == 1 else 2)


def test_catalog_and_custom_pair(tmp_path):
    for stage, models in oc.MODELS.items():
        assert models
        for key, model in models.items():
            assert set(model["files"]) <= oc.CATALOG["files"].keys()
            assert all(
                name.startswith(stage + "/")
                for name in oc.model_files([{"model": key}])
            )
    param = tmp_path / "custom.param"
    param.touch()
    with pytest.raises(ValueError, match="matching"):
        steps(upscale=oc.CUSTOM, upscale_param=param)
    weights = param.with_suffix(".bin")
    weights.write_bytes(b"one")
    args = {"upscale": oc.CUSTOM, "upscale_param": param, "native": 3}
    passes, scale = steps(**args)
    assert not oc.model_files(passes)
    assert scale == 3
    weights.write_bytes(b"different")
    assert passes[0]["model"] != steps(**args)[0][0]["model"]


def test_download_verification_and_retry(tmp_path, monkeypatch):
    data = b"correct"
    sha = hashlib.sha1(b"blob 7\x00" + data).hexdigest()
    monkeypatch.setitem(oc.CATALOG, "files", {"model.bin": [7, sha]})
    monkeypatch.setattr(oc, "model_files", lambda _: {"model.bin"})
    body = Mock()
    body.iter_content.side_effect = [[b"damaged"], [data], [data]]
    get = Mock(side_effect=lambda *a, **k: nullcontext(body))
    monkeypatch.setattr(oc.requests, "get", get)
    root = Path(tmp_path)
    run = lambda: oc.prepare_models(root, [], Progress.noop_progress())  # noqa: E731
    with pytest.raises(ValueError, match="incorrect"):
        run()
    assert not list(root.iterdir())
    run()
    run()
    assert get.call_count == 2
    path = root / "model.bin"
    path.write_bytes(b"broken")
    run()
    assert path.read_bytes() == data


@pytest.mark.skipif(
    not os.environ.get("CHAINNER_OPENCOMIC_GPU_TEST"),
    reason="Optional native Vulkan runtime",
)
def test_native_worker_reuse_cancel_and_alpha():
    ctx = Mock(spec=NodeContext)
    image = np.random.default_rng(42).random((95, 103, 4), dtype=np.float32)
    passes, scale = steps(descreen=oc.DESCREEN)
    try:
        output = process_image(ctx, image, passes, scale)
        np.testing.assert_array_equal(output[:, :, 3], image[:, :, 3])
        worker = ctx.add_cleanup.call_args.args[0].__self__
        proc = psutil.Process(worker.proc.pid)
        native = proc.children(recursive=True)
        assert native
        output = process_image(ctx, image[:, :, 0], passes, scale)
        assert output.shape == image.shape[:2]
        assert set(native) & set(proc.children(recursive=True))
        ctx.check_aborted.side_effect = Aborted()
        with pytest.raises(Aborted):
            process_image(ctx, np.tile(image, (8, 8, 1)), passes, scale)
        assert not psutil.wait_procs([proc, *native], timeout=5)[1]
    finally:
        for call in ctx.add_cleanup.call_args_list:
            call.args[0]()


@pytest.mark.skipif(
    not os.environ.get("CHAINNER_OPENCOMIC_GPU_TEST"),
    reason="Optional native Vulkan runtime",
)
def test_tta_changes_pixels_and_reuses_the_right_network():
    ctx = Mock(spec=NodeContext)
    image = np.random.default_rng(17).random((65, 97, 3), dtype=np.float32)
    off = steps(upscale=oc.UPSCALE)
    on = steps(upscale=oc.UPSCALE, tta=True)
    try:
        plain = process_image(ctx, image, *off)
        worker = ctx.add_cleanup.call_args.args[0].__self__
        proc = psutil.Process(worker.proc.pid)

        def native_networks():
            return [
                child
                for child in proc.children(recursive=True)
                if child.name().lower().removesuffix(".exe") == "upscayl-bin"
            ]

        original = native_networks()
        assert len(original) == 1 and "-x" not in original[0].cmdline()
        augmented = process_image(ctx, image, *on)
        networks = set(native_networks())
        assert len(networks) == 2
        assert any("-x" in child.cmdline() for child in networks)
        assert np.any(plain != augmented)
        np.testing.assert_array_equal(process_image(ctx, image, *on), augmented)
        np.testing.assert_array_equal(process_image(ctx, image, *off), plain)
        assert set(native_networks()) == networks
    finally:
        for call in ctx.add_cleanup.call_args_list:
            call.args[0]()
