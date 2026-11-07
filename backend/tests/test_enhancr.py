"""FIGSR/SMoSR checkpoint and node compatibility; synthetic weights run in CI."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
spandrel = pytest.importorskip("spandrel")
adapter = importlib.import_module("nodes.impl.pytorch.enhancr")
FIGSR = importlib.import_module("nodes.impl.pytorch.architecture.figsr.arch").FIGSR
SMoSR = importlib.import_module("nodes.impl.pytorch.architecture.smosr.arch").SMoSR


@pytest.fixture(scope="module", autouse=True)
def limit_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def make_model(arch, scale=2, rep=True):
    torch.manual_seed(42)
    if arch == "FIGSR":
        return FIGSR(
            dim=16,
            gc=4,
            n_blocks=3,
            expansion_ratio=2,
            scale=scale,
            square_kernel_size=3,
            band_kernel_size=5,
        )
    return SMoSR(dim=8, n_mb=2, scale=scale, rep=rep)


def make_loader():
    registry = spandrel.ArchRegistry()
    adapter.register_enhancr(registry)
    return spandrel.ModelLoader(registry=registry)


@pytest.mark.parametrize("arch", ["FIGSR", "SMoSR"])
@pytest.mark.parametrize("scale", [1, 2, 4])
def test_safetensors_metadata_and_eval_match_reference(tmp_path, arch, scale):
    from safetensors.torch import save_file

    # Save training-form weights. SMoSR must fuse the loaded values during eval,
    # rather than use stale eval_conv tensors stored in the checkpoint.
    reference = make_model(arch, scale)
    path = tmp_path / "wrong-name-8x.safetensors"
    save_file(reference.state_dict(), str(path))
    loaded = make_loader().load_from_file(path).eval()
    reference.eval()
    assert loaded.architecture.id == arch
    assert loaded.scale == scale
    assert loaded.input_channels == loaded.output_channels == 3
    image = torch.rand(1, 3, 9, 13)
    with torch.inference_mode():
        expected = reference(image).clamp(0, 1)
    torch.testing.assert_close(loaded(image), expected)
    assert loaded(image).shape == (1, 3, 9 * scale, 13 * scale)


@pytest.mark.parametrize("rep", [False, True])
def test_smosr_rep_metadata_and_fusion(rep):
    reference = make_model("SMoSR", rep=rep)
    loaded = make_loader().load_from_state_dict(reference.state_dict()).eval()
    reference.eval()
    image = torch.rand(1, 3, 11, 16)
    with torch.inference_mode():
        expected = reference(image).clamp(0, 1)
    torch.testing.assert_close(loaded(image), expected)
    assert ("rep" in loaded.tags) == rep
    for module in loaded.model.modules():
        if type(module).__name__ in ("DOConv2d", "ConvNXC"):
            assert module.forward_func.__name__ in ("eval_forward", "forward_eval")


@pytest.mark.parametrize("arch", ["FIGSR", "SMoSR"])
@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (2, 2), (7, 1), (6, 9)])
def test_tiny_and_odd_inputs(arch, shape):
    loaded = make_loader().load_from_state_dict(make_model(arch).state_dict()).eval()
    result = loaded(torch.rand(1, 3, *shape))
    assert result.shape == (1, 3, shape[0] * 2, shape[1] * 2)
    assert torch.isfinite(result).all()


@pytest.mark.parametrize("arch", ["FIGSR", "SMoSR"])
@pytest.mark.parametrize("change", ["short", "version", "method", "scale", "width"])
def test_bad_metadata_rejected_before_upstream_can_replace_it(arch, change):
    state = make_model(arch).state_dict()
    key = "upscale.MetaUpsample" if arch == "FIGSR" else "upsampler.MetaUpsample"
    meta = state[key].clone()
    if change == "short":
        meta = meta[:3]
    else:
        index, value = {
            "version": (0, 123),
            "method": (1, 4),
            "scale": (2, 4),
            "width": (3, 99),
        }[change]
        meta[index] = value
    state[key] = meta
    with pytest.raises(ValueError, match=arch):
        make_loader().load_from_state_dict(state)


@pytest.mark.parametrize("arch", ["FIGSR", "SMoSR"])
def test_strict_weights_and_consecutive_blocks(arch):
    state = make_model(arch).state_dict()
    prefix = "gfisr_body_half_2" if arch == "FIGSR" else "blocks_2"
    key = next(k for k in state if k.startswith(prefix + ".0."))
    state[key.replace(".0.", ".9.", 1)] = state.pop(key)
    with pytest.raises(ValueError, match="consecutive"):
        make_loader().load_from_state_dict(state)
    state = make_model(arch).state_dict()
    del state["in_to_dim.bias" if arch == "FIGSR" else "short.bias"]
    with pytest.raises(RuntimeError, match="Missing key"):
        make_loader().load_from_state_dict(state)


def test_registration_preserves_existing_and_does_not_claim_gfisrv2():
    registry = spandrel.ArchRegistry()
    native = adapter.FIGSRArch()
    registry.add(spandrel.ArchSupport.from_architecture(native))
    adapter.register_enhancr(registry)
    adapter.register_enhancr(registry)
    assert len(list(registry)) == 2
    assert (
        next(s for s in registry if s.architecture.id == "FIGSR").architecture is native
    )
    gfisrv2 = importlib.import_module(
        "nodes.impl.pytorch.architecture.gfisrv2.arch"
    ).GFISRV2(dim=8, n_blocks=2)
    for support in registry:
        assert not support.architecture.detect(gfisrv2.state_dict())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for FP16")
@pytest.mark.parametrize("arch", ["FIGSR", "SMoSR"])
def test_fp16_and_fft_dtype(arch):
    loaded = (
        make_loader()
        .load_from_state_dict(make_model(arch).state_dict())
        .eval()
        .cuda()
        .half()
    )
    dtypes = []
    handles = [
        module.register_forward_pre_hook(
            lambda _module, inputs: dtypes.append(inputs[0].dtype)
        )
        for module in loaded.model.modules()
        if type(module).__name__ in ("CustomRfft2Wrap", "CustomIrfft2Wrap")
    ]
    try:
        result = loaded(torch.rand(1, 3, 9, 13, device="cuda", dtype=torch.float16))
        assert result.dtype == torch.float16
        assert torch.isfinite(result).all()
        if arch == "FIGSR":
            assert dtypes and set(dtypes) == {torch.float32}
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize(
    ("filename", "arch"),
    [
        ("2x_enhancr_da_figsr.safetensors", "FIGSR"),
        ("2x_enhancr_da_smosr_v1.safetensors", "SMoSR"),
    ],
)
@pytest.mark.parametrize("half", [False, True])
def test_local_checkpoints_through_chainner_nodes(filename, arch, half):
    model_dir = os.environ.get("CHAINNER_GFISRV2_MODEL_DIR")
    if not model_dir or not torch.cuda.is_available():
        pytest.skip("Set CHAINNER_GFISRV2_MODEL_DIR and use CUDA for real-weight tests")
    model_root = Path(model_dir)

    import numpy as np

    from api import NodeContext, SettingsParser
    from nodes.impl.upscale.auto_split_tiles import NO_TILING, TileSize
    from packages.chaiNNer_pytorch.pytorch.io.load_model import load_model_node
    from packages.chaiNNer_pytorch.pytorch.processing.upscale_image import upscale
    from packages.chaiNNer_pytorch.settings import PyTorchSettings

    class Context(NodeContext):
        @property
        def aborted(self) -> bool:
            return False

        @property
        def paused(self) -> bool:
            return False

        @property
        def settings(self) -> SettingsParser:
            return SettingsParser(
                {"use_cpu": False, "use_fp16": half, "gpu_index": "0"}
            )

        @property
        def storage_dir(self) -> Path:
            return model_root

        def set_progress(self, progress):
            pass

        def add_cleanup(self, fn, after="chain"):
            pass

    loaded, _, _ = load_model_node(Context(), model_root / filename)
    assert loaded.architecture.id == arch
    assert loaded.scale == 2
    assert loaded.dtype == (torch.float16 if half else torch.float32)
    settings = PyTorchSettings(
        use_cpu=False, use_fp16=half, gpu_index=0, budget_limit=0
    )
    image = np.random.default_rng(42).random((51, 67, 3), dtype=np.float32)
    for tile in (NO_TILING, TileSize(48)):
        result = upscale(image, loaded, tile, settings, None)
        assert result.shape == (102, 134, 3)
        assert np.isfinite(result).all()
        assert result.min() >= 0 and result.max() <= 1
