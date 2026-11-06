"""GFISRv2 compatibility tests; no downloaded weights are needed for CPU CI."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
spandrel = pytest.importorskip("spandrel")
adapter = importlib.import_module("nodes.impl.pytorch.gfisrv2")
GFISRV2 = importlib.import_module(
    "nodes.impl.pytorch.architecture.gfisrv2.arch"
).GFISRV2


@pytest.fixture(scope="module", autouse=True)
def limit_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def make_model(scale=4, channels=3):
    torch.manual_seed(42)
    return GFISRV2(
        dim=8, n_blocks=2, scale=scale, in_nc=channels, out_nc=channels
    ).eval()


def make_loader():
    registry = spandrel.ArchRegistry()
    adapter.register_gfisrv2(registry)
    return spandrel.ModelLoader(registry=registry)


@pytest.mark.parametrize("scale", [1, 2, 4])
@pytest.mark.parametrize("channels", [1, 3])
def test_checkpoint_metadata_overrides_filename(tmp_path, scale, channels):
    original = make_model(scale, channels)
    # Loading may not rely on 4x/RGB/default-size assumptions or the file's name.
    path = tmp_path / "wrong-name-8x-RGB.pth"
    torch.save({"params_ema": original.state_dict()}, path)
    loaded = make_loader().load_from_file(path)
    assert isinstance(loaded, spandrel.ImageModelDescriptor)
    assert loaded.architecture.id == "GFISRV2"
    assert loaded.scale == scale
    assert loaded.input_channels == channels
    assert loaded.output_channels == channels
    assert loaded.supports_half
    assert not loaded.supports_bfloat16
    assert loaded.tiling == spandrel.ModelTiling.DISCOURAGED
    image = torch.rand(1, channels, 9, 13)
    with torch.inference_mode():
        expected = original(image).clamp(0, 1)
    actual = loaded(image)
    assert actual.shape == (1, channels, 9 * scale, 13 * scale)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (2, 2), (7, 1)])
def test_tiny_inputs_are_padded_and_cropped(shape):
    loaded = make_loader().load_from_state_dict(make_model().state_dict())
    image = torch.rand(1, 3, *shape)
    output = loaded(image)
    assert output.shape == (1, 3, shape[0] * 4, shape[1] * 4)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    "metadata",
    [
        [3, 1, 4],  # truncated
        [2, 1, 4, 8, 3, 32, 4],  # unknown version
        [3, 4, 4, 8, 3, 32, 4],  # unsupported upsampler
        [3, 1, 0, 8, 3, 32, 4],  # invalid scale
        [3, 1, 4, 16, 3, 32, 4],  # wrong feature width
        [3, 1, 2, 8, 3, 32, 4],  # scale inconsistent with weights
    ],
)
def test_invalid_metadata_is_not_silently_replaced(metadata):
    state = make_model().state_dict()
    state["upscale.MetaUpsample"] = torch.tensor(metadata, dtype=torch.uint8)
    with pytest.raises(ValueError, match="GFISRv2"):
        make_loader().load_from_state_dict(state)


def test_missing_weight_is_rejected_by_strict_loading():
    state = make_model().state_dict()
    del state["gfisr_body.1.fc2.weight"]
    with pytest.raises(RuntimeError, match="Missing key"):
        make_loader().load_from_state_dict(state)


def test_gap_in_block_indices_is_rejected():
    state = make_model().state_dict()
    state["gfisr_body.2.gamma"] = state.pop("gfisr_body.1.gamma")
    with pytest.raises(ValueError, match="consecutive"):
        make_loader().load_from_state_dict(state)


def test_unrelated_architecture_is_not_claimed():
    assert not adapter.GFISRv2Arch().detect({"conv.weight": torch.zeros(3, 3, 3, 3)})


def test_registration_is_idempotent_and_preserves_existing_support():
    registry = spandrel.ArchRegistry()
    native = adapter.GFISRv2Arch()
    registry.add(spandrel.ArchSupport.from_architecture(native))
    adapter.register_gfisrv2(registry)
    adapter.register_gfisrv2(registry)
    supports = list(registry)
    assert len(supports) == 1
    assert supports[0].architecture is native


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for FP16")
def test_fp16_keeps_fft_in_fp32():
    loaded = make_loader().load_from_state_dict(make_model().state_dict()).cuda().half()
    fft_dtypes = []
    handles = []
    for module in loaded.model.modules():
        if type(module).__name__ in ("CustomRfft2Wrap", "CustomIrfft2Wrap"):
            handles.append(
                module.register_forward_pre_hook(
                    lambda _module, inputs: fft_dtypes.append(inputs[0].dtype)
                )
            )
    try:
        image = torch.rand(1, 3, 9, 13, device="cuda", dtype=torch.float16)
        output = loaded(image)
        assert output.dtype == torch.float16
        assert torch.isfinite(output).all()
        assert fft_dtypes and set(fft_dtypes) == {torch.float32}
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize(
    ("filename", "scale"),
    [
        ("4x_wtp_MangaScale_GfisrV2.pth", 4),
        ("2x_umzi_DigitalArt_GFISRV2.pth", 2),
    ],
)
@pytest.mark.parametrize("half", [False, True])
def test_local_checkpoints_through_chainner_nodes(filename, scale, half):
    model_dir = os.environ.get("CHAINNER_GFISRV2_MODEL_DIR")
    if not model_dir:
        pytest.skip("Set CHAINNER_GFISRV2_MODEL_DIR for opt-in local node integration")
    if not torch.cuda.is_available():
        pytest.skip("Local node integration requires CUDA")
    assert model_dir is not None
    model_root = Path(model_dir)

    # Optional runtime packages are only imported when a real checkpoint is supplied.
    import numpy as np
    from spandrel import ImageModelDescriptor

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

    loaded, _directory, _name = load_model_node(Context(), model_root / filename)
    assert isinstance(loaded, ImageModelDescriptor)
    assert loaded.scale == scale
    assert loaded.dtype == (torch.float16 if half else torch.float32)
    options = PyTorchSettings(use_cpu=False, use_fp16=half, gpu_index=0, budget_limit=0)
    image = np.random.default_rng(42).random((51, 67, 3), dtype=np.float32)
    for tile in (NO_TILING, TileSize(48)):
        result = upscale(image, loaded, tile, options, None)
        assert result.shape == (51 * scale, 67 * scale, 3)
        assert np.isfinite(result).all()
        assert result.min() >= 0 and result.max() <= 1
