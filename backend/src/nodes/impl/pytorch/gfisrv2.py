"""Local Spandrel adapter for GFISRv2; no changes to the installed Spandrel package."""

from __future__ import annotations

from spandrel import (
    MAIN_REGISTRY,
    Architecture,
    ArchRegistry,
    ArchSupport,
    ImageModelDescriptor,
    ModelTiling,
    SizeRequirements,
    StateDict,
)

from .architecture.gfisrv2.arch import GFISRV2


def _detect(state_dict: StateDict) -> bool:
    return all(
        key in state_dict
        for key in (
            "in_to_dim.weight",
            "gfisr_body.0.gamma",
            "gfisr_body.0.norm.scale",
            "gfisr_body.0.conv.pconv.fdc.weight",
            "gfisr_body.0.conv.pconv.fpe.weight",
            "upscale.MetaUpsample",
        )
    )


class GFISRv2Arch(Architecture[GFISRV2]):
    def __init__(self) -> None:
        super().__init__(id="GFISRV2", name="GFISRv2", detect=_detect)

    def load(self, state_dict: StateDict) -> ImageModelDescriptor[GFISRV2]:
        # Metadata is authoritative: never infer scale/channels from a filename.
        meta = state_dict["upscale.MetaUpsample"]
        if meta.ndim != 1 or meta.numel() != 7:
            raise ValueError("GFISRv2 requires seven UniUpsampleV3 metadata values.")
        version, method, scale, meta_dim, out_channels, mid_dim, _groups = (
            int(value) for value in meta.tolist()
        )
        if version != 3 or method != 1:
            raise ValueError(
                "This GFISRv2 adapter supports UniUpsampleV3 "
                "pixelshuffledirect checkpoints only."
            )
        if scale < 1 or out_channels < 1:
            raise ValueError("GFISRv2 scale and output channels must be positive.")

        dim, in_channels = state_dict["in_to_dim.weight"].shape[:2]
        if meta_dim != dim:
            raise ValueError("GFISRv2 metadata does not match its feature channels.")
        if state_dict["upscale.0.weight"].shape[0] != out_channels * scale**2:
            raise ValueError("GFISRv2 metadata does not match its upsampling weights.")

        block_indices = sorted(
            int(key.split(".")[1])
            for key in state_dict
            if key.startswith("gfisr_body.") and key.endswith(".gamma")
        )
        if not block_indices or block_indices != list(range(len(block_indices))):
            raise ValueError("GFISRv2 block indices must be consecutive from zero.")
        hidden_channels = state_dict["gfisr_body.0.fc1.weight"].shape[0] // 2

        model = GFISRV2(
            in_nc=in_channels,
            out_nc=out_channels,
            dim=dim,
            expansion_ratio=hidden_channels / dim,
            scale=scale,
            mid_dim=mid_dim,
            n_blocks=len(block_indices),
            upsampler="pixelshuffledirect",
        )
        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="SR" if scale > 1 else "Restoration",
            tags=[f"{len(block_indices)}blocks", f"{dim}dim"],
            supports_half=True,
            # BF16 runs, but tests with the MangaScale weights show large deviations.
            supports_bfloat16=False,
            scale=scale,
            input_channels=in_channels,
            output_channels=out_channels,
            size_requirements=SizeRequirements(minimum=2),
            # Fourier features depend on the whole tile. Prefer the largest tile
            # that fits in VRAM; independent tiles are not numerically equivalent.
            tiling=ModelTiling.DISCOURAGED,
        )


def register_gfisrv2(registry: ArchRegistry | None = None) -> None:
    """Register once; retain native Spandrel support if it already has this ID."""
    if registry is None:
        registry = MAIN_REGISTRY
    registry.add(
        ArchSupport.from_architecture(GFISRv2Arch()),
        ignore_duplicates=True,
    )
