"""Local Spandrel adapters for the pinned FIGSR and SMoSR implementations."""

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

from .architecture.figsr.arch import FIGSR
from .architecture.smosr.arch import SMoSR


def _metadata(state: StateDict, key: str, length: int, name: str) -> list[int]:
    tensor = state[key]
    if tensor.ndim != 1 or tensor.numel() != length:
        raise ValueError(f"{name} requires {length} upsampler metadata values.")
    values = tensor.tolist()
    if any(value != int(value) for value in values):
        raise ValueError(f"{name} metadata must contain integers.")
    return [int(value) for value in values]


def _sequence_length(state: StateDict, prefix: str, name: str) -> int:
    indices = sorted(
        {int(key.split(".")[1]) for key in state if key.startswith(prefix + ".")}
    )
    if not indices or indices != list(range(len(indices))):
        raise ValueError(f"{name} {prefix} indices must be consecutive from zero.")
    return len(indices)


class FIGSRArch(Architecture[FIGSR]):
    def __init__(self) -> None:
        super().__init__(
            id="FIGSR",
            name="FIGSR",
            detect=lambda state: all(
                key in state
                for key in (
                    "in_to_dim.weight",
                    "gfisr_body_half.0.norm.scale",
                    "gfisr_body_half.0.conv.fu.fdc.weight",
                    "gfisr_body_half_2.0.conv.fu.fpe.weight",
                    "cat_to_dim.weight",
                    "upscale.MetaUpsample",
                )
            ),
        )

    def load(self, state_dict: StateDict) -> ImageModelDescriptor[FIGSR]:
        version, method, scale, meta_dim, out_channels, mid_dim, groups = _metadata(
            state_dict, "upscale.MetaUpsample", 7, "FIGSR"
        )
        if version != 3 or method != 1:
            raise ValueError("FIGSR requires UniUpsampleV3 pixelshuffledirect weights.")
        dim, in_channels = state_dict["in_to_dim.weight"].shape[:2]
        if (
            scale < 1
            or meta_dim != dim
            or in_channels != 3
            or out_channels != 3
            or mid_dim < 1
            or groups != 4
        ):
            raise ValueError("FIGSR has invalid or unsupported RGB upsampler metadata.")
        if state_dict["upscale.0.weight"].shape[:2] != (
            out_channels * scale**2,
            dim,
        ):
            raise ValueError("FIGSR metadata does not match its upsampling weights.")
        first = _sequence_length(state_dict, "gfisr_body_half", "FIGSR")
        second = _sequence_length(state_dict, "gfisr_body_half_2", "FIGSR") - 1
        if second not in (first, first + 1):
            raise ValueError("FIGSR block halves do not match the upstream topology.")
        blocks = first + second
        hidden = state_dict["gfisr_body_half.0.fc1.weight"].shape[0] // 2
        gc = state_dict["gfisr_body_half.0.conv.convh.weight"].shape[0]
        model = FIGSR(
            in_nc=in_channels,
            out_nc=out_channels,
            dim=dim,
            expansion_ratio=hidden / dim,
            scale=scale,
            mid_dim=mid_dim,
            n_blocks=blocks,
            gc=gc,
            square_kernel_size=state_dict["gfisr_body_half.0.conv.convhw.weight"].shape[
                2
            ],
            band_kernel_size=state_dict["gfisr_body_half.0.conv.convh.weight"].shape[2],
            upsampler="pixelshuffledirect",
        )
        return ImageModelDescriptor(
            model,
            state_dict.copy(),  # Upstream load_state_dict replaces the metadata entry.
            architecture=self,
            purpose="SR" if scale > 1 else "Restoration",
            tags=[f"{blocks}blocks", f"{dim}dim"],
            supports_half=True,
            supports_bfloat16=False,
            scale=scale,
            input_channels=in_channels,
            output_channels=out_channels,
            size_requirements=SizeRequirements(minimum=6),
            tiling=ModelTiling.DISCOURAGED,
        )


class SMoSRArch(Architecture[SMoSR]):
    def __init__(self) -> None:
        super().__init__(
            id="SMoSR",
            name="SMoSR",
            detect=lambda state: all(
                key in state
                for key in (
                    "short.weight",
                    "blocks_1.0.short.weight",
                    "blocks_1.0.body.0.eval_conv.weight",
                    "blocks_2.0.body.4.eval_conv.weight",
                    "end_block.1.eval_conv.weight",
                    "upsampler.MetaUpsample",
                )
            ),
        )

    def load(self, state_dict: StateDict) -> ImageModelDescriptor[SMoSR]:
        version, method, scale, meta_dim, out_channels, mid_dim, groups, rep = (
            _metadata(state_dict, "upsampler.MetaUpsample", 8, "SMoSR")
        )
        if version != 254 or method != 1:
            raise ValueError(
                "SMoSR requires UniUpsampleV4_light pixelshuffledirect weights."
            )
        dim, in_channels = state_dict["blocks_1.0.body.0.eval_conv.weight"].shape[:2]
        if (
            scale < 1
            or out_channels < 1
            or mid_dim < 1
            or rep not in (0, 1)
            or groups != 4
            or meta_dim != dim + in_channels * scale**2
        ):
            raise ValueError("SMoSR has invalid or unsupported upsampler metadata.")
        if state_dict["upsampler.0.eval_conv.weight"].shape[:2] != (
            out_channels * scale**2,
            meta_dim,
        ) or state_dict["short.weight"].shape[:2] != (
            in_channels * scale**2,
            in_channels,
        ):
            raise ValueError("SMoSR metadata does not match its upsampling weights.")
        blocks = _sequence_length(state_dict, "blocks_2", "SMoSR")
        model = SMoSR(
            in_ch=in_channels,
            out_ch=out_channels,
            dim=dim,
            scale=scale,
            rep=bool(rep),
            n_mb=blocks,
            upsampler="pixelshuffledirect",
            upsampler_mid_dim=mid_dim,
        )
        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="SR" if scale > 1 else "Restoration",
            tags=[f"{blocks}blocks", f"{dim}dim", "rep" if rep else "DOConv"],
            supports_half=True,
            supports_bfloat16=False,
            scale=scale,
            input_channels=in_channels,
            output_channels=out_channels,
            size_requirements=SizeRequirements(minimum=3),
        )


def register_enhancr(registry: ArchRegistry | None = None) -> None:
    """Register local adapters once, preserving future native implementations."""
    if registry is None:
        registry = MAIN_REGISTRY
    registry.add(
        ArchSupport.from_architecture(FIGSRArch()),
        ArchSupport.from_architecture(SMoSRArch()),
        ignore_duplicates=True,
    )
