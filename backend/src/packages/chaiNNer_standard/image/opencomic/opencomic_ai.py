from __future__ import annotations

from pathlib import Path

import numpy as np

from api import NodeContext
from nodes.groups import Condition, if_enum_group, if_group
from nodes.impl.opencomic.pipeline import CUSTOM, DESCREEN, MODELS, OFF, steps_for
from nodes.impl.opencomic.runtime import process_image
from nodes.properties.inputs import (
    BoolInput,
    DropDownInput,
    FileInput,
    ImageInput,
    NumberInput,
)
from nodes.properties.outputs import ImageOutput

from .. import opencomic_group


def model_input(stage: str, label: str, default: str):
    return DropDownInput(
        "string",
        label,
        [
            {"option": "Off / 关闭", "value": OFF},
            *[
                {"option": model["name"], "value": key}
                for key, model in MODELS[stage].items()
            ],
            {"option": "Custom NCNN / 自定义模型", "value": CUSTOM},
        ],
        default_value=default,
    )


def custom_param(label: str):
    return (
        FileInput(label, "param", [".param"])
        .make_optional()
        .with_docs(
            "Select the .param file. Its matching .bin must have the same name and be in the same directory."
        )
    )


@opencomic_group.register(
    schema_id="chainner:image:opencomic_ai",
    name="OpenComic AI",
    description="Run OpenComic artifact removal, descreening and upscaling in reader order on Vulkan. Each stage can be switched off. Model downloads are verified and reused.",
    icon="BsImages",
    inputs=[
        ImageInput(channels=[1, 3, 4]).with_id(0),
        model_input("artifact-removal", "Artifact Removal / 去伪影", OFF).with_id(1),
        if_enum_group(1, CUSTOM)(custom_param("Artifact Model").with_id(2)),
        model_input("descreen", "Descreening / 去网点", DESCREEN).with_id(3),
        if_enum_group(3, CUSTOM)(custom_param("Descreen Model").with_id(4)),
        if_group(~Condition.enum(3, OFF))(
            BoolInput("Keep Big Halftones / 保留大网点").with_id(5),
            if_group(Condition.bool(5, True))(
                NumberInput(
                    "Minimum Halftone Size",
                    default=3,
                    min=0,
                    max=100,
                    precision=2,
                    unit="%",
                ).with_id(6)
            ),
        ),
        model_input("upscale", "Upscaling / 超分", OFF).with_id(7),
        if_enum_group(7, CUSTOM)(
            custom_param("Upscale Model").with_id(8),
            NumberInput("Custom Model Native Scale", default=2, min=1, max=16).with_id(
                9
            ),
        ),
        if_group(~Condition.enum(7, OFF))(
            BoolInput("Auto Scale / 自动倍率", default=True).with_id(10),
            if_group(Condition.bool(10, False))(
                NumberInput("Scale", min=1, max=8, default=2).with_id(11)
            ),
            NumberInput("Max Input Megapixels", default=2, min=0, precision=2)
            .with_id(12)
            .with_docs(
                "Larger inputs skip only the upscale stage. Set 0 for unlimited. Auto Scale uses a 2 MP reference when unlimited."
            ),
            if_enum_group(
                7, [k for k, m in MODELS["upscale"].items() if m.get("noise")]
            )(
                NumberInput("Denoise Level", min=-1, max=3, default=0)
                .with_id(13)
                .with_docs(
                    "Real-CUGAN and Waifu2x use the nearest level supported by the selected model."
                )
            ),
        ),
        BoolInput("Advanced / 高级设置", default=False).with_id(14),
        if_group(Condition.bool(14, True))(
            NumberInput("Tile Size (0 = Auto)", default=0, min=0, max=8192).with_id(15),
            BoolInput("Test-Time Augmentation", default=False).with_id(16),
            BoolInput("Force FP32", default=False).with_id(17),
        ),
    ],
    outputs=[
        ImageOutput(
            image_type="Image { channels: Input0.channels }", assume_normalized=True
        )
    ],
    node_context=True,
)
def opencomic_ai_node(
    context: NodeContext,
    image: np.ndarray,
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
    advanced: bool,
    tile: int,
    tta: bool,
    fp32: bool,
) -> np.ndarray:
    h, w = image.shape[:2]
    steps, factor = steps_for(
        w,
        h,
        artifact,
        artifact_param,
        descreen,
        descreen_param,
        keep,
        min_size,
        upscale,
        upscale_param,
        native,
        auto,
        scale,
        max_mp,
        noise,
        tile if advanced else 0,
        advanced and tta,
        advanced and fp32,
    )
    return process_image(context, image, steps, factor)
