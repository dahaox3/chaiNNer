from __future__ import annotations

import numpy as np

from nodes.impl.comicread import return_image
from nodes.properties.inputs import ImageInput, TextInput

from .. import io_group


@io_group.register(
    schema_id="chainner:image:return_image",
    name="Return Image",
    description="Returns a processed PNG to the ComicReadScript request. Connect Request ID directly from Receive Image. A processing error stops the service; fix the workflow and run again.",
    icon="BsImages",
    inputs=[ImageInput(channels=[1, 3, 4]), TextInput("Request ID")],
    outputs=[],
    side_effects=True,
)
def return_image_node(image: np.ndarray, request_id: str) -> None:
    return_image(image, request_id)
