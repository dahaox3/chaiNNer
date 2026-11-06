from __future__ import annotations

from pathlib import Path

import numpy as np

from api import Generator, IteratorOutputInfo, NodeContext
from nodes.impl.comicread import ImageService
from nodes.properties.inputs import DirectoryInput, NumberInput, TextInput
from nodes.properties.outputs import ImageOutput, TextOutput

from .. import io_group


@io_group.register(
    schema_id="chainner:image:receive_image",
    name="Receive Image",
    description="Starts a local ComicReadScript image service. Connect Image through processing nodes to Return Image, and connect Request ID directly. Run once to keep serving; Stop closes the server. Changing the workflow requires restarting it. Only one service iterator is allowed per workflow.",
    icon="BsImages",
    inputs=[
        NumberInput("Port", default=5004, min=1024, max=65535),
        TextInput("Access Token", min_length=16, has_handle=False).with_docs(
            "Use a random token of at least 16 characters. Enter the same token in ComicReadScript."
        ),
        DirectoryInput("Cache Directory", must_exist=False),
        NumberInput("Cache Limit (GiB)", default=5, min=0, max=100),
    ],
    outputs=[ImageOutput(), TextOutput("Request ID")],
    iterator_outputs=IteratorOutputInfo(outputs=[0, 1], length_type="uint"),
    kind="generator",
    side_effects=True,
    node_context=True,
)
def receive_image_node(
    context: NodeContext, port: int, token: str, cache_directory: Path, cache_limit: int
) -> Generator[tuple[np.ndarray, str]]:
    service = ImageService(
        port,
        token,
        cache_directory,
        cache_limit * 1024**3,
        context.execution_fingerprint,
        context,
    )
    context.add_cleanup(service.close, after="chain")

    def supplier():
        while True:
            yield service.next_image()

    return Generator.from_iter(supplier, expected_length=0).with_streaming()
