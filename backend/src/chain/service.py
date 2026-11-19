"""Validation and stable cache identity for an explicitly enabled image service."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from api import ExecutionOptions, NodeId, registry

from .chain import Chain, GeneratorNode

RECEIVE = "chainner:image:receive_image"
RETURN = "chainner:image:return_image"
VERSION = "comicread-bridge-1"


def service_fingerprint(chain: Chain, options: ExecutionOptions) -> str:
    receivers = chain.nodes_with_schema_id(RECEIVE)
    if not receivers:
        return ""
    returns = chain.nodes_with_schema_id(RETURN)
    generators = [
        node for node in chain.nodes.values() if isinstance(node, GeneratorNode)
    ]
    if len(receivers) != 1 or len(returns) != 1 or len(generators) != 1:
        raise ValueError(
            "An image service requires exactly one Receive Image, one Return Image, and no other iterators."
        )
    receiver, sender = receivers[0], returns[0]
    request_edge = chain.edge_to(sender.id, sender.data.inputs[1].id)
    if (
        request_edge is None
        or request_edge.source.id != receiver.id
        or request_edge.source.output_id != receiver.data.outputs[1].id
    ):
        raise ValueError(
            "Connect Request ID directly from Receive Image to Return Image."
        )
    visited: set[str] = set()

    def uses_received_image(node_id: NodeId) -> bool:
        if node_id in visited:
            return False
        visited.add(node_id)
        for edge in chain.edges_to(node_id):
            if edge.source.id == receiver.id:
                if edge.source.output_id == receiver.data.outputs[0].id:
                    return True
            elif uses_received_image(edge.source.id):
                return True
        return False

    image_edge = chain.edge_to(sender.id, sender.data.inputs[0].id)
    if image_edge is None or not (
        (
            image_edge.source.id == receiver.id
            and image_edge.source.output_id == receiver.data.outputs[0].id
        )
        or (
            image_edge.source.id != receiver.id
            and uses_received_image(image_edge.source.id)
        )
    ):
        raise ValueError("Return Image must process the image from Receive Image.")

    nodes = []
    files = []
    settings = {}
    for node_id, node in sorted(chain.nodes.items()):
        values = chain.inputs.inputs.get(node_id, {})
        nodes.append(
            (
                node_id,
                node.schema_id,
                sorted(values.items()),
                sorted(
                    (e.source.id, e.source.output_id, e.target.input_id)
                    for e in chain.edges_to(node_id)
                ),
            )
        )
        is_opencomic = node.schema_id == "chainner:image:opencomic_ai"
        if is_opencomic:
            from nodes.impl.opencomic.pipeline import (
                ADVANCED_INPUT,
                CACHE_VERSION,
                TTA_INPUT,
            )

            # Older TTA-on results were produced without augmentation. Preserve
            # the existing disk cache for graphs that cannot enable TTA.
            uses_tta = all(
                values.get(i) or chain.edge_to(node_id, i) is not None
                for i in (ADVANCED_INPUT, TTA_INPUT)
            )
            version = (*CACHE_VERSION, "daemon-tta-1") if uses_tta else CACHE_VERSION
            nodes.append(("opencomic", version))
        for node_input in node.data.inputs:
            if node_input.kind == "file":
                value = values.get(node_input.id)
                if isinstance(value, str | Path):
                    path = Path(value).resolve()
                    paths = (
                        (path, path.with_suffix(".bin")) if is_opencomic else (path,)
                    )
                    for model_path in paths:
                        info = model_path.stat()
                        files.append((str(model_path), info.st_size, info.st_mtime_ns))
        package_id = registry.get_package(node.schema_id).id
        settings[package_id] = options.get_package_settings_json(package_id)
    data = json.dumps(
        [VERSION, nodes, files, settings],
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    return hashlib.sha256(data.encode()).hexdigest()
