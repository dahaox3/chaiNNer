"""Real loopback HTTP, cache and executor lifecycle regression tests."""

from __future__ import annotations

import asyncio
import importlib
import socket
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest
import requests

from api import ExecutionOptions, NodeId, SettingsParser, registry
from chain.cache import OutputCache
from chain.chain import FunctionNode
from chain.json import parse_json
from chain.service import service_fingerprint
from events import EventQueue
from nodes.impl.comicread import (
    BridgeError,
    ImageCache,
    ImageService,
    decode_image,
    encode_image,
    return_image,
)
from process import (
    ExecutionId,
    Executor,
    RegularOutput,
    _ExecutorNodeContext,
    _IterationTimer,
)
from progress_controller import Aborted, ProgressController

TOKEN = "comicread-test-token-0123456789"


def image_bytes(value=0.4):
    return encode_image(np.full((9, 13, 3), value, np.float32))


def context(tmp_path):
    return _ExecutorNodeContext(
        ProgressController(),
        SettingsParser({}),
        tmp_path,
        execution_fingerprint="test-pipeline",
    )


@pytest.fixture
def service(tmp_path):
    instance = ImageService(
        0, TOKEN, tmp_path / "cache", 1024**2, "test-pipeline", context(tmp_path)
    )
    yield instance
    instance.close()


def api(service, route, method="GET", **kwargs):
    with requests.Session() as session:
        session.trust_env = False
        return session.request(
            method,
            f"http://127.0.0.1:{service.port}/v1{route}",
            headers={"Authorization": f"Bearer {TOKEN}"},
            timeout=3,
            **kwargs,
        )


def upload(service, data=None, identifier=None, priority=0):
    return api(
        service,
        "/jobs",
        "POST",
        files={
            "image": (
                "page.png",
                data if data is not None else image_bytes(),
                "image/png",
            )
        },
        data={"request_id": identifier or uuid.uuid4().hex, "priority": priority},
    )


def test_http_auth_validation_and_roundtrip(service):
    with requests.Session() as session:
        session.trust_env = False
        assert (
            session.get(
                f"http://127.0.0.1:{service.port}/v1/status", timeout=2
            ).status_code
            == 401
        )
    assert api(service, "/status").json()["protocol"] == 1
    assert upload(service, b"not an image").status_code == 400
    assert not service.works
    response = upload(service)
    assert response.status_code == 200
    job = response.json()
    assert api(service, f"/jobs/{job['id']}/result").status_code == 409
    image, request_id = service.next_image()
    return_image(np.repeat(np.repeat(image, 2, axis=0), 2, axis=1), request_id)
    result = api(service, f"/jobs/{job['id']}/result")
    assert result.headers["Content-Type"] == "image/png"
    assert decode_image(result.content).shape == (18, 26, 3)
    assert upload(service).json()["state"] == "ready"


def test_alias_cancellation_priority_and_idempotency(service):
    identifier = uuid.uuid4().hex
    assert upload(service, identifier=identifier, priority=5).status_code == 200
    assert upload(service, identifier=identifier).json()["id"] == identifier
    alias = upload(service, priority=3).json()["id"]
    assert len(service.works) == 1
    assert api(service, f"/jobs/{identifier}", "DELETE").status_code == 200
    assert upload(service, identifier=identifier).status_code == 409
    current = upload(service, image_bytes(0.8), priority=0).json()["id"]
    assert (
        api(
            service, f"/jobs/{alias}/priority", "POST", json={"priority": 0}
        ).status_code
        == 200
    )
    image, request_id = service.next_image()
    assert image.mean() == pytest.approx(0.4, abs=0.01)
    return_image(image, request_id)
    assert api(service, f"/jobs/{alias}").json()["state"] == "ready"
    assert api(service, f"/jobs/{current}").json()["state"] == "queued"
    # Cancellation before a delayed upload must also prevent orphan GPU work.
    late_id = uuid.uuid4().hex
    assert api(service, f"/jobs/{late_id}", "DELETE").status_code == 200
    assert upload(service, image_bytes(0.9), late_id).status_code == 409


def test_queue_bound_and_clear(service):
    for i in range(8):
        assert upload(service, image_bytes(i / 10)).status_code == 200
    assert upload(service, image_bytes(0.99)).status_code == 429
    assert api(service, "/cache", "DELETE").status_code == 409
    for job in list(service.jobs):
        api(service, f"/jobs/{job}", "DELETE")
    assert api(service, "/cache", "DELETE").status_code == 200


def test_disk_cache_restarts_and_profile_invalidation(service, tmp_path):
    upload(service)
    image, request_id = service.next_image()
    return_image(image, request_id)
    cache_path = service.cache.directory
    service.close()
    second = ImageService(
        0, TOKEN, cache_path, 1024**2, "test-pipeline", context(tmp_path)
    )
    try:
        assert upload(second).json()["cached"] is True
    finally:
        second.close()
    third = ImageService(
        0, TOKEN, cache_path, 1024**2, "different-model-or-scale", context(tmp_path)
    )
    try:
        assert upload(third).json()["state"] == "queued"
    finally:
        third.close()


def test_cache_lru_and_clear_only_own_files(tmp_path):
    cache = ImageCache(tmp_path, 10)
    unrelated = tmp_path / "my-picture.png"
    unrelated.write_bytes(b"preserve")
    cache.put("a" * 64, b"aaaa")
    cache.put("b" * 64, b"bbbb")
    assert cache.get("a" * 64) == b"aaaa"
    cache.put("c" * 64, b"cccc")
    assert cache.get("a" * 64) is not None
    assert cache.get("b" * 64) is None
    cache.clear()
    assert unrelated.read_bytes() == b"preserve"


def test_alpha_and_color_roundtrip():
    image = np.array([[[0.1, 0.2, 0.8, 0.4]]], np.float32)
    np.testing.assert_allclose(decode_image(encode_image(image)), image, atol=1 / 255)
    with pytest.raises(BridgeError):
        decode_image(b"broken")


def test_stream_timing_is_bounded():
    timer = _IterationTimer(ProgressController(), streaming=True)
    for _ in range(1000):
        timer.add()
    assert timer.iterations == 1000
    assert len(timer.times) == 100


@pytest.fixture
def service_chain(tmp_path, monkeypatch):
    receiver = importlib.import_module(
        "packages.chaiNNer_standard.image.io.receive_image"
    )
    sender = importlib.import_module("packages.chaiNNer_standard.image.io.return_image")
    for data in receiver.io_group.nodes:
        monkeypatch.setitem(registry.nodes, data.schema_id, (data, receiver.io_group))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    chain = parse_json(
        [
            {
                "id": "receive",
                "schemaId": "chainner:image:receive_image",
                "nodeType": "generator",
                "parent": None,
                "inputs": [
                    {"type": "value", "value": value}
                    for value in (port, TOKEN, str(tmp_path / "cache"), 1)
                ],
            },
            {
                "id": "return",
                "schemaId": "chainner:image:return_image",
                "nodeType": "regularNode",
                "parent": None,
                "inputs": [
                    {"type": "edge", "id": "receive", "index": index}
                    for index in (0, 1)
                ],
            },
        ]
    )
    return chain, port, sender


async def wait_until_listening(port):
    for _ in range(50):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return
        except OSError:
            await asyncio.sleep(0.02)
    raise AssertionError("Service did not start")


@pytest.mark.asyncio
async def test_executor_wait_pause_stop_and_port_release(
    service_chain, tmp_path, monkeypatch
):
    chain, port, _ = service_chain
    receiver = importlib.import_module(
        "packages.chaiNNer_standard.image.io.receive_image"
    )
    factory = Mock(wraps=receiver.ImageService)
    monkeypatch.setattr(receiver, "ImageService", factory)
    pool = ThreadPoolExecutor(max_workers=2)
    executor = Executor(
        ExecutionId("comicread-test"),
        chain,
        False,
        ExecutionOptions({}),
        asyncio.get_running_loop(),
        EventQueue(),
        pool,
        tmp_path,
        parent_cache=OutputCache(static_data={NodeId("return"): RegularOutput([])}),
    )
    task = asyncio.create_task(executor.run())
    try:
        await wait_until_listening(port)
        executor.pause()

        # Pausing must not hold the service lock or block status requests.
        class Address:
            pass

        address = Address()
        address.port = port
        response = await asyncio.to_thread(api, address, "/status")
        assert response.json()["paused"] is True
        executor.resume()
        job = await asyncio.to_thread(upload, address)
        for _ in range(50):
            state = await asyncio.to_thread(api, address, f"/jobs/{job.json()['id']}")
            if state.json()["state"] == "ready":
                break
            await asyncio.sleep(0.02)
        assert state.json()["state"] == "ready"
        assert factory.call_count == 1
        executor.kill()
        with pytest.raises(Aborted):
            await asyncio.wait_for(task, 2)
        replacement = ImageService(
            port, TOKEN, tmp_path / "other", 0, "test", context(tmp_path)
        )
        replacement.close()
    finally:
        executor.kill()
        if not task.done():
            await asyncio.gather(task, return_exceptions=True)
        pool.shutdown(wait=True)


@pytest.mark.asyncio
async def test_downstream_failure_closes_listener(service_chain, tmp_path, monkeypatch):
    chain, port, sender = service_chain

    def fail(_image, _request_id):
        raise RuntimeError("simulated inference failure")

    monkeypatch.setattr(sender, "return_image", fail)
    with ThreadPoolExecutor(max_workers=2) as pool:
        executor = Executor(
            ExecutionId("comicread-error"),
            chain,
            False,
            ExecutionOptions({}),
            asyncio.get_running_loop(),
            EventQueue(),
            pool,
            tmp_path,
        )
        task = asyncio.create_task(executor.run())
        try:
            await wait_until_listening(port)

            class Address:
                pass

            address = Address()
            address.port = port
            await asyncio.to_thread(upload, address)
            with pytest.raises(Exception, match="simulated inference failure"):
                await asyncio.wait_for(task, 3)
            with (
                pytest.raises(OSError),
                socket.create_connection(("127.0.0.1", port), timeout=0.2),
            ):
                pass
        finally:
            executor.kill()
            if not task.done():
                await asyncio.gather(task, return_exceptions=True)


def test_workflow_identity_and_missing_return(service_chain):
    chain, _, _ = service_chain
    first = service_fingerprint(chain, ExecutionOptions({}))
    assert first == service_fingerprint(chain, ExecutionOptions({}))
    chain.inputs.inputs["receive"][3] = 2
    assert first != service_fingerprint(chain, ExecutionOptions({}))
    chain.remove_node("return")
    with pytest.raises(ValueError, match="exactly one"):
        service_fingerprint(chain, ExecutionOptions({}))


def test_model_file_and_precision_invalidate_identity(
    service_chain, tmp_path, monkeypatch
):
    from nodes.properties.inputs import PthFileInput

    chain, _, sender = service_chain
    model = tmp_path / "model.pth"
    model.write_bytes(b"weights-v1")
    data = replace(
        chain.nodes["return"].data,
        schema_id="test:comicread_file",
        inputs=[PthFileInput().with_id(0)],
        side_effects=False,
    )
    monkeypatch.setitem(registry.nodes, data.schema_id, (data, sender.io_group))
    node = FunctionNode(NodeId("file"), data.schema_id)
    chain.add_node(node)
    chain.inputs.set(node.id, data.inputs[0].id, str(model))
    original = service_fingerprint(chain, ExecutionOptions({}))
    assert original != service_fingerprint(
        chain, ExecutionOptions({"chaiNNer_standard": {"precision": "fp16"}})
    )
    model.write_bytes(b"different-weights-v2")
    assert original != service_fingerprint(chain, ExecutionOptions({}))


def test_port_conflict_does_not_close_existing_listener(service, tmp_path):
    with pytest.raises(OSError):
        ImageService(
            service.port,
            TOKEN,
            tmp_path / "second-cache",
            0,
            "other",
            context(tmp_path),
        )
    assert api(service, "/status").status_code == 200
