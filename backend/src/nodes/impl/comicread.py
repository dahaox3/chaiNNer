"""Loopback image bridge for ComicReadScript. No model-specific inference here."""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import re
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from email.message import EmailMessage
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np
from PIL import Image, ImageOps

from api import NodeContext

PROTOCOL_VERSION = 1
MAX_UPLOAD = 64 * 1024**2
MAX_PIXELS = 40_000_000
MAX_RESULT = 256 * 1024**2
MAX_QUEUE = 8
MAX_JOBS = 128
JOB_TTL = 600
_SERVICES: dict[str, ImageService] = {}
_SERVICES_LOCK = threading.Lock()


class BridgeError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


class LoopbackServer(ThreadingHTTPServer):
    # Windows SO_REUSEADDR can admit two live listeners on the same port.
    allow_reuse_address = os.name != "nt"
    daemon_threads = True

    def server_bind(self):
        if os.name == "nt":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def decode_image(data: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.width * source.height > MAX_PIXELS:
                raise BridgeError("Input image exceeds 40 million pixels.", 413)
            oriented = ImageOps.exif_transpose(source)
            mode = (
                "RGBA"
                if "A" in oriented.getbands() or "transparency" in oriented.info
                else "RGB"
            )
            image = np.array(oriented.convert(mode))
        image = cv2.cvtColor(
            image, cv2.COLOR_RGBA2BGRA if mode == "RGBA" else cv2.COLOR_RGB2BGR
        )
        return image.astype(np.float32) / 255
    except BridgeError:
        raise
    except Exception as error:
        raise BridgeError("Invalid or unsupported image.") from error


def encode_image(image: np.ndarray) -> bytes:
    if not np.isfinite(image).all():
        raise ValueError("Image contains non-finite values.")
    success, encoded = cv2.imencode(
        ".png", np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
    )
    if not success or encoded.nbytes > MAX_RESULT:
        raise ValueError("Could not encode result, or PNG exceeds 256 MiB.")
    return encoded.tobytes()


class ImageCache:
    """Only owns SHA256-named PNG files directly inside its directory."""

    def __init__(self, directory: Path, limit: int):
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.limit = limit
        self.lock = threading.RLock()
        self.last_touch = max((p.stat().st_mtime_ns for p in self._files()), default=0)
        with self.lock:
            self._evict()

    def _touch(self, path: Path):
        # NTFS rounds timestamps; make consecutive accesses distinguishable.
        self.last_touch = max(time.time_ns(), self.last_touch + 1000)
        os.utime(path, ns=(self.last_touch, self.last_touch))

    def _files(self) -> list[Path]:
        return [
            p
            for p in self.directory.glob("*.png")
            if re.fullmatch(r"[a-f0-9]{64}\.png", p.name) and not p.is_symlink()
        ]

    def _evict(self):
        files = sorted(self._files(), key=lambda p: p.stat().st_mtime_ns)
        size = sum(p.stat().st_size for p in files)
        for path in files:
            if size <= self.limit:
                break
            size -= path.stat().st_size
            path.unlink(missing_ok=True)

    def get(self, key: str) -> bytes | None:
        with self.lock:
            path = self.directory / f"{key}.png"
            if not path.is_file() or path.is_symlink():
                return None
            try:
                data = path.read_bytes()
                self._touch(path)
                return data
            except FileNotFoundError:
                return None

    def put(self, key: str, data: bytes):
        if len(data) > self.limit:
            return
        with self.lock:
            temporary = self.directory / f"{uuid.uuid4().hex}.tmp"
            try:
                temporary.write_bytes(data)
                temporary.replace(self.directory / f"{key}.png")
                self._touch(self.directory / f"{key}.png")
                self._evict()
            finally:
                temporary.unlink(missing_ok=True)

    def clear(self):
        with self.lock:
            for path in self._files():
                path.unlink(missing_ok=True)


@dataclass
class Work:
    key: str
    image: np.ndarray | None
    sequence: int
    state: str = "queued"
    result: bytes | None = None
    error: str | None = None
    cached: bool = False


@dataclass
class Job:
    id: str
    work: Work
    priority: int
    cancelled: bool = False
    touched: float = field(default_factory=time.monotonic)


class ImageService:
    def __init__(
        self,
        port: int,
        token: str,
        directory: Path,
        limit: int,
        fingerprint: str,
        context: NodeContext,
    ):
        if len(token) < 16:
            raise ValueError("Access Token must contain at least 16 characters.")
        if not fingerprint:
            raise ValueError("Run the complete workflow to start the image service.")
        self.id = uuid.uuid4().hex
        self.token = token
        self.fingerprint = fingerprint
        self.context = context
        self.cache = ImageCache(directory, limit)
        self.condition = threading.Condition(threading.RLock())
        self.jobs: dict[str, Job] = {}
        self.cancelled: dict[str, float] = {}
        self.works: dict[str, Work] = {}
        self.closed = False
        self.sequence = 0
        self.active: Work | None = None
        self.slots = threading.BoundedSemaphore(12)
        self.server = LoopbackServer(("127.0.0.1", port), self._handler())
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True
        )
        with _SERVICES_LOCK:
            _SERVICES[self.id] = self
        self.thread.start()

    @property
    def port(self) -> int:
        return self.server.server_address[1]

    def prune(self):
        cutoff = time.monotonic() - JOB_TTL
        for identifier, job in list(self.jobs.items()):
            if job.cancelled or job.touched < cutoff:
                del self.jobs[identifier]
        # Browsers normally DELETE after downloading. Bound retained results even
        # when a client disappears without doing so (keep one oversized result).
        ready = sorted(
            {
                job.work.key: job.work
                for job in sorted(self.jobs.values(), key=lambda job: job.touched)
                if job.work.result is not None
            }.values(),
            key=lambda work: max(
                job.touched for job in self.jobs.values() if job.work is work
            ),
        )
        retained_bytes = sum(len(work.result or b"") for work in ready)
        for work in ready[:-1]:
            if retained_bytes <= 128 * 1024**2:
                break
            retained_bytes -= len(work.result or b"")
            self.jobs = {
                identifier: job
                for identifier, job in self.jobs.items()
                if job.work is not work
            }
        self.cancelled = {
            identifier: touched
            for identifier, touched in self.cancelled.items()
            if touched >= cutoff
        }
        while len(self.cancelled) > 1024:
            self.cancelled.pop(next(iter(self.cancelled)))
        retained = {job.work.key for job in self.jobs.values() if not job.cancelled}
        for key, work in list(self.works.items()):
            if key not in retained and work is not self.active:
                del self.works[key]

    def submit(self, data: bytes, identifier: str, priority: int) -> dict:
        if not re.fullmatch(r"[A-Za-z0-9_-]{20,80}", identifier):
            raise BridgeError(
                "request_id must be a unique opaque identifier (20-80 characters)."
            )
        key = hashlib.sha256(self.fingerprint.encode() + b"\0" + data).hexdigest()
        with self.condition:
            self.prune()
            if self.closed:
                raise BridgeError("Service stopped.", 503)
            if identifier in self.cancelled:
                raise BridgeError("Request already cancelled.", 409)
            if identifier in self.jobs:
                job = self.jobs[identifier]
                if job.work.key != key:
                    raise BridgeError(
                        "request_id already belongs to another image.", 409
                    )
                return self.job_state(job)
            if len(self.jobs) >= MAX_JOBS:
                raise BridgeError("Too many live requests. Cancel unused jobs.", 429)
            work = self.works.get(key)
            if work is None:
                cached = self.cache.get(key)
                if (
                    cached is None
                    and sum(w.state == "queued" for w in self.works.values())
                    >= MAX_QUEUE
                ):
                    raise BridgeError("Queue is full.", 429)
                image = None if cached is not None else decode_image(data)
                self.sequence += 1
                work = Work(
                    key,
                    image,
                    self.sequence,
                    state="ready" if cached is not None else "queued",
                    result=cached,
                    cached=cached is not None,
                )
                self.works[key] = work
            job = Job(identifier, work, max(0, priority))
            self.jobs[identifier] = job
            self.condition.notify_all()
            return self.job_state(job)

    def job_state(self, job: Job) -> dict:
        return {
            "id": job.id,
            "state": "cancelled" if job.cancelled else job.work.state,
            "cached": job.work.cached,
            "error": job.work.error,
        }

    def get_job(self, identifier: str) -> Job:
        job = self.jobs.get(identifier)
        if job is None:
            raise BridgeError("Job not found or expired.", 404)
        job.touched = time.monotonic()
        return job

    def next_image(self) -> tuple[np.ndarray, str]:
        if self.active is not None:
            raise RuntimeError("Previous image did not reach Return Image.")
        while not self.closed:
            self.context.suspend()
            with self.condition:
                if self.closed:
                    break
                self.prune()
                candidates = [
                    (job.priority, job.work.sequence, job)
                    for job in self.jobs.values()
                    if not job.cancelled and job.work.state == "queued"
                ]
                if candidates:
                    job = min(candidates, key=lambda value: value[:2])[2]
                    work = job.work
                    work.state = "processing"
                    self.active = work
                    assert work.image is not None
                    image, work.image = work.image, None
                    return image, f"{self.id}:{work.key}"
                self.condition.wait(0.1)
        raise RuntimeError("Image service stopped.")

    def complete(self, key: str, image: np.ndarray):
        data = encode_image(image)
        with self.condition:
            work = self.active
            if self.closed or work is None or work.key != key:
                raise ValueError("Request does not belong to the active image.")
            self.cache.put(key, data)
            work.result = data
            work.state = "ready"
            self.active = None
            self.condition.notify_all()

    def close(self):
        with self.condition:
            if self.closed:
                return
            self.closed = True
            for work in self.works.values():
                if work.state in ("queued", "processing"):
                    work.state, work.error, work.image = (
                        "failed",
                        "Workflow stopped or failed. Restart it in chaiNNer.",
                        None,
                    )
            self.condition.notify_all()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        with _SERVICES_LOCK:
            _SERVICES.pop(self.id, None)
        with self.condition:
            self.jobs.clear()
            self.works.clear()
            self.active = None

    def _handler(self):
        service = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object):
                # Never log tokens, image bodies, or signed source URLs.
                pass

            def setup(self):
                super().setup()
                self.connection.settimeout(10)

            def _reply(self, status: int, data: bytes | dict):
                is_image = isinstance(data, bytes)
                body = data if is_image else json.dumps(data).encode()
                self.send_response(status)
                self.send_header(
                    "Content-Type", "image/png" if is_image else "application/json"
                )
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header(
                    "Access-Control-Allow-Headers", "Authorization, Content-Type"
                )
                self.send_header(
                    "Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS"
                )
                self.send_header("Access-Control-Allow-Private-Network", "true")
                self.end_headers()
                self.wfile.write(body)

            def _body(self) -> bytes:
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                except ValueError as error:
                    raise BridgeError("Invalid content length.") from error
                if size <= 0 or size > MAX_UPLOAD:
                    raise BridgeError("Upload must be between 1 byte and 64 MiB.", 413)
                data = self.rfile.read(size)
                if len(data) != size:
                    raise BridgeError("Incomplete upload.")
                return data

            def _dispatch(self):
                if self.command == "OPTIONS":
                    self._reply(200, {})
                    return
                if not hmac.compare_digest(
                    self.headers.get("Authorization", ""), f"Bearer {service.token}"
                ):
                    raise BridgeError("Invalid access token.", 401)
                route = urlsplit(self.path).path
                if route == "/v1/status" and self.command == "GET":
                    with service.condition:
                        service.prune()
                        state = {
                            "protocol": PROTOCOL_VERSION,
                            "fingerprint": service.fingerprint,
                            "paused": service.context.paused,
                            "queued": sum(
                                w.state == "queued" for w in service.works.values()
                            ),
                            "cache_limit": service.cache.limit,
                        }
                    self._reply(200, state)
                    return
                if route == "/v1/cache" and self.command == "DELETE":
                    with service.condition:
                        if service.active or any(
                            w.state == "queued" for w in service.works.values()
                        ):
                            raise BridgeError(
                                "Wait for current tasks before clearing cache.", 409
                            )
                        service.cache.clear()
                        service.jobs.clear()
                        service.works.clear()
                    self._reply(200, {"cleared": True})
                    return
                if route == "/v1/jobs" and self.command == "POST":
                    content_type = self.headers.get("Content-Type", "")
                    if not content_type.startswith("multipart/form-data;"):
                        raise BridgeError("Expected multipart/form-data.")
                    message = BytesParser(policy=default).parsebytes(
                        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode()
                        + self._body()
                    )
                    if (
                        not isinstance(message, EmailMessage)
                        or not message.is_multipart()
                    ):
                        raise BridgeError("Invalid multipart body.")
                    parts = {
                        part.get_param(
                            "name", header="content-disposition"
                        ): part.get_payload(decode=True)
                        for part in message.iter_parts()
                    }
                    data = parts.get("image")
                    if not isinstance(data, bytes) or not data:
                        raise BridgeError("Missing image file.")
                    identifier = parts.get(
                        "request_id", uuid.uuid4().hex.encode()
                    ).decode()
                    priority = int(parts.get("priority", b"0"))
                    self._reply(200, service.submit(data, identifier, priority))
                    return
                match = re.fullmatch(
                    r"/v1/jobs/([A-Za-z0-9_-]{20,80})(/result|/priority)?", route
                )
                if match:
                    if not match[2] and self.command == "DELETE":
                        with service.condition:
                            service.cancelled[match[1]] = time.monotonic()
                            if match[1] in service.jobs:
                                service.jobs[match[1]].cancelled = True
                            service.prune()
                        self._reply(
                            200, {"id": match[1], "state": "cancelled", "cached": False}
                        )
                        return
                    with service.condition:
                        job = service.get_job(match[1])
                        if match[2] == "/result" and self.command == "GET":
                            if (
                                job.cancelled
                                or job.work.state != "ready"
                                or job.work.result is None
                            ):
                                raise BridgeError("Result is not available.", 409)
                            result: bytes | dict = job.work.result
                        elif (not match[2] and self.command in ("GET", "PATCH")) or (
                            match[2] == "/priority" and self.command == "POST"
                        ):
                            if self.command in ("PATCH", "POST"):
                                job.priority = max(
                                    0, int(json.loads(self._body())["priority"])
                                )
                            result = service.job_state(job)
                        else:
                            raise BridgeError("Unknown endpoint.", 404)
                    self._reply(200, result)
                    return
                raise BridgeError("Unknown endpoint.", 404)

            def do_GET(self):
                if not service.slots.acquire(blocking=False):
                    self._reply(503, {"error": "Too many HTTP requests."})
                    return
                try:
                    self._dispatch()
                except BridgeError as error:
                    self._reply(error.status, {"error": str(error)})
                except (ValueError, KeyError, UnicodeError):
                    self._reply(400, {"error": "Invalid request."})
                except (OSError, TimeoutError):
                    self.close_connection = True
                finally:
                    service.slots.release()

            do_POST = do_GET  # noqa: N815 - stdlib HTTP method names
            do_PATCH = do_GET  # noqa: N815
            do_DELETE = do_GET  # noqa: N815
            do_OPTIONS = do_GET  # noqa: N815

        return Handler


def return_image(image: np.ndarray, request_id: str):
    service_id, separator, key = request_id.partition(":")
    with _SERVICES_LOCK:
        service = _SERVICES.get(service_id)
    if not separator or service is None:
        raise ValueError(
            "Unknown image request. Connect Request ID from Receive Image."
        )
    service.complete(key, image)
