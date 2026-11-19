"""Run the official OpenComic worker over private pipes, scoped to one workflow."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from weakref import WeakKeyDictionary

import cv2
import numpy as np
import psutil

from api import NodeContext
from nodes.impl.image_utils import normalize, to_uint8
from nodes.utils.utils import get_h_w_c

from .pipeline import prepare_models

SCRIPT = Path(__file__).with_name("worker.cjs")


class Worker:
    def __init__(self, runtime: Path, models: Path):
        node = runtime / ("node.exe" if os.name == "nt" else "node")
        if not node.is_file():
            raise ValueError(
                f"Missing OpenComic runtime at {runtime}. See OPENCOMIC.md."
            )
        self.proc = subprocess.Popen(
            [str(node), str(SCRIPT), str(runtime), str(models)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.events: queue.Queue[str | None] = queue.Queue()
        self.lock = threading.Lock()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        assert self.proc.stdout
        try:
            for line in self.proc.stdout:
                self.events.put(line)
        finally:
            self.events.put(None)

    def run(self, context: NodeContext, source: Path, dest: Path, steps: list[dict]):
        with self.lock:
            assert self.proc.stdin
            self.proc.stdin.write(
                json.dumps({"source": str(source), "dest": str(dest), "steps": steps})
                + "\n"
            )
            self.proc.stdin.flush()
            last_response = time.monotonic()
            while True:
                context.check_aborted()
                try:
                    line = self.events.get(timeout=0.1)
                except queue.Empty:
                    if time.monotonic() - last_response > 900:
                        raise TimeoutError(
                            "OpenComic stopped responding for 15 minutes."
                        ) from None
                    continue
                if line is None:
                    raise RuntimeError(
                        f"The OpenComic worker exited unexpectedly (exit {self.proc.poll()})."
                    )
                last_response = time.monotonic()
                event = json.loads(line)
                if "error" in event:
                    raise RuntimeError(f"OpenComic: {event['error']}")
                if event.get("done"):
                    break
                if "progress" in event:
                    context.set_progress(0.2 + 0.79 * max(0, min(1, event["progress"])))

    def close(self):
        # Capture only children of this specific worker, before its PID can exit.
        if self.proc.stdout and self.proc.stdout.closed:
            return
        try:
            parent = psutil.Process(self.proc.pid)
            children = (
                parent.children(recursive=True) if self.proc.poll() is None else []
            )
        except psutil.NoSuchProcess:
            children = []
        if self.proc.stdin and not self.proc.stdin.closed:
            try:
                self.proc.stdin.close()
            except BrokenPipeError:
                pass
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(children, timeout=2)
        self.reader.join(timeout=2)
        if self.proc.stdout:
            self.proc.stdout.close()


_workers: WeakKeyDictionary[NodeContext, Worker] = WeakKeyDictionary()
_workers_lock = threading.Lock()


def process_image(
    context: NodeContext, image: np.ndarray, steps: list[dict], scale: int
):
    if not steps:
        return image
    runtime = Path(
        os.environ.get(
            "CHAINNER_OPENCOMIC_RUNTIME", SCRIPT.parents[4] / "opencomic-runtime"
        )
    ).resolve()
    base = runtime.parent.parent
    models = Path(
        os.environ.get("CHAINNER_OPENCOMIC_MODELS", base / "models" / "opencomic")
    ).resolve()
    scratch = base / "runtime-temp" / "opencomic"
    models.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    prepare_models(models, steps, context)
    h, w, channels = get_h_w_c(image)
    with TemporaryDirectory(dir=scratch) as folder:
        source = Path(folder) / "input.png"
        dest = Path(folder) / "output.png"
        # The native reader models operate on 8-bit RGB. Keep the graph's alpha
        # channel at its original precision and resize it only when needed.
        rgb = to_uint8(image[:, :, :3] if channels > 1 else image, normalized=True)
        if channels == 1:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
        ok, encoded = cv2.imencode(".png", rgb, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        if not ok:
            raise ValueError("Could not encode the OpenComic input PNG.")
        encoded.tofile(source)
        context.suspend()
        with _workers_lock:
            worker = _workers.get(context)
            if worker is None or worker.proc.poll() is not None:
                worker = Worker(runtime, models)
                _workers[context] = worker
                context.add_cleanup(worker.close)
        try:
            worker.run(context, source, dest, steps)
        except BaseException:
            # Also reap the native process before releasing this image's files.
            worker.close()
            raise
        result = cv2.imdecode(np.fromfile(dest, dtype=np.uint8), cv2.IMREAD_COLOR)
        if result is None or result.shape[:2] != (h * scale, w * scale):
            actual = result.shape[:2] if result is not None else None
            raise ValueError(
                f"OpenComic returned size {actual} (H,W), expected {(h * scale, w * scale)}. Check the model scale."
            )
        if channels == 1:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        result = normalize(result)
        if channels == 4:
            alpha = image[:, :, 3]
            if scale != 1:
                alpha = cv2.resize(
                    alpha, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR
                )
            result = np.dstack((result, alpha))
        context.set_progress(1)
        return result
