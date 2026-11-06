# ComicReadScript image bridge

This fork adds two ordinary image nodes. In ComicReadScript, select the chaiNNer
upscale backend and configure the local URL and token. Translation remains a
separate service; translated pages can be fed through the same image workflow.

## Workflow

Connect `Receive Image → Upscale Image → Return Image`. Connect the receiver's
`Request ID` directly to the return node. Connect a model to Upscale Image as usual.
Receive Image needs a port (5004 by default), a random access token of at least
16 characters, a writable cache directory and a size limit in GiB (5 by default).
Zero disables disk caching. An image can pass through additional processing nodes.

Click Run once. The workflow keeps accepting images until Stop. Pause keeps the
listener open and queues requests. Stop while idle releases the listener promptly;
during inference, cancellation takes effect at the next cooperative checkpoint.
An inference or downstream node error stops the service and releases its resources.
Correct the workflow (for example, reduce tile size after an out-of-memory error)
and run it again. Restart after changing parameters or replacing model files.

Exactly one receiver, one return node and no other iterators are supported in a
service workflow. Ordinary finite batch workflows keep their existing behavior.
The service workflow occupies the main executor while serving.

## HTTP v1

All routes except OPTIONS require `Authorization: Bearer <token>`. The listener
binds only to `127.0.0.1`; it does not expose the internal chaiNNer /run API.
The bridge never accepts filesystem paths, model paths or remote image URLs.

| Route | Request / response |
| --- | --- |
| `GET /v1/status` | Protocol version, workflow fingerprint, paused state and queue length |
| `POST /v1/jobs` | multipart `image`, unique `request_id` (20–80 URL-safe characters), optional numeric `priority` (0 highest); returns `{id,state,cached,error}` |
| `GET /v1/jobs/{id}` | Same job state; states are queued, processing, ready, failed |
| `GET /v1/jobs/{id}/result` | PNG bytes; 409 until ready |
| `POST /v1/jobs/{id}/priority` | JSON `{priority: number}`; affects queued work |
| `DELETE /v1/jobs/{id}` | Cancels this subscriber, including uploads not yet received |
| `DELETE /v1/cache` | Clears managed PNG files when no queued or active work exists |

Responses are 200 on success; malformed uploads return 400, wrong tokens 401,
unknown/expired jobs 404, unavailable results or conflicts 409, excessive uploads
413, and a full queue 429. Bodies on failure are JSON `{error: string}`. Uploads
are limited to 64 MiB and decoded input to 40 million pixels. The queue holds up
to eight distinct images; matching content shares inference but has independent
subscriber IDs. One subscriber cancelling does not cancel another. An active GPU
operation is allowed to finish and populate the disk cache after cancellation.

Completed subscriptions should be DELETEd after downloading. Abandoned jobs expire
after ten minutes. Retained result memory is bounded, and the executor drains
preview tasks and caps stream timing history instead of accumulating them forever.

## Cache and maintenance

Cache keys combine uploaded bytes with a stable hash of the graph, literal inputs,
package execution settings and input-file paths, lengths and modification times.
The bridge implementation version is included. Increase `VERSION` in
`backend/src/chain/service.py` when inference/cache semantics change. Cache files
are SHA256-named PNGs, atomically written and evicted by last access. Clear only
removes owned files with this name pattern; unrelated files are preserved.

Keep `main` as the upstream mirror and merge it into `gfisrv2-support`. The bridge
is isolated from model implementations. The small generic execution extensions
are an opt-in `Generator.streaming` flag, a read-only
`NodeContext.execution_fingerprint` property, optional streaming progress data,
and reliable cleanup after every execution exit. Existing node APIs default to
their previous behavior. `api.Aborted` and the executor now share one exception.
Service runs initialize the receiver once and reload static workflow inputs rather
than reusing GUI preview caches, so a changed model file or precision is respected.

Run `python -m pytest backend/tests/test_comicread.py -q`, normal backend tests,
Ruff and frontend type/lint checks after merging. Optional CUDA checkpoint tests
use `CHAINNER_GFISRV2_MODEL_DIR`; no weights are committed or downloaded by CI.
See the [reader fork](https://github.com/dahaox3/ComicReadScript/tree/chainner-upscale).
