# GFISRv2 support and fork maintenance

This fork adds native PyTorch support for GFISRv2 checkpoints using
UniUpsampleV3 `pixelshuffledirect`, including:

- `4x_wtp_MangaScale_GfisrV2.pth`
- `2x_umzi_DigitalArt_GFISRV2.pth`

Use the existing **PyTorch Load Model → Upscale Image** nodes. In the PyTorch
settings, **Use FP16 Mode** selects FP16 when enabled and FP32 when disabled.
FFT operations retain the upstream implementation's explicit FP32 conversions.
BF16 is not advertised as supported: local MangaScale tests found substantially
larger numerical deviations than FP16. The model's existing weights are used
directly; weights and their licenses are not redistributed in this repository.

The adapter reads scale, channels, depth and width from checkpoint metadata and
weights, not the filename. It validates the metadata before the upstream model
can replace it and uses strict state-dictionary loading. Other upsamplers and
pixel-unshuffle variants are not claimed as supported by this adapter.

## Keep the delta small

- `backend/src/nodes/impl/pytorch/gfisrv2.py`: the Spandrel adapter and registration.
- `backend/src/nodes/impl/pytorch/architecture/gfisrv2/`: an unmodified, pinned copy
  of the author's implementation, its MIT license and upstream commit reference.
- `backend/src/packages/chaiNNer_pytorch/pytorch/io/load_model.py`: one import and
  one registration call; the original loader/inference logic is preserved.
- `backend/tests/test_gfisrv2.py` and `.github/workflows/gfisrv2.yml`: synthetic
  CPU CI plus optional tests using local weights and the actual chaiNNer nodes.
- `pyrightconfig.json`: ignore diagnostics from the unmodified vendored source;
  the adapter remains checked. Existing Ruff rules already exclude this vendor path.

No forked Spandrel dependency, startup downloads, precision setting changes or
parallel inference implementation are introduced. Registration is idempotent
and retains an existing `GFISRV2` architecture if Spandrel later provides one.
If a future Spandrel release uses another ID, review detection and remove this
fallback explicitly after validating the native implementation.

## Branches and upstream updates

Keep `main` as a clean mirror of `chaiNNer-org/chaiNNer:main`. Keep this feature
on `gfisrv2-support`. Merge upstream changes into the feature branch and test;
do not force-push over the custom branch with the GitHub “Sync fork” action.

First-time remote setup, if needed:

```powershell
git remote add upstream https://github.com/chaiNNer-org/chaiNNer.git
```

For each update, start with a clean working tree:

```powershell
git fetch upstream
git switch main
git merge --ff-only upstream/main
git push origin main
git switch gfisrv2-support
git merge main
```

Resolve any conflicts, then run the checks below before pushing. Most changes
are new files, so frontend updates should usually merge cleanly. Conflicts are
still possible when upstream changes the model loader, Spandrel APIs, dependency
versions or the build system. A successful Git merge alone does not establish
runtime compatibility. Push only after tests pass:

```powershell
git push origin gfisrv2-support
```

GitHub's **GFISRv2 compatibility** workflow runs automatically on that branch
and can also be run manually. It uses synthetic models and never needs access
to the local model directory. Review the rest of upstream's CI for larger merges.

## Local checks

Use a Python environment with chaiNNer's PyTorch dependencies and the development
requirements installed. Keep it separate from the running app's environment.

```powershell
python -m pytest backend/tests/test_gfisrv2.py -v
python -m ruff check backend
python -m ruff format backend --check
python -m pyright backend/src/nodes/impl/pytorch/gfisrv2.py
```

Opt in to the private-checkpoint tests on a CUDA machine:

```powershell
$env:CHAINNER_GFISRV2_MODEL_DIR = 'F:\Reline_Models_2026-11-06\models'
python -m pytest backend/tests/test_gfisrv2.py -v
```

These tests exercise both local models in FP32 and FP16 through the actual
**Load Model** and upscale functions, including odd image sizes and tiled output.
The tests verify loading, shapes, range and finite values; they do not claim tiled
output is identical to a full-image pass or establish real manga visual quality.

## Run or package the fork

Follow the normal chaiNNer build requirements (see the root README). With the
backend dependencies installed and a compatible Node.js version:

```powershell
npm ci
npm start
# Or create a Windows portable ZIP:
npm run make-win-zip -- --no-sign
```

Use a separate directory for fork builds. Updating an official chaiNNer install
does not install this fork's changes. Avoid treating edits inside a versioned
`AppData/Local/chaiNNer/app-*` directory as the maintained source of truth.

## Retire the fork patch when upstream catches up

The best long-term outcome is native Spandrel support. Once it handles these
checkpoints and preserves FP32 FFT behavior in FP16 mode, run the local node
tests with it, remove the vendored adapter and its loader registration, and
retain regression coverage. This avoids maintaining a second architecture
implementation indefinitely.
# Additional architectures

This branch also supports FIGSR and SMoSR. See [ENHANCR.md](ENHANCR.md)
for supported checkpoint variants, precision and maintenance instructions.
