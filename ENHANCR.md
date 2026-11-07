# FIGSR and SMoSR support

This fork adds native PyTorch loading for these tested checkpoints:

- `2x_enhancr_da_figsr.safetensors`: FIGSR, 2x RGB.
- `2x_enhancr_da_smosr_v1.safetensors`: SMoSR rep, 2x RGB.

Use the existing Load Model and Upscale Image nodes, including the ComicRead
Receive/Return workflow. No ONNX conversion or extra runtime package is needed.
FP32 and CUDA FP16 are supported. FIGSR's FFT operations stay in FP32.
BF16 is not advertised. FP16 is approximate; FP32 is preferable when exact
agreement with the author's FP32 inference matters, especially smooth gradients.
FIGSR uses global Fourier features, so tile size can affect the result.

## Maintenance

`backend/src/nodes/impl/pytorch/enhancr.py` contains both Spandrel adapters.
The source is pinned to Resselt 1.4.1 commit
`4a0c2c163f0ba98b7c548221590bec27fe80a861` under the separate `architecture/figsr`
and `architecture/smosr` directories, with licenses and provenance notes.
Only SMoSR's `typing.Self` import is adjusted for Python 3.10 compatibility.
The installed Spandrel and Python environment are not patched.

The adapters validate versioned pixelshuffledirect metadata, tensor dimensions,
and block topology before strict weight loading. Other upsamplers are explicitly
rejected. SMoSR retains upstream eval-time convolution fusion; load weights and
call eval before converting to FP16. Registration preserves future native support
with matching architecture IDs.

Keep `main` tracking upstream and merge upstream changes into `gfisrv2-support`.
Review the single registration call in Load Model after a merge. When updating
vendored architecture code, update provenance and compare real-weight outputs
against an isolated original Resselt release.

## Validation

CPU CI uses synthetic safetensors checkpoints: variable scales, rep/non-rep,
odd/tiny dimensions, strict loading, invalid metadata, and registration isolation.
CUDA additionally verifies FP16 and FIGSR FFT dtypes. Both Python 3.10 and 3.12
are covered by `.github/workflows/gfisrv2.yml`.

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests/test_enhancr.py -q
$env:CHAINNER_GFISRV2_MODEL_DIR='F:\Reline_Models_2026-11-06\models'
.\.venv\Scripts\python.exe -m pytest backend/tests -q
```

The opt-in real-weight tests load through chaiNNer's actual nodes and exercise
whole-image and tiled inference for FP32 and FP16. Model weights are not included.
