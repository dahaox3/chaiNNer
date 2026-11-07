# FIGSR upstream source

- Repository: https://github.com/rewaifu/resselt
- Commit: `4a0c2c163f0ba98b7c548221590bec27fe80a861` (Resselt 1.4.1).
- Original file: `resselt/archs/figsr/arch.py`, copied verbatim.
- Original architecture: https://github.com/enhancr/figsr
- License: MIT. See `../RESSelt-LICENSE.md`.
- Local modifications: none.

The maintained Spandrel adapter is in `nodes/impl/pytorch/enhancr.py`.
It currently accepts RGB UniUpsampleV3 pixelshuffledirect checkpoints.
FFT and inverse FFT stay in FP32 in upstream code even during FP16 inference.
Review source updates and run `backend/tests/test_enhancr.py` before refreshing.
Model weights are not included and retain their own licenses.
