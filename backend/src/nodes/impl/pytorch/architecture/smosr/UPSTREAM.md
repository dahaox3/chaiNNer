# SMoSR upstream source

- Repository: https://github.com/rewaifu/resselt
- Commit: `4a0c2c163f0ba98b7c548221590bec27fe80a861` (Resselt 1.4.1).
- Original file: `resselt/archs/smosr/arch.py`.
- Original architecture: https://github.com/umzi2/SMoSR
- License: MIT. See `../RESSelt-LICENSE.md`.
- Local modifications: import `Self` from `typing_extensions` for Python 3.10.

The maintained Spandrel adapter is in `nodes/impl/pytorch/enhancr.py`.
It accepts UniUpsampleV4_light (version 254) pixelshuffledirect checkpoints,
with or without rep convolutions. Keep upstream `eval()` reparameterization:
it must run after loading the checkpoint, before converting to FP16.
Review source updates and run `backend/tests/test_enhancr.py` before refreshing.
Model weights are not included and retain their own licenses.
