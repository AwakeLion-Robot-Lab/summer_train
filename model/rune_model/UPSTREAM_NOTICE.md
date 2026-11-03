# RuneDetectionModel notice

The five-keypoint ONNX model and OpenVINO deployment approach in this project
are derived from https://github.com/SZURPVision/RuneDetectionModel, with specific reference to
upstream `infer/fivepoints_infer.cpp`.

The detector follows its configurable minimum-valid-keypoint policy (default 3),
valid-keypoint mean confidence ranking, and valid-keypoint mean-center NMS.
Per-point confidence and validity are preserved for downstream pose processing.

The debug overlay in `src/rune/debug_overlay.cpp` also follows that file's
five-keypoint color palette and black-background class/confidence label style,
with an additional four-point target outline. It is shared by the live runtime
and the offline `rune_detect` tool.

- Copyright: (c) 2024 RoboMaster RobotPilots
- License: MIT
- Upstream model: `model/model-0624.onnx` (master branch)
- Local model: `model/rune_model/model-0624.onnx`
- Model SHA-256: `e6e6bddf1f793ece1b9b351b8db061dbb026d32b8a6d71cb39a8fd21ce9cb917`
- License text: `model/rune_model/LICENSE` (upstream: `LICENSE`)

No model or deployment library from the non-public reference project is used.
