# Bundled AP-10K animal pose weights

`rtmpose_ap10k.onnx` is the unchanged `end2end.onnx` member of this official OpenMMLab release:

https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-ap10k_pt-aic-coco_210e-256x256-7a041aa1_20230206.zip

- Model: RTMPose-m, AP-10K, 210 epochs, 17 animal keypoints.
- ONNX SHA-256: `1cfd1c86e0d9e5d5f95178bcd95ee9a4e8386a624cd3c57519f27ff58cac7f28`
- Original ZIP SHA-256: `2d75445331cf2f21d6e164430f96ffa765cd874872965ae1736932dda03987f0`
- Network input: float32 `(batch,3,256,256)`, RGB normalized NCHW.
- Outputs: `simcc_x`, `simcc_y`, each `(batch,17,512)`; original AP-10K order.
- Coordinate split ratio: 2.0. Confidence is the minimum of the two raw axis maxima.

The accompanying `deploy.json`, `detail.json`, and `pipeline.json` are upstream provenance metadata. Some legacy entries say 192 × 256; the actual graph and AP-10K training configuration use 256 × 256. This CLI uses the latter. Do not edit the binary or substitute a human model with the same number of keypoints.

The application contains its own small inference wrapper and does not need MMPose, MMCV, MMDetection, PyTorch, or rtmlib installed. Model attribution and upstream links are in `THIRD_PARTY_NOTICES.md`. The application's MIT license excludes third-party weights and metadata.

## YOLOX-Tiny dog detector

`yolox_tiny.onnx` is the unchanged official YOLOX-Tiny ONNX model from:

https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx

- SHA-256: `427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7`
- Input: float32 `(1,3,416,416)`, BGR values 0–255, top-left letterbox filled with 114; no ImageNet normalization.
- Output: `(1,3549,85)`, grid/stride decoding with strides 8/16/32. The selected class is COCO dog index 16, not the human detector class.
- Dog confidence: objectness × dog-class score, threshold 0.30; class-specific suppression IoU 0.45.
- License/provenance: official [YOLOX project](https://github.com/Megvii-BaseDetection/YOLOX), Apache 2.0; upstream license saved as `YOLOX-LICENSE.txt`.

This detector supplies current-frame whole-dog boxes, not anatomical keypoints. The AP-10K pose model remains responsible for landmarks. Both use the requested verified CUDA device by default. Explicit `--device cpu` uses OpenCV DNN. The `--bbox X Y WIDTH HEIGHT` mode skips the detector and uses a fixed whole-dog crop. See the root README for CLI usage.
