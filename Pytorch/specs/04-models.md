# Specification: Models

## Backbones (timm)
- Pre-configured support via `configs/model/*.yaml`:
  - `resnet50`
  - `convnext_base`
  - `swin_base_patch4_window7_224`
- Any valid `timm` model name can be supplied in the config (`model.backbone`).

## Head replacement
- Models are instantiated using `timm.create_model(model_name, pretrained=..., num_classes=...)` in `build_model()`.
- The `num_classes` is derived dynamically from the number of `labels` configured in the dataset config.
- A runtime check ensures the model's output `num_classes` matches the requested `num_classes`.

## Pretrained weights
- Use timm pretrained weights when `model.pretrained: true`.

## Image Sizer and normalization
- The model configuration dictates `image_size`.
- `normalize` handles image normalization (e.g., `normalize: imagenet` uses ImageNet mean/std; `normalize: none` skips it).

## Container Notes (Apptainer)
Model configuration and `timm` backbone selection behave the same inside the Apptainer image; use `cuda-apptainer.sh exec <cmd>` or `apptainer-run-*.sh` to run it containerized.
