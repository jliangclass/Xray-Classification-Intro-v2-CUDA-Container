# Specification: Hydra Configuration

## Root config (`configs/config.yaml`)
- Must include defaults:
  - dataset: `chestxray14`
  - model: `resnet50`
  - optimizer: `adamw`
  - scheduler: `reduce_lr_on_plateau`
  - train: `default`
- `dataset_dir`: Centralized root directory definition for all datasets. Override globally using `dataset_dir=...` or `+dataset_dir=...` on the CLI.
- Hydra run directory: `outputs/${dataset.name}` (organized by dataset name)
- Use `hydra.job.chdir: true` so outputs are local to run dir.
- Hydra's internal logging is disabled via `override hydra/hydra_logging: none` and `override hydra/job_logging: none` to prevent automatic log file creation (like `train_v2_lightning.log`) - training scripts create their own `training.log` file in the `{run_name}` directory.
- `run_name` is constructed in `config.yaml` using Hydra resolvers: `${dataset.name}_${model.backbone}_${train.optimizer}_${format_lr:${train.lr}}_${now:%Y%m%d}_${now:%H%M%S}`
  - Uses custom `format_lr` resolver (registered in training scripts) to format learning rate (replaces `.` with `_`)
  - Uses `${now:%Y%m%d}` for date and `${now:%H%M%S}` for time
- The `.hydra` directory is automatically placed in `outputs/{dataset_name}/{run_name}/.hydra` via `hydra.output_subdir: ${run_name}/.hydra` since `run_name` is known at config load time.

## Container Notes (Apptainer)
Hydra CLI overrides passed to the training entrypoint (e.g. `dataset=... model=... train.batch_size=...`) work the same whether you run directly on the host or inside the container via:
- `./cuda-apptainer.sh exec <cmd>`
- `./apptainer-run-*.sh` (SLURM wrappers that invoke `cuda-apptainer.sh exec`).

## Dataset config fields
- `name`: dataset name
- `dataset_root`: By default inherits from the global `dataset_dir`. Can be overridden specifically for the active dataset using `dataset.dataset_root=...` on the CLI.
- `train_ann`, `val_ann`, `test_ann`: paths to annotation CSV files
- `train_image_root`, `val_image_root`, `test_image_root`: root folders
- `image_path_key`: column name in CSV (default `Path`)
- `image_path_append`: optional suffix
- `labels`: list of label names
- `num_classes`: int

## Model config fields
- `backbone`: full timm model name, e.g. `resnet50`, `densenet121`, `efficientnet_b0`, `convnext_base`, `swin_base_patch4_window7_224` (config file: `swin_base.yaml`)
- `pretrained`: bool
- `image_size`: int
- `normalize`: `imagenet` or `none`

## Train config fields
- `batch_size`, `num_workers`, `epochs`
- `lr`, `weight_decay`, `momentum` (if applicable)
- `eval_every`
- `seed`
- `save_predictions`, `save_best_model`
- `monitor`, `monitor_mode`, `patience` (default: 10, early stopping patience in epochs)

## Optimizer config (`configs/optimizer/*.yaml`)
- Settings specific to the chosen optimizer algorithm (e.g., Adam, AdamW, SGD).

## Scheduler config (`configs/scheduler/*.yaml`)
- Settings for dynamic learning rate adjustments (e.g., ReduceLROnPlateau, CosineAnnealingLR, none).