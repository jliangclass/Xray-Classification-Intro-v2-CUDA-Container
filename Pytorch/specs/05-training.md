# Specification: Training

## Loss
- Use BCEWithLogits with masking for labels == -1.

## Optimizers and Schedulers
- `adamw`, `adam`, or `sgd`
- Weight decay and momentum supported.
- Schedulers supported: `reduce_lr_on_plateau`, `cosine_annealing`, or `none`. Warmup epochs supported for `reduce_lr_on_plateau`.

## Precision
- Mixed precision is handled automatically using `torch.amp.autocast`. Uses `bfloat16` if Ampere+ GPU or HPU is detected, otherwise `float16`.

## Loop
- Standard epoch loop without tqdm progress bar (uses logging).
- Log every batch for first 20 batches, then every 10th batch (for epochs <= 2) or every 50th batch (for epochs > 2).
- Log format includes: rank, dataset name, optimizer, epoch, global step, batch index, batch timing, data timing, learning rates, loss, elapsed time, ETA.
- All INFO level logs are saved to `training.log` file under `outputs/{dataset_name}/{run_name}/` directory.
- Save `checkpoints/last_model.pt` at end of training under `outputs/{dataset_name}/{run_name}/checkpoints/` directory.
- Save `checkpoints/best_model.pt` when monitored metric improves under `outputs/{dataset_name}/{run_name}/checkpoints/` directory.

## Evaluation Augmentation Rules
- Train transform: Resize to `image_size`, Random Rotation (7 deg), ToTensor. (No horizontal flip).
- Val/Test Transform: 
  - Remove 10-crop and center crop; only use "no crop".
  - For 256x256 model architecture, directly resize to 224x224 and then pass on transformation pipeline.
  - For models accepting 512x512 (with 512x512 image size base), only resize to 512x512 and then pass to the model.
  - Disable horizontal flipping in all cases during evaluation.

## Early Stopping
- Early stopping based on `patience` (default: 10 evals).
- Monitors the validation metric specified by `monitor` (default: `mean_auroc`).
- Stops training if the monitored metric does not improve for `patience` consecutive evaluation steps.
- Improvement is determined by `monitor_mode`: `max` (higher is better) or `min` (lower is better).
- When early stopping triggers, logs the best metric value and the epoch at which it was achieved.
- Training always saves `checkpoints/last_model.pt` even if early stopping occurs.

## Container Notes (Apptainer)
- The training entrypoint (`src/train_v1_classic.py`) can be run inside the container using `./cuda-apptainer.sh exec <cmd>` and containerized SLURM wrappers `./apptainer-run-*.sh`.
