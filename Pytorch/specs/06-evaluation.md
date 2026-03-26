# Specification: Evaluation

## Implementation
- Evaluation functions are in `src/evaluate.py`:
  - `evaluate()`: Main evaluation function that runs model inference and computes metrics
  - `calculate_subset_metrics()`: Calculate metrics for label subsets
  - `multilabel_metrics()`: Calculate multilabel metrics from probabilities and targets

## Metrics
- Use torchmetrics:
  - `MultilabelAUROC`
- Call `update()` on each batch and `compute()` at end.
- Use `ignore_index=-1` to skip uncertain labels.
- Report `mean_auroc` and `per_label_auroc` for all labels.
- Save validation and test AUROC metrics to `auroc_list_mean.csv` and `auroc_list_per_label.csv` using `save_val_auroc()`. This tracks `val_mean_auroc`, `test_mean_auroc`, and per-label metrics across evaluation steps with a `date_time` column.

## Label Subsets (test_labels)
- If `test_labels` is defined in the dataset config, calculate metrics for specified label subsets.
- `test_labels` is a dictionary mapping subset names to lists of label names.
- Common subsets: `top5`, `top12`, `top6`, `top14`, etc.
- For each subset, calculate:
  - `{split}_mean_auroc_{subset_name}` (e.g., `val_mean_auroc_top5`, `test_mean_auroc_top12`)
- Metrics are calculated for both validation and test sets.
- Example config:
  ```yaml
  test_labels:
    top5:
      - Atelectasis
      - Cardiomegaly
      - Consolidation
      - Edema
      - Pleural Effusion
    top12:
      - Atelectasis
      - Cardiomegaly
      - Consolidation
      - Edema
      - Enlarged Cardiomediastinum
      - Fracture
      - Lung Lesion
      - Lung Opacity
      - Pleural Effusion
      - Pleural Other
      - Pneumonia
      - Pneumothorax
  ```

## Predictions
- Save both validation and test predictions at each evaluation step.
- Saved in split-specific subdirectories: `val/` and `test/`
- Filename format:
  - `val/preds_epochXXX_iterYYYYYY.csv`
  - `test/preds_epochXXX_iterYYYYYY.csv`
- CSV columns (predictions only, no ground truth):
  - `Path`: original image path from dataset CSV (before image_root is applied)
  - `__key__`: original `__key__` column value from dataset CSV (if present in original dataset)
  - `pred_<label>`: prediction probability for each label (rounded to 4 decimal places)
- Implementation: `save_predictions()` function in `src/evaluate.py`

## Container Notes (Apptainer)
- Evaluation code (`src/evaluate.py`) is run the same way whether invoked directly on the host or via `./cuda-apptainer.sh exec <cmd>` / `./apptainer-run-*.sh`.
