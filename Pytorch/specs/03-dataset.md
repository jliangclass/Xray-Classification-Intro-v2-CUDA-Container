# Specification: Dataset

## Dataset Size Statistics

| Dataset | Total Images | Full Size | 1024x1024 | 256x256 |
|---------|--------------|-----------|-----------|-----------|
| NIH Chest X-ray 14 | 112,123 | ~44 GB | ~42 GB | ~3.3 GB |
| CheXpert v1.0 | 187,273 | ~447 GB | ~83 GB | ~5.7 GB |
| MIMIC-CXR JPEG | 377,031 | ~575 GB | ~100 GB | ~12.8 GB |

## CSV format
- Must contain a column with image paths (default `Path`).
- Must contain one column per label with values:
  - `1` positive
  - `0` negative
  - `-1` or NaN = ignore (not used in loss/metrics)

## Dataset class (`src/dataset.py`)
- Reads CSV using pandas.
- Converts paths to absolute if needed using `image_root` and `image_path_append`.
- Loads image with PIL, converts to RGB.
- Applies torchvision transforms.
- Returns 5-tuple: `(image_tensor, label_tensor, full_image_path, original_path, original_key)`
  - `image_tensor`: transformed image tensor
  - `label_tensor`: label tensor with values 1 (positive), 0 (negative), -1 (ignore)
  - `full_image_path`: absolute path after `image_root` and `image_path_append` are applied
  - `original_path`: original path value from CSV (before `image_root` is applied)
  - `original_key`: value from `__key__` column in CSV (if present, otherwise `None`)
- Stores original paths and keys before path modification for use in prediction saving.
- Validation: optional random path checks (fail fast).

## Visualization (`src/visualize_training.ipynb`)
- Jupyter notebook for visualizing training dataset samples
- Features:
  - Loads dataset configuration from `configs/dataset/*.yaml` files
  - Interactive dropdown widget to select dataset (ipywidgets)
  - Displays random samples in 3-column grid layout
  - Shows ground truth labels (positive, uncertain, no finding)
  - Computes and displays dataset statistics (label distribution, samples with/without findings)
- Usage:
  - Open `src/visualize_training.ipynb` in Jupyter
  - Select dataset from dropdown widget
  - Click "Visualize Selected Dataset" button to load and visualize

## Container Notes (Apptainer)
- The container script `cuda-apptainer.sh` binds common host dataset paths (`/scratch`, `/data`, `/mnt/local/dataset`) into the container.
- You can provide extra binds via the host `EXTRA_MOUNTS` environment variable when using `cuda-apptainer.sh` / the container wrappers.
