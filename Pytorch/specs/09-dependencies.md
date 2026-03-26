# Specification: Dependencies

> [!IMPORTANT]
> **PyTorch Version & GPU Compatibility:**
> When running on Tesla V100 GPUs (Compute Capability 7.0), you **must** use PyTorch 2.5.1 (or a compatible older version). Newer PyTorch versions (2.6+) from pip have dropped out-of-the-box support for Compute Capability 7.0 (`sm_70`). Running newer versions on a V100 will crash with `CUDA error: no kernel image is available for execution on the device`. To ensure cluster compatibility, our `requirements.txt` specifically pins `torch==2.5.1` and `torchvision==0.20.1` uniformly across our CUDA 12.4.1 pipeline.
> 
> *Note: This pinned version natively leverages CUDA 12.4 (`torch==2.5.1` default) and fully supports **A100** (Compute Capability 8.0), **RTX 8000 Quadro** (Compute Capability 7.5), as well as **Tesla V100** GPUs!*

Required:
- torch (Pinned to `==2.5.1` natively utilizing `12.4` for Tesla V100 / SM_70 compatibility)
- torchvision (Pinned to `==0.20.1` natively utilizing `12.4` for compatibility with torch 2.5.1)
- pandas
- numpy
- pyyaml
- tqdm
- pillow
- hydra-core
- omegaconf
- torchmetrics
- timm
- psutil

Optional:
- ipywidgets - for interactive Jupyter notebook widgets (required for visualize_training.ipynb)

## Container Notes (Apptainer)
- When using Apptainer, most runtime environment setup comes from `apptainer-cuda.def` (built by `./cuda-apptainer.sh build`).
- The container execution scripts (`cuda-apptainer.sh exec` and `apptainer-run-*.sh`) wrap the same Python entrypoints, so Hydra config/CLI behavior stays consistent.