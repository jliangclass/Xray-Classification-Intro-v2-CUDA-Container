# Specification: Project Structure

```
Pytorch/
  configs/
    config.yaml
    dataset/
      chestxray14.yaml
      chexpert.yaml
      mimic.yaml
      vindr_cxr.yaml
    model/
      resnet50.yaml
      convnext_base.yaml
      swin_base.yaml
    train/
      default.yaml
  src/
    train_v1_classic.py
    dataset.py
    utils.py
    analyze_output.py
    visualize_training.ipynb
  requirements.txt
  specs/
```

## Rules
- `configs/config.yaml` defines Hydra defaults and run directory.
- Dataset/model/train configs are selectable by Hydra group name.
- `src/train_v1_classic.py` - Core PyTorch training script (entrypoint)

## Apptainer Files (Container Option)
- `cuda-apptainer.sh` builds and runs the Apptainer image (`apptainer-cuda.sif`) and provides `exec` for commands.
- `apptainer-run-*.sh` are SLURM wrappers that run the same training commands inside the container.
- `apptainer-cuda.def` defines the container environment.
