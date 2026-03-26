# Specification: Run Scripts

## Host SLURM Scripts (`run-*.local.sh`, e.g., `run-xray14-resnet50-best.local.sh`)
- SBATCH scripts explicitly configured for optimal performance on SLURM clusters.
- Configures GPUs (supports torchrun DDP for `NUM_GPUS > 1`), CPUs, and memory limits.
- Sets output directory to `$SCRATCH/outputs-Xray-Classification-Intro-v2/$CLUSTER`.
- Directly invokes `src/train_v1_classic.py` (via `torchrun` if multi-gpu, or standard python).
- Handles environment setup using `source setup_env.sh` to ensure compatibility.

## Apptainer Container (`cuda-apptainer.sh` + `apptainer-run-*.sh`)
This repo can run the same training/evaluation code either directly on the host (above) or inside a prebuilt Apptainer image.

### Build / Run / Exec
- Build the image: `./cuda-apptainer.sh build` (creates `apptainer-cuda.sif`).
- Interactive shell: `./cuda-apptainer.sh run`
- Execute a command inside the container: `./cuda-apptainer.sh exec <cmd>`

Common examples from the `Pytorch/` directory:
```bash
./cuda-apptainer.sh exec python src/train_v1_classic.py dataset=chestxray14
./cuda-apptainer.sh exec torchrun --nproc_per_node=2 src/train_v1_classic.py dataset=chestxray14
```

### Execute the preconfigured SLURM wrappers
- Use the containerized SLURM scripts: `sbatch ./apptainer-run-*.sh`
- Example: `sbatch ./apptainer-run-xray14-resnet50-best.sh`



