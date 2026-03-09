# Training on GPU Cluster

This document outlines the steps and files required to launch the SFT training on a remote GPU cluster (e.g., Slurm, PBS) using 8x H100 GPUs.

## Required Files for Git
If you are pushing the training configurations to GitHub to clone on your cluster, you only need to commit the following core configuration files:

1. `config/llamafactory/sft_qwen3_full.yaml` (The main LLaMA-Factory training config)
2. `config/llamafactory/ds_z3_config.json` (DeepSpeed ZeRO-3 configuration)
3. `data/dataset_info.json` (Crucial: tells LLaMA-Factory how to parse your JSONL)
4. `scripts/run_sft_8H100.sh` (Your entry point script)

*(Optional but recommended)*:
- `accelerate_config.yaml` 

## What not to push
- Do NOT push your actual model folder (`models/Qwen3-8B-Base/`) - this is too large. Download it directly on the cluster using `huggingface-cli`.
- Do NOT push the 20k dataset (`data/sft_pairs_20k.jsonl`) if it's too large for GitHub (>100MB). Either use Git LFS or SCP/rsync it directly to the cluster.

## Launching on the Cluster
Once cloned on the cluster, ensure your dataset and base model are in the correct paths:
- `./models/Qwen3-8B-Base/`
- `./data/sft_pairs_20k.jsonl`

Then simply run your cluster's job submission script (e.g., `sbatch`) which should eventually execute:
```bash
bash scripts/run_sft_8H100.sh
```
