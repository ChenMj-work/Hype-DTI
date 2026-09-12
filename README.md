# Hype-DTI

Minimal reproducibility repository for Hype-DTI. It contains only the complete
training code and the five folds of the AGO, E-, and Blocker 10-shot datasets.

## Repository contents

```text
configs/hype_dti_full.json     Frozen experiment configuration
kge/                           Complete model and training code
tools/download_kge_checkpoint.py
tools/precompute_self_embeddings.py
tools/check_artifacts.py
tools/run_all_folds.py
var_data/                      AGO, E-, and Blocker data (five folds each)
environment.yml                Reproducible conda environment
```

Large pretrained weights and generated outputs are not stored in Git.

## Environment

The experiments were run on a Linux GPU server with:

- Python 3.8
- PyTorch 1.8.1
- CUDA 11.1
- DGL 0.6.0
- ChemBERTa, ESM-1b, and ESM2 pretrained models

Create the pinned environment:

```bash
conda env create -f environment.yml
conda activate hype-dti
```

`environment.yml` pins the remaining packages: matplotlib 3.4.2, NumPy
1.24.4, pandas 2.0.3, scikit-learn 0.24.1, NetworkX 3.1, DGL-LifeSci 0.2.9,
fair-esm 2.0.0, gdown 5.2.0, RDKit 2021.9.2.1, and Transformers 4.38.2.

A CUDA GPU is required for practical reproduction. The training pipeline loads
the 650M-parameter ESM-1b model. `--device -1` selects CPU, but a complete CPU
run is not practical.

Run every command below from the repository root.

## Prepare pretrained artifacts

Download the public DRKG embedding checkpoint:

```bash
python tools/download_kge_checkpoint.py
```

The downloader stores it in `var_models/` and verifies SHA-256:

```text
A2FC0281D2399A0B67D9554CF89AB3AE09BEE4059EE2A6FF308C37F375DC9F12
```

Generate the ChemBERTa drug embeddings and ESM2 protein embeddings:

```bash
python tools/precompute_self_embeddings.py --device cuda:0
```

This step downloads `seyonec/ChemBERTa-zinc-base-v1` and
`facebook/esm2_t33_650M_UR50D` from Hugging Face on first use. It is resumable.
For a smaller GPU, reduce `--batch_size` and `--protein_batch_size`.

Verify all data, embeddings, and the DRKG checkpoint before training:

```bash
python tools/check_artifacts.py
```

The expected generated files are:

```text
var_models/2024-04-28_10_01_40.24__kgeSLHstd_main.py--save--dataset__a-10--device__6--gate__kge.pth
var_features/pretrained_self/pretrained_self_drug_embed.pt
var_features/pretrained_self/pretrained_self_protein_embed.pt
```

The training entry point additionally downloads ESM-1b through `fair-esm` on
first use. On an offline server, populate the Hugging Face and ESM caches in
advance.

## Run one fold

```bash
python kge/std_main.py \
  --dataset ago_10shots_0 \
  --device 0 \
  --seed 0 \
  --load_kge_model 2024-04-28_10_01_40.24__kgeSLHstd_main.py--save--dataset__a-10--device__6--gate__kge.pth
```

Valid dataset names are:

```text
ago_10shots_0       ... ago_10shots_4
e-_10shots_0        ... e-_10shots_4
blocker_10shots_0   ... blocker_10shots_4
```

Each run executes the complete Hype-DTI pipeline: relation expert training,
relation-to-self pseudo-label generation, hybrid self-expert training,
self-to-relation distillation, and gate training/evaluation.

## Run all datasets

```bash
python tools/run_all_folds.py --device 0 --seed 0
```

This command runs all 15 folds sequentially. Outputs are written to:

```text
var_models/<dataset>/    Model checkpoints
var_pkls/<dataset>/      Predictions and diagnostics
```

To prevent validation/test pairs from entering new pseudo-label candidate
pools, add `--filter-protected-pseudo-candidates`. This changes the historical
protocol and should be reported as a new audited run rather than an exact
reproduction of archived metrics.

## Reproducibility notes

- The complete hyperparameter configuration is frozen in
  `configs/hype_dti_full.json`.
- Use the same seed, software versions, and comparable CUDA hardware when
  comparing results. Different CUDA kernels can cause small numerical changes.
- The 15 split directories, entity dictionaries, SMILES, and protein sequences
  are included. Model weights and language-model caches must be downloaded or
  generated as described above.
- The repository must remain the working directory because the training entry
  point resolves `var_data/`, `var_models/`, and `var_pkls/` relative to it.
