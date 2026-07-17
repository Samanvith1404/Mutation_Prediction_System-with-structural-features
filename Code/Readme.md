# Epitope Escape Mutation Prediction — GNN Training Pipeline

A graph neural network pipeline that predicts which residues in an antibody-binding
region ("epitope sphere") on a viral antigen are likely to carry escape mutations.
Each binding region is represented as a graph: residues are nodes, spatial/sequence
proximity defines edges, and node features combine ESM protein-language-model
embeddings with structural descriptors. The model does per-residue binary
classification with a learnable decision threshold.

## How it fits together

```
sphere_radius_mapped_dataset.csv  ─┐
comprehensive_pdb_parsed_metrics.csv ─┼─▶ data_loader.py ──▶ model.py ──▶ inference.py
                                     │         (graphs)      (EGNN + trainer)  (metrics/report)
                          esm_utils.py (ESM-C embeddings, cached)
                                     │
                                train.py  (orchestrates everything, CLI entry point)
```

`verify_setup.py` is a standalone pre-flight check that confirms all files,
data, dependencies, and GPU support are in place before a training run.

## File-by-file

### `train.py`
Main entry point / CLI orchestrator. Wires together every other module into a
9-step pipeline:
1. Initializes `ESMEmbeddingManager`, a thin wrapper that either talks to the
   real ESM-C API (via `esm_utils.ESMCClient`) or silently falls back to
   random placeholder embeddings if no API key/SDK is available — this keeps
   the pipeline runnable end-to-end even without live embeddings.
2. Loads the sphere and structural-metrics CSVs and builds train/val/test
   `DataLoader`s via `data_loader.create_data_loaders`.
3. Builds the `RegionScorer` model (`model.py`), with input dimension
   `1280 (ESM) + 9 (structural features)`.
4. Wraps it in an `EpitopeTrainer` (loss, optimizer, checkpointing).
5. Runs the full training loop.
6. Serializes training history (loss curves, metrics per epoch) to JSON.
7. Evaluates the trained model on train/val/test sets via
   `inference.EpitopeEvaluator`.
8. Generates a written evaluation report.
9. Dumps the full run configuration (model + training hyperparameters) to
   `config.json` for reproducibility.

Run with:
```bash
python train.py \
  --sphere_data sphere_radius_mapped_dataset.csv \
  --metrics_data comprehensive_pdb_parsed_metrics.csv \
  --esm_cache /path/to/esm_cache \
  --esm_token $ESM_API_KEY \
  --batch_size 32 --num_epochs 100
```
All hyperparameters (hidden dim, message-passing depth, learning rate,
positive-class weight, similarity threshold, etc.) are exposed as CLI flags —
see `python train.py --help`.

### `data_loader.py`
Turns the two source CSVs into PyTorch Geometric graphs.

- **`EpitopeGraphDataset`** — a `torch_geometric.data.Dataset` subclass. For
  each split (`train`/`val`/`test`) it:
  - Merges the sphere-mapping and structural-metrics CSVs on their shared
    columns.
  - Filters rows by a minimum GISAID-to-PDB **sequence similarity threshold**
    (default 0.90).
  - Buckets rows into similarity tiers (`0.80–0.90`, `0.90–0.95`, `0.95–1.0`)
    and splits train/val/test **by tier** rather than randomly — high-identity
    strains (tier1) go to training, a sampled slice of tier2 goes to
    validation, and the most-diverged strains (tier3) are held out entirely
    for testing. This is a deliberate design to test generalization to more
    evolutionarily distant strains, not just random hold-out.
  - For each row, builds one graph per epitope "sphere" (a binding region of
    `n_residues` around an antibody contact site):
    - Fetches ESM-C embeddings per residue (via the passed-in `esm_manager`,
      i.e. `ESMCClient`), or falls back to random placeholder vectors if no
      manager/API key is available.
    - Appends structural node features — solvent accessibility (RSA), surface
      normal vectors, and local geometry descriptors (currently placeholder
      random values in `_get_structural_features`, meant to be replaced with
      real dMaSIF-style structural features).
    - Builds edges via **k-nearest-neighbor** in embedding space (`k=10`) plus
      sequence-adjacency edges (residues within 4 positions of each other).
    - Computes 2D edge attributes from surface-normal orientation (dot
      product + norm of the difference) between connected residues.
    - Labels are currently assigned to **random** residue positions within
      each sphere based on the known mutation count for that row (a
      placeholder pending true residue-level mutation coordinates).
- **`create_data_loaders(...)`** — convenience function that builds all three
  `EpitopeGraphDataset` splits and wraps them in PyG `DataLoader`s.

> **Note:** `_get_structural_features` and the mutation-label assignment in
> `_process` currently use `np.random` placeholders. These are the two spots
> to wire up real structural feature extraction and true per-residue mutation
> coordinates before this becomes a production training pipeline.

### `model.py`
Defines the GNN architecture and training loop.

- **`StructuralMessagePassing`** — an EGNN-style message-passing layer (built
  on `torch_geometric.nn.MessagePassing`, `aggr='mean'`). At each layer,
  messages are computed from concatenated source/target node features plus
  edge attributes, then aggregated and used (with a residual-style
  concatenation) to update node features via a small MLP + ReLU.
- **`RegionScorer`** — the full model: stacks `num_mp_layers` of
  `StructuralMessagePassing`, followed by an `num_final_layers`-deep MLP
  classification head producing one logit per residue. Also owns a
  **learnable decision threshold** (`nn.Parameter`, initialized at 0.5) used
  to binarize predicted probabilities — rather than hard-coding 0.5, the
  model can shift its operating point during training.
- **`WeightedBCELoss`** — binary cross-entropy with a `pos_weight` multiplier
  on the positive (mutation) class, since escape-mutation residues are rare
  relative to non-mutated residues in each sphere.
- **`EpitopeTrainer`** — training harness: Adam optimizer,
  `ReduceLROnPlateau` scheduler, gradient clipping, early stopping
  (`max_patience=10` epochs without val-loss improvement), and checkpointing
  of the best model (weights, optimizer state, threshold) to disk. Computes
  precision/recall/F1/MCC/AUROC each validation epoch via
  `_compute_metrics`.

### `esm_utils.py`
Wraps Meta's official **ESM-C** SDK (`esm.sdk.forge.ESMCForgeInferenceClient`)
for fetching protein-language-model embeddings, pointed at a BioHub.ai
inference endpoint.

- **`ESMCClient`** — handles the encode → logits round-trip needed to pull
  per-residue embeddings out of ESM-C (`ESMProtein` → `client.encode()` →
  `client.logits(..., return_embeddings=True)`), with:
  - **Disk caching** keyed by an MD5 hash of the sequence, tracked in a JSON
    manifest (`manifest.json`) so repeated runs skip re-fetching identical
    sequences.
  - **Failure memoization** — sequences that previously failed to embed are
    remembered and skipped on subsequent calls rather than retried.
  - `batch_embed(...)` for embedding many strains at once with a progress
    bar, and `get_cache_stats()` / `clear_cache()` for cache management.
- **`EmbeddingLoader`** — a thin high-level convenience wrapper around
  `ESMCClient` exposing `load_for_dataset(...)` and the expected embedding
  dimensionality (1536 for `esmc-6b-2024-12`).
- Runnable standalone (`python esm_utils.py <api_key>`) to smoke-test that a
  given API key can successfully fetch one embedding.

> Note: `esm_utils.py` and `train.py` disagree slightly on embedding size —
> `esm_utils.py`'s `EmbeddingLoader` documents ESM-C-6b as 1536-dim, while
> `model.py`/`train.py` assume a 1280-dim ESM embedding (`in_channels=1280+9`).
> Confirm which ESM-C model/dimension you're actually using and keep the
> model's `in_channels` consistent with it.

### `inference.py`
*(imported by `train.py` as `EpitopeEvaluator` — not included in this upload,
but based on usage: computes comprehensive test-time metrics per loader
(`evaluate_loader`), including per-PDB-structure breakdowns
(`_compute_per_pdb_metrics`), and produces a written evaluation report via
`generate_report`.)*

### `verify_setup.py`
Standalone environment/sanity checker, run before training to catch problems
early. Checks, in order:
1. **Project files present** — `train.py`, `data_loader.py`, `model.py`,
   `inference.py`, `esm_utils.py`, `requirements.txt`, `README.md`,
   `QUICKSTART.md`.
2. **Input data files present** — the sphere and metrics CSVs.
3. **Python dependencies installed** — torch, torch-geometric, numpy, pandas,
   scikit-learn, matplotlib, scipy, requests, tqdm (with version reporting).
4. **CUDA/GPU availability** — reports device name and memory if available.
5. **Source-level structure validation** — greps each core file for expected
   class/function names (e.g. confirms `RegionScorer`, `EpitopeTrainer`,
   `EpitopeGraphDataset`, `EpitopeEvaluator` are actually defined) to catch
   incomplete or corrupted files.
6. **ESM-C API key** — checks `ESM_API_KEY`/`ESMC_API_KEY` environment
   variables and prints usage instructions.

Prints a colored pass/fail summary and exits with status 0/1 accordingly —
suitable for use as a CI smoke test or a manual pre-training checklist.

### `requirements.txt`
Python dependencies: PyTorch, PyTorch Geometric, NumPy/Pandas/SciPy,
scikit-learn, matplotlib/seaborn (plotting), tqdm, HuggingFace Transformers,
Biopython, and `requests`.

## Known placeholders / TODOs

These are the parts of the current codebase that are explicitly stubbed out
and should be replaced before treating results as final:

- **Structural node features** (`data_loader._get_structural_features`) are
  random noise, not real RSA/surface-normal/geometry values.
- **Per-residue mutation labels** are randomly assigned within each sphere
  (matching only the *count* of known mutations, not their true positions).
- **ESM embeddings fall back to random vectors** whenever no API key/SDK is
  available, so a run without `--esm_token` will silently train on noise —
  worth adding a loud warning or hard failure mode for production runs.
- **Embedding dimension mismatch** between `esm_utils.py` (1536) and
  `model.py`/`train.py` (1280) should be reconciled.
