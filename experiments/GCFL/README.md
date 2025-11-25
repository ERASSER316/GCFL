# GCFL GCBlock-aware Federated Learning

This repository extends the GCFL experiments with GCBlock-style re-parameterization for communication-efficient federated learning.

## Module overview

- `Models/GCBlock.py`
  - Implements `GCBlock`, a multi-branch 3×3 block with optional 1×1 and identity branches during training.
  - Provides `build_fused_state_dict` to fuse each GCBlock into a single 3×3 kernel/bias pair for upload.
  - Provides `load_fused_weights_into_gc_model` to inflate fused weights back into training-time GCBlocks with variant **A** (zero auxiliary branches) or **B** (keep auxiliary branches as local adapters).
- `Models/CNNs.py`
  - Baseline `CNN_1`, `CNN_2`, `CNN_3` remain unchanged.
  - Adds `GC_CNN_1` and `GC_CNN_2`, which swap the second convolution for a `GCBlock`.
  - `build_model` helper instantiates baseline or GCBlock variants based on `model_name`/`use_gcblock`.
- `Standalone.py`
  - Homogeneous federated loop with FedAvg aggregation.
  - When GCBlock mode is enabled, clients fuse their GCBlocks before upload, the server averages fused weights, and clients inflate fused weights according to the selected variant before the next round.
  - Logs per-round communication cost (parameter/byte counts) and can append them to the result CSV.

## Running GCBlock-FL experiments

```bash
python -m experiments.GCFL.Standalone \
  --data-name cifar10 \
  --data-path data \
  --num-nodes 100 \
  --fraction 0.1 \
  --num-steps 500 \
  --save-path Results/GCFL_gc_cnn1 \
  --epochs 10 \
  --batch-size 512 \
  --nkernels 16 \
  --model gc_cnn1 \
  --use-gcblock true \
  --gcfl-variant A \
  --log-comm-to-csv true

python -m experiments.GCFL.Standalone \
  --data-name cifar10 \
  --data-path data \
  --num-nodes 100 \
  --fraction 0.1 \
  --num-steps 500 \
  --save-path Results/GCFL_gc_cnn1 \
  --epochs 10 \
  --batch-size 512 \
  --nkernels 16 \
  --model gc_cnn1 \
  --use-gcblock true \
  --gcfl-variant B \
  --log-comm-to-csv true

python -m experiments.GCFL.Standalone \
  --data-name cifar10 \
  --data-path data \
  --num-nodes 100 \
  --fraction 0.1 \
  --save-path Results/GCFL_cnn1 \
  --num-steps 500 \
  --epochs 10 \
  --batch-size 512 \
  --nkernels 16 \
  --model cnn1 \
  --use-gcblock false \
  --gcfl-variant A \
  --log-comm-to-csv true
```

Key flags:
- `--model {cnn1, gc_cnn1}`: choose baseline or GCBlock CNN.
- `--use-gcblock`: toggles GCBlock fusion-aware flow (enabled automatically for `gc_cnn1`).
- `--gcfl-variant {A,B}`: select auxiliary-branch handling on download.
- `--log-comm-to-csv`: append communication metrics to the CSV output.

## Fusion/inflation flow

1. **Local training** uses the full GCBlock (multi-branch) architecture.
2. **Upload**: `build_fused_state_dict` produces fused 3×3 kernels/biases for every GCBlock; non-GC parameters are sent unchanged.
3. **Server aggregation**: FedAvg runs over the fused GCBlock weights and non-GC parameters separately.
4. **Download**: `load_fused_weights_into_gc_model` injects the fused kernels back into each client's GCBlocks using variant **A** (zero auxiliary branches) or **B** (keep auxiliary branches), then training resumes for the next round.

## Notes

- GCBlock fusion requires BatchNorm running statistics; allow a few warm-up steps before relying on the fused view.
- Variant **B** treats auxiliary branches as client-specific adapters while still contributing to the fused upload.
