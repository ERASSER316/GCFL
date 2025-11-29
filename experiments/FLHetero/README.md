# Heterogeneous Federated Learning with GCBlock

This folder contains the GCBlock-enabled heterogeneous FL setup used by `FedMRL_hetero.py` and the CNN backbones under `Models/`. It documents how to launch experiments, what the GCBlock aggregation variants mean, and where to hook into the fusion utilities.

## Key Components
- **GCBlock** (`GCBlock.py`): multi-branch convolution with optional 3×3, 1×1, and identity paths that can be fused into a single convolution for aggregation. Supports λ-weighted fusion via `get_equivalent_kernel_bias` and auxiliary fusion helpers for personalized variants.
- **CNN_5_small** (`Models/CNNs.py`): dual-branch backbone combining the original convolutional stack with a parallel GCBlock stack; feature maps are summed before the classifier to keep the output shape unchanged.
- **Federated driver** (`FedMRL_hetero.py`): orchestrates local training, builds fused GCBlock parameters for upload with `build_fused_state_dict`, and reloads aggregated weights with `load_fused_weights_into_gc_model`.

## Running an Experiment
Example command for CIFAR-10 with GCBlock variant B and custom fusion weights:
```bash
python experiments/FLHetero/FedMRL_hetero.py \
  --data-name cifar10 \
  --num-nodes 20 \
  --fraction 0.2 \
  --nkernels 16 \
  --lambda-3x3 1.2 \
  --lambda-1x1 0.75 \
  --lambda-id 0.8 \
  --gc-variant B \
  --epochs 10 \
  --num-steps 100
```

### Important Flags
- `--lambda-3x3`, `--lambda-1x1`, `--lambda-id`: weights applied when fusing GCBlock branches before aggregation.
- `--gc-variant`: GCBlock aggregation logic. `B` preserves local auxiliary paths and adjusts only the main 3×3 branch; `A` zeroes auxiliary paths each round.
- Standard FL knobs such as `--num-nodes`, `--fraction`, `--batch-size`, `--epochs`, and `--num-steps` control sampling and optimization.

## Aggregation Workflow
1. **Local training** keeps GCBlocks in multi-branch mode.
2. **Upload** via `build_fused_state_dict`, which collects standard parameters unchanged and exports fused GCBlock kernels/biases.
3. **Server averaging** applies FedAvg on fused parameters.
4. **Download** through `load_fused_weights_into_gc_model`:
   - Variant **B** subtracts the auxiliary contribution (second 3×3, 1×1, identity) from the fused kernel and loads the remainder into the main 3×3 path, preserving local personalization.
   - Variant **A** zeroes auxiliary paths and loads the fused kernel directly into the main 3×3 path.

## Tips
- Keep GCBlocks in training mode during FL; only call `switch_to_deploy` if you need a single fused convolution for inference after training is complete.
- Ensure both branches in `CNN_5_small` share the same downsampling (stride/pooling) so the element-wise fusion remains shape-compatible.
- The fusion utilities operate on state dicts; if you add new GCBlocks, no extra wiring is needed as long as they are registered as modules within your model.
