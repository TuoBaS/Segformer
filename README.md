# PyTorch SegFormer

This project is a clean PyTorch reproduction of SegFormer for ADE20K semantic segmentation.
The current B0 setting targets the official single-scale ADE20K validation result around
37.4 mIoU.

## Project Status

- SegFormer B0-B5 model configs.
- MiT encoder and All-MLP decoder implementation.
- ADE20K train/val dataloaders with `reduce_zero_label=True`.
- Iteration-based training with AdamW, poly LR, warmup, AMP, checkpointing, and W&B logging.
- Config-driven ADE20K augmentation pipeline with random resize, random crop, flip,
  photometric distortion, normalization, and padding.
- Independent validation CLI.
- Image/folder inference CLI with raw masks, colored masks, and overlay visualizations.
- Optional LCAR decoder improvements and edge auxiliary supervision for ablation studies.

## Environment

Install PyTorch for your CUDA version first, then install the remaining dependencies:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements/requirements.txt
```

## Data Layout

Expected ADE20K layout:

```text
data/
  ADEChallengeData2016/
    images/
      training/
      validation/
    annotations/
      training/
      validation/
  ADE20K_TestData/
    testing/
pretrained/
  mit_b0.pth
  mit_b1.pth
  ...
```

The ADE20K annotations use raw labels `0..150`. This project maps label `0` to
`ignore_index=255` and foreground labels `1..150` to `0..149`.

## Training

```bash
python train.py --config configs/segformer_b0.yaml
```

Resume from a full checkpoint:

```bash
python train.py --config configs/segformer_b0.yaml --load checkpoints/segformer_b0_iter16000.pth
```

The base config uses `seed: 42`. Training now seeds Python, NumPy, PyTorch, CUDA, and
DataLoader workers.

Validate all runnable configs before starting expensive experiments:

```bash
python tools/validate_configs.py
```

## Proposed Improvement

The baseline keeps the original SegFormer All-MLP decoder. The proposed lightweight
decoder variants only change the decoder, leaving the MiT-B0 backbone and ADE20K training
recipe unchanged:

- `SGFR`: semantic-guided feature recalibration. The deepest decoder feature generates
  channel gates for all four scales.
- `DCR`: depthwise contextual refinement. The fused 1/4-resolution feature is refined with
  lightweight multi-dilation depthwise branches.
- `LCAR`: combines `SGFR + DCR`.
- `LCAR+Edge`: adds an auxiliary semantic-boundary head during training. The boundary target
  is generated from ADE20K masks and is not used at inference.

Suggested ablation order:

```bash
python train.py --config configs/segformer_b0.yaml
python train.py --config configs/segformer_b0_sgfr.yaml
python train.py --config configs/segformer_b0_dcr.yaml
python train.py --config configs/segformer_b0_lcar.yaml
python train.py --config configs/segformer_b0_lcar_edge.yaml
```

Run a short engineering sanity check before full training:

```bash
python train.py --config configs/segformer_b0_lcar_edge_sanity.yaml
```

Count parameters:

```bash
python tools/count_params.py --config configs/segformer_b0.yaml
python tools/count_params.py --config configs/segformer_b0_lcar_edge.yaml
```

## Validation

Run validation only:

```bash
python eval.py ^
  --config configs/segformer_b0.yaml ^
  --checkpoint checkpoints/segformer_b0_best.pth ^
  --output outputs/eval_b0.json
```

The printed metric dictionary contains precision, recall, F1, and mean IoU.

## Inference

Single image:

```bash
python infer.py ^
  --config configs/segformer_b0.yaml ^
  --checkpoint checkpoints/segformer_b0_best.pth ^
  --input path/to/image.jpg ^
  --output outputs/demo
```

Folder inference with multi-scale + flip TTA:

```bash
python infer.py ^
  --config configs/segformer_b0.yaml ^
  --checkpoint checkpoints/segformer_b0_best.pth ^
  --input data/ADE20K_TestData/testing ^
  --output outputs/ade20k_test ^
  --scales 0.5 0.75 1.0 1.25 1.5 1.75 ^
  --flip
```

Outputs:

```text
outputs/demo/
  masks/    # uint8 class-id masks, values 0..149
  color/    # colored masks
  overlay/  # image-mask overlays
```

## Notes for Paper Experiments

Use this repository as the readable reproduction and prototype implementation. For larger
ablation studies, it is still useful to mirror the final modifications into MMSegmentation
so that distributed training, logging, test-time augmentation, and comparison baselines are
standardized.
