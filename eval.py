import argparse
import json
import logging
import os
import sys

import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data_process.builder import build_val_dataloader
from evaluate import evaluate
from utils.config import load_config
from utils.model_utils import build_model_from_config, load_checkpoint
from utils.reproducibility import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SegFormer on ADE20K validation set.")
    parser.add_argument("--config", default="configs/segformer_b0.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output", default=None, help="Optional JSON file for metrics.")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    conf = load_config(args.config)
    seed_everything(conf.get("seed", 42), deterministic=conf.get("deterministic", False))

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model_from_config(conf, encoder_pretrained=False).to(device)
    load_info = load_checkpoint(model, args.checkpoint, device=device, strict=args.strict)
    if load_info["missing_keys"] or load_info["unexpected_keys"]:
        logging.warning("Missing keys: %s", load_info["missing_keys"][:20])
        logging.warning("Unexpected keys: %s", load_info["unexpected_keys"][:20])

    data_cfg = conf["data"]
    aug_cfg = conf.get("augmentation", {})
    val_cfg = data_cfg["val"]
    val_aug_cfg = aug_cfg.get("val", {})
    normalize_cfg = aug_cfg.get("normalize", {})
    val_loader = build_val_dataloader(
        img_dir=val_cfg["img_dir"],
        mask_dir=val_cfg["mask_dir"],
        batch_size=args.batch_size or data_cfg.get("val_batch_size", 2),
        num_workers=args.num_workers if args.num_workers is not None else data_cfg.get("num_workers", 4),
        img_scale=tuple(val_aug_cfg.get("img_scale", [2048, 512])),
        normalize=normalize_cfg,
        reduce_zero_label=data_cfg.get("reduce_zero_label", True),
        seed=conf.get("seed", None),
    )

    metrics = evaluate(
        model,
        val_loader,
        device,
        num_classes=conf["model"].get("num_classes", 150),
        amp_enabled=conf.get("amp", {}).get("enabled", False),
        ignore_index=conf.get("loss", {}).get("ignore_index", 255),
        max_batches=args.max_batches if args.max_batches is not None else conf.get("evaluation", {}).get("max_batches", None),
    )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
