import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.config import load_config
from utils.model_utils import build_model_from_config


def parse_args():
    parser = argparse.ArgumentParser(description="Count model parameters.")
    parser.add_argument("--config", default="configs/segformer_b0.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    conf = load_config(args.config)
    model = build_model_from_config(conf, encoder_pretrained=False)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder = sum(p.numel() for p in model.encoder.parameters())
    decoder = sum(p.numel() for p in model.decoder.parameters())

    print(f"config:    {args.config}")
    print(f"variant:   {conf['model'].get('variant', 'unknown')}")
    print(f"total:     {total / 1e6:.3f} M")
    print(f"trainable: {trainable / 1e6:.3f} M")
    print(f"encoder:   {encoder / 1e6:.3f} M")
    print(f"decoder:   {decoder / 1e6:.3f} M")


if __name__ == "__main__":
    main()
