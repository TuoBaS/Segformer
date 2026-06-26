import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.config import load_config
from utils.model_utils import build_model_from_config, load_checkpoint
from utils.visualization import ade20k_palette, colorize_mask, overlay_mask


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Run SegFormer inference on one image or a folder.")
    parser.add_argument("--config", default="configs/segformer_b0.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Image file or directory.")
    parser.add_argument("--output", default="outputs/inference")
    parser.add_argument("--device", default=None)
    parser.add_argument("--scales", type=float, nargs="+", default=[1.0])
    parser.add_argument("--flip", action="store_true", help="Enable horizontal flip test-time augmentation.")
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def list_images(input_path):
    path = Path(input_path)
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS)


def resize_keep_ratio(image, target_short):
    width, height = image.size
    if height < width:
        new_h = target_short
        new_w = round(width * target_short / height)
    else:
        new_w = target_short
        new_h = round(height * target_short / width)
    return image.resize((new_w, new_h), Image.BILINEAR)


def image_to_tensor(image, divisor=32):
    arr = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD

    h, w = tensor.shape[-2:]
    pad_h = int(np.ceil(h / divisor) * divisor - h)
    pad_w = int(np.ceil(w / divisor) * divisor - w)
    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value=0)
    return tensor.unsqueeze(0), (h, w)


@torch.no_grad()
def predict_image(model, image, device, base_short=512, scales=(1.0,), flip=False, amp_enabled=False):
    orig_w, orig_h = image.size
    logits_sum = None
    num_views = 0

    for scale in scales:
        target_short = max(1, int(round(base_short * scale)))
        resized = resize_keep_ratio(image, target_short)
        tensor, valid_size = image_to_tensor(resized)
        tensor = tensor.to(device)

        with torch.autocast(device_type="cuda", enabled=amp_enabled and device.type == "cuda"):
            logits = model(tensor)
        logits = logits[..., :valid_size[0], :valid_size[1]]
        logits = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        logits_sum = logits if logits_sum is None else logits_sum + logits
        num_views += 1

        if flip:
            flipped = tensor.flip(dims=[-1])
            with torch.autocast(device_type="cuda", enabled=amp_enabled and device.type == "cuda"):
                flip_logits = model(flipped)
            flip_logits = flip_logits.flip(dims=[-1])
            flip_logits = flip_logits[..., :valid_size[0], :valid_size[1]]
            flip_logits = F.interpolate(flip_logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            logits_sum = logits_sum + flip_logits
            num_views += 1

    pred = (logits_sum / num_views).argmax(dim=1)[0]
    return pred.cpu().numpy().astype(np.uint8)


def save_prediction(image, mask, image_path, output_dir, palette, alpha):
    stem = Path(image_path).stem
    output_dir = Path(output_dir)
    mask_dir = output_dir / "masks"
    color_dir = output_dir / "color"
    overlay_dir = output_dir / "overlay"
    mask_dir.mkdir(parents=True, exist_ok=True)
    color_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    color = colorize_mask(mask, palette)
    overlay = overlay_mask(np.asarray(image), color, alpha=alpha)

    Image.fromarray(mask).save(mask_dir / f"{stem}.png")
    Image.fromarray(color).save(color_dir / f"{stem}.png")
    Image.fromarray(overlay).save(overlay_dir / f"{stem}.png")


def main():
    args = parse_args()
    conf = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = build_model_from_config(conf, encoder_pretrained=False).to(device)
    load_checkpoint(model, args.checkpoint, device=device, strict=args.strict)
    model.eval()

    base_short = conf.get("augmentation", {}).get("val", {}).get("img_scale", [2048, 512])[1]
    amp_enabled = conf.get("amp", {}).get("enabled", False)
    palette = ade20k_palette(conf["model"].get("num_classes", 150))

    images = list_images(args.input)
    if not images:
        raise FileNotFoundError(f"No images found in {args.input}")

    for image_path in images:
        image = Image.open(image_path).convert("RGB")
        mask = predict_image(
            model,
            image,
            device,
            base_short=base_short,
            scales=args.scales,
            flip=args.flip,
            amp_enabled=amp_enabled,
        )
        save_prediction(image, mask, image_path, args.output, palette, args.alpha)
        print(f"saved: {image_path}")


if __name__ == "__main__":
    main()
