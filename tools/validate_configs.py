import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.config import load_config


REQUIRED_TOP_LEVEL = (
    "model",
    "data",
    "optimizer",
    "lr_scheduler",
    "runner",
    "augmentation",
    "loss",
)

REQUIRED_ENCODER_FIELDS = (
    "embed_dims",
    "num_heads",
    "depths",
    "sr_ratios",
    "mlp_ratios",
    "drop_path_rate",
)


def _as_pair(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a pair, got {value!r}")
    return tuple(value)


def _validate_positive_int(conf, dotted_name, errors):
    current = conf
    for part in dotted_name.split("."):
        if not isinstance(current, dict) or part not in current:
            errors.append(f"missing required field: {dotted_name}")
            return
        current = current[part]
    if not isinstance(current, int) or current <= 0:
        errors.append(f"{dotted_name} must be a positive integer, got {current!r}")


def validate_config(path):
    errors = []
    conf = load_config(str(path))

    for key in REQUIRED_TOP_LEVEL:
        if key not in conf:
            errors.append(f"missing top-level section: {key}")

    if errors:
        return errors

    model_cfg = conf["model"]
    encoder_cfg = model_cfg.get("encoder", {})
    decoder_cfg = model_cfg.get("decoder", {})
    data_cfg = conf["data"]
    aug_cfg = conf["augmentation"]
    train_aug = aug_cfg.get("train", {})
    val_aug = aug_cfg.get("val", {})
    runner_cfg = conf["runner"]

    for field in ("variant", "img_size", "num_classes", "encoder", "decoder"):
        if field not in model_cfg:
            errors.append(f"model.{field} is required")

    for field in REQUIRED_ENCODER_FIELDS:
        if field not in encoder_cfg:
            errors.append(f"model.encoder.{field} is required")

    encoder_lengths = [
        len(encoder_cfg.get("embed_dims", [])),
        len(encoder_cfg.get("num_heads", [])),
        len(encoder_cfg.get("depths", [])),
        len(encoder_cfg.get("sr_ratios", [])),
        len(encoder_cfg.get("mlp_ratios", [])),
    ]
    if any(length != 4 for length in encoder_lengths):
        errors.append("model.encoder list fields must all have 4 stages")

    if "decoder_dim" not in decoder_cfg:
        errors.append("model.decoder.decoder_dim is required")

    for split in ("train", "val"):
        split_cfg = data_cfg.get(split)
        if not isinstance(split_cfg, dict):
            errors.append(f"data.{split} section is required")
            continue
        for field in ("img_dir", "mask_dir"):
            if field not in split_cfg:
                errors.append(f"data.{split}.{field} is required")

    _validate_positive_int(conf, "data.crop_size", errors)
    _validate_positive_int(conf, "runner.max_iters", errors)
    _validate_positive_int(conf, "runner.checkpoint_interval", errors)
    _validate_positive_int(conf, "runner.eval_interval", errors)

    try:
        train_img_scale = _as_pair(train_aug.get("img_scale", [2048, 512]), "augmentation.train.img_scale")
        val_img_scale = _as_pair(val_aug.get("img_scale", [2048, 512]), "augmentation.val.img_scale")
        if train_img_scale[1] <= 0 or val_img_scale[1] <= 0:
            errors.append("augmentation img_scale short side must be positive")
    except ValueError as exc:
        errors.append(str(exc))

    try:
        ratio_range = _as_pair(train_aug.get("ratio_range", [0.5, 2.0]), "augmentation.train.ratio_range")
        if ratio_range[0] <= 0 or ratio_range[1] <= 0 or ratio_range[0] > ratio_range[1]:
            errors.append("augmentation.train.ratio_range must be positive and ordered")
    except ValueError as exc:
        errors.append(str(exc))

    crop_size = data_cfg.get("crop_size")
    img_size = model_cfg.get("img_size")
    if isinstance(crop_size, int) and isinstance(img_size, int) and crop_size != img_size:
        errors.append(f"data.crop_size ({crop_size}) and model.img_size ({img_size}) differ")

    if runner_cfg.get("max_iters", 0) < runner_cfg.get("eval_interval", 0):
        errors.append("runner.max_iters should be >= runner.eval_interval for full experiments")

    return errors


def iter_config_paths(args):
    if args.config:
        for item in args.config:
            yield Path(item)
        return

    configs_dir = ROOT_DIR / "configs"
    yield from sorted(path for path in configs_dir.glob("*.yaml") if not path.name.startswith("_"))


def main():
    parser = argparse.ArgumentParser(description="Validate SegFormer YAML configs without importing torch.")
    parser.add_argument("--config", nargs="*", help="One or more config files. Defaults to configs/*.yaml")
    args = parser.parse_args()

    failed = False
    for path in iter_config_paths(args):
        try:
            errors = validate_config(path)
        except Exception as exc:  # noqa: BLE001 - report config loading failures clearly.
            failed = True
            print(f"[FAIL] {path}: {exc}")
            continue

        if errors:
            failed = True
            print(f"[FAIL] {path}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"[ OK ] {path}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
