import torch

from models.segmentor import SegFormer


def build_model_from_config(conf, encoder_pretrained=None):
    model_cfg = conf["model"]
    if encoder_pretrained is None:
        encoder_pretrained = model_cfg.get("encoder_pretrained", True)

    return SegFormer(
        img_size=model_cfg.get("img_size", 512),
        num_classes=model_cfg.get("num_classes", 150),
        encoder_pretrained=encoder_pretrained,
        encoder_config=model_cfg["encoder"],
        decoder_config=model_cfg["decoder"],
    )


def extract_model_state(checkpoint):
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if "model" in checkpoint:
            return checkpoint["model"]
    return checkpoint


def strip_module_prefix(state_dict):
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


def load_checkpoint(model, checkpoint_path, device="cpu", strict=True):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = strip_module_prefix(extract_model_state(checkpoint))
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    return {
        "checkpoint": checkpoint,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }
