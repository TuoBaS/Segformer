import numpy as np


def ade20k_palette(num_classes=150):
    rng = np.random.RandomState(42)
    return rng.randint(40, 220, (num_classes, 3), dtype=np.uint8)


def colorize_mask(mask, palette=None, ignore_index=255):
    if palette is None:
        palette = ade20k_palette()

    mask = np.asarray(mask)
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = (mask != ignore_index) & (mask >= 0) & (mask < len(palette))
    color[valid] = palette[mask[valid]]
    return color


def overlay_mask(image, color_mask, alpha=0.55):
    image = np.asarray(image).astype(np.float32)
    color_mask = np.asarray(color_mask).astype(np.float32)
    blended = image * (1.0 - alpha) + color_mask * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)
