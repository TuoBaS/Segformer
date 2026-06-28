import math

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .dataset import ADE20KDataset
from .transforms import get_train_transforms, get_val_transforms
from utils.reproducibility import make_generator, seed_worker


def pad_collate_fn(batch, ignore_index=255, divisor=32):
    images = [item["image"] for item in batch]
    masks = [item["mask"] for item in batch]

    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    max_h = math.ceil(max_h / divisor) * divisor
    max_w = math.ceil(max_w / divisor) * divisor

    padded_images = []
    for img in images:
        pad_h = max_h - img.shape[1]
        pad_w = max_w - img.shape[2]
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, (0, pad_w, 0, pad_h), value=0)
        padded_images.append(img)

    padded_masks = []
    for mask in masks:
        pad_h = max_h - mask.shape[0]
        pad_w = max_w - mask.shape[1]
        if pad_h > 0 or pad_w > 0:
            mask = F.pad(mask, (0, pad_w, 0, pad_h), value=ignore_index)
        padded_masks.append(mask)

    return {
        "image": torch.stack(padded_images),
        "mask": torch.stack(padded_masks),
    }


def build_train_dataloader(
    img_dir,
    mask_dir,
    batch_size=2,
    num_workers=4,
    crop_size=512,
    img_scale=(2048, 512),
    ratio_range=(0.5, 2.0),
    flip_prob=0.5,
    photo_distortion=True,
    normalize=None,
    cat_max_ratio=0.75,
    reduce_zero_label=True,
    pin_memory=True,
    seed=None,
):
    dataset = ADE20KDataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        transforms=get_train_transforms(
            crop_size=crop_size,
            img_scale=img_scale,
            ratio_range=ratio_range,
            flip_prob=flip_prob,
            photo_distortion=photo_distortion,
            normalize=normalize,
            cat_max_ratio=cat_max_ratio,
        ),
        reduce_zero_label=reduce_zero_label,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker if seed is not None else None,
        generator=make_generator(seed),
    )


def build_val_dataloader(
    img_dir,
    mask_dir,
    batch_size=2,
    num_workers=4,
    img_scale=(2048, 512),
    normalize=None,
    reduce_zero_label=True,
    pin_memory=True,
    seed=None,
):
    dataset = ADE20KDataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        transforms=get_val_transforms(img_scale=img_scale, normalize=normalize),
        reduce_zero_label=reduce_zero_label,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=pad_collate_fn,
        worker_init_fn=seed_worker if seed is not None else None,
        generator=make_generator(seed),
    )
