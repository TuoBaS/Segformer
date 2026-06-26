import torch
from torch.nn import functional as F


def compute_boundary_target(masks, ignore_index=255, kernel_size=3):
    """Build binary semantic boundary targets from class masks.

    A pixel is marked as boundary when classes inside its local window are not all
    identical. Windows touching ignore pixels are excluded from the edge loss.
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")

    valid = masks != ignore_index
    safe_masks = masks.clone()
    safe_masks = safe_masks.masked_fill(~valid, 0).float().unsqueeze(1)

    padding = kernel_size // 2
    local_max = F.max_pool2d(safe_masks, kernel_size, stride=1, padding=padding)
    local_min = -F.max_pool2d(-safe_masks, kernel_size, stride=1, padding=padding)
    boundary = (local_max != local_min).float()

    valid_float = valid.float().unsqueeze(1)
    valid_count = F.avg_pool2d(
        valid_float,
        kernel_size,
        stride=1,
        padding=padding,
        count_include_pad=False,
    )
    valid_boundary = (valid_count >= 1.0).float()
    return boundary * valid_boundary, valid_boundary
