import torch
from tqdm import tqdm

from utils.Score import compute_metrics


@torch.no_grad()
def evaluate(model, val_loader, device, num_classes=150, amp_enabled=False,
             ignore_index=255, max_batches=None):
    """Evaluate semantic segmentation metrics on a validation loader.

    The implementation accumulates a confusion matrix with torch.bincount instead
    of expanding predictions to one-hot tensors. This keeps the metric identical
    while making ADE20K full-image validation much faster and lighter.
    """
    model.eval()

    confusion = torch.zeros((num_classes, num_classes), dtype=torch.float64, device=device)
    autocast_enabled = amp_enabled and device.type == "cuda"

    with torch.autocast(device_type="cuda", enabled=autocast_enabled):
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = batch["image"].to(device=device, non_blocking=True)
            true_masks = batch["mask"].to(device=device, dtype=torch.long, non_blocking=True)

            pred_logits = model(images)
            pred_masks = pred_logits.argmax(dim=1)

            valid = true_masks != ignore_index
            target = true_masks[valid]
            pred = pred_masks[valid].clamp(0, num_classes - 1)

            if target.numel() == 0:
                continue

            indices = target * num_classes + pred
            confusion += torch.bincount(
                indices,
                minlength=num_classes * num_classes,
            ).reshape(num_classes, num_classes).to(confusion.dtype)

    tp = torch.diag(confusion)
    fp = confusion.sum(dim=0) - tp
    fn = confusion.sum(dim=1) - tp

    model.train()
    return compute_metrics(tp, fp, fn)
