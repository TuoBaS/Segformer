from torch import Tensor


def dice_score(input: Tensor, target: Tensor, reduce_batch: bool = False, epsilon: float = 1e-6, ignore_background: bool = True):
    """
    :param input: 预测概率图 [B, C, H, W] (经过 Softmax)
    :param target: 真实标签的 One-Hot 编码 [B, C, H, W]
    :param ignore_background: 是否忽略 Class 0 (背景)
    """
    assert input.dim() == 4, "Multiclass Dice requires [B, C, H, W] tensors"

    # 如果是二分类且忽略背景，我们只截取 C=1 (及之后) 的通道参与计算
    if ignore_background and input.shape[1] > 1:
        input = input[:, 1:, ...]
        target = target[:, 1:, ...]

    sum_dim = (0, 2, 3) if reduce_batch else (2, 3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)

    dice_per_class = (inter + epsilon) / (sets_sum + epsilon)

    # 返回所有参与计算的类的平均 Dice: 此时就剩下通道 C 也就是类别数
    return dice_per_class.mean()

def compute_metrics(tp: Tensor, fp: Tensor, fn: Tensor,EPSILON = 1e-6) -> dict:
    """
    分子不再加 EPSILON，防止 TP=0 且 FP=0 时指标强制等于 1.0 的 Bug
    """
    precision_per_class = tp / (tp + fp + EPSILON)
    recall_per_class = tp / (tp + fn + EPSILON)
    f1_per_class = (2 * tp) / (2 * tp + fp + fn + EPSILON)
    iou_per_class = tp / (tp + fp + fn + EPSILON)

    return {
        'Precision': precision_per_class.mean().item(),
        'Recall': recall_per_class.mean().item(),
        'F1-Score': f1_per_class.mean().item(),
        'Mean IoU': iou_per_class.mean().item(),
    }