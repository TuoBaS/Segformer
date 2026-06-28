import sys
import os

# 将项目根目录加入 sys.path，确保 models / data_process / utils 等子目录可正确导入
_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

import argparse
import logging
import torch
import numpy as np
import wandb
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

from data_process.builder import build_train_dataloader, build_val_dataloader
from evaluate import evaluate
from utils.boundary import compute_boundary_target
from utils.config import load_config
from utils.model_utils import build_model_from_config
from utils.reproducibility import seed_everything


# =====================================================================
# 1. 多项式学习率调度器 (Polynomial Decay + Linear Warmup)
# =====================================================================
class PolynomialLRWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    """
    自定义学习率调度器：结合了 线性预热 (Linear Warmup) 与 多项式衰减 (Polynomial Decay)。
    完全对齐 OpenMMLab (MMSegmentation) 中 SegFormer 官方所采用的 Poly 策略。

    学习率随迭代次数（iter）的变化分为两个阶段：
        1. Warmup 阶段 (iter < warmup_iters):
           学习率从 base_lr * warmup_ratio 线性增长到 base_lr。
        2. Decay 阶段 (iter >= warmup_iters):
           学习率从 base_lr 按照多项式曲线衰减到 min_lr，公式为：
           lr = (base_lr - min_lr) * (1 - progress)^power + min_lr
    """

    def __init__(self, optimizer, max_iters, warmup_iters=1500, warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, last_epoch=-1):
        self.max_iters = max_iters  # 训练的总迭代次数（对应公式中的总数）
        self.warmup_iters = warmup_iters  # 预热阶段的迭代次数（默认前1500步进行预热）
        self.warmup_ratio = warmup_ratio  # 预热初始学习率的倍率系数（默认从极小的值开始）
        self.power = power  # 多项式衰减的幂指数（power=1.0 时退化为线性衰减）
        self.min_lr = min_lr  # 最终衰减到的最小学习率（通常设为 0）
        # 调用父类 _LRScheduler 的初始化函数，初始化优化器和上一次的 epoch/iter 计数
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        计算当前迭代步（self.last_epoch）下，各个参数组应该享有的学习率。
        注意：在 PyTorch 的底层设计中，scheduler 里的 last_epoch 实际上代表的是 iteration（迭代步数）。
        """
        # ---- 阶段一：线性预热阶段 ----
        if self.last_epoch < self.warmup_iters:
            # 计算当前在预热阶段的进度比例，范围从 0 到 1
            # max(1, ...) 是为了防止 warmup_iters 被设置为 0 时发生除以零的错误
            alpha = self.last_epoch / max(1, self.warmup_iters)

            # 线性插值计算预热系数：从起始的 warmup_ratio 线性增长到 1.0
            warmup_factor = self.warmup_ratio + (1.0 - self.warmup_ratio) * alpha

            # 遍历优化器中的所有基础学习率（base_lrs），乘上当前的预热系数并返回
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        # ---- 阶段二：多项式衰减阶段 ----
        else:
            # 计算衰减阶段的进度：当前超出预热的步数 / 总衰减步数，范围从 0 到 1
            progress = (self.last_epoch - self.warmup_iters) / max(1, self.max_iters - self.warmup_iters)

            # 计算衰减因子：(1 - 进度)^power，并使用 max(0.0, ...) 确保因子不为负数（防止总步数超限）
            factor = max(0.0, (1.0 - progress) ** self.power)

            # 根据多项式衰减公式，计算每个参数组的当前学习率并返回
            return [self.min_lr + (base_lr - self.min_lr) * factor for base_lr in self.base_lrs]


# =====================================================================
# 2. 差异化参数优化器构建函数 (Param-wise Optimizer Builder)
# =====================================================================
def build_optimizer(model, opt_cfg, model_cfg):
    """
    构建 AdamW 优化器，并支持官方的差异化参数配置 (Paramwise Configuration)。

    核心设计动机（对齐 SegFormer 官方训练技巧）：
        1. 针对 LayerNorm 层和位置编码层（dwconv）：不施加权重衰减 (Weight Decay = 0)。
           因为这些层通常含有强烈的偏置或尺度缩放特征，进行衰减会损害模型的表达能力。
        2. 针对解码头 (Decoder Head)：给它赋予更高的学习率倍数 (通常是主干网络的 10 倍)。
           因为 Backbone (MiT) 通常使用了 ImageNet 预训练权重，只需要微调；而 Decoder 则是从头训练的，需要跑得更快。
    """
    # 从优化器配置字典中获取 paramwise 字典，若不存在则默认为空字典
    paramwise = opt_cfg.get('paramwise', {})
    # 获取解码头的学习率放大倍数，默认放大 10.0 倍
    head_lr_mult = paramwise.get('head_lr_mult', 10.0)

    # 获取基础学习率（用于 Backbone/Encoder）
    base_lr = opt_cfg['lr']
    # 获取基础权重衰减系数（默认应用到全连接、卷积等权重上）
    base_wd = opt_cfg['weight_decay']

    # 初始化 4 个空列表，用于将模型参数精确地划分到 4 个不同的特征优化组
    encoder_params = []  # 组 1: 编码器（主干网络）的普通权重（需要衰减）
    encoder_no_decay_params = []  # 组 2: 编码器中的 Norm 层、偏置和位置编码（不需要衰减）
    decoder_params = []  # 组 3: 解码头的普通权重（需要高学习率 + 衰减）
    decoder_no_decay_params = []  # 组 4: 解码头中的 Norm 层、偏置和位置编码（需要高学习率 + 不衰减）

    # 遍历网络中所有包含名称的参数
    for name, param in model.named_parameters():
        # 如果该参数不要求计算梯度（例如被冻结的层），则直接跳过，不加入优化器
        if not param.requires_grad:
            continue

        # 核心判断 1：通过参数名字是否以 'decoder.' 开头，判断它是否属于解码头
        is_decoder = name.startswith('decoder.')

        # 核心判断 2：通过名字中是否包含 'norm'，判断它是否是归一化层（如 LayerNorm, BatchNorm）
        is_norm = 'layer_norm' in name or 'norm' in name

        # 核心判断 3：在 SegFormer 中，MixFFN 内部的深度分离卷积（dwconv）充当了隐式位置编码（PE）
        # 官方策略规定位置编码层同样不进行权重衰减
        is_pos_block = 'dwconv' in name

        # ---- 根据上述判断，将参数精确分流到对应的 4 个桶中 ----
        if is_decoder:
            # 如果是解码头，再进一步细分是否需要进行 weight decay
            if is_norm or is_pos_block:
                decoder_no_decay_params.append(param)  # 归一化/位置编码 -> 进入解码头无衰减组
            else:
                decoder_params.append(param)  # 普通卷积/全连接 -> 进入解码头常规组
        else:
            # 如果是编码器（主干网络），同样细分是否需要进行 weight decay
            if is_norm or is_pos_block:
                encoder_no_decay_params.append(param)  # 归一化/位置编码 -> 进入主干无衰减组
            else:
                encoder_params.append(param)  # 普通卷积/全连接 -> 进入主干常规组

    # 将划分好的 4 个参数列表，打包成 PyTorch AdamW 能够识别的差异化参数组字典格式 (param_groups)
    param_groups = [
        # 组 1：主干网络常规参数，享用基础学习率，应用常规权重衰减
        {'params': encoder_params, 'lr': base_lr, 'weight_decay': base_wd},

        # 组 2：主干网络特殊参数（Norm等），享用基础学习率，权重衰减强制设为 0.0
        {'params': encoder_no_decay_params, 'lr': base_lr, 'weight_decay': 0.0},

        # 组 3：解码头常规参数，学习率放大 head_lr_mult 倍，应用常规权重衰减
        {'params': decoder_params, 'lr': base_lr * head_lr_mult, 'weight_decay': base_wd},

        # 组 4：解码头特殊参数（Norm等），学习率放大 head_lr_mult 倍，权重衰减强制设为 0.0
        {'params': decoder_no_decay_params, 'lr': base_lr * head_lr_mult, 'weight_decay': 0.0},
    ]

    # 获取 AdamW 的一阶和二阶动量系数 betas 传参，若无配置则默认为标准的 [0.9, 0.999]
    betas = tuple(opt_cfg.get('betas', [0.9, 0.999]))

    # 实例化 PyTorch 官方的 AdamW 优化器
    # 此时传入的不再是单一的 model.parameters()，而是我们精心定制好的差异化字典列表 param_groups
    optimizer = torch.optim.AdamW(param_groups, lr=base_lr, betas=betas, weight_decay=base_wd)

    # 返回配置完美的优化器对象
    return optimizer


def save_training_checkpoint(model, optimizer, scheduler, iteration, best_mIoU, conf, save_path):
    ckpt = {
        'iter': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_mIoU': best_mIoU,
        'config': conf,
    }
    torch.save(ckpt, save_path)
    return save_path


# ============================================================
# 训练主循环 (Iteration-based)
# ============================================================
def train(device, model, conf, checkpoint_dict=None):
    # ⭐1、解构配置
    data_cfg = conf['data']
    opt_cfg = conf['optimizer']
    sched_cfg = conf['lr_scheduler']
    runner_cfg = conf['runner']
    loss_cfg = conf.get('loss', {})
    model_cfg = conf['model']
    aug_cfg = conf.get('augmentation', {})
    eval_cfg = conf.get('evaluation', {})

    max_iters = runner_cfg['max_iters']
    checkpoint_interval = runner_cfg.get('checkpoint_interval', 4000)
    eval_interval = runner_cfg.get('eval_interval', 16000)
    log_interval = runner_cfg.get('log_interval', 50)
    final_eval = runner_cfg.get('final_eval', True)
    save_final_checkpoint = runner_cfg.get('save_final_checkpoint', True)

    work_dir = conf.get('work_dir', 'checkpoints')
    os.makedirs(work_dir, exist_ok=True)

    # ⭐2、构建 DataLoader ----
    train_data_cfg = data_cfg['train']
    val_data_cfg = data_cfg['val']
    train_aug_cfg = aug_cfg.get('train', {})
    val_aug_cfg = aug_cfg.get('val', {})
    normalize_cfg = aug_cfg.get('normalize', {})

    train_loader = build_train_dataloader(
        img_dir=train_data_cfg['img_dir'],
        mask_dir=train_data_cfg['mask_dir'],
        batch_size=data_cfg.get('batch_size', 2),
        num_workers=data_cfg.get('num_workers', 4),
        crop_size=data_cfg.get('crop_size', 512),
        img_scale=tuple(train_aug_cfg.get('img_scale', [2048, 512])),
        ratio_range=tuple(train_aug_cfg.get('ratio_range', [0.5, 2.0])),
        flip_prob=train_aug_cfg.get('flip_prob', 0.5),
        photo_distortion=train_aug_cfg.get('photo_distortion', True),
        normalize=normalize_cfg,
        cat_max_ratio=train_aug_cfg.get('cat_max_ratio', 0.75),
        reduce_zero_label=data_cfg.get('reduce_zero_label', True),
        seed=conf.get('seed', None),
    )

    val_loader = build_val_dataloader(
        img_dir=val_data_cfg['img_dir'],
        mask_dir=val_data_cfg['mask_dir'],
        batch_size=data_cfg.get('val_batch_size', 2),
        num_workers=data_cfg.get('num_workers', 4),
        img_scale=tuple(val_aug_cfg.get('img_scale', [2048, 512])),
        normalize=normalize_cfg,
        reduce_zero_label=data_cfg.get('reduce_zero_label', True),
        seed=conf.get('seed', None),
    )

    logging.info(f'训练集: {len(train_loader.dataset)} 张, Batch: {len(train_loader)}')
    logging.info(f'验证集: {len(val_loader.dataset)} 张, Batch: {len(val_loader)}')

    # ⭐3、优化器 & 调度器 ----
    optimizer = build_optimizer(model, opt_cfg, model_cfg)

    scheduler = PolynomialLRWithWarmup(
        optimizer,
        max_iters=max_iters,
        warmup_iters=sched_cfg.get('warmup_iters', 1500),
        warmup_ratio=sched_cfg.get('warmup_ratio', 1e-6),
        power=sched_cfg.get('power', 1.0),
        min_lr=sched_cfg.get('min_lr', 0.0),
    )

    # ⭐4、损失函数 ----
    ignore_index = loss_cfg.get('ignore_index', 255)
    criterion = CrossEntropyLoss(ignore_index=ignore_index)
    edge_cfg = loss_cfg.get('edge_aux', {})
    edge_aux_enabled = edge_cfg.get('enabled', False)
    edge_weight = edge_cfg.get('weight', 0.2)
    edge_kernel_size = edge_cfg.get('kernel_size', 3)
    edge_pos_weight = edge_cfg.get('pos_weight', 3.0)

    # ⭐5、断点恢复 ----
    start_iter = 0
    best_mIoU = 0.0
    if checkpoint_dict is not None:
        if 'model_state_dict' in checkpoint_dict:
            model.load_state_dict(checkpoint_dict['model_state_dict'])
            optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint_dict['scheduler_state_dict'])
            start_iter = checkpoint_dict.get('iter', 0)
            best_mIoU = checkpoint_dict.get('best_mIoU', 0.0)
            logging.info(f'已从断点恢复: iter={start_iter}, best_mIoU={best_mIoU:.4f}')

    # ---- 混合精度 (可选) ----
    amp_enabled = conf.get('amp', {}).get('enabled', False)
    autocast_enabled = amp_enabled and device.type == 'cuda'
    scaler = torch.amp.GradScaler(enabled=autocast_enabled)

    # ⭐6、WandB ----
    variant = model_cfg.get('variant', 'b0')
    wandb_cfg = conf.get('wandb', {})
    wandb_enabled = wandb_cfg.get('enabled', True)
    wandb_log_images = wandb_cfg.get('log_images', True)
    if wandb_enabled:
        wandb_kwargs = {
            'project': wandb_cfg.get('project', f'SegFormer-{variant}'),
            'resume': wandb_cfg.get('resume', 'allow'),
            'config': conf,
        }
        if wandb_cfg.get('mode') is not None:
            wandb_kwargs['mode'] = wandb_cfg['mode']
        wandb.init(**wandb_kwargs)

    # ⭐7、训练循环 ----
    model.train()
    train_iter = iter(train_loader)
    running_loss = 0.0
    running_edge_loss = 0.0
    last_eval_iter = start_iter if start_iter % eval_interval == 0 else 0
    last_images = None   # 保留最后一个 batch 用于可视化
    last_masks = None

    pbar = tqdm(range(start_iter, max_iters), desc='Training', unit='iter',
                initial=start_iter, total=max_iters)
    for current_iter in pbar:
        # 获取下一个 batch (DataLoader 耗尽后自动重置)
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        images = batch['image'].to(device, non_blocking=True)
        masks = batch['mask'].to(device, non_blocking=True)

        # 保留当前 batch 用于后续可视化 (仅引用，不额外占用显存)
        last_images = images
        last_masks = masks

        # 前向 + 反向
        optimizer.zero_grad(set_to_none=True)
        edge_loss_value = 0.0
        with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
            outputs = model(images, return_aux=edge_aux_enabled)
            seg_logits = outputs['seg'] if isinstance(outputs, dict) else outputs
            loss = criterion(seg_logits, masks)

            if edge_aux_enabled and isinstance(outputs, dict) and 'edge' in outputs:
                edge_target, edge_valid = compute_boundary_target(
                    masks,
                    ignore_index=ignore_index,
                    kernel_size=edge_kernel_size,
                )
                edge_loss_map = F.binary_cross_entropy_with_logits(
                    outputs['edge'],
                    edge_target,
                    reduction='none',
                )
                edge_loss_map = edge_loss_map * (1.0 + edge_target * (edge_pos_weight - 1.0))
                edge_loss = (edge_loss_map * edge_valid).sum() / edge_valid.sum().clamp_min(1.0)
                loss = loss + edge_weight * edge_loss
                edge_loss_value = edge_loss.detach().item()

        scaler.scale(loss).backward()   # 将 loss 放大 (防止 FP16 梯度下溢)，然后反向传播
        scaler.step(optimizer)          # 先将梯度缩小回原始尺度，再执行优化器更新 (若检测到 inf/nan 则跳过本次更新)
        scaler.update()                 # 根据本次是否出现 inf/nan，动态调整下一次的缩放因子
        scheduler.step()                # 更新学习率 (polynomial decay + warmup)

        running_loss += loss.item()
        running_edge_loss += edge_loss_value

        # 更新进度条
        pbar.set_postfix(loss=f'{loss.item():.4f}', lr=f'{optimizer.param_groups[0]["lr"]:.6f}')

        # ---- 日志 ----
        if (current_iter + 1) % log_interval == 0:
            iter_num = current_iter + 1
            avg_loss = running_loss / log_interval
            avg_edge_loss = running_edge_loss / log_interval
            lr = optimizer.param_groups[0]['lr']
            logging.info(f'Iter [{iter_num}/{max_iters}] loss={avg_loss:.4f} edge={avg_edge_loss:.4f} lr={lr:.6f}')
            log_dict = {'train/loss': avg_loss, 'train/lr': lr}
            if edge_aux_enabled:
                log_dict['train/edge_loss'] = avg_edge_loss
            if wandb_enabled:
                wandb.log(log_dict, step=iter_num)
            running_loss = 0.0
            running_edge_loss = 0.0

        # ---- 定期评估 & 保存 ----
        if (current_iter + 1) % eval_interval == 0:
            iter_num = current_iter + 1
            metrics = evaluate(model, val_loader, device,
                               num_classes=model_cfg['num_classes'],
                               amp_enabled=autocast_enabled,
                               max_batches=eval_cfg.get('max_batches', None))
            val_mIoU = metrics['Mean IoU']
            last_eval_iter = iter_num
            logging.info(f'Eval @ Iter {iter_num}: mIoU={val_mIoU:.4f} (best={best_mIoU:.4f})')
            if wandb_enabled:
                wandb.log({'val/mIoU': val_mIoU, **{f'val/{k}': v for k, v in metrics.items()}}, step=iter_num)

            # ---- Wandb 可视化: 原图 / 真实 mask / 预测 mask ----
            if wandb_enabled and wandb_log_images and last_images is not None:
                try:
                    model.eval()

                    # 反归一化: ImageNet 标准化 → 原始 RGB
                    # albumentations Normalize: (img - mean*255) / (std*255)
                    # 反归一化: (tensor * std + mean) * 255
                    img_vis = last_images[0].cpu()
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img_vis = torch.clamp((img_vis * std + mean) * 255, 0, 255)
                    img_vis = img_vis.permute(1, 2, 0).numpy().astype('uint8')

                    # 真实 mask
                    true_mask_vis = last_masks[0].cpu().numpy()

                    # 预测 mask (eval 模式, 关闭 dropout/BN 训练统计)
                    with torch.no_grad():
                        pred_logits = model(last_images)
                    pred_mask_vis = pred_logits.argmax(dim=1)[0].cpu().numpy()

                    # 类别着色: 0-149 用随机色, 255(ignore) 用黑色
                    rng = np.random.RandomState(42)
                    colors = rng.randint(40, 220, (150, 3), dtype=np.uint8)

                    def colorize(mask):
                        h, w = mask.shape
                        rgb = np.zeros((h, w, 3), dtype=np.uint8)
                        valid = (mask >= 0) & (mask < 150)
                        rgb[valid] = colors[mask[valid]]
                        return rgb

                    gt_vis = colorize(true_mask_vis)
                    pred_vis = colorize(pred_mask_vis)

                    wandb.log({
                        'Original': wandb.Image(img_vis),
                        'Ground_Truth': wandb.Image(gt_vis),
                        'Prediction': wandb.Image(pred_vis),
                    }, step=iter_num)

                    model.train()
                except Exception as e:
                    logging.warning(f'Wandb 可视化失败: {e}')
                    model.train()

            # 保存最优 checkpoint
            if val_mIoU > best_mIoU:
                best_mIoU = val_mIoU
                save_path = os.path.join(work_dir, f'segformer_{variant}_best.pth')
                save_training_checkpoint(model, optimizer, scheduler, iter_num, best_mIoU, conf, save_path)
                logging.info(f'已保存最优模型: {save_path}')

            model.train()

        # 定期保存 checkpoint (非最优)
        if (current_iter + 1) % checkpoint_interval == 0:
            iter_num = current_iter + 1
            save_path = os.path.join(work_dir, f'segformer_{variant}_iter{iter_num}.pth')
            save_training_checkpoint(model, optimizer, scheduler, iter_num, best_mIoU, conf, save_path)

    if final_eval and max_iters != last_eval_iter:
        metrics = evaluate(model, val_loader, device,
                           num_classes=model_cfg['num_classes'],
                           amp_enabled=autocast_enabled,
                           max_batches=eval_cfg.get('max_batches', None))
        val_mIoU = metrics['Mean IoU']
        logging.info(f'Final Eval @ Iter {max_iters}: mIoU={val_mIoU:.4f} (best={best_mIoU:.4f})')
        if wandb_enabled:
            wandb.log({'val/final_mIoU': val_mIoU, **{f'val/final_{k}': v for k, v in metrics.items()}}, step=max_iters)
        if val_mIoU > best_mIoU:
            best_mIoU = val_mIoU
            save_path = os.path.join(work_dir, f'segformer_{variant}_best.pth')
            save_training_checkpoint(model, optimizer, scheduler, max_iters, best_mIoU, conf, save_path)
            logging.info(f'已保存最优模型: {save_path}')

    if save_final_checkpoint:
        final_path = os.path.join(work_dir, f'segformer_{variant}_final.pth')
        save_training_checkpoint(model, optimizer, scheduler, max_iters, best_mIoU, conf, final_path)
        logging.info(f'已保存最终模型: {final_path}')

    # ---- 训练结束 ----
    logging.info(f'训练完成! 最终 best mIoU = {best_mIoU:.4f}')
    if wandb_enabled:
        wandb.finish()


# ====================================================================================
def get_args():
    parser = argparse.ArgumentParser(description='SegFormer Configuration')
    parser.add_argument('--config', type=str, default='configs/segformer_b0.yaml')
    parser.add_argument('--load', type=str, default=None, help='断点恢复或加载已训练模型的 checkpoint 路径')
    return parser.parse_args()

if __name__ == '__main__':
    # 设置日志格式
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    # 加载配置
    args = get_args()
    conf = load_config(args.config)
    seed_everything(conf.get('seed', 42), deterministic=conf.get('deterministic', False))
    model_cfg = conf['model']
    logging.info(f'配置文件: {args.config}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    model = build_model_from_config(conf).to(device=device)

    # 断点恢复 / 加载权重
    checkpoint_dict = None
    if args.load:
        checkpoint = torch.load(args.load, map_location=device)
        if 'model_state_dict' in checkpoint:
            checkpoint_dict = checkpoint
            logging.info(f'检测到完整断点! 将从 iter={checkpoint.get("iter", 0)} 恢复。')
        else:
            # 纯权重文件 (无优化器状态)
            if 'mask_values' in checkpoint:
                del checkpoint['mask_values']
            model.load_state_dict(checkpoint, strict=False)
            logging.info(f'已加载纯模型权重: {args.load}')

    try:
        train(
            device=device,
            model=model,
            conf=conf,
            checkpoint_dict=checkpoint_dict,
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('GPU OutOfMemoryError! 尝试清空缓存...')
        torch.cuda.empty_cache()
        raise
