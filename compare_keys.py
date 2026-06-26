"""
预训练权重 vs 模型参数 对比工具
运行方式: python compare_keys.py
输出: compare_keys_result.txt (可在 PyCharm 中直接打开查看)
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from models.segmentor import SegFormer


def main():
    # ---- 构建模型 (不加载预训练权重) ----
    enc_cfg = {
        'embed_dims': [32, 64, 160, 256],
        'num_heads': [1, 2, 5, 8],
        'depths': [2, 2, 2, 2],
        'mlp_ratios': [4, 4, 4, 4],
    }
    dec_cfg = {'decoder_dim': 256}

    model = SegFormer(
        img_size=512, num_classes=150,
        encoder_pretrained=False,  # 不自动加载预训练权重
        encoder_config=enc_cfg, decoder_config=dec_cfg
    )

    # ---- 获取模型参数 (仅编码器部分) ----
    model_keys = set()
    for name, _ in model.encoder.named_parameters():
        model_keys.add(name)

    # ---- 加载预训练权重 ----
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pretrained', 'mit_b0.pth')
    state_dict = torch.load(ckpt_path, map_location='cpu')

    # 去除 segformer.encoder. 前缀，跳过 classifier
    pretrained_keys = set()
    skipped = []
    for key in state_dict.keys():
        if key.startswith('classifier'):
            skipped.append(key)
            continue
        if key.startswith('segformer.encoder.'):
            pretrained_keys.add(key.replace('segformer.encoder.', '', 1))

    # ---- 对比 ----
    matched = sorted(model_keys & pretrained_keys)
    missing_in_model = sorted(pretrained_keys - model_keys)      # 预训练有，模型没有
    unexpected_in_model = sorted(model_keys - pretrained_keys)    # 模型有，预训练没有

    # ---- 写入结果文件 ----
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'compare_keys_result.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("  预训练权重 (mit_b0.pth) vs 模型参数 对比结果\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"模型编码器参数总数:  {len(model_keys)}\n")
        f.write(f"预训练权重参数总数:  {len(pretrained_keys)} (已跳过 classifier)\n")
        f.write(f"跳过的分类头参数:    {len(skipped)}\n\n")

        f.write(f"匹配成功:            {len(matched)}\n")
        f.write(f"模型缺失 (预训练有): {len(missing_in_model)}\n")
        f.write(f"模型多余 (预训练无): {len(unexpected_in_model)}\n\n")

        # 匹配的参数
        f.write("-" * 80 + "\n")
        f.write(f"  匹配的参数 ({len(matched)})\n")
        f.write("-" * 80 + "\n")
        for k in matched:
            f.write(f"  {k}\n")

        # 模型缺失的参数
        if missing_in_model:
            f.write("\n" + "-" * 80 + "\n")
            f.write(f"  模型缺失的参数 ({len(missing_in_model)}) — 预训练有但模型没有\n")
            f.write("-" * 80 + "\n")
            for k in missing_in_model:
                f.write(f"  {k}\n")

        # 模型多余的参数
        if unexpected_in_model:
            f.write("\n" + "-" * 80 + "\n")
            f.write(f"  模型多余的参数 ({len(unexpected_in_model)}) — 模型有但预训练没有\n")
            f.write("-" * 80 + "\n")
            for k in unexpected_in_model:
                f.write(f"  {k}\n")

        # 跳过的参数
        if skipped:
            f.write("\n" + "-" * 80 + "\n")
            f.write(f"  已跳过的分类头参数 ({len(skipped)})\n")
            f.write("-" * 80 + "\n")
            for k in skipped:
                f.write(f"  {k}\n")

        # ---- 逐参数详细对比 (含 shape) ----
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("  逐参数 Shape 对比\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'参数名':<65} {'预训练 Shape':<20} {'模型 Shape':<20} {'匹配'}\n")
        f.write("-" * 80 + "\n")

        all_keys = sorted(model_keys | pretrained_keys)
        for k in all_keys:
            pt_shape = str(tuple(state_dict['segformer.encoder.' + k].shape)) if 'segformer.encoder.' + k in state_dict else "-"
            md_shape = ""
            if k in model_keys:
                for name, param in model.encoder.named_parameters():
                    if name == k:
                        md_shape = str(tuple(param.shape))
                        break
            match_str = "OK" if pt_shape == md_shape else "MISMATCH" if pt_shape != "-" and md_shape != "" else ""
            f.write(f"  {k:<63} {pt_shape:<20} {md_shape:<20} {match_str}\n")

    print(f"结果已保存到: {out_path}")
    print(f"匹配: {len(matched)} | 缺失: {len(missing_in_model)} | 多余: {len(unexpected_in_model)}")


if __name__ == '__main__':
    main()
