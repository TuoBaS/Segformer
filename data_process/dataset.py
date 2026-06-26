import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# 支持的图像文件扩展名
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


class ADE20KDataset(Dataset):
    """
    ADE20K 语义分割数据集。

    ADE20K 标签说明:
        原始标签: 0 = 背景(wall/void), 1~150 = 前景类别
        reduce_zero_label=True 时: 前景类别减 1 变为 0~149, -1背景映射为 255 (ignore_index)
        这与官方 MMSeg 配置 reduce_zero_label=True 对齐。

    目录结构:
        img_dir/
            ADE_train_00000001.jpg
            ADE_train_00000002.jpg
            ...
        mask_dir/
            ADE_train_00000001.png    ← 注意: mask 是 .png 格式
            ADE_train_00000002.png
            ...
    """
    def __init__(self, img_dir, mask_dir, transforms=None, reduce_zero_label=True, ignore_index=255):
        """
        :param img_dir:            图像文件夹路径
        :param mask_dir:           标注 mask 文件夹路径
        :param transforms:         albumentations Compose 变换 pipeline
        :param reduce_zero_label:  是否将标签减 1 (ADE20K 标准做法, 0~149 为有效类别)
        :param ignore_index:       忽略标签值 (默认 255, 对应背景/无效像素)
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.reduce_zero_label = reduce_zero_label
        self.ignore_index = ignore_index

        # 收集所有图像文件名 (仅保留图像扩展名), 并排序保证可复现
        all_files = sorted(os.listdir(img_dir))
        self.img_names = [f for f in all_files if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS]

        assert len(self.img_names) > 0, f"在 {img_dir} 中未找到任何图像文件"

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # 关键修复: mask 文件名 = 图像文件名去掉扩展名 + .png
        # ADE20K 中图像是 .jpg, 标注是 .png, 二者 base name 相同
        mask_base = os.path.splitext(img_name)[0]
        mask_path = os.path.join(self.mask_dir, mask_base + '.png')

        # 读取图像 (RGB) 和标注 (灰度)
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path), dtype=np.int64)

        # reduce_zero_label: 将 ADE20K 的标签 0~150 映射为 -1~149
        # 其中 -1 (即原标签 0, 背景) 通过 +255 变为 ignore_index=255
        if self.reduce_zero_label:
            mask = mask - 1
            mask[mask == -1] = self.ignore_index

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        return {
            'image': image.float(),
            'mask': mask.long()
        }





if __name__ == "__main__":
    # 快速验证数据集加载
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import torch
    from data_process.transforms import get_train_transforms, get_val_transforms

    # 测试路径 (请根据实际数据路径修改)
    test_img_dir = '../data/ADEChallengeData2016/images/training'
    test_mask_dir = '../data/ADEChallengeData2016/annotations/training'

    if os.path.exists(test_img_dir):
        dataset = ADE20KDataset(
            img_dir=test_img_dir,
            mask_dir=test_mask_dir,
            transforms=get_train_transforms(crop_size=512),
            reduce_zero_label=True
        )
        print(f"数据集大小: {len(dataset)}")

        sample = dataset[0]
        print(f"Image shape: {sample['image'].shape}")  # [3, 512, 512]
        print(f"Mask shape:  {sample['mask'].shape}")    # [512, 512]
        print(f"Mask unique values (sample): {torch.unique(sample['mask'])[:10]}")
        print(f"Mask dtype: {sample['mask'].dtype}")     # torch.int64
    else:
        print(f"测试数据路径不存在: {test_img_dir}")
        print("请确认 ADE20K 数据已放置在 data/ 目录下。")