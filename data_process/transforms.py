"""
    数据增强 Pipeline — 对齐 SegFormer 官方 MMSeg 配置

    官方训练 pipeline (ADE20K):
        1. Resize (ratio_range=[0.5, 2.0], img_scale=(2048, 512)) # 随机缩放
        2. RandomCrop (crop_size=512, cat_max_ratio=0.75)         # 随机裁剪
        3. RandomFlip (prob=0.5)                                  # 随机翻转
        4. PhotoMetricDistortion (brightness, contrast, saturation, hue) # 色彩扭曲
        5. Normalize (ImageNet mean/std)                          # 归一化
        6. Pad (to crop_size, pad_val=0, seg_pad_val=255)         # 边缘填充

    官方测试 pipeline:
        1. Resize (keep_ratio=True, size_divisor=32)              # 保持比例缩放，并保证是32的倍数
        2. Normalize (ImageNet mean/std)                          # 归一化
        3. ToTensor                                               # 转为张量
"""
import numpy as np                        # 引入 numpy 处理矩阵运算
import albumentations as A                # 引入 albumentations 图像增强库
from albumentations import ToTensorV2     # 引入转 PyTorch Tensor 的工具


# ==================== ImageNet 归一化常量 (与官方一致) ====================
# albumentations 库在执行 Normalize 时，内部会自动将输入的 [0, 1] 标准差乘以 255。
# 因此这里不能传 [123.675, 116.28, 103.53] 这种尺度，必须传 [0, 1] 尺度的均值/标准差。
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB三通道的均值
IMAGENET_STD = [0.229, 0.224, 0.225]   # RGB三通道的标准差


class PhotoMetricDistortion(A.ImageOnlyTransform):
    """
    模拟 MMSeg 的 PhotoMetricDistortion (光度失真):
    随机调整图像的亮度、对比度、饱和度、色调。
    继承 ImageOnlyTransform 是因为这个变换只改变图像，不改变 Mask(分割标签)。
    """
    def __init__(self, brightness_delta=32, contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5), hue_delta=18, p=0.5):
        super().__init__(p=p) # p 是触发该数据增强的概率
        self.brightness_delta = brightness_delta # 亮度变化幅度
        self.contrast_range = contrast_range     # 对比度变化范围
        self.saturation_range = saturation_range # 饱和度变化范围
        self.hue_delta = hue_delta               # 色调变化幅度

    def apply(self, img, **params):
        # 1. 随机亮度偏移 (50% 概率触发)
        if np.random.randint(2): # 随机生成0或1
            # 随机生成一个在 [-32, 32] 之间的亮度增量
            delta = np.random.uniform(-self.brightness_delta, self.brightness_delta)
            # 转为 float32 防止溢出，加上亮度增量
            img = img.astype(np.float32) + delta

        # 2. 随机对比度调整 (50% 概率触发)
        if np.random.randint(2):
            # 随机生成一个对比度乘子 (0.5 到 1.5)
            alpha = np.random.uniform(*self.contrast_range)
            # 像素值乘以对比度系数
            img = img.astype(np.float32) * alpha

        # 准备进行色调和饱和度调整，必须先将像素截断在 0-255 范围内，并转回 uint8
        img = np.clip(img, 0, 255).astype(np.uint8)
        import cv2 # 局部导入 cv2 处理色彩空间转换
        # 将 RGB 图像转换到 HSV 色彩空间 (色调、饱和度、明度)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)

        # 3. 随机饱和度调整 (50% 概率触发)
        if np.random.randint(2):
            alpha = np.random.uniform(*self.saturation_range)
            # 对 HSV 的 S通道（通道1，即饱和度）乘以随机系数
            img_hsv[:, :, 1] = img_hsv[:, :, 1] * alpha

        # 4. 随机色调偏移 (50% 概率触发)
        if np.random.randint(2):
            delta = np.random.uniform(-self.hue_delta, self.hue_delta)
            # 对 HSV 的 H通道（通道0，即色调）加上随机偏移量
            img_hsv[:, :, 0] = img_hsv[:, :, 0] + delta

        # 修正越界值：OpenCV 中 8位图像的 Hue 范围是 0-180，Saturation 是 0-255
        img_hsv[:, :, 0] = np.clip(img_hsv[:, :, 0], 0, 180)
        img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1], 0, 255)

        # 调整完毕后，从 HSV 空间转回 RGB 空间
        img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return img


class RandomResize(A.DualTransform):
    """
    随机缩放 — 对齐官方。
    DualTransform 表示这个变换会同时、同比例地作用于 Image 和 Mask(标签)。
    """
    def __init__(self, short_size=512, ratio_range=(0.5, 2.0), p=1.0):
        super().__init__(p=p)
        self.short_size = short_size # 目标短边长度基准值
        self.ratio_range = ratio_range # 缩放比例的随机范围

    def apply(self, img, scale_factor=1.0, **params):
        import cv2
        h, w = img.shape[:2] # 获取原图高宽
        # 计算出新的短边长度：基准长度 * 随机出来的缩放比例
        new_short = int(self.short_size * scale_factor)

        # 判断哪条边是短边，按比例计算新的高宽
        if h < w: # 如果高是短边
            new_h, new_w = new_short, int(w * new_short / h)
        else:     # 如果宽是短边或正方形
            new_w, new_h = new_short, int(h * new_short / w)
        # 对图像进行线性插值缩放
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def apply_to_mask(self, mask, scale_factor=1.0, **params):
        # 标签 (Mask) 的缩放逻辑与图像完全一致
        import cv2
        h, w = mask.shape[:2]
        new_short = int(self.short_size * scale_factor)
        if h < w:
            new_h, new_w = new_short, int(w * new_short / h)
        else:
            new_w, new_h = new_short, int(h * new_short / w)
        # 注意：分割 Mask 缩放必须使用最近邻插值 (INTER_NEAREST)！
        # 因为 Mask 里面的值代表类别索引（如 1 代表人, 2 代表车），不能出现 1.5 这种小数。
        return cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    def get_params(self):
        # 每次执行时，在 ratio_range (0.5 - 2.0) 内随机生成一个缩放因子
        scale_factor = np.random.uniform(*self.ratio_range)
        return {'scale_factor': scale_factor} # 传递给 apply 和 apply_to_mask 使用


class PadToSize(A.DualTransform):
    """
    填充到目标尺寸。因为下一步要进行固定尺寸裁剪（比如 512x512），
    如果经过 Resize 后图片太小了（比如 400x400），裁剪就会报错。这里提前补全。
    """
    def __init__(self, size=(512, 512), pad_val=0, seg_pad_val=255, p=1.0):
        super().__init__(p=p)
        self.size = size  # 目标 (H, W)
        self.pad_val = pad_val         # 图片填充值 (默认 0，黑色)
        self.seg_pad_val = seg_pad_val # 标签填充值 (默认 255，即 ignore_index，计算 loss 时会忽略)

    def apply(self, img, **params):
        h, w = img.shape[:2]
        target_h, target_w = self.size
        # 计算需要填充的高度和宽度（如果原图比目标大，则 max 返回 0，不需要填充）
        pad_h = max(target_h - h, 0)
        pad_w = max(target_w - w, 0)
        if pad_h > 0 or pad_w > 0:
            # np.pad 语法：((上, 下), (左, 右), (通道维度不填充))
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)),
                         mode='constant', constant_values=self.pad_val)
        # 如果原本图像比目标还大，这里[:target_h, :target_w] 起到了左上角裁剪的作用
        return img[:target_h, :target_w]

    def apply_to_mask(self, mask, **params):
        # 对 Mask 执行完全相同的填充逻辑
        h, w = mask.shape[:2]
        target_h, target_w = self.size
        pad_h = max(target_h - h, 0)
        pad_w = max(target_w - w, 0)
        if pad_h > 0 or pad_w > 0:
            # Mask 通常是 2D 的，所以只有前两个维度的 padding
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)),
                          mode='constant', constant_values=self.seg_pad_val)
        return mask[:target_h, :target_w]


class RandomCropWithFilter(A.DualTransform):
    """
    带类别占比过滤机制的随机裁剪 (核心技术点)。
    避免裁剪出一张图全是背景（或全是某一个类别），导致模型训练崩溃或低效。
    """
    def __init__(self, crop_size=512, cat_max_ratio=0.75, max_retries=10, p=1.0):
        super().__init__(p=p)
        self.crop_size = crop_size         # 裁剪尺寸
        self.cat_max_ratio = cat_max_ratio # 单一类别允许的最大面积占比 (0.75 = 75%)
        self.max_retries = max_retries     # 如果不合格，最大重试次数 (10次)

    def apply(self, img, crop_x=0, crop_y=0, **params):
        # 实际执行裁剪图像 (坐标由 get_params_dependent_on_data 提供)
        return img[crop_y:crop_y + self.crop_size, crop_x:crop_x + self.crop_size]

    def apply_to_mask(self, mask, crop_x=0, crop_y=0, **params):
        # 实际执行裁剪 Mask
        return mask[crop_y:crop_y + self.crop_size, crop_x:crop_x + self.crop_size]

    def get_params_dependent_on_data(self, params, data):
        # 核心逻辑：在这个方法里寻找合格的裁剪坐标 (x, y)
        img = data['image']
        mask = data.get('mask')
        h, w = img.shape[:2]
        cs = self.crop_size

        for _ in range(self.max_retries):
            # 随机生成左上角的裁剪点
            y = np.random.randint(0, max(1, h - cs + 1))
            x = np.random.randint(0, max(1, w - cs + 1))

            if mask is not None:
                # 截取这块区域的 Mask 并展平为一维数组
                crop_mask = mask[y:y + cs, x:x + cs].ravel()
                # 找出这块区域包含的所有类别标签
                labels = np.unique(crop_mask)
                # 剔除 ignore_index (255)
                labels = labels[labels != 255]

                if len(labels) > 0:
                    # 统计这块区域除去 255 以外，各个标签出现的像素数量
                    _, counts = np.unique(crop_mask[crop_mask != 255], return_counts=True)
                    # 如果占比最大的那个类别，没有超过总面积的 75%
                    if np.max(counts) < self.cat_max_ratio * cs * cs:
                        return {'crop_x': x, 'crop_y': y} # 合格，返回坐标
                # 如果代码走到这里，说明要么全是 255，要么某个类占比 > 75%，循环重试
            else:
                # 如果没有 Mask（通常是预测阶段，尽管预测通常不调用这个，但为了代码健壮性）
                return {'crop_x': x, 'crop_y': y}

        # 如果重试 10 次都没找到合格的，只能被迫返回最后一次的坐标
        return {'crop_x': x, 'crop_y': y}

    def get_transform_init_args_names(self):
        return ('crop_size', 'cat_max_ratio', 'max_retries')


# ==================== Pipeline 构建函数 ====================

def get_train_transforms(crop_size=512, img_scale=(2048, 512)):
    """
    暴露给外部调用的函数：获取训练用的数据增强 Pipeline
    """
    return A.Compose([
        # 1. 随机缩放：保证短边在 512 的 0.5 到 2.0 倍之间波动
        RandomResize(short_size=img_scale[1], ratio_range=(0.5, 2.0), p=1.0),

        # 2. 补边：如果缩放后图片不足 512x512，则右方和下方补黑边 (防止下一步裁剪报错)
        PadToSize(size=(crop_size, crop_size), pad_val=0, seg_pad_val=255),

        # 3. 随机裁剪：裁出 512x512，如果某个类别占比 >75% 就重裁
        RandomCropWithFilter(crop_size=crop_size, cat_max_ratio=0.75),

        # 4. 随机水平翻转 (50% 概率)
        A.HorizontalFlip(p=0.5),

        # 5. 光度失真 (模拟亮度/对比度/饱和度/色调的变化)
        PhotoMetricDistortion(p=0.5),

        # 6. ImageNet 归一化 (减去均值除以方差，利于模型收敛)
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),

        # 7. 转为 PyTorch 张量格式，并自动将维度从 (H, W, C) 变成 (C, H, W)
        ToTensorV2(),
    ])


def get_val_transforms(img_scale=(2048, 512)):
    """
    暴露给外部调用的函数：获取验证用的数据处理 Pipeline
    验证集不需要任何随机操作（不能随机裁剪、翻转等），确保每次评估指标一致。
    """
    return A.Compose([
        # 1. 保持宽高比缩放：将最短边对齐到 512
        A.SmallestMaxSize(max_size=img_scale[1]),

        # 2. 补边到 32 的倍数
        # 非常关键：SegFormer(MiT编码器) 会进行 4 次跨步为 2 的下采样 (2*2*2*2 = 16，后续有stride处理总计需要32的倍数对齐)
        # 输入图片的宽和高必须能被 32 整除，否则上采样恢复时尺寸对不上会报错！
        A.PadIfNeeded(
            min_height=None, min_width=None,
            pad_height_divisor=32, pad_width_divisor=32, # 要求边长被32整除
            border_mode=0, fill=0, fill_mask=255,        # 补黑边，标签补 255
        ),

        # 3. 归一化
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),

        # 4. 转 Tensor
        ToTensorV2(),
    ])


def get_test_transforms(scales=(1.0,), flip=False, img_scale=(2048, 512)):
    """
    多尺度测试 (TTA) pipeline (测试集使用)。
    测试通常和验证类似，但有可能会传入不同的 scale 甚至需要水平翻转，来融合结果。
    """
    transforms = [
        # 短边缩放
        A.SmallestMaxSize(max_size=img_scale[1]),
        # 对齐 32 倍数
        A.PadIfNeeded(
            min_height=None, min_width=None,
            pad_height_divisor=32, pad_width_divisor=32,
            border_mode=0, fill=0, fill_mask=255,
        ),
        # 归一化
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        # 转 Tensor
        ToTensorV2(),
    ]
    # 如果开启翻转测试，则强制插入翻转操作
    if flip:
        transforms.insert(1, A.HorizontalFlip(p=1.0))
    return A.Compose(transforms)


# ==================== 测试代码块 ====================
if __name__ == "__main__":
    # 当直接运行这个 python 脚本时，会执行这里的测试代码，验证 pipeline 是否报错

    # 造一张假图: 600 高, 800 宽, 3 通道 (RGB)
    dummy_img = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
    # 造一个假 Mask: 600 高, 800 宽, 类别从 0 到 149
    dummy_mask = np.random.randint(0, 150, (600, 800), dtype=np.int64)

    # 1. 验证训练流程
    train_tf = get_train_transforms(crop_size=512)
    result = train_tf(image=dummy_img, mask=dummy_mask)
    # 打印结果尺寸
    print(f"[Train] Image: {result['image'].shape}, Mask: {result['mask'].shape}")
    # 期望输出: Image [3, 512, 512], Mask [512, 512] (3 跑到前面是因为 ToTensorV2 做了轴调换 HWC -> CHW)

    # 2. 验证验证流程
    val_tf = get_val_transforms()
    result = val_tf(image=dummy_img, mask=dummy_mask)
    print(f"[Val]   Image: {result['image'].shape}, Mask: {result['mask'].shape}")
    # 验证流程没有 crop，只有对短边 512 的缩放和 32 倍数的 pad

    print("Transforms pipeline 验证通过。")