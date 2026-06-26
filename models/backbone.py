import logging
from functools import partial

import torch
import torch.nn as nn
from timm.layers import to_2tuple, DropPath


'''
    ⭐1、重叠块嵌入 (Overlapped Patch Embedding)
        核心思想：使用步长小于卷积核大小的卷积（即 CNN 的滑窗机制），在降采样的同时保留局部的连续性。
        作用类似 Swin 里的 PatchMerging，但有重叠。

        实际上在 Segformer 中 由于作者关注patch之间的连续性，因此舍弃了Swin Transformer中 直接将 Patch在通道维度进行硬叠加的操作
        从而改为了使用 Stage1: K = 7, S = 4, P = 3 
                     Stage2 ~ 4: K = 3, S = 2, P = 1


        Stage1: 
                [B,C,H,W] -> [B, H*W/16, embed_dim]    new_H = H/4,   new_W = W/4

        Stage2~4:
                [B,C,H,W] -> [B, H*W/4, embed_dim]     new_H = H/2,   new_W = W/2

        return x, new_H, new_W  
'''
class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_channels=3, embed_dim=768):
        '''
        :param img_size:    当前输入特征图大小
        :param patch_size:  卷积核大小
        :param stride:      卷积步长
        :param in_channels: 当前特征图每个Patch的通道数
        :param embed_dim:   要下采样 输出的通道数
        '''
        super(OverlapPatchEmbed, self).__init__()
        # Embed的作用就是进行下采样，所以要先获取 padding,
        # Stage1: K = 7, S = 4, P = 3            即 第一次下采样 4倍            S=4     img_size / 4
        # Stage2 ~ 4: K = 3, S = 2, P = 1        即 之后三次 每次下采样 2 倍     S=2     img_size / 8 --> img_size / 16 --> img_size / 32
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H_new, self.W_new = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H_new * self.W_new

        # padding 的计算逻辑，保证输出分辨率符合预期
        pad_h = patch_size[0] // 2
        pad_w = patch_size[1] // 2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride, padding=(pad_h, pad_w))
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # [B, C, H, W] -> [B, embed_dim, H_new, W_new]
        x = self.proj(x)

        _, _, H_new, W_new = x.shape

        # [B, embed_dim, H_new, W_new] -> [B, embed_dim, H_new * W_new] -> [B, H_new * W_new, embed_dim]
        x = x.flatten(2).transpose(1, 2)
        x = self.layer_norm(x)

        return x, H_new, W_new


'''
    ⭐2、高效自注意力机制 (Efficient Self-Attention)
        核心操作：因为高分辨率特征图的 N (H*W) 太大，Attention 复杂度 O(N^2) 吃不消。
        这里使用空间缩减比例 (sr_ratio) 直接通过卷积对 K 和 V 进行降采样，将序列长度缩短，从而极大降低计算量。



        [B,N,C] ->  [B,N,C]
'''
class EfficientSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., linear_drop=0., sr_ratio=1):
        super(EfficientSelfAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qk_scale = qk_scale or head_dim ** -0.5

        # 将 Q/K/V/SR/Norm 放入 self 子模块 (对齐 HuggingFace: attention.self.*)
        self.self = nn.Module()
        self.self.query = nn.Linear(dim, dim, bias=qkv_bias)
        self.self.key = nn.Linear(dim, dim, bias=qkv_bias)
        self.self.value = nn.Linear(dim, dim, bias=qkv_bias)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            # 使用卷积对 K 和 V 的空间维度进行降采样 从而减少计算量
            self.self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.self.layer_norm = nn.LayerNorm(dim)

        self.attn_drop = nn.Dropout(p=attn_drop)

        # 输出投影 (对齐 HuggingFace: attention.output.dense)
        self.output = nn.Module()
        self.output.dense = nn.Linear(dim, dim)
        self.output_dropout = nn.Dropout(p=linear_drop)

    def forward(self, x, H, W):
        B, N, C = x.shape

        # 1、获取 Q
        # [B,N,C] --> [B,N,C] --> [B,N,num_heads,head_dim] --> [B,num_heads,N,head_dim]
        Q = self.self.query(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # 2、获取 K 和 V，并根据 sr_ratio 进行序列缩减
        if self.sr_ratio > 1:
            # [B,N,C] -> [B,C,N] -> [B,C,H,W]
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)

            # 减小 kv的序列长度
            # [B,C,H,W] -> [B,C,H/R,W/R] -> [B,H/R,W/R,C] -> [B,H*W/R^2,C]
            x_ = self.self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.self.layer_norm(x_)

            # [B,N/R^2,C] -> [B,N/R^2,num_heads,head_dim]
            K = self.self.key(x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            V = self.self.value(x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        else:
            K = self.self.key(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            V = self.self.value(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # 3、计算注意力
        # [B, num_heads, N, head_dim] @ [B, num_heads, head_dim, N/(sr^2)] = [B, num_heads, N, N/(sr^2)]
        attn = (Q @ K.transpose(-2, -1)) * self.qk_scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # [B, num_heads, N, N/(sr^2)] @ [B, num_heads, N/(sr^2), head_dim] = [B, num_heads, N, head_dim]
        # -> [B, N, num_heads, head_dim] -> [B, N, C]
        attn = (attn @ V).transpose(1, 2).reshape(B, N, C)

        # 4、线性投影
        x = self.output.dense(attn)
        x = self.output_dropout(x)
        return x


'''
    ⭐3、深度可分离卷积模块 (DWConvModule)
        包装 Depth-wise Conv，用于 MixFFN 中注入局部位置信息。
'''
class DWConvModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 3x3 Depth-wise Conv，groups=dim 确保通道互不干扰
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=True)

    def forward(self, x):
        return self.dwconv(x)


'''
    ⭐4、混合前向传播网络 (Mix-FFN)
        核心操作：在传统的两个 Linear 层之间，插入一个 3x3 的深度可分离卷积 (Depth-wise Conv)。
        作者认为，通过 CNN Padding 泄露的位置信息足以替代 Absolute/Relative Positional Encoding，
        因此 SegFormer 不需要显示的位置编码，对不同分辨率的图片泛化能力极强。


         [B,N,C] ->  [B,N,C]
'''
class MixFFN(nn.Module):
    def __init__(self, in_channels, hidden=None, out_channels=None, act_layer=nn.GELU, drop=0.):
        super(MixFFN, self).__init__()

        out_channels = out_channels or in_channels
        hidden = hidden or in_channels

        self.dense1 = nn.Linear(in_channels, hidden)
        self.dwconv = DWConvModule(hidden)
        self.act = act_layer()

        self.dense2 = nn.Linear(hidden, out_channels)

        self.drop = nn.Dropout(p=drop)

    def forward(self, x, H, W):
        B, N, C = x.shape

        # 1、升维: [B, N, C] -> [B, N, hidden_features]
        x = self.dense1(x)

        # 2、转为图像排布: [B, N, hidden_features] -> [B, hidden_features, N] -> [B, hidden_features, H, W]
        x = x.transpose(1, 2).reshape(B, -1, H, W)

        # 3、注入局部位置信息: 3x3 卷积
        x = self.dwconv(x)

        # 4、展平拉回: [B, hidden_features, H, W] -> [B, hidden_features, N] -> [B, N, hidden_features]
        x = x.flatten(2).transpose(1, 2)

        x = self.act(x)
        x = self.drop(x)

        # 5、降维还原: [B, N, hidden_features] -> [B, N, out_features]
        x = self.dense2(x)
        x = self.drop(x)

        return x


'''
    ⭐5、MiT 编码器块 (Block)
        结构：LN + Attention (带缩减) + 残差 + LN + Mix-FFN + 残差
'''
class Block(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, sr_ratio=1, qk_scale=None, attn_drop=0.,
                 linear_drop=0., mlp_ratio=4., act_layer=nn.GELU, norm_layer=nn.LayerNorm, drop_path=0.):
        super(Block, self).__init__()

        self.layer_norm_1 = norm_layer(dim)

        self.attention = EfficientSelfAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, linear_drop, sr_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.layer_norm_2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MixFFN(dim, mlp_hidden_dim, dim, act_layer, linear_drop)

    def forward(self, x, H, W):
        # Attention 阶段
        x = x + self.drop_path(self.attention(self.layer_norm_1(x), H, W))
        # FFN 阶段 (带有3x3 Conv，需要传入 H 和 W)
        x = x + self.drop_path(self.mlp(self.layer_norm_2(x), H, W))
        return x


'''
    ⭐⭐⭐6、完整的 MixVisionTransformer (MiT)
        包含了四个阶段 (Stages)，每个阶段产出不同分辨率的特征图，类似 ResNet 的层级结构。
        默认参数为 SegFormer-B0 模型的配置。
        参数命名对齐官方 HuggingFace transformers 格式，可直接加载预训练权重。
'''
class MixVisionTransformer(nn.Module):
    def __init__(self, img_size=224, in_channels=3, embed_dims=[32, 64, 160, 256],
                 num_heads=[1, 2, 5, 8], qkv_bias=True, qk_scale=None, sr_ratios=[8, 4, 2, 1], attn_drop=0.,
                 linear_drop=0., drop_path=0.1, mlp_ratios=[4, 4, 4, 4], depths=[2, 2, 2, 2],
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 encoder_pretrained=True):
        super(MixVisionTransformer, self).__init__()

        self.depths = depths

        # dpr 为所有的 block 分配逐渐增加的 DropPath 失活率
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]

        # ---------- 4 个 Stage 的 Patch Embedding ----------
        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        in_chs = [in_channels] + list(embed_dims[:-1])

        self.patch_embeddings = nn.ModuleList()
        for i in range(4):
            self.patch_embeddings.append(
                OverlapPatchEmbed(
                    img_size=img_size // (2 ** (i + 1)) if i > 0 else img_size,
                    patch_size=patch_sizes[i],
                    stride=strides[i],
                    in_channels=in_chs[i],
                    embed_dim=embed_dims[i]
                )
            )

        # ---------- 4 个 Stage 的 Transformer Blocks ----------
        self.block = nn.ModuleList()
        for s in range(4):
            stage_blocks = nn.ModuleList()
            for i in range(depths[s]):
                stage_blocks.append(
                    Block(
                        dim=embed_dims[s],
                        num_heads=num_heads[s],
                        qkv_bias=qkv_bias,
                        sr_ratio=sr_ratios[s],
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        linear_drop=linear_drop,
                        mlp_ratio=mlp_ratios[s],
                        norm_layer=norm_layer,
                        drop_path=dpr[sum(depths[:s]) + i]
                    )
                )
            self.block.append(stage_blocks)

        # ---------- 4 个 Stage 的 LayerNorm ----------
        self.layer_norm = nn.ModuleList([norm_layer(embed_dims[i]) for i in range(4)])

        # 如果启用预训练，在默认初始化之后加载 ImageNet-1K 权重 (覆盖默认初始化)
        self.encoder_pretrained = encoder_pretrained
        if encoder_pretrained:
            self.init_weights(pretrained=True)

    # ========== 以下为新增：ImageNet-1K 预训练权重加载 ==========
    def init_weights(self, pretrained=None):

        if pretrained is None or not pretrained:
            return

        import os
        if isinstance(pretrained, bool):
            # 自动推断变体名: 根据 embed_dims[0] 和 depths 判断
            dim0 = self.patch_embeddings[0].proj.out_channels
            if dim0 == 32:
                variant = 'b0'
            else:
                # b1-b5 都是 embed_dims[0]=64，通过 depths 总和区分
                depth_sum = sum(self.depths)
                depth_variant_map = {10: 'b1', 16: 'b2', 28: 'b3', 41: 'b4', 52: 'b5'}
                variant = depth_variant_map.get(depth_sum, 'b1')
            # 使用项目根目录的绝对路径
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            pretrained = os.path.join(project_root, 'pretrained', f'mit_{variant}.pth')

        if not os.path.exists(pretrained):
            logging.warning(f'[MiT] 预训练权重文件不存在: {pretrained}，跳过加载。'
                            f'请从官方渠道下载并放入 pretrained/ 目录。')
            return
        logging.info(f'[MiT] 正在加载预训练权重: {pretrained}')
        checkpoint = torch.load(pretrained, map_location='cpu')

        # HuggingFace 格式的 checkpoint 直接是 state_dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # 去除 segformer.encoder. 前缀，跳过 classifier 分类头
        mapped_state_dict = {}
        skipped_keys = []

        for key, value in state_dict.items():
            if key.startswith('classifier'):
                skipped_keys.append(key)
                continue
            if key.startswith('segformer.encoder.'):
                mapped_state_dict[key.replace('segformer.encoder.', '', 1)] = value

        # 加载权重
        missing, unexpected = self.load_state_dict(mapped_state_dict, strict=False)

        if missing:
            logging.warning(f'[MiT] Missing keys ({len(missing)}): {missing[:10]}...'
                            if len(missing) > 10 else f'[MiT] Missing keys: {missing}')
        if unexpected:
            logging.warning(f'[MiT] Unexpected keys ({len(unexpected)}): {unexpected[:10]}...'
                            if len(unexpected) > 10 else f'[MiT] Unexpected keys: {unexpected}')
        if skipped_keys:
            logging.info(f'[MiT] 已跳过分类头权重 ({len(skipped_keys)}): {skipped_keys[:5]}...')

        loaded_count = len(mapped_state_dict) - len(unexpected)
        logging.info(f'[MiT] 预训练权重加载完成！成功加载 {loaded_count} 个参数。')

    def forward(self, x):
        B = x.shape[0]
        outs = []

        for s in range(4):
            # Patch Embedding: 下采样
            x, H, W = self.patch_embeddings[s](x)
            # Transformer Blocks: 特征交互
            for blk in self.block[s]:
                x = blk(x, H, W)
            # LayerNorm + 转回 BCHW 排布
            x = self.layer_norm[s](x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs
