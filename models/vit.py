import os
import re

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F



class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5  # スケーリング係数

        # QKV変換用の線形層
        self.qkv = nn.Linear(dim, dim * 3)

        # 出力投影用の線形層
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape  # バッチサイズ、トークン数、次元

        # QKV変換
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # それぞれ [B, heads, N, head_dim]

        # 注意スコア計算
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, N, N]
        attn = attn.softmax(dim=-1)  # softmax適用

        # 加重和計算
        x = (attn @ v).transpose(1, 2)  # [B, N, heads, head_dim]
        x = x.reshape(B, N, C)  # [B, N, dim]

        # 出力投影
        x = self.proj(x)

        return x



class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop_path=0.):
        super().__init__()
        # Layer Normalization 1
        self.norm1 = nn.LayerNorm(dim)

        # マルチヘッド自己注意機構
        self.attn = MultiHeadAttention(dim, num_heads)

        # Layer Normalization 2
        self.norm2 = nn.LayerNorm(dim)

        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x):
        # 自己注意機構 + 残差接続
        x = x + self.attn(self.norm1(x))

        # MLP + 残差接続
        x = x + self.mlp(self.norm2(x))

        return x


class ViT(nn.Module):
    def __init__(self,
                 num_classes, in_channels=3,
                 grid_size=16, patch_size=16,
                 embed_dim=768, num_heads=12, depth=12,
                 mlp_ratio=4.0,
                 dynamic_img_size=False,
                 ):
        super().__init__()
        self.grid_size = grid_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth
        self.dynamic_img_size = dynamic_img_size

        self.img_size = grid_size * patch_size
        self.n_patches = grid_size ** 2

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))

        # Transformerエンコーダー
        self.transformer = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        # Layer Norm
        self.norm = nn.LayerNorm(embed_dim)

        # 分類ヘッド
        self.mlp_head = nn.Linear(embed_dim, num_classes)

        self._init_weights()


    def _init_weights(self):
        # Initialize position embeddings and class token
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Apply to all parameters
        self.apply(self._init_weights_recursive)


    def _init_weights_recursive(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def interpolate_pos_encoding(self, x_size):
        # 入力サイズからグリッドサイズを計算
        new_grid_h = x_size[2] // self.patch_size
        new_grid_w = x_size[3] // self.patch_size

        # クラストークン部分を除いて位置エンコーディングを取得
        # [B, N+1, C] -> [B, H, W, C]
        pos_tokens = self.pos_embed[:, 1:, :].reshape(1, self.grid_size, self.grid_size, -1)

        # 新しいグリッドサイズへ補間（双三次補間）
        pos_tokens = F.interpolate(
            pos_tokens.permute(0, 3, 1, 2),  # [B, C, H, W]
            size=(new_grid_h, new_grid_w),
            mode='bicubic',
            align_corners=False
        )

        # 元の形状に戻す
        # [B, H, W, C] -> [B, N, C]
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(
            1, new_grid_h * new_grid_w, -1
        )

        # クラストークンと結合して返す
        # [B, N, C] -> [B, N+1, C]
        return torch.cat([self.pos_embed[:, :1, :], pos_tokens], dim=1)


    def forward(self, x, activate=False):
        B, C, H, W = x.shape

        if self.dynamic_img_size and (H != self.img_size or W != self.img_size):
            pos_embed = self.interpolate_pos_encoding(x.shape)
        else:
            pos_embed = self.pos_embed # [B, N+1, embed_dim]

        x = self.proj(x) # [B, H*W, embed_dim]
        x = x.flatten(2)  # [B, embed_dim, H*W]
        x = x.transpose(1, 2)  # [B, H*W(=N), embed_dim]

        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, N+1, embed_dim]
        x = x + pos_embed

        for block in self.transformer:
            x = block(x)

        # 正規化
        x = self.norm(x)

        x = x[:, 0]  # クラストークンの表現のみ抽出 [B, embed_dim]
        x = self.mlp_head(x)  # [B, num_classes]

        if activate:
            if self.num_classes > 1:
                x = torch.softmax(x, dim=-1)
            else:
                x = torch.sigmoid(x)

        return x


def create_vit(
        variant:str,
        num_classes:int,
        in_channels=3,
        grid_size=16,
        patch_size=16,
        mlp_ratio=4.0,
        dynamic_img_size=True
        ):
    variant_params = {
        'tiny': {
            'embed_dim': 192,
            'num_heads': 3,
            'depth': 12,
        },
        'small': {
            'embed_dim': 384,
            'num_heads': 6,
            'depth': 12,
        },
        'base': {
            'embed_dim': 768,
            'num_heads': 12,
            'depth': 12,
        },
        'large': {
            'embed_dim': 1024,
            'num_heads': 16,
            'depth': 24,
        }
    }[variant]

    # 共通パラメータと特定バリアントのパラメータを結合
    params = {
        'grid_size': grid_size,
        'in_channels': in_channels,
        'patch_size': patch_size,
        'mlp_ratio': mlp_ratio,
        'dynamic_img_size': dynamic_img_size,
        **variant_params
    }

    return ViT(num_classes, **params)
