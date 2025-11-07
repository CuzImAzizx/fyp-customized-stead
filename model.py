import torch
import torch.nn as nn
import torch.nn.functional as F
from performer_pytorch import SelfAttention

# ---------------------------
# Block Definitions
# ---------------------------

class ConvBlock(nn.Module):
    def __init__(self, dim, ff_mult=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.conv = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim, dim, kernel_size=1),
            nn.BatchNorm3d(dim),
        )
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * dim, dim)
        )

    def forward(self, x):
        B, T, H, W, C = x.shape
        y = self.norm1(x)
        y = y.permute(0, 4, 1, 2, 3).contiguous()  # B,C,T,H,W
        y = self.conv(y)
        y = y.permute(0, 2, 3, 4, 1).contiguous()  # B,T,H,W,C
        x = x + y

        y = self.norm2(x)
        y = self.ff(y)
        return x + y

class AttnBlock(nn.Module):
    def __init__(self, dim, heads=1, dropout=0.0, attn_dropout=0.0, ff_mult=4):
        super().__init__()
        self.attn = SelfAttention(dim, heads=heads, dropout=attn_dropout)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * dim, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        B, T, H, W, C = x.shape
        y = self.norm(x)
        y = y.reshape(B, T*H*W, C)
        y = self.attn(y)
        y = y.reshape(B, T, H, W, C)
        x = x + y
        y = self.norm2(x)
        return x + self.ff(y)

class ResBottleBlock(nn.Module):
    def __init__(self, dim, temporal=True, dropout=0.0):
        super().__init__()
        width = max(1, dim // 4)
        self.conv1 = nn.Conv2d(dim, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, dim, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.temporal = temporal
        if temporal:
            self.tdw = nn.Conv3d(dim, dim, kernel_size=(3,1,1), padding=(1,0,0), groups=dim, bias=False)
            self.tbn = nn.BatchNorm3d(dim)

    def forward(self, x):
        B, T, H, W, C = x.shape
        y = x.permute(0,1,4,2,3).reshape(B*T, C, H, W)
        identity = y
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        y = self.relu(y + identity)
        y = self.drop(y)
        y = y.view(B, T, C, H, W).permute(0,1,3,4,2)

        if self.temporal:
            z = y.permute(0,4,1,2,3)
            z = self.tdw(z)
            z = self.tbn(z)
            z = F.relu(z, inplace=True)
            z = z.permute(0,2,3,4,1)
            y = y + z

        return y

class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, expand_ratio=4, stride=1, se_ratio=0.25, dropout=0.0):
        super(MBConv, self).__init__()
        hidden_dim = in_ch * expand_ratio
        self.expand = nn.Conv2d(in_ch, hidden_dim, kernel_size=1, bias=False) if expand_ratio != 1 else None
        self.bn0 = nn.BatchNorm2d(hidden_dim) if expand_ratio != 1 else None
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        se_hidden = max(1, int(in_ch * se_ratio))
        self.se_reduce = nn.Conv2d(hidden_dim, se_hidden, 1)
        self.se_expand = nn.Conv2d(se_hidden, hidden_dim, 1)
        self.project = nn.Conv2d(hidden_dim, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.use_residual = (stride == 1 and in_ch == out_ch)

    def forward(self, x):
        identity = x
        if self.expand is not None:
            x = F.silu(self.bn0(self.expand(x)))
        x = F.silu(self.bn1(self.dwconv(x)))
        se = F.adaptive_avg_pool2d(x, 1)
        se = F.silu(self.se_reduce(se))
        se = torch.sigmoid(self.se_expand(se))
        x = x * se
        x = self.bn2(self.project(x))
        if self.use_residual:
            x = x + identity
        return self.dropout(x)

class MBConv3DWrapper(nn.Module):
    def __init__(self, in_ch, out_ch, expand_ratio=4, se_ratio=0.25, dropout=0.0):
        super().__init__()
        self.mbconv = MBConv(in_ch, out_ch, expand_ratio, 1, se_ratio, dropout)

    def forward(self, x):
        B, T, H, W, C = x.shape
        x = x.permute(0, 1, 4, 2, 3).contiguous().view(B*T, C, H, W)
        x = self.mbconv(x)
        C2, H2, W2 = x.shape[1:]
        x = x.view(B, T, C2, H2, W2).permute(0,1,3,4,2).contiguous()
        return x

# ---------------------------
# TAM Block
# ---------------------------

class TAMBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, reduction=16):
        super().__init__()
        self.temporal_conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(dim, dim // reduction)
        self.fc2 = nn.Linear(dim // reduction, dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, T, H, W, C = x.shape
        x_reshaped = x.permute(0, 4, 1, 2, 3)
        x_gap = x_reshaped.mean(dim=[3, 4])
        y = self.temporal_conv(x_gap)
        y = self.bn(y)
        y = self.relu(y)
        y = y.mean(dim=-1)
        y = self.fc2(F.relu(self.fc1(y)))
        y = self.sigmoid(y).view(B, C, 1, 1, 1)
        x_mod = x_reshaped * y
        return x_mod.permute(0, 2, 3, 4, 1)
    
    
class ResNetTAMBlock(nn.Module):
    """
    Composed block: ResBottleBlock -> BatchNorm3d -> ReLU -> TAMBlock.
    Keeps shapes consistent: internal blocks expect [B,T,H,W,C].
    Use this when you want "ResNet -> BatchNorm -> ReLU -> TAM".
    """
    def __init__(self, dim, temporal=True, dropout=0.0):
        super().__init__()
        # use your existing ResBottleBlock (frame-wise spatial bottleneck + temporal depthwise conv)
        self.res = ResBottleBlock(dim=dim, temporal=temporal, dropout=dropout)
        # BatchNorm3d operates on shape [B, C, T, H, W]
        self.bn3d = nn.BatchNorm3d(dim)
        self.relu = nn.ReLU(inplace=True)
        # use existing TAMBlock (expects [B, T, H, W, C])
        self.tam = TAMBlock(dim)

    def forward(self, x):  # x: [B, T, H, W, C]
        # 1) ResBottleBlock (returns [B, T, H, W, C])
        x = self.res(x)

        # 2) BatchNorm3d requires [B, C, T, H, W]
        x_bn = x.permute(0, 4, 1, 2, 3).contiguous()  # -> [B, C, T, H, W]
        x_bn = self.bn3d(x_bn)
        x_bn = self.relu(x_bn)

        # back to [B, T, H, W, C]
        x = x_bn.permute(0, 2, 3, 4, 1).contiguous()

        # 3) TAMBlock (applies temporal adaptive gating, returns [B, T, H, W, C])
        x = self.tam(x)

        return x
    
    
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1, ff_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * dim, dim)
        )

    def forward(self, x):
        # x: [B, T, H, W, C]
        B, T, H, W, C = x.shape
        x_ = x.reshape(B, T * H * W, C)  # flatten space into tokens
        y = self.norm1(x_)
        attn_out, _ = self.attn(y, y, y)
        x_ = x_ + attn_out
        y = self.norm2(x_)
        y = self.ff(y)
        out = x_ + y
        out = out.reshape(B, T, H, W, C)
        return out

# ---- ResNet + Transformer composite ----
class ResNetTransformerBlock(nn.Module):
    """
    ResNet -> BatchNorm3d -> ReLU -> Transformer
    """
    def __init__(self, dim, temporal=True, dropout=0.0, num_heads=4, ff_mult=4):
        super().__init__()
        self.res = ResBottleBlock(dim=dim, temporal=temporal, dropout=dropout)
        self.bn3d = nn.BatchNorm3d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.transformer = TransformerBlock(dim=dim, num_heads=num_heads, dropout=dropout, ff_mult=ff_mult)

    def forward(self, x):  # [B, T, H, W, C]
        x = self.res(x)
        x_bn = x.permute(0, 4, 1, 2, 3).contiguous()
        x_bn = self.bn3d(x_bn)
        x_bn = self.relu(x_bn)
        x = x_bn.permute(0, 2, 3, 4, 1).contiguous()
        x = self.transformer(x)
        return x

# ---------------------------
# Model
# ---------------------------

class Model(nn.Module):
    def __init__(self, dims=(32, 64), depths=(2, 2), block_types=('c', 'a'),
                 dropout=0.0, attn_dropout=0.0, ff_mult=4):
        super().__init__()

        self.norm = nn.LayerNorm(192)
        self.proj = nn.Linear(192, dims[0])
        self.stages = nn.ModuleList()

        for depth, stage_dim, block_type in zip(depths, dims, block_types):
            
            prev_dim = dims[0]
            if stage_dim != prev_dim:
                self.stages.append(nn.Linear(prev_dim, stage_dim))
            prev_dim = stage_dim
            
            if block_type == "c":
                for _ in range(depth):
                    self.stages.append(ConvBlock(dim=stage_dim, ff_mult=ff_mult, dropout=dropout))
            elif block_type == "a":
                for _ in range(depth):
                    self.stages.append(AttnBlock(stage_dim, 1, dropout, attn_dropout, ff_mult=ff_mult))
            elif block_type == "r":
                for _ in range(depth):
                    self.stages.append(ResBottleBlock(dim=stage_dim, temporal=True, dropout=dropout))
            elif block_type == "e":
                print("EEEEEEEEEEEEEEEEEEE EfficientNet Block")
                for _ in range(depth):
                    self.stages.append(
                        MBConv3DWrapper(stage_dim, stage_dim, expand_ratio=4, se_ratio=0.25, dropout=dropout)
                    )
            elif block_type == "t":
                print("TTTTTTTTTTTTTTTTTTTTTTTTTTT TAM Block")
                for _ in range(depth):
                    self.stages.append(TAMBlock(stage_dim))
            elif block_type == "rt":
                # ResNet -> BatchNorm -> ReLU -> TAM
                print("RRRRRRRRRRRR ResNet -> BatchNorm -> ReLU -> TAM")
                for _ in range(depth):
                    self.stages.append(ResNetTAMBlock(stage_dim, temporal=True, dropout=dropout))
            elif block_type == "rtr":
                print("RRRRRRRRRRRR ResNet -> BatchNorm -> ReLU -> Transformer")
                for _ in range(depth):
                    self.stages.append(ResNetTransformerBlock(stage_dim, temporal=True, dropout=dropout))
                    
            else:
                raise ValueError(f"Unknown block type: {block_type}")

        self.pool = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout)
        self.head_norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], 1)

    def forward(self, x):
        # Convert from [B, C, T, H, W] → [B, T, H, W, C]
        x = x.permute(0, 2, 3, 4, 1).contiguous()
    
        x = self.norm(x)
        x = self.proj(x)
    
        for stage in self.stages:
            x = stage(x)
    
        # Back to [B, C, T, H, W] for pooling
        x = x.permute(0, 4, 1, 2, 3)
        x = self.pool(x).squeeze(-1).squeeze(-1).squeeze(-1)
        x = self.dropout(self.head_norm(x))
        out = self.head(x)
        return out, x