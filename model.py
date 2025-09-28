import torch
from torch import nn
from utils import FeedForward, DECOUPLED
from performer_pytorch import Performer
import torch.nn.functional as F

class AttnBlock(nn.Module):
    def __init__(self, dim, depth, dropout, attn_dropout, heads = 16, ff_mult = 2):
        super().__init__()
        self.performer = Performer(dim = dim, 
                                   depth = depth, 
                                   heads = heads, 
                                   dim_head = dim // heads, 
                                   causal = False,
                                   ff_mult = ff_mult,
                                   local_attn_heads = 8,
                                   local_window_size = dim // 8,
                                   ff_dropout = dropout,
                                   attn_dropout = attn_dropout,
                                   )

    def forward(self, x):
        B, T, H, W, C = x.shape
        x = x.view(B, -1, C)
        x = self.performer(x)
        x = x.view(B, T, H, W, C)
        return x

class ConvBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        ff_mult = 2,
        dropout = 0.,
        heads = 16,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.conv = DECOUPLED(dim, heads)
        self.ff = FeedForward(dim, ff_mult, dropout)

    def forward(self, x):
        x = x + self.conv(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class MBConv3DWrapper(nn.Module):
    def __init__(self, in_ch, out_ch, expand_ratio=4, se_ratio=0.25, dropout=0.0):
        super().__init__()
        self.mbconv = MBConv(in_ch, out_ch, expand_ratio, 1, se_ratio, dropout)

    def forward(self, x):
        # x: [B, T, H, W, C]
        B, T, H, W, C = x.shape
        x = x.permute(0, 1, 4, 2, 3).contiguous()  # → [B, T, C, H, W]
        x = x.view(B*T, C, H, W)                   # → [B*T, C, H, W]
        x = self.mbconv(x)                         # apply 2D MBConv
        _, C2, H2, W2 = x.shape
        x = x.view(B, T, C2, H2, W2)               # → [B, T, C, H, W]
        x = x.permute(0, 1, 3, 4, 2).contiguous()  # → [B, T, H, W, C]
        return x


class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, expand_ratio=4, stride=1, se_ratio=0.25, dropout=0.0):
        super(MBConv, self).__init__()
        hidden_dim = in_ch * expand_ratio

        # Expansion
        self.expand = nn.Conv2d(in_ch, hidden_dim, kernel_size=1, bias=False) if expand_ratio != 1 else None
        self.bn0 = nn.BatchNorm2d(hidden_dim) if expand_ratio != 1 else None

        # Depthwise
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)

        # Squeeze-and-Excite
        se_hidden = max(1, int(in_ch * se_ratio))
        self.se_reduce = nn.Conv2d(hidden_dim, se_hidden, 1)
        self.se_expand = nn.Conv2d(se_hidden, hidden_dim, 1)

        # Projection
        self.project = nn.Conv2d(hidden_dim, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.dropout = nn.Dropout(dropout)

        self.use_residual = (stride == 1 and in_ch == out_ch)

    def forward(self, x):
        identity = x

        # Expansion
        if self.expand is not None:
            x = F.silu(self.bn0(self.expand(x)))
        else:
            x = identity

        # Depthwise
        x = F.silu(self.bn1(self.dwconv(x)))

        # Squeeze-and-Excite
        se = F.adaptive_avg_pool2d(x, 1)
        se = F.silu(self.se_reduce(se))
        se = torch.sigmoid(self.se_expand(se))
        x = x * se

        # Projection
        x = self.bn2(self.project(x))

        if self.use_residual:
            x = x + identity
        return self.dropout(x)
    
class ResBottleBlock(nn.Module):
    """
    ResNet-50 style bottleneck applied per-frame (2D), plus optional cheap temporal depthwise conv.
    Expects channels-last input: (B, T, H, W, C) and returns same shape.
    """
    def __init__(self, dim, temporal=True, dropout=0.0):
        super().__init__()
        width = max(1, dim // 4)  # bottleneck width

        # Frame-wise spatial bottleneck (on each frame independently)
        self.conv1 = nn.Conv2d(dim, width, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, dim, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(dim)
        self.relu  = nn.ReLU(inplace=True)
        self.drop  = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Optional temporal mixing: depthwise 3x1x1 over T
        self.temporal = temporal
        if temporal:
            self.tdw   = nn.Conv3d(dim, dim, kernel_size=(3,1,1), padding=(1,0,0), groups=dim, bias=False)
            self.tbn   = nn.BatchNorm3d(dim)

    def forward(self, x):  # x: (B, T, H, W, C)
        B, T, H, W, C = x.shape

        # frame-wise 2D bottleneck
        y = x.permute(0,1,4,2,3).reshape(B*T, C, H, W)  # (B*T, C, H, W)
        identity = y
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        y = self.relu(y + identity)
        y = self.drop(y)
        y = y.view(B, T, C, H, W).permute(0,1,3,4,2)     # back to (B, T, H, W, C)

        # optional temporal depthwise conv
        if self.temporal:
            z = y.permute(0,4,1,2,3)                     # (B, C, T, H, W)
            z = self.tdw(z)
            z = self.tbn(z)
            z = F.relu(z, inplace=True)
            z = z.permute(0,2,3,4,1)                     # (B, T, H, W, C)
            y = y + z                                    # residual add

        return y

# main class

class Model(nn.Module):
    def __init__(
        self,
        *,
        dropout = 0.2,
        attn_dropout = 0.1,
        ff_mult = 4,
        dims = (192, 128),
        depths = (3, 3),
        block_types = ('c', 'a')
    ):
        dims = dims
        depths = depths
        block_types = block_types
        super().__init__()
        self.init_dim, *_, last_dim = dims

        self.stages = nn.ModuleList([])

        for ind, (depth, block_types) in enumerate(zip(depths, block_types)):
            is_last = ind == len(depths) - 1
            stage_dim = dims[ind]
            
            if block_types == "c":
                for _ in range(depth):
                    self.stages.append(
                        ConvBlock(dim=stage_dim, ff_mult=ff_mult, dropout=dropout)
                    )
            elif block_types == "a":
                for _ in range(depth):
                    self.stages.append(
                        AttnBlock(stage_dim, 1, dropout, attn_dropout, ff_mult=ff_mult)
                    )
            elif block_types == "r":   # NEW: ResNet-style bottleneck
                for _ in range(depth):
                    self.stages.append(
                        ResBottleBlock(dim=stage_dim, temporal=True, dropout=dropout)
                    )
            elif block_types == "e":   # NEW: EfficientNet-style block
                print("EEEEEEEEEEEEEEEEEEE EfficientNet Block")
                for _ in range(depth):
                    self.stages.append(
                        MBConv3DWrapper(stage_dim, stage_dim, expand_ratio=4, se_ratio=0.25, dropout=dropout)
                    )

            else:
                raise ValueError(f"Unknown block type: {block_types}")                
            if not is_last:
                self.stages.append(
                    nn.Sequential(
                        nn.LayerNorm(stage_dim),
                        nn.Linear(stage_dim, dims[ind + 1])
                    )
                )

        self.norm0 = nn.LayerNorm(192)
        self.linear = nn.Linear(192, dims[0])
        self.norm = nn.LayerNorm(last_dim)
        self.fc = nn.Linear(last_dim, 1)
        self.drop_out = nn.Dropout(dropout)
        self.pooling = nn.AdaptiveMaxPool3d((1, 1, 1))
        
    def forward(self, x):

        x = x.permute(0, 2, 3, 4, 1)

        if x.shape[4] != self.init_dim:
            x = self.linear(self.norm0(x))

        for stage in self.stages:
            x = stage(x)

        
        x = x.permute(0, 4, 1, 2, 3)
        x = self.pooling(x).squeeze()

        
        x = self.drop_out(x)
        x = self.norm(x)
        logits = self.fc(x)
        return logits, x
