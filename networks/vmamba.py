import time
import math
import os
import cv2
import numpy as np
from functools import partial
from typing import Optional, Callable
from rms_norm import RMSNorm
from rotary import apply_rotary_emb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from differential_transformer import MultiheadDiffAttn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
try:
    # from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm (in which causal_conv1d is needed)
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    import numpy as np
    
    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop
    

    assert not with_complex

    flops = 0 # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """
    
    in_for_flops = B * D * N   
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops 
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """
    
    return flops

"""
 1. 核心架构逻辑将原有的单路径 VM-UNet 修改为 SSM-CNN 双流并行编码器 结构 ：低频路径（LF Branch）：
    采用原论文的 VSS Block ，利用 Mamba 的线性复杂度和长程建模能力捕捉肋骨的全局解剖结构 。
    高频路径（HF Branch）：将 VSS Block 替换为 卷积层（CNN/ResBlock），利用卷积的局部归纳偏置精准捕捉肋骨边缘和细微骨折线。
2. 图像预处理（拉普拉斯分频）在模型输入 forward 的第一步进行空间域分频：输入：$448 \times 448$ 原始图像。
    LF 图像：通过高斯模糊后下采样，生成 $224 \times 224$ 图像。
    HF 图像：将 LF 上采样回 $448$ 后与原图相减，得到 $448 \times 448$ 的残差边缘图。
3. 编码器设计（双流并行与对齐）LF 编码器 (VSS 路径)：输入：$224 \times 224$。参数：patch_size=2, stride=2 。特征图流：$112 \rightarrow 56 \rightarrow 28 \rightarrow 14$。
    HF 编码器 (CNN 路径)：输入：$448 \times 448$。参数：patch_size=4, stride=4（通过第一层大步长卷积或 PatchEmbed 实现对齐）。特征图流：$112 \rightarrow 56 \rightarrow 28 \rightarrow 14$。
    对齐意义：确保两路在每一个 Stage 的特征图尺寸和通道数（96, 192, 384, 768）完全一致 。
4. 跳跃连接与融合策略 (Skip Connection)采用 “先融合，后跳跃” 的极简逻辑，符合 VM-UNet 减少额外参数的设计思路 ：
    Stage 内融合：在 Encoder 的每个 Stage 结束时，将 LF 路特征与 HF 路特征直接进行 元素级相加（Element-wise Addition）。
    跳跃连接传递：将相加后的融合特征图作为 skip_connection 传递给 Decoder 。
Decoder 接收：Decoder 将上采样后的特征与融合后的 Skip 特征再次相加 

"""

class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer.1. Default: None
       
    PatchEmbed2D：图像 → Patch token（用卷积实现“切块+线性投影”）
    """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        
        # 使用2D卷积通过步长的方式将图像进行分块 + 线性投影（投影的参数保持不变）用卷积实现速度更快
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 如果传了 nn.LayerNorm：
        # 对最后一维 embed_dim 做 LayerNorm（这就是为什么要 permute 到 [B,H,W,C]，LayerNorm默认对最后维做）
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    # x: [B, C, H, W] ->(proj(x)) [B, embed_dim, H/ps, W/ps] （这里 ps=patch_size） -> [B, H/ps, W/ps, embed_dim]
    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    
    PatchMerging2D：2×2 合并下采样（空间减半、通道翻倍）
    这是 Swin 里的经典 Patch Merging：把 2×2 四个位置的 token 拼在一起，然后线性降维。
    空间分辨率减半,通道数翻倍（C → 2C）
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        # 它取四个子采样网格：x0：偶行偶列, x1：奇行偶列, x2：偶行奇列, x3：奇行奇列
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        # SHAPE_FIX 在干嘛（奇数尺寸处理）如果 H 或 W 是奇数，会出现四个分支尺寸不一致的问题。强行裁掉最后多出来的一行/一列，保证能拼起来。
        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
        
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C ,在最后一个维度 concate
        # 在奇数裁剪场景下用 H//2, W//2 重新规整一下形状。
        x = x.view(B, H//2, W//2, 4 * C)  # B H/2*W/2 4*C
        
        # LayerNorm(4C)
        x = self.norm(x)
        # Linear(4C -> 2C)（也就是 self.reduction）
        x = self.reduction(x)

        return x
    
"""  
PatchExpand2D：上采样 ×2（像素重排/PixelShuffle 思想）
通常用于解码器，把 [B,H,W,C] 放大到 [B,2H,2W,C/2]（空间×2，通道减小），实现“反 PatchMerging”。
输入 x: [B, H, W, C] -> 输出：[B, 2H, 2W, C/2]
"""
class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        # 这里的 dim 被作者当成“上采样之后的通道数（目标通道）”，而不是“输入通道数”。
        # 把输入 (B, H, W, 2·dim) 上采样成 (B, 2H, 2W, dim)
        self.dim = dim*2
        # 上采样倍数
        self.dim_scale = dim_scale
        # 通过线性层直接上采样
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        # 用于高效地重排张量的维度。它支持通过简单的表达式日实现复杂的维度变换，通常用于深度学习中的数据预处理或模型输入/输出的变换，例如重新划分张量维度，可以实现数组的转置、拆分、合并等操作。
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        x= self.norm(x)

        return x
    
""" 
Final_PatchExpand2D：最后一次上采样 ×4（恢复到原图尺度）
"""
class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        # 放缩比例为4倍
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        x= self.norm(x)

        return x

"""
mamba块
SS2D 是什么：把 1D 的 Mamba/SSM 扫描扩展到 2D 图像
"""
class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        # d_state="auto", # 20240109
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        # 输入/输出的通道维（token embedding dim）
        self.d_model = d_model
        # SSM 的状态维（每个通道的状态长度 N），越大表达力越强但更耗算
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        # 内部扩展倍率
        self.expand = expand
        # 内部维度
        self.d_inner = int(self.expand * self.d_model)

        # 生成步长 Δt 的低秩维度
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # 输入投影 + 门控
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        # 深度可分离卷积,每个通道单独卷积，不混通道
        # 作用：在长程 scan 之前先注入一点局部纹理能力（非常常见）
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        # 激活函数
        self.act = nn.SiLU()

        # 每个方向都有一套 x_proj（参数不共享）
        # 但作者不保留 4 个 Linear 模块，而是把权重堆成一个张量 x_proj_weight，后面用 einsum 一次算完（更快、更省 Python 开销）
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        # 每个通道，每个方向都有自己的dt
        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        
        # 对角转移矩阵A ，每一个方向一个
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        # 门控 D
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        # 输出处理
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod  # 装饰器把类里的一个函数变成静态方法：不需要 self（实例）参数,也不需要 cls（类）参数. 调用时可以用 类名.方法() 或 实例.方法()，但它不会自动传入任何对象
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        # mamba 的 selective_scan_fn
        self.selective_scan = selective_scan_fn
        
        B, C, H, W = x.shape
        # 把二维展平成一维序列长度
        L = H * W
        # 四个扫描方向
        K = 4

        #k = 1：WH  k=0：HW 正向（左→右，行优先）, 正向（上→下，列优先）, k = 2：HW 反向, k = 3：WH 反向
        # view 等价于 reshape, transpose() 交换两个维度, transpose() 转置后张量的内存布局通常不是连续的，直接 .view(...) 会报错或得到错误结果。.contiguous() 会拷贝一份连续内存，保证后面 view 安全。 stack([...], dim=1) 把两条序列堆起来，增加一个“方向维”。
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        # flip(L, dims=[-1]) 把序列维 L 反过来（时间反向/序列反向）。实现 4个方向
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        # 对每个方向 k，都做一次“线性层”投影（不带 bias）(做矩阵乘法)。 xs.view [b k d l], self.x_proj_weight [k c d], output [b k c l]
        # 本质上做了 4 个不同的 Linear（每个方向一套权重），作者用 einsum + stack权重 的方式把 4 个 Linear 合成一次大计算，减少 Python 模块开销。
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        
        # 把 x_dbl 切成三段(在维度2上)  dts：[B, K, dt_rank, L] （生成步长 Δt 的低秩表示）, Bs：[B, K, d_state, L] （SSM 的 B 参数，随位置变化）, Cs：[B, K, d_state, L] （SSM 的 C 参数，随位置变化）
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        # 把低秩的 dt 表达扩展为“每个通道都有自己的 dt”，更灵活。
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        # 整理形状 + 转 float32（为了 scan 稳定）
        # 因为 selective_scan_fn 往往是按 “通道维并行” 扫描的：把方向 K 和通道 C 合并成一个大通道轴，统一做一次扫描。
        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        # 门控, 每个（方向×通道）一个 D（skip/直通系数）
        Ds = self.Ds.float().view(-1) # (k * d)
        # 构造 SSM 的状态转移参数 A
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        # 每个（方向×通道）一个 dt bias（后面配合 softplus 得到正的 dt）
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        
        # 把反向扫描的输出再 flip 回来，使它们的第 t 个位置对应原来正向序列的第 t 个位置。
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        # 把 WH 展平的序列转回 HW 展平
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        # 返回四个方向的结果,HW 正向扫描结果  HW 反向扫描（翻回后）  WH 正向扫描（转回 HW 顺序）  WH 反向扫描（翻回并转回）
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y


    def forward(self, x: torch.Tensor, **kwargs):
        # print(f"x shape before unpacking: {x.shape}")
        B, H, W, C = x.shape
        xz = self.in_proj(x)  # [B,H,W,2*d_inner]
        # 将xz分块
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)  x,z: [B,H,W,d_inner]
        x = x.permute(0, 3, 1, 2).contiguous()  # [B,d_inner,H,W]
        x = self.act(self.conv2d(x)) # (b, d, h, w)
        
        # 得到 mamba 的四个方向的结果
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        # layer 归一化
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        
        # mamba 层输出
        return out


# 一个“Pre-LN + SS2D + 残差”的基本块（相当于 Transformer Block 的注意力子层）
class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        # 归一化层的“构造器”，默认是 LayerNorm，并且 eps 设为 1e-6。
        # Callable[..., nn.Module] 表示：你传进来的东西“能被调用并返回一个 nn.Module
        # partial(nn.LayerNorm, eps=1e-6) 相当于固定住 eps 参数，之后只需要传 normalized_shape：
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x

# 编码器 多个VSSBlock + downsample 组合而成
class VSSLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        downsample=None, 
        use_checkpoint=False, 
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None


    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        
        if self.downsample is not None:
            x = self.downsample(x)

        return x
    

# 解码器 多个VSSBlock + upsample 组合而成
class VSSLayer_up(nn.Module):
    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        upsample=None, 
        use_checkpoint=False, 
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)
        ])
        
        if True:
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x, skip=None):
        # 1) 先上采样
        if self.upsample is not None:
            x = self.upsample(x)
        # 2) 再融合 skip
        if skip is not None:
            x = x + skip
        # 3) 最后做 block refine
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x
    



# Resnet 块,数据的形状保持不变
class ResBlock(nn.Module):
    def __init__(self, dim, drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=1) 
        self.act = nn.GELU()
        self.norm2 = norm_layer(dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # x: B, H, W, C
        # 保存残差
        shortcut = x
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2) # B, C, H, W
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1) # B, H, W, C
        x = self.act(x)
        x = self.norm2(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1)
        x = shortcut + self.drop_path(x)
        return x

# 堆 ResBlock + 可选下采样
class CNNLayer(nn.Module):
    def __init__(self, dim, depth, drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, **kwargs):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            ResBlock(dim=dim, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample is not None else None
    
    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class GuidedAttentionFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 优化：为了节省显存，不直接使用 concat+conv(2C->1)，而是拆分为两个 Conv(C->1) 相加
        # 数学上 Conv(concat(a,b)) == Conv_a(a) + Conv_b(b) + bias
        # 输出形状不变,得到的是门控权重
        self.conv_lf = nn.Conv2d(dim, 1, kernel_size=3, padding=1, bias=True)
        self.conv_hf = nn.Conv2d(dim, 1, kernel_size=3, padding=1, bias=False) # 只需要一个bias
        self.act = nn.Sigmoid()

    def forward(self, x_lf, x_hf):
        # x_lf, x_hf: B, H, W, C
        B, H, W, C = x_lf.shape
        
        # 显存优化策略：
        # 1. 避免 torch.cat 创建 2C 大小的张量副本
        # 2. 分别处理 permute，减小峰值显存
        
        # 处理 LF 分支
        # permute 返回的是 view，但 conv2d 可能在内部需要 contiguous
        # 我们这里显式 permute，但不调用 contiguous()，让 PyTorch 的 conv2d 自动处理 stride
        # 如果 conv2d 不支持非连续内存，它会自己构建副本，但此时我们没有持有中间的 cat 张量
        x_lf_p = x_lf.permute(0, 3, 1, 2) # B, C, H, W
        mask_lf = self.conv_lf(x_lf_p)
        
        # 处理 HF 分支
        x_hf_p = x_hf.permute(0, 3, 1, 2) # B, C, H, W
        mask_hf = self.conv_hf(x_hf_p)
        
        # 相加生成 Mask (B, 1, H, W),过激活函数是为了保障[0,1]
        mask = self.act(mask_lf + mask_hf)
        
        # 恢复维度 B, H, W, 1
        mask = mask.permute(0, 2, 3, 1)

        # 融合计算: out = x_lf + x_hf * mask
        # 使用 In-place 操作减少显存分配
        # 这里的计算逻辑：
        # 1. x_hf * mask -> 生成一个临时张量 (B, H, W, C)
        # 2. + x_lf
        out = x_hf * mask
        # 为什么不给 x_lf 上 mask ? 感觉谁上都差不多
        out.add_(x_lf) # In-place add
        return out


class HighFrequencyGuidedUpsampling(nn.Module):
    """
    高频特征引导的联合上采样模块 (High-Frequency Guided Joint Upsampling)
    利用原图的高频线索 (如边缘、纹理) 来引导低分辨率的 Logits 上采样到原图尺寸，
    从而获得更锐利和准确的边界预测。
    """
    def __init__(self, in_channels, guide_channels=3, compress_dim=16):
        super().__init__()
        # 联合特征通道压缩与对齐
        self.align = nn.Sequential(
            nn.Conv2d(in_channels + guide_channels, compress_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(compress_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(compress_dim, compress_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(compress_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(compress_dim, in_channels, kernel_size=1)
        )
        
        # 零初始化残差连接的最后一层，确保未训练时等效于纯双线性插值，不破坏早期收敛
        nn.init.normal_(self.align[-1].weight, std=0.001)
        if self.align[-1].bias is not None:
            nn.init.constant_(self.align[-1].bias, 0)
            
    def forward(self, logits_small, guide_high_res):
        # 先用双线性插值将小图放大到大图尺寸
        logits_up = F.interpolate(logits_small, size=guide_high_res.shape[2:], mode='bilinear', align_corners=False)
        # 将放大后的 logits 与原图高频线索拼接
        fused = torch.cat([logits_up, guide_high_res], dim=1)
        # 预测残差并加上双线性插值的结果
        return logits_up + self.align(fused)


class Uncertainty_Guide_Enhancement(nn.Module):
    """
    专为医学图像 (细长连续结构) 改造的 UARB 模块。
    核心思想：全图视野学习 + 晚期软门控 + 局部拓扑保护 (2D CNN 代替 Mamba)
    """
    def __init__(self, dim, num_classes=31, d_state=None):
        super().__init__()
        # 接收 d_state 参数以兼容你原来的代码调用，但在这里我们不用它
        
        # 1. 局部拓扑保护：使用 2D 卷积替代 1D Mamba 扫描
        self.compress_dim = 16
        self.refine_net = nn.Sequential(
            # 通道压缩
            nn.Conv2d(dim, self.compress_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.compress_dim),
            nn.ReLU(inplace=True),
            
            # 3x3 卷积，专门捕捉 2D 连续边缘，保证骨骼不断裂
            nn.Conv2d(self.compress_dim, self.compress_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.compress_dim),
            nn.ReLU(inplace=True),
            
            # 输出矫正 Logits
            nn.Conv2d(self.compress_dim, num_classes, kernel_size=1)
        )

        # 2. 残差零初始化 (Zero-Initialization)
        # 保证模型初期输出全 0，不干扰主干网络的健康收敛
        nn.init.normal_(self.refine_net[-1].weight, mean=0.0, std=0.001)
        if self.refine_net[-1].bias is not None:
            nn.init.constant_(self.refine_net[-1].bias, 0.0)

    def forward(self, features: torch.Tensor, uncertainty: torch.Tensor):
        # features: [B, H, W, C] (channels-last)
        # uncertainty: [B, 1, H, W]
        
        # 转换为标准的卷积输入格式 [B, C, H, W]
        x_bchw = features.permute(0, 3, 1, 2).contiguous()

        # 对齐尺寸
        if uncertainty.shape[2:] != x_bchw.shape[2:]:
            uncertainty = F.interpolate(uncertainty, size=x_bchw.shape[2:], mode="bilinear", align_corners=True)

        # 核心改造 1：全图视野学习 (不截断输入！)
        # 让网络看到完整的肋骨走向，计算出全局的候选矫正量
        delta_logits_full = self.refine_net(x_bchw)

        # 核心改造 2：形态学软化
        # 用 3x3 平均池化对 uncertainty 进行软化，消除尖锐的锯齿边缘
        u_map_soft = F.avg_pool2d(uncertainty, kernel_size=3, stride=1, padding=1)


        # 核心改造 3：晚期软门控 (Late Soft-Gating)
        # 算完之后，只保留不确定区域的残差值
        delta_logits = delta_logits_full * u_map_soft

        return delta_logits


def _cfg_get(cfg, key, default):
    if cfg is None:
        return default
    return cfg.get(key, default)


def _binary_entropy_map(probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = probs.clamp(min=eps, max=1.0 - eps)
    return -(probs * torch.log(probs) + (1.0 - probs) * torch.log(1.0 - probs)) / math.log(2.0)


def _mean_normalize_map(x: torch.Tensor, dims=(2, 3), cap: float = 4.0, eps: float = 1e-6) -> torch.Tensor:
    denom = x.detach().mean(dim=dims, keepdim=True).clamp_min(eps)
    return torch.clamp(x / denom, min=0.0, max=cap)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dims=(2, 3), eps: float = 1e-6) -> torch.Tensor:
    mask = mask.type_as(x)
    x_detached = x.detach()
    num = (x_detached * mask).sum(dim=dims, keepdim=True)
    den = mask.sum(dim=dims, keepdim=True)
    fallback = x_detached.mean(dim=dims, keepdim=True)
    return torch.where(den > 0, num / den.clamp_min(1.0), fallback).clamp_min(eps)


def _masked_normalize_map(
    x: torch.Tensor,
    mask: torch.Tensor,
    dims=(2, 3),
    cap: float = 4.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    denom = _masked_mean(x, mask=mask, dims=dims, eps=eps)
    return torch.clamp(x / denom, min=0.0, max=cap)


def _binary_morphological_boundary(mask: torch.Tensor, radius: int = 2) -> torch.Tensor:
    radius = int(radius)
    if radius <= 0:
        return mask.clamp(0.0, 1.0)

    kernel = 2 * radius + 1
    mask = mask.clamp(0.0, 1.0)
    dilated = F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=radius)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel, stride=1, padding=radius)
    return (dilated - eroded).clamp(0.0, 1.0)


def build_gt_guided_hard_maps(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    rib_channels: int = 24,
    alpha: float = 0.75,
    beta: float = 0.25,
    fn_weight: float = 1.5,
    fp_weight: float = 0.5,
    bg_uncertainty_scale: float = 0.25,
    uncertainty_type: str = "entropy",
    pixel_agg_mode: str = "max",
    pixel_agg_blend: float = 0.7,
    boundary_radius: int = 2,
) -> dict:
    probs = torch.sigmoid(logits)
    target = target.type_as(probs)

    b, c, h, w = probs.shape
    rib_limit = min(int(rib_channels), c)

    class_mask = torch.zeros((1, c, 1, 1), device=probs.device, dtype=probs.dtype)
    class_mask[:, :rib_limit] = 1.0

    if valid_mask is None:
        valid = torch.ones((b, c, 1, 1), device=probs.device, dtype=probs.dtype)
    else:
        valid = valid_mask[:, :c].type_as(probs).unsqueeze(-1).unsqueeze(-1)
    valid = valid * class_mask

    if str(uncertainty_type).lower() == "quadratic":
        uncertainty = 4.0 * probs * (1.0 - probs)
    else:
        uncertainty = _binary_entropy_map(probs)

    boundary_band = torch.zeros_like(probs)
    focus_region = torch.zeros_like(probs)
    if rib_limit > 0:
        boundary_small = _binary_morphological_boundary(target[:, :rib_limit], radius=boundary_radius)
        boundary_band[:, :rib_limit] = boundary_small
        focus_region[:, :rib_limit] = torch.clamp(target[:, :rib_limit] + boundary_small, min=0.0, max=1.0)

    fn_map = target * (1.0 - probs)
    fp_map = (1.0 - target) * probs * boundary_band
    error_map = fn_weight * fn_map + fp_weight * fp_map

    uncertainty_pos = target * uncertainty
    uncertainty_neg = (1.0 - target) * uncertainty * probs * boundary_band
    uncertainty_map = uncertainty_pos + bg_uncertainty_scale * uncertainty_neg

    hard_core = (alpha * error_map + beta * uncertainty_map) * valid

    channel_hard = torch.zeros((b, c, 1, 1), device=probs.device, dtype=probs.dtype)
    if rib_limit > 0:
        channel_focus = focus_region[:, :rib_limit]
        channel_score = _masked_mean(
            hard_core[:, :rib_limit],
            mask=channel_focus,
            dims=(2, 3),
        )
        channel_hard[:, :rib_limit] = channel_score
    channel_hard = channel_hard * valid

    boundary_hard = hard_core * boundary_band

    if rib_limit > 0:
        boundary_overlap = boundary_band[:, :rib_limit].sum(dim=1, keepdim=True).clamp_min(1.0)
        max_map = boundary_hard[:, :rib_limit].max(dim=1, keepdim=True)[0]
        mean_map = boundary_hard[:, :rib_limit].sum(dim=1, keepdim=True) / boundary_overlap

        mode = str(pixel_agg_mode).lower()
        if mode == "overlap_mean":
            pixel_hard = mean_map
        elif mode == "hybrid":
            pixel_hard = float(pixel_agg_blend) * max_map + (1.0 - float(pixel_agg_blend)) * mean_map
        elif mode == "mean":
            pixel_hard = boundary_hard[:, :rib_limit].mean(dim=1, keepdim=True)
        else:
            pixel_hard = max_map
    else:
        pixel_hard = torch.zeros((b, 1, h, w), device=probs.device, dtype=probs.dtype)

    return {
        'probs': probs,
        'channel_hard': channel_hard.clamp_min(0.0),
        'pixel_hard': pixel_hard.clamp_min(0.0),
        'boundary_hard': boundary_hard.clamp_min(0.0),
        'boundary_band': boundary_band,
        'focus_region': focus_region,
        'valid_channel_mask': valid,
        'fn_map': fn_map * valid,
        'fp_map': fp_map * valid,
        'error_map': error_map * valid,
        'uncertainty_map': uncertainty_map * valid,
        'hard_core': hard_core,
    }


def build_soft_weight_maps(
    hard_dict: dict,
    lambda_channel: float = 0.35,
    lambda_boundary: float = 1.50,
    lambda_error: float = 0.75,
    error_gamma: float = 1.50,
    weight_cap: float = 6.0,
) -> dict:
    channel_hard = hard_dict['channel_hard']
    boundary_hard = hard_dict['boundary_hard']
    boundary_band = hard_dict['boundary_band']
    error_map = hard_dict['error_map']
    focus_region = hard_dict['focus_region']
    valid_channel_mask = hard_dict['valid_channel_mask']

    channel_norm = _masked_normalize_map(
        channel_hard,
        mask=valid_channel_mask,
        dims=(1, 2, 3),
        cap=float(weight_cap),
    )

    boundary_norm = _masked_normalize_map(
        boundary_hard,
        mask=boundary_band,
        dims=(2, 3),
        cap=float(weight_cap),
    )

    error_norm = _masked_normalize_map(
        error_map,
        mask=focus_region,
        dims=(2, 3),
        cap=float(weight_cap),
    )

    channel_soft = torch.clamp(1.0 + float(lambda_channel) * channel_norm, min=1.0, max=float(weight_cap))
    boundary_soft = torch.clamp(1.0 + float(lambda_boundary) * boundary_norm, min=1.0, max=float(weight_cap))
    error_soft = torch.clamp(1.0 + float(lambda_error) * error_norm.pow(float(error_gamma)), min=1.0, max=float(weight_cap))

    # tri_soft = torch.clamp(channel_soft * boundary_soft * error_soft, min=1.0, max=float(weight_cap))
    tri_soft = torch.clamp(boundary_soft * error_soft, min=1.0, max=float(weight_cap))
    print("Total weight max:", tri_soft.max().item())
    print("Number of pixels > 1.0:", (tri_soft > 1.0).sum().item())
    return {
        'pixel_soft_weight': boundary_soft,
        'channel_soft_weight': channel_soft,
        'boundary_soft_weight': boundary_soft,
        'error_soft_weight': error_soft,
        'tri_soft_weight': tri_soft,
        'hybrid_soft_weight': tri_soft,
    }


class VSSM(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, num_classes=24, depths=[2, 2, 2, 2], depths_decoder=[2, 2, 2, 1],
                 dims=[96, 192, 384, 768], dims_decoder=[768, 384, 192, 96], d_state=16, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        # 两路分支的 PatchEmbed: LF (224->112) 使用 patch_size=2, HF (448->112) 使用 patch_size=4
        self.patch_embed_lf = PatchEmbed2D(patch_size=2, in_chans=in_chans, embed_dim=self.embed_dim,
                           norm_layer=norm_layer if patch_norm else None)
        self.patch_embed_hf = PatchEmbed2D(patch_size=4, in_chans=in_chans, embed_dim=self.embed_dim,
                           norm_layer=norm_layer if patch_norm else None)

        # WASTED absolute position embedding ======================
        # 没用到,APE（绝对位置编码）
        self.ape = False
        if self.ape:
            self.patches_resolution = self.patch_embed_hf.patches_resolution
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # linspace(a,b,c) [a,b]线性取c个 生成线性间距向量
        # encoder：越深的 block，drop_path 越大（更强正则）
        # decoder：反过来（因为 decoder 从“最深”往“最浅”走）
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]

        # 编码器 encoder
        # 双一路编码器：LF 和 HF（结构一致），以及每层的融合层（Concat + Linear 1x1）
        self.layers_lf = nn.ModuleList()
        self.layers_hf = nn.ModuleList()
        # 计算每层输出通道（layer 输出后如果有 downsample，则输出通道为下一层的 dims）
        layer_out_dims = []
        for i_layer in range(self.num_layers):
            out_dim = dims[i_layer + 1] if i_layer < self.num_layers - 1 else dims[i_layer]
            layer_out_dims.append(out_dim)

        for i_layer in range(self.num_layers):
            layer_lf = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            # HF 路径：使用 CNNLayer 替换 VSSLayer 以节省显存并捕捉局部特征
            layer_hf = CNNLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers_lf.append(layer_lf)
            self.layers_hf.append(layer_hf)

        # 融合层：使用 GuidedAttentionFusion 模块（对应 4 个 Stage）
        self.fusion_attention = nn.ModuleList([
            GuidedAttentionFusion(dim=dims[i_layer]) for i_layer in range(self.num_layers)
        ])

        # UGCL策略: 不确定性引导的矫正学习
        final_d_state = math.ceil(dims[0] / 6) if d_state is None else d_state
        
        # 解码器 decoder
        self.layers_up = nn.ModuleList()
        
        for i_layer in range(self.num_layers):
            layer = VSSLayer_up(
                dim=dims_decoder[i_layer],
                depth=depths_decoder[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpand2D if (i_layer != 0) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers_up.append(layer)
        
        # Decoder Heads (moved out of the loop above to modify indexing)
        self.dec_heads = nn.ModuleList()
        # Modify: Only keep Stage 1 and Stage 2 UARB.
        # i_layer 0 -> Stage 4 (dims_decoder[0]) -> remove
        # i_layer 1 -> Stage 3 (dims_decoder[1]) -> remove
        # i_layer 2 -> Stage 2 (dims_decoder[2]) -> keep
        # i_layer 3 -> Stage 1 (dims_decoder[3]) -> keep
        for i_layer in range(self.num_layers):
            if i_layer < 2:
                continue
            
            # Deep Supervision Head for each layer
            head = nn.ModuleDict({
                # 将特征对齐到统一通道数(例如 24)或者直接使用 decoder output dim
                # 这里为了简单, 直接使用当前维度的 conv 来做分类
                'conv': nn.Conv2d(dims_decoder[i_layer], num_classes, 1),
                # 联合上采样: 这里输入 num_classes 和 3(对应 hf_residual 通道)
                'guided_up': HighFrequencyGuidedUpsampling(in_channels=num_classes, guide_channels=3),
                # UARB 接受当前维度
                'uarb': Uncertainty_Guide_Enhancement(dim=dims_decoder[i_layer], num_classes=num_classes, d_state=final_d_state)
            })
            self.dec_heads.append(head)

        self.final_up = Final_PatchExpand2D(dim=dims_decoder[-1], dim_scale=4, norm_layer=norm_layer)
        # 像素点分类 输出头 (用于最终的高分辨率输出, 对应 Stage 1 的输出经过 upsample)
        # 但如果是 Deep Supervision, Stage 1 的输出就是最高分辨率之前的那个 feature.
        # 这里的 self.layers_up 最后一层输出的分辨率是 1/4 (如果 input 是 448, PatchEmbed=4, so 112. Layer up 0->112? No.
        # Let's trace resolution:
        # Encoder:
        # LF(PatchEmbed(p=2) -> 112) -> Layer0(112) -> Down(56) -> Layer1(56) -> Down(28) -> Layer2(28) -> Down(14) -> Layer3(14)
        # Decoder(dims_decoder=[768, 384, 192, 96]):
        # Layer0(input 14 + skip 14) -> 14. (dims 768)
        # Layer1(Upsample(14->28) + skip 28) -> 28. (dims 384)
        # Layer2(Upsample(28->56) + skip 56) -> 56. (dims 192)
        # Layer3(Upsample(56->112) + skip 112) -> 112. (dims 96)
        
        # self.final_up does 4x upsample? 112 * 4 = 448. Yes.
        # So Layer3 output is 1/4 resolution.
        
        # User wants UARB at *every layer* of decoder.
        # Layer 0 (14x14), Layer 1 (28x28), Layer 2 (56x56), Layer 3 (112x112).
        # These are the 4 stages.
        
        # The heads added above are for these 4 stages.
        
        # For the final output (main_logits), we usually take the last layer output (Layer 3), upsample it to 448x448.
        # Wait, the user said: "Stage 4为最低分辨率瓶颈层，Stage 1为最高分辨率输出前层".
        # In my loop: i_layer 0 -> Layer 0 (lowest res, 14x14). This corresponds to "Stage 4" in user's mind (bottleneck).
        # i_layer 3 -> Layer 3 (highest feature res, 112x112). This corresponds to "Stage 1".
        
        # The user wants "forward 函数除了返回最终的主预测图 main_logits 外，还需要返回一个包含 4 个 UARB 输出... 的列表".
        # Main logits usually come from the highest resolution branch, possibly upsampled to original image size.
        
        # Keep final_up and final_conv for the "Main Prediction" (inference mode)?
        # Or does the user want the UARB enhanced result to be the main prediction?
        # User says: "Main_DiceLoss(main_logits, GT) + weighted UARB losses".
        # This implies `main_logits` is the final result, and UARB outputs are auxiliary deep supervision.
        # BUT, the user also said: "将现有的 UARB 模块... 无缝接入到解码器的每一层中".
        # And "UARB 仅在训练阶段使用！... 如果是推理测试阶段（eval 模式），请直接跳过 UARB 计算".
        # This suggests that for inference, we don't use UARB at all. We just use the encoder-decoder backbone.
        # The original code used UARB at the *very end* for refinement.
        # Now UARB is only for *auxiliary loss during training*.
        
        # So I will keep `self.final_up` and `self.final_conv` for the main inference path (and for generating the main_logits for training loss).
        
        # But wait, does `main_logits` benefit from UARB?
        # "UARB 仅在训练阶段使用" -> So inference `main_logits` is pure backbone.
        # But during training, `main_logits` is also compared to GT. 
        # Does `main_logits` come from the backbone output *before* UARB? Yes.
        
        # 恢复 final_conv 定义
        self.final_conv = nn.Conv2d(dims_decoder[-1] // 4, num_classes, 1)
        self.fuse_proj = nn.Linear(1536, 768)

        # 定义了但在 forward 里没用
        self.apply(self._init_weights)
        self.diff_attn = MultiheadDiffAttn(embed_dim=dims[-1], depth=4, num_heads=8)




    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore  # 编译/脚本化时不要把这个函数转换成 TorchScript（忽略它）。
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, patch_embed):
        # 已废弃；保留以兼容调用，但不再用于双路流水线
        skip_list = []
        x = patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        # drop out
        x = self.pos_drop(x)

        for layer in self.layers_lf:
            skip_list.append(x)
            x = layer(x)
        return x, skip_list

    def forward_features_dual(self, lf_embed, hf_embed):
        """双路编码器前向：接收已通过 PatchEmbed 的 LF 与 HF 特征（B,H,W,C），在每层后进行融合。
        返回最终特征与每层的融合 skip_list（用于 Decoder）。
        """
        skip_list = []

        lf = lf_embed
        hf = hf_embed

        # 添加 dropout（与之前行为一致）
        if self.ape:
            lf = lf + (self.absolute_pos_embed if self.absolute_pos_embed.shape[-1] == lf.shape[-1] else 0)
            hf = hf + (self.absolute_pos_embed if self.absolute_pos_embed.shape[-1] == hf.shape[-1] else 0)
        lf = self.pos_drop(lf)
        hf = self.pos_drop(hf)

        for i in range(self.num_layers):
            # 在进入本层编码前，先融合当前两个分支的特征作为 skip connection
            # 使用 GuidedAttentionFusion
            fused_pre = self.fusion_attention[i](lf, hf)

            skip_list.append(fused_pre)

            # 各自编码器独立前向，保持 LF/HF 分离
            lf = self.layers_lf[i](lf)
            hf = self.layers_hf[i](hf)

        # 最终使用编码后的输出做一次融合作为 decoder 的输入
        # 使用最后一层的 GuidedAttentionFusion 
        fused_post = self.fusion_attention[self.num_layers - 1](lf, hf)

        return fused_post, skip_list

    def forward_features_up(self, x, skip_list):
        outs = []
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                skip = skip_list[-(inx + 1)]
                x = layer_up(x, skip)
            outs.append(x)
        return outs

    def forward_final(self, outs, hf_guide=None, target=None, valid_mask=None, uarb_cfg=None, epoch=None, iter=None, force_uarb=False):
        uarb_outputs = []
        stage_loss_weights = list(_cfg_get(uarb_cfg, 'stage_loss_weights', [0.0, 0.30]))

        if (self.training or force_uarb) and target is not None:
            rib_channels = int(_cfg_get(uarb_cfg, 'rib_channels', min(24, self.num_classes)))
            alpha = float(_cfg_get(uarb_cfg, 'alpha', 0.75))
            beta = float(_cfg_get(uarb_cfg, 'beta', 0.25))
            fn_weight = float(_cfg_get(uarb_cfg, 'fn_weight', 1.5))
            fp_weight = float(_cfg_get(uarb_cfg, 'fp_weight', 0.5))
            bg_uncertainty_scale = float(_cfg_get(uarb_cfg, 'bg_uncertainty_scale', 0.25))
            uncertainty_type = _cfg_get(uarb_cfg, 'uncertainty_type', 'entropy')
            pixel_agg_mode = _cfg_get(uarb_cfg, 'pixel_agg_mode', 'max')
            pixel_agg_blend = float(_cfg_get(uarb_cfg, 'pixel_agg_blend', 0.7))

            boundary_radius = int(_cfg_get(uarb_cfg, 'boundary_radius', 2))
            lambda_channel = float(_cfg_get(uarb_cfg, 'lambda_channel', 0.35))
            lambda_boundary = float(_cfg_get(uarb_cfg, 'lambda_boundary', 1.50))
            lambda_error = float(_cfg_get(uarb_cfg, 'lambda_error', 0.75))
            error_gamma = float(_cfg_get(uarb_cfg, 'error_gamma', 1.50))
            weight_cap = float(_cfg_get(uarb_cfg, 'weight_cap', 6.0))

            gate_mode = str(_cfg_get(uarb_cfg, 'uarb_gate_mode', 'pixel')).lower()
            detach_hard_map = bool(_cfg_get(uarb_cfg, 'detach_hard_map', True))

            for i, x_feat in enumerate(outs):
                if i < 2:
                    continue

                stage_head_index = i - 2
                stage_scale = stage_loss_weights[stage_head_index] if stage_head_index < len(stage_loss_weights) else 0.0
                if stage_scale <= 0.0:
                    continue

                head = self.dec_heads[stage_head_index]
                x_bchw = x_feat.permute(0, 3, 1, 2).contiguous()
                logits_init = head['conv'](x_bchw)

                if hf_guide is not None:
                    logits_init_up = head['guided_up'](logits_init, hf_guide)
                else:
                    logits_init_up = F.interpolate(logits_init, size=target.shape[2:], mode='bilinear', align_corners=False)

                logits_for_hard = logits_init_up.detach() if detach_hard_map else logits_init_up

                hard_dict = build_gt_guided_hard_maps(
                    logits=logits_for_hard,
                    target=target,
                    valid_mask=valid_mask,
                    rib_channels=rib_channels,
                    alpha=alpha,
                    beta=beta,
                    fn_weight=fn_weight,
                    fp_weight=fp_weight,
                    bg_uncertainty_scale=bg_uncertainty_scale,
                    uncertainty_type=uncertainty_type,
                    pixel_agg_mode=pixel_agg_mode,
                    pixel_agg_blend=pixel_agg_blend,
                    boundary_radius=boundary_radius,
                )

                soft_weights = build_soft_weight_maps(
                    hard_dict=hard_dict,
                    lambda_channel=lambda_channel,
                    lambda_boundary=lambda_boundary,
                    lambda_error=lambda_error,
                    error_gamma=error_gamma,
                    weight_cap=weight_cap,
                )

                if gate_mode == 'channel':
                    gate_map = hard_dict['channel_hard']
                else:
                    gate_map = hard_dict['pixel_hard']
                if detach_hard_map:
                    gate_map = gate_map.detach()

                delta_logits = head['uarb'](x_feat, gate_map)
                if delta_logits.shape[1] > rib_channels:
                    delta_logits = delta_logits.clone()
                    delta_logits[:, rib_channels:, :, :] = 0.0

                # 本次文档默认不打开 residual correction，先把变量控制在“软权重设计”这一处。
                logits_final_small = logits_init
                # 如果你后续要继续验证 UARB 残差矫正，再改成：
                # logits_final_small = logits_init + delta_logits

                if hf_guide is not None:
                    logits_final_up = head['guided_up'](logits_final_small, hf_guide)
                else:
                    logits_final_up = F.interpolate(logits_final_small, size=target.shape[2:], mode='bilinear', align_corners=False)

                uarb_outputs.append({
                    'logits_init': logits_init_up,
                    'logits_final': logits_final_up,
                    'target_stage': target,
                    'valid_mask_stage': valid_mask,
                    'stage_loss_weight': stage_scale,
                    'pixel_hard_map': hard_dict['pixel_hard'].detach(),
                    'channel_hard_map': hard_dict['channel_hard'].detach(),
                    'boundary_band_map': hard_dict['boundary_band'].detach(),
                    'boundary_hard_map': hard_dict['boundary_hard'].detach(),
                    'error_map': hard_dict['error_map'].detach(),
                    'uncertainty_map': hard_dict['uncertainty_map'].detach(),
                    'hard_core_map': hard_dict['hard_core'].detach(),
                    'pixel_soft_weight': soft_weights['pixel_soft_weight'].detach(),
                    'channel_soft_weight': soft_weights['channel_soft_weight'].detach(),
                    'boundary_soft_weight': soft_weights['boundary_soft_weight'].detach(),
                    'error_soft_weight': soft_weights['error_soft_weight'].detach(),
                    'tri_soft_weight': soft_weights['tri_soft_weight'].detach(),
                    'hybrid_soft_weight': soft_weights['hybrid_soft_weight'].detach(),
                })

        x_last = outs[-1]
        x_decoder = self.final_up(x_last)
        x_decoder_bchw = x_decoder.permute(0, 3, 1, 2)
        main_logits = self.final_conv(x_decoder_bchw)

        return main_logits, uarb_outputs


    """
    这里得到的“高频/低频”是空间域（spatial domain）上的高低频分量，不是在傅里叶域（frequency domain）里做 FFT 后再按频谱切出来的那种“频率域高低频”。
    """
    def forward(self, x, epoch=None, iter=None, target=None, valid_mask=None, uarb_cfg=None, force_uarb=False):
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        # 1) 拉普拉斯分频：LF 路径 - 高斯平滑并下采样到 H/2 x W/2（假设输入为 2x 的分辨率）
        # 高斯核大小 5×5，标准差 1.0。
        # 作用：做一个低通滤波，把纹理/边缘抹掉一些，只保留大结构。
        kernel_size = 5
        sigma = 1.0
        # 构造高斯核
        # 高斯核可以看作是将样本映射到无穷维的特征空间，从而捕捉到更加丰富的特征关系；在原始空间中线性不可分的问题，可能在映射后的高维空间中被线性分割。
        # -2.5 ~ 3 生成 -2，-1，0，1，2
        a = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, device=x.device, dtype=x.dtype)
        # [X,Y] = meshgrid(x,y) 基于向量 x 和 y 中包含的坐标返回二维网格坐标。X 是一个矩阵，每一行是 x 的一个副本；Y 也是一个矩阵，每一列是 y 的一个副本。坐标 X 和 Y 表示的网格有 length(y) 个行和 length(x) 个列。
        xx, yy = torch.meshgrid(a, a, indexing='ij')
        # exp(-(x²+y²)/2σ²)：高斯公式
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma * sigma))
        # 归一化，让核的和等于 1（不改变整体亮度/均值）
        kernel = kernel / kernel.sum()
        # [C, 1, 5, 5]
        # 这是为了配合下面的 groups=C：实现每个通道各自用同一个高斯核做卷积（互不混合通道）。
        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(C, 1, 1, 1)

        lf_smoothed = F.conv2d(x, weight=kernel, padding=kernel_size // 2, groups=C)
        # 降采样,低频信息本来变化慢，用更低分辨率表示更划算；也让后面 LF 分支可以更专注全局结构。
        # interpolate() 插值函数 mode='bilinear' 线性插值
        lf_down = F.interpolate(lf_smoothed, size=(H // 2, W // 2), mode='bilinear', align_corners=False)

        # HF 路径为残差：HF = 原图 - 上采样(LF)
        lf_up = F.interpolate(lf_down, size=(H, W), mode='bilinear', align_corners=False)
        hf_residual = x - lf_up

        # 2) PatchEmbed：LF 使用 patch_size=2,stride=2 产生 112x112；HF 使用 patch_size=4,stride=4 也产生 112x112
        lf_emb = self.patch_embed_lf(lf_down)
        hf_emb = self.patch_embed_hf(hf_residual)

        # 3) 双路编码器前向并融合
        x_fused, skip_list = self.forward_features_dual(lf_emb, hf_emb)

        # 4) 解码器（单路），接收融合后的 skip connections
        # x_ups is a list of features at each stage.
        x_ups = self.forward_features_up(x_fused, skip_list)
        
        main_logits, uarb_outputs = self.forward_final(
            x_ups, 
            hf_guide=hf_residual, 
            target=target, 
            valid_mask=valid_mask, 
            uarb_cfg=uarb_cfg, 
            epoch=epoch, 
            iter=iter, 
            force_uarb=force_uarb
        )

        if self.training or force_uarb:
            return main_logits, uarb_outputs
        else:
            return main_logits