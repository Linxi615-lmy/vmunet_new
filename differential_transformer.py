import math
import torch
import torch.nn.functional as F
from torch import nn
from rms_norm import RMSNorm
from rotary import apply_rotary_emb

import math
import torch
import torch.nn.functional as F
from torch import nn

import math
import torch
import torch.nn.functional as F
from torch import nn




def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class MultiheadDiffAttn(nn.Module):
    def __init__(
        self,
        embed_dim,
        depth, # current layer index
        
        num_heads,
        num_kv_heads=None,
        input_size=448,
        posi_scale=1,
        patch_size=4
    ):
        super().__init__()
        self.embed_dim = embed_dim


        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.n_rep = self.num_heads // self.num_kv_heads

        # 调整 head_dim 的计算方式
        self.head_dim = embed_dim // num_heads // 2
        # self.head_dim = embed_dim // num_heads 
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)
        # self.subln = RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)
        self.input_size = input_size
        self.posi_scale = posi_scale
        self._precomputed_freqs_cis = None
        self.patch_size = patch_size
    
    def build_rel_pos(self, x, start_pos=0):
        if self._precomputed_freqs_cis is None:
            # 获取图像空间维度信息
            h = w = int(self.input_size // self.patch_size)
            self.flatten_input_size = h * w  # 更新展平后的尺寸
            
            # 生成二维坐标网格
            rows = torch.arange(h, device=x.device).float()
            cols = torch.arange(w, device=x.device).float()
            grid_rows, grid_cols = torch.meshgrid(rows, cols, indexing='ij')
            
            # 展平坐标矩阵
            grid_rows = grid_rows.reshape(-1)  # (h*w,)
            grid_cols = grid_cols.reshape(-1)  # (h*w,)
            
            # 生成频率基（行和列共享同一组频率基）
            angle = 1.0 / ((10000 * self.posi_scale) ** 
                        torch.linspace(0, 1, self.head_dim//4, 
                                    dtype=torch.float, device=x.device))
            
            # 计算行和列的角度编码（交替排列）
            angles_rows = grid_rows[:, None] * angle[None, :]  # (h*w, d//4)
            angles_cols = grid_cols[:, None] * angle[None, :]  # (h*w, d//4)
            
            # 合并行列角度（交替维度）
            angles = torch.zeros((h*w, self.head_dim//2), 
                            device=x.device, dtype=torch.float)
            angles[:, 0::2] = angles_rows  # 偶数位置放行编码
            angles[:, 1::2] = angles_cols  # 奇数位置放列编码
            
            self._precomputed_freqs_cis = angles

        # 获取当前窗口的位置编码
        current_angles = self._precomputed_freqs_cis[start_pos:start_pos+x.size(1)]
        
        # 计算旋转矩阵分量
        cos = torch.cos(current_angles).to(x.dtype)  # (seq_len, d//2)
        sin = torch.sin(current_angles).to(x.dtype)  # (seq_len, d//2)
        
        return (cos, sin)

    def forward(
        self,
        x,
        hf_x,  # 高频分支
        attn_mask=None,
    ):
        # 确保输入为 (batch_size, height, width, channels)
        bsz, h, w, embed_dim = x.size()
        src_len = h * w  # 把 2D 变成 1D 序列长度

        # 调整输入形状：(bz, h, w, c) -> (bz, seq_len, embed_dim)
        x = x.view(bsz, src_len, embed_dim)
        hf_x = hf_x.view(bsz, src_len, embed_dim)
        # 计算标准 Query, Key, Value
        q = self.q_proj(x)  
        k = self.k_proj(x)  
        v = self.v_proj(x)  

        # 计算高频分支的 Query, Key
        hf_q = self.q_proj(hf_x)  
        hf_k = self.k_proj(hf_x)  
        # 打印中间张量形状
        # print(f"q shape: {q.shape}")

        # 重新调整形状
        # q = q.view(bsz, src_len, self.num_heads, self.head_dim)
        # k = k.view(bsz, src_len, self.num_kv_heads, self.head_dim)
        # v = v.view(bsz, src_len, self.num_kv_heads, self.head_dim)
        q = q.view(bsz, src_len, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len,   self.num_kv_heads, 2 * self.head_dim)
        
        # hf_q = hf_q.view(bsz, src_len, 2 * self.num_heads, self.head_dim)
        # hf_k = hf_k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
        hf_q = hf_q.view(bsz, src_len, 2 * self.num_heads, self.head_dim)
        hf_k = hf_k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)




        # 确保形状调整正确，可添加打印语句检查
        # print(f"q shape after view: {q.shape}")
        # print(f"k shape after view: {k.shape}")
        # print(f"v shape after view: {v.shape}")
        # print(f"hf_q shape after view: {hf_q.shape}")
        # print(f"hf_k shape after view: {hf_k.shape}")
        # pose_embed = self.pose_embed[:src_len, :]  # 确保 seq_len 一致

        # print(f"self.pose_embed shape: {self.pose_embed.shape}")

        # # 应用 Rotary Embedding
        rel_pos_x = self.build_rel_pos(x)
        rel_pos_hf = self.build_rel_pos(hf_x)

        q = apply_rotary_emb(q, *rel_pos_x ,interleaved=True)
        k = apply_rotary_emb(k, *rel_pos_x, interleaved=True)
        hf_q = apply_rotary_emb(hf_q, *rel_pos_hf,interleaved=True)
        hf_k = apply_rotary_emb(hf_k, *rel_pos_hf,  interleaved=True)

        offset = src_len - src_len
        q = q.transpose(1, 2)
        k = repeat_kv(k.transpose(1, 2), self.n_rep)
        v = repeat_kv(v.transpose(1, 2), self.n_rep)
        # 调整 v 的形状使其和 attn_weights 匹配
        hf_q = hf_q.transpose(1, 2)
        hf_k = repeat_kv(hf_k.transpose(1, 2), self.n_rep)

        # 计算注意力得分
        q *= self.scaling
        hf_q *= self.scaling

        attn_weights_x = torch.matmul(q, k.transpose(-1, -2))  # 原始注意力
        attn_weights_hf = torch.matmul(hf_q, hf_k.transpose(-1, -2))  # 高频注意力

        # 应用注意力 mask
        if attn_mask is None:
            attn_mask_x = torch.triu(
                torch.zeros([src_len, src_len])
                .float()
                .fill_(float("-inf"))
                .type_as(attn_weights_x),
                1 + offset,
            )
        if attn_mask is None:
            attn_mask_hf = torch.triu(
                torch.zeros([src_len, src_len])
                .float()
                .fill_(float("-inf"))
                .type_as(attn_weights_hf),
                1 + offset,
            )
        
        attn_weights_x = torch.nan_to_num(attn_weights_x)
        attn_weights_hf = torch.nan_to_num(attn_weights_hf)
        
        attn_weights_x += attn_mask_x
        attn_weights_hf += attn_mask_hf

        # 计算 softmax
        attn_weights_x = F.softmax(attn_weights_x, dim=-1, dtype=torch.float32).type_as(attn_weights_x)
        attn_weights_hf = F.softmax(attn_weights_hf, dim=-1, dtype=torch.float32).type_as(attn_weights_hf)

        # 计算 λ 值
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        # 计算最终的注意力权重
        # attn_weights = attn_weights_1 - lambda_full * attn_weights_2
        attn_weights_x = attn_weights_x.view(bsz, self.num_heads, 2, src_len,src_len)  #[8, 8, 2, 196, 196]
        attn_weights_hf = attn_weights_hf.view(bsz, self.num_heads, 2, src_len,src_len)  #[8, 8, 2, 196, 196]
        attn_weights_xh0 = attn_weights_x[:, :, 0] + attn_weights_hf[:, :, 0]


        attn_weights_xh1 = attn_weights_x[:, :, 1] + attn_weights_hf[:, :, 1]
        attn_weights = attn_weights_xh0 - lambda_full * attn_weights_xh1

        # 检查 attn_weights 和 v 的形状
        # print(f"attn_weights shape: {attn_weights.shape}")

        # 计算注意力输出
        attn = torch.matmul(attn_weights, v)
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)

        attn = attn.transpose(1, 2).reshape(bsz, src_len, self.num_heads * 2 * self.head_dim)
        # attn = attn.transpose(1, 2).reshape(bsz, src_len, self.num_heads * self.head_dim)


        # 添加调试信息
        # print(f"attn shape: {attn.shape}")
        # print(f"out_proj in_features: {self.out_proj.in_features}")
        # print(f"out_proj out_features: {self.out_proj.out_features}")

        # 输出投影
        attn = self.out_proj(attn)
        h = w = int(math.sqrt(src_len))  # 计算 h 和 w
        attn = attn.view(bsz, h, w, embed_dim)  # 重新调整形状

        # print(f"attn111111111111111 shape: {attn.shape}")
        return attn