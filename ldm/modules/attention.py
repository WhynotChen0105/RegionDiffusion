
from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

# from ldm.modules.diffusionmodules.util import checkpoint, FourierEmbedder
from torch.utils import checkpoint


def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)

class SelfAttention(nn.Module):
    def __init__(self, query_dim, heads=8, dim_head=64, dropout=0., efficient_attention=False):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)
        self.efficient_attention =efficient_attention
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, x, attn_mask=None):
        q = self.to_q(x)  # B*N*(H*C)
        k = self.to_k(x)  # B*N*(H*C)
        v = self.to_v(x)  # B*N*(H*C)

        B, N, HC = q.shape
        H = self.heads
        C = HC // H
        # W = int(math.sqrt(N))
        if attn_mask is not None and N in attn_mask.keys(): # not use flash attention
            q = q.view(B, N, H, C).permute(0, 2, 1, 3).reshape(B * H, N, C)  # (B*H)*N*C
            k = k.view(B, N, H, C).permute(0, 2, 1, 3).reshape(B * H, N, C)  # (B*H)*N*C
            v = v.view(B, N, H, C).permute(0, 2, 1, 3).reshape(B * H, N, C)  # (B*H)*N*C

            sim = torch.einsum('b i c, b j c -> b i j', q, k) * self.scale  # (B*H)*N*N

            # print(f"only use the mask in {math.sqrt(N)}")
            attn_mask = attn_mask[N] # B*HW*HW
            # _, HW, _ = attn_mask.shape
            attn_mask = attn_mask.view(B, 1, N, N).repeat(1,self.heads,1,1)
            attn_mask_sim = attn_mask.view(B * self.heads, N, N)  # (B * head, HW, HW)

            sim[attn_mask_sim == 0] = -torch.finfo(sim.dtype).max


            attn = sim.softmax(dim=-1)  # (B*H)*N*N

            out = torch.einsum('b i j, b j c -> b i c', attn, v)  # (B*H)*N*C
            out = out.view(B, H, N, C).permute(0, 2, 1, 3).reshape(B, N, (H * C))  # B*N*(H*C)
        else:
            if self.efficient_attention:
                q = q.view(B, N, H, C).permute(0, 2, 1, 3)  # B*H*N*C
                k = k.view(B, N, H, C).permute(0, 2, 1, 3)  # B*H*N*C
                v = v.view(B, N, H, C).permute(0, 2, 1, 3)  # B*H*N*C
                with torch.backends.cuda.sdp_kernel():
                    out = F.scaled_dot_product_attention(q, k, v, attn_mask=None)

            else:
                q = q.view(B, N, H, C).permute(0, 2, 1, 3).reshape(B * H, N, C)  # (B*H)*N*C
                k = k.view(B, N, H, C).permute(0, 2, 1, 3).reshape(B * H, N, C)  # (B*H)*N*C
                v = v.view(B, N, H, C).permute(0, 2, 1, 3).reshape(B * H, N, C)  # (B*H)*N*C
                sim = torch.einsum('b i c, b j c -> b i j', q, k) * self.scale  # (B*H)*N*N
                attn = sim.softmax(dim=-1)  # (B*H)*N*N
                out = torch.einsum('b i j, b j c -> b i c', attn, v)  # (B*H)*N*C

            out = out.contiguous().view(B, H, N, C).permute(0, 2, 1, 3).reshape(B, N, (H * C))  # B*N*(H*C)


        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, heads=8, dim_head=64, dropout=0, efficient_attention=False):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.efficient_attention = efficient_attention

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(key_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(value_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def fill_inf_from_mask(self, sim, mask):
        if mask is not None:
            B, M = mask.shape
            mask = mask.unsqueeze(1).repeat(1, self.heads, 1).reshape(B * self.heads, 1, -1)
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg_value)
        return sim

    def forward(self, x, key, value, mask=None):
        q = self.to_q(x)  # B*N*(H*C)
        k = self.to_k(key)  # B*M*(H*C)
        v = self.to_v(value)  # B*M*(H*C)

        B, N, HC = q.shape
        _, M, _ = key.shape
        H = self.heads
        C = HC // H

        q = q.view(B, N, H, C).permute(0, 2, 1, 3)  # B*H*N*C
        k = k.view(B, M, H, C).permute(0, 2, 1, 3)  # B*H*M*C
        v = v.view(B, M, H, C).permute(0, 2, 1, 3)  # B*H*M*C

        if self.efficient_attention:
            # Flash attention requires q,k,v to have the same last dimension and to be a multiple of 8 and less than
            # or equal to 128. If the last dimension of q,k,v is larger than 128, we cannot use flash_attention.
            # https://github.com/Dao-AILab/flash-attention/issues/108
            with torch.backends.cuda.sdp_kernel():
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        else:
            q = q.reshape(B * H, N, C)  # (B*H)*N*C
            k = k.reshape(B * H, M, C)  # (B*H)*M*C
            v = v.reshape(B * H, M, C)  # (B*H)*M*C

            sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale  # (B*H)*N*M
            self.fill_inf_from_mask(sim, mask)
            attn = sim.softmax(dim=-1)  # (B*H)*N*M

            out = torch.einsum('b i j, b j d -> b i d', attn, v)  # (B*H)*N*C

        out = out.contiguous().view(B, H, N, C).permute(0, 2, 1, 3).reshape(B, N, (H * C))  # B*N*(H*C)

        return self.to_out(out)


class GatedCrossAttentionDense(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, n_heads, d_head):
        super().__init__()

        self.attn = CrossAttention(query_dim=query_dim, key_dim=key_dim, value_dim=value_dim, heads=n_heads,
                                   dim_head=d_head)
        self.ff = FeedForward(query_dim, glu=True)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)))
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)))

        # this can be useful: we can externally change magnitude of tanh(alpha)
        # for example, when it is set to 0, then the entire model is same as original one
        self.scale = 1

    def forward(self, x, objs):
        x = x + self.scale * torch.tanh(self.alpha_attn) * self.attn(self.norm1(x), objs, objs)
        x = x + self.scale * torch.tanh(self.alpha_dense) * self.ff(self.norm2(x))

        return x


class GatedRegionCrossAttentionDense(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, n_heads, d_head, use_self=False):
        super().__init__()
        self.use_self = use_self
        self.attn1 = CrossAttention(query_dim=query_dim, key_dim=key_dim, value_dim=value_dim, heads=n_heads, dim_head=d_head)
        # self.attn2 = CrossAttention(query_dim=query_dim, key_dim=key_dim, value_dim=value_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, glu=True)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm3 = nn.LayerNorm(query_dim)

        self.scale_attn1 = nn.Parameter(torch.tensor(0.))
        self.scale_ff = nn.Parameter(torch.tensor(0.))
        # add the self_attention
        if self.use_self:
            self.linear = nn.Linear(key_dim, query_dim)
            self.attn2 = SelfAttention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
            self.scale_attn2 = nn.Parameter(torch.tensor(0.))
            self.norm2 = nn.LayerNorm(query_dim)
        # for example, when it is set to 0, then the entire model is same as original one
        self.scale = 1

    def forward(self, x, objs, attn_masks):
        _, N, _ = x.shape
        x = x + self.scale * torch.tanh(self.scale_attn1) * self.attn1(self.norm1(x), objs, objs, attn_masks)
        if self.use_self:
            x = x + self.scale * torch.tanh(self.scale_attn2) * self.attn2(self.norm2(torch.cat([x, self.linear(objs)], dim=1)), attn_masks)[:,:N,]
        x = x + self.scale * torch.tanh(self.scale_ff) * self.ff(self.norm3(x))

        return x


class GatedSelfAttentionDense(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, d_head):
        super().__init__()

        # we need a linear projection since we need cat visual feature and obj feature
        self.linear = nn.Linear(context_dim, query_dim)

        self.attn = SelfAttention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, glu=True)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)))
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)))

        # this can be useful: we can externally change magnitude of tanh(alpha)
        # for example, when it is set to 0, then the entire model is same as original one
        self.scale = 1

    def forward(self, x, objs):
        N_visual = x.shape[1]
        objs = self.linear(objs)

        x = x + self.scale * torch.tanh(self.alpha_attn) * self.attn(self.norm1(torch.cat([x, objs], dim=1)))[:,
                                                           0:N_visual, :]
        x = x + self.scale * torch.tanh(self.alpha_dense) * self.ff(self.norm2(x))

        return x

class GatedRegionSelfAttentionDense(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, d_head, efficient_attention=False):
        super().__init__()

        # we need a linear projection since we need cat visual feature and obj feature
        self.linear = nn.Linear(context_dim, query_dim)

        self.attn = SelfAttention(query_dim=query_dim, heads=n_heads, dim_head=d_head, efficient_attention=efficient_attention)
        self.ff = FeedForward(query_dim, glu=True)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)))
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)))

        # this can be useful: we can externally change magnitude of tanh(alpha)
        # for example, when it is set to 0, then the entire model is same as original one
        self.scale = 1

    def forward(self, x, objs, attn_mask):
        N_visual = x.shape[1]
        objs = self.linear(objs)

        x = x + self.scale * torch.tanh(self.alpha_attn) * self.attn(self.norm1(torch.cat([x, objs], dim=1)), attn_mask=attn_mask)[:,
                                                           0:N_visual, :]
        x = x + self.scale * torch.tanh(self.alpha_dense) * self.ff(self.norm2(x))

        return x

class BasicTransformerBlock(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, n_heads, d_head, fuser_type, use_checkpoint=True, use_fuser=True, efficient_attention=False):
        super().__init__()
        self.attn1 = SelfAttention(query_dim=query_dim, heads=n_heads, dim_head=d_head, efficient_attention=efficient_attention)
        self.ff = FeedForward(query_dim, glu=True)
        self.attn2 = CrossAttention(query_dim=query_dim, key_dim=key_dim, value_dim=value_dim, heads=n_heads,
                                    dim_head=d_head, efficient_attention=efficient_attention)
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.norm3 = nn.LayerNorm(query_dim)
        self.use_checkpoint = use_checkpoint
        self.use_fuser = use_fuser
        self.fuser_type = fuser_type
        if use_fuser:
            if fuser_type == "gatedSA":
                # note key_dim here actually is context_dim
                self.fuser = GatedSelfAttentionDense(query_dim, key_dim, n_heads, d_head)
            if fuser_type == "gatedCA":
                self.fuser = GatedCrossAttentionDense(query_dim, key_dim, value_dim, n_heads,d_head)
            elif fuser_type == 'gatedRSA':
                self.fuser =  GatedRegionSelfAttentionDense(query_dim, key_dim, n_heads, d_head, efficient_attention)
            else:
                assert False

    def forward(self, x, context, objs, attn_mask):

        if self.use_checkpoint and x.requires_grad:
            return checkpoint.checkpoint(self._forward, x, context, objs, attn_mask, use_reentrant=True)
        else:
            return self._forward(x, context, objs, attn_mask)

    def _forward(self, x, context, objs, attn_mask):
        # print(context)
        x = self.attn1(self.norm1(x)) + x
        if self.use_fuser:
            if self.fuser_type in ['gatedRSA']:
                x = self.fuser(x, objs, attn_mask)  # identity mapping in the beginning
            else:
                x = self.fuser(x, objs)
        x = self.attn2(self.norm2(x), context, context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    def __init__(self, in_channels, key_dim, value_dim, n_heads, d_head, depth=1, fuser_type=None, use_checkpoint=True,
                 use_fuser=True,efficient_attention=False):
        super().__init__()
        self.in_channels = in_channels
        query_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        self.use_fuser = use_fuser

        self.proj_in = nn.Conv2d(in_channels,
                                 query_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(query_dim, key_dim, value_dim, n_heads, d_head, fuser_type,
                                   use_checkpoint=use_checkpoint, use_fuser=self.use_fuser, efficient_attention=efficient_attention)
             for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(query_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context, objs, attn_mask):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context, objs, attn_mask)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in