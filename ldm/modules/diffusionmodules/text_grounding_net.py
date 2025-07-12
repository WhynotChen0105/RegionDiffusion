import torch
import torch.nn as nn
from ldm.modules.attention import BasicTransformerBlock
from ldm.modules.diffusionmodules.util import checkpoint, FourierEmbedder
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super(PositionalEmbedding, self).__init__()
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        positions = torch.arange(0, x.size(1)).unsqueeze(0).to(x.device)
        return self.embedding(positions)


class PositionNet(nn.Module):
    def __init__(self, in_dim, out_dim, fourier_freqs=8, mask_sizes=[64, 32]):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mask_sizes = mask_sizes
        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_embedder = PositionalEmbedding(99, in_dim)
        self.position_dim = fourier_freqs * 2 * 4  # 2 is sin&cos, 4 is xyxy
        self.linears = nn.Sequential(
            nn.Linear(self.in_dim + self.position_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )
        self.null_positive_feature = torch.nn.Parameter(torch.zeros([self.in_dim]))
        self.null_position_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))

    def forward(self, boxes, masks, positive_embeddings):
        B, N, _ = boxes.shape
        masks = masks.unsqueeze(-1)
        attn_masks = dict()
        # construct the region attention mask
        for size in self.mask_sizes:
            attn_mask = torch.zeros([B, N, size, size], device=boxes.device)
            # background_mask = torch.ones([B, size, size])
            coordinates = torch.round(boxes * size)  # (B,N,4)
            for b in range(B):
                for n in range(N - 1):
                    if masks[b][n] == 1:
                        x1, y1, x2, y2 = coordinates[b][n]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        attn_mask[b, n, y1:y2, x1:x2] = 1
                        # background_mask[b,y1:y2,x1:x2] = 0
            # attn_mask[:,N-1,:,:] = background_mask # (B,N,size,size)
            attn_mask[:, N - 1, :, :][attn_mask.sum(1) == 0] = 1

            # self_attention masks
            HW = size * size
            # tokens in same box can communicate with each other
            attn_mask_o = attn_mask.view(B * N, HW, 1)
            attn_mask_t = attn_mask.view(B * N, 1, HW)
            attn_mask_sim = torch.bmm(attn_mask_o, attn_mask_t)  # (B * phase_num, HW, HW)
            attn_mask_sim = attn_mask_sim.view(B, N, HW, HW).sum(dim=1)
            attn_mask_sim[attn_mask_sim > 1] = 1  # (B, HW, HW)
            # attn_mask_sim = attn_mask_sim.view(B, 1, HW, HW)
            # attn_mask_sim = attn_mask_sim.repeat(1, self.heads, 1, 1)
            # attn_mask_sim = attn_mask_sim.view(B * self.heads, HW, HW)
            sim_mask = torch.zeros([B, HW + N, HW + N], device=boxes.device)
            sim_mask[:, :HW, :HW] = attn_mask_sim
            # visual tokens only can communicate with the relevant groundings
            attn_mask = attn_mask.view(B, N, HW).permute(0, 2, 1)  # B HW N
            sim_mask[:, :HW, HW:] = attn_mask
            attn_masks[HW + N] = sim_mask

            # embedding position (it may includes padding as placeholder)
        xyxy_embedding = self.fourier_embedder(boxes)  # B*N*4 --> B*N*C
        # learnable null embedding
        positive_null = self.null_positive_feature.view(1, 1, -1)
        xyxy_null = self.null_position_feature.view(1, 1, -1)
        # add position into text_embeddings
        positive_embeddings = positive_embeddings + self.position_embedder(positive_embeddings).repeat(B, 1, 1)
        # replace padding with learnable null embedding 
        positive_embeddings = positive_embeddings * masks + (1 - masks) * positive_null
        xyxy_embedding = xyxy_embedding * masks + (1 - masks) * xyxy_null

        objs = self.linears(torch.cat([positive_embeddings, xyxy_embedding], dim=-1))
        assert objs.shape == torch.Size([B, N, self.out_dim])
        return objs, attn_masks



