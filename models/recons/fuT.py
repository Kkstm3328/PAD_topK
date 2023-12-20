import copy
import math
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from models.initializer import initialize_from_cfg
from torch import Tensor, nn

class fuT(nn.Module):
    def __init__(
            self,
            hidden_dim, # 同uniad, 超参，过程中的channel维度
            batch_size,
            feature_size,
            img_size,
            K,
            dim_in, # MFCN后的channel维度
            neighbor_size,   # 在neighbor_size * neighbor_size的窗口内作atten，其余mask掉
            num_encoder_layers,
            dim_feedforward,
    ):
        """
            hidden_dim: 处理时channel维度
            feature_size: [h, w]
            img_size: 同上
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.h, self.w = feature_size
        self.pos_embedding = PositionEmbeddingLearned([self.h, self.w], hidden_dim)
        self.q_input_proj = nn.Linear(dim_in, hidden_dim)
        self.kv_input_proj = nn.Linear(dim_in, hidden_dim)
        self.transformerEncoder = TransformerEncoder(neighbor_size, feature_size, batch_size, hidden_dim, K, num_encoder_layers, dim_feedforward)#
        self.output_proj = nn.Linear(hidden_dim, dim_in)

        self.upsample = nn.UpsamplingBilinear2d(size=img_size)

    def forward(self, query_features, matched_features_list):
        """
        query_feature: <tensor> output of MFCN, b*14*14*272
        matched_feature_list: <list:tensor> the item.shape == query_feature.shape
        """
        qurey_tokens = rearrange(query_features, "b c h w -> (h w) b c")
        qurey_tokens = self.q_input_proj(qurey_tokens)
        matched_tokens_list = list()
        for matched_features in matched_features_list:
            matched_tokens = rearrange(matched_features, "b c h w -> (h w) b c")
            matched_tokens = self.kv_input_proj(matched_tokens)
            matched_tokens_list.append(matched_tokens)
        
        pos_embed = self.pos_embedding(qurey_tokens)    # 位置编码，供query_img和各个matched_img用

        recon_tokens = self.transformerEncoder(qurey_tokens, matched_tokens_list, pos_embed)

        recon_tokens = self.output_proj(recon_tokens)
        recon_feature = rearrange(recon_tokens, "(h w) b c -> b c h w", h=self.h)

        pred = torch.sqrt(      # 输入和重建的欧氏距离，作甚？->是作差异然后上采样得到gtmask类似物？
            torch.sum((recon_feature - query_features) ** 2, dim=1, keepdim=True)   #会不会输入和重建特征相同地方数值上相差太大所以结果不行
        )  # B x 1 x H x W
            # norm pred
        max_val = torch.max(pred)
        min_val = torch.min(pred)
        pred = (pred - min_val) / (max_val - min_val)
        pred = self.upsample(pred)  

        return recon_feature, pred
    
class TransformerEncoder(nn.Module):
    def __init__(
            self, 
            neighbor_size,
            feature_size,
            batch_size,
            hidden_dim,
            K, 
            num_encoder_layers,
            dim_feedforward,
    ):
        super().__init__()        
        
        self.neighbor_size = neighbor_size
        self.feature_size = feature_size
        self.batch_size = batch_size
        self.num_encoder_layers = num_encoder_layers
        encoder_layer = TransformerEncoderLayer(batch_size, hidden_dim, K, dim_feedforward=dim_feedforward)#
        self.encoder_norm = nn.LayerNorm(hidden_dim)
        self.layers = _get_clones(encoder_layer, self.num_encoder_layers)
        ###uniad的get_clone? 这样的几层权重共享吗 ->不共享

    def forward(self, query_tokens, matched_token_list, pos_embed):
        """
        query_tokens (h w) b c

        得mask
        num_encoder_layer层的IO
        """
        pos_embed = torch.cat(          #pos_embed.shape = (H x W) x C
            [pos_embed.unsqueeze(1)] * min(self.batch_size, query_tokens.shape[1]), dim=1    #为每个batch都复制一份，在最后有时batch会少于batch_size
        )  # (H X W) x B x C

        mask = self.generate_mask(self.feature_size, self.neighbor_size)
        
        rec_tokens = query_tokens
        for layer in self.layers:
            rec_tokens = layer(rec_tokens, matched_token_list, pos_embed, mask)

        # rec_tokens = self.encoder_norm(rec_tokens)

        return rec_tokens
    
    def generate_mask(self, feature_size, neighbor_size):   #得到如论文Fig5的mask_map
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).

        mask: shape==(h * w, h * w), 第一个维度是query_img的
        """
        h, w = feature_size
        hm, wm = neighbor_size, neighbor_size      # h_mask
        mask = torch.zeros(h, w, h, w)
        for idx_h1 in range(h):     #两层for代表query
            for idx_w1 in range(w):
                idx_h2_start = max(idx_h1 - hm // 2, 0)
                idx_h2_end = min(idx_h1 + hm // 2 + 1, h)
                idx_w2_start = max(idx_w1 - wm // 2, 0)
                idx_w2_end = min(idx_w1 + wm // 2 + 1, w)
                mask[
                    idx_h1, idx_w1, idx_h2_start:idx_h2_end, idx_w2_start:idx_w2_end
                ] = 1
        mask = mask.view(h * w, h * w)      # reshape
        # 现在，需要mask掉的值为0，否则1
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
            .cuda()
        )
        return mask

class TransformerEncoderLayer(nn.Module):
    def __init__(
            self, 
            batch_size,
            hidden_dim,
            K,
            dropout_wei = 0.1,
            dim_feedforward = 2048,
    ):
        super().__init__()
        self.atten = MultiCrossAttention(hidden_dim, K)
        self.dropout1 = nn.Dropout(dropout_wei)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.activation = nn.ReLU()
        self.dropout_ffn = nn.Dropout(dropout_wei)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)

        self.dropout2 = nn.Dropout(dropout_wei)
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, src_tokens, matched_token_list, pos_embed, mask):
        """
        要完成：
        add mask
        uniad中同名的任务，其中attention要自己写
        """
        q = src_tokens + pos_embed      # (h w) b c
        k_topK = torch.stack(matched_token_list) + pos_embed      # K (h w) b c
        
        #残差怎么加？Uniad的decoder是只加q
        rec_tokens = self.atten(q, k_topK, torch.stack(matched_token_list), mask)
        rec_tokens = src_tokens + self.dropout1(rec_tokens)
        rec_tokens = self.norm1(rec_tokens)
        tmp_rec_tokens = self.linear2(self.dropout_ffn(self.activation(self.linear1(rec_tokens))))

        rec_tokens = rec_tokens + self.dropout2(tmp_rec_tokens)
        rec_tokens = self.norm2(rec_tokens)

        # pred = 
        return rec_tokens


class MultiCrossAttention(nn.Module):
    """
    实现attention
    q对K个kv作attention再聚合
    """
    def __init__(
            self, 
            channel_dim,
            K,
    ):
        super().__init__()

        self.K = K
        self.aggHeadK = nn.Linear(channel_dim*K, channel_dim)
    
    def forward(
            self,
            q,
            k_topK,
            v_topK,
            mask,
    ):
        """
        input:
            q:  (h w) b c
            k_topK:  K (h w) b c
            v_topK:  K (h w) b c
        """

        q = rearrange(q, "n b c -> b n c")      # n = h * W
        k_topK = rearrange(k_topK, "K n b c -> K b n c")
        v_topK = rearrange(v_topK, "K n b c -> K b n c")

        q_scaled = q / math.sqrt(q.shape[-1])   # torch源码：缩放q以免qk点数字过大
        
        # K个head分别计算
        attn_outputs_listK = []
        for i in range(self.K):
            attn_weights = torch.baddbmm(mask, q_scaled, k_topK[i].transpose(-2, -1), beta=1, alpha=1)     # b * (h w) * (h w)
            # attn_weights = attn_weights * mask  #>>>>>>>>mask会广播吧--------->mask该加还是乘
            attn_weights = torch.softmax(attn_weights, dim=-1)  #》》》》》》》》》》》》先乘mask还是后
            attn_outputs = torch.bmm(attn_weights, v_topK[i])   # b * n * c
            attn_outputs_listK.append(attn_outputs)
        attn_outputs_K = torch.cat(attn_outputs_listK, dim=-1)  # b * n * (c * K)
        #>>>>>>>>>>融合是在这个attention里面外面？
        
        # 沿channel融合
        attn_outputs_K = self.aggHeadK(attn_outputs_K)  # b * n * c
        attn_outputs_K = rearrange(attn_outputs_K, "b n c -> n b c")
        return attn_outputs_K


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    可学习的pos_embed, uniad那拿来的
    feature_size格式: [H, W]
    num_pos_feats: channel的一半
    """

    def __init__(self, feature_size, num_pos_feats=128):
        super().__init__()
        num_pos_feats //= 2
        self.feature_size = feature_size  # H, W
        self.row_embed = nn.Embedding(feature_size[0], num_pos_feats)   #参数：需要编码的个数(序列长度)，输出编码的维度
        self.col_embed = nn.Embedding(feature_size[1], num_pos_feats)   #输入：序列长度，输出：序列长度*嵌入维度
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor):
        i = torch.arange(self.feature_size[1], device=tensor.device)  # W
        j = torch.arange(self.feature_size[0], device=tensor.device)  # H
        x_emb = self.col_embed(i)  # W x C // 2         为什么编码维度等于channel//2->之后沿着C维度cat
        y_emb = self.row_embed(j)  # H x C // 2
        pos = torch.cat(
            [
                torch.cat(
                    [x_emb.unsqueeze(0)] * self.feature_size[0], dim=0      #torch.unsqueeze(index):index对应位置前插入一维
                ),  # H x W x C // 2                # x_emb被装入list，list*n，表示把list复制n次，在把list用cat沿dim=0从1*14*128变成14*14*128
                torch.cat(
                    [y_emb.unsqueeze(1)] * self.feature_size[1], dim=1
                ),  # H x W x C // 2
            ],
            dim=-1,     #dim=-1表示最后一个维度
        ).flatten(
            0, 1
        )  # (H X W) X C
        return pos

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])