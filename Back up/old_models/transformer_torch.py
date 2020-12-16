import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self,attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k ,v, scale=None, attn_mask=None):
        """前向传播.
        Args:
            q: Queries张量，形状为[B, L_q, D_q]
            k: Keys张量，形状为[B, L_k, D_k]
            v: Values张量，形状为[B, L_v, D_v]，一般来说就是k
            scale: 缩放因子，一个浮点标量
            attn_mask: Masking张量，形状为[B, L_q, L_k]

        Returns:
            上下文张量和attetention张量
        """
        attention = torch.bmm(q, k, v, scale)
        if scale:
            attention = torch * scale
        if attn_mask


# https://www.jianshu.com/p/3b550e903e78
