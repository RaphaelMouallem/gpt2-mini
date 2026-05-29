from torch import nn
from torch.nn import functional as F

from model.multiSelfAttentionHead import MultiSelfAttentionHead
from model.feedForward import FeedForward

class AttentionBlock(nn.Module):
    def __init__(self, n_head : int, d_emb : int, block_size : int, dropout : float):
        super().__init__()

        self.ln_1 = nn.LayerNorm(d_emb, bias = True)
        self.ln_2 = nn.LayerNorm(d_emb, bias = True)

        self.attn = MultiSelfAttentionHead(
            n_heads = n_head,
            d_embd = d_emb,
            block_size = block_size,
            dropout = dropout
        )

        self.ffn = FeedForward(d_embd = d_emb, dropout = dropout)

    def forward(self, x):
        #pre norm
        
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))

        return x