import torch
from torch import nn
from torch.nn import functional as F

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # time indices (0,1,2,...seq-1): shape (end,)
    t = torch.arange(end, device=freqs.device)

    # outer product → (end, dim/2) where each row is t * freq
    freqs = torch.outer(t, freqs).float()

    #complex rotations: cos(freq) + i sin(freq)
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    #(a,b,c,d,...) -> [(a + ib), (c + id), ...]
    #(batch, heads, seq, head_dim) -> (batch, heads, seq, head_dim/2)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0) # (1, 1, T, d_k/2)
    
    xq_out = torch.view_as_real(xq_ * freqs_cis[:, :, :xq.shape[2]]).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis[:, :, :xk.shape[2]]).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

class MultiSelfAttentionHead(nn.Module):
    def __init__(self, n_heads: int, d_embd: int, block_size: int, dropout: float, in_proj_bias: bool = True, out_proj_bias: bool = True) -> None:
        super().__init__()

        self.in_proj = nn.Linear(d_embd, 3 * d_embd, bias = in_proj_bias)
        self.out_proj = nn.Linear(d_embd, d_embd, bias = out_proj_bias)
        
        self.n_heads = n_heads
        self.d_heads = d_embd // n_heads
        
        # Precompute RoPE
        self.register_buffer("freqs_cis", precompute_freqs_cis(self.d_heads, block_size))
        
        self.resid_dropout = nn.Dropout(dropout)
        
    def forward(self, x, causal_mask: bool = True):
        # x (latent): (batch, seq_len, dim)
        input_shape = x.shape
        batch_size, sequence_length, dimension = input_shape
        
        # Shape for heads: (Batch_Size, Seq_Len, H, Dim / H)
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_heads)

        # (bact, seq_len, dim) -> (batch, seq_len, dim * 3) -> 3 tensors of shape (batch, seq_len, dim)
        q, k, v = self.in_proj(x).chunk(3, dim = -1)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, H, Seq_Len, Dim / H)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)
        
        freqs_cis_T = self.freqs_cis[:sequence_length].to(q.device)
        q, k = apply_rotary_emb(q, k, freqs_cis_T)

        output = F.scaled_dot_product_attention(q, k, v, is_causal = causal_mask)

        output = output.transpose(1, 2) 

        output = output.reshape(input_shape) 

        output = self.resid_dropout(self.out_proj(output)) 
        
        return output