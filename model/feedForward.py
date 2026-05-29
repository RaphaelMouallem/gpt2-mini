from torch import nn
from torch.nn import functional as F

class FeedForward(nn.Module):
    def __init__(self, d_embd : int, dropout : float, exp_factor : int = 4):
        super().__init__()

        hidden_dim = d_embd * exp_factor

        self.w1_3 = nn.Linear(d_embd, 2 * hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_embd, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x : (batch, seq, d_embd)

        #(batch, seq, d_embd) -> (batch, seq, 8 * d_embd)
        w1_3_out = self.w1_3(x)

        #(batch, seq, 8 * d_embd) -> (batch, seq, 4 * d_embd)
        gate_part, value_part = w1_3_out.chunk(2, dim = -1)

        gate = F.gelu(gate_part)

        swiglu = gate * value_part

        #(batch, seq, d_embd)
        output = self.w2(swiglu)

        return self.dropout(output)