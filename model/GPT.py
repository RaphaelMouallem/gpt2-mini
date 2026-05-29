import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from model.attentionBlock import AttentionBlock

class GPT(nn.Module):
    def __init__(self, config, use_checkpointing : bool = False):
        super().__init__()
        self.config = config
        self.use_checkpointing = use_checkpointing
        
        self.transformer = nn.ModuleDict(dict(
            #(B, T) -> Embeddings (B, T, C)
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            
            h = nn.ModuleList([
                AttentionBlock(
                    n_head=config.n_head, 
                    d_emb=config.n_embd, 
                    block_size=config.block_size, 
                    dropout=config.dropout
                ) 
                for _ in range(config.n_layer)
            ]),
            
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight Tying with (WTE) 
        self.transformer.wte.weight = self.lm_head.weight 
        self.apply(self._init_weights)
        print(f"GPT model initialized with {self.get_num_params()/1e6:.2f}M parameters")

    def get_num_params(self, non_embedding=False):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            #Xavier/Kaiming Uniform
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            #Xavier/Kaiming Uniform
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            #LayerNorm b = 0 and w = 1
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        
        # (B, T, C)
        tok_emb = self.transformer.wte(idx)
        
        x = tok_emb
        
        for block in self.transformer.h: 
            if self.use_checkpointing:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
            
        # (B, T, C)
        x = self.transformer.ln_f(x)
        
        # (B, T, V) - V=vocab_size
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = targets[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p = None, repetition_penalty : float = 1.1, stop_token_ids = None):
        for _ in range(max_new_tokens):
            B, T = idx.size()
            idx_cond = idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)

            if temperature > 0:
                logits = logits[:, -1, :] / temperature
                for b in range(B):
                    prev = torch.unique(idx[b])

                    if stop_token_ids is not None:
                        mask = torch.ones_like(prev, dtype=torch.bool)
                        for stop_id in stop_token_ids:
                            mask &= (prev != stop_id)
                        prev = prev[mask]
                        
                    l = logits[b, prev]
                    logits[b, prev] = torch.where(l < 0, l * repetition_penalty, l / repetition_penalty)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
                idx = torch.cat((idx, idx_next), dim=1)
                continue

            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            if top_k is not None and top_k > 0:
                top_k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            if(stop_token_ids is not None and idx_next.item() in stop_token_ids):
                break

            idx = torch.cat((idx, idx_next), dim=1)

        return idx