class GPTConfig:
    def __init__(self, **kwargs):
        self.vocab_size = kwargs.get('vocab_size', 50257)   # From your BPE training
        self.block_size = kwargs.get('block_size', 1024)    # Sequence length
        self.n_layer = kwargs.get('n_layer', 12)            # Number of transformer blocks
        self.n_head = kwargs.get('n_head', 12)              # Number of attention heads
        self.n_embd = kwargs.get('n_embd', 768)             # Embedding dimension (d_model)
        self.dropout = kwargs.get('dropout', 0.1)           # Dropout rate
        self.bias = kwargs.get('bias', False)               # Use bias in Linear layers (modern practice often excludes it)
