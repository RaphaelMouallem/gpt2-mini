# GPT-2 Mini — Built from Scratch in PyTorch

A decoder-only transformer built from the ground up in PyTorch, trained on Wikipedia 
and fine-tuned on conversational data. This was a learning project with one goal: 
understand how LLMs actually work at the code level, not just the API level.

## Architecture

- 12 transformer blocks, 12 self-attention heads, 768 embedding dimensions
- 151M parameters
- RoPE (Rotary Position Embedding) implemented from scratch using complex number rotations
- SwiGLU activation in the feed-forward blocks (same family as LLaMA)
- Weight tying between token embeddings and the LM head
- Pre-norm architecture with LayerNorm before attention and FFN

## Training

- Pre-trained on a Wikipedia text corpus
- Fine-tuned on a user/bot conversation dataset using masked labels
  (model only learns to predict bot responses, not user inputs)
- Trained across Apple Silicon (MPS) and Google Colab
- bfloat16 mixed precision, gradient accumulation, cosine LR schedule with warmup,
  gradient clipping
- Incremental head scaling strategy: started at 6 heads, scaled to 8, then 12
  to manage RAM constraints during training

## Inference

Supports top-k sampling, nucleus sampling (top-p), temperature scaling, 
and repetition penalty. Includes a basic chat loop for dialog interaction.

## What I learned

Most of the interesting problems were in the details — getting RoPE to apply 
correctly across the head dimension, implementing masked fine-tuning so the loss 
only backpropagates through bot turns, and keeping training stable on MPS without 
running out of memory. The model works, barely holds a conversation, and that's 
kind of the point — building it taught me more about how these systems behave 
than anything else I've done.

## Stack

PyTorch, HuggingFace Tokenizers, PEFT/LoRA (configured for fine-tuning), 
Jupyter, Apple MPS, Google Colab