import torch

@torch.no_grad()
def generate_text(model, tokenizer_processor, prompt: str, max_new_tokens: int, temperature: float = 0.8, top_k: int = 40, top_p = 0.9, stop_token_ids = None):
    device = next(model.parameters()).device 
    model.eval()

    input_ids = tokenizer_processor.encode(prompt).ids

    idx = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0) 

    generated_tokens_tensor = model.generate(
        idx, 
        max_new_tokens=max_new_tokens, 
        temperature=temperature, 
        top_k=top_k,
        top_p=top_p,
        stop_token_ids=stop_token_ids
    )

    generated_tokens_list = generated_tokens_tensor[0].tolist()
    
    decoded_text = tokenizer_processor.decode(generated_tokens_list) 

    model.train() 

    return decoded_text

def chat_with_model(model, tokenizer_processor):
    chat_history = ""
    print("Type 'quit' to stop")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']: break
        
        chat_history += f"{user_input}<|endoftext|>"

        response_full = generate_text(
            model, tokenizer_processor, 
            prompt=chat_history, 
            max_new_tokens=100,
            stop_token_ids=[50256],
            top_p=0.9,
            temperature=0.7
        )
        
        model_reply = response_full[len(chat_history):].replace("<|endoftext|>", "").strip()
        
        print(f"Bot: {model_reply}")
        
        chat_history += f"{model_reply}<|endoftext|>"