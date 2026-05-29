from tokenizers import Tokenizer

class TokenizerProcessor:
    def __init__(self, tokenizer_file_path: str, eos: bool = False):
        self.tokenizer = Tokenizer.from_file(tokenizer_file_path)

        if(eos):
            self.eos_id = 50256
        
    def get_tokenizer(self):
        return self.tokenizer
    
    def encode(self, text):
        return self.tokenizer.encode(text)
    
    def decode(self, ids):
        return self.tokenizer.decode(ids)
    
    def enable_padding(self, direction: str = "right"):        
        self.tokenizer.enable_padding(
            direction=direction,
            pad_id=self.eos_id, 
            pad_token="<|endoftext|>",
        )