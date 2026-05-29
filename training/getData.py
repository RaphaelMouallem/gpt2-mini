import json
import torch
from torch.utils.data import Dataset

def create_masked_tensors(formatted_text: str, tokenizer_processor, max_length: int, IGNORE_INDEX : int = -100, EOS_ID : int = 50256):
    tokenizer = tokenizer_processor.get_tokenizer()

    encoding = tokenizer.encode(formatted_text)
    input_ids = encoding.ids
    labels = list(input_ids)

    mask_labels = True

    for i in range(len(labels)):
        token_id = input_ids[i]

        if(mask_labels):
            labels[i] = IGNORE_INDEX

        if(token_id == EOS_ID):
            mask_labels = not mask_labels

    input_ids = input_ids[:max_length]
    labels = labels[:max_length]

    padding_length = max_length - len(input_ids)

    input_ids = input_ids + [EOS_ID] * padding_length
    labels = labels + [IGNORE_INDEX] * padding_length

    input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return input_ids_tensor, labels_tensor

def load_data(file_path: str):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded {len(data)} conversations from {file_path}")
        return data
    except Exception as e:
        print(f"ERROR loading data : {e}")
        return []
    
def process_conversation_for_sft(conversations) -> str:
    EOS = "<|endoftext|>"
    full_text = ""

    for turn in conversations: 
        full_text += f"{turn['user']}{EOS}{turn['model']}{EOS}"
            
    return full_text

class ConversationDataset(Dataset):
    def __init__(self, raw_data, tokenizer_processor, max_length: int, IGNORE_INDEX : int = -100):
        self.raw_data = raw_data
        self.tokenizer_processor = tokenizer_processor
        self.max_length = max_length
        self.IGNORE_INDEX = IGNORE_INDEX

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        example = self.raw_data[idx]
        conversations = example["conversations"]
        formatted_text = process_conversation_for_sft(conversations)

        input_ids, labels = create_masked_tensors(
            formatted_text,
            self.tokenizer_processor,
            self.max_length
        )

        if input_ids is None or len(input_ids) != self.max_length:
            print("error")
            return {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'labels': torch.full((self.max_length,), self.IGNORE_INDEX, dtype=torch.long)
            }

        return {
            'input_ids': input_ids,
            'labels': labels
        }