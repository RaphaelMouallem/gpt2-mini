import torch
from torch.utils.data import Dataset
from pathlib import Path
import random

class LLMDataset(Dataset):
    def __init__(self, raw_data_dir, tokenizer_processor, block_size=512, split="train", target_mb=None):
        self.tokenizer = tokenizer_processor
        self.block_size = block_size
        self.chunks = []

        raw_data_dir = Path(raw_data_dir)
        pattern = "train_*.txt" if split == "train" else "test_final.txt"
        all_files = list(raw_data_dir.glob(pattern))
        
        if split == "train" and target_mb:
            random.shuffle(all_files)
            selected_files = []
            current_bytes = 0
            target_bytes = target_mb * 1024 * 1024
            for f in all_files:
                if current_bytes >= target_bytes: break
                selected_files.append(f)
                current_bytes += f.stat().st_size
            files_to_load = selected_files
            print(f"Loading random {target_mb}MB subset ({len(files_to_load)} files) for training...")
        else:
            files_to_load = all_files
            print(f"Loading all {len(files_to_load)} files for {split}...")

        token_ids = []
        for file_path in files_to_load:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("="):
                        cleaned_line = line.replace("@-@", "")
                        encoded = self.tokenizer.encode(cleaned_line)
                        token_ids.extend(encoded.ids)

        chunk_size = block_size + 1
        for i in range(0, len(token_ids) - chunk_size, chunk_size):
            self.chunks.append(token_ids[i : i + chunk_size])

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        tokens = torch.tensor(self.chunks[idx], dtype=torch.long)
        return tokens[:-1], tokens[1:]