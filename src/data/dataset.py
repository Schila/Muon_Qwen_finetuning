import os
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

class MoonDataset(Dataset):
    def __init__(self, dataset_name, tokenizer, max_length=512):
        self.dataset_name = dataset_name.replace("/", "_")
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Кэширование токенизированного датасета
        cache_file = f"{self.dataset_name}_tokens.bin"
        if os.path.exists(cache_file):
            print(f"Loading cached tokens from {cache_file}...")
            self.tokens = torch.load(cache_file)
        else:
            print("Tokenizing dataset (this might take a while)...")
            raw_dataset = load_dataset(dataset_name, trust_remote_code=True)
            all_tokens = []
            for text in tqdm(raw_dataset["train"]["text"], desc="Tokenizing"):
                all_tokens.extend(self.tokenizer.encode(text, add_special_tokens=True))
            self.tokens = torch.tensor(all_tokens, dtype=torch.long)
            torch.save(self.tokens, cache_file)

    def __len__(self):
        return len(self.tokens) // self.max_length

    def __getitem__(self, idx):
        start = idx * self.max_length
        return self.tokens[start : start + self.max_length]

def get_dataloaders(model_name: str, batch_size: int, max_length: int = 512):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dataset = MoonDataset("Elriggs/openwebtext-100k", tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader, tokenizer
