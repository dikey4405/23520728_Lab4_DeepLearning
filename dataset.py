import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict

from vocab import Vocabulary 

class TranslationDataset(Dataset):
    def __init__(self, json_path: str, vocab: Vocabulary):
        super().__init__()
        self.vocab = vocab
        with open(json_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[index]
        
        # Lấy raw text
        src_raw = item[self.vocab.src_lang]
        tgt_raw = item[self.vocab.tgt_lang]

        # Mã hóa thành vector
        src_tensor = self.vocab.text_to_indices(src_raw, self.vocab.src_lang)
        tgt_tensor = self.vocab.text_to_indices(tgt_raw, self.vocab.tgt_lang)

        return {
            "src": src_tensor,
            "tgt": tgt_tensor
        }

def translation_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Hàm gom batch và padding dữ liệu."""
    src_batch = [x["src"] for x in batch]
    tgt_batch = [x["tgt"] for x in batch]

    # Padding với giá trị 0 
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=0)

    # Tính độ dài thực tế cho pack_padded_sequence
    src_lens = torch.tensor([len(x) for x in src_batch], dtype=torch.long)
    tgt_lens = torch.tensor([len(x) for x in tgt_batch], dtype=torch.long)

    return {
        "src": src_padded,
        "tgt": tgt_padded,
        "src_len": src_lens,
        "tgt_len": tgt_lens
    }

