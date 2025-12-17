import json
import re
import torch
from typing import List, Dict, Tuple

class Vocabulary:
    def __init__(self, data_path: str, src_lang: str, tgt_lang: str):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        # Định nghĩa các token đặc biệt
        self.token_bos = "<bos>"
        self.token_eos = "<eos>"
        self.token_pad = "<pad>"
        self.token_unk = "<unk>"
        
        self.specials = [self.token_pad, self.token_bos, self.token_eos, self.token_unk]
        self.pad_idx = 0
        self.bos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3

        # Xây dựng từ điển
        src_vocab_set, tgt_vocab_set = self._build_vocab(data_path)

        self.src_w2i, self.src_i2w = self._create_mapping(src_vocab_set)
        self.tgt_w2i, self.tgt_i2w = self._create_mapping(tgt_vocab_set)

    def _build_vocab(self, path: str) -> Tuple[set, set]:
        """Đọc file và tách từ để tạo tập từ vựng."""
        s_vocab = set()
        t_vocab = set()
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        for entry in data:
            s_text = self.clean_text(entry[self.src_lang])
            t_text = self.clean_text(entry[self.tgt_lang])
            
            s_vocab.update(s_text.split())
            t_vocab.update(t_text.split())
            
        return s_vocab, t_vocab

    def _create_mapping(self, vocab_set: set) -> Tuple[Dict, Dict]:
        """Tạo mapping từ word->index và index->word."""
        tokens = self.specials + sorted(list(vocab_set))
        w2i = {word: idx for idx, word in enumerate(tokens)}
        i2w = {idx: word for word, idx in w2i.items()}
        return w2i, i2w

    def clean_text(self, text: str) -> str:
        """Tiền xử lý chuỗi ký tự."""
        text = text.lower()
        # Giữ lại các ký tự chữ cái bao gồm tiếng Việt
        text = re.sub(r"([^\w\sÀ-ỹà-ỹ])", r" \1 ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def text_to_indices(self, text: str, lang: str) -> torch.Tensor:
        """Chuyển câu văn bản thành tensor các chỉ số (indices)."""
        mapping = self.src_w2i if lang == self.src_lang else self.tgt_w2i
        processed_text = self.clean_text(text)
        tokens = processed_text.split()
        
        encoded = [mapping.get(token, mapping[self.token_unk]) for token in tokens]
        encoded = [mapping[self.token_bos]] + encoded + [mapping[self.token_eos]]
        
        return torch.tensor(encoded, dtype=torch.long)
    
    def indices_to_text(self, indices: torch.Tensor, lang: str) -> str:
        """Chuyển tensor chỉ số ngược lại thành câu văn bản."""
        idx_list = indices.tolist()
        mapping = self.src_i2w if lang == self.src_lang else self.tgt_i2w
        
        result_tokens = []
        for idx in idx_list:
            word = mapping.get(int(idx), self.token_unk)
            if word == self.token_eos:
                break
            if word in self.specials:
                continue
            result_tokens.append(word)

        return " ".join(result_tokens)