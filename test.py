import argparse
import yaml
import torch
import evaluate
from torch.utils.data import DataLoader
from tqdm import tqdm

from vocab import Vocabulary
from dataset import TranslationDataset, translation_collate_fn

from LSTM import Seq2Seq
from LSTM_Bai2 import Seq2SeqBahdanau
from LSTM_Bai3 import Seq2SeqLuong

class Evaluator:
    def __init__(self, config_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rouge_metric = evaluate.load("rouge")
        
        # Load config và setup
        self.cfg = self._load_config(config_path)
        self._setup_resources()

    def _load_config(self, path):
        return yaml.safe_load(open(path, "r", encoding="utf-8"))

    def _setup_resources(self):
        print(">> [Test] Loading Resources...")
        data_cfg = self.cfg["data"]
        
        # 1. Load Vocab
        self.vocab = Vocabulary(data_cfg["train"], data_cfg["src_lang"], data_cfg["tgt_lang"])
        
        # 2. Load Test Dataset
        test_ds = TranslationDataset(data_cfg["test"], self.vocab)
        self.test_loader = DataLoader(
            test_ds, 
            batch_size=self.cfg["train"]["batch_size"], 
            shuffle=False, 
            collate_fn=translation_collate_fn
        )
        
        # 3. Load Model Architecture & Weights
        self._load_model()

    def _load_model(self):
        ckpt_path = self.cfg["train"]["save_path"]
        print(f"[Test] Loading Checkpoint: {ckpt_path}")
        
        if not torch.cuda.is_available():
            ckpt = torch.load(ckpt_path, map_location="cpu")
        else:
            ckpt = torch.load(ckpt_path)
            
        model_cfg = self.cfg["model"]
        name = model_cfg["name"].lower()
        hid_dim = model_cfg.get("hidden_dim", 256)
        n_layers = model_cfg.get("num_layers", 3)
        dropout = model_cfg.get("dropout", 0.1)

        # Khởi tạo kiến trúc model 
        if name == "seq2seq":
            self.model = Seq2Seq(self.vocab, hid_dim, n_layers, dropout)
        elif name == "seq2seqbahdanau":
            self.model = Seq2SeqBahdanau(self.vocab, hid_dim, n_layers, dropout)
        elif name == "seq2seqluong":
            self.model = Seq2SeqLuong(self.vocab, hid_dim, n_layers, dropout)
        else:
            raise ValueError(f"Unknown model name: {name}")

        # Load weights

        state_dict = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state_dict)
        
        self.model.to(self.device)
        self.model.eval()

    def run_inference(self, num_examples=3):
        print(" [Test] Running Inference...")
        
        predictions = []
        references = []
        shown_count = 0
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader):
                src = batch["src"].to(self.device)
                src_len = batch["src_len"].to(self.device)
                tgt = batch["tgt"] #  lấy text để đối chiếu
                
                # Gọi hàm inference của model
                pred_ids = self.model.inference(src, src_len)
                
                for i in range(src.size(0)):
                    # Decode IDs sang Text
                    pred_text = self.vocab.indices_to_text(pred_ids[i], self.vocab.tgt_lang)
                    ref_text = self.vocab.indices_to_text(tgt[i], self.vocab.tgt_lang)
                    src_text = self.vocab.indices_to_text(src[i], self.vocab.src_lang)
                    
                    predictions.append(pred_text)
                    references.append(ref_text)
                    
                    #  ví dụ
                    if shown_count < num_examples:
                        print(f"\n--- Example {shown_count + 1} ---")
                        print(f"SRC : {src_text}")
                        print(f"REF : {ref_text}")
                        print(f"PRED: {pred_text}")
                        shown_count += 1

        # Tính điểm ROUGE
        scores = self.rouge_metric.compute(predictions=predictions, references=references)
        
        print("\n TEST RESULTS ")
        for k, v in scores.items():
            print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    args = parser.parse_args()
    
    evaluator = Evaluator(args.config)
    evaluator.run_inference()