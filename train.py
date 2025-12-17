import os
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import evaluate

from vocab import Vocabulary
from dataset import TranslationDataset, translation_collate_fn

from LSTM import Seq2Seq
from LSTM_Bai2 import Seq2SeqBahdanau
from LSTM_Bai3 import Seq2SeqLuong

class Trainer:
    def __init__(self, config_path):
        self.cfg = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rouge_metric = evaluate.load("rouge")
        
        # Khởi tạo dữ liệu và model
        self._prepare_data()
        self._build_model()
        self._setup_optimization()

    def _load_config(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _prepare_data(self):
        print("[Data] Preparing Datasets...")
        data_cfg = self.cfg["data"]
        train_cfg = self.cfg["train"]
        
        # Load Vocab
        self.vocab = Vocabulary(data_cfg["train"], data_cfg["src_lang"], data_cfg["tgt_lang"])
        
        # Load Datasets
        train_ds = TranslationDataset(data_cfg["train"], self.vocab)
        dev_ds = TranslationDataset(data_cfg["dev"], self.vocab)
        
        # Dataloaders
        self.train_loader = DataLoader(
            train_ds, 
            batch_size=train_cfg["batch_size"], 
            shuffle=True, 
            collate_fn=translation_collate_fn
        )
        self.dev_loader = DataLoader(
            dev_ds, 
            batch_size=train_cfg["batch_size"], 
            shuffle=False, 
            collate_fn=translation_collate_fn
        )
        
        print(f" Source Vocab: {len(self.vocab.src_w2i)}")
        print(f" Target Vocab: {len(self.vocab.tgt_w2i)}")

    def _build_model(self):
        """Khởi tạo model """
        model_cfg = self.cfg["model"]
        name = model_cfg["name"].lower()
        
        # Lấy tham số chung từ config
        hid_dim = model_cfg.get("hidden_dim", 256) 
        n_layers = model_cfg.get("num_layers", 3)
        dropout = model_cfg.get("dropout", 0.1)
        
        print(f" [Model] Building: {name.upper()}")
        
        if name == "seq2seq":
            self.model = Seq2Seq(self.vocab, hid_dim, n_layers, dropout)
        elif name == "seq2seqbahdanau":
            self.model = Seq2SeqBahdanau(self.vocab, hid_dim, n_layers, dropout)
        elif name == "seq2seqluong":
            self.model = Seq2SeqLuong(self.vocab, hid_dim, n_layers, dropout)
        else:
            raise ValueError(f"Model '{name}' chưa được hỗ trợ.")
            
        self.model.to(self.device)

    def _setup_optimization(self):
        train_cfg = self.cfg["train"]
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=train_cfg["lr"], 
            weight_decay=train_cfg.get("weight_decay", 0)
        )
        # Bỏ qua token padding khi tính loss
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.vocab.pad_idx)
        self.save_path = train_cfg["save_path"]

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for batch in progress_bar:
            src = batch["src"].to(self.device)
            tgt = batch["tgt"].to(self.device)
            src_len = batch["src_len"].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(src, src_len, tgt, teacher_forcing=True)
            
            # Reshape để tính Loss (Bỏ token đầu tiên của decoder output và target)
            # Output: [Batch, Seq, Vocab] -> [(Batch * Seq-1), Vocab]
            output_flat = output[:, 1:].reshape(-1, output.shape[-1])
            tgt_flat = tgt[:, 1:].reshape(-1)
            
            loss = self.criterion(output_flat, tgt_flat)
            loss.backward()
            
            # Gradient clipping 
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        predictions = []
        references = []
        
        for batch in tqdm(self.dev_loader, desc="Evaluating", leave=False):
            src = batch["src"].to(self.device)
            tgt = batch["tgt"].to(self.device)
            src_len = batch["src_len"].to(self.device)
            
            # 1. Tính Validation Loss (Teacher Forcing)
            out = self.model(src, src_len, tgt, teacher_forcing=True)
            out_flat = out[:, 1:].reshape(-1, out.shape[-1])
            tgt_flat = tgt[:, 1:].reshape(-1)
            loss = self.criterion(out_flat, tgt_flat)
            total_loss += loss.item()
            
            # 2. Tính ROUGE (Inference - Greedy Decoding)

            pred_ids = self.model.inference(src, src_len)
            
            for i in range(src.size(0)):
                pred_text = self.vocab.indices_to_text(pred_ids[i], self.vocab.tgt_lang)
                ref_text = self.vocab.indices_to_text(tgt[i], self.vocab.tgt_lang)
                predictions.append(pred_text)
                references.append(ref_text)
                
        avg_loss = total_loss / len(self.dev_loader)
        rouge_score = self.rouge_metric.compute(predictions=predictions, references=references)
        
        return avg_loss, rouge_score["rougeL"]

    def run(self):
        epochs = self.cfg["train"]["epochs"]
        patience = self.cfg["train"].get("patience", 5)
        best_rouge = 0.0
        no_improve = 0
        
        print("\n>> START TRAINING")
        for epoch in range(1, epochs + 1):
            print(f"\n=== Epoch {epoch}/{epochs} ===")
            
            train_loss = self.train_epoch()
            val_loss, val_rouge = self.evaluate()
            
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss:   {val_loss:.4f} | Val ROUGE-L: {val_rouge:.4f}")
            
            # Checkpoint & Early Stopping
            if val_rouge > best_rouge:
                best_rouge = val_rouge
                no_improve = 0
                
                os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "config": self.cfg,
                    "vocab": self.vocab
                }, self.save_path)
                
                print(f" Model Saved! (New Best ROUGE: {best_rouge:.4f})")
            else:
                no_improve += 1
                
            if no_improve >= patience:
                print("\n Early Stopping .")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    trainer = Trainer(args.config)
    trainer.run()