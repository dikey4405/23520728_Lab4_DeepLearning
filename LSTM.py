import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Tuple

class VanillaEncoder(nn.Module):

    def __init__(self, vocab_size: int, emb_dim: int, hid_dim: int, n_layers: int, dropout: float, pad_idx: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0
        )
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x: [Batch, Seq_Len]
        emb = self.dropout_layer(self.embedding(x))

        # Pack sequence để tính toán cho các câu có độ dài khác nhau
        packed_emb = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        packed_out, (hidden, cell) = self.rnn(packed_emb)
        
        outputs, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
        
        return outputs, (hidden, cell)

class VanillaDecoder(nn.Module):

    def __init__(self, vocab_size: int, emb_dim: int, hid_dim: int, n_layers: int, dropout: float, pad_idx: int):
        super().__init__()
        self.output_dim = vocab_size
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0
        )
        
        self.fc_out = nn.Linear(hid_dim, vocab_size)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, input_tok: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor):
        # input_tok: [Batch] (1 time step)
        input_tok = input_tok.unsqueeze(1) # -> [Batch, 1]
        
        emb = self.dropout_layer(self.embedding(input_tok))
        
        # LSTM forward
        output, (hidden, cell) = self.rnn(emb, (hidden, cell))
        
        # output: [Batch, 1, Hid] -> squeeze -> [Batch, Hid]
        prediction = self.fc_out(output.squeeze(1))
        
        return prediction, hidden, cell

class Seq2Seq(nn.Module):

    def __init__(self, vocab, hid_dim=256, n_layers=3, dropout=0.1):
        super().__init__()
        self.vocab = vocab
        pad_idx = vocab.pad_idx
        src_dim = len(vocab.src_w2i)
        tgt_dim = len(vocab.tgt_w2i)
        
        emb_dim = hid_dim 

        self.encoder = VanillaEncoder(src_dim, emb_dim, hid_dim, n_layers, dropout, pad_idx)
        self.decoder = VanillaDecoder(tgt_dim, emb_dim, hid_dim, n_layers, dropout, pad_idx)

    def forward(self, src, src_len, tgt, teacher_forcing=True):
        batch_size = src.size(0)
        max_len = tgt.size(1)
        vocab_size = self.decoder.output_dim
        
        # Tensor chứa kết quả dự đoán
        outputs = torch.zeros(batch_size, max_len, vocab_size).to(src.device)
        
        # Encoder phase
        _, (hidden, cell) = self.encoder(src, src_len)
        
        # Decoder phase
        # Bắt đầu bằng token <bos>
        curr_input = tgt[:, 0]
        
        for t in range(1, max_len):
            output, hidden, cell = self.decoder(curr_input, hidden, cell)
            outputs[:, t] = output
            
            # Chọn token có xác suất cao nhất
            top1 = output.argmax(1)
            
            # Teacher Forcing
            curr_input = tgt[:, t] if teacher_forcing else top1
            
        return outputs

    def inference(self, src, src_len, max_len=50):
        """Hàm dự đoán """
        self.eval()
        batch_size = src.size(0)
        
        with torch.no_grad():
            _, (hidden, cell) = self.encoder(src, src_len)
            
            curr_input = torch.full((batch_size,), self.vocab.bos_idx, dtype=torch.long, device=src.device)
            
            preds = []
            finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)
            
            for _ in range(max_len):
                output, hidden, cell = self.decoder(curr_input, hidden, cell)
                pred_token = output.argmax(1)
                
                preds.append(pred_token.unsqueeze(1))
                
                # Check EOS
                finished |= (pred_token == self.vocab.eos_idx)
                if finished.all():
                    break
                    
                curr_input = pred_token
                
        return torch.cat(preds, dim=1)