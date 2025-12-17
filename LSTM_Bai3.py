import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LuongEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self.proj = nn.Linear(hid_dim * 2, hid_dim, bias=False)

    def forward(self, x, lengths):
        x = self.dropout(self.embedding(x))
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, (h, c) = self.rnn(packed)
        outputs, _ = pad_packed_sequence(out_packed, batch_first=True, total_length=x.size(1))

        outputs = self.proj(outputs) 
        return outputs, (h, c)

class LuongDecoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout, pad_idx):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)

        self.rnn = nn.LSTM(emb_dim + hid_dim, hid_dim, n_layers, batch_first=True, dropout=dropout)
        
        self.concat = nn.Linear(hid_dim * 2, hid_dim)
        self.out = nn.Linear(hid_dim, vocab_size)

    def forward(self, input_tk, hidden, cell, enc_outs, mask, last_context=None):
        B = input_tk.size(0)
        embedded = self.dropout(self.embedding(input_tk))

        if last_context is None:
            last_context = torch.zeros(B, self.rnn.hidden_size).to(input_tk.device)
            
        rnn_in = torch.cat([embedded, last_context], dim=1).unsqueeze(1)
        rnn_out, (hidden, cell) = self.rnn(rnn_in, (hidden, cell))
        
        # --- Luong Dot Attention ---
        # Score(h_t, h_s) = h_t . h_s
        # rnn_out: [B, 1, H], enc_outs: [B, Seq, H]
        scores = torch.bmm(rnn_out, enc_outs.transpose(1, 2)) # -> [B, 1, Seq]
        
        if mask is not None:
            scores.masked_fill_(mask.unsqueeze(1), -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights, enc_outs).squeeze(1) # -> [B, H]

        concat_vector = torch.cat([rnn_out.squeeze(1), context], dim=1)
        attn_hidden = torch.tanh(self.concat(concat_vector))
        
        # Dự đoán
        prediction = self.out(attn_hidden)
        
        return prediction, hidden, cell, context

class Seq2SeqLuong(nn.Module):
    def __init__(self, vocab, hid_dim=256, n_layers=3, dropout=0.1):
        super().__init__()
        self.vocab = vocab
        pad_idx = vocab.pad_idx
        emb_dim = hid_dim
        
        self.encoder = LuongEncoder(len(vocab.src_w2i), emb_dim, hid_dim, n_layers, dropout, pad_idx)
        self.decoder = LuongDecoder(len(vocab.tgt_w2i), emb_dim, hid_dim, n_layers, dropout, pad_idx)
        
        self.bridge_h = nn.Linear(hid_dim * 2, hid_dim)
        self.bridge_c = nn.Linear(hid_dim * 2, hid_dim)

    def _init_dec_state(self, h, c):
        # h, c: [n_layers*2, batch, hid]
        L = self.encoder.rnn.num_layers
        h = h.view(L, 2, -1, self.encoder.rnn.hidden_size)
        c = c.view(L, 2, -1, self.encoder.rnn.hidden_size)
        
        h_cat = torch.cat([h[:,0], h[:,1]], dim=-1)
        c_cat = torch.cat([c[:,0], c[:,1]], dim=-1)
        
        dec_h = torch.tanh(self.bridge_h(h_cat))
        dec_c = torch.tanh(self.bridge_c(c_cat))
        return dec_h, dec_c

    def forward(self, src, src_len, tgt, teacher_forcing=True):
        enc_out, (h, c) = self.encoder(src, src_len)
        h, c = self._init_dec_state(h, c)
        
        outputs = torch.zeros(src.size(0), tgt.size(1), self.decoder.vocab_size).to(src.device)
        mask = (src == self.vocab.pad_idx)
        
        inp = tgt[:, 0]
        context = None 
        
        for t in range(1, tgt.size(1)):
            out, h, c, context = self.decoder(inp, h, c, enc_out, mask, context)
            outputs[:, t] = out
            inp = tgt[:, t] if teacher_forcing else out.argmax(1)
            
        return outputs

    def inference(self, src, src_len, max_len=50):
        self.eval()
        with torch.no_grad():
            enc_out, (h, c) = self.encoder(src, src_len)
            h, c = self._init_dec_state(h, c)
            mask = (src == self.vocab.pad_idx)
            
            inp = torch.full((src.size(0),), self.vocab.bos_idx, dtype=torch.long, device=src.device)
            preds = []
            context = None
            
            for _ in range(max_len):
                out, h, c, context = self.decoder(inp, h, c, enc_out, mask, context)
                tok = out.argmax(1)
                preds.append(tok.unsqueeze(1))
                if (tok == self.vocab.eos_idx).all(): break
                inp = tok
                
            return torch.cat(preds, dim=1)