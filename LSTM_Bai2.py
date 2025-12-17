import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class AttentionLayer(nn.Module):

    def __init__(self, enc_hid_dim: int, dec_hid_dim: int, attn_dim: int = 256):
        super().__init__()
        self.attn_enc = nn.Linear(enc_hid_dim, attn_dim, bias=False)
        self.attn_dec = nn.Linear(dec_hid_dim, attn_dim, bias=False)
        self.v_proj = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, encoder_out, decoder_hidden, mask=None):
        # encoder_out: [Batch, Seq, Enc_Hid]
        # decoder_hidden: [Batch, Dec_Hid]
        
        # Score = v^T * tanh(W_h * h_enc + W_s * s_dec)
        energy = torch.tanh(self.attn_enc(encoder_out) + self.attn_dec(decoder_hidden).unsqueeze(1))
        scores = self.v_proj(energy).squeeze(-1) # [Batch, Seq]

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        
        # Context = Sum(weights * encoder_out)
        context = torch.bmm(weights.unsqueeze(1), encoder_out).squeeze(1)
        return context, weights

class BiEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(
            emb_dim, hid_dim, n_layers, 
            batch_first=True, bidirectional=True, 
            dropout=dropout if n_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        x = self.dropout(self.embedding(x))
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, (hidden, cell) = self.rnn(packed)
        output, _ = pad_packed_sequence(output, batch_first=True, total_length=x.size(1))
        return output, (hidden, cell)

class AttnDecoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, enc_out_dim, dropout, pad_idx):
        super().__init__()
        self.output_dim = vocab_size
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.attention = AttentionLayer(enc_out_dim, hid_dim)
        
        self.rnn = nn.LSTM(
            emb_dim + enc_out_dim, hid_dim, n_layers, 
            batch_first=True, dropout=dropout if n_layers > 1 else 0
        )
        self.fc_out = nn.Linear(hid_dim + enc_out_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tok, hidden, cell, enc_out, mask):
        input_tok = input_tok.unsqueeze(1) 
        embedded = self.dropout(self.embedding(input_tok)).squeeze(1)

        # Calculate Attention
        # Dùng hidden state lớp cuối cùng của decoder để tính attention
        dec_top_hidden = hidden[-1]
        context, _ = self.attention(enc_out, dec_top_hidden, mask)

        # Input cho LSTM ghép giữa Embedding từ và Context Vector
        rnn_input = torch.cat((embedded, context), dim=1).unsqueeze(1)
        
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        output = output.squeeze(1)
        
        # Dự đoán từ tiếp theo
        prediction = self.fc_out(torch.cat((output, context), dim=1))
        
        return prediction, hidden, cell

class Seq2SeqBahdanau(nn.Module):
    def __init__(self, vocab, hid_dim=256, n_layers=3, dropout=0.1):
        super().__init__()
        self.vocab = vocab
        pad_idx = vocab.pad_idx
        emb_dim = hid_dim 

        self.encoder = BiEncoder(len(vocab.src_w2i), emb_dim, hid_dim, n_layers, dropout, pad_idx)
        
        # Output của Encoder là 2 chiều (hid_dim * 2)
        enc_out_dim = hid_dim * 2
        self.decoder = AttnDecoder(len(vocab.tgt_w2i), emb_dim, hid_dim, n_layers, enc_out_dim, dropout, pad_idx)

        self.fc_h = nn.Linear(enc_out_dim, hid_dim)
        self.fc_c = nn.Linear(enc_out_dim, hid_dim)

    def _bridge_states(self, hidden, cell):
        # hidden: [n_layers*2, batch, hid]
        # Cầ ghép đúng chiều forward và backward của cùng 1 layer
        L, _, B, H = hidden.view(self.encoder.rnn.num_layers, 2, -1, self.encoder.rnn.hidden_size).shape
        
        h_reshape = hidden.view(L, 2, B, H)
        c_reshape = cell.view(L, 2, B, H)
        
        h_cat = torch.cat([h_reshape[:, 0], h_reshape[:, 1]], dim=-1) # [L, B, 2*H]
        c_cat = torch.cat([c_reshape[:, 0], c_reshape[:, 1]], dim=-1)
        
        new_h = torch.tanh(self.fc_h(h_cat))
        new_c = torch.tanh(self.fc_c(c_cat))
        return new_h.contiguous(), new_c.contiguous()

    def forward(self, src, src_len, tgt, teacher_forcing=True):
        enc_out, (h, c) = self.encoder(src, src_len)
        h, c = self._bridge_states(h, c) # Init decoder state
        
        outputs = torch.zeros(src.size(0), tgt.size(1), self.decoder.output_dim).to(src.device)
        mask = (src == self.vocab.pad_idx)
        
        inp = tgt[:, 0]
        for t in range(1, tgt.size(1)):
            out, h, c = self.decoder(inp, h, c, enc_out, mask)
            outputs[:, t] = out
            inp = tgt[:, t] if teacher_forcing else out.argmax(1)
            
        return outputs

    def inference(self, src, src_len, max_len=50):
        self.eval()
        with torch.no_grad():
            enc_out, (h, c) = self.encoder(src, src_len)
            h, c = self._bridge_states(h, c)
            mask = (src == self.vocab.pad_idx)
            
            inp = torch.full((src.size(0),), self.vocab.bos_idx, dtype=torch.long, device=src.device)
            preds = []
            
            for _ in range(max_len):
                out, h, c = self.decoder(inp, h, c, enc_out, mask)
                token = out.argmax(1)
                preds.append(token.unsqueeze(1))
                if (token == self.vocab.eos_idx).all(): break
                inp = token
                
            return torch.cat(preds, dim=1)