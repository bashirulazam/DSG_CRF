from torch import Tensor
import torch
import torch.nn as nn
from lib.within_triplet_static_transformer import Transformer
import math

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, float(emb_size), 2)* math.log(10000) / float(emb_size))
        pos = torch.arange(0, float(maxlen)).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class MyUnaryAttTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(MyUnaryAttTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.att_generator = nn.Linear(emb_size, tgt_vocab_size)
        self.spa_generator = nn.Linear(emb_size, tgt_vocab_size)
        self.con_generator = nn.Linear(emb_size, tgt_vocab_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)


    @property
    def device(self):
        return next(self.parameters()).device
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def generate_subsequent_mask_from_im_idx(self, im_idx):
        sz = len(im_idx)
        mask = torch.zeros((sz, sz), device=self.device)
        for i in range(sz):
            image_idx = im_idx[i]
            mask[i, image_idx == im_idx] = 1
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_mask(self, src_emb, tgt_emb):
        src_seq_len = src_emb.shape[0]
        tgt_seq_len = tgt_emb.shape[0]

        tgt_mask = torch.zeros((tgt_seq_len, tgt_seq_len), device=self.device).type(torch.bool)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=self.device).type(torch.bool)

        return src_mask, tgt_mask

    def forward(self,
                src_emb: Tensor,
                trg: Tensor,
                ):

        tgt_emb = self.tgt_tok_emb(trg)
        src_mask,  tgt_mask = self.create_mask(src_emb, tgt_emb)
        outs, memory = self.transformer(src_emb, tgt_emb,  src_mask, tgt_mask)
        return  self.att_generator(outs), memory
        #This is for Encoder Only Ablation
        #return  self.att_generator(memory), memory

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)


