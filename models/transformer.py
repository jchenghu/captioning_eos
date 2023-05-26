import torch
from models.layers.transformer_layers import FeedForward
from models.layers.generic_layers import EmbeddingLayer
from models.layers.transformer_layers import MultiHeadAttention
from utils.masking import create_pad_mask, create_no_peak_and_pad_mask
from models.captioning_model import CaptioningModel

import torch.nn as nn



class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout_perc):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(dropout_perc)
        self.dropout_2 = nn.Dropout(dropout_perc)

        self.mha_1 = MultiHeadAttention(d_model, num_heads, dropout_perc)
        self.ff = FeedForward(d_model, d_ff, dropout_perc)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.mha_1(q=x2, k=x2, v=x2, mask=mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_perc):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)

        self.dropout_1 = nn.Dropout(dropout_perc)
        self.dropout_2 = nn.Dropout(dropout_perc)
        self.dropout_3 = nn.Dropout(dropout_perc)

        self.mha_2 = MultiHeadAttention(d_model, num_heads, dropout_perc)
        self.mha_1 = MultiHeadAttention(d_model, num_heads, dropout_perc)
        self.ff = FeedForward(d_model, d_ff, dropout_perc)

    def forward(self, x, cross_connection_x, input_attention_mask, cross_attention_mask):

        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.mha_1(q=x2, k=x2, v=x2, mask=input_attention_mask))

        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.mha_2(q=x2, k=cross_connection_x, v=cross_connection_x, mask=cross_attention_mask))

        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x


class Transformer(CaptioningModel):
    def __init__(self, d_model, N_enc, N_dec, ff, num_heads,
                 output_word2idx, max_seq_len, drop_args, rank=0):
        super().__init__()
        self.output_word2idx = output_word2idx
        self.max_seq_len = max_seq_len

        self.N_enc = N_enc
        self.N_dec = N_dec
        self.d_model = d_model

        self.encoders = nn.ModuleList([EncoderLayer(d_model, ff, num_heads, drop_args.enc) for _ in range(N_enc)])
        self.decoders = nn.ModuleList([DecoderLayer(d_model, num_heads, ff, drop_args.dec) for _ in range(N_dec)])

        self.input_embedder_dropout = nn.Dropout(drop_args.enc_input)
        self.input_linear = torch.nn.Linear(2048, d_model)
        self.vocab_linear = torch.nn.Linear(d_model, len(output_word2idx))
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.out_enc_dropout = nn.Dropout(drop_args.other)
        self.out_dec_dropout = nn.Dropout(drop_args.other)

        self.out_embedder = EmbeddingLayer(len(output_word2idx), d_model, drop_args.dec_input)
        self.pos_encoder = nn.Embedding(max_seq_len, d_model)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.trained_steps = 0
        self.rank = rank

    def forward_enc(self, enc_input, enc_input_num_pads):
        pad_mask = create_pad_mask(mask_size=(enc_input.size(0), enc_input.size(1), enc_input.size(1)),
                                   pad_row=enc_input_num_pads,
                                   pad_column=enc_input_num_pads,
                                   rank=self.rank)

        x = self.input_embedder_dropout(self.input_linear(enc_input))
        for i in range(self.N_enc):
            x = self.encoders[i](x=x, mask=pad_mask)
        return x

    def forward_dec(self, cross_input, enc_input_num_pads, dec_input, dec_input_num_pads, apply_log_softmax=False):

        no_peak_and_pad_mask = create_no_peak_and_pad_mask(
                                mask_size=(dec_input.size(0), dec_input.size(1), dec_input.size(1)),
                                num_pads=dec_input_num_pads,
                                rank=self.rank)

        pad_mask = create_pad_mask(mask_size=(dec_input.size(0), dec_input.size(1), cross_input.size(1)),
                                   pad_row=dec_input_num_pads,
                                   pad_column=enc_input_num_pads,
                                   rank=self.rank)

        y = self.out_embedder(dec_input)
        pos_y = torch.arange(dec_input.size(1)).unsqueeze(0).expand(dec_input.size(0), dec_input.size(1)).to(self.rank)
        y = y + self.pos_encoder(pos_y)
        for i in range(self.N_dec):
            y = self.decoders[i](x=y,
                                 cross_connection_x=cross_input,
                                 input_attention_mask=no_peak_and_pad_mask,
                                 cross_attention_mask=pad_mask)

        y = self.vocab_linear(y)

        if apply_log_softmax:
            y = self.log_softmax(y)

        return y
