import torch.nn as nn
import math


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, d_model, dropout_perc):
        super(EmbeddingLayer, self).__init__()
        self.dropout = nn.Dropout(dropout_perc)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.dropout(self.embed(x)) * math.sqrt(float(self.d_model))

