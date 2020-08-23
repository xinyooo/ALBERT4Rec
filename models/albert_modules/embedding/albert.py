import torch
import torch.nn as nn
from .token import TokenEmbedding
from .position import PositionalEmbedding
from models.albert_modules.utils import LayerNorm


class ALBERTEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, max_len, hidden):
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.token_2 = nn.Linear(embed_size, hidden)
        self.position = PositionalEmbedding(max_len=max_len, d_model=hidden)
        self.norm = LayerNorm(hidden)

    def forward(self, sequence):
        seq_len = sequence.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=sequence.device)
        pos = pos.unsqueeze(0).expand_as(sequence)

        x = self.token(sequence)
        x = self.token_2(x)
        x = x + self.position(pos)
        return self.norm(x)
