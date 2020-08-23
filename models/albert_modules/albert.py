from torch import nn as nn
from models.albert_modules.embedding import ALBERTEmbedding
from models.albert_modules.transformer import TransformerLayer
from utils import fix_random_seed_as


class ALBERT(nn.Module):
    def __init__(self, args):
        super().__init__()
        fix_random_seed_as(args.model_init_seed)
        max_len = args.bert_max_len
        num_items = args.num_items
        n_layers = args.bert_num_blocks
        heads = args.bert_num_heads
        vocab_size = num_items + 2
        embed_size = args.bert_embed_size
        hidden = args.bert_hidden_units
        self.hidden = hidden
        self.embedding = ALBERTEmbedding(vocab_size=vocab_size, embed_size=embed_size, max_len=max_len,
                                         hidden=hidden)
        self.transformer_layer = TransformerLayer(n_layers, hidden, heads, hidden * 4)

    def forward(self, x):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        x = self.embedding(x)
        x = self.transformer_layer(x, mask)
        return x

    def init_weights(self):
        pass
