import torch.nn as nn
from .attention import MultiHeadedAttention
from .utils import LayerNorm, PositionwiseFeedForward


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


class TransformerLayer(nn.Module):
    def __init__(self, n_layers, hidden, attn_heads, feed_forward_hidden):
        super().__init__()
        self.n_layers = n_layers
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.projection = nn.Linear(hidden, hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden)
        self.input_sublayer = LayerNorm(hidden)
        self.output_sublayer = LayerNorm(hidden)

    def forward(self, x, mask):
        for _ in range(self.n_layers):
            x = self.attention(x, mask=mask)
            x = self.input_sublayer(x + self.projection(x))
            x = self.output_sublayer(x + self.feed_forward(x))
        return x
