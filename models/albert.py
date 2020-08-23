from .base import BaseModel
from .albert_modules.albert import ALBERT

import torch.nn as nn


class ALBERTModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.albert = ALBERT(args)
        self.out = nn.Linear(self.albert.hidden, args.num_items + 1)

    @classmethod
    def code(cls):
        return 'albert'

    def forward(self, x):
        x = self.albert(x)
        return self.out(x)
