from transformers import RobertaForSequenceClassification
from torch.nn import Module, Linear
import torch
from typing import Tuple


class Reasoning(Module):
    def __init__(self):
        super(Reasoning, self).__init__()
        self.roberta = RobertaForSequenceClassification.from_pretrained('roberta-large')
        # self.classifer = Linear(in_features=768, out_features=2)

    def forward(self, batch) -> Tuple:
        input = batch['input_ids'].long()
        mask = batch['masks']
        label = batch['labels'].long()

        outputs = self.roberta(
            input_ids=input,
            attention_mask=mask,
            labels=label
        )
        return outputs.loss, outputs.logits, None, None
