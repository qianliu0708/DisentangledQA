from transformers import RobertaModel
from torch.nn import Module
from torch import nn
import torch


class GoldenSentenceBert(Module):
    def __init__(self):
        super(GoldenSentenceBert, self).__init__()
        self.bert = RobertaModel.from_pretrained('roberta-large')
        self.classifier = nn.Linear(1024, 2)

    def forward(self, inputs, masks):
        # logits = []
        # inputs = inputs.permute(1, 0, 2)  # [seq_num, batch, 768]
        # masks = masks.permute(1, 0, 2)
        # for i in range(inputs.size(0)):
        #     cls = self.bert(inputs[i], masks[i], return_dict=True).pooler_output  # [batch, hidden]
        #     logit = self.classifier(cls).unsqueeze(0)  # [1, batch, 2]
        #     logits.append(logit)

        return self.classifier(self.bert(inputs, masks, return_dict=True).pooler_output)  # torch.cat(logits, 0).permute(1, 0, 2)