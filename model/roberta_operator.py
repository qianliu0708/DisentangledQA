from transformers import RobertaModel
from torch.nn import Module, Linear
import torch
from typing import Tuple
from torch.nn import CrossEntropyLoss


loss_fn = CrossEntropyLoss()


class ReasoningWithOperator(Module):
    def __init__(self):
        super(ReasoningWithOperator, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-large')
        self.classifer = Linear(in_features=1024 * 2, out_features=2)
        self.gru = torch.nn.GRU(input_size=1024, hidden_size=1024, batch_first=True)

    def forward(
        self,
        input: torch.Tensor,
        mask: torch.Tensor,
        label: torch.Tensor,
        op_len: torch.Tensor,
        op_abstract: torch.Tensor
    ) -> Tuple:
        outputs = self.roberta(
            input_ids=input,
            attention_mask=mask,
            return_dict=True
        ).last_hidden_state
        repre = torch.cat((outputs[:, 0, :], outputs[:, -2, :]), dim=1)
        # repre = []
        # for i in range(outputs.size(0)):
        #     op_rpr, hidden = self.gru(outputs[i, -op_len[i].item():, :].unsqueeze(0))
        #     repre.append(hidden)
        # repre = torch.cat(repre, dim=0)
        # repre = torch.cat((outputs[:, 0, :], repre.squeeze(1)), dim=1)
        logits = self.classifer(repre)
        return loss_fn(logits, label), logits