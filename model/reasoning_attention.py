from typing import Tuple, Dict

from transformers import RobertaModel
from transformers import RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaLayer
from torch.nn import Module, Linear, CrossEntropyLoss


class ReasoningSelfAttention(Module):
    def __init__(self):
        super(ReasoningSelfAttention, self).__init__()
        self.encoder = RobertaModel.from_pretrained('roberta-large')
        self.config = RobertaConfig.from_pretrained('roberta-large')
        self.self_attention = RobertaLayer(self.config)
        self.label_dense = Linear(
            in_features=1024,
            out_features=1024
        )
        self.label_out_proj = Linear(
            in_features=1024,
            out_features=2
        )
        self.operator = Linear(
            in_features=1024,
            out_features=4
        )
        self.loss_fn = CrossEntropyLoss()

    def forward(self, batch: Dict) -> Tuple:
        input = batch['input_ids'].long()
        mask = batch['masks']
        masked_input = batch['masked_input_ids'].long()
        masked_mask = batch['masked_mask']
        label = batch['labels'].long()
        op_label = batch['op'].long()

        original_representation = self.encoder(
            input_ids=input,
            attention_mask=mask
        ).last_hidden_state
        original_representation = self.self_attention(original_representation)[0][:, 0, :]
        original_logits = self.label_out_proj(self.label_dense(original_representation))
        original_loss = self.loss_fn(
            input=original_logits,
            target=label
        )

        masked_representation = self.encoder(
            input_ids=masked_input,
            attention_mask=masked_mask
        ).pooler_output
        masked_logits = self.operator(masked_representation)
        masked_loss = self.loss_fn(
            input=masked_logits,
            target=op_label
        )

        return original_loss, original_logits, masked_loss, masked_logits
