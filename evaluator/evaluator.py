from typing import Tuple
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.nn import Module
import torch


class Evaluator(object):
    def __call__(self, dataloader: DataLoader, model: Module, device: torch.device) -> Tuple:
        all, match, op_match = 0, 0, 0
        for index, batch in tqdm(enumerate(dataloader)):
            for key, tensor in batch.items():
                if type(tensor) == torch.Tensor:
                    batch[key] = tensor.to(device)
            loss, logits, masked_loss, masked_logits = model(batch)
            for i in range(logits.size(0)):
                prd = torch.argmax(logits[i]).item()
                trg = batch['labels'][i].item()
                all += 1
                if prd == trg:
                    match += 1

            # Evaluation on operation: Accuracy
            if masked_loss is not None:
                for i in range(masked_logits.size(0)):
                    prd = torch.argmax(masked_logits[i]).item()
                    trg = batch['op'][i].item()
                    if prd == trg:
                        op_match += 1

        return match / all, op_match / all
