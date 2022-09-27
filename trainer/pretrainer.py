from torch.utils.data import DataLoader
import torch
from torch.nn import CrossEntropyLoss
from transformers import AdamW, get_linear_schedule_with_warmup

from model.roberta import Reasoning
from dataset.golden_dataset import GoldenDataset, Collator
from evaluator.evaluator import Evaluator


class Pretrainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda') if self.args.cuda else torch.device('cpu')

        self.dataset = GoldenDataset(args)
        self.dataloader = None
        self.model = Reasoning()
        self.model.to(self.device)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.args.pretrain_learning_rate,
            weight_decay=self.args.weight_decay,
            correct_bias=False
        )
        total_steps = (self.args.epoch_num * len(self.dataset)) // self.args.batch_size
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(self.args.warmup_rate * total_steps),
            num_training_steps=total_steps
        )

        self.loss_fn = CrossEntropyLoss()
        self.evaluator = Evaluator()

    def save(self):
        torch.save(self.model, self.args.pretrained_model_path)

    def pretrain(self):
        print('Pretraining on BoolQ ...')
        for epoch in range(self.args.pretrain_epoch_num_boolq):
            self.model.train()
            self.dataloader = DataLoader(
                dataset=self.dataset,
                batch_size=self.args.pretrain_batch_size,
                num_workers=self.args.num_workers,
                collate_fn=Collator(),
                pin_memory=True if self.args.cuda else False,
                shuffle=True
            )
            for index, batch in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                for key, tensor in batch.items():
                    batch[key] = tensor.to(self.device)
                loss, logits = self.model(
                    input=batch['input_ids'].long(),
                    mask=batch['masks'],
                    label=batch['labels'].long()
                )
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                if index % 20 == 0:
                    print(f'Dataset: BoolQ\tEpoch: {epoch}/{self.args.epoch_num}\tBatch: {index}/{len(self.dataloader)}\t'
                          f'Loss: {loss.item()}')

        print('Pretraining on 20Q ...')
        for epoch in range(self.args.pretrain_epoch_num_20q):
            self.model.train()
            self.dataloader = DataLoader(
                dataset=self.dataset,
                batch_size=self.args.pretrain_batch_size,
                num_workers=self.args.num_workers,
                collate_fn=Collator(),
                pin_memory=True if self.args.cuda else False,
                shuffle=True
            )
            for index, batch in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                for key, tensor in batch.items():
                    batch[key] = tensor.to(self.device)
                loss, logits = self.model(
                    input=batch['input_ids'].long(),
                    mask=batch['masks'],
                    label=batch['labels'].long()
                )
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                if index % 20 == 0:
                    print(
                        f'Dataset: 20Q\tEpoch: {epoch}/{self.args.epoch_num}\tBatch: {index}/{len(self.dataloader)}\t'
                        f'Loss: {loss.item()}')

        self.save()