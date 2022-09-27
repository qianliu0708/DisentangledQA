from torch.utils.data import DataLoader
import torch
from torch.nn import CrossEntropyLoss
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

from model.golden_sentence_bert import GoldenSentenceBert
from dataset.squad import SquadDataset, SquadDatasetCollator
from evaluator.evaluator import Evaluator


class GoldenSentenceTrainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda') if self.args.cuda else torch.device('cpu')

        self.dataset = SquadDataset(args, 'train')
        self.dev_dataset = SquadDataset(args, 'dev')
        # self.test_dataset = Golden_Dataset('test')
        self.dataloader = None
        self.model = GoldenSentenceBert()
        self.model.to(self.device)

        # listed_params = list(self.model.named_parameters())
        # grouped_parameters = [
        #     {'params': [p for n, p in listed_params if 'bert' in n],
        #      'lr': self.args.tuning_rate,
        #      'weight_decay': self.args.weight_decay},
        #     {'params': [p for n, p in listed_params if 'bert' not in n],
        #      'weight_decay': self.args.weight_decay}
        # ]
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            correct_bias=False
        )
        total_steps = (self.args.epoch_num * len(self.dataset)) // self.args.batch_size
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(self.args.warmup_rate * total_steps),
            num_training_steps=total_steps
        )

        self.loss_fn = CrossEntropyLoss(ignore_index=-1)
        self.evaluator = Evaluator()
        self.max_f1 = 0.

    def calculate_loss(self, logit, golden) -> torch.Tensor:
        logit = logit.reshape(-1, 2)
        golden = golden.reshape(-1)
        return self.loss_fn(logit, golden)

    def load_pretrained(self):
        self.model = torch.load(self.args.pretrained_model_path, map_location=self.device)
        print(f'Pretrained model loaded from {self.args.pretrained_model_path}')

    def save(self):
        print(f'Model saved at {self.args.model_path}')
        torch.save(self.model, self.args.model_path)

    def evaluate(self, dev_dataloader):
        all = 0.
        matched = 0.
        para_triple = {}
        with torch.no_grad():
            for index, batch in enumerate(tqdm(dev_dataloader)):

                for key, tensor in batch.items():
                    batch[key] = tensor.to(self.device)
                logits = self.model(
                    inputs=batch['inputs'],
                    masks=batch['masks']
                )
                predicted = torch.argmax(logits, dim=1)

                for i in range(logits.size(0)):
                    para_index = batch["paras"][i].item()
                    if para_index not in para_triple:
                        para_triple[para_index] = (logits[i][1].item(), predicted[i].item(), batch["labels"][i].item())
                    else:
                        if logits[i][1].item() > para_triple[para_index][0]:
                            para_triple[para_index] = (logits[i][1].item(), predicted[i].item(), batch["labels"][i].item())
                # for i in range(batch['labels'].size(0)):
                #     if batch['labels'][i].item() == 1:
                #         recall_denom += 1
                #     if predicted[i].item() == 1:
                #         precision_denom += 1
                #     if batch['labels'][i].item() == 1 and predicted[i].item() == 1:
                #         matched += 1
                for key, triple in para_triple.items():
                    all += 1
                    if triple[1] == 1 and triple[2] == 1:
                        matched += 1
        # precision = matched / (precision_denom + 1e-5)
        # recall = matched / (recall_denom + 1e-5)
        # f1 = 2 * precision * recall / (precision + recall + 1e-5)
        acc = matched / all
        if acc > self.max_f1:
            self.max_f1 = acc
            print('Update!')
            self.save()
        # print(f'P: {precision}\tR: {recall}\tF1: {f1}')
        print(f'Acc: {acc}')

    def train(self):
        for epoch in range(self.args.epoch_num):

            self.model.eval()
            dev_dataloader = DataLoader(
                dataset=self.dev_dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                collate_fn=SquadDatasetCollator(),
                pin_memory=True if self.args.cuda else False,
                shuffle=False
            )
            print('Evaluating on Dev ...')
            print(len(self.dev_dataset))
            self.evaluate(dev_dataloader)

            self.model.train()
            self.dataloader = DataLoader(
                dataset=self.dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                collate_fn=SquadDatasetCollator(),
                shuffle=True
            )
            for index, batch in enumerate(tqdm(self.dataloader)):

                for key, tensor in batch.items():
                    batch[key] = tensor.to(self.device)
                logits = self.model(
                    inputs=batch['inputs'],
                    masks=batch['masks']
                )
                loss = self.loss_fn(logits, batch["labels"])
                # loss = self.calculate_loss(logits, batch["labels"])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                if index % 500 == 0:

                    print(f'Epoch: {epoch}/{self.args.epoch_num}\tBatch: {index}/{len(self.dataloader)}\t'
                          f'Loss: {loss.item()}')

                # del batch, logits, loss
                torch.cuda.empty_cache()
