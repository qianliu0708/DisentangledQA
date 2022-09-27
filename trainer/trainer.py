from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
from transformers import AdamW, get_linear_schedule_with_warmup

from model.roberta import Reasoning
from model.roberta_operator import ReasoningWithOperator
from model.roberta_operator_abstract import ReasoningWithOperatorAbstract
from model.reasoning_plain import ReasoningPlain
from model.reasoning_attention import ReasoningSelfAttention
from dataset.golden_dataset import GoldenDataset, Collator
from dataset.golden_sentence_dataset import GoldenSentenceDataset
from dataset.last_step_dataset import LastStepDataset
from dataset.ir_avgcls_dataset import IrAvgClsDataset
from dataset.reasoning_dataset import ReasoningDataset, ReasoningCollator
from evaluator.evaluator import Evaluator
from predictor.ir_avgcls_predictor import IrAvrClsPredictor


dataset_dict = {
    "golden_dataset": GoldenDataset,
    "golden_sentence_dataset": GoldenSentenceDataset,
    "last_step_dataset": LastStepDataset,
    "ir_avgcls_dataset": IrAvgClsDataset,
    "reasoning_dataset": ReasoningDataset
}
collator_dict = {
    "collator": Collator,
    "reasoning_collator": ReasoningCollator
}
model_dict = {
    "Reasoning": Reasoning,
    "ReasoningWithOperator": ReasoningWithOperator,
    "ReasoningWithOperatorAbstract": ReasoningWithOperatorAbstract,
    "ReasoningPlain": ReasoningPlain,
    "ReasoningSelfAttention": ReasoningSelfAttention
}


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda') if self.args.cuda else torch.device('cpu')

        self.dataset = dataset_dict[self.args.train_dataset](self.args)
        print(f'Dataset: {self.args.train_dataset}')
        self.dev_dataset = dataset_dict[self.args.dev_dataset](self.args, 'dev')
        self.test_dataset = dataset_dict[self.args.test_dataset](self.args, 'test')
        self.dataloader = None
        self.collator = ReasoningCollator()

        self.model = model_dict[self.args.model_class]()
        print(f'Model: {self.args.model_class}')
        self.model.to(self.device)
        if self.args.eval_only:
            self.load_pretrained_all()
        elif self.args.load_pretrained:
            self.load_pretrained_roberta()

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

        # self.loss_fn = CrossEntropyLoss()
        self.evaluator = Evaluator()
        self.predictor = IrAvrClsPredictor(self.args.prediction_path)
        self.max_acc = 0.

    # def load_pretrained(self):
    #     pretrained_model: torch.nn.Module = torch.load(self.args.pretrained_model_path, map_location=self.device)
    #     pretrained_params = [key for key, value in pretrained_model.named_parameters()]
    #     state_dict = self.model.state_dict()
    #     unloaded_params = []
    #     for key, value in self.model.named_parameters():
    #         if key in pretrained_params:
    #             state_dict[key] = pretrained_model.state_dict()[key]
    #         else:
    #             unloaded_params.append(key)
    #     self.model.load_state_dict(state_dict)
    #     print(f'The following parameters are not loaded from pretrained model: {unloaded_params}')

    def load_pretrained_state_dict(self):
        pretrained_state_dict = torch.load(self.args.pretrained_model_path, map_location=self.device)
        unloaded_params = []
        state_dict = self.model.state_dict()
        for name, param in state_dict.items():
            if self.convert_key_seqcls(name) in pretrained_state_dict:
                state_dict[name] = pretrained_state_dict[self.convert_key_seqcls(name)]
            else:
                unloaded_params.append(name)
        self.model.load_state_dict(state_dict)
        print(f'The following params are not loaded from pretrained model: {unloaded_params}')

    def load_pretrained_roberta(self):
        pretrained_state_dict = torch.load(self.args.pretrained_model_path, map_location=self.device)
        unloaded_params = []
        state_dict = self.model.state_dict()
        for name, param in state_dict.items():
            if self.convert_key_roberta(name) in pretrained_state_dict:
                state_dict[name] = pretrained_state_dict[self.convert_key_roberta(name)]
            else:
                unloaded_params.append(name)
        self.model.load_state_dict(state_dict)
        print(f'The following params are not loaded from a pretrained model: {unloaded_params}')

    def load_pretrained_all(self):
        print('Loading from a previously trained model for evaluation and test ...')
        self.model = torch.load(self.args.trained_model_path)

    def convert_key_seqcls(self, original: str) -> str:
        return '_classifier' + original[7:]

    def convert_key_roberta(self, original: str) -> str:
        if 'encoder' in original.split('.') and 'self_attention' not in original:
            return '_classifier.roberta.' + '.'.join(original.split('.')[1:])
        elif 'dense' in original and 'self_attention' not in original:
            return '_classifier.classifier.dense.' + original.split('.')[-1]
        elif 'out_proj' in original and 'self_attention' not in original:
            return '_classifier.classifier.out_proj.' + original.split('.')[-1]
        else:
            return original

    def save(self):
        print(f'Model saved at {self.args.model_path}')
        torch.save(self.model, self.args.model_path)

    def train(self):
        for epoch in range(self.args.epoch_num):

            # Evaluation
            self.model.eval()
            dev_dataloader = DataLoader(
                dataset=self.dev_dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                collate_fn=self.collator,
                pin_memory=True if self.args.cuda else False,
                shuffle=False
            )
            print('Evaluating on Dev ...')
            with torch.no_grad():
                acc, op_acc = self.evaluator(dev_dataloader, self.model, self.device)
            if op_acc:
                print(f'Dev performance: Accuracy {acc}\tOperator Accuracy: {op_acc}')
            else:
                print(f'Dev performance: Accuracy {acc}')
            if acc > self.max_acc:
                print(f'Update!')
                self.max_acc = acc
                self.save()

                # Prediction
                print(f'Generating predictions on Test ...')
                test_dataloader = DataLoader(
                    dataset=self.test_dataset,
                    batch_size=self.args.batch_size,
                    num_workers=self.args.num_workers,
                    collate_fn=self.collator,
                    pin_memory=True if self.args.cuda else False,
                    shuffle=False
                )
                with torch.no_grad():
                    self.predictor(test_dataloader, self.model, self.device)

            if self.args.eval_only:
                break

            # Training
            self.model.train()
            self.dataloader = DataLoader(
                dataset=self.dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                collate_fn=self.collator,
                pin_memory=True if self.args.cuda else False,
                shuffle=True
            )
            for index, batch in tqdm(enumerate(self.dataloader)):
                for key, tensor in batch.items():
                    if type(tensor) == torch.Tensor:
                        batch[key] = tensor.to(self.device)
                loss, logits, masked_loss, masked_logits = self.model(
                    batch
                )
                if masked_loss is not None:
                    loss = loss + self.args.weightofoperator * masked_loss
                loss.backward()
                if index % self.args.gradient_accumulate_step == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if index % 50 == 0:
                    print(f'Epoch: {epoch}/{self.args.epoch_num}\tBatch: {index}/{len(self.dataloader)}\t'
                          f'Loss: {loss.item()}')
