import torch
import json
from tqdm import tqdm
from transformers import BertTokenizer
from nltk import sent_tokenize


class SquadGoldenSentencePredictor(object):
    def __init__(self, args, split):
        self.arg = args
        self.device = torch.device('cuda') if self.arg.cuda else torch.device('cpu')
        self.model = torch.load(self.arg.model_path, map_location=self.device)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.corpus = json.load(open(self.arg.corpus_path, 'r'))
        if split == 'train':
            self.dataset = json.load(open(self.arg.train_path, 'r', encoding='utf-8'))
            self.gdsent_dataset = self.arg.train_gdsent_path
        else:
            self.dataset = json.load(open(self.arg.dev_path, 'r', encoding='utf-8'))
            self.gdsent_dataset = self.arg.dev_gdsent_path

    def predict(self):
        write_obj = []
        for i, _ in enumerate(tqdm(self.dataset)):
            new_dict = self.dataset[i]
            new_dict["golden_sentence"] = []
            path = self.dataset[i]["evidence"][0]
            for step_index, step in enumerate(path):
                gdsents = []
                for evidence in step:
                    if type(evidence) == list:
                        for paragraph in evidence:
                            question = self.dataset[i]["decomposition"][step_index]
                            context = self.corpus[paragraph]["content"]
                            gdsents += self.find(question, context)
                if gdsents:
                    new_dict["golden_sentence"].append(gdsents)
            write_obj.append(new_dict)

        with open(self.gdsent_dataset, 'w') as f:
            json.dump(write_obj, f)

    def find(self, question, context) -> list:
        gdsent = ""
        score = -1
        sents = sent_tokenize(context)
        for sent in sents:
            input_id = self.tokenizer.convert_tokens_to_ids(
                ['[CLS]'] + self.tokenizer.tokenize(question) + ['[SEP]'] + self.tokenizer.tokenize(sent)
            )
            mask = [1] * len(input_id)
            input_id = torch.tensor(input_id, device=self.device)
            mask = torch.tensor(mask, device=self.device)
            logit = self.model(
                input_id.unsqueeze(0),
                mask.unsqueeze(0)
            )
            if logit.squeeze(0)[1] > score:
                score = logit.squeeze(0)[1]
                gdsent = sent

        return [gdsent]