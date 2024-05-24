import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TextClassificationDataset(Dataset):

    def __init__(self, data, tokenizer_path):
        self.sequences = data
        self.item2id = json.load(open("../dataset/Amazon/metadata.json", "r"))["entity2id"]
        if "gpt" in tokenizer_path:
            self.type = "gpt"
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.tokenizer.pad_token_id = 0
            self.tokenizer.padding_side = "left"
        else:
            self.type = "bert"
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        dict_elem = self.sequences[index]
        item = self.item2id[dict_elem["id"]]
        text = dict_elem["question"]
        label = dict_elem["answer"]
        if self.type == "bert":
            # 在这里我们对文本进行Tokenize处理
            inputs = self.tokenizer(text, truncation=True, padding='max_length', max_length=384)
        else:
            inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=128)
        return {
            'item': item,
            'input_ids': torch.tensor(inputs['input_ids'], dtype = torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype = torch.long),
            'label': torch.tensor(label, dtype = torch.long)
        }



class ReviewPredictionDataset(Dataset):

    def __init__(self, data, tokenizer_path, kg_name):
        self.sequences = data
        self.item2id = json.load(open("../dataset/{}/metadata.json".format(kg_name), "r"))["entity2id"]
        if "gpt" in tokenizer_path:
            self.type = "gpt"
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.tokenizer.pad_token_id = 0
            self.tokenizer.padding_side = "left"
        else:
            self.type = "bert"
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):

        dict_elem = self.sequences[index]
        item = self.item2id[dict_elem["id"]]
        text = dict_elem["review"]
        label = dict_elem["answer"] - 1
        # 在这里我们对文本进行Tokenize处理
        if self.type == "bert":
            # 在这里我们对文本进行Tokenize处理
            inputs = self.tokenizer(text, truncation=True, padding='max_length', max_length=150)
        else:
            inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=150)

        return {
            'item': item,
            'input_ids': torch.tensor(inputs['input_ids'], dtype = torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype = torch.long),
            'label': torch.tensor(label, dtype = torch.long)
        }