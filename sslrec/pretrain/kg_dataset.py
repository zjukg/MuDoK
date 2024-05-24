import json
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict


class PretrainDataset(Dataset):
    def __init__(self, args) -> None:
        super().__init__()
        self.dataset = args["dataset"]
        meta_data = json.load(open("dataset/{}/metadata.json".format(self.dataset), "r"))
        self.entity2id = meta_data["entity2id"]
        self.id2entity = meta_data["id2entity"]
        self.relation2id = meta_data["relation2id"]
        self.id2relation = meta_data["id2relation"]
        self.triples = meta_data["triples"]
        # Padding
        self.max_token_length = args["max_token_length"]
        self.pad_token = len(self.entity2id)

        self.num = len(self.triples)
        self.item_list = defaultdict(list)
        self.tokens = []
        self.rels = []
        self.attention_mask = []
        
        self._preprocess()
        self.token_num = len(self.entity2id) + 1
        self.rel_num = len(self.relation2id) + 1
        

    
    def _preprocess(self):
        for i in range(self.num):
            h, r, t = self.triples[i]
            hid = self.entity2id[h]
            rid = self.relation2id[r]
            tid = self.entity2id[t]
            self.item_list[hid].append((rid, tid))
        for item in self.item_list:
            sequence = [item]
            relation = [0]
            for (a, v) in self.item_list[item]:
                sequence.append(v)
                relation.append(a + 1)
            token_ids, rel_ids, attention_mask = self.padding(sequence, relation)
            self.tokens.append(token_ids)
            self.rels.append(rel_ids)
            self.attention_mask.append(attention_mask)
        type_ids = [1 for _ in range(self.max_token_length)]
        type_ids[0] = 0
        self.type_ids = type_ids


    
    def padding(self, sequence, relation):
        if len(sequence) > self.max_token_length:
            sequence = sequence[: self.max_token_length]
            relation = relation[: self.max_token_length]
            attention_mask = [1] * self.max_token_length
        else:
            m = len(sequence)
            n = self.max_token_length
            sequence = sequence + [self.pad_token] * (n - m)
            relation = relation + [0] * (n - m)
            attention_mask = [1] * m + [0] * (n - m)
        return sequence, relation, attention_mask
        
    

    def __len__(self):
        return len(self.tokens)


    def __getitem__(self, index):
        return {
            "input_ids": torch.tensor(self.tokens[index]),
            "rel_ids": torch.tensor(self.rels[index]),
            "type_ids": torch.tensor(self.type_ids),
            "attention_masks": torch.tensor(self.attention_mask[index])
        }


if __name__ == "__main__":
    args = {
        "dataset": "Amazon",
        "max_token_length": 9
    }
    dataset = PretrainDataset(args)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    for (idx, batch_data) in enumerate(data_loader):
        print(batch_data)
        break
    