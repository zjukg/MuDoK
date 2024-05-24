import os
import argparse
import torch
import logging
import pickle as pkl
from logging import getLogger, Formatter, StreamHandler
from torch.utils.data import DataLoader
from torch.optim import Adam
from kg_dataset import PretrainDataset
from models.modeling_pretrain import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="Douban", type = str)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--dim', default=200, type=int)
    parser.add_argument('--num_epoch', default=5, type=int)
    # parser.add_argument('--num_head', default=2, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--decay', default=0.0, type=float)
    parser.add_argument('--mu', default=0.1, type=float)
    parser.add_argument('--max_token_length', default=8, type=int)
    parser.add_argument('--temperature', default=0.5, type=float)
    args = parser.parse_args()
    return vars(args)


def list_to_dict(list):
    return_dict = {}
    for i in range(len(list)):
        return_dict[list[i][0]] = i
    return return_dict



if __name__ == "__main__":
    args = get_args()
    dataset = PretrainDataset(args)
    data_loader = DataLoader(dataset, batch_size=args["batch_size"], shuffle=True)
    pid2idx = list_to_dict(dataset.tokens)
    processed_dataset = {
        "indexs": pid2idx,
        "input_ids": dataset.tokens,
        "rel_ids": dataset.rels,
        "type_ids": dataset.type_ids,
        "attention_masks": dataset.attention_mask
    }
    pkl.dump(processed_dataset, open("save/Douban-KG.pkl", "wb"))
    