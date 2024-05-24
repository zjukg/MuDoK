import os
import argparse
import torch
import logging
from logging import getLogger, Formatter, StreamHandler
from torch.utils.data import DataLoader
from torch.optim import Adam
from kg_dataset import PretrainDataset
from models.modeling_pretrain import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="Amazon", type = str)
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



if __name__ == "__main__":
    args = get_args()
    dataset = PretrainDataset(args)
    data_loader = DataLoader(dataset, batch_size=args["batch_size"], shuffle=True)

    config = OurConfig(
        vocab_size=dataset.token_num,
        type_vocab_size=3,
        rel_size=dataset.rel_num,
        temperature=args["temperature"]
    )
    model = PreTrainBackbone(config).cuda()

    optimizer = Adam(model.parameters(), lr=args["lr"], weight_decay=args["decay"])
    total_params = sum(p.numel() for group in optimizer.param_groups for p in group['params'])
    trainable_params = sum(p.numel() for group in optimizer.param_groups for p in group['params'] if p.requires_grad)

    # Logger Setting
    log_format = "{}-{}-{}".format(args["dataset"], args["lr"], args["mu"])
    logger = getLogger()
    logger.setLevel(logging.INFO)
    log_format = Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler = StreamHandler()
    stream_handler.setFormatter(log_format)
    logger.addHandler(stream_handler)
    logger.info(args)
    logger.info("Total Params: {}, Trainable Params: {}, Ratio: {}".format(total_params, trainable_params, trainable_params / total_params))
    model.train()
    for epoch in range(args["num_epoch"]):
        for (idx, batch_data) in enumerate(data_loader):
            losses = model.get_loss(
                input_ids = batch_data["input_ids"].cuda(),
                rel_ids = batch_data["rel_ids"].cuda(),
                attention_mask = batch_data["attention_masks"].cuda(),
                token_type_ids = batch_data["type_ids"].cuda()
            )
            loss = losses[0] + args["mu"] * losses[1]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.info("Epoch: {}, Step: {}, Loss: {}".format(epoch + 1, idx, round(loss.item(), 5)))

        # Save Model
        path_base = "save/{}-LR{}-MU{}-Temp{}".format(args["dataset"], args["lr"], args["mu"], config.temperature)
        if not os.path.exists(path_base):
            os.mkdir(path_base)
        path_base = "{}/epoch-{}".format(path_base, epoch + 1)
        if not os.path.exists(path_base):
            os.mkdir(path_base)
        torch.save(model.transformer.embeddings.state_dict(), open("{}/embedding.pth".format(path_base), "wb"))
        torch.save(model.transformer.pooler.state_dict(), open("{}/pooler.pth".format(path_base), "wb"))
        torch.save(model.transformer.encoder.state_dict(), open("{}/encoder.pth".format(path_base), "wb"))
    model.transformer.config.save_pretrained("save/{}-LR{}-MU{}-Temp{}".format(args["dataset"], args["lr"], args["mu"], config.temperature))
    