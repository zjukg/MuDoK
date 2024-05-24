import json
import torch
import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from dataset import ReviewPredictionDataset
from models import BertForCLS

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="douban_review_item", type = str)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--cuda', default=1, type=int)
    parser.add_argument('--num_epoch', default=5, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--eval_size', default=32, type=int)
    parser.add_argument('--model_path', default='roberta-base', type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    args = get_args()
    torch.cuda.set_device(args.cuda)

    dataset = json.load(open("datasets/{}.json".format(args.dataset), "r"))
    train_data = dataset["train"]
    valid_data = dataset["valid"]
    test_data = dataset["test"]
    if "movie" in args.dataset:
        kg_name = "Amazon"
    else:
        kg_name = "Douban"
    train_dataset = ReviewPredictionDataset(data=train_data, tokenizer_path=args.model_path, kg_name=kg_name)
    valid_dataset = ReviewPredictionDataset(data=valid_data, tokenizer_path=args.model_path, kg_name=kg_name)
    test_dataset = ReviewPredictionDataset(data=test_data, tokenizer_path=args.model_path, kg_name=kg_name)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.eval_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.eval_size, shuffle=True)

    model = BertForCLS(args.model_path, class_num=5).cuda()
    optimizer = Adam(model.parameters(), lr=args.lr)
    loss_func = CrossEntropyLoss()

    for epoch in range(args.num_epoch):
        training_bar = tqdm(train_dataloader)
        model = model.train()
        for (idx, batch_data) in enumerate(training_bar):
            input_ids = batch_data['input_ids'].cuda()
            attention_mask = batch_data['attention_mask'].cuda()
            labels = batch_data['label'].cuda()
            model_output = model.forward(input_ids, attention_mask)
            loss = loss_func(model_output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_bar.set_postfix_str("loss={}".format(loss.item()))
        model = model.eval()
        with torch.no_grad():
            valid_predictions = []
            valid_labels = []
            for (idx, batch_data) in enumerate(valid_dataloader):
                input_ids = batch_data['input_ids'].cuda()
                attention_mask = batch_data['attention_mask'].cuda()
                labels = batch_data['label']
                model_output = model.forward(input_ids, attention_mask)
                pred_probs = torch.softmax(model_output, dim=-1)
                pred_result = torch.argmax(pred_probs, dim=-1)
                valid_predictions += pred_result.cpu().numpy().tolist()
                valid_labels += labels
            acc = accuracy_score(y_true=valid_labels, y_pred=valid_predictions)
            macro_f1 = f1_score(y_true=valid_labels, y_pred=valid_predictions, average='macro')
            micro_f1 = f1_score(y_true=valid_labels, y_pred=valid_predictions, average='micro')
            print("Epoch{} Valiatation: Acc={:.4f}, Ma-F1={:.4f}, Mi-F1={:.4f}".format(epoch + 1, acc, macro_f1, micro_f1))
            test_predictions = []
            test_labels = []
            for (idx, batch_data) in enumerate(test_dataloader):
                input_ids = batch_data['input_ids'].cuda()
                attention_mask = batch_data['attention_mask'].cuda()
                labels = batch_data['label']
                model_output = model.forward(input_ids, attention_mask)
                pred_probs = torch.softmax(model_output, dim=-1)
                pred_result = torch.argmax(pred_probs, dim=-1)
                test_predictions += pred_result.cpu().numpy().tolist()
                test_labels += labels
            acc = accuracy_score(y_true=valid_labels, y_pred=valid_predictions)
            macro_f1 = f1_score(y_true=valid_labels, y_pred=valid_predictions, average='macro')
            micro_f1 = f1_score(y_true=valid_labels, y_pred=valid_predictions, average='micro')
            print("Epoch{} Test: Acc={:.4f}, Ma-F1={:.4f}, Mi-F1={:.4f}".format(epoch + 1, acc, macro_f1, micro_f1))

            
