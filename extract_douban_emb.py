import torch
import json
from transformers import BertModel, BertTokenizer

model_path = "downstream_nlp/bert-base-uncased/"

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path).to('cuda:5')


if __name__ == "__main__":
    dataset = "Douban"
    kg = json.load(open("dataset/{}/metadata.json".format(dataset), "r"))
    text_map = json.load(open("data_utils/item2text.json", "r"))
    entity_map = kg["id2entity"]
    n = len(entity_map)
    result = []
    count = 0
    with torch.no_grad():
        for i in range(n):
            entity = entity_map[str(i)]
            text = entity
            count += 1
            print(count)
            inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to('cuda:5')
            outputs = model(**inputs)
            ent_representation = outputs.pooler_output.reshape(-1,)
            result.append(ent_representation.cpu())
    result = torch.stack(result).cpu()
    torch.save(result, open("embeddings/{}2.pth".format(dataset), "wb"))
