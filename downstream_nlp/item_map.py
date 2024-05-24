import json
import pickle as pkl


if __name__ == "__main__":
    kg_data = pkl.load(open("../dataset/Douban/KG.pkl", "rb"))
    index_map = kg_data["indexs"]
    input_ids = [None] * 100000
    rel_ids = [None] * 100000
    type_ids = [None] * 100000
    attention_masks = [None] * 100000
    id_map = {}
    item2id = json.load(open("../dataset/Douban/metadata.json", "r"))["entity2id"]
    dataset = json.load(open("datasets/douban_review_item.json", "r"))
    for split in dataset.keys():
        data = dataset[split]
        n = len(data)
        for i in range(n):
            item_id = item2id[data[i]['id']]
            if item_id in id_map:
                continue
            id_map[item_id] = len(id_map)
            index = index_map[item_id]
            id = id_map[item_id]
            print(id)
            input_ids[id] = kg_data['input_ids'][index]
            rel_ids[id] = kg_data['rel_ids'][index]
            type_ids[id] = kg_data['type_ids']
            attention_masks[id] = kg_data['attention_masks'][index]
    data = {
        'input_ids': input_ids[0: len(id_map)],
        'rel_ids': rel_ids[0: len(id_map)],
        'type_ids': type_ids[0: len(id_map)],
        'attention_masks': attention_masks[0: len(id_map)],
        'id_map': id_map
    }
    pkl.dump(data, open("datasets/douban_review_prompt_map.pkl", "wb"))