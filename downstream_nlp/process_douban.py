import json

from random import shuffle
from collections import defaultdict

if __name__ == "__main__":
    data_file = open("./datasets/music_reviews.txt", "r")
    item_ids = json.load(open("../dataset/Douban/metadata.json", "r"))["entity2id"]
    item_count = defaultdict(int)
    score_count = defaultdict(list)
    for line in data_file.readlines()[1:]:
        splits = line.split('\t')
        if len(splits) < 8:
            continue
        item_id = "music" + splits[1]
        if item_id not in item_ids:
            continue
        rates = int(splits[2].replace('\"', ''))
        item_info = splits[3].replace('\"', '')
        comment = splits[4].replace('\"', '')
        select_data = {
            "id": item_id,
            "review": "Item: {} | Review: {}".format(item_info, comment),
            "answer": rates
        }
        item_count[select_data["id"]] += 1
        score_count[select_data["answer"]].append(select_data)
    print(len(score_count[1]), len(score_count[2]), len(score_count[3]), len(score_count[4]),len(score_count[5]))
    count = 2400
    num_test = int(count * 0.1)
    test_set = []
    valid_set = []
    train_set = []
    for k in score_count:
        shuffle(score_count[k])
        test_set += score_count[k][: num_test]
        valid_set += score_count[k][num_test: 2 * num_test]
        train_set += score_count[k][2 * num_test: count]
    print(test_set[0], valid_set[0])
    print(len(train_set), len(valid_set), len(test_set))
    dataset = {
        'train': train_set,
        'valid': valid_set,
        'test': test_set
    }
    json.dump(dataset, open("datasets/douban_review_item.json", "w"), ensure_ascii=False)
