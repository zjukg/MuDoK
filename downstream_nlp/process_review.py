import json

from random import shuffle
from collections import defaultdict

if __name__ == "__main__":
    data_file = open("./datasets/review_small.json", "r")
    text_map = json.load(open("../data_utils/item2text.json", "r"))
    item_count = defaultdict(int)
    score_count = defaultdict(list)
    for line in data_file.readlines():
        data = eval(line)
        item_text = text_map[data["asin"]]
        prompt = "Review: {}, Product: {} {}\n\n ".format(data["reviewText"], data["asin"], item_text)
        select_data = {
            "id": data["asin"],
            "review": prompt,
            "answer": int(data["overall"])
        }
        item_count[select_data["id"]] += 1
        score_count[select_data["answer"]].append(select_data)
    count = 10000
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
    json.dump(dataset, open("datasets/movie_review_item.json", "w"), ensure_ascii=False)
