import json

from random import shuffle
from collections import defaultdict

if __name__ == "__main__":
    data_file = open("datasets/qa_Video_Games.json", "r")
    text_map = json.load(open("../data_utils/item2text.json", "r"))
    pos_data = []
    neg_data = []
    item_count = defaultdict(int)
    for line in data_file.readlines():
        data = eval(line)
        if data["questionType"] != "yes/no":
            continue
        item_text = text_map[data["asin"]][: 128]
        prompt = "Product: {} {}\n\n Question: {}".format(data["asin"], item_text, data["question"])
        select_data = {
            "id": data["asin"],
            "question": prompt,
            "answer": 1 if data["answerType"] == 'Y' else 0
        }
        if select_data["answer"] == 1:
            pos_data.append(select_data)
        else:
            neg_data.append(select_data)
        item_count[select_data["id"]] += 1

    count = min(len(pos_data), len(neg_data))
    pos_data = pos_data[: count]
    neg_data = neg_data[: count]
    shuffle(pos_data)
    shuffle(neg_data)
    num_test = int(count * 0.1)
    test_set = pos_data[: num_test] + neg_data[: num_test]
    valid_set = pos_data[num_test: 2 * num_test] + neg_data[num_test: 2 * num_test]
    train_set = pos_data[2 * num_test:] + neg_data[2 * num_test:]
    print(len(train_set), len(valid_set), len(test_set))
    dataset = {
        'train': train_set,
        'valid': valid_set,
        'test': test_set
    }
    json.dump(dataset, open("datasets/video_game2.json", "w"), ensure_ascii=False)
