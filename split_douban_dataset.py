import json
import random
import pickle as pkl
from scipy.sparse import csr_matrix
from collections import defaultdict


if __name__ == "__main__":
    dataset = "book"
    # meta_data = json.load(open("data_utils/Amazon/meta_{}.json".format(dataset)))
    rate_data = open("data_utils/Douban/{}reviews_cleaned.txt".format(dataset), "r")
    user_dict = defaultdict(int)
    item_dict = defaultdict(int)
    user_item = defaultdict(list)
    item_user = defaultdict(list)
    for line in rate_data.readlines()[1:]:
        splits = line.split('\t')
        if len(splits) < 2:
            continue
        user, item = splits[0], splits[1]
        user = user.replace('\"', "")
        item = item.replace('\"', "")
        user_item[user].append(item)
        item_user[item].append(user)
    min_threshold = 0
    filtered_user_items = defaultdict(list)
    for i in user_item.keys():
        cand = []
        for item in user_item[i]:
            if len(item_user[item]) < 5:
                continue
            cand.append(item)
        if len(cand) >= 5:
            filtered_user_items[i] = cand
    interactions = defaultdict(list)
    kg_data = pkl.load(open("dataset/Douban/KG.pkl", "rb"))
    item2id = json.load(open("dataset/Douban/metadata.json", "r"))["entity2id"]
    for k in filtered_user_items:
        if k not in user_dict:
            user_dict[k] = len(user_dict)
        for item in filtered_user_items[k]:
            item_id = "{}\"{}\"".format(dataset, item)
            if item_id not in item2id:
                continue
            if item not in item_dict:
                item_dict[item] = len(item_dict)
            interactions[user_dict[k]].append(item_dict[item])
    m = len(user_dict)
    n = len(item_dict)
    s = sum([len(interactions[k]) for k in interactions])
    id2index = kg_data["indexs"]
    item2index = {}
    for item in item_dict:
        item_id = item_dict[item]
        index = id2index[item2id["{}\"{}\"".format(dataset, item)]]
        print(item, item_id, index)
        item2index[item_id] = index
    pkl.dump(item2index, open("sslrec/datasets/general_cf/douban_{}/kg_map.pkl".format(dataset), "wb"))
    train = [[0 for _ in range(n)] for _ in range(m)]
    valid = [[0 for _ in range(n)] for _ in range(m)]
    test = [[0 for _ in range(n)] for _ in range(m)]
    random.seed(42)
    for i in range(m):
        data = interactions[i]
        random.shuffle(data)
        # print(len(data))
        if len(data) < 3:
            continue
        num_valid = max(1, int(len(data) * 0.1))
        num_test = max(1, int(len(data) * 0.1))
        valid_data = data[0: num_valid]
        test_data = data[num_valid: num_valid + num_test]
        train_data = data[num_valid + num_test:]
        for j in train_data:
            train[i][j] = 1
        for j in valid_data:
            valid[i][j] = 1
        for j in test_data:
            test[i][j] = 1
    sparse_train = csr_matrix(train)
    sparse_valid = csr_matrix(valid)
    sparse_test = csr_matrix(test)
    # print(sparse_train.shape, sparse_valid.shape, sparse_test.shape)
    pkl.dump(sparse_train, open("sslrec/datasets/general_cf/douban_{}/train_mat.pkl".format(dataset), "wb"))
    pkl.dump(sparse_valid, open("sslrec/datasets/general_cf/douban_{}/valid_mat.pkl".format(dataset), "wb"))
    pkl.dump(sparse_test, open("sslrec/datasets/general_cf/douban_{}/test_mat.pkl".format(dataset), "wb"))

