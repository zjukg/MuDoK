import json
import scipy
import scipy.sparse
import pickle as pkl

if __name__ == "__main__":
    """
    dataset = "douban_movie"
    train_file = "sslrec/datasets/general_cf/{}/train_mat.pkl".format(dataset)
    valid_file = "sslrec/datasets/general_cf/{}/valid_mat.pkl".format(dataset)
    test_file = "sslrec/datasets/general_cf/{}/test_mat.pkl".format(dataset)
    count = 0
    for data in [train_file, valid_file, test_file]:
        matrix = pkl.load(open(data, "rb"))
        print(matrix.shape, matrix.nnz)
    """
    dataset = "downstream_nlp/datasets/video_game.json"
    data_file = json.load(open(dataset, "r"))
    print(data_file.keys())
    data = data_file['train'] + data_file['valid'] + data_file['test']
    s = set()
    for instance in data:
        s.add(instance['id'])
    print(len(data))
    print(len(s))
        
    