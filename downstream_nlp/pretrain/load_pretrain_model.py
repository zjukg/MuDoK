from transformers import BertModel

def load_pretrain_model(model_path):
    model = BertModel.from_pretrained(model_path)
    print(model.state_dict().keys())

if __name__ == "__main__":
    model_path  = "save/model-v1"
    load_pretrain_model(model_path)