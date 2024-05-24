from models.modeling_pretrain import OurConfig
from models.modeling_ptuning import OurPromptTuningModel
from torch.utils.data import DataLoader
from kg_dataset import PretrainDataset
from pretrain import get_args


if __name__ == "__main__":
    base_path = "save/Amazon-LR0.0003-MU0.01-Temp0.1/"
    config = OurConfig.from_json_file("{}/config.json".format(base_path))
    checkpoint_path = "{}/epoch-5/".format(base_path)
    prompt_model = OurPromptTuningModel(config, checkpoint_path)
    args = get_args()
    dataset = PretrainDataset(args)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    for (idx, batch_data) in enumerate(data_loader):
        output = prompt_model(
            input_ids = batch_data["input_ids"],
            rel_ids = batch_data["rel_ids"],
            attention_mask = batch_data["attention_masks"],
            token_type_ids = batch_data["type_ids"]
        ).pooler_output
        print(output.shape)
