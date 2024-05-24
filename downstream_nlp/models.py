import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoConfig
from pretrain.models.modeling_pretrain import OurConfig
from pretrain.models.modeling_ptuning import OurPromptTuningModel, PKGM


class BertForCLS(nn.Module):
	def __init__(self, model_path, class_num=2) -> None:
		super().__init__()
		self.config = AutoConfig.from_pretrained(model_path)
		self.bert = AutoModel.from_pretrained(model_path)
		self.output = nn.Linear(self.config.hidden_size, class_num)
	
	def forward(self, input_ids, attention_mask):
		bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
		preds = self.output(bert_output)
		return preds


class GPTForCLS(nn.Module):
	def __init__(self, model_path, class_num=2) -> None:
		super().__init__()
		self.config = AutoConfig.from_pretrained(model_path)
		self.gpt = AutoModel.from_pretrained(model_path)
		self.output = nn.Linear(self.config.hidden_size, class_num)
	
	def forward(self, input_ids, attention_mask):
		gpt_output = self.gpt(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
		preds = self.output(gpt_output.squeeze(1)[:, -1, :])
		return preds


class BertKGPrompt(nn.Module):
	def __init__(self, model_path, base_path, checkpoint, prompt_map_path, class_num=2) -> None:
		super().__init__()
		self.config = AutoConfig.from_pretrained(model_path)
		self.bert = AutoModel.from_pretrained(model_path)
		self.output = nn.Linear(self.config.hidden_size, class_num)
		self.prompt_encoder = PromptEncoder(base_path, checkpoint, prompt_map_path)

	
	def forward(self, input_ids, attention_mask, item_ids):
		bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
		prompts = self.prompt_encoder(item_ids)
		preds = self.output(bert_output + prompts)
		return preds
	
class GPTKGPrompt(nn.Module):
	def __init__(self, model_path, base_path, checkpoint, prompt_map_path, class_num=2) -> None:
		super().__init__()
		self.config = AutoConfig.from_pretrained(model_path)
		self.gpt = AutoModel.from_pretrained(model_path)
		self.output = nn.Linear(self.config.hidden_size, class_num)
		self.prompt_encoder = PromptEncoder(base_path, checkpoint, prompt_map_path)

	
	def forward(self, input_ids, attention_mask, item_ids):
		gpt_output = self.gpt(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
		prompts = self.prompt_encoder(item_ids)
		preds = self.output(gpt_output.squeeze(1)[:, -1, :] + prompts)
		return preds


"""
class PromptEncoder(nn.Module):
	def __init__(self, base_path, checkpoint, prompt_map_path) -> None:
		super().__init__()
		prompt_map_data = pkl.load(open(prompt_map_path, "rb"))
		self.prompt_proj = nn.Linear(256, 768)
		self.input_ids = torch.tensor(prompt_map_data['input_ids']).cuda()
		self.rel_ids = torch.tensor(prompt_map_data['rel_ids']).cuda()
		self.attention_masks = torch.tensor(prompt_map_data['attention_masks']).cuda()
		self.type_ids = torch.tensor(prompt_map_data['type_ids']).cuda()
		self.id_map = prompt_map_data['id_map']

		self.config = OurConfig.from_json_file("{}/config.json".format(base_path))
		self.prompt_encoder = OurPromptTuningModel(self.config, base_path + "/" + checkpoint)


	def forward(self, item_ids):
		item_ids = torch.tensor([self.id_map[x.item()] for x in item_ids])
		prompt_embeds = self.prompt_encoder(
			input_ids = self.input_ids[item_ids, :],
			rel_ids = self.rel_ids[item_ids, :],
			attention_mask = self.attention_masks[item_ids, :],
			token_type_ids = self.type_ids[item_ids, :]
		).pooler_output
		return self.prompt_proj(prompt_embeds)
"""

class PromptEncoder(nn.Module):
	def __init__(self, base_path, checkpoint, prompt_map_path) -> None:
		super().__init__()
		prompt_map_data = pkl.load(open(prompt_map_path, "rb"))
		self.prompt_proj = nn.Linear(256, 768)
		self.input_ids = torch.tensor(prompt_map_data['input_ids']).cuda()
		self.rel_ids = torch.tensor(prompt_map_data['rel_ids']).cuda()
		self.attention_masks = torch.tensor(prompt_map_data['attention_masks']).cuda()
		self.type_ids = torch.tensor(prompt_map_data['type_ids']).cuda()
		self.id_map = prompt_map_data['id_map']

		self.config = OurConfig.from_json_file("{}/config.json".format(base_path))
		self.prompt_encoder = PKGM(self.config, base_path + "/" + checkpoint)


	def forward(self, item_ids):
		item_ids = torch.tensor([self.id_map[x.item()] for x in item_ids])
		prompt_embeds = self.prompt_encoder(
			input_ids = self.input_ids[item_ids, :],
			rel_ids = self.rel_ids[item_ids, :],
			attention_mask = self.attention_masks[item_ids, :],
			token_type_ids = self.type_ids[item_ids, :]
		)
		return self.prompt_proj(prompt_embeds)