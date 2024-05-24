import torch as t
import pickle as pkl
from torch import nn
from models.aug_utils import EdgeDrop
from models.base_model import BaseModel
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_params
from collections import defaultdict

from pretrain.models.modeling_pretrain import OurConfig
from pretrain.models.modeling_ptuning import OurPromptTuningModel

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class LightGCN_PT(BaseModel):
	def __init__(self, data_handler):
		super(LightGCN_PT, self).__init__(data_handler)
		self.adj = data_handler.torch_adj
		# New Prompt Encoder
		print("LightGCN With Prompt Tuning.")
		base_path = configs["train"]["pretrain_config"]
		checkpoint = configs["train"]["checkpoint"]
		self.config = OurConfig.from_json_file("{}/config.json".format(base_path))
		self.prompt_encoder = OurPromptTuningModel(self.config, base_path + "/" + checkpoint)
		prompt_dataset = pkl.load(open(configs["train"]["kg_path"], "rb"))
		
		self.indexs = prompt_dataset["indexs"]

		self.layer_num = configs['model']['layer_num']
		self.reg_weight = configs['model']['reg_weight']
		self.keep_rate = configs['model']['keep_rate']

		self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
		self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
		self.item_id_map = defaultdict(int)
		assert data_handler.kg_map is not None
		self.item2index = data_handler.kg_map
		input_ids = []
		rel_ids = []
		attention_masks = []
		type_ids = []
		for i in range(self.item_num):
			index = self.item2index[i]
			input_ids.append(prompt_dataset["input_ids"][index])
			rel_ids.append(prompt_dataset["rel_ids"][index])
			attention_masks.append(prompt_dataset["attention_masks"][index])
			type_ids.append(prompt_dataset["type_ids"])
		self.input_ids = t.tensor(input_ids).cuda()
		self.rel_ids = t.tensor(rel_ids).cuda()
		self.attention_masks = t.tensor(attention_masks).cuda()
		self.type_ids = t.tensor(type_ids).cuda()
		self.prompt_proj = nn.Linear(self.config.hidden_size, self.embedding_size)
		# total_params = sum(p.numel() for p in self.prompt_encoder.parameters())
		# trainable_params = sum(p.numel() for p in self.prompt_encoder.parameters() if p.requires_grad)
		# print(trainable_params, total_params, trainable_params / total_params)

		self.edge_dropper = EdgeDrop()
		self.is_training = True
		self.final_embeds = None
	
	def _propagate(self, adj, embeds):
		return t.spmm(adj, embeds)

	def get_item_prompt(self):
		prompt_embeds = self.prompt_encoder(
			input_ids = self.input_ids,
			rel_ids = self.rel_ids,
			attention_mask = self.attention_masks,
			token_type_ids = self.type_ids
		).pooler_output
		return self.prompt_proj(prompt_embeds)
	
	def forward(self, adj, keep_rate):
		if not self.is_training and self.final_embeds is not None:
			return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:]
		pooler_output = self.get_item_prompt()
		item_embeds = self.item_embeds + 0.01 * pooler_output
		embeds = t.concat([self.user_embeds, item_embeds], axis=0)
		embeds_list = [embeds]
		if self.is_training:
			adj = self.edge_dropper(adj, keep_rate)
		for i in range(self.layer_num):
			embeds = self._propagate(adj, embeds_list[-1])
			embeds_list.append(embeds)
		embeds = sum(embeds_list)# / len(embeds_list)
		self.final_embeds = embeds
		return embeds[:self.user_num], embeds[self.user_num:]
	
	def cal_loss(self, batch_data):
		self.is_training = True
		user_embeds, item_embeds = self.forward(self.adj, self.keep_rate)
		ancs, poss, negs = batch_data
		anc_embeds = user_embeds[ancs]
		pos_embeds = item_embeds[poss]
		neg_embeds = item_embeds[negs]
		bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
		reg_loss = self.reg_weight * reg_params(self)
		loss = bpr_loss + reg_loss
		losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss}
		return loss, losses

	def full_predict(self, batch_data):
		user_embeds, item_embeds = self.forward(self.adj, 1.0)
		self.is_training = False
		pck_users, train_mask = batch_data
		pck_users = pck_users.long()
		pck_user_embeds = user_embeds[pck_users]
		full_preds = pck_user_embeds @ item_embeds.T
		full_preds = self._mask_predict(full_preds, train_mask)
		return full_preds
