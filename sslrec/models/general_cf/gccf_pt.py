import torch as t
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_params
from models.base_model import BaseModel

import pickle as pkl
from pretrain.models.modeling_pretrain import OurConfig
from pretrain.models.modeling_ptuning import OurPromptTuningModel

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class GCNLayer(nn.Module):
	def __init__(self, latdim):
		super(GCNLayer, self).__init__()
		self.W = nn.Parameter(init(t.empty(latdim, latdim)))

	def forward(self, adj, embeds):
		return t.spmm(adj, embeds) # @ self.W (Performs better without W)

class GCCF_PT(BaseModel):
	def __init__(self, data_handler):
		super(GCCF_PT, self).__init__(data_handler)

		self.adj = data_handler.torch_adj
		
		# hyper-parameter
		self.layer_num = configs['model']['layer_num']
		self.reg_weight = configs['model']['reg_weight']
		
		self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
		self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
		self.gcnLayers = nn.Sequential(*[GCNLayer(self.embedding_size) for i in range(self.layer_num)])
		self.is_training = True

		base_path = configs["train"]["pretrain_config"]
		checkpoint = configs["train"]["checkpoint"]
		self.config = OurConfig.from_json_file("{}/config.json".format(base_path))
		self.prompt_encoder = OurPromptTuningModel(self.config, base_path + "/" + checkpoint)
		prompt_dataset = pkl.load(open(configs["train"]["kg_path"], "rb"))

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
	
	def get_item_prompt(self):
		prompt_embeds = self.prompt_encoder(
			input_ids = self.input_ids,
			rel_ids = self.rel_ids,
			attention_mask = self.attention_masks,
			token_type_ids = self.type_ids
		).pooler_output
		return self.prompt_proj(prompt_embeds)
		
	
	def forward(self, adj=None):
		if adj is None:
			adj = self.adj
		if not self.is_training:
			return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:], None
		pooler_output = self.get_item_prompt() * 0.01
		item_embeds = self.item_embeds + 0.01 * pooler_output
		embeds = t.concat([self.user_embeds, item_embeds], axis=0)
		embeds_list = [embeds]
		for gcn in self.gcnLayers:
			embeds = gcn(adj, embeds_list[-1])
			embeds_list.append(embeds)
		embeds = t.concat(embeds_list, dim=-1)
		self.final_embeds = embeds
		return embeds[:self.user_num], embeds[self.user_num:], embeds_list[-1]
	
	def cal_loss(self, batch_data):
		self.is_training = True
		user_embeds, item_embeds, _ = self.forward(self.adj)
		ancs, poss, negs = batch_data
		anc_embeds = user_embeds[ancs]
		pos_embeds = item_embeds[poss]
		neg_embeds = item_embeds[negs]
		bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
		reg_loss = self.reg_weight * reg_params(self)
		loss = bpr_loss + reg_loss
		losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss}
		return loss, losses

	# def _predict_all_wo_mask(self, ancs):
	#     user_embeds, item_embeds = self.forward(self.adj)
	#     pck_users = ancs
	#     pck_user_embeds = user_embeds[pck_users]
	#     full_preds = pck_user_embeds @ item_embeds.T
	#     return full_preds

	def full_predict(self, batch_data):
		user_embeds, item_embeds, _ = self.forward(self.adj)
		self.is_training = False
		pck_users, train_mask = batch_data
		pck_users = pck_users.long()
		pck_user_embeds = user_embeds[pck_users]
		full_preds = pck_user_embeds @ item_embeds.T
		full_preds = self._mask_predict(full_preds, train_mask)
		return full_preds
