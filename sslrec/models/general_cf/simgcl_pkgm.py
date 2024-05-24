import torch as t
import pickle as pkl
from torch import nn
from config.configurator import configs
from models.aug_utils import EmbedPerturb
from models.general_cf.lightgcn import LightGCN
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss

from pretrain.models.modeling_pretrain import OurConfig
from pretrain.models.modeling_ptuning import PKGM

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class SimGCL_PKGM(LightGCN):
	def __init__(self, data_handler):
		super(SimGCL_PKGM, self).__init__(data_handler)
		
		self.cl_weight = configs['model']['cl_weight']
		self.temperature = configs['model']['temperature']
		self.eps = configs['model']['eps']
		self.embed_perturb = EmbedPerturb(eps=self.eps)
		base_path = configs["train"]["pretrain_config"]
		checkpoint = configs["train"]["checkpoint"]
		self.config = OurConfig.from_json_file("{}/config.json".format(base_path))
		self.prompt_encoder = PKGM(self.config, base_path + "/" + checkpoint)
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
		)
		return self.prompt_proj(prompt_embeds)
	
	def forward(self, adj, perturb=False):
		if not perturb:
			return super(SimGCL_PKGM, self).forward(adj, 1.0)
		pooler_output = self.get_item_prompt()
		item_embeds = self.item_embeds + 0.01 * pooler_output
		embeds = t.concat([self.user_embeds, item_embeds], dim=0)
		embeds_list = [embeds]
		for i in range(self.layer_num):
			embeds = self._propagate(adj, embeds_list[-1])
			embeds = self.embed_perturb(embeds)
			embeds_list.append(embeds)
		embeds = sum(embeds_list)
		return embeds[:self.user_num], embeds[self.user_num:]
	
	def _pick_embeds(self, user_embeds, item_embeds, batch_data):
		ancs, poss, negs = batch_data
		anc_embeds = user_embeds[ancs]
		pos_embeds = item_embeds[poss]
		neg_embeds = item_embeds[negs]
		return anc_embeds, pos_embeds, neg_embeds
		
	def cal_loss(self, batch_data):
		self.is_training = True
		user_embeds1, item_embeds1 = self.forward(self.adj, perturb=True)
		user_embeds2, item_embeds2 = self.forward(self.adj, perturb=True)
		user_embeds3, item_embeds3 = self.forward(self.adj, perturb=False)

		anc_embeds1, pos_embeds1, neg_embeds1 = self._pick_embeds(user_embeds1, item_embeds1, batch_data)
		anc_embeds2, pos_embeds2, neg_embeds2 = self._pick_embeds(user_embeds2, item_embeds2, batch_data)
		anc_embeds3, pos_embeds3, neg_embeds3 = self._pick_embeds(user_embeds3, item_embeds3, batch_data)
		bpr_loss = cal_bpr_loss(anc_embeds3, pos_embeds3, neg_embeds3) / anc_embeds3.shape[0]
		cl_loss = cal_infonce_loss(anc_embeds1, anc_embeds2, user_embeds2, self.temperature) + cal_infonce_loss(pos_embeds1, pos_embeds2, item_embeds2, self.temperature)
		cl_loss /= anc_embeds1.shape[0]
		reg_loss = self.reg_weight * reg_params(self)
		cl_loss *= self.cl_weight
		loss = bpr_loss + reg_loss + cl_loss
		losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'cl_loss': cl_loss}
		return loss, losses

	def full_predict(self, batch_data):
		user_embeds, item_embeds = self.forward(self.adj, False)
		self.is_training = False
		pck_users, train_mask = batch_data
		pck_users = pck_users.long()
		pck_user_embeds = user_embeds[pck_users]
		full_preds = pck_user_embeds @ item_embeds.T
		full_preds = self._mask_predict(full_preds, train_mask)
		return full_preds