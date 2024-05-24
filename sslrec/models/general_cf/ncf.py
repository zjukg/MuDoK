import torch as t
import torch.nn.functional as F
from torch import nn
from models.aug_utils import EdgeDrop
from models.base_model import BaseModel
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_params

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class NCF(BaseModel):
	def __init__(self, data_handler):
		super(NCF, self).__init__(data_handler)

		self.adj = data_handler.torch_adj

		self.layer_num = configs['model']['layer_num']
		self.reg_weight = configs['model']['reg_weight']
		self.keep_rate = configs['model']['keep_rate']

		self.user_embeds_mlp = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
		self.item_embeds_mlp = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))

		self.user_embeds_mf = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
		self.item_embeds_mf = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))

		self.mlp_layers = nn.Sequential(
			nn.Linear(self.embedding_size * 2, self.embedding_size),
			nn.ReLU(),
			nn.Linear(self.embedding_size, self.embedding_size),
			nn.ReLU()
		)

		self.pred_layer = nn.Sequential(
			nn.Linear(self.embedding_size * 2, 1),
			nn.Sigmoid()
		)
		

		self.edge_dropper = EdgeDrop()
		self.is_training = True
		self.final_embeds = None
		self.loss = nn.BCELoss()

	
	def _propagate(self, adj, embeds):
		return t.spmm(adj, embeds)
	
	def forward(self, adj, keep_rate):
		if not self.is_training and self.final_embeds is not None:
			return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:]
		return self.user_embeds, self.item_embeds

	def get_scores(self, users_mf, users_mlp, items_mf, items_mlp):
		embed_mlp = t.cat((users_mlp, items_mlp), dim=-1)
		mlp_output = self.mlp_layers(embed_mlp)
		embed_gmf = users_mf * items_mf
		final_input = t.cat((embed_gmf, mlp_output), dim=-1)
		final_output = self.pred_layer(final_input).view(-1)
		return final_output
	
	def cal_loss(self, batch_data):
		self.is_training = True
		# user_embeds, item_embeds = self.forward(self.adj, self.keep_rate)
		ancs, poss, negs = batch_data
		# anc_embeds = user_embeds[ancs]
		# pos_embeds = item_embeds[poss]
		# neg_embeds = item_embeds[negs]
		user_mf, user_mlp = self.user_embeds_mf[ancs], self.user_embeds_mlp[ancs]
		item_mf_pos, item_mlp_pos = self.item_embeds_mf[poss], self.item_embeds_mlp[poss]
		item_mf_neg, item_mlp_neg = self.item_embeds_mf[negs], self.item_embeds_mlp[negs]
		pos_score = self.get_scores(user_mf, user_mlp, item_mf_pos, item_mlp_pos)
		neg_score = self.get_scores(user_mf, user_mlp, item_mf_neg, item_mlp_neg)
		bpr_loss = self.loss(pos_score, t.ones_like(pos_score)) + self.loss(neg_score, t.zeros_like(neg_score))
		reg_loss = self.reg_weight * reg_params(self)
		loss = bpr_loss + reg_loss
		losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss}
		return loss, losses

	def full_predict(self, batch_data):
		# user_embeds, item_embeds = self.forward(self.adj, 1.0)
		
		self.is_training = False
		pck_users, train_mask = batch_data
		pck_users = pck_users.long()
		# pck_user_embeds = user_embeds[pck_users]
		user_mf, user_mlp = self.user_embeds_mf[pck_users], self.user_embeds_mlp[pck_users]
		item_mf, item_mlp = self.item_embeds_mf, self.item_embeds_mlp
		batch_size = user_mf.shape[0]
		item_size = self.item_num
		pred = []
		for i in range(batch_size):
			user_mf_temp = user_mf[i].repeat(item_size, 1)
			user_mlp_temp = user_mlp[i].repeat(item_size, 1)
			pred_temp = self.get_scores(user_mf_temp, user_mlp_temp, item_mf, item_mlp)
			pred.append(pred_temp)
		full_preds = t.stack(pred)
		full_preds = self._mask_predict(full_preds, train_mask)
		return full_preds