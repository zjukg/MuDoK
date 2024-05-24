import torch
import numpy as np
import torch_sparse
import torch.nn as nn
import scipy.sparse as sp
from config.configurator import configs
from models.aug_utils import AdaptiveMask
from models.general_cf.lightgcn import BaseModel
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss

import pickle as pkl
from pretrain.models.modeling_pretrain import OurConfig
from pretrain.models.modeling_ptuning import OurPromptTuningModel

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class DCCF_PT(BaseModel):
	def __init__(self, data_handler):
		super(DCCF_PT, self).__init__(data_handler)

		# prepare adjacency matrix for DCCF
		rows = data_handler.trn_mat.tocoo().row
		cols = data_handler.trn_mat.tocoo().col
		new_rows = np.concatenate([rows, cols + self.user_num], axis=0)
		new_cols = np.concatenate([cols + self.user_num, rows], axis=0)
		plain_adj = sp.coo_matrix((np.ones(len(new_rows)), (new_rows, new_cols)), shape=[self.user_num + self.item_num, self.user_num + self.item_num]).tocsr().tocoo()
		self.all_h_list = list(plain_adj.row)
		self.all_t_list = list(plain_adj.col)
		self.A_in_shape = plain_adj.shape
		self.A_indices = torch.tensor([self.all_h_list, self.all_t_list], dtype=torch.long).cuda()
		self.D_indices = torch.tensor([list(range(self.user_num + self.item_num)), list(range(self.user_num + self.item_num))], dtype=torch.long).cuda()
		self.all_h_list = torch.LongTensor(self.all_h_list).cuda()
		self.all_t_list = torch.LongTensor(self.all_t_list).cuda()
		self.G_indices, self.G_values = self._cal_sparse_adj()
		self.adaptive_masker = AdaptiveMask(head_list=self.all_h_list, tail_list=self.all_t_list, matrix_shape=self.A_in_shape)

		# hyper parameters
		self.layer_num = configs['model']['layer_num']
		self.intent_num = configs['model']['intent_num']
		self.reg_weight = configs['model']['reg_weight']
		self.cl_weight = configs['model']['cl_weight']
		self.temperature = configs['model']['temperature']

		# model parameters
		self.user_embeds = nn.Embedding(self.user_num, self.embedding_size)
		self.item_embeds = nn.Embedding(self.item_num, self.embedding_size)
		self.user_intent = torch.nn.Parameter(init(torch.empty(self.embedding_size, self.intent_num)), requires_grad=True)
		self.item_intent = torch.nn.Parameter(init(torch.empty(self.embedding_size, self.intent_num)), requires_grad=True)

		# train/test
		self.is_training = True
		self.final_embeds = None

		self._init_weight()
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
		self.input_ids = torch.tensor(input_ids).cuda()
		self.rel_ids = torch.tensor(rel_ids).cuda()
		self.attention_masks = torch.tensor(attention_masks).cuda()
		self.type_ids = torch.tensor(type_ids).cuda()
		self.prompt_proj = nn.Linear(self.config.hidden_size, self.embedding_size)
	
	def get_item_prompt(self):
		prompt_embeds = self.prompt_encoder(
			input_ids = self.input_ids,
			rel_ids = self.rel_ids,
			attention_mask = self.attention_masks,
			token_type_ids = self.type_ids
		).pooler_output
		return self.prompt_proj(prompt_embeds)

	def _init_weight(self):
		init(self.user_embeds.weight)
		init(self.item_embeds.weight)

	def _cal_sparse_adj(self):
		A_values = torch.ones(size=(len(self.all_h_list), 1)).view(-1).cuda()
		A_tensor = torch_sparse.SparseTensor(row=self.all_h_list, col=self.all_t_list, value=A_values, sparse_sizes=self.A_in_shape).cuda()
		D_values = A_tensor.sum(dim=1).pow(-0.5)
		G_indices, G_values = torch_sparse.spspmm(self.D_indices, D_values, self.A_indices, A_values, self.A_in_shape[0], self.A_in_shape[1], self.A_in_shape[1])
		G_indices, G_values = torch_sparse.spspmm(G_indices, G_values, self.D_indices, D_values, self.A_in_shape[0], self.A_in_shape[1], self.A_in_shape[1])
		return G_indices, G_values

	def forward(self):
		if not self.is_training and self.final_embeds is not None:
			return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:], None, None, None, None
		pooler_output = self.get_item_prompt() * 0.01
		item_embeds = self.item_embeds.weight + 0.01 * pooler_output
		all_embeds = [torch.concat([self.user_embeds.weight, item_embeds], dim=0)]
		gnn_embeds, int_embeds, gaa_embeds, iaa_embeds = [], [], [], []

		for i in range(0, self.layer_num):
			# Graph-based Message Passing
			gnn_layer_embeds = torch_sparse.spmm(self.G_indices, self.G_values, self.A_in_shape[0], self.A_in_shape[1], all_embeds[i])

			# Intent-aware Information Aggregation
			u_embeds, i_embeds = torch.split(all_embeds[i], [self.user_num, self.item_num], 0)
			u_int_embeds = torch.softmax(u_embeds @ self.user_intent, dim=1) @ self.user_intent.T
			i_int_embeds = torch.softmax(i_embeds @ self.item_intent, dim=1) @ self.item_intent.T
			int_layer_embeds = torch.concat([u_int_embeds, i_int_embeds], dim=0)

			# Adaptive Augmentation
			gnn_head_embeds = torch.index_select(gnn_layer_embeds, 0, self.all_h_list)
			gnn_tail_embeds = torch.index_select(gnn_layer_embeds, 0, self.all_t_list)
			int_head_embeds = torch.index_select(int_layer_embeds, 0, self.all_h_list)
			int_tail_embeds = torch.index_select(int_layer_embeds, 0, self.all_t_list)
			G_graph_indices, G_graph_values = self.adaptive_masker(gnn_head_embeds, gnn_tail_embeds)
			G_inten_indices, G_inten_values = self.adaptive_masker(int_head_embeds, int_tail_embeds)
			gaa_layer_embeds = torch_sparse.spmm(G_graph_indices, G_graph_values, self.A_in_shape[0], self.A_in_shape[1], all_embeds[i])
			iaa_layer_embeds = torch_sparse.spmm(G_inten_indices, G_inten_values, self.A_in_shape[0], self.A_in_shape[1], all_embeds[i])

			# Aggregation
			gnn_embeds.append(gnn_layer_embeds)
			int_embeds.append(int_layer_embeds)
			gaa_embeds.append(gaa_layer_embeds)
			iaa_embeds.append(iaa_layer_embeds)
			all_embeds.append(gnn_layer_embeds + int_layer_embeds + gaa_layer_embeds + iaa_layer_embeds + all_embeds[i])

		all_embeds = torch.stack(all_embeds, dim=1)
		all_embeds = torch.sum(all_embeds, dim=1, keepdim=False)
		user_embeds, item_embeds = torch.split(all_embeds, [self.user_num, self.item_num], 0)
		self.final_embeds = all_embeds
		return user_embeds, item_embeds, gnn_embeds, int_embeds, gaa_embeds, iaa_embeds

	def _cal_cl_loss(self, users, positems, negitems, gnn_emb, int_emb, gaa_emb, iaa_emb):
		users = torch.unique(users)
		items = torch.unique(torch.concat([positems, negitems]))
		cl_loss = 0.0
		for i in range(len(gnn_emb)):
			u_gnn_embs, i_gnn_embs = torch.split(gnn_emb[i], [self.user_num, self.item_num], 0)
			u_int_embs, i_int_embs = torch.split(int_emb[i], [self.user_num, self.item_num], 0)
			u_gaa_embs, i_gaa_embs = torch.split(gaa_emb[i], [self.user_num, self.item_num], 0)
			u_iaa_embs, i_iaa_embs = torch.split(iaa_emb[i], [self.user_num, self.item_num], 0)

			u_gnn_embs = u_gnn_embs[users]
			u_int_embs = u_int_embs[users]
			u_gaa_embs = u_gaa_embs[users]
			u_iaa_embs = u_iaa_embs[users]

			i_gnn_embs = i_gnn_embs[items]
			i_int_embs = i_int_embs[items]
			i_gaa_embs = i_gaa_embs[items]
			i_iaa_embs = i_iaa_embs[items]

			cl_loss += cal_infonce_loss(u_gnn_embs, u_int_embs, u_int_embs, self.temperature) / u_gnn_embs.shape[0]
			cl_loss += cal_infonce_loss(u_gnn_embs, u_gaa_embs, u_gaa_embs, self.temperature) / u_gnn_embs.shape[0]
			cl_loss += cal_infonce_loss(u_gnn_embs, u_iaa_embs, u_iaa_embs, self.temperature) / u_gnn_embs.shape[0]
			cl_loss += cal_infonce_loss(i_gnn_embs, i_int_embs, i_int_embs, self.temperature) / u_gnn_embs.shape[0]
			cl_loss += cal_infonce_loss(i_gnn_embs, i_gaa_embs, i_gaa_embs, self.temperature) / u_gnn_embs.shape[0]
			cl_loss += cal_infonce_loss(i_gnn_embs, i_iaa_embs, i_iaa_embs, self.temperature) / u_gnn_embs.shape[0]
		
		return cl_loss

	def cal_loss(self, batch_data):
		self.is_training = True
		user_embeds, item_embeds, gnn_embeds, int_embeds, gaa_embeds, iaa_embeds = self.forward()
		ancs, poss, negs = batch_data
		anc_embeds = user_embeds[ancs]
		pos_embeds = item_embeds[poss]
		neg_embeds = item_embeds[negs]
		bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
		reg_loss = self.reg_weight * reg_params(self)
		cl_loss = self.cl_weight * self._cal_cl_loss(ancs, poss, negs, gnn_embeds, int_embeds, gaa_embeds, iaa_embeds)
		loss = bpr_loss + reg_loss + cl_loss
		losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'cl_loss': cl_loss}
		return loss, losses

	def full_predict(self, batch_data):
		user_embeds, item_embeds, _, _, _, _ = self.forward()
		self.is_training = False
		pck_users, train_mask = batch_data
		pck_users = pck_users.long()
		pck_user_embeds = user_embeds[pck_users]
		full_preds = pck_user_embeds @ item_embeds.T
		full_preds = self._mask_predict(full_preds, train_mask)
		return full_preds