# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torchmetrics.functional import pairwise_cosine_similarity
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType
import faiss


class RecDCL(GeneralRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(RecDCL, self).__init__(config, dataset)

        # load parameters info
        self.batch_size = config['train_batch_size']
        self.embedding_size = config['embedding_size']

        self.encoder_name = config['encoder']
        self.reg_weight = config['reg_weight']
       
        self.a = config['a']
        self.polyc = config['polyc']
        self.degree = config['degree']
        self.poly_coeff = config['poly_coeff']
        self.bt_coeff = config['bt_coeff']
        self.all_bt_coeff = config['all_bt_coeff']
        self.mom_coeff = config['mom_coeff']
        self.momentum = config['momentum']

        # define layers and loss
        if self.encoder_name == 'MF':
            self.encoder = MFEncoder(self.n_users, self.n_items, self.embedding_size)
        elif self.encoder_name == 'LightGCN':
            self.n_layers = config['n_layers']
            self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
            self.norm_adj = self.get_norm_adj_mat().to(self.device)
            self.encoder = LGCNEncoder(self.n_users, self.n_items, self.embedding_size, self.norm_adj, self.n_layers)
        else:
            raise ValueError('Non-implemented Encoder.')

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        self.bn = nn.BatchNorm1d(self.embedding_size, affine=False)

        layers = []
        embs = str(self.embedding_size) + '-' + str(self.embedding_size) + '-' + str(self.embedding_size)
        sizes = [self.embedding_size] + list(map(int, embs.split('-')))
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # parameters initialization
        self.apply(xavier_normal_initialization)

        self.predictor = nn.Linear(self.embedding_size, self.embedding_size)

        self.u_target_his = torch.randn((self.n_users, self.embedding_size), requires_grad=False).to(self.device)
        self.i_target_his = torch.randn((self.n_items, self.embedding_size), requires_grad=False).to(self.device)

    def get_norm_adj_mat(self):
        # build adj matrix
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def forward(self, user, item):
        user_e, item_e, lightgcn_all_embeddings = self.encoder(user, item)

        with torch.no_grad():
            u_target, i_target = self.u_target_his.clone()[user, :], self.i_target_his.clone()[item, :]
            u_target.detach()
            i_target.detach()

            u_target = u_target * self.momentum + user_e.data * (1. - self.momentum)
            i_target = i_target * self.momentum + item_e.data * (1. - self.momentum)

            self.u_target_his[user, :] = user_e.clone()
            self.i_target_his[item, :] = item_e.clone()
            
        return user_e, item_e, lightgcn_all_embeddings, u_target, i_target

    @staticmethod
    def off_diagonal(x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    
    def bt(self, x, y):
        user_e = self.projector(x) 
        item_e = self.projector(y) 
        c = self.bn(user_e).T @ self.bn(item_e)
        c.div_(user_e.size()[0])
        # sum the cross-correlation matrix
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().div(self.embedding_size)
        off_diag = self.off_diagonal(c).pow_(2).sum().div(self.embedding_size)
        bt = on_diag + self.bt_coeff * off_diag
        return bt

    def poly_feature(self, x):
        user_e = self.projector(x) 
        xx = self.bn(user_e).T @ self.bn(user_e)
        poly = (self.a * xx + self.polyc) ** self.degree 
        return poly.mean().log()

    def loss_fn(self, p, z):  # cosine similarity
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e, item_e, embeddings_list, u_target, i_target = self.forward(user, item)
        user_e_n, item_e_n = F.normalize(user_e, dim=-1), F.normalize(item_e, dim=-1)
        user_e, item_e = self.predictor(user_e), self.predictor(item_e)
        if self.all_bt_coeff == 0:
            bt_loss = 0.0
        else:
            bt_loss = self.bt(user_e_n, item_e_n)

        if self.poly_coeff == 0:
            poly_loss = 0.0
        else:
            poly_loss = self.poly_feature(user_e_n) / 2 + self.poly_feature(item_e_n) / 2 
        
        if self.mom_coeff == 0:
            mom_loss = 0.0
        else:    
            mom_loss = self.loss_fn(user_e, i_target) / 2 + self.loss_fn(item_e, u_target) / 2

        return self.all_bt_coeff * bt_loss + poly_loss * self.poly_coeff + mom_loss * self.mom_coeff
       
    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)
        return torch.mul(user_e, item_e).sum(dim=1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.encoder_name == 'LightGCN':
            if self.restore_user_e is None or self.restore_item_e is None:
                self.restore_user_e, self.restore_item_e, _ = self.encoder.get_all_embeddings()
            user_e = self.restore_user_e[user]
            all_item_e = self.restore_item_e
        else:
            user_e = self.encoder.user_embedding(user)
            all_item_e = self.encoder.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)

class MFEncoder(nn.Module):
    def __init__(self, user_num, item_num, emb_size):
        super(MFEncoder, self).__init__()
        self.user_embedding = nn.Embedding(user_num, emb_size)
        self.item_embedding = nn.Embedding(item_num, emb_size)

    def forward(self, user_id, item_id):
        u_embed = self.user_embedding(user_id)
        i_embed = self.item_embedding(item_id)
        return u_embed, i_embed

    def get_all_embeddings(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        return user_embeddings, item_embeddings


class LGCNEncoder(nn.Module):
    def __init__(self, user_num, item_num, emb_size, norm_adj, n_layers=3):
        super(LGCNEncoder, self).__init__()
        self.n_users = user_num
        self.n_items = item_num
        self.n_layers = n_layers
        self.norm_adj = norm_adj

        self.user_embedding = torch.nn.Embedding(user_num, emb_size)
        self.item_embedding = torch.nn.Embedding(item_num, emb_size)

    def get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def get_all_embeddings(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings, lightgcn_all_embeddings

    def forward(self, user_id, item_id):
        user_all_embeddings, item_all_embeddings, lightgcn_all_embeddings = self.get_all_embeddings()
        u_embed = user_all_embeddings[user_id]
        i_embed = item_all_embeddings[item_id]
        return u_embed, i_embed, lightgcn_all_embeddings
