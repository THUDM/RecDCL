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


class BT4Rec(GeneralRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(BT4Rec, self).__init__(config, dataset)

        # load parameters info
        self.batch_size = config['train_batch_size']
        self.embedding_size = config['embedding_size']
        self.gamma = config['gamma']
        self.encoder_name = config['encoder']
        self.reg_weight = config['reg_weight']
        self.a = config['a']
        self.polyc = config['polyc']
        self.degree = config['degree']
        self.bt_coeff = config['bt_coeff']
        self.warm_up = config['warm_up']
        self.uniform_cof = config['uniform_cof']

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
        user_e, item_e = self.encoder(user, item)
        return F.normalize(user_e, dim=-1), F.normalize(item_e, dim=-1)

    @staticmethod
    def alignment(x, y, alpha=2):
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()

    # @staticmethod
    def uniformity(self, input, t=2):
        return torch.pdist(input, p=2).pow(2).mul(-t).exp().mean().log()

    @staticmethod
    def off_diagonal(x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    
    @staticmethod
    def eigh_batch(x, t=2, eps=1e-3):
        f = (x.shape[0] - 1) / x.shape[0]      # 方差调整系数
        x_reducemean = x - torch.mean(x, axis=0)
        numerator = torch.matmul(x_reducemean.T, x_reducemean) / x.shape[0]
        var_ = x.var(axis=0).reshape(x.shape[1], 1)
        denominator = torch.sqrt(torch.matmul(var_, var_.T)) * f
        xx = numerator / denominator
        # return eigenvalues and eigenvectors of x
        # xx1 = xx.pow(2).mul(-2).exp()
        vals = torch.linalg.eigvalsh(xx)
        # return -vals[vals > eps].log().sum()
        return vals[vals > eps].log().sum()

    def poly_feature(self, x):
        user_e = self.projector(x) 
        xx = self.bn(user_e).T @ self.bn(user_e)
        poly = (self.a * xx + self.polyc) ** self.degree 
        return poly.mean().log()

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
    
    def mse(self, x, y):
        x = self.projector(x) 
        y = self.projector(y) 
        return F.mse_loss(x, y)
    
    def std(self, x, y):
        x = self.projector(x) 
        y = self.projector(y)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2
        return std_loss

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_e, item_e = self.forward(user, item)
        # repr_loss = self.mse(user_e, item_e)
        std_loss =self.std(user_e, item_e)
        # vicreg_loss = self.vicreg(user_e, item_e)
        bt_loss = self.bt(user_e, item_e)
        # align = self.alignment(user_e, item_e)
        # uniform = (self.uniformity(user_e) + self.uniformity(item_e)) / 2
        poly_loss = self.poly_feature(item_e) / 2 + self.poly_feature(user_e) / 2 
        # return repr_loss + std_loss +
        # self.gamma = min(self.gamma, (self.gamma * float(num_batch) / self.warm_up))
        # print('self.warm_up: ', self.warm_up)
        # print('self.num_batch: ', num_batch)
        # print('self.gamma: ', self.gamma, '\n')
        return bt_loss + self.gamma * poly_loss + std_loss
        # return (1 - self.gamma) * bt_loss + self.gamma * poly_loss # vicreg_loss 

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
                self.restore_user_e, self.restore_item_e = self.encoder.get_all_embeddings()
            user_e = self.restore_user_e[user]
            all_item_e = self.restore_item_e
        else:
            user_e = self.encoder.user_embedding(user)
            all_item_e = self.encoder.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)

    def vicreg(self, x, y):
        x = self.projector(x) 
        y = self.projector(y) 
        repr_loss = F.mse_loss(x, y)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2
        cov_x = (x.T @ x) / (self.batch_size - 1)
        cov_y = (y.T @ y) / (self.batch_size - 1)
        cov_loss = self.off_diagonal(cov_x).pow_(2).sum().div(self.embedding_size) + self.off_diagonal(cov_y).pow_(2).sum().div(self.embedding_size)
        # loss = (
        #     self.sim_coeff * repr_loss
        #     + self.std_coeff * std_loss
        #     + self.cov_coeff * cov_loss
        # )
        loss = 25 * repr_loss + 25 * std_loss + 1 * cov_loss
        return loss


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
        return user_all_embeddings, item_all_embeddings

    def forward(self, user_id, item_id):
        user_all_embeddings, item_all_embeddings = self.get_all_embeddings()
        u_embed = user_all_embeddings[user_id]
        i_embed = item_all_embeddings[item_id]
        return u_embed, i_embed