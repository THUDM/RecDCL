# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import pairwise_cosine_similarity

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType


class BT4Rec(GeneralRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(BT4Rec, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.gamma = config['gamma']
        self.encoder_name = config['encoder']
        self.reg_weight = config['reg_weight']

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

    # @staticmethod
    # def alignment(x, y, alpha=2):
    #     return (x - y).norm(p=2, dim=1).pow(alpha).mean()

    @staticmethod
    def uniformity(x, t=2):
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    @staticmethod
    def off_diagonal(x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def dpp(x, y):
        pass

        
    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_e, item_e = self.forward(user, item)
        # uniform = self.uniformity(user_e) + self.uniformity(item_e)
        # user_e = self.projector(user_e) 
        # item_e = self.projector(item_e) 
        # # uniform = self.uniformity(user_e) + self.uniformity(item_e)
        # # c = self.bn(user_e).T @ self.bn(item_e)
        # c = user_e.T @ item_e

        # # sum the cross-correlation matrix between all gpus
        # c.div_(user_e.size()[0])
        # # torch.distributed.all_reduce(c)

        # on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        # off_diag = self.off_diagonal(c).pow_(2).sum()
        # loss = on_diag
        # loss += self.reg_weight * off_diag
        # loss += 5 * uniform

        # c = self.bn(user_e) @ self.bn(item_e.T)
        c = user_e @ item_e.T
        c.div_(user_e.size()[0])
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum() 
        user_e_copy, item_e_copy = user_e.clone(), item_e.clone()
        item_item_sim = pairwise_cosine_similarity(item_e_copy.detach(), item_e_copy.detach())
        user_user_sim = pairwise_cosine_similarity(user_e_copy.detach(), user_e_copy.detach())
        user_loss, item_loss = 0., 0.
        # # for i in range(user_e.size()[0]):
        # #     user_i = user_e[i] @ item_e.T
        # #     user_loss = user_loss + user_i.pow(2).log()
        # # user_loss = user_loss + item_item_sim.log()

        # # for j in range(item_e.size()[0]):
        # #     item_j = item_e[j] @ user_e.T
        # #     item_loss = item_loss + item_j.pow(2).log()
        # # item_loss = item_loss + user_user_sim.log()
        # for i in range(user_e.size()[0]):
        #     user_i = user_e[i] @ item_e.T
        #     user_kernel = user_i.reshape(item_item_sim.shape[0], 1) * item_item_sim * user_i.reshape(1, item_item_sim.shape[0])
        #     user_loss = user_loss + user_kernel.pow(2).mul(-2).exp().mean().log()
        # user_loss = user_loss / user_e.size()[0]

        for i in range(user_e.size()[0]):
            user_i = user_e[i] @ item_e.T
            user_i_diag = torch.diag(user_i)
            user_kernel = user_i_diag @ item_item_sim @ user_i_diag.T
            user_loss = user_loss + 1 / user_kernel.mean().log()
        user_loss = user_loss / user_e.size()[0]
        print("user_loss: ", user_loss)

        for j in range(item_e.size()[0]):
            item_j = item_e[j] @ user_e.T
            # item_j_diag = torch.diag(item_j)
            item_kernel = item_j.reshape((user_user_sim.shape[0], 1)) * user_user_sim * item_j.reshape((1, user_user_sim.shape[0]))
            # item_kernel = item_j_diag @ user_user_sim @ item_j_diag.T
            item_loss = item_loss + 1 / item_kernel.mean().log()
        item_loss = item_loss / item_e.size()[0]
        # print("user_loss: ", user_loss) 
        print("item_loss: ", item_loss) 
        # user_item = user_e @ item_e.T
        # item_user = item_e @ user_e.T
        # user_item_T = (user_e @ item_e.T).T
        # item_user_T = (item_e @ user_e.T).T
        # item_item_sim = pairwise_cosine_similarity(item_e_copy.detach(), item_e_copy.detach())
        # user_user_sim = pairwise_cosine_similarity(user_e_copy.detach(), user_e_copy.detach())
        # user_kernel = user_item @ item_item_sim @ user_item_T
        # user_loss = user_kernel.mean().log()
        # item_kernel = item_user @ user_user_sim @ item_user_T
        # item_loss = item_kernel.mean().log()
        loss = on_diag + self.reg_weight * off_diag # + 1 * (user_loss + item_loss)
        # user_item = user_e.T @ item_e
        # item_user = item_e.T @ user_e
        # user_item_T = (user_e.T @ item_e).T
        # item_user_T = (item_e.T @ user_e).T
        # item_item_sim = pairwise_cosine_similarity(item_e_copy.T.detach(), item_e_copy.T.detach())
        # user_user_sim = pairwise_cosine_similarity(user_e_copy.T.detach(), user_e_copy.T.detach())
        # user_kernel = user_item @ item_item_sim @ user_item_T
        # user_loss = user_kernel.mean().log()
        # item_kernel = item_user @ user_user_sim @ item_user_T
        # item_loss = item_kernel.mean().log()
        # loss = on_diag + 0.005 * off_diag + 10 * (user_loss + item_loss)


        # u_mean = torch.mean(user_e, dim=0)
        # i_mean = torch.mean(item_e, dim=0)
        # c = (user_e - u_mean).T @ (item_e - i_mean)
        # c.div_(user_e.size()[0])
        # loss = torch.linalg.svd(c)[1].sum()
        return loss

        # align = self.alignment(user_e, item_e)
        # uniform = self.gamma * (self.uniformity(user_e) + self.uniformity(item_e)) / 2

        # return align + uniform

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
