import math
import pdb
import numpy as np
import torch
from torch import Tensor, nn
from model_init import PETER
import torch.optim
import torch.nn.functional as F
from torch.nn import Parameter, Linear, Embedding, Module, BatchNorm1d, LayerNorm, PReLU
from configuration_3 import parse
args = parse()

class Attention(nn.Module):
    def __init__(self, sentence_embed_dim):
        super(Attention, self).__init__()
        self.embed_dim = sentence_embed_dim
        self.tanh = nn.Tanh()

    def forward(self, query, keys, values):
        """
            e = 512, k = num_reviews
            query shape   :   N X query_len X embed_dim   : (nqe)
            keys shape    :   N X key_len X embed_dim     : (nke)
            values shape  :   N X key_len X embed_dim     : (nke)
        """
        energy = torch.einsum("nqe,nke->nqk", [query, keys])
        attention = torch.softmax(energy, dim=-1)
        output = torch.einsum("nqk,nke->nqe", [attention, values])
        return attention, output

class OrderEmbedding(Module):
    def __init__(self, num_embeddings, embedding_dim, learnable=True, BatchNorm=True):
        super(OrderEmbedding, self).__init__()
        self.class_embedding = Parameter(torch.Tensor(num_embeddings, embedding_dim))
        torch.nn.init.xavier_normal_(self.class_embedding)
        norm_range = 1*torch.linspace(-1, 1, num_embeddings).view(-1, 1)
        self.norm_range = Parameter(norm_range, requires_grad=False)
        self.learnable = learnable
        if learnable:
            self.order_embedding = Parameter(torch.Tensor(1, embedding_dim))
            torch.nn.init.xavier_normal_(self.order_embedding)
        else:
            self.order_embedding = Parameter(torch.ones((1, embedding_dim))/math.sqrt(embedding_dim), requires_grad=False)
        self.batchNorm = BatchNorm1d(embedding_dim) if BatchNorm else None

    def forward(self, index_tensor):
        order_embed = self.norm_range@F.relu(self.order_embedding)
        if self.batchNorm:
            order_embed = self.batchNorm(order_embed)
        edge_embedding = self.class_embedding + order_embed
        return F.embedding(index_tensor, edge_embedding)

class MLP(nn.Module):
    def __init__(self, emsize=512):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(emsize, emsize)
        self.linear2 = nn.Linear(emsize, int(emsize/2))
        self.linear3 = nn.Linear(int(emsize/2), 1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.linear1.weight.data.uniform_(-initrange, initrange)
        self.linear2.weight.data.uniform_(-initrange, initrange)
        self.linear3.weight.data.uniform_(-initrange, initrange)
        self.linear1.bias.data.zero_()
        self.linear2.bias.data.zero_()
        self.linear3.bias.data.zero_()

    def forward(self, hidden):  # (batch_size, emsize)
        mlp_vector = self.linear1(hidden)
        mlp_vector = self.linear2(mlp_vector)
        rating = self.linear3(mlp_vector)
        rating = torch.squeeze(rating)
        return rating

def data_in_one(inputdata):
    min = np.nanmin(inputdata)
    max = np.nanmax(inputdata)
    outputdata = (inputdata-min)/(max-min)
    return outputdata

class Model(nn.Module):
    def __init__(self, src_len, tgt_len, pad_idx, nuser, nitem, ntokens, lamda):
        super(Model, self).__init__()
        self.user_embed_ra = nn.Embedding(num_embeddings=nuser, embedding_dim=args.emsize)
        self.user_embed_re = nn.Embedding(num_embeddings=nuser, embedding_dim=args.emsize)
        self.item_embed_ra = nn.Embedding(num_embeddings=nitem, embedding_dim=args.emsize)
        self.item_embed_re = nn.Embedding(num_embeddings=nitem, embedding_dim=args.emsize)
        # self.rating_embed = OrderEmbedding(num_embeddings=5, embedding_dim=args.emsize)
        self.rating_embed = nn.Embedding(num_embeddings=7, embedding_dim=args.emsize)
        self.pad_idx = pad_idx
        self.lamda = lamda
        self.linear1 = nn.Linear(args.emsize * 2, args.emsize, bias=True).to('cuda')
        self.linear2 = nn.Linear(args.emsize * 2, args.emsize, bias=True).to('cuda')
        self.linear3 = nn.Linear(args.emsize * 2, args.emsize).to('cuda')
        self.linear4 = nn.Linear(args.emsize * 2, args.emsize, bias=True).to('cuda')
        self.attn = Attention(args.emsize).to('cuda')
        self.layer_norm = LayerNorm(args.emsize).to('cuda')
        self.softmax = nn.Softmax(dim=-1).to('cuda')
        self.model = PETER(args.peter_mask, src_len, tgt_len, pad_idx, nuser, nitem, ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to('cuda')
        self.bn = nn.BatchNorm1d(512).to('cuda')
        self.de_ra = nn.Linear(1, tgt_len, bias=True).to('cuda')
        self.de_re = nn.Linear(args.emsize, args.emsize, bias=True).to('cuda')
        self.hidden2token = nn.Linear(args.emsize, ntokens)
        self.GELU = nn.GELU().to('cuda')

        self.mlp = MLP(args.emsize).to('cuda')
        self.remlp = MLP(args.emsize).to('cuda')

        self.W_1 = nn.Parameter(torch.FloatTensor(args.emsize, args.emsize), requires_grad=True).to('cuda')  # rating
        torch.nn.init.kaiming_uniform_(self.W_1)
        self.W_2 = nn.Parameter(torch.FloatTensor(args.emsize, args.emsize), requires_grad=True).to('cuda') # review
        torch.nn.init.kaiming_uniform_(self.W_2)
        self.init_weights()

    def init_weights(self):
        self.user_embed_ra.weight.data.normal_(mean=0.0, std=0.1)
        self.item_embed_ra.weight.data.normal_(mean=0.0, std=0.1)
        self.user_embed_re.weight.data.normal_(mean=0.0, std=0.1)
        self.item_embed_re.weight.data.normal_(mean=0.0, std=0.1)

    def forward(self, user, item, rating, fake_r, seq,  mode="Train"):
        # rating related
        user_embed_ra = self.user_embed_ra(user)
        item_embed_ra = self.item_embed_ra(item)
        interaction = torch.cat((user_embed_ra, item_embed_ra), -1) # 256*2, 512
        uira_emb = self.linear1(interaction) # 256, 512

        if mode == "Train":
            seq_pre = True
        else:
            seq_pre = False

        ra_c = torch.matmul(uira_emb, self.W_1)
        ra_s = uira_emb - ra_c  # 256.512
        ui_ra_spa_emb = uira_emb
        rating_ln = self.mlp(ui_ra_spa_emb)

        encoder_distri, encoder_hidden, con_hidden = self.model(seq, uira_emb, seq_prediction=seq_pre)

        uire_emb = encoder_hidden #+ ra_s.unsqueeze(0).repeat((encoder_hidden.shape[0], 1, 1))
         # 256.512
        re_c = torch.matmul(uire_emb, self.W_2)  # 16.256.512
        re_s = uire_emb - re_c  # 16.256.512
        remlp = self.remlp(re_s.mean(0))

        de_ra = self.de_ra(uira_emb.unsqueeze(2)).permute(2, 0, 1)  # 256.512.1
        de_ra_c = self.de_ra(ra_c.unsqueeze(2)).permute(2, 0, 1)  # 256.512.1
        de_ra_s = self.de_ra(ra_s.unsqueeze(2)).permute(2, 0, 1)  # 256.512.1

        de_re = self.de_re(uire_emb.mean(0).squeeze())  # 256.512
        de_re_c = self.de_re(re_c.mean(0).squeeze())  # 256.512
        de_re_s = self.de_re(re_s.mean(0).squeeze())  # 256.512

        ui_re_spa_emb = uire_emb + self.lamda * ra_s.unsqueeze(0).repeat((encoder_hidden.shape[0], 1, 1))
        # ui_re_spa_emb = F.normalize(ui_re_spa_emb)
        if mode == "Train":
            word_prob = self.hidden2token(ui_re_spa_emb)  # (tgt_len, batch_size, ntoken)
            ui_re_distri = F.log_softmax(word_prob, dim=-1)
        else:
            word_prob = self.hidden2token(ui_re_spa_emb[-1])  # (tgt_len, batch_size, ntoken)
            ui_re_distri = F.log_softmax(word_prob, dim=-1)



        return ra_c, ra_s, re_c, re_s, uira_emb, uire_emb, \
               de_ra, de_re, de_ra_c, de_ra_s, de_re, de_re_c, de_re_s, \
               rating_ln, ui_re_distri, con_hidden, remlp


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True):
    result = torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine).to('cuda')
    return result