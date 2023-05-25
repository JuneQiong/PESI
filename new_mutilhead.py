import math
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True):
        super(MyMultiheadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.kdim = self.head_dim
        self.vdim = self.head_dim
        self.num_heads = num_heads
        self.dropout = dropout
        assert self.head_dim * num_heads == self.embed_dim
        self.q_proj_weight = torch.empty(embed_dim, embed_dim, requires_grad=True, device=torch.device('cuda:0'))
        self.k_proj_weight = torch.empty(embed_dim, embed_dim, requires_grad=True, device=torch.device('cuda:0'))
        self.v_proj_weight = torch.empty(embed_dim, embed_dim, requires_grad=True, device=torch.device('cuda:0'))
        # self.q_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))  # embed_dim = kdim * num_heads
        # self.k_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))  # W_k,  embed_dim = kdim * num_heads
        # self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))  # W_v,  embed_dim = vdim * num_heads
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.qr = torch.empty(256 * self.num_heads, 17, requires_grad=True, device=torch.device('cuda:0'))
        self.kr = torch.empty(256 * self.num_heads, 17, requires_grad=True, device=torch.device('cuda:0'))
        self.vr = torch.empty(17, self.embed_dim // self.num_heads, requires_grad=True, device=torch.device('cuda:0'))

        self.init_weights()

    def init_weights(self):
        self.q_proj_weight = nn.init.normal_(self.q_proj_weight, mean=0.0, std=0.1)
        self.k_proj_weight = nn.init.normal_(self.k_proj_weight, mean=0.0, std=0.1)
        self.v_proj_weight = nn.init.normal_(self.v_proj_weight, mean=0.0, std=0.1)

        self.qr = nn.init.normal_(self.qr, mean=0.0, std=0.1)
        self.kr = nn.init.normal_(self.kr, mean=0.0, std=0.1)
        self.vr = nn.init.normal_(self.vr, mean=0.0, std=0.1)


    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):

        return multi_head_attention_forward(query, key, value, self.num_heads, self.qr, self.vr, self.kr,
                                            self.dropout, self.out_proj.weight, self.out_proj.bias,
                                            training=self.training,
                                            key_padding_mask=key_padding_mask,
                                            q_proj_weight=self.q_proj_weight,
                                            k_proj_weight=self.k_proj_weight,
                                            v_proj_weight=self.v_proj_weight,
                                            attn_mask=attn_mask)
def multi_head_attention_forward(
    query,  # [tgt_len,batch_size, embed_dim]
    key,  # [src_len, batch_size, embed_dim]
    value,  # [src_len, batch_size, embed_dim]
    num_heads,
    qr, vr, kr,
    dropout_p,
    out_proj_weight, # [embed_dim = vdim * num_heads, embed_dim]
    out_proj_bias,
    training=True,
    key_padding_mask=None,  # [batch_size,src_len/tgt_len]
    q_proj_weight=None,  # [embed_dim,kdim * num_heads]
    k_proj_weight=None,  # [embed_dim, kdim * num_heads]
    v_proj_weight=None,  # [embed_dim, vdim * num_heads]
    attn_mask=None,  # [tgt_len,src_len]
    ):



    q = F.linear(query, q_proj_weight)
    #  [tgt_len,batch_size,embed_dim] x [embed_dim,kdim * num_heads] = [tgt_len,batch_size,kdim * num_heads]
    k = F.linear(key, k_proj_weight)
    # [src_len, batch_size,embed_dim] x [embed_dim,kdim * num_heads] = [src_len,batch_size,kdim * num_heads]
    v = F.linear(value, v_proj_weight)
    # [src_len, batch_size,embed_dim] x [embed_dim,vdim * num_heads] = [src_len,batch_size,vdim * num_heads]


    tgt_len, bsz, embed_dim = query.size()  # [tgt_len,batch_size, embed_dim]


    src_len = key.size(0)
    head_dim = embed_dim // num_heads  # num_heads * head_dim = embed_dim
    scaling = float(head_dim) ** -0.5
    q = q * scaling  # [query_len,batch_size,kdim * num_heads]

    if attn_mask is not None:  # [tgt_len,src_len] or [num_heads*batch_size,tgt_len, src_len]
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)  # [1, tgt_len,src_len]
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 2D attn_mask is not correct.')
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 3D attn_mask is not correct.')

    ui = query[0, :, :]  # [256, 512]
    ui = ui.contiguous().view(bsz * num_heads, head_dim).transpose(0, 1)  # [64,2048]
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1) # [2048, 17, 64]
    # [batch_size * num_heads,tgt_len,kdim]
    pdb.set_trace()
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)  # [batch_size * num_heads,src_len,kdim]
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)  # [batch_size * num_heads,src_len,vdim]
    qr = torch.matmul(ui, qr) #[64,2048] * [2048, 17] =  64*17
    kr = torch.matmul(ui, kr)
    #
    # weight = torch.ones(tgt_len, 1).to("cuda")
    # weight[0] = 1.5

    # Qr = torch.matmul(q, qr[:, 0:tgt_len])
    # Kr = torch.matmul(k, kr[:, 0:tgt_len])
    # QK = torch.matmul(torch.mul(q, weight), torch.mul(k, weight).transpose(1, 2))
    pdb.set_trace()
    q_r = torch.matmul(q, qr[:, 0:tgt_len]) # [[2048, 17, 64] * [64*17] = [2048, 17, 17]
    k_r = torch.matmul(k, kr[:, 0:tgt_len]) #
    attn_output_weights = (torch.bmm(q, k.transpose(1, 2)) + q_r + k_r)
    bn = nn.BatchNorm1d(tgt_len).to('cuda')

    attn_output_weights = bn(attn_output_weights)
    # bn(relu(attn_output_weights[0]))
    # print(attn_output_weights[0][0])
    # print(attn_output_weights[0][:, 0])
    # attn_output_weights = (0.1 * Qr + 0.1 * Kr + QK)
    # [batch_size * num_heads,tgt_len,kdim] x [batch_size * num_heads, kdim, src_len]
    if attn_mask is not None:
        attn_output_weights += attn_mask  # [batch_size * num_heads, tgt_len, src_len]
    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        #  [batch_size, num_heads, tgt_len, src_len]
        attn_output_weights = attn_output_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        # [batch_size,src_len]---[batch_size,1,1,src_len]
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)
        # [batch_size * num_heads, tgt_len, src_len]
    attn_output_weights = F.softmax(attn_output_weights, dim=-1)  # [batch_size * num_heads, tgt_len, src_len]
    # attn_output_weights = attn_output_weights.view(-1, bsz, tgt_len, src_len)
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights,  v + vr[0:tgt_len, :])
    # Z = [batch_size * num_heads, tgt_len, src_len]  x  [batch_size * num_heads,src_len,vdim]
    # = # [batch_size * num_heads,tgt_len,vdim]
    # num_heads  Attention(Q,K,V)

    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    # transpose [tgt_len, batch_size* num_heads ,kdim]
    # view [tgt_len,batch_size,num_heads*kdim]
    attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
    Z = F.linear(attn_output, out_proj_weight, out_proj_bias)
    #  [tgt_len,batch_size,embed_dim]
    return Z, (attn_output_weights.sum(dim=1) / num_heads)