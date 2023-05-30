import math
import copy
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as func
from typing import Tuple, Optional
from torch import Tensor
import torch.nn.functional as F
# from new_mutilhead import MyMultiheadAttention

class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.self_attn = MyMultiheadAttention(d_model, nhead)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = func.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> \
    Tuple[Tensor, Tensor]:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2, attn = self.self_attn(src, src, src, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)
        # src2, attn = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn


class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> \
    Tuple[Tensor, Tensor]:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src
        attns = []

        for mod in self.layers:
            output, attn = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            attns.append(attn)
        attns = torch.stack(attns)

        if self.norm is not None:
            output = self.norm(output)

        return output, attns


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
        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return func.relu
    elif activation == "gelu":
        return func.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # d_model: word embedding size
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len,) -> (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model/2,)
        '''
        probably to prevent from rounding error
        e^(idx * (-log 10000 / d_model)) -> (e^(log 10000))^(- idx / d_model) -> 10000^(- idx / d_model) -> 1/(10000^(idx / d_model))
        since idx is an even number, it is equal to that in the formula
        '''
        pe[:, 0::2] = torch.sin(position * div_term)  # even number index, (max_len, d_model/2)
        pe[:, 1::2] = torch.cos(position * div_term)  # odd number index
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, d_model) -> (1, max_len, d_model) -> (max_len, 1, d_model)
        self.register_buffer('pe', pe)  # will not be updated by back-propagation, can be called via its name

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MLP(nn.Module):
    def __init__(self, emsize=512):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(emsize, emsize)
        self.linear2 = nn.Linear(emsize, 1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU();

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.linear1.weight.data.uniform_(-initrange, initrange)
        self.linear2.weight.data.uniform_(-initrange, initrange)
        self.linear1.bias.data.zero_()
        self.linear2.bias.data.zero_()

    def forward(self, hidden):  # (batch_size, emsize)
        mlp_vector = self.relu(self.linear1(hidden))  # (batch_size, emsize)
        rating = self.sigmoid(self.linear2(mlp_vector));
        rating = torch.squeeze(rating)
        return rating


def generate_square_subsequent_mask(total_len):
    mask = torch.triu(torch.full((total_len, total_len), float('-inf'), device='cuda'), diagonal=1)
    # mask = torch.tril(torch.ones(total_len, total_len))  # (total_len, total_len), lower triangle -> 1.; others 0.
    # mask = mask == 0  # lower -> False; others True
    return mask


def generate_peter_mask(src_len, tgt_len):
    total_len = src_len + tgt_len
    mask = generate_square_subsequent_mask(total_len)
    mask[0, 1] = False  # allow to attend for user and item
    return mask

def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True):
    if torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class PETER(nn.Module):
    def __init__(self, peter_mask, src_len, tgt_len, pad_idx, nuser, nitem, ntoken, emsize, nhead, nhid, nlayers,
                 dropout=0.5):
        super(PETER, self).__init__()
        self.pos_encoder = PositionalEncoding(emsize, dropout)  # emsize: word embedding size
        encoder_layers = TransformerEncoderLayer(emsize, nhead, nhid,
                                                 dropout)  # nhid: dim_feedforward, one basic layer, including multi-head attention and FFN
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)  # loop over the one above
        # self.user_embeddings = nn.Embedding(nuser, emsize)
        # self.item_embeddings = nn.Embedding(nitem, emsize)
        self.word_embeddings = nn.Embedding(ntoken, emsize)
        self.hidden2token = nn.Linear(emsize, ntoken)
        self.GELU = nn.GELU().to('cuda')
        self.attn = Attention(emsize).to('cuda')
        # mbart has one extra layer_norm
        self.layer_norm = LayerNorm(emsize)
        self.ui_len = 2
        self.src_len = src_len
        self.pad_idx = pad_idx
        self.emsize = emsize
        if peter_mask:
            self.attn_mask = generate_peter_mask(src_len, tgt_len+1)
        else:
            self.attn_mask = generate_square_subsequent_mask(src_len + tgt_len + 1)

        # self.attn_mask = generate_square_subsequent_mask(src_len + tgt_len + 1)
        self.self_attn = nn.MultiheadAttention(emsize, 1, dropout=dropout)
        self.linear3 = nn.Linear(emsize * 3, emsize).to('cuda')
        self.init_weights()


    def init_weights(self):
        initrange = 0.1
        # self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        # self.item_embeddings.weight.data.uniform_(-initrange, initrange)
        self.word_embeddings.weight.data.normal_(mean=0.0, std=0.02)
        self.hidden2token.weight.data.normal_(mean=0.0, std=0.02)
        self.hidden2token.bias.data.normal_(mean=0.0, std=0.02)

    def predict_context(self, hidden):
        context_prob = self.hidden2token(hidden[1])  # (batch_size, ntoken)   hidden torch.Size([18, 128, 512])
        log_context_dis = func.log_softmax(context_prob, dim=-1)
        return log_context_dis

    def predict_rating(self, hidden):
        rating = self.recommender(hidden[0])  # (batch_size,)
        return rating

    def predict_seq(self, hidden):
        word_prob = self.hidden2token(hidden[self.src_len + 1:])  # (tgt_len, batch_size, ntoken)
        log_word_prob = func.log_softmax(word_prob, dim=-1)
        return log_word_prob

    def generate_token(self, hidden):
        word_prob = self.hidden2token(hidden[-1])  # (batch_size, ntoken)
        log_word_prob = func.log_softmax(word_prob, dim=-1)
        return log_word_prob

    def forward(self, text, ui_embedding, seq_prediction=True):
        '''
        :param user: (batch_size,), torch.int64
        :param item: (batch_size,), torch.int64
        :param text: (total_len - ui_len, batch_size), torch.int64
        :param seq_prediction: bool
        :param context_prediction: bool
        :param rating_prediction: bool
        :return log_word_prob: target tokens (tgt_len, batch_size, ntoken) if seq_prediction=True; the last token (batch_size, ntoken) otherwise.
        :return log_context_dis: (batch_size, ntoken) if context_prediction=True; None otherwise.
        :return rating: (batch_size,) if rating_prediction=True; None otherwise.
        :return attns: (nlayers, batch_size, total_len, total_len)
        '''
        device = text.device
        batch_size = text.size(1)
        total_len = text.size(0)  # deal with generation when total_len != src_len + tgt_len
        # see nn.MultiheadAttention for attn_mask and key_padding_mask
        attn_mask = self.attn_mask[:total_len+1, :total_len+1].to(device)  # (total_len, total_len)
        left = torch.zeros(batch_size, 1).bool().to(device)  # (batch_size, ui_len)
        right = text.t() == self.pad_idx  # replace pad_idx with True and others with False, (batch_size, total_len - ui_len)
        key_padding_mask = torch.cat([left, right], 1)  # (batch_size, total_len)

        w_src = self.word_embeddings(text)  # (total_len - ui_len, batch_size, emsize) 16,64,512
        src = w_src  # (total_len, batch_size, emsize)
        # src = src + ui_embedding.unsqueeze(0).repeat((total_len, 1, 1))
        # ui_embedding.unsqueeze(0).repeat((total_len, 1, 1))

        # attention ui and src
        # st = self.attn(src, ui_embedding.unsqueeze(0), ui_embedding.unsqueeze(0))
        # hu = self.layer_norm(self.GELU(src + st))

        # concat ui and src   for generate explaniation]-
        st = torch.cat([ui_embedding.unsqueeze(0), src], dim=0)

        hu = st * math.sqrt(self.emsize)
        hu = self.pos_encoder(hu)
        # 18,64,512,    18,18,   ,64,18
        hidden, attns = self.transformer_encoder(hu, attn_mask, key_padding_mask)  # (total_len, batch_size, emsize) vs. (nlayers, batch_size, total_len_tgt, total_len_src)
        if seq_prediction:
            # hidden, attns = self.transformer_encoder(hu, src_key_padding_mask=key_padding_mask)
            # hidden, attns = self.transformer_encoder(hu, attn_mask, key_padding_mask)
            log_word_prob = self.predict_seq(hidden)  # (tgt_len, batch_size, ntoken)
        else:
            # hidden, attns = self.transformer_encoder(hu, attn_mask, key_padding_mask)
            log_word_prob = self.generate_token(hidden)  # (batch_size, ntoken)
        # get contrastive loss hidden vector
        hidden[self.src_len + 1:].  transpose(0, 1)
        dim0, dim2 = (hidden[self.src_len + 1:].mean(0)).shape
        feature_con = (hidden[self.src_len + 1:].mean(0)).view(dim0, 1, dim2)

        x_normalized = F.normalize(feature_con, dim=-1)
        # hidden = hidden + ui_embedding.unsqueeze(0).repeat((hidden.shape[0], 1, 1))

        return log_word_prob, hidden[self.src_len + 1:], x_normalized
