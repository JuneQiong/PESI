import os
import math
import pdb
import numpy as np
import torch
import heapq
import random
import pickle
import datetime
from rouge import rouge
from bleu import compute_bleu
from textblob import TextBlob


# def different(references, generated):
#     real_senti = []
#     fake_senti = []
#     true_num = 0
#     for (real, fake) in zip(references, generated):
#         if len(real) > 2:
#             blob = TextBlob(real).sentences[0].sentiment.polarity
#         else:
#             blob = 0
#         if blob < -0.0001:
#             real_blob_senti = 0
#         elif blob >= -0.0001 and blob <= 0.0001:
#             real_blob_senti = 1
#         else:
#             real_blob_senti = 2
#         real_senti.append(real_blob_senti)
#
#         if len(fake) > 2:
#             blob = TextBlob(fake).sentences[0].sentiment.polarity
#         else:
#             blob = 0
#         if blob < -0.0001:
#             fake_blob_senti = 0
#         elif blob >= -0.0001 and blob <= 0.0001:
#             fake_blob_senti = 1
#         else:
#             fake_blob_senti = 2
#         fake_senti.append(fake_blob_senti)
#
#     pdb.set_trace()
#     from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score
#     import sklearn.utils.class_weight
#     mf1 = f1_score(real_senti, fake_senti, average='macro')
#     mf2 = f1_score(real_senti, fake_senti, average='micro')
#     f1_score(real_senti, fake_senti, labels=[0, 1, 2], average='weighted')
#     accuracy_score(real_senti, fake_senti, normalize=True, sample_weight=None)
#     precision_score(real_senti, fake_senti, average="weighted")
#     from metric import cal_precision_and_recall
#     cal_precision_and_recall(real_senti, fake_senti)
#     rouge(real_senti, fake_senti)

def rouge_score(references, generated):
    """both are a list of strings"""
    score = rouge(generated, references)
    rouge_s = {k: (v * 100) for (k, v) in score.items()}
    '''
    "rouge_1/f_score": rouge_1_f,
    "rouge_1/r_score": rouge_1_r,
    "rouge_1/p_score": rouge_1_p,
    "rouge_2/f_score": rouge_2_f,
    "rouge_2/r_score": rouge_2_r,
    "rouge_2/p_score": rouge_2_p,
    "rouge_l/f_score": rouge_l_f,
    "rouge_l/r_score": rouge_l_r,
    "rouge_l/p_score": rouge_l_p,
    '''
    return rouge_s


def bleu_score(references, generated, n_gram=4, smooth=False):
    """a list of lists of tokens"""
    formatted_ref = [[ref] for ref in references]
    bleu_s, _, _, _, _, _ = compute_bleu(formatted_ref, generated, n_gram, smooth)
    return bleu_s * 100


def two_seq_same(sa, sb):
    if len(sa) != len(sb):
        return False
    for (wa, wb) in zip(sa, sb):
        if wa != wb:
            return False
    return True


def unique_sentence_percent(sequence_batch):
    unique_seq = []
    for seq in sequence_batch:
        count = 0
        for uni_seq in unique_seq:
            if two_seq_same(seq, uni_seq):
                count += 1
                break
        if count == 0:
            unique_seq.append(seq)

    return len(unique_seq) / len(sequence_batch), len(unique_seq)


def feature_detect(seq_batch, feature_set):
    feature_batch = []
    for ids in seq_batch:
        feature_list = []
        for i in ids:
            if i in feature_set:
                feature_list.append(i)
        feature_batch.append(set(feature_list))

    return feature_batch


def feature_matching_ratio(feature_batch, test_feature):
    count = 0
    for (fea_set, fea) in zip(feature_batch, test_feature):
        if fea in fea_set:
            count += 1

    return count / len(feature_batch)


def feature_coverage_ratio(feature_batch, feature_set):
    features = set()
    for fb in feature_batch:
        features = features | fb

    return len(features) / len(feature_set)


def feature_diversity(feature_batch):
    list_len = len(feature_batch)

    total_count = 0
    for i, x in enumerate(feature_batch):
        for j in range(i + 1, list_len):
            y = feature_batch[j]
            total_count += len(x & y)

    denominator = list_len * (list_len - 1) / 2
    return total_count / denominator


def mean_absolute_error(predicted, max_r, min_r, mae=True):
    total = 0
    for (r, p) in predicted:
        if p > max_r:
            p = max_r
        if p < min_r:
            p = min_r

        sub = p - r
        if mae:
            total += abs(sub)
        else:
            total += sub ** 2

    return total / len(predicted)


def root_mean_square_error(predicted, max_r, min_r):
    mse = mean_absolute_error(predicted, max_r, min_r, False)
    return math.sqrt(mse)


class WordDictionary:
    def __init__(self):
        self.idx2word = ['<bos>', '<eos>', '<pad>', '<unk>']
        self.__predefine_num = len(self.idx2word)
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}
        self.__word2count = {}

    def add_sentence(self, sentence):
        for w in sentence.split():
            self.add_word(w)

    def add_word(self, w):
        if w not in self.word2idx:
            self.word2idx[w] = len(self.idx2word)
            self.idx2word.append(w)
            self.__word2count[w] = 1
        else:
            self.__word2count[w] += 1

    def __len__(self):
        return len(self.idx2word)

    def keep_most_frequent(self, max_vocab_size=20000):
        if len(self.__word2count) > max_vocab_size:
            frequent_words = heapq.nlargest(max_vocab_size, self.__word2count, key=self.__word2count.get)
            self.idx2word = self.idx2word[:self.__predefine_num] + frequent_words
            self.word2idx = {w: i for i, w in enumerate(self.idx2word)}


class EntityDictionary:
    def __init__(self):
        self.idx2entity = []
        self.entity2idx = {}

    def add_entity(self, e):
        if e not in self.entity2idx:
            self.entity2idx[e] = len(self.idx2entity)
            self.idx2entity.append(e)

    def __len__(self):
        return len(self.idx2entity)


def func(n, end, start = 1):
    list(range(start, n)) + list(range(n+1, end))
    return list(range(start, n)) + list(range(n+1, end))

class DataLoader:
    def __init__(self, data_path, index_dir, vocab_size):
        self.word_dict = WordDictionary()
        self.user_dict = EntityDictionary()
        self.item_dict = EntityDictionary()
        self.max_rating = float('-inf')
        self.min_rating = float('inf')
        self.initialize(data_path)
        self.word_dict.keep_most_frequent(vocab_size)
        self.__unk = self.word_dict.word2idx['<unk>']
        self.feature_set = set()
        self.train, self.valid, self.test = self.load_data(data_path, index_dir)

    def initialize(self, data_path):
        assert os.path.exists(data_path)
        reviews = pickle.load(open(data_path, 'rb'))
        for review in reviews:
            self.user_dict.add_entity(review['user'])
            self.item_dict.add_entity(review['item'])
            (fea, adj, tem, sco) = review['template']
            self.word_dict.add_sentence(tem)
            self.word_dict.add_word(fea)
            rating = review['rating']
            if self.max_rating < rating:
                self.max_rating = rating
            if self.min_rating > rating:
                self.min_rating = rating

    def load_data(self, data_path, index_dir):
        data_raw = []
        reviews = pickle.load(open(data_path, 'rb'))
        # amamov: actors, story, charactor
        # yelp: service prices atmosphere
        # trip: staff  service location
        # amoclo: quality  price  color wear  material
        for review in reviews:
            (fea, adj, tem, sco) = review['template']
            data_raw.append({'user': self.user_dict.entity2idx[review['user']],
                             'item': self.item_dict.entity2idx[review['item']],
                             'rating': review['rating'],
                             'text': self.seq2ids(tem),
                             'text_word': tem,
                             'feature': self.word_dict.word2idx.get(fea, self.__unk),
                             # 'feature2': self.word_dict.word2idx.get("service", self.__unk),
                             # 'feature3': self.word_dict.word2idx.get("location", self.__unk),

                             # 'feature4': self.word_dict.word2idx.get("director", self.__unk),
                             # 'feature5': self.word_dict.word2idx.get("performance", self.__unk),
                             })
            if fea in self.word_dict.word2idx:
                self.feature_set.add(fea)
            else:
                self.feature_set.add('<unk>')
        train_index, valid_index, test_index = self.load_index(index_dir)
        #
        data_new = []
        data_init = []
        pos_num = 0
        neg_num = 0
        neu_num = 0
        for item in data_raw:
            item.update({'label': 1})

            if item['rating'] > 3:
                item.update({'r_label': 6})
            elif item['rating'] == 3:
                item.update({'r_label': 3})
            else:
                item.update({'r_label': 0})
            item.update({'fake_r': item['r_label']})

            # if len(item['text_word']) > 2:
            #     blob = TextBlob(item['text_word']).sentences[0].sentiment.polarity
            # else:
            #     blob = 0
            # if blob < -0.0001:
            #     blob_senti = 0
            # elif blob >= -0.0001 and blob <= 0.0001:
            #     blob_senti = 1
            # else:
            #     blob_senti = 2
            # item.update({'textblob': blob_senti})

            # itemnew = item.copy()
            # listr = [0, 3, 6]
            # listr.remove(int(item['r_label']))
            #
            # rand_rating = np.random.choice(listr)
            # r = func(int(item['r_label']), 3)
            # rand_rating = np.random.choice(r)
            # itemnew.update({'fake_r': rand_rating})
            # itemnew.update({'label': 0})

            # data_new.append(itemnew)
            data_init.append(item)
        # fw = open("trip_textbolb.pickle", "wb")
        # pickle.dump(data_init, fw)
        # data_init = pickle.load(open(r"E:\keyan\C-PETER\PESI-distangle-motsf\trip_textbolb.pickle", 'rb'))
        train, valid, test = [], [], []
        for idx in train_index:
            train.append(data_init[idx])
            # train.append(data_new[idx])
        for idx in valid_index:
            valid.append(data_init[idx])
            # valid.append(data_new[idx])
        for idx in test_index:
            test.append(data_init[idx])
            # test.append(data_new[idx])

        # train = sorted(train, key=lambda e: e.__getitem__('item'))
        random.shuffle(train)
        random.shuffle(valid)
        test = sorted(test, key=lambda e: e.__getitem__('rating'))
        return train, valid, test

    def seq2ids(self, seq):
        return [self.word_dict.word2idx.get(w, self.__unk) for w in seq.split()]

    def load_index(self, index_dir):
        assert os.path.exists(index_dir)
        with open(os.path.join(index_dir, 'train.index'), 'r') as f:
            train_index = [int(x) for x in f.readline().split(' ')]
        with open(os.path.join(index_dir, 'validation.index'), 'r') as f:
            valid_index = [int(x) for x in f.readline().split(' ')]
        with open(os.path.join(index_dir, 'test.index'), 'r') as f:
            test_index = [int(x) for x in f.readline().split(' ')]
        return train_index, valid_index, test_index


def sentence_format(sentence, max_len, bos, eos, pad):
    length = len(sentence)
    if length >= max_len:
        return [bos] + sentence[:max_len] + [eos]
    else:
        return [bos] + sentence + [eos] + [pad] * (max_len - length)


class Batchify:
    def __init__(self, data, word2idx, seq_len=15, batch_size=128, shuffle=False):
        bos = word2idx['<bos>']
        eos = word2idx['<eos>']
        pad = word2idx['<pad>']
        u, i, r, t, l, fr, f = [], [], [], [], [], [], []
        for x in data:
            u.append(x['user'])
            i.append(x['item'])
            r.append(x['rating'])
            t.append(sentence_format(x['text'], seq_len, bos, eos, pad))
            f.append([x['feature']])
            # f2.append([x['feature2']])
            # f3.append([x['feature3']])
            # f4.append([x['feature4']])
            # f5.append([x['feature5']])
            l.append(x['rating'])
            fr.append(x['fake_r'])

        self.user = torch.tensor(u, dtype=torch.int64).contiguous()
        self.item = torch.tensor(i, dtype=torch.int64).contiguous()
        self.rating = torch.tensor(r, dtype=torch.float).contiguous()
        self.seq = torch.tensor(t, dtype=torch.int64).contiguous()
        self.feature = torch.tensor(f, dtype=torch.int64).contiguous()
        # self.feature2 = torch.tensor(f2, dtype=torch.int64).contiguous()
        # self.feature3 = torch.tensor(f3, dtype=torch.int64).contiguous()
        # self.feature4 = torch.tensor(f4, dtype=torch.int64).contiguous()
        # self.feature5 = torch.tensor(f5, dtype=torch.int64).contiguous()

        self.label = torch.tensor(l, dtype=torch.float).contiguous()
        self.fake_rating = torch.tensor(fr, dtype=torch.int64).contiguous()
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.sample_num = len(data)
        self.index_list = list(range(self.sample_num))
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0

    def next_batch(self):
        if self.step == self.total_step:
            self.step = 0
            if self.shuffle:
                random.shuffle(self.index_list)
        start = self.step * self.batch_size
        offset = start + self.batch_size
        if offset > self.sample_num:
            self.step += 1
            index = self.index_list[start:offset]
            index1 = self.index_list[0:offset - self.sample_num]
            index[len(index):len(index1)] = index1
            user = self.user[index]  # (batch_size,)
            item = self.item[index]
            rating = self.rating[index]
            seq = self.seq[index]  # (batch_size, seq_len)
            feature = self.feature[index]  # (batch_size, 1)
            # feature2 = self.feature2[index]
            # feature3 = self.feature3[index]
            # feature4 = self.feature4[index]
            # feature5 = self.feature5[index]
            label = self.label[index]
            fake_r = self.fake_rating[index]
        else:
            self.step += 1
            index = self.index_list[start:offset]
            user = self.user[index]  # (batch_size,)
            item = self.item[index]
            rating = self.rating[index]
            seq = self.seq[index]  # (batch_size, seq_len)
            feature = self.feature[index]  # (batch_size, 1)
            # feature2 = self.feature2[index]
            # feature3 = self.feature3[index]
            # feature4 = self.feature4[index]
            # feature5 = self.feature5[index]
            label = self.label[index]
            fake_r = self.fake_rating[index]
        return user, item, rating, seq, label, fake_r, feature

def now_time():
    return '[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + ']: '


def ids2tokens(ids, word2idx, idx2word):
    eos = word2idx['<eos>']
    tokens = []
    for i in ids:
        if i == eos:
            break
        tokens.append(idx2word[i])
    return tokens