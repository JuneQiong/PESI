import os
import math
import pdb

import torch
import argparse
import torch.nn as nn
from Logger import create_log_dir
from configuration_3 import parse
from SupConLoss import SupConLoss
from Contrastive_loss import *
import time
import torch
import random
import numpy as np
from auxiliary_3 import Model
from FacalLoss import FocalLoss
from util_init import rouge_score, bleu_score, DataLoader, Batchify, now_time, ids2tokens, unique_sentence_percent, \
    feature_detect, feature_matching_ratio, feature_coverage_ratio, feature_diversity, \
    root_mean_square_error, mean_absolute_error
from metric import cal_precision_and_recall
import torch.nn.functional as F
args = parse()
args.lr = 1
args.rat_lr = 0.1
args.ll_lr = 1e-6
args.batch_size = 256
args.gamma = 0.1
# args.checkpoint = 'mov_6ly_01_0154l_3x/'
# args.outf = 'mov_6ly_01_0154l_3x'
args.use_feature =False
args.seed = 2021

day_now = time.strftime("%Y%m%d", time.localtime())
date_now = time.strftime("%H%M%S", time.localtime())
log_path = './{}_log/'.format(day_now)
logger = create_log_dir(log_path, '{}_{}_log.txt'.format(date_now, args.outf))
logger.info("model init: src=wrc.   src = src + ui_embedding.unsqueeze(0).repeat((total_len, 1, 1))  uire_emb = encoder_hidden")
args.outf = date_now + args.outf
args.checkpoint = log_path + args.checkpoint

# def seed_torch(seed=args.seed):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)  # In order to prohibit hash randomization, the experiment can be reproduced
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.enabled = True
# seed_torch()

if args.data_path is None:
    logger.error('--data_path should be provided for loading data')
if args.index_dir is None:
    logger.error('--index_dir should be provided for loading data splits')

logger.info('-' * 40 + 'ARGUMENTS' + '-' * 40)
for arg in vars(args):
    logger.info('{:40} {}'.format(arg, getattr(args, arg)))
logger.info('-' * 40 + 'ARGUMENTS' + '-' * 40)

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        logger.info(now_time() + 'WARNING: You have a CUDA device, so you should probably run with --cuda')
device = torch.device('cuda' if args.cuda else 'cpu')

if not os.path.exists(args.checkpoint):
    os.makedirs(args.checkpoint)
model_path = os.path.join(args.checkpoint, '{}.pt'.format(args.outf))
prediction_path = os.path.join(args.checkpoint, args.outf)

###############################################################################
# Load data
###############################################################################

logger.info(now_time() + 'Loading data')
corpus = DataLoader(args.data_path, args.index_dir, args.vocab_size)
word2idx = corpus.word_dict.word2idx
idx2word = corpus.word_dict.idx2word
feature_set = corpus.feature_set
train_data = Batchify(corpus.train, word2idx, args.words, args.batch_size, shuffle=True)
val_data = Batchify(corpus.valid, word2idx, args.words, args.batch_size)
test_data = Batchify(corpus.test, word2idx, args.words, args.batch_size)
logger.info(now_time() + 'Loading data success')
###############################################################################
# Build the model
###############################################################################

if args.use_feature:
    src_len = train_data.feature.size(1)
else:
    src_len = 0  # [u, i]
tgt_len = args.words + 1  # added <bos> or <eos>
ntokens = len(corpus.word_dict)
nuser = len(corpus.user_dict)
nitem = len(corpus.item_dict)
pad_idx = word2idx['<pad>']
model = Model(src_len, tgt_len, pad_idx, nuser, nitem, ntokens).to(device)

# definition loss
# neg71501 - neu96935 - pos273347

neg_num = 1/71501
neu_num = 1/96935
pos_num = 1/273347

text_criterion = nn.NLLLoss(ignore_index=pad_idx).to(device)  # ignore the padding when computing loss
# rev_criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([neg_num, neu_num, pos_num])).float()).to(device)
# rev_criterion = FocalLoss(class_num=3, gamma=2)
# rev_criterion = nn.CrossEntropyLoss().to(device)
Loss_MSE = torch.nn.MSELoss(reduction='mean').to(device)
rating_criterion = nn.MSELoss().to(device)
# rav_criterion = nn.MSELoss().to(device)
# con_criterion = SupConLoss(temperature=args.temp).to(device)
selfcon_criterion = Contrastive_loss().to(device)

opt_text = torch.optim.SGD([p for p in model.parameters()], lr=args.lr, weight_decay=1e-4)
sch_text = torch.optim.lr_scheduler.StepLR(opt_text, 3, gamma=args.gamma)
# opt_con = torch.optim.SGD([p for p in model.parameters()], lr=args.con_lr, weight_decay=1e-4)
# sch_con = torch.optim.lr_scheduler.StepLR(opt_con, 3, gamma=0.25)
opt_rat = torch.optim.SGD([p for p in model.parameters()], lr=args.rat_lr, weight_decay=1e-4)
sch_rat = torch.optim.lr_scheduler.StepLR(opt_rat, 3, gamma=0.1)
opt_ll = torch.optim.SGD([p for p in model.parameters()], lr=args.ll_lr, weight_decay=1e-4)
sch_ll = torch.optim.lr_scheduler.StepLR(opt_ll, 3, gamma=0.25)

###############################################################################
# Training code
###############################################################################


def predict(log_context_dis, topk):
    word_prob = log_context_dis.exp()  # (batch_size, ntoken)
    if topk == 1:
        context = torch.argmax(word_prob, dim=1, keepdim=True)  # (batch_size, 1)
    else:
        context = torch.topk(word_prob, topk, 1)[1]  # (batch_size, topk)
    return context  # (batch_size, topk)

def decoder_loss_function(img_rep, de_txt, de_txt_c, de_txt_s, t):
    img = F.normalize(img_rep, dim=1)
    txt = F.normalize(de_txt, dim=1)
    txt_c = F.normalize(de_txt_c, dim=1)
    txt_s = F.normalize(de_txt_s, dim=1)
    pos_1 = torch.sum(img * txt_c, dim=1)
    pos_2 = torch.sum(img * txt, dim=1)
    neg_1 = torch.sum(img * txt_s, dim=1)
    pos_1_h = torch.exp(pos_1 / t)
    pos_2_h = torch.exp(pos_2 / t)
    neg_1_h = torch.exp(neg_1 / t)
    loss_1 = -torch.mean(torch.log(pos_1_h/(pos_1_h + pos_2_h + neg_1_h) + 1e-24))
    loss_2 = -torch.mean(torch.log(pos_2_h/(pos_2_h + neg_1_h) + 1e-24))
    return loss_1 + loss_2

def train(data, flag):
    # Turn on training mode which enables dropout.
    model.train()

    text_loss = 0.
    rating_loss = 0.
    loss_loss = 0.
    review_loss = 0.
    self_con_loss = 0.
    total_sample = 0
    rating_predict = []
    rating_data = []
    while True:
        user, item, rating, seq, label, fake_r, feature = data.next_batch()  # (batch_size, seq_len), data.step += 1
        batch_size = user.size(0)
        user = user.to(device)  # (batch_size,)
        item = item.to(device)
        rating = rating.to(device)
        label = label.to(device)
        fake_r = fake_r.to(device)
        feature = feature.t().to(device)
        seq = seq.t().to(device)  # (tgt_len + 1, batch_size)


        if args.use_feature:
            text = torch.cat([feature, seq[:-1]], 0)  # (src_len + tgt_len - 2, batch_size)
        else:
            text = seq[:-1]  # (src_len + tgt_len - 2, batch_size)

        # text
        opt_text.zero_grad()
        ra_c, ra_s, re_c, re_s, uira_emb, uire_emb, \
        de_ra, de_re, de_ra_c, de_ra_s, de_re, de_re_c, de_re_s, \
        rating_ln, ui_re_distri, con_hidden, remlp = model(user, item, rating, fake_r, text,
                                                                            mode="Train")
        t_loss = text_criterion(ui_re_distri.view(-1, ntokens), seq[1:].reshape((-1,)))
        t_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt_text.step()

        opt_rat.zero_grad()
        ra_c, ra_s, re_c, re_s, uira_emb, uire_emb, \
        de_ra, de_re, de_ra_c, de_ra_s, de_re, de_re_c, de_re_s, \
        rating_ln, ui_re_distri, con_hidden, remlp = model(user, item, rating, fake_r, text,
                                                    mode="Train")
        r_loss = rating_criterion(rating_ln, rating)
        r_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt_rat.step()

        opt_ll.zero_grad()
        ra_c, ra_s, re_c, re_s, uira_emb, uire_emb, \
        de_ra, de_re, de_ra_c, de_ra_s, de_re, de_re_c, de_re_s, \
        rating_ln, ui_re_distri, con_hidden, remlp = model(user, item, rating, fake_r, text,
                                                           mode="Train")
        L_sim = Loss_MSE(ra_c, re_c.mean(0).squeeze())
        L_ort = Loss_MSE(ra_s, re_s.mean(0).squeeze())
        #
        decoder_loss_T2V = decoder_loss_function(uira_emb, de_re, de_re_c, de_re_s, 0.5)
        decoder_loss_V2T = decoder_loss_function(uire_emb, de_ra, de_ra_c, de_ra_s, 0.5)
        #
        l_loss = L_sim-L_ort + decoder_loss_T2V + decoder_loss_V2T

        rls_loss = l_loss
        rls_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt_ll.step()

        rating_loss += batch_size * r_loss.item()
        loss_loss += batch_size * l_loss.item()
        text_loss += batch_size * t_loss.item()
        # r_loss, l_loss, t_loss, self_loss
        total_sample += batch_size

        if data.step % args.log_interval == 0 or data.step == data.total_step:
            cur_t_loss = text_loss / total_sample
            cur_r_loss = rating_loss / total_sample
            cur_l_loss = loss_loss / total_sample
            cur_sc_loss = self_con_loss / total_sample
            con_rv_loss = review_loss/ total_sample
            logger.info(
                now_time() + 'text loss {:4.4f} | rating loss {:4.4f} | lloss {:4.4f} | rv loss {:4.4f} |sc loss {:4.4f} |{:5d} batches/{:5d} batches'.format(
                    cur_t_loss, cur_r_loss, cur_l_loss, con_rv_loss, cur_sc_loss, data.step,
                    data.total_step))
            text_loss = 0.
            rating_loss = 0.
            loss_loss = 0.
            self_con_loss = 0.
            total_sample = 0
            rating_predict.extend(rating_ln.t().squeeze().tolist())
            rating_data.extend(rating.float().tolist())
        if data.step == data.total_step:
            break
    # precision, recall, accuracy, f1_score = cal_precision_and_recall(rating_data, rating_predict)
    #
    # logger.info(
    #         now_time() + ' precision  {},recall  {}, accuracy  {}, f1_score   {} in valid stage'.format(round(precision, 4), round(recall, 4), round(accuracy, 4), round(f1_score, 4)))
    predicted_rating = [(r, p) for (r, p) in zip(rating_data, rating_predict)]
    MAE = mean_absolute_error(predicted_rating, corpus.max_rating, corpus.min_rating)
    RMSE = root_mean_square_error(predicted_rating, corpus.max_rating, corpus.min_rating)
    logger.info(now_time() + 'RMSE {:7.4f}'.format(RMSE) + 'MAE {:7.4f}'.format(MAE) + "in training")

    # logger.info(now_time() + 'MAE {:7.4f}'.format(MAE))

def evaluate(data, flag):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    text_loss = 0.
    rating_loss = 0.
    loss_loss = 0.
    self_con_loss = 0.
    total_sample = 0
    rating_predict = []
    rating_data = []
    review_loss = 0.
    idss_predict = []
    seq_data = []
    with torch.no_grad():
        while True:
            user, item, rating, seq, label, fake_r, feature = data.next_batch()  # (batch_size, seq_len), data.step += 1
            batch_size = user.size(0)
            user = user.to(device)  # (batch_size,)
            item = item.to(device)
            rating = rating.to(device)
            label = label.to(device)
            fake_r = fake_r.to(device)
            feature = feature.t().to(device)
            seq = seq.t().to(device)  # (tgt_len + 1, batch_size)

            if args.use_feature:
                text = torch.cat([feature, seq[:-1]], 0)  # (src_len + tgt_len - 2, batch_size)
            else:
                text = seq[:-1]  # (src_len + tgt_len - 2, batch_size)

            ra_c, ra_s, re_c, re_s, uira_emb, uire_emb, \
            de_ra, de_re, de_ra_c, de_ra_s, de_re, de_re_c, de_re_s, \
            rating_ln, ui_re_distri, con_hidden, remlp = model(user, item, rating, fake_r, text,
                                                        mode="Train")
            #
            t_loss = text_criterion(ui_re_distri.view(-1, ntokens), seq[1:].reshape((-1,)))
            L_sim = Loss_MSE(ra_c, re_c.mean(0).squeeze())

            L_ort = Loss_MSE(ra_s, re_s.mean(0).squeeze())

            decoder_loss_T2V = decoder_loss_function(uira_emb, de_re, de_re_c, de_re_s, 0.5)
            decoder_loss_V2T = decoder_loss_function(uire_emb, de_ra, de_ra_c, de_ra_s, 0.5)

            l_loss = L_sim-L_ort + decoder_loss_T2V + decoder_loss_V2T
            r_loss = rating_criterion(rating_ln, rating)

            rating_loss += batch_size * r_loss.item()
            loss_loss += batch_size * l_loss.item()
            text_loss += batch_size * t_loss.item()

            rating_predict.extend(rating_ln.t().squeeze().tolist())
            rating_data.extend(rating.float().tolist())

            total_sample += batch_size

            if data.step == data.total_step:
                break
        predicted_rating = [(r, p) for (r, p) in zip(rating_data, rating_predict)]
        MAE = mean_absolute_error(predicted_rating, corpus.max_rating, corpus.min_rating)
        RMSE = root_mean_square_error(predicted_rating, corpus.max_rating, corpus.min_rating)
        logger.info(now_time() + 'RMSE {:7.4f}'.format(RMSE) + 'MAE {:7.4f}'.format(MAE) + "in valid")
    return text_loss / total_sample, rating_loss / total_sample, \
           loss_loss / total_sample,  self_con_loss / total_sample, review_loss / total_sample

def generate(data):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    idss_predict = []
    seq_data = []
    rating_predict = []
    rating_data = []
    real_rating = []

    with torch.no_grad():
        while True:
            user, item, rating, seq, label, fake_r, feature = data.next_batch()
            user = user.to(device)  # (batch_size,)
            item = item.to(device)
            rating = rating.to(device)
            fake_r = fake_r.to(device)
            feature = feature.t().to(device)

            bos = seq[:, 0].unsqueeze(0).to(device)  # (1, batch_size)

            if args.use_feature:
                # movie =torch.full((1, 256), 11).to(device)
                text = torch.cat([feature, bos], 0)  # (src_len - 1, batch_size)
            else:
                text = bos  # (src_len - 1, batch_size)
            start_idx = text.size(0)
            for idx in range(args.words):
                # produce a word at each step
                if idx == 0:
                    ra_c, ra_s, re_c, re_s, uira_emb, uire_emb, \
                    de_ra, de_re, de_ra_c, de_ra_s, de_re, de_re_c, de_re_s, \
                    rating_ln, ui_re_distri, con_hidden, remlp = model(user, item, rating, fake_r, text,
                                                                mode="Test")  # (batch_size, ntoken) vs. (batch_size, ntoken) vs. (batch_size,)
                    rating_predict.extend(rating_ln.t().squeeze().tolist())
                    real_rating.extend(rating.float().tolist())
                    rating_data.extend(rating.float().tolist())
                    seq_data.extend(seq.tolist())
                else:
                    ra_c, ra_s, re_c, re_s, uira_emb, uire_emb, \
                    de_ra, de_re, de_ra_c, de_ra_s, de_re, de_re_c, de_re_s, \
                    rating_ln, ui_re_distri, con_hidden, remlp = model(user, item, rating, fake_r, text,
                                                                mode="Test")  # (batch_size, ntoken)
                word_prob = ui_re_distri.exp()  # (batch_size, ntoken)
                word_idx = torch.argmax(word_prob, dim=1)  # (batch_size,), pick the one with the largest probability
                text = torch.cat([text, word_idx.unsqueeze(0)], 0)  # (len++, batch_size)
            ids = text[start_idx:].t().tolist()  # (batch_size, seq_len)
            idss_predict.extend(ids)

            if data.step == data.total_step:
                break


    predicted_rating = [(r, p) for (r, p) in zip(rating_data, rating_predict)]
    RMSE = root_mean_square_error(predicted_rating, corpus.max_rating, corpus.min_rating)
    logger.info(now_time() + 'RMSE {:7.4f}'.format(RMSE))
    MAE = mean_absolute_error(predicted_rating, corpus.max_rating, corpus.min_rating)
    logger.info(now_time() + 'MAE {:7.4f}'.format(MAE))

    # text
    tokens_test = [ids2tokens(ids[1:], word2idx, idx2word) for ids in seq_data]
    tokens_predict = [ids2tokens(ids, word2idx, idx2word) for ids in idss_predict]
    BLEU1 = bleu_score(tokens_test, tokens_predict, n_gram=1, smooth=False)
    logger.info(now_time() + 'BLEU-1 {:7.4f}'.format(BLEU1))
    BLEU4 = bleu_score(tokens_test, tokens_predict, n_gram=4, smooth=False)
    logger.info(now_time() + 'BLEU-4 {:7.4f}'.format(BLEU4))
    USR, USN = unique_sentence_percent(tokens_predict)
    logger.info(now_time() + 'USR {:7.4f} | USN {:7}'.format(USR, USN))
    feature_batch = feature_detect(tokens_predict, feature_set)
    # DIV = feature_diversity(feature_batch)  # time-consuming
    # logger.info(now_time() + 'DIV {:7.4f}'.format(DIV))
    FCR = feature_coverage_ratio(feature_batch, feature_set)
    logger.info(now_time() + 'FCR {:7.4f}'.format(FCR))
    feature_test = [idx2word[i] for i in data.feature.squeeze(1).tolist()]  # ids to words
    FMR = feature_matching_ratio(feature_batch, feature_test)
    logger.info(now_time() + 'FMR {:7.4f}'.format(FMR))
    text_test = [' '.join(tokens) for tokens in tokens_test]
    text_predict = [' '.join(tokens) for tokens in tokens_predict]
    ROUGE = rouge_score(text_test, text_predict)  # a dictionary
    for (k, v) in ROUGE.items():
        logger.info(now_time() + '{} {:7.4f}'.format(k, v))
    text_out = ''
    for (rating, real, fake) in zip(real_rating, text_test, text_predict):
        text_out += '{}\n{}\n{}\n\n'.format(rating, real, fake)
    return text_out


# Loop over epochs.
logger.info("========================================")
logger.info("========================================")
best_val_loss = float('inf')
best_val_con_loss = float('inf')
endure_count = 0
flag = 1
for epoch in range(1, args.epochs + 1):
    logger.info(now_time() + 'epoch {}'.format(epoch))

    train(train_data, flag=flag)
    val_t_loss, val_r_loss, val_l_loss, val_sc_loss, val_rv_loss = evaluate(val_data, flag=flag)
    val_loss = val_t_loss + 0.1 * val_r_loss + 0.1 * val_l_loss #+ val_rv_loss
    if val_loss < best_val_loss:  # or val_con_loss < best_val_con_loss:
        best_val_loss = val_loss
        with open(model_path, 'wb') as f:
            torch.save(model, f)
    else:
        endure_count += 1
        logger.info(now_time() + 'Endured {} time(s)'.format(endure_count))
        if endure_count == args.endure_times:
            logger.info(now_time() + 'Cannot endure it anymore | Exiting from early stop')
            break

        sch_rat.step()
        sch_text.step()
        sch_ll.step()
        # logger.info(now_time() + 'Learning rate set to {:2.8f}'.format(scheduler.get_last_lr()[0]))

    logger.info(
        now_time() + 'text ppl {:4.4f} | rating loss {:4.4f} | vl loss {:4.4f} | rv loss {:4.4f}  | v-sc loss {:4.4f} | v-loss {:4.4f} on validation'.format(
            val_t_loss, val_r_loss, val_l_loss, val_rv_loss, val_sc_loss, val_loss))

# Load the best saved model.
with open(model_path, 'rb') as f:
    model = torch.load(f).to(device)

# Run on test data.
# test_c_loss, test_t_loss, test_r_loss = evaluate(test_data)
# logger.info('=' * 89)
# logger.info(now_time() + 'context ppl {:4.4f} | text ppl {:4.4f} | rating loss {:4.4f} on test | End of training'.format(
#     math.exp(test_c_loss), math.exp(test_t_loss), test_r_loss))

# val_t_loss, val_r_loss, val_l_loss, val_sc_loss, val_rv_loss = evaluate(val_data, flag=1)

logger.info(now_time() + 'Generating text')
text_o = generate(test_data)
with open('{}.txt'.format(prediction_path), 'w', encoding='utf-8') as f:
    f.write(text_o)
logger.info(now_time() + 'Generated text saved to ({}.txt)'.format(prediction_path))
