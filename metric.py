import pdb
from collections import defaultdict
import numpy as np
import pandas as pd

'''
@author:
Enneng Yang (ennengyang@gmail.com)
evalution metric

'''

# classification task
def cal_precision_and_recall(labels, preds):
    predss = [round(i,0) for i in preds]
    true_positive_samples = (np.array(preds) * np.array(labels) >= 0.5).tolist().count(True)
    false_positive_samples = (np.array(preds) * (1 - np.array(labels)) >= 0.5).tolist().count(True)
    false_negative_samples = (np.array(preds) * np.array(labels) < 0.5).tolist().count(True)
    true_negative_samples = (np.array(preds) * (1 - np.array(labels)) < 0.5).tolist().count(True)

    precision, recall, accuracy, f1_score = 0., 0., 0., 0.
    if (true_positive_samples + false_positive_samples) > 0 and (true_positive_samples + false_negative_samples) > 0:
        precision = float(true_positive_samples) / (true_positive_samples + false_positive_samples)
        recall = float(true_positive_samples) / (true_positive_samples + false_negative_samples)
        accuracy = float(true_positive_samples + true_negative_samples) / (true_positive_samples + false_positive_samples + false_negative_samples + true_negative_samples)
        if (precision + recall) > 0:
            f1_score = 2 * precision * recall / (precision + recall)
    return round(precision, 8), round(recall, 8), round(accuracy, 8), round(f1_score, 8)

def class_metrics(labels, preds):
    # 判断正确的个数
    pred_label = [round(i,0) for i in preds]
    true_positive_samples = (np.array(pred_label) == np.array(labels)).sum()
