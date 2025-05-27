""" Eval metrics and related

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import numpy as np
from sklearn.metrics import confusion_matrix as cm_sklearn
from sklearn.metrics import f1_score

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

def F1_Score(count_pred, count):
    f1_positive= 2*count_pred[1]/(2*count_pred[1] + count[0] - count_pred[0] + count[1] - count_pred[1])
    f1_negative= 2*count_pred[0]/(2*count_pred[0] + count[0] - count_pred[0] + count[1] - count_pred[1])
    f1 = (f1_positive + f1_negative)/2
    return 100*f1

def Count(y_pred, y_true):
    predictions = torch.greater(y_pred[:,1], y_pred[:,0]).int()
    count_pred= [0, 0]
    for i in range(len(y_true)):
        if y_true[i] == predictions[i]:
            count_pred[int(y_true[i].item())] += 1
        else:
            pass
    return count_pred


def AUC(output, target):
    auc= roc_auc_score(target, output)
    return auc

def PR_AUC(output, target):
    precision, recall, thresholds = precision_recall_curve(target, output)
    auc_score = auc(recall, precision)
    return auc_score

def print_network(net):
    num_params = 0
    num_params_train = 0
    
    for param in net.parameters():
            n = param.numel()
            num_params += n
            if param.requires_grad:
                    num_params_train += n      
    return num_params, num_params_train    


def opt_f1(thresholds, output, target):
    f1_scores = []
    for threshold in thresholds:   
        # obtain class prediction based on threshold
        y_predictions = np.where(output>=threshold, 1, 0) 

        #f1 scores
        f1 = f1_score(target, y_predictions, average='macro')
        f1_scores.append(f1)
        
    print('\nBest threshold: ', thresholds[np.argmax(f1_scores)], 'Best F1-Score: ', np.max(f1_scores))