import torch
import torch.nn.functional as F
import numpy as np
import math
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report, average_precision_score


def compute_loss(pos_score, neg_score, device):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).to(device)
    return F.binary_cross_entropy_with_logits(scores, labels)


def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).cpu().data.numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)


def evaluation(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).cpu().numpy()
    outputs = np.asarray([1 if score >= 0.5 else 0 for score in scores])
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()

    return accuracy_score(labels, outputs), f1_score(labels, outputs), roc_auc_score(labels, scores)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


if __name__ == '__main__':
    print(f1_score(np.array([1, 0, 0, 0, 1]), np.array([1, 0, 1, 0, 1])))
