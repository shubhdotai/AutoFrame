"""Inference-compatible TalkNet/LR-ASD loss heads."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class lossAV(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCELoss()
        self.FC = nn.Linear(128, 2)

    def forward(self, x, labels=None, r=1):
        x = x.reshape(-1, x.shape[-1])
        logits = self.FC(x)
        if labels is None:
            return logits[:, 1].detach().cpu().numpy()

        probs = F.softmax(logits / r, dim=-1)[:, 1]
        nloss = self.criterion(probs, labels.reshape(-1).float())
        pred_score = F.softmax(logits, dim=-1)
        pred_label = torch.round(pred_score)[:, 1]
        correct_num = (pred_label == labels.reshape(-1)).sum().float()
        return nloss, pred_score, pred_label, correct_num


class lossV(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCELoss()
        self.FC = nn.Linear(128, 2)

    def forward(self, x, labels, r=1):
        x = x.reshape(-1, x.shape[-1])
        logits = self.FC(x)
        probs = F.softmax(logits / r, dim=-1)
        return self.criterion(probs[:, 1], labels.reshape(-1).float())

