import torch
import torchvision
from torch import nn
import numpy as np


class LossConfidence(nn.Module):
    def __init__(self, reduction="sum"):
        super(LossConfidence, self).__init__()
        self.BCE = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, preds, targets):
        preds_conf = preds[:, 0]
        targets_conf = targets[:, 0]
        loss = self.BCE(preds_conf, targets_conf)

        matches = (preds_conf > 0) == targets_conf
        
        return loss, matches


class ContrastiveLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(ContrastiveLoss, self).__init__()
        self.reduction = reduction

    def forward(self, preds, labels):
        pos = preds[labels == 1].reshape(-1, 1)
        neg = preds[labels == 0].reshape(1, -1).repeat(pos.shape[0], 1)

        probs = torch.nn.functional.softmax(torch.cat([pos, neg], dim=1), dim=-1)
        probs_pos = probs[:, 0:1]
        
        loss = - (1 - probs_pos) * torch.log(probs_pos + 1e-6)
        
        loss = loss.mean() if self.reduction == "mean" else loss.sum()
        matches = torch.argmax(probs, dim=1) == 0
        
        return loss, matches

class Scheduler:
    def __init__(self, max_value, min_value, num_iters, num_warmup_iters, warmup_start_value):
        self.base_value = max_value
        self.final_value = min_value
        self.num_base_iters = num_iters

        num_base_iters = num_iters - num_warmup_iters
        base_iters = np.arange(num_base_iters)
        warmup_schedule = np.linspace(warmup_start_value, max_value, num_warmup_iters)
        base_schedule = min_value + 0.5 * (max_value - min_value) * (1 + np.cos(np.pi * base_iters / num_base_iters))

        self.schedule = np.concatenate([warmup_schedule, base_schedule])

    def get_value(self, iteration):
        if iteration >= len(self.schedule):
            return self.final_value
        else:
            return self.schedule[iteration]
        

class TrainStatistics:
    def __init__(self, loss_reduction_type):
        self.loss_reduction_type = loss_reduction_type
        
        self.data_dict = {"conf": None}
        self.reset_stats()

    def reset_stats(self):
        data_dict = {"conf": None}
        
        for key in data_dict.keys():
            data_dict[key] = {"loss": 0, "n": 0, "matches": 0}
                
        self.data_dict = data_dict

    def update_stats(self, key, loss, matches):

        loss = loss.detach().cpu().item()
        matches = matches.detach().cpu().numpy()
        
        n = len(matches)
        
        self.data_dict[key]["loss"] += (loss * n) if self.loss_reduction_type == "mean" else loss
        self.data_dict[key]["matches"] += np.sum(matches)

        self.data_dict[key]["n"] += n

    def get_stats(self):
        output_dict = {}
        for key in self.data_dict.keys():
            loss_mean = self.data_dict[key]["loss"] / self.data_dict[key]["n"]
            matches = self.data_dict[key]["matches"] / self.data_dict[key]["n"]
            
            output_dict[key] = {"loss": loss_mean, "matches": matches}

        return output_dict
        
        
