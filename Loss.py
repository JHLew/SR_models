import torch
import torch.nn as nn


class Thresh_Error(nn.Module):
    def __init__(self, threshold=5, squared=True, penalty='over'):
        super(Thresh_Error, self).__init__()
        self.t = threshold
        self.squared = squared
        assert penalty in ['over', 'under'], 'penalty over of under threshold'
        self.penalty = penalty

    def forward(self, pred, target):
        dist = torch.abs(pred - target)
        shift_dist = dist - self.t
        if self.penalty == 'under':
            shift_dist = -shift_dist
        thresholded = torch.relu(shift_dist)
        if self.squared:
            thresholded = thresholded ** 2
        if self.penalty == 'under':
            thresholded = -thresholded

        return thresholded.mean()
