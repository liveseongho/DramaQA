import math

from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


def nll_loss(output, target):
    return F.nll_loss(output, target)

def cross_entropy(output, target):
	return F.cross_entropy(output, target)

class MarginRankingLoss(nn.MarginRankingLoss):
	def __init__(self, margin = 0.0):
		super(MarginRankingLoss, self).__init__(margin=margin)


class CrossEntropyLoss(nn.CrossEntropyLoss):
	def __init__(self, padding_idx=0):
		super(CrossEntropyLoss, self).__init__(ignore_index=padding_idx)