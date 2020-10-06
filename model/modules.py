import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from einops import rearrange


class ContextMatching(nn.Module):
    def __init__(self, channel_size):
        super(ContextMatching, self).__init__()
        
        self.mlp = nn.Linear(channel_size * 2, 1, bias=False)

    @classmethod
    def get_u_tile(cls, s, s2):
        a_weight = F.softmax(s, dim=2)  # [B, t1, t2]
        a_weight.data.masked_fill_(a_weight.data != a_weight.data, 0)  # remove nan from softmax on -inf
        
        u_tile = torch.bmm(a_weight, s2)  # [B, t1, t2] * [B, t2, D] -> [B, t1, D]
        return u_tile

    def forward(self, s1, l1, s2, l2, mask2d=False):
        t1 = s1.size(1) 
        t2 = s2.size(1)
        repeat_s1 = s1.unsqueeze(2).repeat(1, 1, t2, 1)  # [B, T1, T2, D]
        repeat_s2 = s2.unsqueeze(1).repeat(1, t1, 1, 1)  # [B, T1, T2, D]
        packed_s1_s2 = torch.cat([repeat_s1, repeat_s2], dim=3)  # [B, T1, T2, D*3]
        s = self.mlp(packed_s1_s2).squeeze(dim=3)  # s is the similarity matrix from biDAF paper. [B, T1, T2]
        '''
        s = torch.bmm(s1, s2.transpose(1, 2))
        '''

        if not mask2d:
            s_mask = s.data.new(*s.size()).fill_(1).bool()  # [B, T1, T2]
            # Init similarity mask using lengths
            for i, (l_1, l_2) in enumerate(zip(l1, l2)):
                s_mask[i][:l_1, :l_2] = 0

            s_mask = Variable(s_mask)
            s.data.masked_fill_(s_mask.data.bool(), -float("inf"))
        else:
            l1 = l1.view(-1)
            s_mask = torch.arange(max(l1), device=l1.device).expand(len(l1), max(l1)) >= l1.unsqueeze(1)
            s_mask = s_mask.view(s1.shape[0], -1).unsqueeze(2).repeat(1,1,max(l2))   

            for i, l_2 in enumerate(l2):
                s_mask[i][:, l_2:] = True
            s.masked_fill_(s_mask, -float("inf"))
        
        u_tile = self.get_u_tile(s, s2)
        
        return u_tile


class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()

        self.size = d_model
          # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm



class MHAttn(nn.Module):
    def __init__(self, heads, hidden, d_model, dropout = 0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = int(hidden/heads)
        self.h = heads

        self.q_linear = nn.Linear(d_model, hidden)
        self.v_linear = nn.Linear(d_model, hidden)
        self.k_linear = nn.Linear(d_model, hidden)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        # calculate attention using function we will define next
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_k*self.h)

        output = self.out(concat)

        return output

    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1).repeat(1,self.h,1,1)
            scores = scores.masked_fill_(mask, -float("inf"))

        scores = F.softmax(scores, dim=-1)

        scores = scores.transpose(-2, -1).repeat(1,1,1,self.d_k)

        if dropout is not None:
            scores = dropout(scores)

        #output = torch.matmul(scores, v)
        output = scores * v

        return output

