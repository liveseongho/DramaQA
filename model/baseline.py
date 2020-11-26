import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from torch.autograd import Variable

from . mlp import MLP
from . rnn import RNNEncoder, mean_along_time
from transformers import BertConfig, BertModel


class DotProdSim(BaseModel):
    def __init__(self, pt_emb, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(pt_emb.shape[0], pt_emb.shape[1])
        self.embedding.weight.data.copy_(torch.from_numpy(pt_emb))

    def forward(self, x):
        q = x['que']
        a = x['ans']
        ql = x['que_l']
        al = x['ans_l']

        q = self.embedding(q)
        a = self.embedding(a)
        q = mean_along_time(q,  ql)
        a = [mean_along_time(a[:, i], al[:, i]) for i in range(5)]
        sim = torch.stack([F.cosine_similarity(q, a[i], dim=1) for i in range(5)])
        sim = sim.transpose(0, 1)

        return sim


class LongestAnswer(BaseModel):
    def __init__(self, pt_emb, **kwargs):
        super().__init__()

    def forward(self, x):
        al = x['ans_len']

        return al


class ShortestAnswer(BaseModel):
    def __init__(self, pt_emb, **kwargs):
        super().__init__()

    def forward(self, x):
        al = 1/x['ans_len']

        return al


class QABert(BaseModel):
    def __init__(self, pt_emb, **kwargs):
        super().__init__()
        D = kwargs["n_dim"]  # pt_emb.shape[1]

        self.sent_embedder = BertModel.from_pretrained('bert-base-cased')
        for param in self.sent_embedder.parameters():
            param.requires_grad = True
        self.distil2imsize = nn.Linear(768, D)
        self.dim_reduce = nn.Linear(768, 300)
        self.get_score = MLP(300, 1, [100], 2)

    def bert_encoder(self, sentence_dict, dict_l):
        B = sentence_dict['input_ids'].shape[0]
        T1 = sentence_dict['input_ids'].shape[1]
        for k, v in sentence_dict.items():
            sentence_dict[k] = v.view(-1, v.shape[-1])
        input_ids = sentence_dict['input_ids']
        attention_mask = sentence_dict['attention_mask']
        token_type_ids = sentence_dict['token_type_ids']
        o = self.sent_embedder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        o = mean_along_time(self.dim_reduce(o.view(B, T1, o.shape[-1])), dict_l)
        return o

    def forward(self, x):
        # cls token, fine-tuning
        qas_dict, qas_l = x['qa'], x['qa_l']
        B = qas_dict[0]['input_ids'].shape[0]
        e_qas = [self.bert_encoder(qas_d, qas_l[i]) for i, qas_d in enumerate(qas_dict)]

        #e_qas = [self.get_score(torch.mean(emb, dim=1)) for emb in e_qas]
        e_qas = [self.get_score(emb) for emb in e_qas]
        e_qas = torch.stack(e_qas, 0).transpose(0, 1).squeeze()

        return e_qas.view (B, -1)


class RNNMLP(BaseModel):
    def __init__(self, pt_emb, **kwargs):
        super().__init__()
        V = pt_emb.shape[0]
        D = pt_emb.shape[1]
        self.embedding = nn.Embedding(V, D).requires_grad_(False)

        self.bilstm_sub = RNNEncoder(300, 150, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm", return_hidden=False)
        self.bilstm_vis = RNNEncoder(512, 150, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm", return_hidden=False)

        self.mlp = MLP(3 * D, 1, [D, 100], 3)

    def forward(self, x):

        qa = [x['qa'].transpose(0, 1)[i] for i in range(5)]
        qa_l = [x['qa_l'].transpose(0, 1)[i] for i in range(5)]
        sub, sub_l = x['filtered_sub'], x['filtered_sub_len']

        bbfts, bbfts_l = x['bbfts'], x['bbfts_l']

        e_qa = [mean_along_time(self.embedding(qa[i]), qa_l[i]) for i in range(5)]

        sub = self.embedding(sub)
        sub, _ = self.bilstm_sub(sub, sub_l)
        sub = mean_along_time(sub, sub_l)

        vis, _ = self.bilstm_vis(bbfts, bbfts_l)
        vis = mean_along_time(vis, bbfts_l)

        concat = torch.stack([torch.cat([e_qa[i], sub, vis], dim=1) for i in range(5)], dim=1)

        final_score = self.mlp(concat).squeeze()


        return final_score


class MemNet(BaseModel):
    def __init__(self):
        super().__init__()

    def forward(self, u, story, story_l, story_l_l):
        # u: (B, D)
        # story: (B, T1, D)
        u = u.unsqueeze(2)

        p = torch.bmm(story, u) # (B, T1, 1)
        p = p.view(p.shape[0], -1)

        p_mask = p.data.new(*p.size()).fill_(1).bool()
        # Init similarity mask using lengths
        for i, l in enumerate(story_l):
            p_mask[i][:l] = 0

        p_mask = Variable(p_mask)
        p.data.masked_fill_(p_mask.data.bool(), -float("inf"))

        p = F.softmax(p, dim=1)

        p = p.unsqueeze(2).expand(-1, -1, story.shape[2])

        o = torch.sum(story * p, dim=1) # (B,D)

        return o+u.squeeze()


class MemN2N(BaseModel):
    def __init__(self, pt_emb, **kwargs):
        super().__init__()
        V = pt_emb.shape[0]
        D = pt_emb.shape[1]
        self.embedding = nn.Embedding(V, D).requires_grad_(False)
        self.memnet = MemNet()
        self.linear_sub = nn.Linear(D, D)
        self.linear_vis = nn.Linear(3*D + 512, D)

    def sentence_embedding(self, story, story_l, story_l_l):
        story = story.view(-1, story.shape[2], story.shape[3])
        story = mean_along_time(story, story_l_l.view(-1))
        story = story.view(story_l.shape[0], -1, story.shape[-1])

        return story

    def get_score(self, e_s, e_q, e_a):
        ga_1 = [torch.sum(e_s * e_a[i], dim=1).unsqueeze(1) for i in range (5)]
        ga_1 = F.softmax(torch.cat(ga_1, dim=1), dim=1)

        return ga_1

    def forward(self, x):
        q, q_l = x['que'], x['que_l']
        a = [x['ans'].transpose(0, 1)[i] for i in range(5)]
        a_l = [x['ans_l'].transpose(0, 1)[i] for i in range(5)]
        sub, sub_l, sub_l_l = x['sub'], x['sub_l'], x['sub_l_l']
        vmeta, vmeta_l = x['vmeta'], x['bbfts_l']
        bbfts, bbfts_l, bbfts_l_l = x['bbfts'], x['bbfts_l'], x['bbfts_l_l']

        e_q = mean_along_time(self.linear_sub(self.embedding(q)), q_l)
        e_a = [mean_along_time(self.linear_sub(self.embedding(a[i])), a_l[i]) for i in range(5)]

        sub = self.linear_sub(self.embedding(sub))
        sub = self.sentence_embedding(sub, sub_l, sub_l_l)

        vis = self.embedding(vmeta)
        vis = vis.view(bbfts.shape[0], bbfts.shape[1], bbfts.shape[2], -1)
        vis = torch.cat([vis, bbfts], dim=3)

        vis = self.linear_vis(vis)
        vis = self.sentence_embedding(vis, bbfts_l, bbfts_l_l)

        e_s = self.memnet(e_q, sub, sub_l, sub_l_l)
        e_v = self.memnet(e_q, vis, bbfts_l, bbfts_l_l)


        for i in range(0):
            e_s = self.memnet(e_s, sub, sub_l, sub_l_l)
            e_v = self.memnet(e_v, vis, bbfts_l, bbfts_l_l)


        ga_2 = [torch.sum(e_q*e_a[i], dim=1).unsqueeze(1) for i in range (5)]
        ga_2 = F.softmax(torch.cat(ga_2, dim=1), dim=1)

        final_score = self.get_score(e_s, e_q, e_a) + self.get_score(e_v, e_q, e_a) + ga_2

        return final_score

