"""
 Multi-level Context Matching code
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from transformers import BertConfig, BertModel

from . rnn import RNNEncoder, max_along_time, mean_along_time
from . modules import ContextMatching

speaker_name = [
    'None', # index 0: unknown speaker
    'Anna', 'Chairman', 'Deogi', 'Dokyung', 'Gitae',
    'Haeyoung1', 'Haeyoung2', 'Heeran', 'Hun', 'Jeongsuk',
    'Jinsang', 'Jiya', 'Kyungsu', 'Sangseok', 'Seohee',
    'Soontack', 'Sukyung', 'Sungjin', 'Taejin', 'Yijoon'
]


class MCM(BaseModel):
    def __init__(self, pt_emb, **kwargs):
        super().__init__()
        visual_dim = kwargs['visual_dim']
        dropout_p = kwargs['dropout_p']

        self.opts = kwargs['options']
        self.remove_metadata = kwargs['remove_metadata']
        self.remove_coreference = kwargs['remove_coreference']

        D = kwargs["n_dim"]  # pt_emb.shape[1]
        self.V = pt_emb
        self.spk_indexes = []
        for spk in speaker_name:
            self.spk_indexes.append(self.V.get(spk))

        clf_dim = 0
        if self.opts['subs_low']:
            clf_dim += 1
        if self.opts['subs_high']:
            clf_dim += 1
        if self.opts['visual_low']:
            clf_dim += 1
        if self.opts['visual_high']:
            clf_dim += 1

        self.sent_embedder = BertModel.from_pretrained('bert-base-cased')
        for param in self.sent_embedder.parameters():
            param.requires_grad = False
        self.distil2imsize = nn.Linear(768, D)

        self.bilstm_qa = RNNEncoder(300, 150, bidirectional=True, dropout_p=dropout_p, n_layers=1, rnn_type="lstm", return_hidden=False)

        '''
        if not kwargs['remove_coreference']:
            self.bilstm_subs = RNNEncoder(321, 150, bidirectional=True, dropout_p=dropout_p, n_layers=1, rnn_type="lstm", return_hidden=False)
        else:
            self.bilstm_subs = RNNEncoder(300, 150, bidirectional=True, dropout_p=dropout_p, n_layers=1, rnn_type="lstm", return_hidden=False)
        '''
        if self.opts['subs_low'] or self.opts['subs_high']:
            self.cmat_subs = ContextMatching(2*D+21)
            if not kwargs['remove_coreference']:
                self.conv_pool_subs = Conv1d(D*2+23, D*2)
            else:
                self.conv_pool_subs = Conv1d(D*3+1, D*2)
            self.clf_subs = nn.Sequential(nn.Linear(D*2, 1), nn.Softmax(dim=1))
        '''
        if self.opts['subs_high']:
            self.bilstm_subs_high = RNNEncoder(300, 150, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm", return_hidden=False)
            self.cmat_subs_high = ContextMatching(D)
            self.conv_pool_subs_high = Conv1d(D*3+2, D*2)
            self.clf_subs_high = nn.Sequential(nn.Linear(D*2, 1), nn.Softmax(dim=1))
        '''

        '''
        if not kwargs['remove_metadata']:
            self.bilstm_bbfts = RNNEncoder(visual_dim+D*2+21, 150, bidirectional=True, dropout_p=dropout_p, n_layers=1, rnn_type="lstm", return_hidden=False)
        else:
            self.bilstm_bbfts = RNNEncoder(visual_dim, 150, bidirectional=True, dropout_p=dropout_p, n_layers=1, rnn_type="lstm", return_hidden=False)
        '''
        if self.opts['visual_low'] or self.opts['visual_high']:
            self.cmat_bbfts = ContextMatching(863)
            if not kwargs['remove_metadata']:
                self.conv_pool_bbfts = Conv1d(864, D*2)
            else:
                self.conv_pool_bbfts = Conv1d(D*3, D*2)
            self.clf_bbfts = nn.Sequential(nn.Linear(D*2, 1), nn.Softmax(dim=1))

        '''
        if self.opts['visual_high']:
            self.bilstm_bbfts_high = RNNEncoder(300, 150, bidirectional=True, dropout_p=dropout_p, n_layers=1, rnn_type="lstm", return_hidden=False)
            self.cmat_bbfts_high = ContextMatching(D)
            self.conv_pool_bbfts_high = Conv1d(D*3+1, D*2)
            self.clf_bbfts_high = nn.Sequential(nn.Linear(D*2, 1), nn.Softmax(dim=1))
        '''
        self.character = nn.Parameter(torch.randn(22, D, dtype=torch.float), requires_grad=True)
        self.ch_emb1 = nn.Linear(300, 321)
        self.ch_emb2 = nn.Linear(300, 563)
        self.dim_reduce = nn.Linear(600, 30)

    def bert_encoder(self, sentence_dict):
        B = sentence_dict['input_ids'].shape[0]
        T1 = sentence_dict['input_ids'].shape[1]
        for k, v in sentence_dict.items():
            sentence_dict[k] = v.view(-1, v.shape[-1])
        input_ids = sentence_dict['input_ids']
        attention_mask = sentence_dict['attention_mask']
        token_type_ids = sentence_dict['token_type_ids']
        o = self.sent_embedder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        o = o.view(B, T1, -1, o.shape[-1])
        return self.distil2imsize(o)

    def bert_encoder_vgraph(self, vgraphs):
        # vgraphs.shape: torch.Size([batch_size, shot, frames, 2])
        B = vgraphs.shape[0]
        Nshot = vgraphs.shape[1]
        Nframe = vgraphs.shape[2]

        vgraphs = vgraphs.view(-1, 2)
        input_ids = vgraphs
        attention_mask = vgraphs.data.new(*vgraphs.size()).fill_(1)
        token_type_ids = vgraphs.data.new(*vgraphs.size()).fill_(0)

        o = self.sent_embedder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        o = o.view(B, Nshot, Nframe, 2, o.shape[-1])

        return self.distil2imsize(o).view(B, Nshot, Nframe, -1)

    def forward(self, x):
        B = x['spkr'].shape[0]
        # q, q_len = x['que'], x['que_l']
        qas_dict, qas_l = x['qa'], x['qa_l']
        out_concat = []

        # Embed question
        # q: (B, #words, 300)
        '''
        print(ans)
        q_idx = self._to_one_hot(q, self.V, mask=q_len)
        e_q = self.embedding(q)
        q_ctx, _ = self.bilstm_qa(e_q, q_len)
        q_attn, _ = self.bilstm_qa_attn(e_q, q_len)
        '''
        # Embed answres
        # ans: [(B, #words, 300) x 5]
        # qas = qas.transpose(0, 1)
        # qas_l = qas_l.transpose(0,1)
        e_qas = [self.bert_encoder(qas_d).squeeze() for qas_d in qas_dict]

        qas = [qas_dict[i]['input_ids'] for i in range(5)]  #.transpose(0, 1)
        # qas = qas.view(B, 5, -1).transpose(0, 1)
        # qas_l = qas_l
        qas_idx = [self._to_one_hot(qas[i], len(self.V), mask=qas_l[i]) for i in range(5)]
        qas_idx = [torch.sum(qas_idx[i], dim=1) for i in range(5)]

        # QA + BiLSTM baseline
        concat_qa = [(self.get_name(qas[i], qas_l[i])).type(torch.cuda.FloatTensor) for i in range(5)]
        concat_qa_none = [(torch.sum(concat_qa[i], dim=1) == 0).unsqueeze(1).type(torch.cuda.FloatTensor) for i in range(5)]
        concat_qa_none = [torch.cat([concat_qa[i], concat_qa_none[i]], dim=1) for i in range(5)]
        concat_qa_none = [concat_qa_none[i].unsqueeze(2).expand(-1, -1, self.character.shape[-1]) for i in range(5)]
        q_c = [concat_qa_none[i] * self.character for i in range(5)]

        # Embed subtitle
        # subs: (B, #sentence * #words, 300)
        # sub_len: (B)
        # subs_len: (B, #sentence)
        subs_dict, subs_len = x['sub'], x['sub_l_l']
        subs = subs_dict['input_ids']

        N_sentence = subs.shape[1]
        N_word = subs_len.max().item()

        subs_idx = self._to_one_hot(subs, len(self.V), mask=subs_len, type3d=True)
        subs_idx = subs_idx.view(B, -1, subs_idx.shape[-1])
        qas_idx = [qas_idx[i].unsqueeze(1).expand(-1, subs_idx.shape[1], -1) for i in range(5)]
        subs_qa_flag = [torch.sum(subs_idx * qas_idx[i], dim=2) for i in range(5)]
        subs_qa_flag = [(subs_qa_flag[i] > 0).type(torch.cuda.FloatTensor).unsqueeze(2) for i in range(5)]
        subs = self.bert_encoder(subs_dict) # (B, T, max_w, 300)

        # Embed speaker

        spk, sub_len = x['spkr'], x['sub_l']
        if not self.remove_coreference:
            self.change_index(spk)
            spk_onehot = self._to_one_hot(spk, 21, mask=sub_len)
            spk_onehot = spk_onehot.unsqueeze(2).expand(-1, -1, N_word, -1)
            subs_spk = torch.cat([subs, spk_onehot], dim=3)

            spk_onehot = spk_onehot.reshape(B, -1, 21)
            spk_flag = [torch.matmul(spk_onehot, concat_qa[i].unsqueeze(2)) for i in range(5)]
            spk_flag = [(spk_flag[i] > 0).type(torch.cuda.FloatTensor) for i in range(5)]

            # c_subs = self.stream_context(subs_spk, subs_len, self.bilstm_subs)
            c_subs = subs_spk
        else:
            spk_flag = None
            c_subs = self.stream_context(subs, subs_len, self.bilstm_subs)

        #c_subs = c_subs + subs.view(B, N_sentence*N_word, c_subs.shape[-1])
        c_subs = c_subs.view(B, N_sentence*N_word, c_subs.shape[-1])

        if self.opts['subs_low']:
            qas_l = [qas_l[i] for i in range(5)]
            out_subs = self.cmat_conv_pool(
                c_subs, subs_len,
                e_qas, qas_l,
                self.cmat_subs,
                self.conv_pool_subs,
                self.clf_subs,
                spk_flag, mask2d=True, subs_qa_flag=subs_qa_flag)
            out_concat.append(out_subs)
        else:
            out_subs = 0

        if self.opts['subs_high']:
            c_subs = c_subs.view(B, N_sentence, N_word, c_subs.shape[-1])
            out_subs_high = self.cmat_conv_pool_high(c_subs, sub_len, subs_len,
                                            e_qas, qas_l,
                                            q_c,
                                            None,
                                            self.cmat_subs,
                                            self.conv_pool_subs,
                                            self.clf_subs, spk_flag, subs_qa_flag = subs_qa_flag)
            out_concat.append(out_subs_high)
        else:
            out_subs_high = 0

        # Embed visual
        bbfts, bbfts_l, bbfts_len = x['bbfts'], x['bbfts_l'], x['bbfts_l_l']
        vgraphs = x['vmeta']
        N_scene = bbfts.shape[1]
        N_shot = bbfts_len.max().item()
        if not self.remove_metadata:
            vgraphs_p = vgraphs[:, :, :, 0]
            self.change_index(vgraphs_p)
            vgraphs_p_onehot = self._to_one_hot(vgraphs_p, 21, mask=bbfts_len, type3d=True)
            vgraphs_be = vgraphs[:, :, :, 1:3].contiguous()
            vgraphs_be = self.dim_reduce(self.bert_encoder_vgraph(vgraphs_be))
            # vgraphs_be = self.embedding(vgraphs_be.view(B, -1)).view(B, N_scene, N_shot, -1)

            vgraphs_full = torch.cat([bbfts, vgraphs_be, vgraphs_p_onehot], dim=3)

            vgraphs_p_onehot = vgraphs_p_onehot.reshape(B, -1, 21)
            vis_flag = [torch.matmul(vgraphs_p_onehot, concat_qa[i].unsqueeze(2)) for i in range(5)]
            vis_flag = [(vis_flag[i] > 0).type(torch.cuda.FloatTensor) for i in range(5)]  # > 0

            # c_bbfts = self.stream_context(vgraphs_full, bbfts_len, self.bilstm_bbfts)  # vgraphs_full
            c_bbfts = vgraphs_full
        else:
            vis_flag = None
            c_bbfts = self.stream_context(bbfts, bbfts_len, self.bilstm_bbfts)  # vgraphs_full

        c_bbfts = c_bbfts.view(B, N_scene*N_shot, c_bbfts.shape[-1])
        if self.opts['visual_low']:
            out_bbfts = self.cmat_conv_pool(
                c_bbfts, bbfts_len,
                e_qas, qas_l,
                self.cmat_bbfts,
                self.conv_pool_bbfts,
                self.clf_bbfts,
                vis_flag, mask2d=True)
            out_concat.append(out_bbfts)
        else:
            out_bbfts = 0

        c_bbfts = c_bbfts.view(B, N_scene, N_shot, c_bbfts.shape[-1])
        if self.opts['visual_high']:
            out_bbfts_high = self.cmat_conv_pool_high(
                c_bbfts, bbfts_l, bbfts_len,
                e_qas, qas_l,
                q_c,
                None,
                self.cmat_bbfts,
                self.conv_pool_bbfts,
                self.clf_bbfts, vis_flag)
            out_concat.append(out_bbfts_high)
        else:
            out_bbfts_high = 0
        # out_qa: (B, 5, 4)
        # out_subs: (B, 5, 1)
        # out_bbfts: (B, 5, 1)

        out_final = torch.cat(out_concat, dim=2)

        #out = self.final_clf(out_qa*out_final)
        out = torch.sum(out_final, dim=-1)
        #out = out_subs+out_subs_high+out_bbfts+out_bbfts_high

        #print(out_final)

        return out.view(B, -1)


    def stream_context(self, stream, stream_l, context_embed):
        # stream: (B, N, M, D)
        # stream_l: (B, N)
        B = stream.shape[0]
        stream = stream.view(-1, stream.shape[2], stream.shape[3])
        stream, _ = context_embed(stream, stream_l.view(-1))
        stream = stream.view(B, -1, stream.shape[2])

        return stream


    def cmat_conv_pool(self, ctx, ctx_len, ans, ans_len, cmat, conv_pool, clf, ctx_flag=None,  mask2d=False, subs_qa_flag=None):
        # ctx: (B, T1*T2, D)
        # ctx_len: (B, T1)

        # Context Maching module
        #u_q = cmat(ctx, ctx_len, q, q_len, mask2d=True)
        u_a = [cmat(ctx, ctx_len, ans[i], ans_len[i], mask2d=True) for i in range(5)]

        # Concatenation
        if subs_qa_flag is None:
            concat_all = [torch.cat([ctx, u_a[i], ctx_flag[i]], dim=-1) for i in range(5)]
        else:
            concat_all = [torch.cat([ctx, u_a[i], ctx_flag[i], subs_qa_flag[i]], dim=-1) for i in range(5)]
        # Conv1D & max-pool
        maxout = [conv_pool(concat_all[i], ctx_len) for i in range(5)]
        #print(maxout[0].shape)
        answers = torch.stack(maxout, dim=1)
        out = clf(answers)

        return out

    def cmat_conv_pool_high(self, ctx, ctx_len, ctx_len_len, ans, ans_len, q_c, context_embed, cmat, conv_pool, clf, ctx_flag=None, subs_qa_flag=None):
        # ctx: (B, T1, T2, D)
        # ctx_len: (B)
        # q_c: [(B, 22, D)]
        if ctx_flag is not None:
            ctx_flag = [ctx_flag[i].view(ctx.shape[0], ctx.shape[1], ctx.shape[2], 1) for i in range(5)]
            ctx_flag = [torch.sum(ctx_flag[i], dim=2) for i in range(5)]
        if subs_qa_flag is not None:
            subs_qa_flag = [subs_qa_flag[i].view(ctx.shape[0], ctx.shape[1], ctx.shape[2], 1) for i in range(5)]
            subs_qa_flag = [torch.sum(subs_qa_flag[i], dim=2) for i in range(5)]
            q_c = [self.ch_emb1(q_c[i]) for i in range(5)]
        else:
            q_c = [self.ch_emb2(q_c[i]) for i in range(5)]

        ctx_attn = [torch.matmul(q_c[i].unsqueeze(1).expand(-1,ctx.shape[1],-1,-1), ctx.transpose(2,3)) for i in range(5)]
        ctx_attn = [F.softmax(ctx_attn[i], dim=3) for i in range(5)]
        ctx = [torch.matmul(ctx_attn[i], ctx) for i in range(5)]
        ctx = [torch.sum(ctx[i], dim=2) for i in range(5) for i in range(5)]
        '''
        ctx = torch.cat([ctx, ctx_flag], dim=3)
        ctx = ctx.view(-1, ctx.shape[2], ctx.shape[3])
        ctx = mean_along_time(ctx, ctx_len_len)
        '''
        #ctx = [context_embed(ctx[i], ctx_len)[0] for i in range(5)]

        # Context Maching module
        #u_q = [cmat(ctx[i], ctx_len, q, q_len) for i in range(5)]
        u_a = [cmat(ctx[i], ctx_len, ans[i], ans_len[i]) for i in range(5)]

        # Concatenation
        if subs_qa_flag is None:
            concat_all = [torch.cat([ctx[i], u_a[i], ctx_flag[i]], dim=-1) for i in range(5)]
        else:
            concat_all = [torch.cat([ctx[i], u_a[i], ctx_flag[i], subs_qa_flag[i]], dim=-1) for i in range(5)]

        # Conv1D & max-pool
        maxout = [conv_pool(concat_all[i], ctx_len) for i in range(5)]
        #print(maxout[0].shape)
        answers = torch.stack(maxout, dim=1)
        out = clf(answers)

        return out


    def mask2d(self, ts, l):
        mask = torch.arange(max(l), device=l.device).expand(len(l), max(l)) >= l.unsqueeze(1)
        ts.masked_fill_(mask.expand(-1, ts.shape[1]), 0)
        return ts

    def len_to_mask(self, lengths, len_max=None):
        if len_max is None:
            len_max = lengths.max().item()
        mask = torch.arange(len_max, device=lengths.device).expand(len(lengths), len_max) >= lengths.unsqueeze(1)
        #mask = torch.as_tensor(mask, dtype=torch.bool)

        return mask, len_max

    def _to_one_hot(self, y, n_dims, mask, type3d=False):
        if not type3d:
            '''
              y: (B, T)
              mask: (B)
              return:  (B, T, n_dims)
            '''
            scatter_dim = len(y.size())
            y_tensor = y.type(torch.cuda.LongTensor).view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), n_dims).type(torch.cuda.FloatTensor)
            out = zeros.scatter(scatter_dim, y_tensor, 1)

            out_mask,_ = self.len_to_mask(mask, out.shape[1])
            out_mask = out_mask.unsqueeze(2).repeat(1, 1, n_dims)
        else:
            '''
              y: (B, T1, T2)
              mask: (B, T1)
              return:  (B, T1, T2, n_dims)
            '''
            scatter_dim = len(y.size())
            y_tensor = y.type(torch.cuda.LongTensor).view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), n_dims).type(torch.cuda.FloatTensor)
            out = zeros.scatter(scatter_dim, y_tensor, 1)

            out_mask, _ = self.len_to_mask(mask.view(-1), y.shape[2])
            out_mask = out_mask.view(out.shape[0], out.shape[1], out.shape[2])
            out_mask = out_mask.unsqueeze(3).repeat(1,1,1,n_dims)

        return out.masked_fill_(out_mask, 0)

    def change_index(self, tensor):
        for i, spk_i in enumerate(self.spk_indexes):
            tensor.masked_fill_(tensor==spk_i, i)
        return tensor

    def get_name(self, x, x_l):
        mask_ = x.data.new(*x.size()).fill_(0).bool()
        for spk_i in self.spk_indexes:
            mask_ = mask_ + (x == spk_i)

        x_mask = x.masked_fill(mask_==False, 21)  # 7330 is None index

        self.change_index(x_mask)

        x_onehot = self._to_one_hot(x_mask, 22, x_l)
        x_sum = torch.sum(x_onehot[:,:,:21], dim=1)
        return x_sum > 0

    def get_name2(self, x, x_l):
        mask_ = x.data.new(*x.size()).fill_(0).bool()
        for spk_i in self.spk_indexes:
            mask_ = mask_ + (x == spk_i)

        x_mask = x.masked_fill(mask_==False, 7330)  # 7330 is None index

        self.change_index(x_mask)

        x_onehot = self._to_one_hot(x_mask, 22, x_l)
        x_sum = torch.sum(x_onehot[:,:,1:22], dim=1)
        return x_sum > 0

class Conv1d(nn.Module):
    def __init__(self, n_dim, out_dim):
        super().__init__()
        out_dim = int(out_dim/4)
        self.conv_k1 = nn.Conv1d(n_dim, out_dim, kernel_size=1, stride=1)
        self.conv_k2 = nn.Conv1d(n_dim, out_dim, kernel_size=2, stride=1)
        self.conv_k3 = nn.Conv1d(n_dim, out_dim, kernel_size=3, stride=1)
        self.conv_k4 = nn.Conv1d(n_dim, out_dim, kernel_size=4, stride=1)

    def forward(self, x, mask_len):
        # x : (B, T, 5*D)
        x_pad = torch.zeros(x.shape[0], 3,x.shape[2]).type(torch.cuda.FloatTensor)
        x = torch.cat([x, x_pad], dim=1)
        x1 = F.relu(self.conv_k1(x.transpose(1, 2)))[:,:,:-3]
        x2 = F.relu(self.conv_k2(x.transpose(1, 2)))[:,:,:-2]
        x3 = F.relu(self.conv_k3(x.transpose(1, 2)))[:,:,:-1]
        x4 = F.relu(self.conv_k4(x.transpose(1, 2)))
        out = torch.cat([x1, x2, x3, x4], dim=1)
        out = out.transpose(1, 2)
        out = self.mask3d2d(out, mask_len)
        return max_along_time(out, None)

    def mask3d2d(self, ts, l):
        B = ts.shape[0]
        l = l.view(-1)
        ts = ts.reshape(l.shape[0], -1, ts.shape[-1])
        mask = torch.arange(max(l), device=l.device).expand(len(l), max(l)) >= l.unsqueeze(1)
        ts.masked_fill_(mask.unsqueeze(2).expand(-1, -1, ts.shape[2]), 0)
        ts = ts.view(B, -1, ts.shape[-1])
        return ts
