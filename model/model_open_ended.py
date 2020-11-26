"""
 Multi-level Context Matching code
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

from . rnn import RNNEncoder #, max_along_time, mean_along_time
from . modules import ContextMatching

import random
from torch import optim

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=50):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden) 
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights
  
class OpenEnded(BaseModel):
    def __init__(self, pt_emb, **kwargs):
        super().__init__()
        visual_dim = kwargs['visual_dim'] # image dimension;s size
        dropout_p = kwargs['dropout_p']

        self.opts = kwargs['options'] 
        self.remove_metadata = kwargs['remove_metadata'] # bounding box etc
        self.remove_coreference = kwargs['remove_coreference']
        self.vocab = pt_emb
        D = kwargs["n_dim"]#pt_emb.shape[1]
        self.V = pt_emb.shape[0]

        self.embedding = nn.Embedding(pt_emb.shape[0], D) # what pt_emb means? pretrained, 300 dim, 7000 *300), glove, words num
        self.embedding.weight.data.copy_(torch.from_numpy(pt_emb))
        
        # q
        self.bilstm_que = RNNEncoder(300, 150, bidirectional=True, dropout_p=dropout_p, n_layers=1, rnn_type="lstm", return_hidden=False)
        self.bilstm_subs = RNNEncoder(321, 150, bidirectional=True, dropout_p=dropout_p, n_layers=1, rnn_type="lstm", return_hidden=False)
        self.bilstm_bbfts = RNNEncoder(visual_dim+D*2+21, 150, bidirectional=True, dropout_p=dropout_p, n_layers=1, rnn_type="lstm", return_hidden=False)

        self.cmat_subs = ContextMatching(D)
        if not kwargs['remove_coreference']:
            # For representation for timeline by merging one vector
            self.conv_pool_subs = Conv1d(D*3+2, D*2)
        self.cmat_bbfts = ContextMatching(D)
        if not kwargs['remove_metadata']:
            self.conv_pool_bbfts = Conv1d(D*3+1, D*2)

        self.character = nn.Parameter(torch.randn(22, D, dtype=torch.float), requires_grad=True)
        self.subs_linear = nn.Linear(300, 256)

        # initializing for open-ended
        self.SOS_token = 0
        self.EOS_token = 1
        self.teacher_forcing_ratio = 0.5
        self.hidden_size = 256 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.learning_rate = 0.01
        self.criterion = nn.NLLLoss()

        i_n_words = self.V
        o_n_words = self.V
        
        self.encoder = EncoderRNN(i_n_words, self.hidden_size).to(self.device)
        self.decoder = AttnDecoderRNN(self.hidden_size, o_n_words, dropout_p=0.1).to(self.device)
        self.encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=self.learning_rate)
        self.decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=self.learning_rate)

    def forward(self, x): # what x means?
        data = x
        encoder_hidden = self.encoder.initHidden()
	 
	# video

        subs, subs_len = x['sub'], x['sub_l_l']

        B = subs.shape[0]
        N_sentence = subs.shape[1]
        N_word = subs_len.max().item()
        subs = self.embedding(subs) # (B, T, max_w, 300)

        # Embed speaker

        spk, sub_len = x['spkr'], x['sub_l']
        if not self.remove_coreference:
            spk_onehot = self._to_one_hot(spk, 21, mask=sub_len)
            spk_onehot = spk_onehot.unsqueeze(2).expand(-1, -1, N_word, -1)
            subs_spk = torch.cat([subs, spk_onehot], dim=3)

            spk_onehot = spk_onehot.reshape(B, -1, 21)
            # by using subs_spk , make context vector
            c_subs = self.stream_context(subs_spk, subs_len, self.bilstm_subs)
        else:
            spk_flag = None
            c_subs = self.stream_context(subs, subs_len, self.bilstm_subs)

        c_subs = c_subs + subs.view(B, N_sentence*N_word, c_subs.shape[-1])
        c_subs = torch.mean(c_subs, dim=1)
        c_subs = self.subs_linear(c_subs)
        encoder_hidden = c_subs.unsqueeze(1)
        
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        input_tensor = data['que']
        target_tensor = data['ans']
        input_tensor = input_tensor.transpose(1, 0)
        target_tensor = target_tensor.transpose(1, 0)
        input_length = len(input_tensor)
        target_length = len(target_tensor)
 
        encoder_outputs = torch.zeros(50, self.encoder.hidden_size, device=self.device)
    
        loss = 0
    
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]
    
        decoder_input = torch.tensor([[self.SOS_token]], device=self.device)
    
        decoder_hidden = encoder_hidden
    
        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
       
        decoder_outputs = torch.empty(target_length, 1, self.V).to(self.device)
        if use_teacher_forcing:
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_outputs[di] = decoder_output
                decoder_input = target_tensor[di]
    
        else:
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()
                decoder_outputs[di] = decoder_output
                if decoder_input.item() == self.EOS_token:
                    break
    
        return decoder_outputs

    def stream_context(self, stream, stream_l, context_embed):
        # stream: (B, N, M, D)
        # stream_l: (B, N)
        B = stream.shape[0]
        stream = stream.view(-1, stream.shape[2], stream.shape[3])
        stream, _ = context_embed(stream, stream_l.view(-1))
        stream = stream.view(B, -1, stream.shape[2])

        return stream


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
        x_pad = torch.zeros(x.shape[0],3,x.shape[2]).type(torch.cuda.FloatTensor)
        x = torch.cat([x, x_pad], dim=1)
        x1 = F.relu(self.conv_k1(x.transpose(1,2)))[:,:,:-3]
        x2 = F.relu(self.conv_k2(x.transpose(1,2)))[:,:,:-2]
        x3 = F.relu(self.conv_k3(x.transpose(1,2)))[:,:,:-1]
        x4 = F.relu(self.conv_k4(x.transpose(1,2)))
        out = torch.cat([x1, x2, x3, x4], dim=1)
        out = out.transpose(1,2)
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
