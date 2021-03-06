import json
import pickle
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict

import torch
import numpy as np
import torch.nn.functional as F
import copy

SPECIAL_TOKENS = ["<bos>", "<eos>", "<que>", "<ans>", "<speaker>", 
                  "<subtitle>", "<video>", "<pad>"]

def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits

def data_for_gpt(data, tokenizer):
    bos, eos, que, ans, speaker, subtitle  = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-2]) 
    token_type_ids = []
    sequence = []
    for spkr, script in zip(data['spkr'], data['script']):
        sequence_spkr = [speaker] + [spkr] + [eos]
        if len(script) == 1:
            sequence_script = [subtitle] + [bos] + [script[0]] + [eos] + [eos]
        else :
            sequence_script = [subtitle] + [bos] + script[0] + [eos] + [eos]
        sequence = sequence + sequence_spkr + sequence_script
        token_type_ids = token_type_ids + [speaker] * len(sequence_spkr) + [subtitle] * len(sequence_script)

    sequence_que = [que] + [bos] + data['que'][0] + [eos] + [eos]
    sequence = sequence + sequence_que
    token_type_ids = token_type_ids + [que] * len(sequence_que)

    sequence_ans = [ans] + data['ans'][0] + [eos]
    lm_labels = ([-1] * len(sequence)) + sequence_ans
    token_type_ids = token_type_ids + [ans] * len(sequence_ans)
    input_ids = sequence + sequence_ans
    input_ids = torch.Tensor(input_ids).long() 
    token_type_ids = torch.Tensor(token_type_ids).long() 
    lm_labels = torch.Tensor(lm_labels).long() 

    return input_ids, token_type_ids, lm_labels

def data_for_answer(input_ids, token_type_ids, answer, tokenizer, device):
    bos, eos, que, ans  = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-4])
    if len(answer) == 0 :
        return input_ids, token_type_ids 
    sequence_ans = answer
    tokens_type_ans = [ans] * len(sequence_ans)

    result_input_ids = torch.Tensor([input_ids.tolist()[0] + sequence_ans]).long()
    result_token_type_ids = torch.Tensor([token_type_ids.tolist()[0] + tokens_type_ans]).long()
    result_input_ids = result_input_ids.to(device)
    result_token_type_ids = result_token_type_ids.to(device)

    return result_input_ids, result_token_type_ids

def masking_answer(input_ids, token_type_ids, tokenizer):
    ans = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[3])
    return_input_ids, return_token_type_ids = [], []
    input_ids = torch.squeeze(input_ids, 0)
    token_type_ids = torch.squeeze(token_type_ids, 0) 
    for input_id, token_type_id in zip(input_ids, token_type_ids):
        if token_type_id.item() != ans:
            return_input_ids.append(input_id)
            return_token_type_ids.append(token_type_id)

    return_input_ids.append(ans)
    return_token_type_ids.append(ans)
    input_ids = torch.Tensor(return_input_ids).long()
    token_type_ids = torch.Tensor(return_token_type_ids).long()

    return torch.unsqueeze(input_ids, 0), torch.unsqueeze(token_type_ids, 0)

def sample_sequence(model, input_ids, token_type_ids, tokenizer, device, max_length = 15, temperature = 0.7, top_k = 0, top_p = 0.9, min_length = 1, video=None):
    current_output = []
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    input_ids, token_type_ids = masking_answer(input_ids, token_type_ids, tokenizer)
    input_ids = input_ids.to(device)
    token_type_ids = token_type_ids.to(device)
    
    for i in range(max_length):
        input_embs, token_ids = data_for_answer(input_ids, token_type_ids, current_output, tokenizer, device)
        input_embs = model.transformer.wte(input_embs)
        if video is not None:
            input_embs = torch.cat([model.video_ff(video), input_embs], dim=1)
            token_ids = torch.cat([torch.ones((1, video.size(1))).long().cuda() * tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-2]), token_ids], dim=1)
    

        logits = model(input_embs, token_type_ids = token_ids)[0]

        logits = logits[0, -1, :] / temperature
        logits = top_filtering(logits, top_k = top_k, top_p = top_p)
        probs = F.softmax(logits, dim=-1)
        prev = torch.multinomial(probs, 1)

        if i < min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                prev = torch.multinomial(probs, num_samples=1)
        
        if prev.item() in special_tokens_ids:
            break

        current_output.append(prev.item())

    return current_output

def beam_search(model, input_ids, token_type_ids, tokenizer, device, max_length=15, min_length=1, penalty = 0.3, beam_size = 5,  video=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    current_output = []
    hyplist =[([], 0., current_output)]
    best_state = None
    comp_hyplist = []

    input_ids, token_type_ids = masking_answer(input_ids, token_type_ids, tokenizer)
    input_ids = input_ids.to(device)
    token_type_ids = token_type_ids.to(device)

  
    for i in range(max_length):
        new_hyplist = []
        argmin = 0
        for out, lp, st in hyplist:
            input_embs, token_ids = data_for_answer(input_ids, token_type_ids, st, tokenizer, device)
            input_embs = model.transformer.wte(input_embs)
            if video is not None:
                input_embs = torch.cat([model.video_ff(video), input_embs], dim=1)
                token_ids = torch.cat([torch.ones((1, video.size(1))).long().cuda() * tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-2]), token_ids], dim=1)
        

            logits = model(input_embs, token_type_ids = token_ids)[0]

            logp = F.log_softmax(logits, dim=-1)[:, -1, :]
            lp_vec = logp.cpu().data.numpy() + lp
            lp_vec = np.squeeze(lp_vec)

            if i >= min_length:
                new_lp = lp_vec[tokenizer.eos_token_id] + penalty * (len(out) + 1)
                comp_hyplist.append((out, new_lp))
                if best_state is None or best_state < new_lp:
                    best_state = new_lp
            count = 1
            for o in np.argsort(lp_vec)[::-1]:
                if o == tokenizer.unk_token_id or o == tokenizer.eos_token_id:
                    continue
                new_lp = lp_vec[o]
                if len(new_hyplist) == beam_size:
                    if new_hyplist[argmin][1] < new_lp:
                        new_st = copy.deepcopy(st)
                        new_st.append(int(o))
                        new_hyplist[argmin] = (out + [o], new_lp, new_st)
                        argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0]
                    else:
                        break
                else:
                    new_st = copy.deepcopy(st)
                    new_st.append(int(o))
                    new_hyplist.append((out + [o], new_lp, new_st))
                    if len(new_hyplist) == beam_size:
                        argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0]
                count += 1
        hyplist = new_hyplist 
    if len(comp_hyplist) > 0:
        maxhyps = sorted(comp_hyplist, key=lambda h: -h[1])[:1]
        return maxhyps
    else:
        return [([], 0)]


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def load_pickle(fname):
    fname = Path(fname)
    with fname.open('rb') as handle:
        return pickle.load(handle)


def save_pickle(content, fname):
    fname = Path(fname)
    with fname.open('wb') as handle:
        pickle.dump(content, handle)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if n == 0:
            return
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
