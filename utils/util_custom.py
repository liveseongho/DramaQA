import re
import numpy as np
import torch
import json
from datetime import datetime
import torch.nn.functional as F


SPECIAL_TOKENS = ["<bos>", "<eos>", "<desc>", "<speaker>", "<subtitle>",
                  "<bounding_feature>", "<person>", "<behavior>", "<emotion>", "<video>", "<pad>"]

def batch_to_device(args, batch, device):
    # net_input_key = [*args]
    # net_input = {k: batch[k] for k in net_input_key}
    for key, value in batch.items():
        if torch.is_tensor(value):
            batch[key] = value.to(device).contiguous()
        elif type(value) is dict:
            for k, v in value.items():
                batch[key][k] = torch.tensor(v, dtype=torch.long).to(device).contiguous()
        elif type(value) is list:
            if type(value[0]) is dict:
                for i, data in enumerate(value):
                    for k, v in data.items():
                        batch[key][i][k] = torch.tensor(v, dtype=torch.long).to(device).contiguous()
            elif len(value) == 5:
                for i, data in enumerate(value):
                    batch[key][i] = torch.tensor(data, dtype=torch.long).to(device).contiguous()


    ans_idx = batch.get('correct_idx', None)
    if torch.is_tensor(ans_idx):
        ans_idx = ans_idx.to(device).contiguous()

    return batch, ans_idx


def get_now():
    now = datetime.now()
    return now.strftime('%Y-%m-%d-%H-%M-%S')


def make_jsonl(path, overwrite=False):
    if (overwrite or not path.is_file()) and path.suffix == '.jsonl':
        path_json = path.parent / path.name[:-1]
        with open(str(path_json), 'r') as f:
            li = json.load(f)
        with open(str(path), 'w') as f:
            for line in li:
                f.write("{}\n".format(json.dumps(line)))


def to_string(vocab, x):
    # x: float tensor of size BSZ X LEN X VOCAB_SIZE
    # or idx tensor of size BSZ X LEN
    if x.dim() > 2:
        x = x.argmax(dim=-1)

    res = []
    for i in range(x.shape[0]):
        sent = x[i]
        li = []
        for j in sent:
            if j not in vocab.special_ids:
                li.append(vocab.itos[j])
        sent = ' '.join(li)
        res.append(sent)

    return res


def get_max_size(t):
    if hasattr(t, 'shape'):
        if not torch.is_tensor(t):
            t = torch.from_numpy(t)
        return list(t.shape), t.dtype
    else:
        # get max
        t = [get_max_size(i) for i in t]
        dtype = t[0][1]
        t = [i[0] for i in t]
        return [len(t), *list(np.array(t).max(axis=0))], dtype


def pad_tensor(x, val=0):
    max_size, dtype = get_max_size(x)
    storage = torch.full(max_size, val).type(dtype)

    def add_data(ids, t):
        if hasattr(t, 'shape'):
            if not torch.is_tensor(t):
                t = torch.from_numpy(t)
            storage[tuple(ids)] = t
        else:
            for i in range(len(t)):
                add_data([*ids, i], t[i])

    add_data([], x)

    return storage


def pad2d(data, pad_val, dtype, reshape3d=False, last_dim=0):
    '''
        data: list of sequence
        return:
            p_data: (batch_size, max_length_of_sequence)
            p_length: (batch_size)
    '''
    batch_size = len(data)
    length = [len(row) for row in data]
    max_length = max(length)
    shape = (batch_size, max_length)
    p_length = torch.tensor(length, dtype=torch.long)  # no need to pad

    if isinstance(pad_val, list):
        p_data = torch.tensor(pad_val, dtype=dtype)
        p_data = p_data.repeat(batch_size, max_length // len(pad_val))
    else:
        p_data = torch.full(shape, pad_val, dtype=dtype)

    for i in range(batch_size):
        d = torch.tensor(data[i], dtype=dtype)
        p_data[i, :len(d)] = d

    if reshape3d:
        p_data = p_data.view(batch_size, -1, last_dim)
        p_length = p_length / last_dim

    return p_data, p_length


def pad3d(data, pad_val, dtype, reshape4d=False, last_dim=0):
    '''
        data: list of list of sequence
        return:
            p_data: (batch_size, max_length_of_list_of_seq, max_length_of_sequence)
            p_dim1_length: (batch_size)
            p_dim2_length: (batch_size, max_length_of_list_of_seq)
    '''
    batch_size = len(data)
    dim2_length = [[len(dim2) for dim2 in dim1] for dim1 in data]
    max_dim1_length = max(len(dim1) for dim1 in data)
    max_dim2_length = max(l for row in dim2_length for l in row)
    data_shape = (batch_size, max_dim1_length, max_dim2_length)
    p_dim2_length, p_dim1_length = pad2d(dim2_length, 0, torch.long)

    if isinstance(pad_val, list):
        p_data = torch.tensor(pad_val, dtype=dtype)
        p_data = p_data.repeat(batch_size, max_dim1_length, max_dim2_length // len(pad_val))
    else:
        p_data = torch.full(data_shape, pad_val, dtype=dtype)

    for i in range(batch_size):
        row = data[i]
        for j in range(len(row)):
            if None in row[j]:

                for idx, d in enumerate(row[j]):
                    if d is None:
                        row[j][idx] = 0
                row[j] = row[j].astype(np.int64)

            d = torch.tensor(row[j], dtype=dtype)
            p_data[i, j, :len(d)] = d

    if reshape4d:
        p_data = p_data.view(batch_size, max_dim1_length, -1, last_dim)
        p_dim2_length = p_dim2_length / last_dim

    return p_data, p_dim1_length, p_dim2_length


def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'),
                  filter_value=-float('Inf')):
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

def data_for_answer(input_ids, token_type_ids, answer, tokenizer, device):
    bos, eos, desc  = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:3])
    if len(answer) == 0 :
        return input_ids, token_type_ids
    sequence_ans = answer
    tokens_type_ans = [desc] * len(sequence_ans)

    result_input_ids = torch.Tensor([input_ids.tolist()[0] + sequence_ans]).long()
    result_token_type_ids = torch.Tensor([token_type_ids.tolist()[0] + tokens_type_ans]).long()
    result_input_ids = result_input_ids.to(device)
    result_token_type_ids = result_token_type_ids.to(device)

    return result_input_ids, result_token_type_ids


def masking_answer(input_ids, token_type_ids, tokenizer):
    desc = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[2])
    return_input_ids, return_token_type_ids = [], []
    input_ids = torch.squeeze(input_ids, 0)
    token_type_ids = torch.squeeze(token_type_ids, 0)
    for input_id, token_type_id in zip(input_ids, token_type_ids):
        if token_type_id.item() != desc:
            return_input_ids.append(input_id)
            return_token_type_ids.append(token_type_id)

    return_input_ids.append(desc)
    return_token_type_ids.append(desc)
    input_ids = torch.Tensor(return_input_ids).long()
    token_type_ids = torch.Tensor(return_token_type_ids).long()

    return torch.unsqueeze(input_ids, 0), torch.unsqueeze(token_type_ids, 0)


def sample_sequence(model, input_ids, token_type_ids, tokenizer, device,
                    max_length = 30, temperature = 0.7, top_k = 8, top_p = 0.9,
                    min_length = 1, video=None, bbfts = None):
    current_output = []
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    input_ids, token_type_ids = masking_answer(input_ids, token_type_ids, tokenizer)
    input_ids = input_ids.to(device)
    token_type_ids = token_type_ids.to(device)

    for i in range(max_length):
        input_embs, token_ids = data_for_answer(input_ids, token_type_ids, current_output, tokenizer, device)
        input_embs = model.transformer.wte(input_embs)
        if bbfts is not None:
            input_embs = torch.cat([bbfts, input_embs], dim = 1)
            token_ids = torch.cat([torch.ones((bbfts.size(0), bbfts.size(1))).long().cuda() * tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[5]), token_ids], dim=1)
        if video is not None:
            input_embs = torch.cat([model.video_ff(video), input_embs], dim=1)
            token_ids = torch.cat([torch.ones((video.size(0), video.size(1))).long().cuda() * tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-2]), token_ids], dim=1)

        logits = model(input_embs, token_type_ids = token_ids)[0]
        logits = logits[0, -1, :] / temperature
        logits = top_filtering(logits, top_k = top_k, top_p = top_p)
        probs = F.softmax(logits, dim=-1)
        prev = torch.multinomial(probs, 1)

        if i < min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                print(special_tokens_ids)
                print(prev.item())
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break

        current_output.append(prev.item())

    return current_output

def get_episode_id(vid):
    return int(vid[13:15])  # vid format: AnotherMissOh00_000_0000


def get_scene_id(vid):
    return int(vid[16:19])  # vid format: AnotherMissOh00_000_0000


def get_shot_id(vid):
    return int(vid[20:24])  # vid format: AnotherMissOh00_000_0000


frame_id_re = re.compile('IMAGE_(\d+)')
def get_frame_id(img_file_name):
    return int(frame_id_re.search(img_file_name).group(1)) # img_file_name format: IMAGE_0000070227
