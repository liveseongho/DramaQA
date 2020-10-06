import re
import numpy as np
import torch
import json
from datetime import datetime


def batch_to_device(args, batch, device):
    net_input_key = [*args]
    net_input = {k: batch[k] for k in net_input_key}
    for key, value in net_input.items():
        if torch.is_tensor(value):
            net_input[key] = value.to(device).contiguous()

    ans_idx = batch.get('correct_idx', None)
    if torch.is_tensor(ans_idx):
        ans_idx = ans_idx.to(device).contiguous()

    return net_input, ans_idx


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



def get_episode_id(vid):
    return int(vid[13:15]) # vid format: AnotherMissOh00_000_0000


def get_scene_id(vid):
    return int(vid[16:19]) # vid format: AnotherMissOh00_000_0000


def get_shot_id(vid):
    return int(vid[20:24]) # vid format: AnotherMissOh00_000_0000


frame_id_re = re.compile('IMAGE_(\d+)')
def get_frame_id(img_file_name):
    return int(frame_id_re.search(img_file_name).group(1)) # img_file_name format: IMAGE_0000070227