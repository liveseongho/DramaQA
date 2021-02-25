from base import BaseDataLoader

from collections import defaultdict

import torch

from utils import *
from .preprocess_script_gpt2 import empty_sub, preprocess_text
from .preprocess_image import process_video
from .modules_language import get_tokenizer
from .data_loaders_bert import MultiModalData_BERT
import os
import numpy as np
from pathlib import Path

from itertools import chain
from torch.utils.data import Dataset
from transformers import *
# debug
from pprint import pprint

SPECIAL_TOKENS = ["<bos>", "<eos>", "<que>", "<ans>", "<speaker>", "<subtitle>", "<video>", "<pad>"]
SPECIAL_TOKENS_DICT = {'bos_token': "<bos>", 'eos_token': "<eos>", 'additional_special_tokens': ["<que>", "<ans>", "<speaker>", "<subtitle>", "<video>"], 'pad_token': "<pad>"}
#MODEL_INPUTS = ["input_ids", "token_type_ids","lm_labels"]
#PADDED_INPUTS = ["input_ids", "token_type_ids","lm_labels"]

eos_token = '<eos>'
pad_token = '<pad>'

speaker_name = [
    'None', # index 0: unknown speaker
    'Anna', 'Chairman', 'Deogi', 'Dokyung', 'Gitae',
    'Haeyoung1', 'Haeyoung2', 'Heeran', 'Hun', 'Jeongsuk',
    'Jinsang', 'Jiya', 'Kyungsu', 'Sangseok', 'Seohee',
    'Soontack', 'Sukyung', 'Sungjin', 'Taejin', 'Yijoon'
]
modes = ['train', 'val']#, 'test']

speaker_index = {name: index for index, name in enumerate(speaker_name)}
n_speakers = len(speaker_name)

# torch datatype
int_dtype = torch.long
float_dtype = torch.float

def tokenize(obj,tokenizer):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(o)) for n, o in obj.items())
    return list(tokenize(o) for o in obj)    


class TextData:
    def __init__(self, args, mode, tokenizer, split_tool, vocab=None):
        self.args = args
        self.tokenizer = tokenizer

#        self.vocab_path = args['vocab_path']
#        self.vocab_path = '/data/dataset/AnotherMissOh/vocab_gpt2.pickle'
        self.json_data_path = {m: self.get_data_path(args, mode=m, ext='.json') for m in modes}
        self.pickle_data_path = {m: self.get_data_path(args, mode=m, ext='.pickle') for m in modes}
        self.nc_data_path = {m: self.get_data_path(args, mode=m, ext='_nc.pickle') for m in modes}

        self.max_sen_len = args['max_word_per_sentence']

        # load vocab and preprocess dataset (words are converted into index)
#        if os.path.isfile(self.vocab_path):  # Use cached vocab if it exists.
#            print('Vocab exists!')
#            self.vocab = load_pickle(self.vocab_path)
#        else:  # There is no cached vocab. Build vocabulary and preprocess text data
#            print('There is no cached vocab.')
#            #self.tokenizer, _ = get_tokenizer(args)
#            self.vocab = build_word_vocabulary(self.args, self.tokenizer, self.json_data_path)
        if not self.args['remove_coreference']:
            preprocess_text(self.tokenizer, split_tool, self.json_data_path, self.pickle_data_path)
        else:
            preprocess_text(self.tokenizer, split_tool, self.json_data_path, self.nc_data_path)
    
        # load data
        if not self.args['remove_coreference']:
            print("Loading processed dataset from path: %s." % self.pickle_data_path[mode])
            self.data = load_pickle(self.pickle_data_path[mode])
        else:
            print("Loading processed dataset from path: %s." % self.nc_data_path[mode])
            self.data = load_pickle(self.nc_data_path[mode])

        ###### Special indices ######
        #self.tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
        bos, eos, que, ans, speaker, subtitle  = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-2]) 
        self.none_index = speaker_index['None']
        self.pad_index = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1])
        self.eos_index = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[1])

    def get_script(self, subtitle):
        spkr_of_sen_l = []  # list of speaker of subtitle sentences
        sub_in_sen_l = []   # list of subtitle sentences

        max_sentence = self.args['max_sentence_per_scene']
        if subtitle != empty_sub:  # subtitle exists
            subs = subtitle["contained_subs"]
            n_sentence = 1

            for s in subs:
                if n_sentence > max_sentence:
                    break
                n_sentence = n_sentence + 1
                spkr = s["speaker"]
                utter = s["utter"]
                spkr_of_sen_l.append(spkr)
                if len(utter) > self.max_sen_len:
                    del utter[self.max_sen_len:]
                    utter[-1] = self.eos_index
                if self.args['cc_spkr']:
                    utter = [spkr] + utter
                sub_in_sen_l.append(utter)
        else:  # No subtitle
            spkr_of_sen_l.append(self.none_index)  # add None speaker
            sub_in_sen_l.append([self.pad_index])  # add <pad>

        return spkr_of_sen_l, sub_in_sen_l


    def get_data_path(self, args, mode='train', ext='.json'):
        name = Path(args['data_path']).name.split('_')
        name.insert(1, mode)
        name = '_'.join(name)
        path = Path(args['data_path']).parent / name
        path = path.parent / (path.stem + ext)
        return path


class ImageData:
    def __init__(self, args, mode, vocab):
        self.args = args
        args['device'] = torch.device('cuda:0')

#        self.vocab = vocab
#        self.pad_index = vocab.get_index(pad_token)
#        self.none_index = speaker_index['None']
#        self.visual_pad = [self.none_index, self.pad_index, self.pad_index]

        self.image_path = args['image_path']
        self.image_dim = args['image_dim']

        self.processed_video_path = self.get_processed_video_path(self.image_path)
#        if not os.path.isfile(self.processed_video_path[mode]):
#            process_video(args, self.processed_video_path, speaker_index, vocab)

        print("Loading processed video input from path: %s." % self.processed_video_path[mode])
        self.image_dt = load_pickle(self.processed_video_path[mode])

    def get_processed_video_path(self, image_path):
        return {m: Path(image_path) / 'cache' / ('processed_video_' + m + '.pickle') for m in modes}

    def get_bbft(self, vid, flatten=False):
        bb_features = []
        visual_graphs = []

        shot = get_shot_id(vid)
        shot_contained = self.image_dt[vid[:19]] if shot == 0 else {vid[:24]: self.image_dt[vid[:19]][vid[:24]]}

        max_frame_per_shot = self.args['max_frame_per_shot']
        max_shot_per_scene = self.args['max_shot_per_scene']

        shot_num = 1
        for shot_vid, shot in shot_contained.items():
            if shot_num > max_shot_per_scene:
                break
            shot_num = shot_num + 1
            bb_feature = []
            vis_graph = []

            frame_num = 1
            # if len(shot.keys()) > max_frame_per_shot:
            #    np.random.uniform(0, len(shot.keys()), max_frame_per_shot)
            # TODO: frame sampling
            for frame_id, frame in shot.items():
                if frame_num > max_frame_per_shot:
                    break
                frame_num = frame_num + 1
                if self.args['remove_metadata']:
                    bb_feature.extend([frame['full_image']])
                else:
                    bb_feature.extend(frame['person_fulls'])
                vis_graph.extend(frame['persons'])

            if not bb_feature:
#                vis_graph = self.visual_pad
                bb_feature = [np.zeros(self.image_dim)]
            bb_feature = np.reshape(np.concatenate(bb_feature), (-1))
#            vis_graph = np.reshape(vis_graph, (-1))
            bb_features.append(bb_feature)
#            visual_graphs.append(vis_graph)

        return bb_features#, visual_graphs


class MultiModalData_GPT2(Dataset):
    def __init__(self, args, mode):
        assert mode in modes, "mode should be %s." % (' or '.join(modes))

        self.tokenizer_class = GPT2Tokenizer
        self.tokenizer = self.tokenizer_class.from_pretrained('gpt2')
        self.tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
        self.pad_token = self.tokenizer.pad_token_id

        self.args = args
        self.mode = mode

        ###### Text ######
        split_tool, _ = get_tokenizer(args)
        text_data = TextData(args, mode, self.tokenizer, split_tool)
        self.text = text_data.data
        self.get_script = text_data.get_script
#        self.vocab = text_data.vocab

        ###### Image #####
#        image_data = ImageData(args, mode, self.vocab)
#        self.image = image_data
#        self.image_dim = image_data.image_dim

        ###### Constraints ######
        self.max_sub_len = args['max_sub_len']
        self.max_image_len = args['max_image_len']

        ###### Special indices ######
#        self.none_index = speaker_index['None']
#        self.pad_index = self.vocab.stoi.get(pad_token)
#        self.eos_index = self.vocab.stoi.get(eos_token)

        self.inputs = args['inputs']

    def __len__(self):
        return len(self.text)

    def process_text(self, idx):
        text = self.text[idx]
        qid = text['qid']
        que = text['que']
        answer = text['answer']
        subtitle = text['subtitle']
        correct_idx = text['correct_idx'] if self.mode != 'test' else None
        q_level_logic = text['q_level_logic']
        bos, eos = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:2])
        str_ans = text['str_answer']
        change_str = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(str_ans))
        convert_str = self.tokenizer.convert_ids_to_tokens(answer[0])
        final = self.tokenizer.convert_tokens_to_string(convert_str)

        data = {
            'que': que,
            'ans': answer,
            'q_level_logic': q_level_logic,
            'qid': qid,
            'str_ans' : change_str
        }

        script_types = ['sentence', 'word']
        assert self.args['script_type'] in script_types, "scrtip_type should be %s." % (' or '.join(script_types))

        spkr, script = self.get_script(subtitle)
        if self.args['script_type'] == 'word':
            # Concatenate subtitle sentences
            sub_in_word_l = []
            spkr_of_word_l = []
            max_sub_len = self.max_sub_len
            n_words = 0
            for spkr, s in zip(spkr, script):
                sen_len = len(s)

                n_words += sen_len
                sub_in_word_l.extend([spkr])

                sub_in_word_l.extend(s)
                spkr_of_word_l.extend(spkr for i in range(sen_len))  # 1:1 correspondence between word and speaker

                if n_words > max_sub_len:
                    del sub_in_word_l[max_sub_len:], spkr_of_word_l[max_sub_len:]
                    sub_in_word_l[-1] = self.eos_index

                    break
            script = sub_in_word_l
            spkr = spkr_of_word_l
            # spkr = np.concatenate(spkr, axis=0)
            # script = np.concatenate(script, axis=0)

        data['spkr'] = spkr
        data['script'] = script

        return data

    
    def process_image(idx, data):
#        text = self.text[idx]
#        vid = text['vid']
#        visual_types = ['shot', 'frame']
#        assert self.args['visual_type'] in visual_types, "visual_typoe should be %s." % (' or '.join(visual_types))
#
#        vfeatures, vmetas = self.image.get_bbft(vid)
#
#        if self.args['visual_type'] == 'frame':
#            # vfeatures: [(num_frames*512), (num_frames*512), ...]
#            # vmetas: [(num_frames*3*512), ...]
#            vfeatures = np.concatenate(vfeatures, axis=0)
#            vmetas = np.concatenate(vmetas, axis=0)
#
#        data['bbfts'] = vfeatures
#        data['vgraphs'] = vmetas
        return data

    def __getitem__(self, idx):

        data = self.process_text(idx)
        
        input_ids, token_type_ids, lm_labels = data_for_gpt(data, self.tokenizer)

#        data = process_image(idx, data)
        
        # currently not tensor yet
        return input_ids, token_type_ids, lm_labels, data['str_ans']

    def padding(self, seq, pad_token):
        max_len = max([i.size(0) for i in seq])
        if len(seq[0].size()) == 1:
            result = torch.ones((len(seq), max_len)).long() * pad_token
        else:
            result = torch.ones((len(seq), max_len, seq[0].size(-1))).float()
        for i in range(len(seq)):
            result[i, :seq[i].size(0)] = seq[i]
        return result

    # data padding
    def collate_fn(self, batch):
        collected = defaultdict(list)
        input_ids_list, token_type_ids_list, lm_labels_list, answer_list = [], [], [], []
        for data in batch:
            input_ids_list.append(data[0])
            token_type_ids_list.append(data[1])
            lm_labels_list.append(data[2])
            answer_list.append(data[3])
         
        input_ids = self.padding(input_ids_list, self.pad_token) 
        token_type_ids = self.padding(token_type_ids_list, self.pad_token)
        lm_labels = self.padding(lm_labels_list, -1)
#        input_mask = input_ids != self.pad_token
        
        return input_ids, token_type_ids, lm_labels, answer_list#, input_mask

