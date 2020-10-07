from base import BaseDataLoader

from collections import defaultdict

import torch

from utils import *
from .preprocess_script import merge_qa_subtitle, empty_sub, build_word_vocabulary, preprocess_text
from .preprocess_image import preprocess_images, process_video
from .modules_language import get_tokenizer
import os
from tqdm import tqdm
import numpy as np
from pathlib import Path

from torch.utils.data import Dataset

# debug
from pprint import pprint

sos_token = '<sos>'
eos_token = '<eos>'
pad_token = '<pad>'
unk_token = '<unk>'

speaker_name = [
    'None', # index 0: unknown speaker
    'Anna', 'Chairman', 'Deogi', 'Dokyung', 'Gitae',
    'Haeyoung1', 'Haeyoung2', 'Heeran', 'Hun', 'Jeongsuk',
    'Jinsang', 'Jiya', 'Kyungsu', 'Sangseok', 'Seohee',
    'Soontack', 'Sukyung', 'Sungjin', 'Taejin', 'Yijoon'
]
modes = ['train', 'val', 'test']

speaker_index = {name: index for index, name in enumerate(speaker_name)}
n_speakers = len(speaker_name)

# torch datatype
int_dtype = torch.long
float_dtype = torch.float


class ImageData:
    def __init__(self, args, mode, vocab):
        self.args = args
        args['device'] = torch.device('cuda:0')

        self.vocab = vocab
        self.pad_index = vocab.get_index(pad_token)
        self.none_index = speaker_index['None']
        self.visual_pad = [self.none_index, self.pad_index, self.pad_index]

        self.image_path = args['image_path']
        self.image_dim = args['image_dim']

        self.processed_video_path = self.get_processed_video_path(self.image_path)
        if not os.path.isfile(self.processed_video_path[mode]):
            process_video(args, self.processed_video_path, speaker_index, vocab)

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
            #if len(shot.keys()) > max_frame_per_shot:
            #    np.random.uniform(0, len(shot.keys()), max_frame_per_shot)

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
                vis_graph = self.visual_pad
                bb_feature = [np.zeros(self.image_dim)]
            bb_feature = np.reshape(np.concatenate(bb_feature), (-1))
            vis_graph = np.reshape(vis_graph, (-1))
            bb_features.append(bb_feature)
            visual_graphs.append(vis_graph)

        return bb_features, visual_graphs


class TextData:
    def __init__(self, args, mode, vocab=None):
        self.args = args

        self.vocab_path = args['vocab_path']

        self.json_data_path = {m: self.get_data_path(args, mode=m, ext='.json') for m in modes}
        self.pickle_data_path = {m: self.get_data_path(args, mode=m, ext='.pickle') for m in modes}
        self.nc_data_path = {m: self.get_data_path(args, mode=m, ext='_nc.pickle') for m in modes}
        self.bert_data_path = {m: self.get_data_path(args, mode=m, ext='_bert.pickle') for m in modes}

        if args['bert']:
            print('BERT mode ON')
            if not os.path.isfile(self.bert_data_path[mode]):
                self.tokenizer, self.vocab = get_tokenizer(args, self.special_tokens)
                self.preprocess_text(self.vocab, self.tokenizer, self.bert_data_path)
            self.data = load_pickle(self.bert_data_path[mode])
        else:
            if os.path.isfile(self.vocab_path): # Use cached vocab if it exists.
                print('Vocab exists!')
                self.vocab = load_pickle(self.vocab_path)
            else: # There is no cached vocab. Build vocabulary and preprocess text data
                print('There is no cached vocab.')
                self.tokenizer = get_tokenizer(args)
                self.vocab = build_word_vocabulary(self.args, self.tokenizer, self.json_data_path)
                if not self.args['remove_coreference']:
                    preprocess_text(self.vocab, self.tokenizer, self.json_data_path, self.pickle_data_path)
                else:
                    preprocess_text(self.vocab, self.tokenizer, self.json_data_path, self.nc_data_path)

            if not self.args['remove_coreference']:
                print("Loading processed dataset from path: %s." % self.pickle_data_path[mode])
                self.data = load_pickle(self.pickle_data_path[mode])
            else:
                print("Loading processed dataset from path: %s." % self.nc_data_path[mode])
                self.data = load_pickle(self.nc_data_path[mode])

    def get_data_path(self, args, mode='train', ext='.json'):
        name = Path(args['data_path']).name.split('_')
        name.insert(1, mode)
        name = '_'.join(name)
        path = Path(args['data_path']).parent / name
        path = path.parent / (path.stem + ext)

        return path

class MultiModalData(Dataset):
    def __init__(self, args, mode):
        assert mode in modes, "mode should be %s." % (' or '.join(modes))

        self.args = args
        self.mode = mode

        ###### Text ######
        text_data = TextData(args, mode)
        self.text = text_data.data
        self.vocab = text_data.vocab

        ###### Image #####
        image_data = ImageData(args, mode, self.vocab)
        self.image = image_data
        self.image_dim = image_data.image_dim

        ###### Constraints ######
        self.max_sen_len = args['max_word_per_sentence']
        self.max_sub_len = args['max_sub_len']
        self.max_image_len = args['max_image_len']

        ###### Special indices ######
        self.none_index = speaker_index['None']
        self.pad_index = self.vocab.stoi.get(pad_token)
        self.eos_index = self.vocab.stoi.get(eos_token)

        self.inputs = args['inputs']

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.text[idx]
        qid = text['qid']
        que = text['que']
        ans = text['answers']
        subtitle = text['subtitle']
        correct_idx = text['correct_idx'] if self.mode != 'test' else None
        q_level_logic = text['q_level_logic']

        vid = text['vid']

        data = {
            'que': que,
            'ans': ans,
            'correct_idx': correct_idx,

            'q_level_logic': q_level_logic,
            'qid': qid
        }

        script_types = ['sentence', 'word']
        assert self.args['script_type'] in script_types, "scrtip_type should be %s." % (' or '.join(script_types))

        spkr_of_sen_l = []  # list of speaker of subtitle sentences
        sub_in_sen_l = []   # list of subtitle sentences
        if self.args['script_type'] == 'sentence':
            max_sentence = self.args['max_sentence_per_scene']
            if subtitle != empty_sub: # subtitle exists
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
                    # utter = [spkr]+utter ## tvqa
                    sub_in_sen_l.append(utter)

            else: # No subtitle
                spkr_of_sen_l.append(self.none_index) # add None speaker
                sub_in_sen_l.append([self.pad_index]) # add <pad>

            data['sub_in_sen'] = sub_in_sen_l
            data['spkr_of_sen'] = spkr_of_sen_l

        elif self.args['script_type'] == 'word':
            # Concatenate subtitle sentences
            sub_in_word_l = []; spkr_of_word_l = []
            max_sub_len = self.max_sub_len
            n_words = 0
            for spkr, s in zip(spkr_of_sen_l, sub_in_sen_l):
                sen_len = len(s)
                #n_words += sen_len+1 #### tvqa

                sub_in_word_l.extend(s)
                #sub_in_word_l.extend([spkr]) ### tvqa
                spkr_of_word_l.extend(spkr for i in range(sen_len)) # 1:1 correspondence between word and speaker

                if n_words > max_sub_len:
                    del sub_in_word_l[max_sub_len:], spkr_of_word_l[max_sub_len:]
                    sub_in_word_l[-1] = self.eos_index

                    break

            data['sub_in_word'] = sub_in_word_l
            data['spkr_of_word'] = spkr_of_word_l

        visual_types = ['shot', 'frame']
        assert self.args['visual_type'] in visual_types, "visual_typoe should be %s." % (' or '.join(visual_types))

        vfeatures, vmetas = self.image.get_bbft(vid)

        if self.args['visual_type'] == 'frame':
            # vfeatures: [(num_frames*512), (num_frames*512), ...]
            # vmetas: [(num_frames*3*512), ...]
            vfeatures = np.concatenate(vfeatures, axis=0)
            vmetas = np.concatenate(vmetas, axis=0)

        data['bbfts'] = vfeatures
        data['vgraphs'] = vmetas

        # currently not tensor yet
        return data

    # data padding
    def collate_fn(self, batch):
        collected = defaultdict(list)
        for data in batch:
            for key, value in data.items():
                collected[key].append(value)
        que, que_l = self.pad2d(collected['que'], self.pad_index, int_dtype)
        ans, _, ans_l = self.pad3d(collected['ans'], self.pad_index, int_dtype)
        qa_concat = [[collected['que'][j]+collected['ans'][j][i] for i in range(5)] for j in range(len(collected['que']))]
        qa_concat, _, qa_concat_l = self.pad3d(qa_concat, self.pad_index, int_dtype)
        correct_idx = torch.tensor(collected['correct_idx'], dtype=int_dtype) if self.mode != 'test' else None # correct_idx does not have to be padded

        data = {
            'que': que, 'que_l': que_l,
            'ans': ans, 'ans_l': ans_l,
            'qa': qa_concat, 'qa_l': qa_concat_l,
            'q_level_logic': collected['q_level_logic'],
            'qid': collected['qid']
        }

        if correct_idx is not None:
            data['correct_idx'] = correct_idx

        if self.args['script_type'] == 'word':
            spkr_of_w, _ = self.pad2d(collected['spkr_of_word'], self.none_index, int_dtype)
            sub_in_w, sub_in_w_l = self.pad2d(collected['sub_in_word'], self.pad_index, int_dtype)
            data['filtered_sub'] = sub_in_w
            data['filtered_sub_len'] = sub_in_w_l
        elif self.args['script_type'] == 'sentence':
            spkr_of_s, _ = self.pad2d(collected['spkr_of_sen'], self.none_index, int_dtype)
            sub_in_s, sub_in_s_l, sub_s_l = self.pad3d(collected['sub_in_sen'], self.pad_index, int_dtype)
            data['sub'] = sub_in_s
            data['sub_l'] = sub_in_s_l
            data['sub_l_l'] = sub_s_l
            data['spkr'] = spkr_of_s

        if self.args['visual_type'] == 'frame':
            bbfts, bbfts_l = self.pad2d(collected['bbfts'], 0, float_dtype, reshape3d=True, last_dim=self.image_dim)
            bbfts_l_l = None
            vgraphs, vgraphs_l = self.pad2d(collected['vgraphs'], self.image.visual_pad, int_dtype)
            vgraphs_l_l = None
            data['bbfts'] = bbfts
            data['bbfts_l'] = bbfts_l
            data['bbfts_l_l'] = bbfts_l_l
            data['vmeta'] = vgraphs
        elif self.args['visual_type'] == 'shot':
            bbfts, bbfts_l, bbfts_l_l = self.pad3d(collected['bbfts'], 0, float_dtype, reshape4d=True, last_dim=self.image_dim)
            vgraphs, vgraphs_l, vgraphs_l_l = self.pad3d(collected['vgraphs'], self.image.visual_pad, int_dtype, reshape4d=True, last_dim=3)

            data['bbfts'] = bbfts
            data['bbfts_l'] = bbfts_l
            data['bbfts_l_l'] = bbfts_l_l
            data['vmeta'] = vgraphs

        # currently not in the device yet
        return data

    def pad2d(self, data, pad_val, dtype, reshape3d=False, last_dim=0):
        batch_size = len(data)
        length = [len(row) for row in data]
        max_length = max(length)
        shape = (batch_size, max_length)
        p_length = torch.tensor(length, dtype=int_dtype) # no need to pad

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
            p_length = p_length/last_dim

        return p_data, p_length

    def pad3d(self, data, pad_val, dtype, reshape4d=False, last_dim=0):
        batch_size = len(data)
        dim2_length = [[len(dim2) for dim2 in dim1] for dim1 in data]
        max_dim1_length = max(len(dim1) for dim1 in data)
        max_dim2_length = max(l for row in dim2_length for l in row)
        data_shape = (batch_size, max_dim1_length, max_dim2_length)
        p_dim2_length, p_dim1_length = self.pad2d(dim2_length, 0, int_dtype)

        if isinstance(pad_val, list):
            p_data = torch.tensor(pad_val, dtype=dtype)
            p_data = p_data.repeat(batch_size, max_dim1_length, max_dim2_length // len(pad_val))
        else:
            p_data = torch.full(data_shape, pad_val, dtype=dtype)

        for i in range(batch_size):
            row = data[i]
            for j in range(len(row)):
                d = torch.tensor(row[j], dtype=dtype)
                p_data[i, j, :len(d)] = d

        if reshape4d:
            p_data = p_data.view(batch_size, max_dim1_length, -1, last_dim)
            p_dim2_length = p_dim2_length/last_dim

        return p_data, p_dim1_length, p_dim2_length


class DramaQADataLoader(BaseDataLoader):
    def __init__(self, mode, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, vocab=None, **kwargs):
        dataset = MultiModalData(kwargs, mode=mode)
        self.vocab = dataset.vocab
        super().__init__(dataset, batch_size, shuffle, validation_split, num_workers, dataset.collate_fn)
