from base import BaseDataLoader

from collections import defaultdict

import torch

from utils import *
from .preprocess_script import empty_sub, build_word_vocabulary, preprocess_text
from .preprocess_image import process_video
from .modules_language import get_tokenizer
import os
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


class TextData:
    def __init__(self, args, mode, vocab=None):
        self.args = args

        self.vocab_path = args['vocab_path']

        self.json_data_path = {m: self.get_data_path(args, mode=m, ext='.json') for m in modes}
        self.pickle_data_path = {m: self.get_data_path(args, mode=m, ext='.pickle') for m in modes}
        self.nc_data_path = {m: self.get_data_path(args, mode=m, ext='_nc.pickle') for m in modes}

        self.max_sen_len = args['max_word_per_sentence']

        # load vocab and preprocess dataset (words are converted into index)
        print('BERT mode ON')
        self.bert_data_path = {m: self.get_data_path(args, mode=m, ext='_bert.pickle') for m in modes}

        self.tokenizer, self.vocab = get_tokenizer(args)
        if not os.path.isfile(self.bert_data_path[mode]):
            self.preprocess_text(self.vocab, self.tokenizer, self.bert_data_path)

        # load data
        print("Loading processed dataset from path: %s." % self.bert_data_path[mode])
        self.data = load_pickle(self.bert_data_path[mode])

        ###### Special indices ######
        self.none_index = speaker_index['None']
        self.pad_index = self.vocab.stoi.get(pad_token)
        self.eos_index = self.vocab.stoi.get(eos_token)

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
                vis_graph = self.visual_pad
                bb_feature = [np.zeros(self.image_dim)]
            bb_feature = np.reshape(np.concatenate(bb_feature), (-1))
            vis_graph = np.reshape(vis_graph, (-1))
            bb_features.append(bb_feature)
            visual_graphs.append(vis_graph)

        return bb_features, visual_graphs


class MultiModalData_BERT(Dataset):
    def __init__(self, args, mode):
        assert mode in modes, "mode should be %s." % (' or '.join(modes))

        self.args = args
        self.mode = mode

        ###### Text ######
        text_data = TextData(args, mode)
        self.text = text_data.data
        self.get_script = text_data.get_script
        self.vocab = text_data.vocab

        ###### Image #####
        image_data = ImageData(args, mode, self.vocab)
        self.image = image_data
        self.image_dim = image_data.image_dim

        ###### Constraints ######
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
        correct_idx = torch.tensor(collected['correct_idx'], dtype=int_dtype) if self.mode != 'test' else None

        data = {
            'correct_idx': correct_idx,
            'q_level_logic': collected['q_level_logic'],
            'qid': collected['qid']
        }

        if self.args['cc_qa']:
            qa_concat = [[collected['que'][j] + collected['ans'][j][i] for i in range(5)] for j in range(len(collected['que']))]
            qa_concat, _, qa_concat_l = pad3d(qa_concat, self.pad_index, int_dtype)
            data['qa'] = qa_concat
            data['qa_l'] = qa_concat_l
        else:
            que, que_l = pad2d(collected['que'], self.pad_index, int_dtype)
            ans, _, ans_l = pad3d(collected['ans'], self.pad_index, int_dtype)
            data['que'] = que
            data['que_l'] = que_l
            data['ans'] = ans
            data['ans_l'] = ans_l

        if self.args['script_type'] == 'word':
            spkr, spkr_l = pad2d(collected['spkr'], self.none_index, int_dtype)
            sub, sub_l = pad2d(collected['script'], self.pad_index, int_dtype)
            sub_l_l = None
        elif self.args['script_type'] == 'sentence':
            spkr, spkr_l = pad2d(collected['spkr'], self.none_index, int_dtype)
            sub, sub_l, sub_l_l = pad3d(collected['script'], self.pad_index, int_dtype)

        data['spkr'] = spkr
        data['sub'] = sub
        data['sub_l'] = sub_l
        data['sub_l_l'] = sub_l_l

        if self.args['visual_type'] == 'frame':
            bbfts, bbfts_l = pad2d(collected['bbfts'], 0, float_dtype, reshape3d=True, last_dim=self.image_dim)
            bbfts_l_l = None
            vgraphs, vgraphs_l = pad2d(collected['vgraphs'], self.image.visual_pad, int_dtype)
        elif self.args['visual_type'] == 'shot':
            bbfts, bbfts_l, bbfts_l_l = pad3d(collected['bbfts'], 0, float_dtype, reshape4d=True, last_dim=self.image_dim)
            vgraphs, vgraphs_l, _ = pad3d(collected['vgraphs'], self.image.visual_pad, int_dtype, reshape4d=True, last_dim=3)

        data['bbfts'] = bbfts
        data['bbfts_l'] = bbfts_l
        data['bbfts_l_l'] = bbfts_l_l
        data['vmeta'] = vgraphs

        # currently not in the device yet