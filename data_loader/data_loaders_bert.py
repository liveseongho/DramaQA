from collections import defaultdict

import torch

from utils import *
from .preprocess_script import empty_sub, build_word_vocabulary, preprocess_text
from .preprocess_image import preprocess_images, load_visual
from os.path import isfile

from .modules_language import get_tokenizer
import os
import numpy as np
from pathlib import Path

from torch.utils.data import Dataset

# debug
from pprint import pprint

sos_token = '[SOS]'
eos_token = '[EOS]'
pad_token = '[PAD]'
unk_token = '[UNK]'

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


        # load vocab and preprocess dataset (words are converted into index)
        print('BERT mode ON')

        self.bert_data_path = {m: self.get_data_path(args, mode=m, ext='_bert.pickle') for m in modes}

        self.tokenizer, self.vocab = get_tokenizer(args)

        '''
        if not os.path.isfile(self.bert_data_path[mode]):
            preprocess_text(self.vocab, self.tokenizer, self.json_data_path, self.bert_data_path)

        # load data
        print("Loading dataset from path: %s." % self.bert_data_path[mode])
        self.data = load_pickle(self.bert_data_path[mode])
        '''
        self.data = read_json(self.json_data_path[mode])

        ###### Special indices ######
        self.none_index = self.vocab.get("None")
        self.pad_index = self.vocab.get(pad_token)
        self.eos_index = self.vocab.get(eos_token)

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
                spkr = self.vocab.get(s["speaker"])
                utter = s["utter"]
                spkr_of_sen_l.append(spkr)
                sub_in_sen_l.append(utter)
        else:  # No subtitle
            spkr_of_sen_l.append(self.vocab.get("None"))  # add None speaker
            sub_in_sen_l.append(pad_token)  # add <pad>

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
        self.pad_index = vocab.get(pad_token)
        self.none_index = vocab.get('None')
        self.visual_pad = [self.none_index, self.pad_index, self.pad_index, self.pad_index, self.pad_index]

        self.image_path = args['image_path']
        self.image_cache_path = args['image_cache_path']
        self.image_dim = args['image_dim']
        self.image_feature_path = args['image_feature']

        self.processed_video_path = self.get_processed_video_path(self.image_cache_path)
        if not os.path.isfile(self.processed_video_path[mode]):
            self.process_video(args, self.processed_video_path, self.image_feature_path, speaker_index, vocab)

        print("Loading processed video input from path: %s." % self.processed_video_path[mode])
        self.image_dt = load_pickle(self.processed_video_path[mode])

    def get_processed_video_path(self, image_path):
        return {m: Path(image_path) / ('processed_video_bert_' + m + '.pickle') for m in modes}


    def process_video(self, args, save_path, feature_path, speaker_index, vocab):
        pad_index = vocab.get(pad_token)
        none_index = speaker_index['None']

        visuals = load_visual(args)
        if isfile(feature_path):
            features = load_pickle(feature_path)
        else:
            print("saving processed_video ...")
            features = preprocess_images(args, visuals)
            save_pickle(features, feature_path)

        """
        {
            full_image:   full_image (tensor of shape (512,)),
            person_list:      [[person1_id_idx, behavior1_idx, emotion1_idx], ...],
            person_fulls: [person_full1 (tensor of shape (512,)), ... ]
        }
        """
        full_images = features['full_image']
        person_fulls = features['person_full']

        new_visuals = defaultdict(dict)
        for i in range(1, 19):
            for key, value in visuals[i].items():
                frame_id = value['frame_id']
                scene_id = frame_id[:19]
                vid = frame_id[:24]
                f = get_frame_id(frame_id)


                if scene_id in new_visuals:
                    if vid in new_visuals[scene_id]:
                        new_visuals[scene_id][vid][f] = value
                    else:
                        new_visuals[scene_id][vid] = defaultdict(dict)
                        new_visuals[scene_id][vid][f] = value
                else:
                    new_visuals[scene_id] = defaultdict(dict)
                    new_visuals[scene_id][vid] = defaultdict(dict)
                    new_visuals[scene_id][vid][f] = value
        visuals = new_visuals

        for scene_vid, vid_dict in full_images.items():
            for vid, frames in vid_dict.items():
                for key, value in frames.items():
                    frames[key] = {
                        'full_image': value,
                        'person_list': [],
                        'person_fulls': [],
                        'object_list': [],
                        'place' : ''
                    }
                    if vid not in visuals[scene_vid] or key not in visuals[scene_vid][vid]:
                        continue

                    visual = visuals[scene_vid][vid][key]
                    processed_p = frames[key]['person_list']

                    for person in visual["persons"]:
                        person_id = person['person_id'].title()
                        person_id_idx = none_index if person_id == '' else speaker_index[person_id]  # none -> None

                        person_info = person['person_info']

                        behavior = person_info['behavior'].lower()
                        behavior_idx = pad_index if behavior == '' else vocab.get(behavior.split()[0])

                        emotion = person_info['emotion'].lower()
                        emotion_idx = pad_index if emotion == '' else vocab.get(emotion)

                        if len(person['related_objects']) > 0:
                            related_obj_id = person['related_objects'][0]['related_object_id'].lower()
                            related_obj_id_idx = vocab.get(related_obj_id)
                            relation = person['person_info']['predicate'].lower()
                            relation_idx = vocab.get(relation)
                        else:
                            related_obj_id_idx = pad_index
                            relation_idx = pad_index

                        processed = [person_id_idx, behavior_idx, emotion_idx, related_obj_id_idx, relation_idx] # Don't convert visual to a tensor yet
                        processed_p.append(processed)

                    if processed_p:
                        frames[key]['person_fulls'] = list(person_fulls[scene_vid][vid][key])

                    obj_list = list()
                    for obj in visual["objects"]:
                        obj_id = obj['object_id']
                        obj_id_idx = vocab.get(obj_id)
                        obj_list.append(obj_id_idx)

                    frames[key]['object_list'] = obj_list
                    frames[key]['place'] = pad_index if visual['place'] == '' else vocab.get(visual['place'])

        vids_list = list()
        qa_paths = {m: Path(args['qa_path']) / 'AnotherMissOhQA_{}_set.json'.format(m) for m in modes}
        qa_set = list()

        for mode in modes:
            qa_set.append(read_json(qa_paths[mode]))

        for i, set_og in enumerate(qa_set):
            vids_l = set()
            for d in set_og:
                vids_l.add(d['vid'][:-5])
            vids_list.append(vids_l)

        train_vids, val_vids, test_vids = vids_list
        scene_vids = {'train': vids_list[0], 'val': vids_list[1], 'test': vids_list[2]}

        full_images_by_modes = {mode: {k: full_images[k] for k in scene_vids[mode]} for mode in modes}
        for mode in modes:
            save_pickle(full_images_by_modes[mode], save_path[mode])


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
                vis_graph.extend(frame['person_list'])

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
        self.tokenizer = text_data.tokenizer
        self.get_script = text_data.get_script
        self.vocab = text_data.vocab

        ###### Image #####
        image_data = ImageData(args, mode, self.vocab)
        self.image = image_data
        self.image_dim = image_data.image_dim

        ###### Constraints ######
        self.max_sub_len = args['max_sub_len']
        self.max_image_len = args['max_image_len']
        self.max_sen_len = args['max_word_per_sentence']

        ###### Special indices ######
        self.none_index = speaker_index['None']
        self.pad_index = self.vocab.get(pad_token)
        self.eos_index = self.vocab.get(eos_token)

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

        script_types = ['sentence']
        assert self.args['script_type'] in script_types, "scrtip_type should be %s." % (' or '.join(script_types))

        spkr, script = self.get_script(subtitle)

        data['spkr'] = spkr
        data['script'] = script

        visual_types = ['shot']
        assert self.args['visual_type'] in visual_types, "visual_typoe should be %s." % (' or '.join(visual_types))

        vfeatures, vmetas = self.image.get_bbft(vid)

        data['bbfts'] = vfeatures
        data['vgraphs'] = vmetas

        # currently not tensor yet
        return data

    def bs2bi(self, batch_sentences):
        batch_size = len(batch_sentences)
        T1 = [len(batch_sentences[i]) for i in range(batch_size)]
        T_max = max(T1)
        sentences = []

        for i in range(batch_size):
            if len(batch_sentences[i]) < T_max:
                batch_sentences[i].extend([pad_token] * (T_max-len(batch_sentences[i])))

            sentences.extend(batch_sentences[i])

        o = self.tokenizer(sentences, padding=True, truncation=True, max_length=40)

        o = {k: [v[T_max*i:T_max*(i+1)] for i in range(batch_size)] for k, v in o.items()}

        max_l_l = [[sum(o['attention_mask'][i][j]) for j in range(T_max)] for i in range(batch_size)]

        # max_l = [max(max_l_l[i]) for i in range(batch_size)]
        return o, torch.tensor(T1), torch.tensor(max_l_l)

    def bs2bi2d(self, batch_sentences):
        batch_size = len(batch_sentences)

        o = self.tokenizer(batch_sentences, padding=True, truncation=True, max_length=40)
        max_l = [sum(o['attention_mask'][i]) for i in range(batch_size)]
        o = {k: v for k, v in o.items()}
        # max_l = [max(max_l_l[i]) for i in range(batch_size)]
        return o, torch.tensor(max_l)

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
            qa_concat = [[collected['que'][j] + '? ' + collected['ans'][j][i] + '. ' for j in range(len(collected['que']))] for i in range(5)]
            qa_concat_dict, max_l, qa_concat_l = [], [], []
            for i in range(5):
                input1, input2 = self.bs2bi2d(qa_concat[i])
                qa_concat_dict.append(input1)
                qa_concat_l.append(input2)

            #qa_concat, _, qa_concat_l = pad3d(qa_concat, self.pad_index, int_dtype)
            data['qa'] = qa_concat_dict
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
            # sub, sub_l, sub_l_l = pad3d(collected['script'], self.pad_index, int_dtype)
            sub, sub_l, sub_l_l = self.bs2bi(collected['script'])

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
            vgraphs, vgraphs_l, _ = pad3d(collected['vgraphs'], self.image.visual_pad, int_dtype, reshape4d=True, last_dim=5)

        data['bbfts'] = bbfts
        data['bbfts_l'] = bbfts_l
        data['bbfts_l_l'] = bbfts_l_l
        data['vmeta'] = vgraphs

        # currently not in the device yet
        return data
