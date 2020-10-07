from base import BaseDataLoader

from collections import defaultdict

import torch

from utils import *
from .preprocess_script import merge_qa_subtitle, empty_sub
from .preprocess_image import preprocess_images
from .modules_language import Vocab, get_tokenizer
import os
import re
from tqdm import tqdm
import numpy as np

from torch.utils.data import Dataset

# debug
from pprint import pprint

modes = ['train', 'val', 'test']

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
        self.image_dt = self.load_images(args)


    def load_images(self, args):
        features, visuals = preprocess_images(args)

        """
        {
            full_image:   full_image (tensor of shape (512,)),
            persons:      [[person1_id_idx, behavior1_idx, emotion1_idx], ...],
            person_fulls: [person_full1 (tensor of shape (512,)), ... ]
        }
        """
        full_images = features['full_image']
        person_fulls = features['person_full']

        new_visuals = defaultdict(dict)
        for i in range(1, 19):
            for key, value in visuals[i].items():
                frame_id  = value['frame_id']
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
                        'persons': [],
                        'person_fulls': []
                    }
                    if vid not in visuals[scene_vid] or key not in visuals[scene_vid][vid]:
                        continue

                    visual = visuals[scene_vid][vid][key]
                    processed_p = frames[key]['persons']

                    for person in visual["persons"]:
                        person_id = person['person_id'].title()
                        person_id_idx = self.none_index if person_id == '' else speaker_index[person_id] # none -> None

                        person_info = person['person_info']

                        behavior = person_info['behavior'].lower()
                        behavior_idx = self.pad_index if behavior == '' else self.vocab.get_index(behavior.split()[0])

                        emotion = person_info['emotion'].lower()
                        emotion_idx= self.pad_index if emotion == '' else self.vocab.get_index(emotion)

                        processed = [person_id_idx, behavior_idx, emotion_idx] # Don't convert visual to a tensor yet
                        processed_p.append(processed)

                    if processed_p:
                        frames[key]['person_fulls'] = list(person_fulls[scene_vid][vid][key])

        return full_images


    def get_bbft(self, vid, flatten=False):
        bb_features = []
        visual_graphs = []

        episode = get_episode_id(vid)
        scene = get_scene_id(vid)
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

        self.line_keys = ['que']
        self.list_keys = ['answers']
        self.contained_subs_keys = ['speaker', 'utter']

        self.glove_path = args['glove_path']
        self.vocab_path = args['vocab_path']
        self.subtitle_path = args['subtitle_path']
        self.visual_path = args['visual_path']
        self.json_data_path = {m: get_data_path(args, mode=m, ext='.json') for m in modes}

        self.pickle_data_path = {m: get_data_path(args, mode=m, ext='.pickle') for m in modes}
        self.nc_data_path = {m: get_data_path(args, mode=m, ext='_nc.pickle') for m in modes}
        self.bert_data_path = {m: get_data_path(args, mode=m, ext='_bert.pickle') for m in modes}

        self.eos_re = re.compile(r'[\s]*[.?!]+[\s]*')

        self.special_tokens = [sos_token, eos_token, pad_token, unk_token]

        if args['bert']:
            print('BERT mode ON')
            self.tokenizer, self.vocab = get_tokenizer(args, self.special_tokens)
            if not os.path.isfile(self.bert_data_path[mode]):
                self.preprocess_text(self.vocab, save_path=self.bert_data_path)
            self.data = load_pickle(self.bert_data_path[mode])
        else:
            self.tokenizer = get_tokenizer(args)
            if os.path.isfile(self.vocab_path): # Use cached vocab if it exists.
                print('Using cached vocab')
                self.vocab = load_pickle(self.vocab_path)
            else: # There is no cached vocab. Build vocabulary and preprocess text data
                print('There is no cached vocab.')
                self.vocab = self.build_word_vocabulary()
                if not self.args['remove_coreference']:
                    self.preprocess_text(self.vocab, save_path=self.pickle_data_path)
                else:
                    self.preprocess_text(self.vocab, save_path=self.nc_data_path)

            if not self.args['remove_coreference']:
                self.data = load_pickle(self.pickle_data_path[mode])
            else:
                self.data = load_pickle(self.nc_data_path[mode])

    # borrowed this implementation from load_glove of tvqa_dataset.py (TVQA),
    # which borrowed from @karpathy's neuraltalk.
    def build_word_vocabulary(self, word_count_threshold=0):
        print("Building word vocabulary starts.")
        print('Merging QA and subtitles.')
        self.merge_text_data()

        print("Loading glove embedding at path: %s." % self.glove_path)
        glove_full, embedding_dim = self.load_glove(self.glove_path)
        glove_keys = glove_full.keys()

        modes_str = "/".join(modes)
        print("Glove Loaded. Building vocabulary from %s QA-subtitle data and visual." % (modes_str))
        self.raw_texts = {mode: read_json(self.json_data_path[mode]) for mode in modes}
        all_sentences = []
        for text in self.raw_texts.values():
            for e in text:
                for k in self.line_keys:
                    all_sentences.append(e[k])

                for k in self.list_keys:
                    all_sentences.extend(e[k])

                subtitle = e['subtitle']

                if subtitle != empty_sub:
                    for sub in subtitle['contained_subs']:
                        for k in self.contained_subs_keys:
                            all_sentences.append(sub[k])

        visual = read_json(self.visual_path)
        text_in_visual = set()
        for frames in visual.values():
            for frame in frames:
                for person in frame["persons"]:
                    person_info = person['person_info']
                    text_in_visual.add(person_info['behavior'])
                    text_in_visual.add(person_info['emotion'])

        #text_in_visual.remove('')
        all_sentences.extend(text_in_visual)

        # Find all unique words and count their occurence
        word_counts = {}
        for sentence in all_sentences:
            for w in self.line_to_words(sentence, sos=False, eos=False, downcase=True):
                word_counts[w] = word_counts.get(w, 0) + 1

        n_all_words = len(word_counts)
        print("The number of all unique words in %s data: %d." % (modes_str, n_all_words))

        # Remove words that have no Glove embedding vector, or speaker names.
        # Speaker names will be added later with random vectors.
        unk_words = [w for w in word_counts if w not in glove_keys or w.title() in speaker_name]
        for w in unk_words:
            del word_counts[w]

        n_glove_words = len(word_counts)
        n_unk_words = n_all_words - n_glove_words
        print("The number of all unique words in %s data that uses GloVe embeddings: %d. "
              '%.2f%% words are treated as %s or speaker names.'
              % (modes_str, n_glove_words, 100 * n_unk_words / n_all_words, unk_token))

        # Accept words whose occurence counts are greater or equal to the threshold.
        vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold and w not in self.special_tokens]
        print("Vocabulary size %d (speakers and %s excluded) using word_count_threshold %d." %
              (len(vocab), ' '.join(self.special_tokens), word_count_threshold))

        # Build index and vocabularies.
        print("Building word2idx, idx2word mapping.")

        # speaker name
        word2idx = {name.lower(): idx for name, idx in speaker_index.items()}
        idx2word = {idx: token for token, idx in word2idx.items()}
        offset = len(word2idx)

        # special characters
        for idx, w in enumerate(self.special_tokens):
            word2idx[w] = idx + offset
            idx2word[idx + offset] = w
        offset = offset + len(self.special_tokens)

        # all words in vocab
        for idx, w in enumerate(vocab):
            word2idx[w] = idx + offset
            idx2word[idx + offset] = w

        print("word2idx size: %d, idx2word size: %d." % (len(word2idx), len(idx2word)))

        # Build GloVe matrix
        print('Building GloVe matrix')

        np.random.seed(0)
        glove_matrix = np.zeros([len(idx2word), embedding_dim])
        n_glove = n_unk = n_name = n_zero = 0
        unk_words = []
        for i in range(len(idx2word)):
            w = idx2word[i]

            if w.title() in speaker_name[1:]: # Remove 'None' from speaker name to use GloVe vector.
                w_embed = np.random.randn(embedding_dim) * 0.4
                n_name += 1
            elif w in glove_keys:
                w_embed = glove_full[w]
                n_glove += 1
            elif w == pad_token:
                w_embed = 0 # zero vector
                n_zero += 1
            else: # <eos>, <sos> are all mapped to <unk>
                w_embed = glove_full[unk_token]
                n_unk += 1
                unk_words.append(w)

            glove_matrix[i, :] = w_embed


        print("Vocab embedding size is :", glove_matrix.shape)
        print('%d words are initialized with known GloVe vectors, '
              '%d words (names) are randomly initialized, '
              '%d words (%s) are initialized as 0, and '
              '%d words (%s) are initialized with %s GloVe vectors.'
              % (n_glove, n_name, n_zero, pad_token, n_unk, ' '.join(unk_words), unk_token))
        print("Building vocabulary done.")

        vocab = Vocab(glove_matrix, idx2word, word2idx, self.special_tokens)

        print("Saving vocab as pickle.")
        save_pickle(vocab, self.vocab_path)

        return vocab

    def preprocess_text(self, vocab, save_path):
        print('Splitting long subtitles and converting words in text data to indices, timestamps from string to float.')
        word2idx = vocab.stoi
        texts = {mode: read_json(self.json_data_path[mode]) for mode in modes}
        for text in texts.values():
            for e in text:
                for k in self.line_keys:
                    e[k] = self.line_to_indices(e[k], word2idx)

                for k in self.list_keys:
                    e[k] = [self.line_to_indices(line, word2idx) for line in e[k]]

                subtitle = e['subtitle']

                if subtitle != empty_sub:
                    subtitle['et'] = float(subtitle['et'])
                    subtitle['st'] = float(subtitle['st'])

                    new_subs = []

                    for sub in subtitle['contained_subs']:
                        sub['et'] = float(sub['et'])
                        sub['st'] = float(sub['st'])
                        sub['speaker'] = speaker_index[sub['speaker']] # to speaker index
                        split_subs = self.split_subtitle(sub, to_indices=True, word2idx=word2idx)
                        new_subs.extend(split_subs)

                    subtitle['contained_subs'] = new_subs

        '''
        for key, text in texts.items():
            texts[key] = [e for e in text if e['q_level_logic'] >= 3]

        '''
        print("Saving converted data as pickle.")
        for mode in modes:
            save_pickle(texts[mode], save_path[mode])

        #self.extract_val_ch_only(texts['val'])
        del texts


    # borrowed this implementation from TVQA (load_glove of tvqa_dataset.py)
    def load_glove(self, glove_path):
        glove = {}

        with open(glove_path, encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                values = line.strip('\n').split(' ')
                word = values[0]
                vector = np.asarray([float(e) for e in values[1:]])
                glove[word] = vector

        embedding_dim = len(vector)

        return glove, embedding_dim

    def split_subtitle(self, sub, sos=True, eos=True, to_indices=False, word2idx=None):
        if to_indices == True and word2idx == None:
            raise ValueError('word2idx should be given when to_indices is True')

        n_special_tokens = sos + eos # True == 1, False == 0
        st, et = sub['st'], sub['et']
        t_range = et - st
        speaker = sub['speaker']

        utters = self.split_string(sub['utter'], sos=sos, eos=eos)
        if to_indices:
            utters = [self.words_to_indices(words, word2idx) for words in utters]

        if len(utters) == 1:
            sub['utter'] = utters[0]

            return [sub]

        utters_len = np.array([len(u) - n_special_tokens for u in utters]) # -2 for <sos> and <eos>
        ratio = utters_len.cumsum() / utters_len.sum()
        ets = st + ratio * t_range
        sts = [st] + list(ets[:-1])

        subs = [dict(speaker=speaker, st=s, et=e, utter=u) for s, e, u in zip(sts, ets, utters)]

        return subs

    # Split a string with multiple sentences to multiple strings with one sentence.
    def split_string(self, string, min_sen_len=3, sos=True, eos=True):
        split = self.eos_re.split(string)
        split = list(filter(None, split)) # remove ''
        split = [self.line_to_words(s, sos=sos, eos=eos) for s in split] # tokenize each split sentence

        # Merge short sentences to adjacent sentences
        n_special_tokens = sos + eos # True == 1, False == 0
        no_short = []
        i = 0
        n_sentences = len(split)
        while i < n_sentences:
            length = len(split[i]) - n_special_tokens # -2 for <sos> and <eos>
            if length < min_sen_len:
                if i == 0:
                    if n_sentences == 1:
                        s = split[i] # 0
                    else:
                        # concatenate split[0] and split[1]
                        # if eos == True (== 1), exclude <eos> from split[0] (split[i][:-1])
                        # else                 ,           just use split[0] (split[i][:len(split[i])])
                        #
                        # if sos == True (== 1), exclude <sos> from split[1] (split[i + 1][1:])
                        # else                 ,           just use split[1] (split[i + 1][0:])
                        s = split[i][:len(split[i])-eos] + split[i + 1][sos:]
                        i += 1

                    no_short.append(s)
                else:
                    no_short[-1] = no_short[-1][:len(no_short[-1])-eos] + split[i][sos:]
            else:
                s = split[i]
                no_short.append(s)

            i += 1

        return no_short

    def clean_string(self, string):
        string = re.sub(r"[^A-Za-z0-9!?.]", " ", string) # remove all special characters except ! ? .
        string = re.sub(r"\.{2,}", ".", string)
        string = re.sub(r"\s{2,}", " ", string)

        return string.strip()

    def remove_coreference(self, line):
        remove_flag = True
        while remove_flag:
            remove_flag = False
            if '(' in line:
                speaker_candidates = line.split('(')[1].split(')')[0]
                for s in speaker_candidates.split(','):
                    if s in speaker_name:
                        remove_flag = True

                if remove_flag:
                    line = line.replace('(' + speaker_candidates + ')', '')
            else:
                remove_flag = False
        return line


    # borrowed this implementation from TVQA (line_to_words of tvqa_dataset.py)
    def line_to_words(self, line, sos=True, eos=True, downcase=True):
        if self.args['remove_coreference']:
            line = self.remove_coreference(line)

        line = self.clean_string(line)
        tokens = self.tokenizer(line.lower()) if downcase else self.tokenizer(line)

        words = [sos_token] if sos else []
        words = words + [w for w in tokens if w != ""]
        words = words + [eos_token] if eos else words

        return words

    def words_to_indices(self, words, word2idx):
        indices = [word2idx.get(w, word2idx[unk_token]) for w in words]

        return indices

    def line_to_indices(self, line, word2idx, sos=True, eos=True, downcase=True):
        words = self.line_to_words(line, sos=sos, eos=eos, downcase=downcase)
        indices = self.words_to_indices(words, word2idx)

        return indices

    def merge_text_data(self):
        for mode in modes:
            ext = '.json'
            new_path = self.json_data_path[mode]
            qa_path = new_path.parent / (new_path.stem[:new_path.stem.find('_script')] + ext)
            subtitle_path = self.subtitle_path
            merge_qa_subtitle(new_path, qa_path, subtitle_path)


def get_data_path(args, mode='train', ext='.json'):
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
        shot_contained = text['shot_contained']
        vid = text['vid']
        episode = get_episode_id(vid)
        scene = get_scene_id(vid)

        spkr_of_sen_l = []  # list of speaker of subtitle sentences
        sub_in_sen_l = []   # list of subtitle sentences
        mean_fi_l = []      # list of meaned full_image features
        all_fi_l = []       # list of all full_image features
        all_pfu_l = []      # list of all person_full features
        sample_v_l = []     # list of one sample visual
        all_v_l = []        # list of all visual
        data = {
            'que': que,
            'ans': ans,
            'correct_idx': correct_idx,

            'q_level_logic': q_level_logic,
            'qid': qid
        }

        script_types = ['sentence', 'word']
        assert self.args['script_type'] in script_types, "scrtip_type should be %s." % (' or '.join(script_types))

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

