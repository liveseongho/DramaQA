from collections import defaultdict

import torch
import nltk

from utils import *
from .load_subtitle import merge_qa_subtitle, empty_sub
from .preprocess_image import preprocess_images

import os
import re
from tqdm import tqdm
import numpy as np

from torch.utils.data import Dataset, DataLoader

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

# Refer to https://docs.scipy.org/doc/numpy/user/basics.subclassing.html
# for information about subclassing np.ndarray
# 
# Refer to https://stackoverflow.com/questions/26598109/preserve-custom-attributes-when-pickling-subclass-of-numpy-array
# for information about pickling custom attributes of subclasses of np.ndarray
class Vocab(np.ndarray):
    def __new__(cls, input_array, idx2word, word2idx, special_tokens):
        obj = np.asarray(input_array).view(cls)
        
        obj.itos = idx2word
        obj.stoi = word2idx
        obj.specials = special_tokens
        obj.special_ids = [word2idx[token] for token in special_tokens]
        for token in special_tokens:
            setattr(obj, token[1:-1], token) # vocab.sos = '<sos>' ... 

        return obj

    def __array_finalize__(self, obj):
        if obj is None: 
            return

        self.itos = getattr(obj, 'itos', None)
        self.stoi = getattr(obj, 'stoi', None)
        self.specials = getattr(obj, 'specials', None)
        self.special_ids = getattr(obj, 'special_ids', None)
        if self.special_ids is not None:
            for token in obj.specials:
                attr = token[1:-1]
                setattr(self, attr, getattr(obj, attr, None))
        
    def __reduce__(self):
        pickled_state = super(Vocab, self).__reduce__()
        new_state = pickled_state[2] + (self.__dict__,)

        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        self.__dict__.update(state[-1])
        super(Vocab, self).__setstate__(state[0:-1])

    def get_word(self, idx):
        return self.itos[idx]

    def get_index(self, word):
        return self.stoi.get(word, self.stoi[unk_token])

class ImageData:
    def __init__(self, args, vocab):
        self.args = args

        self.vocab = vocab
        self.pad_index = vocab.get_index(pad_token)
        self.none_index = speaker_index['None']
        self.visual_pad = [self.none_index, self.pad_index, self.pad_index] 

        self.image_path = args.image_path
        self.image_dim = args.image_dim
        self.image_dt = self.load_images(args)
        self.structure = self.build_structure(args)

    def load_images(self, args):
        features, visuals = preprocess_images(args)
        image_dt = self.join(features, visuals)

        return image_dt

    def build_structure(self, args):
        image_path = self.image_path
        episode_dirs = sorted(e for e in os.listdir(image_path) if e.startswith("AnotherMissOh"))

        episodes = {}

        for e in episode_dirs:
            episodes[get_episode_id(e)] = {}
            scenes = episodes[get_episode_id(e)]

            episode_path = image_path / e
            scene_dirs = sorted(s for s in os.listdir(episode_path) if s.isnumeric())

            for s in scene_dirs:
                scenes[int(s)] = {}
                shots = scenes[int(s)]

                scene_path = episode_path / s
                shot_dirs = sorted(sh for sh in os.listdir(scene_path) if sh.isnumeric())

                for sh in shot_dirs:
                    shot_path = scene_path / sh
                    shots[int(sh)] = sorted(get_frame_id(f.split('.')[0]) for f in os.listdir(shot_path) if f.startswith('IMAGE'))

        return episodes

    def join(self, features, visuals):
        """
        After: 

        full_images = [
            {}, # empty dict

            { (episode1)
                frame_id1: {
                    full_image:   full_image (tensor of shape (512,)),
                    persons:      [[person1_id_idx, behavior1_idx, emotion1_idx], ...],
                    person_fulls: [person_full1 (tensor of shape (512,)), ... ] 
                },
                ...
            },

            ...

            { (episode18)
                ...
            }
        ]
        """
        full_images = features['full_image']
        person_fulls = features['person_full']

        for frames in full_images:
            for key, value in frames.items():
                frames[key] = {
                    'full_image': value,
                    'persons': [],
                    'person_fulls': []
                }

        for e in range(1, 18 + 1):
            master_dict = full_images[e]
            pfu_dict = person_fulls[e]
            visual_dict = visuals[e]

            for frame, info in master_dict.items():
                if frame not in visual_dict: # no visual for this frame
                    continue

                visual = visual_dict[frame]
                processed_persons = master_dict[frame]['persons']
                for person in visual["persons"]:
                    person_id = person['person_id'].title()
                    person_id_idx = self.none_index if person_id == '' else speaker_index[person_id] # none -> None

                    person_info = person['person_info']

                    behavior = person_info['behavior'].lower()
                    behavior_idx = self.pad_index if behavior == '' else self.vocab.get_index(behavior.split()[0]) 

                    emotion = person_info['emotion'].lower()
                    emotion_idx= self.pad_index if emotion == '' else self.vocab.get_index(emotion) 

                    processed = [person_id_idx, behavior_idx, emotion_idx] # Don't convert visual to a tensor yet
                    processed_persons.append(processed)

                # when processed_persons is empty, pfu_dict[frame] contains 
                # full_image feature. Just ignore this.
                if processed_persons: # not empty
                    master_dict[frame]['person_fulls'] = list(pfu_dict[frame]) # (N, C) np.array -> list of N arrays of shape (C,)

        return full_images

    def get_image_by_vid(self, episode, scene, shot_contained):
        first_shot = shot_contained[0]
        last_shot = shot_contained[-1]
        first_frame = self.structure[episode][scene][first_shot][0]
        last_frame = self.structure[episode][scene][last_shot][-1]

        return self.get_image_by_frame(episode, first_frame, last_frame + 1) # add + 1 to include last_frame

    def get_image_by_time(self, episode, st, et):
        return self.get_image_by_frame(episode, int(st * 25), int(et * 25))

    def get_image_by_frame(self, episode, start_frame_id, end_frame_id):
        frames_in_episode = self.image_dt[episode]

        cnt = 0       # number of frames
        mean_fi = 0   # mean of full_image features 
        all_fi = []   # list of full_image features  (aligned with person)
        all_pfu = []  # list of person_full features (aligned with person)
        all_v = []    # list of visuals              (aligned with person)
        sample_v = [] # first visual in the range

        cur_id = start_frame_id
        added_8 = False
        while cur_id < end_frame_id:
            if cur_id in frames_in_episode: # found a frame in a certain shot
                frame = frames_in_episode[cur_id]

                p = frame['persons']
                sample_v = p if sample_v == [] else sample_v
                all_v.extend(p) # include all visual info of a frame

                fi = frame['full_image']
                mean_fi += fi
                all_fi.extend(fi for i in range(len(p)))

                pfu = frame['person_fulls']
                all_pfu.extend(pfu)

                cnt += 1

                # adjacent frame ids in a shot differ by 8, so
                # add 8 to cur_id to go to the next frame directly
                cur_id += 8 
                added_8 = True
            else:
                if added_8:
                    cur_id -= 8 # move back

                # increment by 1 until a frame in a certain shot is found
                cur_id += 1
                added_8 = False

        if cnt == 0:
            mean_fi = np.zeros(self.image_dim)
        else:
            mean_fi /= cnt

        if not all_fi: # empty
            all_fi.append(np.zeros(self.image_dim))

        if not sample_v: # empty
            sample_v = self.visual_pad
        else:
            sample_v = sample_v[0] # just select the first one

        if not all_v: # empty: all_v and all_v are empty at the same time
            all_v = [self.visual_pad]
            all_pfu.append(np.zeros(self.image_dim))

        # Don't convert visual to tensors yet
        return mean_fi, all_fi, sample_v, all_v, all_pfu


class TextData:
    def __init__(self, args, vocab=None):
        self.val_types = ['all', 'ch_only']
        if args.val_type not in self.val_types:
            raise ValueError('val_type should be ' + ' or '.join(self.val_types))

        self.args = args

        self.line_keys = ['que']
        self.list_keys = ['answers']
        self.contained_subs_keys = ['speaker', 'utter']

        self.glove_path = args.glove_path
        self.vocab_path = args.vocab_path
        self.subtitle_path = args.subtitle_path
        self.visual_path = args.visual_path
        self.json_data_path = {m: get_data_path(args, mode=m, ext='.json') for m in modes}
        self.pickle_data_path = {m: get_data_path(args, mode=m, ext='.pickle') for m in modes}
        
        self.tokenizer = get_tokenizer(args)
        self.eos_re = re.compile(r'[\s]*[.?!]+[\s]*')        

        self.special_tokens = [sos_token, eos_token, pad_token, unk_token]

        if vocab is None:
            if os.path.isfile(self.vocab_path): # Use cached vocab if it exists.
                print('Using cached vocab')
                vocab = load_pickle(self.vocab_path) 
            else: # There is no cached vocab. Build vocabulary and preprocess text data
                print('There is no cached vocab.')
                vocab = self.build_word_vocabulary()
                self.preprocess_text(vocab)

        self.vocab = vocab

        self.data = {m: load_pickle(self.pickle_data_path[m]) for m in modes} 
        if args.val_type == 'ch_only' :
            self.data['val'] = load_pickle(get_data_path(args, mode='val_ch_only', ext='.pickle'))
    
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
        self.raw_texts = {mode: load_json(self.json_data_path[mode]) for mode in modes}
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

        visual = load_json(self.visual_path)
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

    def preprocess_text(self, vocab):
        print('Splitting long subtitles and converting words in text data to indices, timestamps from string to float.')
        word2idx = vocab.stoi
        texts = self.raw_texts # self.raw_texts is assigned in self.build_word_vocabulary
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

        print("Saving converted data as pickle.")
        for mode in modes:
            save_pickle(texts[mode], self.pickle_data_path[mode])

        del self.raw_texts

        self.extract_val_ch_only(texts['val'])

    def extract_val_ch_only(self, data):
        print('Extracting QAs that contains 5 different people in answer')

        new_data = []
        for qa in tqdm(data):
            ans = qa['answers']
            ans = [torch.tensor(ans[i], dtype=int_dtype) for i in range(5)]

            persons = set(idx.item() for i in range(5) for idx in ans[i][ans[i] < n_speakers])

            if len(persons) >= 5:
                new_data.append(qa)

        save_pickle(new_data, get_data_path(self.args, mode='val_ch_only', ext='.pickle'))

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

    # borrowed this implementation from TVQA (line_to_words of tvqa_dataset.py)
    def line_to_words(self, line, sos=True, eos=True, downcase=True):
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
    name = args.data_path.name.split('_')
    name.insert(1, mode)
    name = '_'.join(name)
    path = args.data_path.parent / name
    path = path.parent / (path.stem + ext)

    return path

class MultiModalData(Dataset):
    def __init__(self, args, text_data, image_data, mode):
        if mode not in modes:
            raise ValueError("mode should be %s." % (' or '.join(modes)))

        self.args = args
        self.mode = mode

        ###### Text ######
        self.text = text_data.data[mode]
        self.vocab = text_data.vocab

        ###### Image ######
        self.image = image_data
        self.image_dim = image_data.image_dim

        ###### Constraints ######
        self.max_sen_len = args.max_sentence_len
        self.max_sub_len = args.max_sub_len
        self.max_image_len = args.max_image_len

        ###### Special indices ######
        self.none_index = speaker_index['None']
        self.pad_index = self.vocab.get_index(pad_token)
        self.eos_index = self.vocab.get_index(eos_token)

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

        if subtitle != empty_sub: # subtitle exists
            subs = subtitle["contained_subs"]

            for s in subs:
                spkr = s["speaker"]
                utter = s["utter"]
                spkr_of_sen_l.append(spkr)
                if len(utter) > self.max_sen_len:
                    del utter[self.max_sen_len:]
                    utter[-1] = self.eos_index
                sub_in_sen_l.append(utter)

                # # get image by st and et
                # st = s["st"]
                # et = s["et"]
                # if et - st > 90: 
                #     et = st + 3
                # mean_fi, all_fi, sample_v, all_v, all_pfu = self.image.get_image_by_time(episode, st, et)
                # mean_fi_l.append(mean_fi)
                # all_fi_l.extend(all_fi)
                # sample_v_l.append(sample_v)
                # all_v_l.extend(all_v)
                # all_pfu_l.extend(all_pfu)

            # get image by shot_contained
            mean_fi, all_fi, sample_v, all_v, all_pfu = self.image.get_image_by_vid(episode, scene, shot_contained) # get all image in the scene/shot
            mean_fi_l.append(mean_fi)
            all_fi_l.extend(all_fi)
            sample_v_l.append(sample_v)
            all_v_l.extend(all_v)
            all_pfu_l.extend(all_pfu)
        else: # No subtitle
            spkr_of_sen_l.append(self.none_index) # add None speaker
            sub_in_sen_l.append([self.pad_index]) # add <pad>
            mean_fi, all_fi, sample_v, all_v, all_pfu = self.image.get_image_by_vid(episode, scene, shot_contained) # get all image in the scene/shot
            mean_fi_l.append(mean_fi)
            all_fi_l.extend(all_fi)
            sample_v_l.append(sample_v)
            all_v_l.extend(all_v)            
            all_pfu_l.extend(all_pfu)

        # Concatenate subtitle sentences
        sub_in_word_l = []; spkr_of_word_l = []
        max_sub_len = self.max_sub_len
        n_words = 0
        for spkr, s in zip(spkr_of_sen_l, sub_in_sen_l):
            sen_len = len(s)
            n_words += sen_len

            sub_in_word_l.extend(s)
            spkr_of_word_l.extend(spkr for i in range(sen_len)) # 1:1 correspondence between word and speaker

            if n_words > max_sub_len:
                del sub_in_word_l[max_sub_len:], spkr_of_word_l[max_sub_len:] 
                sub_in_word_l[-1] = self.eos_index

                break

        # Remove paddings in between and flatten visuals 
        filtered_v = []; filtered_fi = []; filtered_pfu = []
        max_img_len = self.max_image_len
        n_img = 0
        for v, fi, pfu in zip(all_v_l, all_fi_l, all_pfu_l):
            if v != self.image.visual_pad:
                filtered_v.extend(v) # flatten visuals 
                filtered_fi.append(fi)
                filtered_pfu.append(pfu)
                n_img += 1
                if n_img > max_img_len:
                    del filtered_fi[max_img_len:], filtered_v[max_img_len * 3:], filtered_pfu[max_img_len:]

                    break

        # Pad empty data
        if not sub_in_word_l: # empty
            sub_in_word_l.append(self.pad_index)
            spkr_of_word_l.append(self.none_index)

        if not filtered_v: # empty: filtered_v, filtered_fi, and filtered_pfu are empty at the same time
            filtered_v = self.image.visual_pad
            filtered_pfu.append(np.zeros(self.image_dim))
            filtered_fi = all_fi_l # use all image
            if len(filtered_fi) > max_img_len:
                del filtered_fi[max_img_len:]

        data = {
            'que': que,
            'ans': ans,
            'correct_idx': correct_idx,

            'sub_in_sen': sub_in_sen_l,
            'spkr_of_sen': spkr_of_sen_l,
            'mean_fi': mean_fi_l,
            'sample_v': sample_v_l,

            'spkr_of_word': spkr_of_word_l,
            'sub_in_word': sub_in_word_l,
            'filtered_fi': filtered_fi,
            'filtered_v': filtered_v,
            'filtered_pfu': filtered_pfu,

            'q_level_logic': q_level_logic,
            'qid': qid
        }
        
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
        correct_idx = torch.tensor(collected['correct_idx'], dtype=int_dtype) if self.mode != 'test' else None # correct_idx does not have to be padded
        
        spkr_of_s, _ = self.pad2d(collected['spkr_of_sen'], self.none_index, int_dtype)
        mean_fi, _, _ = self.pad3d(collected['mean_fi'], 0, float_dtype)
        sample_v, _, _ = self.pad3d(collected['sample_v'], self.image.visual_pad, int_dtype)
        sub_in_s, sub_in_s_l, sub_s_l = self.pad3d(collected['sub_in_sen'], self.pad_index, int_dtype)

        f_v, f_v_l = self.pad2d(collected['filtered_v'], self.image.visual_pad, int_dtype)
        f_fi, f_fi_l, _ = self.pad3d(collected['filtered_fi'], 0, float_dtype)
        f_pfu, f_pfu_l, _ = self.pad3d(collected['filtered_pfu'], 0, float_dtype) 
        spkr_of_w, _ = self.pad2d(collected['spkr_of_word'], self.none_index, int_dtype)
        sub_in_w, sub_in_w_l = self.pad2d(collected['sub_in_word'], self.pad_index, int_dtype)

        q_level_logic = collected['q_level_logic'] # No need to convert to torch.Tensor
        qid = collected['qid'] # No need to convert to torch.Tensor
        
        data = {
            'que': que,
            'answers': ans, 
            'que_len': que_l,
            'ans_len': ans_l,

            'subtitle': sub_in_s, 
            'speaker': spkr_of_s, 
            'images': mean_fi,
            'sample_visual': sample_v,
            'sub_len': sub_in_s_l,
            'sub_sentence_len': sub_s_l,

            'filtered_visual': f_v,
            'filtered_sub': sub_in_w,
            'filtered_speaker': spkr_of_w,
            'filtered_image': f_fi,
            'filtered_person_full': f_pfu,
            'filtered_visual_len': f_v_l,
            'filtered_sub_len': sub_in_w_l,
            'filtered_image_len': f_fi_l,
            'filtered_person_full_len': f_pfu_l,

            'q_level_logic': q_level_logic,
            'qid': qid
        }

        if correct_idx is not None:
            data['correct_idx'] = correct_idx 

        return data

    def pad2d(self, data, pad_val, dtype):
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

        return p_data, p_length

    def pad3d(self, data, pad_val, dtype):
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

        return p_data, p_dim1_length, p_dim2_length

def get_tokenizer(args):
    tokenizers = {
        'nltk': nltk.word_tokenize,
        'nonword': re.compile(r'\W+').split,
    }

    return tokenizers[args.tokenizer.lower()]

def load_data(args, vocab=None):
    print('Loading text data')
    text = TextData(args, vocab)
    vocab = text.vocab

    print('Load image data')
    image = ImageData(args, vocab)

    train_dataset = MultiModalData(args, text, image, mode='train')
    valid_dataset = MultiModalData(args, text, image, mode='val')
    test_dataset  = MultiModalData(args, text, image, mode='test')

    train_iter = DataLoader(
        train_dataset, 
        batch_size=args.batch_sizes[0],
        shuffle=args.shuffle[0], 
        num_workers=args.num_workers,
        collate_fn=train_dataset.collate_fn
    )

    val_iter = DataLoader(
        valid_dataset, 
        batch_size=args.batch_sizes[1],
        shuffle=args.shuffle[1],
        num_workers=args.num_workers,
        collate_fn=valid_dataset.collate_fn
    )

    test_iter = DataLoader(
        test_dataset, 
        batch_size=args.batch_sizes[2],
        shuffle=args.shuffle[2],
        num_workers=args.num_workers,
        collate_fn=test_dataset.collate_fn
    )

    return {'train': train_iter, 'val': val_iter, 'test': test_iter}, vocab


def get_iterator(args, vocab=None):
    iters, vocab = load_data(args, vocab)
    print("Data loading done")

    return iters, vocab

