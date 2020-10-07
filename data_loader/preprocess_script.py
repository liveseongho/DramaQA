import os, re
import json
import numpy as np
from pathlib import Path
from unidecode import unidecode
from .modules_language import Vocab
from tqdm import tqdm
from utils import read_json, write_json, save_pickle

# debug
from pprint import pprint

empty_sub = '.'

sos_token = '<sos>'
eos_token = '<eos>'
pad_token = '<pad>'
unk_token = '<unk>'
special_tokens = [sos_token, eos_token, pad_token, unk_token]

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


def load_subtitle(subtitle_path):
    subtitles = {}
    speakers = {}
    subtitles = read_json(subtitle_path)

    for vid, v in subtitles.items():
        for sub in subtitles[vid]["contained_subs"]:
            sub["utter"] = unidecode(sub["utter"])

    return subtitles


def merge_qa_subtitle(new_path, qa_path, subtitle_path):
    print("Processing subtitle data")

    subtitles = load_subtitle(subtitle_path)
    qa = read_json(qa_path)

    res = []
    for row in tqdm(qa):
        if row['vid'].endswith('_0000'):
            # scene question
            vid = row['vid']
            vid_prefix = vid[:vid.find('_0000')]

            shot_subtitles = []
            shot_st = []
            shot_et = []
            subtitle_sts = set()

            # if vid starts with vid_prefix,
            # add sub to shot_subtitles if the same sub has not been added yet
            for vid, subs in subtitles.items():
                if not vid.startswith(vid_prefix):
                    continue

                shot_st.append(subs['st'])
                shot_et.append(subs['et'])

                for sub in subs['contained_subs']:
                    st = sub['st']

                    if st in subtitle_sts:
                        continue

                    subtitle_sts.add(st)
                    shot_subtitles.append((float(st), sub))

            shot_st = sorted(shot_st, key=float)
            shot_et = sorted(shot_et, key=float)
            shot_subtitles.sort()

            if shot_subtitles:
                row['subtitle'] = {}
                row['subtitle']["contained_subs"] = [sub for st, sub in shot_subtitles]
                row['subtitle']["st"] = shot_st[0]
                row['subtitle']["et"] = shot_et[-1]
            else:
                row['subtitle'] = ''
        else:
            # shot question
            if row['vid'] in subtitles:
                row['subtitle'] = subtitles[row['vid']]
            else:
                row['subtitle'] = ''

        if row['subtitle'] == '':
            row['subtitle'] = empty_sub  # prevent empty string
        res.append(row)

    write_json(res, new_path)


def merge_text_data(args, json_data_path):
    subtitle_path = args['subtitle_path']
    for mode in modes:
        ext = '.json'
        new_path = json_data_path[mode]
        qa_path = new_path.parent / (new_path.stem[:new_path.stem.find('_script')] + ext)
        subtitle_path = subtitle_path
        merge_qa_subtitle(new_path, qa_path, subtitle_path)


# borrowed this implementation from load_glove of tvqa_dataset.py (TVQA),
# which borrowed from @karpathy's neuraltalk.
def build_word_vocabulary(args, tokenizer, json_data_path, word_count_threshold=0):
    print("Building word vocabulary starts.")
    print('Merging QA and subtitles.')

    merge_text_data(args, json_data_path)

    glove_path = args['glove_path']
    print("Loading glove embedding at path: %s." % glove_path)
    glove_full, embedding_dim = load_glove(glove_path)
    glove_keys = glove_full.keys()

    modes_str = "/".join(modes)
    print("Glove Loaded. Building vocabulary from %s QA-subtitle data and visual." % (modes_str))
    raw_texts = {mode: read_json(json_data_path[mode]) for mode in modes}
    all_sentences = []
    for text in raw_texts.values():
        for e in text:
            all_sentences.append(e['que'])
            all_sentences.extend(e['answers'])

            subtitle = e['subtitle']

            if subtitle != empty_sub:
                for sub in subtitle['contained_subs']:
                    for k in ['speaker', 'utter']:
                        all_sentences.append(sub[k])

    visual = read_json(args['visual_path'])
    text_in_visual = set()
    for frames in visual.values():
        for frame in frames:
            for person in frame["persons"]:
                person_info = person['person_info']
                text_in_visual.add(person_info['behavior'])
                text_in_visual.add(person_info['emotion'])

    # text_in_visual.remove('')
    all_sentences.extend(text_in_visual)

    # Find all unique words and count their occurence
    word_counts = {}
    for sentence in all_sentences:
        for w in line_to_words(sentence, tokenizer, sos=False, eos=False, downcase=True):
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
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold and w not in special_tokens]
    print("Vocabulary size %d (speakers and %s excluded) using word_count_threshold %d." %
            (len(vocab), ' '.join(special_tokens), word_count_threshold))

    # Build index and vocabularies.
    print("Building word2idx, idx2word mapping.")

    # speaker name
    word2idx = {name.lower(): idx for name, idx in speaker_index.items()}
    idx2word = {idx: token for token, idx in word2idx.items()}
    offset = len(word2idx)

    # special characters
    for idx, w in enumerate(special_tokens):
        word2idx[w] = idx + offset
        idx2word[idx + offset] = w
    offset = offset + len(special_tokens)

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

        if w.title() in speaker_name[1:]:  # Remove 'None' from speaker name to use GloVe vector.
            w_embed = np.random.randn(embedding_dim) * 0.4
            n_name += 1
        elif w in glove_keys:
            w_embed = glove_full[w]
            n_glove += 1
        elif w == pad_token:
            w_embed = 0  # zero vector
            n_zero += 1
        else:  # <eos>, <sos> are all mapped to <unk>
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

    vocab = Vocab(glove_matrix, idx2word, word2idx, special_tokens)

    print("Saving vocab as pickle.")
    save_pickle(vocab, args['vocab_path'])

    return vocab

def preprocess_text(vocab, tokenizer, json_data_path, save_path):
    print('Splitting long subtitles and converting words in text data to indices, timestamps from string to float.')
    word2idx = vocab.stoi

    texts = {mode: read_json(json_data_path[mode]) for mode in modes}
    for text in texts.values():
        for e in text:
            e['que'] = line_to_indices(e['que'], tokenizer, word2idx)
            e['answers'] = [line_to_indices(line, tokenizer, word2idx) for line in e['answers']]

            subtitle = e['subtitle']

            if subtitle != empty_sub:
                subtitle['et'] = float(subtitle['et'])
                subtitle['st'] = float(subtitle['st'])

                new_subs = []

                for sub in subtitle['contained_subs']:
                    sub['et'] = float(sub['et'])
                    sub['st'] = float(sub['st'])
                    sub['speaker'] = speaker_index[sub['speaker']] # to speaker index
                    split_subs = split_subtitle(sub, tokenizer, to_indices=True, word2idx=word2idx)
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
def load_glove(glove_path):
    glove = {}

    with open(glove_path, encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            values = line.strip('\n').split(' ')
            word = values[0]
            vector = np.asarray([float(e) for e in values[1:]])
            glove[word] = vector

    embedding_dim = len(vector)

    return glove, embedding_dim

def split_subtitle(sub, tokenizer, sos=True, eos=True, to_indices=False, word2idx=None):
    if to_indices == True and word2idx == None:
        raise ValueError('word2idx should be given when to_indices is True')

    n_special_tokens = sos + eos # True == 1, False == 0
    st, et = sub['st'], sub['et']
    t_range = et - st
    speaker = sub['speaker']

    utters = split_string(sub['utter'], tokenizer, sos=sos, eos=eos)
    if to_indices:
        utters = [words_to_indices(words, word2idx) for words in utters]

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
def split_string(string, tokenizer, min_sen_len=3, sos=True, eos=True):
    eos_re = re.compile(r'[\s]*[.?!]+[\s]*')
    split = eos_re.split(string)
    split = list(filter(None, split)) # remove ''
    split = [line_to_words(s, tokenizer, sos=sos, eos=eos) for s in split] # tokenize each split sentence

    # Merge short sentences to adjacent sentences
    n_special_tokens = sos + eos  # True == 1, False == 0
    no_short = []
    i = 0
    n_sentences = len(split)
    while i < n_sentences:
        length = len(split[i]) - n_special_tokens  # -2 for <sos> and <eos>
        if length < min_sen_len:
            if i == 0:
                if n_sentences == 1:
                    s = split[i]  # 0
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


def clean_string(string):
    string = re.sub(r"[^A-Za-z0-9!?.]", " ", string) # remove all special characters except ! ? .
    string = re.sub(r"\.{2,}", ".", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip()


def remove_coreference(line):
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
def line_to_words(line, tokenizer, remove_c=False, sos=True, eos=True, downcase=True):
    if remove_c:
        line = remove_coreference(line)

    line = clean_string(line)
    tokens = tokenizer(line.lower()) if downcase else tokenizer(line)

    words = [sos_token] if sos else []
    words = words + [w for w in tokens if w != ""]
    words = words + [eos_token] if eos else words

    return words


def words_to_indices(words, word2idx):
    indices = [word2idx.get(w, word2idx[unk_token]) for w in words]

    return indices


def line_to_indices(line, tokenizer, word2idx, sos=True, eos=True, downcase=True):
    words = line_to_words(line, tokenizer, sos=sos, eos=eos, downcase=downcase)
    indices = words_to_indices(words, word2idx)

    return indices
