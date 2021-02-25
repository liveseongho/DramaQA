import re
import numpy as np
from unidecode import unidecode
from .modules_language import Vocab
from tqdm import tqdm
from utils import read_json, write_json, save_pickle

empty_sub = '.'

sos_token = '<sos>'
eos_token = '<eos>'
pad_token = '<pad>'
unk_token = '<unk>'
special_tokens = [sos_token, eos_token, pad_token, unk_token]

speaker_name = [
    'None',  # index 0: unknown speaker
    'Anna', 'Chairman', 'Deogi', 'Dokyung', 'Gitae',
    'Haeyoung1', 'Haeyoung2', 'Heeran', 'Hun', 'Jeongsuk',
    'Jinsang', 'Jiya', 'Kyungsu', 'Sangseok', 'Seohee',
    'Soontack', 'Sukyung', 'Sungjin', 'Taejin', 'Yijoon'
]
modes = ['train', 'val']#, 'test']

speaker_index = {name: index for index, name in enumerate(speaker_name)}
n_speakers = len(speaker_name)

def tokenize(obj,tokenizer):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(o)) for n, o in obj.items())
    return list(tokenize(o, tokenizer) for o in obj)    



def load_subtitle(subtitle_path):
    subtitles = {}
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

# Todo(Donggeon) : emotion, action add
def preprocess_text(tokenizer, split_tool, json_data_path, save_path):
    print('Splitting long subtitles and converting words in text data to indices, timestamps from string to float.')

    texts = {mode: read_json(json_data_path[mode]) for mode in modes}
    for text in texts.values():
        for e in text:
            e['que'] = [tokenize(e['que'], tokenizer)]
            answer = e['answers'][e['correct_idx']] 
            e['str_answer'] = answer
            e['answer'] = [tokenize(answer, tokenizer)]
       
            subtitle = e['subtitle']

            if subtitle != empty_sub:
                subtitle['et'] = float(subtitle['et'])
                subtitle['st'] = float(subtitle['st'])

                new_subs = []

                for sub in subtitle['contained_subs']:
                    sub['et'] = float(sub['et'])
                    sub['st'] = float(sub['st'])
                    sub['speaker'] = speaker_index[sub['speaker']]  # to speaker index
                    split_subs = split_subtitle(sub, tokenizer, split_tool, to_indices=True)
                    new_subs.extend(split_subs)

                subtitle['contained_subs'] = new_subs 

    '''
    for key, text in texts.items():
        texts[key] = [e for e in text if e['q_level_logic'] >= 3]

    '''
    print("Saving converted data as pickle.")
    for mode in modes:
        save_pickle(texts[mode], save_path[mode])


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


def split_subtitle(sub, tokenizer, split_tool, sos=True, eos=True, to_indices=False, word2idx=None):
#    if to_indices and word2idx is None:
#        raise ValueError('word2idx should be given when to_indices is True')

    n_special_tokens = sos + eos  # True == 1, False == 0
    st, et = sub['st'], sub['et']
    t_range = et - st
    speaker = sub['speaker']

    utters = split_string(sub['utter'], split_tool, sos=sos, eos=eos)
    if to_indices:
        utters = [tokenize(words, tokenizer) for words in utters]
#        utters = [words_to_indices(words, word2idx) for words in utters]

    if len(utters) == 1:
        sub['utter'] = utters[0]

        return [sub]

    utters_len = np.array([len(u) - n_special_tokens for u in utters])  # -2 for <sos> and <eos>
    ratio = utters_len.cumsum() / utters_len.sum()
    ets = st + ratio * t_range
    sts = [st] + list(ets[:-1])

    subs = [dict(speaker=speaker, st=s, et=e, utter=u) for s, e, u in zip(sts, ets, utters)]

    return subs


# Split a string with multiple sentences to multiple strings with one sentence.
def split_string(string, tokenizer, min_sen_len=3, sos=True, eos=True):
    eos_re = re.compile(r'[\s]*[.?!]+[\s]*')
    split = eos_re.split(string)
    split = list(filter(None, split))  # remove ''
    split = [line_to_words(s, tokenizer, sos=sos, eos=eos) for s in split]  # tokenize each split sentence

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
                    s = split[i][:len(split[i]) - eos] + split[i + 1][sos:]
                    i += 1

                no_short.append(s)
            else:
                no_short[-1] = no_short[-1][:len(no_short[-1]) - eos] + split[i][sos:]
        else:
            s = split[i]
            no_short.append(s)

        i += 1

    return no_short


def clean_string(string):
    string = re.sub(r"[^A-Za-z0-9!?.]", " ", string)  # remove all special characters except ! ? .
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
