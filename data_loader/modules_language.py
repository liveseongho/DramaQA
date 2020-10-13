import numpy as np
from transformers import BertTokenizer
import torchtext
import nltk
import re
import itertools

speaker_name = [
    'None', # index 0: unknown speaker
    'Anna', 'Chairman', 'Deogi', 'Dokyung', 'Gitae',
    'Haeyoung1', 'Haeyoung2', 'Heeran', 'Hun', 'Jeongsuk',
    'Jinsang', 'Jiya', 'Kyungsu', 'Sangseok', 'Seohee',
    'Soontack', 'Sukyung', 'Sungjin', 'Taejin', 'Yijoon'
]


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
            setattr(obj, token[1:-1], token)  # vocab.sos = '<sos>' ...

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
        return self.stoi.get(word, self.stoi['<unk>'])


def get_tokenizer(args, special_tokens=None):
    if args['bert']:
        tok = BertTokenizer.from_pretrained('bert-base-cased')
        '''
            bos_token=sos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            additional_special_tokens=speaker_name)
        '''
        for i, spk in enumerate(speaker_name):
            tok.vocab[i] = spk
            #print(tok.vocab[i])
        print(tok.vocab[20])
        for v in itertools.islice(tok.vocab, 100):
            v = speaker_name[0]
            print(v)
        for v in itertools.islice(tok.vocab, 100):
            print(v)
        vocab = torchtext.vocab.Vocab(tok.vocab, min_freq=args['vocab_freq'], specials=[])
        return tok, vocab
    else:
        tokenizers = {
            'nltk': nltk.word_tokenize,
            'nonword': re.compile(r'\W+').split,
        }

        return tokenizers[args['tokenizer'].lower()], None
