import os
import io
import pickle
from typing import List, Dict, Tuple

import youtokentome as yttm


PAD_TOKEN = ''
UNK_TOKEN = '<UNK>'
BEG_TOKEN = '\b'
END_TOKEN = '\n'

PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1
BEG_TOKEN_ID = 2
END_TOKEN_ID = 3

tmp_dir = '../tmp'


class TextTokenizer:
    def __init__(self, name):
        self.name = name
        self.itos = dict()
        self.stoi = dict()
        self.is_char_level = False
        self.bpe_model_path = None
        self.bpe = None
        self.vocab_size = -1
        self.max_len = -1

    def fit(self, texts, bpe_items):
        if bpe_items == 0:
            # char-level представление текста
            self.is_char_level = True

            all_chars = set()
            for text in texts:
                all_chars.update(text)

            self.max_len = 2 + max(map(len, texts))

            self.stoi = dict((c, i) for i, c in enumerate(all_chars, start=4))
            self.stoi[PAD_TOKEN] = PAD_TOKEN_ID
            self.stoi[UNK_TOKEN] = UNK_TOKEN_ID
            self.stoi[BEG_TOKEN] = BEG_TOKEN_ID
            self.stoi[END_TOKEN] = END_TOKEN_ID

            self.itos = dict((i, s) for s, i in self.stoi.items())
            self.vocab_size = len(self.stoi)
        else:
            # Используется BPE модель для сегментации текста
            self.is_char_level = False

            # Готовим корпус для обучения BPE модели
            bpe_corpus = os.path.join(tmp_dir, 'bpe_corpus.txt')

            with io.open(bpe_corpus, 'w', encoding='utf-8') as wrt:
               for text in texts:
                   wrt.write(text+'\n')

            self.bpe_model_path = os.path.join(tmp_dir, 'template2poem_keras_transformers.yttm.{}'.format(self.name))
            self.bpe = yttm.BPE.train(data=bpe_corpus, vocab_size=bpe_items, model=self.bpe_model_path,
                                      pad_id=PAD_TOKEN_ID, unk_id=UNK_TOKEN_ID,
                                      bos_id=BEG_TOKEN_ID, eos_id=END_TOKEN_ID)
            os.remove(bpe_corpus)

            self.itos = dict(enumerate(self.bpe.vocab()))
            self.itos[PAD_TOKEN_ID] = PAD_TOKEN
            self.itos[UNK_TOKEN_ID] = UNK_TOKEN
            self.itos[BEG_TOKEN_ID] = BEG_TOKEN
            self.itos[END_TOKEN_ID] = END_TOKEN
            self.vocab_size = len(self.itos)

            self.max_len = 0
            for text in texts:
                tx = self.bpe.encode([text], output_type=yttm.OutputType.SUBWORD)
                self.max_len = max(self.max_len, len(tx[0])+2)

    def get_pad_id(self):
        return PAD_TOKEN_ID

    def save(self, f):
        pickle.dump(self.name, f)
        pickle.dump(self.is_char_level, f)
        pickle.dump(self.bpe_model_path, f)
        pickle.dump(self.vocab_size, f)
        pickle.dump(self.stoi, f)
        pickle.dump(self.itos, f)
        pickle.dump(self.max_len, f)

    @staticmethod
    def load(f):
        t = TextTokenizer(None)
        t.name = pickle.load(f)
        t.is_char_level = pickle.load(f)
        t.bpe_model_path = pickle.load(f)
        t.vocab_size = pickle.load(f)
        t.stoi = pickle.load(f)
        t.itos = pickle.load(f)
        t.max_len = pickle.load(f)
        if t.is_char_level is False:
            t.bpe = yttm.BPE(t.bpe_model_path)
        return t

    def tokenize(self, text):
        if self.is_char_level:
            return [BEG_TOKEN] + list(text) + [END_TOKEN]
        else:
            return [BEG_TOKEN] + self.bpe.encode([text], output_type=yttm.OutputType.SUBWORD) + [END_TOKEN]

    def encode(self, text, pad_to_len=0, add_walls=True):
        if self.is_char_level:
            if add_walls:
                ids = [BEG_TOKEN_ID] + [self.stoi[c] for c in text] + [END_TOKEN_ID]
            else:
                ids = [self.stoi[c] for c in text]
        else:
            if add_walls:
                ids = [BEG_TOKEN_ID] + self.bpe.encode([text], output_type=yttm.OutputType.ID)[0] + [END_TOKEN_ID]
            else:
                raise NotImplementedError()

        if pad_to_len > 0:
            l = len(ids)
            if l < pad_to_len:
                ids += [PAD_TOKEN_ID] * (pad_to_len - l)
            elif l > pad_to_len:
                ids = ids[:pad_to_len]

        return ids

    def decode(self, tok_ids: List[int]):
        #tok_ids = tok_ids.tolist()

        if END_TOKEN_ID in tok_ids:
            tok_ids = tok_ids[:tok_ids.index(END_TOKEN_ID)]

        if self.is_char_level:
            return ''.join(self.itos[i] for i in tok_ids if i not in (PAD_TOKEN_ID, UNK_TOKEN_ID, BEG_TOKEN_ID))
        else:
            sx = [self.itos[i] for i in tok_ids if i not in (PAD_TOKEN_ID, UNK_TOKEN_ID, BEG_TOKEN_ID)]
            return ''.join(sx).replace('▁', ' ').strip()

    def decode_token(self, tok_id):
        assert(self.is_char_level)
        return self.itos[tok_id]
