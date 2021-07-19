import pickle
import os
import pathlib

import numpy as np

from .transcriber_model import create_model
from .text_tokenizer import TextTokenizer
from .special_tokens import *


class Text2Phonems:
    def __init__(self):
        self.config = None
        self.input_tokenizer = None
        self.output_tokenizer = None
        self.model = None

    def load(self, model_dir=None):
        if model_dir is None:
            model_dir = str(pathlib.Path(__file__).resolve().parent)

        with open(os.path.join(model_dir, 'text2transcription.config'), 'rb') as f:
            self.config = pickle.load(f)
            self.input_tokenizer = TextTokenizer.load(f)
            self.output_tokenizer = TextTokenizer.load(f)

        _, _, self.model = create_model(self.config, self.input_tokenizer, self.output_tokenizer)
        self.model.load_weights(filepath=os.path.join(model_dir, 'text2transcription.model'))

        self.max_seq_len = self.config['max_seq_len']

    def convert1(self, text):
        tokenized_input_sentence = np.zeros((1, self.max_seq_len), dtype=np.int32)
        tokenized_input_sentence[0, :] = self.input_tokenizer.encode(text, self.max_seq_len)

        tokenized_target_sentence = np.zeros((1, self.max_seq_len), dtype=np.int32)
        tokenized_target_sentence[0, 0] = BEG_TOKEN_ID

        for i in range(self.max_seq_len-1):
            predictions = self.model({'encoder_inputs': tokenized_input_sentence,
                                      'decoder_inputs': tokenized_target_sentence})

            sampled_token_index = np.argmax(predictions[0, i, :])
            tokenized_target_sentence[0, i+1] = sampled_token_index

        predicted_output = self.output_tokenizer.decode(tokenized_target_sentence[0, :].tolist())

        return predicted_output


if __name__ == '__main__':
    t2f = Text2Phonems()
    t2f.load('../tmp')
    res = t2f.convert1('кошка ловит мышку')
    print(res)



