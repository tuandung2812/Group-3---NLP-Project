# import spacy
import collections
# from tqdm.notebook import tqdm
import youtokentome as yttm
import os
import json
import datetime

# spacy_eng = spacy.load("en_core_web_sm")


class Vocabulary():
    '''Vocabulary class: extract all words from corpus, save all words that appears more than freq_threshold times'''

    def __init__(self, freq_threshold=3):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>", 4: "<upper>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3, "<upper>": 4}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def build_vocab(self, sentence_list):
        # Build the vocab
        idx = 4
        word_counter = collections.Counter([word for sentence in sentence_list for word in sentence])

        for word, count in word_counter.items():
            if count >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
        # print(self.stoi)

    def numericalize(self, sent):
        # Turn a sentence into list of word ID
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in sent
        ]

    def save_vocab(self, vocab_path):
        #     with open(itos_path, 'w') as itos_f:
        #         json.dump(self.itos, itos_f)
        #     with open(stoi_path, 'w') as stoi_f:
        #         json.dump(self.stoi, stoi_f)
        with open(vocab_path, 'w') as f:
            vocab = {'itos': self.itos, 'stoi': self.stoi}
            json.dump(vocab, f)

    def load_vocab(self, vocab_path):
        with open(vocab_path, 'r') as f:
            #         print(type(itos_f.read()))
            vocab = json.load(f)
            self.itos = vocab['itos']
            self.stoi = vocab['stoi']


