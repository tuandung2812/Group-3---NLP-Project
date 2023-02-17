from src.vocabulary import Vocabulary
import datetime
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm
import youtokentome as yttm
import os
import json
import random
import underthesea
import nltk
import nltk.data
nltk.download('punkt')

seed  = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



def add_upper_token(sent: list):
    i = 0
    sent_len = len(sent)

    new_sent = []
    while i < sent_len:
        word = sent[i]
        upper = False
        for char in word:
            if char.isupper():
                upper = True
        if upper:
            new_sent.append('<upper>')
        new_sent.append(word.lower())
        i += 1
        sent_len = len(sent)

    return new_sent


class StanfordEnViDataset(Dataset):
    '''The dataset class:
      Read all sentence pairs from src_data_file and tgt_data_file
      Create vobulary, then turns all these pairs into list of word IDs'''

    def __init__(self, src_data_file, tgt_data_file, en_vocab_path, vi_vocab_path, input_type="bpe", bpe_en_file=None,
                 bpe_vi_file=None, max_seq_len=512, back_translation=False):
        '''
        Params:
          src_data_file: path to sents of source data
          tgt_data_file: path to sents of target data
          max_seq_len: maximum number of tokens in a sentence. Truncate our sentences if the input is longer
          en_vocab, vi_vocab: objects of Vocabulary class. If we already have en_vocab and vi_vocab, we avoid creating them again.
          back_translation: if we're using this for back translation, set this to True
        '''
        self.max_seq_len = max_seq_len
        self.src, self.tgt = [], []

        if input_type == 'bpe':
            bpe_en = yttm.BPE(model=bpe_en_file)
            bpe_vi = yttm.BPE(model=bpe_vi_file)

        # Create and build dataset
        with open(src_data_file, encoding="utf-8") as src_f, open(tgt_data_file, encoding="utf-8") as tgt_f:

            for src_sent, tgt_sent in zip(src_f, tgt_f):
                # src_sent = src_sent.strip().lower()
                # tgt_sent = tgt_sent.strip().lower()
                if input_type == 'bpe':
                    if back_translation is False:
                        en_sent = bpe_en.encode(src_sent, output_type=yttm.OutputType.SUBWORD)
                        vi_sent = bpe_vi.encode(tgt_sent, output_type=yttm.OutputType.SUBWORD)

                        en_sent = add_upper_token(en_sent)
                        vi_sent = add_upper_token(vi_sent)
                        if len(en_sent) + 2 >= max_seq_len:
                            en_sent = en_sent[0:max_seq_len - 2]

                        if len(vi_sent) + 2 >= max_seq_len:
                            vi_sent = vi_sent[0:max_seq_len - 2]

                        self.src.append(en_sent)
                        self.tgt.append(vi_sent)

                    else:
                        en_sent = bpe_en.encode(tgt_sent, output_type=yttm.OutputType.SUBWORD)
                        vi_sent = bpe_vi.encode(src_sent, output_type=yttm.OutputType.SUBWORD)

                        en_sent = add_upper_token(en_sent)
                        vi_sent = add_upper_token(vi_sent)

                        if len(en_sent) + 2 >= max_seq_len:
                            en_sent = en_sent[0:max_seq_len - 2]

                        if len(vi_sent) + 2 >= max_seq_len:
                            vi_sent = vi_sent[0:max_seq_len - 2]
                        # if len(en_sent) + 2 < max_seq_len and len(vi_sent) + 1 < max_seq_len:
                        self.src.append(vi_sent)
                        self.tgt.append(en_sent)

                elif input_type == 'word_segmentation':

                    if back_translation is False:
                        # en_sent = bpe_en.encode(src_sent, output_type=yttm.OutputType.SUBWORD, dropout_prob = 0.1)
                        en_sent = nltk.word_tokenize(src_sent)
                        vi_sent = underthesea.word_tokenize(tgt_sent)

                        en_sent = add_upper_token(en_sent)
                        vi_sent = add_upper_token(vi_sent)

                        # vi_sent = bpe_vi.encode(tgt_sent, output_type=yttm.OutputType.SUBWORD, dropout_prob = 0.1)
                        if len(en_sent) + 1 >= max_seq_len:
                            en_sent = en_sent[0:max_seq_len - 2]

                        if len(vi_sent) + 1 >= max_seq_len:
                            vi_sent = vi_sent[0:max_seq_len - 2]

                        self.src.append(en_sent)
                        self.tgt.append(vi_sent)

                    else:
                        en_sent = nltk.word_tokenize(tgt_sent)
                        vi_sent = underthesea.word_tokenize(src_sent)

                        en_sent = add_upper_token(en_sent)
                        vi_sent = add_upper_token(vi_sent)

                        # en_sent = bpe_en.encode(tgt_sent, output_type=yttm.OutputType.SUBWORD, dropout_prob = 0.1)
                        # vi_sent = bpe_vi.encode(src_sent, output_type=yttm.OutputType.SUBWORD, dropout_prob = 0.1)
                        if len(en_sent) + 1 >= max_seq_len:
                            en_sent = en_sent[0:max_seq_len - 2]

                        if len(vi_sent) + 1 >= max_seq_len:
                            vi_sent = vi_sent[0:max_seq_len - 2]
                        self.src.append(vi_sent)
                        self.tgt.append(en_sent)

                elif input_type == 'tokenization':
                    if back_translation is False:
                        # en_sent = bpe_en.encode(src_sent, output_type=yttm.OutputType.SUBWORD, dropout_prob = 0.1)
                        en_sent = nltk.word_tokenize(src_sent)
                        vi_sent = nltk.word_tokenize(tgt_sent)

                        en_sent = add_upper_token(en_sent)
                        vi_sent = add_upper_token(vi_sent)

                        # vi_sent = bpe_vi.encode(tgt_sent, output_type=yttm.OutputType.SUBWORD, dropout_prob = 0.1)
                        if len(en_sent) + 1 >= max_seq_len:
                            en_sent = en_sent[0:max_seq_len - 2]

                        if len(vi_sent) + 1 >= max_seq_len:
                            vi_sent = vi_sent[0:max_seq_len - 2]

                        self.src.append(en_sent)
                        self.tgt.append(vi_sent)

                    else:
                        en_sent = nltk.word_tokenize(tgt_sent)
                        vi_sent = nltk.word_tokenize(src_sent)

                        en_sent = add_upper_token(en_sent)
                        vi_sent = add_upper_token(vi_sent)

                        if len(en_sent) + 1 >= max_seq_len:
                            en_sent = en_sent[0:max_seq_len - 2]

                        if len(vi_sent) + 1 >= max_seq_len:
                            vi_sent = vi_sent[0:max_seq_len - 2]
                        self.src.append(vi_sent)
                        self.tgt.append(en_sent)

            # Create en_vocab and vi_vocab if neccessary
            if os.path.isfile(en_vocab_path) == False or os.path.isfile(vi_vocab_path) == False:
                self.vi_vocab = Vocabulary(3)
                self.en_vocab = Vocabulary(3)
                if back_translation is False:
                    self.vi_vocab.build_vocab(self.tgt)
                    self.en_vocab.build_vocab(self.src)
                else:
                    self.vi_vocab.build_vocab(self.src)
                    self.en_vocab.build_vocab(self.tgt)

                self.vi_vocab.save_vocab(vi_vocab_path)
                self.en_vocab.save_vocab(en_vocab_path)

            else:
                self.vi_vocab = Vocabulary(3)
                self.en_vocab = Vocabulary(3)
                self.en_vocab.load_vocab(en_vocab_path)
                self.vi_vocab.load_vocab(vi_vocab_path)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, index):
        # Override __getitem__ method of parent class (torch.utils.data.Dataset class)
        en_sent_numericalized = [self.en_vocab.stoi["<SOS>"]]
        en_sent_numericalized += self.en_vocab.numericalize(self.src[index])
        en_tensor = np.full(self.max_seq_len, self.en_vocab.stoi["<PAD>"])
        en_tensor[:len(en_sent_numericalized)] = en_sent_numericalized
        en_tensor = torch.Tensor(en_tensor).long()

        vi_sent_numericalized = [self.vi_vocab.stoi["<SOS>"]]
        vi_sent_numericalized += self.vi_vocab.numericalize(self.tgt[index])
        vi_sent_numericalized += [self.vi_vocab.stoi["<EOS>"]]
        vi_tensor = np.full(self.max_seq_len, self.vi_vocab.stoi["<PAD>"])
        vi_tensor[:len(vi_sent_numericalized)] = vi_sent_numericalized
        vi_tensor = torch.Tensor(vi_tensor).long()

        return en_tensor, vi_tensor


def get_loader(
        src_data_file,
        tgt_data_file,
        en_vocab_path,
        vi_vocab_path,
        input_type='bpe',
        bpe_en_file=None,
        bpe_vi_file=None,
        max_seq_len=512,
        en_vocab=None,
        vi_vocab=None,
        back_translation=False,
        batch_size=16,
        num_workers=0,
        shuffle=True,
        pin_memory=True
):
    '''
    Function to create DataLoader: group sentences into batches of size batch_size

    Params:
      src_data_file, tgt_data_file: path to source and target data file
      max_seq_len: max number of tokens in a sentence
      en_vocab, vi_vocab: object of Vocabulary class. If we already have en_vocab and vi_vocab, we avoid creating them again
      back_translation: if we're using this for back translation, set this to True
      batch_size: number of sentences in a batch
      num_workers, shuffle, pin_memory: not very important
    Returns:
      loader: object of DataLoader class
      dataset: object of StanfordEnViDataset class
      en_vocab, vi_vocab: object of Vocabulary class. This is vocab of English and Vietnamese
    '''
    dataset = StanfordEnViDataset(src_data_file, tgt_data_file, en_vocab_path, vi_vocab_path, input_type, bpe_en_file,
                                  bpe_vi_file, max_seq_len, back_translation)
    en_vocab = dataset.en_vocab
    vi_vocab = dataset.vi_vocab
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory
    )
    return loader, dataset, en_vocab, vi_vocab