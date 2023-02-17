import torch
import torch.nn as nn
import math

from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import numpy as np
import random
import datetime
seed  = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class EnViTransformer(nn.Module):
    '''
    Class for Transformer architecture
    '''

    def __init__(
            self,
            embedding_size,
            src_vocab_size,
            tgt_vocab_size,
            src_pad_idx,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
            max_seq_len,
            device,
    ):
        '''Params:
          embedding_size: size of word embedding
          src_vocab_size, tgt_vocab_size: size of source and target vocab
          src_pad_idx: index of <PAD> token of source vocabulary
          num_heads: number of attention heads
          num_encoder_layers: number of encoder layers
          forward_expansion: size of vector after forward expansion
          dropout: dropout rate
          max_seq_len: max number of tokens in a sentence
          device: device (GPU or CPU)'''

        super(EnViTransformer, self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.src_position_embedding = nn.Embedding(max_seq_len, embedding_size)
        self.tgt_word_embedding = nn.Embedding(tgt_vocab_size, embedding_size)
        self.tgt_position_embedding = nn.Embedding(max_seq_len, embedding_size)

        self.device = device
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(embedding_size, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        '''Make mask to skip computation on <PAD> tokens'''
        src_mask = src == self.src_pad_idx

        # (N, src_len)
        return src_mask.to(self.device)

    def forward(self, src, tgt):
        # src, tgt: (N, max_seq_len)
        N, src_seq_length = src.shape
        N, tgt_seq_length = tgt.shape

        # position embedding size (N, max_seq_len)
        src_positions = (
            torch.arange(0, src_seq_length, dtype=torch.int64)
                .unsqueeze(0)
                .expand(N, src_seq_length)
                .to(self.device)
        )
        tgt_positions = (
            torch.arange(0, tgt_seq_length, dtype=torch.int64)
                .unsqueeze(0)
                .expand(N, tgt_seq_length)
                .to(self.device)
        )

        # shape: (N, max_seq_len, embedding_size)
        embed_src = self.dropout(
            (self.src_word_embedding(src) + self.src_position_embedding(src_positions))
        )
        embed_trg = self.dropout(
            (self.tgt_word_embedding(tgt) + self.tgt_position_embedding(tgt_positions))
        )

        src_padding_mask = self.make_src_mask(src)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_length).to(
            self.device
        )

        # shape: (N, max_seq_len, embedding_size)
        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=tgt_mask,
        )
        # shape: (N, max_seq_len, tgt_vocab_size)
        out = self.fc_out(out)
        return out
