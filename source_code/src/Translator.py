from src.vocabulary import Vocabulary
from src.dataset import StanfordEnViDataset,get_loader, add_upper_token
from src.model import EnViTransformer
import datetime

from src.utils import save_checkpoint, load_checkpoint, training_step, validate, beam_search
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm
import youtokentome as yttm
import os
import json
import torch.nn as nn
import math

from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

import random
# import spacy
import collections
from tqdm.notebook import tqdm
import youtokentome as yttm
import os
import json
import random
import underthesea
import transformers
import nltk
import nltk.data
nltk.download('punkt')


class TransformerTranslator():

    def __init__(self, config, device, input_type='bpe', back_translation=False, load_pretrained=True):
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.config = config
        self.input_type = input_type
        self.device = device
        self.back_translation = back_translation
        self.train_src_file = self.config['data']['train']['src_file']
        self.train_tgt_file = self.config['data']['train']['tgt_file']

        if self.input_type == 'bpe':
            self.bpe_en_file = self.config['utils']['bpe']['bpe_en_file']
            self.bpe_vi_file = self.config['utils']['bpe']['bpe_vi_file']
            yttm.BPE.train(data=self.train_src_file, vocab_size=5000, model=self.bpe_en_file)
            yttm.BPE.train(data=self.train_tgt_file, vocab_size=5000, model=self.bpe_vi_file)
            self.bpe_en = yttm.BPE(model=self.bpe_en_file)
            self.bpe_vi = yttm.BPE(model=self.bpe_vi_file)

        else:
            self.bpe_en_file = None
            self.bpe_vi_file = None
            self.bpe_en = bpe_en = None
            self.bpe_vi = bpe_vi = None

        self.dev_src_file = self.config['data']['dev']['src_file']
        self.dev_tgt_file = self.config['data']['dev']['tgt_file']
        # training hparam
        self.num_epochs = self.config['train']['num_epochs']
        self.learning_rate = self.config['train']['learning_rate']
        self.batch_size = self.config['train']['batch_size']
        # set up training phase

        self.max_sent_len = self.config['model']['max_sent_len']

        self.en_vocab_path = self.config['utils']['vocab']['en']
        self.vi_vocab_path = self.config['utils']['vocab']['vi']

        if os.path.isfile(self.en_vocab_path) and os.path.isfile(self.vi_vocab_path):
            self.en_vocab = Vocabulary(3)
            self.en_vocab.load_vocab(self.en_vocab_path)
            self.vi_vocab = Vocabulary(3)
            self.vi_vocab.load_vocab(self.vi_vocab_path)
        else:
            self.train_loader, _, self.en_vocab, self.vi_vocab = get_loader(self.train_src_file, self.train_tgt_file,
                                                                            en_vocab_path=self.en_vocab_path,
                                                                            vi_vocab_path=self.vi_vocab_path,
                                                                            input_type=self.input_type,
                                                                            bpe_en_file=self.bpe_en_file,
                                                                            bpe_vi_file=self.bpe_vi_file,
                                                                            max_seq_len=self.max_sent_len,
                                                                            batch_size=self.batch_size,
                                                                            back_translation=self.back_translation)

            self.dev_loader, _, _, _ = get_loader(self.dev_src_file, self.dev_tgt_file,
                                                  en_vocab_path=self.en_vocab_path,
                                                  vi_vocab_path=self.vi_vocab_path,
                                                  input_type=self.input_type,
                                                  bpe_en_file=self.bpe_en_file,
                                                  bpe_vi_file=self.bpe_vi_file,
                                                  max_seq_len=self.max_sent_len,
                                                  batch_size=self.batch_size
                                                  , back_translation=self.back_translation)

        if self.back_translation == False:
            self.src_vocab = self.en_vocab
            self.tgt_vocab = self.vi_vocab
        else:
            self.src_vocab = self.vi_vocab
            self.tgt_vocab = self.en_vocab

            # model hparam
        self.src_vocab_size = len(self.src_vocab)
        self.tgt_vocab_size = len(self.tgt_vocab)
        self.embedding_size = self.config['model']['embedding_size']
        self.num_heads = self.config['model']['num_heads']
        self.num_encoder_layers = self.config['model']['num_encoder_layers']
        self.num_decoder_layers = self.config['model']['num_decoder_layers']
        self.dropout = self.config['model']['dropout']
        self.max_len = self.max_sent_len
        self.forward_expansion = self.config['model']['forward_expansion']
        self.checkpoint_path = self.config['model']['checkpoint_path']
        self.src_pad_idx = self.en_vocab.stoi["<PAD>"]
        # self.pad_idx = en_vocab.stoi["<PAD>"]

        self.model = EnViTransformer(
            self.embedding_size,
            self.src_vocab_size,
            self.tgt_vocab_size,
            self.src_pad_idx,
            self.num_heads,
            self.num_encoder_layers,
            self.num_decoder_layers,
            self.forward_expansion,
            self.dropout,
            self.max_len,
            self.device,
        ).to(self.device)

        if os.path.isfile(self.checkpoint_path) and load_pretrained:
            load_checkpoint(self.checkpoint_path, self.model)

        self.train_loss_runs = self.config['utils']['training_runs']['train_loss']
        self.dev_loss_runs = self.config['utils']['training_runs']['dev_loss']
        self.lr_runs = self.config['utils']['training_runs']['learning_rate']

    def train(self):
        train_loader, _, en_vocab, vi_vocab = get_loader(self.train_src_file, self.train_tgt_file,
                                                         en_vocab_path=self.en_vocab_path,
                                                         vi_vocab_path=self.vi_vocab_path,
                                                         input_type=self.input_type,
                                                         bpe_en_file=self.bpe_en_file,
                                                         bpe_vi_file=self.bpe_vi_file,
                                                         max_seq_len=self.max_sent_len,
                                                         batch_size=self.batch_size,
                                                         back_translation=self.back_translation)

        dev_loader, _, _, _ = get_loader(self.dev_src_file, self.dev_tgt_file,
                                         en_vocab_path=self.en_vocab_path,
                                         vi_vocab_path=self.vi_vocab_path,
                                         input_type=self.input_type,
                                         bpe_en_file=self.bpe_en_file,
                                         bpe_vi_file=self.bpe_vi_file,
                                         max_seq_len=self.max_sent_len,
                                         batch_size=self.batch_size
                                         , back_translation=self.back_translation)

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.99, 0.98))

        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(
            self.num_epochs * len(train_loader) * 0.15),
                                                                 num_training_steps=self.num_epochs * len(
                                                                     train_loader))

        criterion = nn.CrossEntropyLoss(ignore_index=self.src_pad_idx)

        # Tensorboard for nice plot
        writer_train = SummaryWriter(self.train_loss_runs)
        writer_dev = SummaryWriter(self.dev_loss_runs)
        writer_lr = SummaryWriter(self.lr_runs)

        step = 0
        best_val_loss = 999
        lrs = []
        train_loss = []
        val_loss = []

        start_time = datetime.datetime.now()
        for epoch in range(self.num_epochs):
            print(
                "**************************************************************************    Epoch number {}    **************************************************************************".format(
                    epoch + 1))
            losses = []
            for batch_idx, (src, tgt) in enumerate(tqdm(train_loader, position=0, leave=True)):
                step += 1
                loss, lr = training_step(self.model, optimizer, scheduler, src, tgt, criterion, self.device)
                lrs.append(lr)
                losses.append(loss)

                writer_lr.add_scalar('Learning rate', lr, step)

                if (step == 1) or step % 250 == 0 or step == len(train_loader) * self.num_epochs:
                    eval_loss = validate(self.model, dev_loader, criterion, self.device)

                    train_loss.append(sum(losses) / len(losses))
                    val_loss.append(eval_loss)
                    current_time = datetime.datetime.now()
                    time_taken = current_time - start_time
                    print("Current step: {0}, epoch: {1}".format(step, epoch + 1), end="    ")
                    print("Training loss: ", sum(losses) / len(losses), end="    ")
                    print("Evaluation loss: ", eval_loss, end="    ")
                    print("Time elapsed: ", time_taken, end="    ")
                    writer_train.add_scalar('Loss', sum(losses) / len(losses), step)
                    writer_dev.add_scalar('Loss', eval_loss, step)

                    if eval_loss < best_val_loss:
                        print("(Best weights saved!) ")
                        best_val_loss = eval_loss
                        checkpoint = {
                            "state_dict": self.model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict()}

                        # Save the checkpoint with the lowest loss
                        save_checkpoint(checkpoint, self.checkpoint_path)


                    else:
                        print()

    def inference(self, inp_data, tensor_input=False):

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True

        model = self.model
        load_checkpoint(self.checkpoint_path, model)

        # Tokenize our source sentence
        if not tensor_input:
            src_sents = [sent.strip() for sent in inp_data]
            # print(src_sents)
            src_tensors = []
            for i in range(len(src_sents)):
                src_sent = src_sents[i]
                if self.input_type == 'bpe':
                    if self.back_translation is False:
                        src_sent = self.bpe_en.encode(src_sent, output_type=yttm.OutputType.SUBWORD)
                    else:
                        src_sent = self.bpe_vi.encode(src_sent, output_type=yttm.OutputType.SUBWORD)

                elif self.input_type == 'word_segmentation':
                    if self.back_translation is False:
                        src_sent = nltk.word_tokenize(src_sent)
                    else:
                        src_sent = underthesea.word_tokenize(src_sent)

                elif self.input_type == 'tokenization':
                    src_sent = nltk.word_tokenize(src_sent)

                src_sent = add_upper_token(src_sent)

                if len(src_sent) + 2 >= self.max_sent_len:
                    src_sent = src_sent[0:self.max_sent_len - 2]

                    # Numericalize our input sentence into tensors
                src_sent_numericalized = [self.src_vocab.stoi["<SOS>"]]
                src_sent_numericalized += self.src_vocab.numericalize(src_sent)
                src_tensor = np.full(self.max_sent_len, self.src_vocab.stoi["<PAD>"])
                src_tensor[:len(src_sent_numericalized)] = src_sent_numericalized
                src_tensor = torch.Tensor(src_tensor).long()
                src_tensors.append(src_tensor)
            #             print(src_tensor.shape)
            #         src_tensor = src_tensor.unsqueeze(0)
            src_tensors = torch.stack(src_tensors, dim=0)
            inp_data = src_tensors.to(self.device)

        else:
            inp_data = inp_data.to(self.device)

        # Generate predictions using beam search, with k = 5
        predictions, log_probabilities = beam_search(model, inp_data, predictions=self.max_sent_len, beam_width=5)
        predict_sents = []
        for p in predictions:
            predict_sents.append([[self.tgt_vocab.itos[str(i)] for i in l] for l in p.tolist()])

        if not tensor_input:
            results = predict_sents
            #         print(results)
            # Remove and <SOS> and <EOS> tokens and every tokens that come after <EOS>, remove "_" from bpe
            for i in range(len(results)):
                for j in range(len(results[i])):
                    translation = results[i][j]
                    translation.pop(0)
                    if '<EOS>' in translation:
                        end_index = translation.index('<EOS>')
                        translation = translation[0:end_index]

                    if self.input_type == "bpe":
                        translation = ("".join(translation))
                    else:
                        translation = (" ".join(translation))
                    translation = translation.replace("▁", " ")
                    translation = translation.strip()
                    if translation[-2:] == " .":
                        translation = translation[:-2] + "."
                    if translation[-2:] == " ?":
                        translation = translation[:-2] + "?"
                    if translation[-2:] == " !":
                        translation = translation[:-2] + "!"

                    translation = translation.replace("<upper>", " <upper>")
                    translation = translation.split(" ")
                    final_translation = ""
                    for k in range(len(translation)):
                        word = translation[k]
                        if word != "<upper>":
                            if translation[k - 1] == '<upper>':
                                if len(word) > 1:
                                    word = word[0].upper() + word[1:]
                                final_translation += word + " "
                            else:
                                final_translation += word + " "

                    results[i][j] = final_translation

        else:
            results = predict_sents

            for i in range(len(results)):
                if results[i - 1] == '<upper>':
                    if results[i][0].isalpha():
                        results[i] = results[i][0].upper() + results[i][1:]
                    else:
                        results[i] = results[i][0] + results[i][1].upper() + results[i][2:]

            # Remove and <SOS> and <EOS> tokens and every tokens that come after <EOS>, remove "_" from bpe
            for i in range(len(results)):
                for j in range(len(results[i])):
                    translation = results[i][j]
                    translation.pop(0)
                    if '<EOS>' in translation:
                        end_index = translation.index('<EOS>')
                        translation = translation[0:end_index]
                    if self.input_type == "bpe":
                        translation = ("".join(translation))
                    else:
                        translation = (" ".join(translation))
                    translation = translation.replace("▁", " ")
                    translation = translation.strip()
                    if translation[-2:] == " .":
                        translation = translation[:-2] + "."
                    if translation[-2:] == " ?":
                        translation = translation[:-2] + "?"
                    if translation[-2:] == " !":
                        translation = translation[:-2] + "!"

                    translation = translation.replace("<upper>", " <upper>")
                    translation = translation.split(" ")
                    final_translation = ""
                    for k in range(len(translation)):
                        word = translation[k]
                        if word != "<upper>":
                            if translation[k - 1] == '<upper>':
                                if len(word) > 1:
                                    word = word[0].upper() + word[1:]
                                final_translation += word + " "
                            else:
                                final_translation += word + " "

                    results[i][j] = final_translation.strip()

        results = [sents[0] for sents in results]

        return results