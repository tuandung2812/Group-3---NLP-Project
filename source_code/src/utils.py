import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm
import youtokentome as yttm
import os
import json
import torch.nn as nn
import math
import datetime
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.utils.data as tud

import random

def save_checkpoint(state, file_name):
    torch.save(state, file_name)

def load_checkpoint(checkpoint_path, model):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_path,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["state_dict"], strict = False)


def training_step(model, optimizer, scheduler, src, tgt, criterion, device):
    """
    Is called every step to train the model
    """

    model.train()

    inp_data = src.to(device)
    target = tgt.to(device)

    # forward prop
    output = model(inp_data, target[:, :-1])

    output = output.reshape(-1, output.shape[2])
    target = target[:, 1:].reshape(-1)

    optimizer.zero_grad()

    loss = criterion(output, target)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    optimizer.step()
    lr = optimizer.param_groups[0]["lr"]
    scheduler.step()

    return loss.item(), lr

def validate(model,dev_loader, criterion, device):
  eval_losses = []
  model.eval()
  with torch.no_grad():
    for idx, (src, tgt) in enumerate(dev_loader):
      inp_data = src.to(device)
      target = tgt.to(device)
      # forward prop
      output = model(inp_data, target[:, :-1])
      output = output.reshape(-1, output.shape[2])
      target = target[:, 1:].reshape(-1)
      eval_loss = criterion(output, target)
      eval_losses.append(eval_loss.item())
  mean_eval_loss = sum(eval_losses) / len(eval_losses)

  return mean_eval_loss

def beam_search(
    model,
    X,
    predictions = 50,
    beam_width = 5,
    batch_size = 64,
):
    with torch.no_grad():
        Y = torch.ones(X.shape[0], 1).to(next(model.parameters()).device).long()
        # The next command can be a memory bottleneck, can be controlled with the batch
        # size of the predict method.
        next_probabilities = model.forward(X, Y)[:, -1, :]
        vocabulary_size = next_probabilities.shape[-1]
        probabilities, next_chars = next_probabilities.squeeze().log_softmax(-1)\
        .topk(k = beam_width, axis = -1)
        Y = Y.repeat((beam_width, 1))
        next_chars = next_chars.reshape(-1, 1)
        Y = torch.cat((Y, next_chars), axis = -1)
        # This has to be minus one because we already produced a round
        # of predictions before the for loop.
        predictions_iterator = range(predictions - 1)
        for i in predictions_iterator:
            dataset = tud.TensorDataset(X.repeat((beam_width, 1, 1)).transpose(0, 1).flatten(end_dim = 1), Y)
            loader = tud.DataLoader(dataset, batch_size = batch_size)
            next_probabilities = []
            iterator = iter(loader)
            for x, y in iterator:
                next_probabilities.append(model.forward(x, y)[:, -1, :].log_softmax(-1))
            next_probabilities = torch.cat(next_probabilities, axis = 0)
            next_probabilities = next_probabilities.reshape((-1, beam_width, next_probabilities.shape[-1]))
            probabilities = probabilities.unsqueeze(-1) + next_probabilities
            probabilities = probabilities.flatten(start_dim = 1)
            probabilities, idx = probabilities.topk(k = beam_width, axis = -1)
            next_chars = torch.remainder(idx, vocabulary_size).flatten().unsqueeze(-1)
            best_candidates = (idx / vocabulary_size).long()
            best_candidates += torch.arange(Y.shape[0] // beam_width, device = X.device).unsqueeze(-1) * beam_width
            Y = Y[best_candidates].flatten(end_dim = -2)
            Y = torch.cat((Y, next_chars), axis = 1)
        return Y.reshape(-1, beam_width, Y.shape[-1]), probabilities