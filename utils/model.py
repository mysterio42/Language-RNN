import random
import string

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam


def load_model(model, path_extension: str):
    model.load_state_dict(torch.load(path_extension))


def generate_model_name(size=5):
    letters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters) for _ in range(size))


def train_model(model, data, seq_len, learning_rate, clip, n_epochs, save_model):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()
    model.train()
    end = data.size(1) - seq_len
    for epoch in range(n_epochs):

        for i in range(0, end, seq_len):
            inputs = data[:, i:i + seq_len]
            targets = data[:, (i + 1):(i + 1) + seq_len]

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, targets.reshape(-1))

            loss.backward()

            clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            if i % (seq_len * 5) == 0:
                print(
                    f'Epoch [{epoch + 1}/{n_epochs}], Step[{i}/{end}], Loss: {loss.item()}, Perplexity: {np.exp(loss.item())}')
    if save_model:
        torch.save(model.state_dict(), 'weights/model-' + generate_model_name(5) + '.pkl')
