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


def train_model(model, train_loader, learning_rate, clip, n_epochs, save_model):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()
    model.train()
    end = len(train_loader)
    for epoch in range(n_epochs):

        for i, (features, labels) in enumerate(train_loader):

            optimizer.zero_grad()

            outputs = model(features)

            loss = criterion(outputs, labels.reshape(-1))

            loss.backward()

            clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            if i % 20 == 0:
                print(
                    f'Epoch [{epoch + 1}/{n_epochs}], Step[{i}/{end}], Loss: {loss.item()}, Perplexity: {np.exp(loss.item())}')
    if save_model:
        torch.save(model.state_dict(), 'weights/model-' + generate_model_name(5) + '.pkl')
