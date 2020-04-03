import torch
import torch.nn as nn


class LanguageRNN(nn.Module):

    def __init__(self, vocab_dim,embed_dim, hidden_dim, layer_dim):
        super(LanguageRNN, self).__init__()

        self.layer_dim = layer_dim

        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(num_embeddings=vocab_dim,
                                      embedding_dim=embed_dim)

        self.lstm = nn.LSTM(input_size=embed_dim,
                            hidden_size=hidden_dim,
                            num_layers=layer_dim,
                            batch_first=True)

        self.fc = nn.Linear(in_features=hidden_dim, out_features=vocab_dim)

    def forward(self, x):

        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).detach()
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).detach()

        x = self.embedding(x)

        out, hn = self.lstm(x, (h0,c0))

        # out = out.contiguous().view(-1, self.hidden_dim)

        out = out.reshape(out.size(0)*out.size(1),out.size(2))

        out = self.fc(out)

        return out
