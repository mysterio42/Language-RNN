import torch
from torch.utils.data import Dataset, DataLoader


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)


class Corpus(Dictionary):

    def __init__(self):
        super(Corpus, self).__init__()

    def _process(self, path):
        next_words = []
        with open(path, 'r') as f:
            for line in f:
                words = line.split()
                if len(words) == 0:
                    continue
                words += ['<eos>']
                next_words.extend(words)
                for word in words:
                    self.add_word(word)
        return next_words

    def tokenize(self, path, seq_len):
        next_words = self._process(path)
        encoded = torch.zeros(size=(len(next_words),), dtype=torch.int64)
        for i, word in enumerate(next_words):
            encoded[i] = self.word2idx[word]

        # num_sequences = encoded.size(0) // seq_len
        # encoded = encoded[:num_sequences * seq_len]
        return encoded


class LanguageData(Dataset, Corpus):

    def __init__(self, path, seq_len, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = self.tokenize(path, seq_len)
        self.seq_len = seq_len

    def __getitem__(self, item):
        x_sl = slice(item, item + self.seq_len)
        y_sl = slice(item + 1, item + 1 + self.seq_len)
        return self.data[x_sl], self.data[y_sl]

    def __len__(self):
        return len(self.data) - self.seq_len - 1


def loaders(path, batch_size, seq_len):
    language_dataset = LanguageData(path=path, seq_len=seq_len)
    vocab_dim = len(language_dataset.word2idx)
    train_loader = DataLoader(dataset=language_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return vocab_dim, train_loader
