import torch


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

    def tokenize(self,path,batch_size=20):
        torch.set_printoptions(edgeitems=80)
        next_words = self._process(path)
        encoded = torch.zeros(size=(len(next_words),), dtype=torch.int64)
        for i, word in enumerate(next_words):
            encoded[i] = self.word2idx[word]

        num_batches = encoded.size(0) // batch_size
        encoded = encoded[:num_batches*batch_size]
        return encoded.view(size=(batch_size,-1))


    def __len__(self):
        return len(self.word2idx)


