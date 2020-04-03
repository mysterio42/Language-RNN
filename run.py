import argparse

import torch

from network.net import LanguageRNN
from utils.data import Corpus
from utils.model import train_model, load_model

torch.manual_seed(1)


def parse_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected')

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Model learning rate  default: 0.001')

    parser.add_argument('--clip', type=float, default=0.5,
                        help='Embedding dim default: 0.5')

    parser.add_argument('--embed', type=int, default=256,
                        help='Embedding dim default: 256')

    parser.add_argument('--hidden', type=int, default=512,
                        help='Hidden dim default: 512')

    parser.add_argument('--layers', type=int, default=2,
                        help='LSTM Layer dim default: 2')

    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of Epochs default: 5')

    parser.add_argument('--batch', type=int, default=20,
                        help='batch size default: 20')

    parser.add_argument('--seq', type=int, default=30,
                        help='Sequence size default: 30')

    parser.add_argument('--load', type=str2bool, default=True,
                        help='True: Load trained model  False: Train model default: True')
    parser.print_help()
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    corpus = Corpus()
    encoded = corpus.tokenize('data/train.txt')
    vocab_dim = len(corpus)

    model = LanguageRNN(vocab_dim=vocab_dim,
                        embed_dim=args.embed,
                        hidden_dim=args.hidden,
                        layer_dim=args.layers)
    if args.load:
        # model_name = 'model-8zi3p.pkl'
        # load_model(model, 'weights/{}'.format(model_name))
        pass

    else:
        train_model(model=model, data=encoded,
                    seq_len=args.seq, learning_rate=args.lr, clip=args.clip,n_epochs=args.epochs,
                    save_model=True)
