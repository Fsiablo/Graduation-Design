import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--char_len', type=int, default=0x9FFF)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--max_len', type=int, default=1)
    return parser.parse_args()
