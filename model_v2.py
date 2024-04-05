import torch
import torch.nn as nn
from settings import parse_args
#Encoder+Decoder
class Model(nn.Module):
    def __init__(self,args):
        super(Model,self).__init__()
        self.args=args
        self.encoder=Encoder(args)
        self.decoder=Decoder(args)

    def forward(self,X, labels=None):
        state = self.decoder.init_state(self.encoder(X))
        out, state = self.decoder(X, state)
        if labels is not None:
            tmp = out.permute(0, 2, 1)

            labels = labels.squeeze(dim=1)
            loss = nn.functional.cross_entropy(tmp, labels)
            out_t=torch.argmax(out.reshape(-1, self.args.char_len),dim=1)
            label_t=labels.reshape(-1)
            acc = torch.sum(out_t==label_t)/out_t.shape[0]
            return loss, acc
        return out
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder,self).__init__()
        self.embedding=nn.Embedding(args.char_len,args.embedding_dim)
        self.rnn = nn.GRU(args.embedding_dim, args.hidden_size, args.num_layers,
                          dropout=args.dropout)
    def forward(self,X):
        X = self.embedding(X)
        X = X.permute(1, 0, 2)
        output, state = self.rnn(X)
        return output, state
class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(args.char_len, args.embedding_dim)
        self.rnn = nn.GRU(args.embedding_dim+args.hidden_size, args.hidden_size, args.num_layers,
                          dropout=args.dropout)
        self.dense = nn.Linear(args.hidden_size, args.char_len)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]
    def forward(self,X,state):
        X = self.embedding(X).permute(1, 0, 2)
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        return output, state