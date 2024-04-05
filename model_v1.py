import torch
import torch.nn as nn
from settings import parse_args
#双层LSTM
class Model(nn.Module):
    def __init__(self,args):
        super(Model,self).__init__()
        self.args=args
        self.embedding=nn.Embedding(args.char_len,args.embedding_dim)
        self.encoder=nn.LSTM(input_size=args.embedding_dim,hidden_size=args.hidden_size,num_layers=args.num_layers)
        self.decoder=nn.LSTM(input_size=args.hidden_size,hidden_size=args.hidden_size,num_layers=args.num_layers)
        self.fc=nn.Linear(args.hidden_size,args.char_len)

    def forward(self,inputs, labels=None):
        out=self.embedding(inputs)
        out, (_, _) = self.encoder(out)
        out=torch.split(out,self.args.max_len,dim=1)
        out = out[-1].expand([out[-1].shape[0], self.args.batch_size, self.args.hidden_size])
        print(out.shape)
        out, (_, _) = self.decoder(out)
        out = self.fc(out)
        if labels is not None:
            tmp = out.permute(0, 2, 1)
            labels = labels.squeeze(dim=1)
            loss = nn.functional.cross_entropy(tmp, labels)
            out_t=torch.argmax(out.reshape(-1, self.args.char_len),dim=1)
            label_t=labels.reshape(-1)
            acc = torch.sum(out_t==label_t)/out_t.shape[0]
            return loss, acc
        return out