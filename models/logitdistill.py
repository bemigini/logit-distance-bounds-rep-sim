from argparse import ArgumentParser

import torch

from utils.args import add_management_args, add_experiment_args
from utils.losses import L1_Loss_Logits, L2_Loss_Logits
from utils.conf import get_device


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Learning via'
                                        'Concept Extractor .')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser

class LogitDistill(torch.nn.Module):
    NAME = 'logitdistill'
    def __init__(self, encoder, args, n_classes=7): 
        super(LogitDistill, self).__init__()

        if args.dataset == 'cifar100':
            n_classes = 100

        self.n_classes = n_classes
        self.n_labels  = n_classes
        self.repr_dim  = encoder.repr_dim

        # bones of the model 
        self.encoder = encoder
        self.linear = torch.nn.Linear(self.repr_dim, self.n_labels, bias=False)
        
        # opt and device
        self.opt = None
        self.args = args
        self.device = get_device()


    def forward(self, x):
        # get embeddings 
        z = self.encoder(x)
        log_y = self.linear(z)

        return {'EMBS': z, 'LOGITS': log_y}

    @staticmethod
    def get_loss(args):
        if args.loss_type=='l1':
            return L1_Loss_Logits
        elif args.loss_type=='l2':
            return L2_Loss_Logits
        else:
            NotImplementedError(f'Loss type not implemented: {args.loss_type}')


    def start_optim(self, args):
        self.opt = torch.optim.Adam(self.parameters(), args.lr)

    def return_unembs(self):
        return self.linear.weight.data.detach().cpu().numpy()
    
    def initialize_weights(self):
        torch.nn.init.xavier_uniform_(self.linear.weight)
        print('Initialized Linear Layer Weights with Xavier Uniform')