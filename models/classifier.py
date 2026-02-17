from argparse import ArgumentParser

import torch

from utils.args import add_management_args, add_experiment_args
from utils.losses import Classification_Loss
from utils.conf import get_device


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Learning via'
                                        'Concept Extractor .')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser

class Classifier(torch.nn.Module):
    NAME = 'classifier'
    def __init__(self, encoder, args, n_classes=7): 
        super(Classifier, self).__init__()

        if args.dataset == 'cifar100':
            n_classes = 100

        self.n_classes = n_classes
        self.n_labels  = n_classes
        self.repr_dim  = encoder.repr_dim

        # bones of the model 
        self.encoder = encoder

        self.linear = torch.nn.Linear(self.repr_dim, self.n_labels, bias=False)

        # number of images, and how to split them
        
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
        return Classification_Loss
        # if args.dataset in ['addmnist', 'shortmnist']:
        #     return ADDMNIST_Concept_Match
        # else: return NotImplementedError('Wrong Choice')
        
    def start_optim(self, args):
        self.opt = torch.optim.Adam(self.parameters(), args.lr)

    def return_unembs(self):
        return self.linear.weight.data.detach().cpu().numpy()
    
    def initialize_weights(self):
        torch.nn.init.xavier_uniform_(self.linear.weight)
        print('Initialized Linear Layer Weights with Xavier Uniform')