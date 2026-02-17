import sys, os

import torch
import argparse
import importlib
import setproctitle, socket, uuid
import datetime

import models

from dsets import get_dataset
from dsets.preprocess_sub import get_and_save_SUB_dino_embs
from models import get_model, get_teacher_model, get_teacher_encoder
from utils.train import train
from utils.conf import set_random_seed
from utils.args import add_management_args, get_all_models
from utils.checkpoint import create_load_ckpt

from utils.distill import distill_knowledge_to_dataset

from utils.posthoc_eval import posthoc_eval

conf_path = os.getcwd() + "."
sys.path.append(conf_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Logit distillation', allow_abbrev=False)
    parser.add_argument('--model', type=str,default='cext', help='Model for inference.', choices=get_all_models())
    parser.add_argument('--load_best_args', action='store_true', help='Loads the best arguments for each method, '
                             'dataset and memory buffer.') 
    
    torch.set_num_threads(4)

    add_management_args(parser)
    args = parser.parse_known_args()[0]
    mod = importlib.import_module('models.' + args.model)

    # LOAD THE PARSER SPECIFIC OF THE MODEL, WITH ITS SPECIFICS
    get_parser = getattr(mod, 'get_parser') 
    parser = get_parser()

    args = parser.parse_args() # this is the return

    # load args related to seed etc.
    set_random_seed(args.seed) if args.seed is not None else set_random_seed(42)
    
    return args

def main(args):

    # Add uuid, timestamp and hostname for logging
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()

    if args.dataset == 'sub':
        sub_dino_emb_path = 'data/SUB_dinov2_embeddings.pt'
        if not os.path.exists(sub_dino_emb_path):
            get_and_save_SUB_dino_embs()

    dataset = get_dataset(args)

    # Load dataset, model, loss, and optimizer
    encoder  = dataset.get_backbone()
    n_classes = dataset.get_n_classes()
    model = get_model(args, encoder, n_classes)
    if not isinstance(model, models.classifier.Classifier):
        model.initialize_weights()

    # get loss and start optimizer
    loss  = model.get_loss(args)
    model.start_optim(args)

    if args.teacher_ckpt is not None:
        print('\n--- Loading Teacher Model from Checkpoint ---\n')

        teacher_encoder = get_teacher_encoder(args)
        teacher_model = get_teacher_model(args, teacher_encoder, n_classes=n_classes) 
        teacher_model = create_load_ckpt(teacher_model, args, is_teacher=True)
        teacher_model.eval()
        dataset = distill_knowledge_to_dataset(teacher_model, dataset, args)
    else:
        teacher_model = None

    # SAVE A BASE MODEL OR LOAD IT, LOAD A CHECKPOINT IF PROVIDED
    model = create_load_ckpt(model, args)

    # set job name
    setproctitle.setproctitle('{}_{}_{}'.format( args.model, args.buffer_size if 'buffer_size' in args else 0, args.dataset))

    # perform posthoc evaluation/ cl training/ joint training
    print('    Chosen device:', model.device)
    if args.posthoc:
        posthoc_eval(model, dataset, teacher_model = teacher_model, args = args)
    else: train(model, dataset, loss, args, teacher=teacher_model)

    print('\n ### Closing ###')

if __name__ == '__main__':
    args = parse_args()
    
    print(args)
    
    main(args)