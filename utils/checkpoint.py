import torch
from utils.conf import create_path

def create_load_ckpt(model, args, is_teacher=False):
    create_path('data/runs')

    if args.checkin is not None:
        model.load_state_dict(torch.load(args.checkin))

    elif args.teacher_ckpt is not None and is_teacher:
        model.load_state_dict(torch.load(args.teacher_ckpt))

    return model


def make_checkpoint_name(epoch, seed, args, t_seed_overwrite = -1, model_type_overwrite = '', loss_type_overwrite = ''):

    flag = ''
    if args.checkin is not None:
        flag += '-finetuned'

    backbone_str = f'-{args.backbone}' if args.backbone != '' else ''
    final_dim_str = f'-{args.final_dim}' if args.final_dim > 2 else ''

    if args.teacher_ckpt is None or args.teacher_ckpt == '':
        teacher_seed = ''
    elif t_seed_overwrite > -1:
        teacher_seed = f'-tseed{t_seed_overwrite}'
    else:
        t_seed = args.teacher_ckpt.split('-seed')[1].split('.')[0]
        teacher_seed = f'-tseed{t_seed}'
    
    if model_type_overwrite == '':
        model_type_str = args.model
    else:
        model_type_str = model_type_overwrite

    if model_type_str =='logitdistill':
        if loss_type_overwrite == '':
            loss_t = f'-{args.loss_type}'
        else:
            loss_t = f'-{loss_type_overwrite}'
    else:
        loss_t = ''

    name = f'{args.dataset}-{args.modality}{flag}-{model_type_str}{loss_t}{backbone_str}{final_dim_str}{teacher_seed}-epoch{epoch}-seed{seed}'

    return name


def save_model_ckpt(model, args, epoch, seed):
    create_path('data/runs')

    checkpoint_name = make_checkpoint_name(epoch, seed, args)
    PATH = f'data/runs/{checkpoint_name}.pt'
    torch.save(model.state_dict(), PATH)
    print('\n \n Saved model checkpoint at', PATH)

def load_teacher(model, args):
    if args.teacher_ckpt is not None:
        model.load_state_dict(torch.load(args.teacher_ckpt))
        print('Loaded teacher from', args.teacher_ckpt)
    return model
