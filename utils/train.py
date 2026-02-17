import os
import datetime

import torch
import numpy as np

from utils.status import progress_bar
from dsets.utils.base_dataset import BaseDataset
from dsets.utils.base_dataset import BaseDataset
from dsets.synthetic import plot_2d_data
from utils.checkpoint import save_model_ckpt, make_checkpoint_name
from utils import save_load_json

from utils.posthoc_eval import evaluate_train_test


def train(model, dataset: BaseDataset, _loss, args, teacher=None):

    # Default Setting for Training
    model.to(model.device)
    train_loader, _, test_loader = dataset.get_data_loaders()

    scheduler1 = torch.optim.lr_scheduler.ConstantLR(model.opt, factor=0.001, total_iters=20)
    scheduler2 = torch.optim.lr_scheduler.ExponentialLR(model.opt, gamma=args.exp_decay)

    milestone = 3 if args.dataset == 'cifar100' else 20
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        model.opt,
        schedulers=[scheduler1, scheduler2],
        milestones=[milestone],
    )
    eval_freq = 50 if args.n_epochs >= 200 else 5

    metrics_d = {
        'epoch_num': [],
        'loss': [],        
        'train': {            
            'acc': [],
            'f1': [],
            'acc_to_teacher': [],
            'f1_to_teacher': [],
            'kl_to_teacher': [],
            'min_prob_teacher':[],
            'min_prob_student':[],
            'cca': [],
            'm_cca': [],
            'd_rep':[],
            'd_LLD':[]
            },
        'test':{
            'acc': [],
            'f1': [],
            'acc_to_teacher': [],
            'f1_to_teacher': [],
            'kl_to_teacher': [],
            'min_prob_teacher':[],
            'min_prob_student':[],
            'cca': [],
            'm_cca': [],
            'd_rep':[],
            'd_LLD':[]
            }
        }

    now = datetime.datetime.now()
    print(f'\n--- Start of Training ---\n{now.strftime("%Y-%m-%d %H:%M:%S")}\n')

    # default for warm-up
    model.opt.zero_grad()
    model.opt.step()

    loader = train_loader

    for epoch in range(args.n_epochs):

        model.train()
        for i, data in enumerate(loader):
            images, labels, tlogits = data
            images, labels, tlogits = images.to(model.device), labels.to(model.device), tlogits.to(model.device)

            out_dict = model(images)

            out_dict.update({'INPUTS': images, 'LABELS': labels, 'TLOGITS': tlogits})

            model.opt.zero_grad()
            loss, losses = _loss(out_dict, args)

            loss.backward()
            model.opt.step()

            if i % 10 ==0: progress_bar(i, len(loader)-9, epoch, loss.item())
        
        if args.generate_gif:
            with torch.no_grad():
                Z, Y = [], []
                for data in test_loader:
                    images, labels, _ = data 
                    images, labels = images.to(model.device), labels.to(model.device)
                    embs = model.encoder(images)
                    
                    Z.append(embs.cpu().numpy())
                    Y.append(labels.cpu().numpy())
                Z = np.concatenate(Z, axis=0)
                Y =  np.concatenate(Y, axis=0)
                if not os.path.exists('data/GIFs'):
                    os.makedirs('data/GIFs')
                plot_2d_data(Z, Y, filename=f'data/GIFs/{args.dataset}-{args.model}-epoch{epoch}-seed{args.seed}_embeddings.png', s=2)

        # Save metrics with eval frequency
        epoch_num = epoch + 1
        if ((epoch_num) % eval_freq) == 0:
            metrics_d['epoch_num'].append(epoch_num)
            metrics_d['loss'].append(loss.item())
            
            _, _, _, _, acc_metrics, rep_metrics = evaluate_train_test(
                model, teacher_model=teacher, train_loader=train_loader, test_loader=test_loader, args=args)
            
            if teacher is None:
                yac, yf1, yac_train, yf1_train = acc_metrics
            else:
                yac, yf1, yac_train, yf1_train, s_to_t_acc, s_to_t_f1, s_to_t_acc_train, s_to_t_f1_train, kl_div_test, kl_div_train, t_min_prob_test, s_min_prob_test, t_min_prob_train, s_min_prob_train = acc_metrics

                metrics_d['test']['acc_to_teacher'].append(s_to_t_acc)
                metrics_d['test']['f1_to_teacher'].append(s_to_t_f1)
                metrics_d['train']['acc_to_teacher'].append(s_to_t_acc_train)
                metrics_d['train']['f1_to_teacher'].append(s_to_t_f1_train)

                if args.all_metrics:
                    metrics_d['test']['kl_to_teacher'].append(kl_div_test)
                    metrics_d['train']['kl_to_teacher'].append(kl_div_train)

                    metrics_d['test']['min_prob_teacher'].append(t_min_prob_test)
                    metrics_d['test']['min_prob_student'].append(s_min_prob_test)
                    metrics_d['train']['min_prob_teacher'].append(t_min_prob_train)
                    metrics_d['train']['min_prob_student'].append(s_min_prob_train)

                    cca_score_test, cca_score_train, d_rep_test, d_rep_train, d_LLD_test, d_LLD_train, m_cca_test, m_cca_train = rep_metrics
                    metrics_d['test']['d_rep'].append(d_rep_test)
                    metrics_d['train']['d_rep'].append(d_rep_train)
                    metrics_d['test']['d_LLD'].append(d_LLD_test)
                    metrics_d['train']['d_LLD'].append(d_LLD_train)
                    metrics_d['test']['m_cca'].append(m_cca_test)
                    metrics_d['train']['m_cca'].append(m_cca_train)
                else:
                    cca_score_test, cca_score_train = rep_metrics

                metrics_d['test']['cca'].append(cca_score_test)
                metrics_d['train']['cca'].append(cca_score_train)


            metrics_d['test']['acc'].append(yac)
            metrics_d['test']['f1'].append(yf1)
            metrics_d['train']['acc'].append(yac_train)
            metrics_d['train']['f1'].append(yf1_train)            


        # update at end of the epoch 
        scheduler.step()

    # SAVE MODEL CHECKPOINT
    save_model_ckpt(model, args, epoch, args.seed)
    checkpoint_name = make_checkpoint_name(epoch, args.seed, args)
    json_path = f'data/runs/metrics-{checkpoint_name}.json'
    save_load_json.save_as_json(metrics_d, json_path)

    # Evaluate performance
    post_train_args = args
    post_train_args.all_metrics = False
    y_true_test, _, y_pred_test, s_embs, _, _ = evaluate_train_test(
        model, teacher_model=teacher, train_loader=train_loader, test_loader=test_loader, args=post_train_args)

    if args.dataset == 'synthetic':
        flag = '' if args.checkin is None else '-finetuned'
        loc = f'data/{args.dataset}-{args.modality}{flag}-{args.model}-epoch{epoch}-seed{args.seed}_embeddings.png'
        plot_2d_data(s_embs, y_true_test, filename=loc, s=2)

    print('--- Training Finished ---')
