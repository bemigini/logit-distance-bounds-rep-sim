"""

Plot illustrating the bound from theorem F.1. 

"""

import os

from argparse import Namespace

import torch 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.cross_decomposition import CCA

from tqdm import tqdm

from dsets import get_dataset
from models import get_model, get_teacher_model, get_teacher_encoder
from utils import save_load_json
from utils.checkpoint import create_load_ckpt, make_checkpoint_name
from utils.metrics import evaluate_reprs

from figures.metric_plots_cifar import load_model_and_save_metrics_to_dict



def combine_metrics_low_dim_student():
    result_folder = 'data/runs'
    combined_metrics_name = 'low_dim_distillation_metrics.json'
    combined_metrics_path = os.path.join(result_folder, combined_metrics_name)

    args = Namespace(seed=0, all_metrics=True, checkin=None, checkout=None, teacher_ckpt='data/runs/cifar100-standard-classifier-pretrain_resnet50v2-50-epoch9-seed0.pt', 
                            dataset='cifar100', modality='standard',
                            model='logitdistill', teacher='classifier', entropy=False, 
                            backbone='resnet18', teacher_backbone='resnet50', n_epochs=50, 
                            final_dim = 25, teacher_final_dim = 50)
    
    student_seeds=[0, 1, 2, 3, 4]
    teacher_seeds=[0] 
    model_types=['logitdistill']
    loss_types=['l2']
    final_epoch=49

    combined_d = {
        'teacher_seed':[],
        'student_seed': [],
        'model_type':[],
        'epoch':[],
        'loss_train':[],        
        'acc_train':[],
        'f1_train':[],
        'acc_to_teacher_train': [],
        'f1_to_teacher_train': [],
        'cca_train':[],
        'm_cca_train':[],
        'd_rep_train':[],
        'd_LLD_train':[],
        'kl_div_train':[],
        'acc_test':[],
        'f1_test':[],
        'acc_to_teacher_test': [],
        'f1_to_teacher_test': [],
        'cca_test':[],
        'm_cca_test':[],
        'd_rep_test':[],
        'd_LLD_test':[],
        'kl_div_test':[]
    }

    for current_model_type in model_types:
        for current_t_seed in teacher_seeds:
            for current_s_seed in tqdm(student_seeds):
                for i, current_loss_type in enumerate(loss_types):
                    if current_model_type == 'kldistill' and i != 0:
                        continue                    
                    combined_d = load_model_and_save_metrics_to_dict(final_epoch, current_s_seed, args, current_t_seed, current_model_type, current_loss_type, result_folder, combined_d)

    save_load_json.save_as_json(combined_d, combined_metrics_path)


def calculate_moment_matrix_eigenvalues_and_canonical_correlations():
    """
    Calculation of moment matrix eigenvalues for use in the theorem F.1 bound
    """
    result_folder = 'data/runs'
    args = Namespace(seed=0, data_dir = result_folder, validate=False,
                     batch_size = 32, num_workers = 2,
                     all_metrics=True, checkin=None, checkout=None, teacher_ckpt='data/runs/cifar100-standard-classifier-pretrain_resnet50v2-50-epoch9-seed0.pt', 
                            dataset='cifar100', modality='standard',
                            model='logitdistill', loss_type = 'l2', teacher='classifier', entropy=False, 
                            backbone='resnet18', teacher_backbone='resnet50', n_epochs=50, 
                            final_dim = 25, teacher_final_dim = 50)
    
    n_components = args.final_dim
    student_seeds=[0, 1, 2, 3, 4]
    student_epochs = [10, 20, 30, 40, 50]

    dataset = get_dataset(args)
    _, _, test_loader = dataset.get_data_loaders()
    encoder  = dataset.get_backbone()
    n_classes = dataset.get_n_classes()
    model = get_model(args, encoder, n_classes)

    teacher_encoder = get_teacher_encoder(args)
    teacher_model = get_teacher_model(args, teacher_encoder, n_classes=n_classes) 
    teacher_model = create_load_ckpt(teacher_model, args, is_teacher=True)
    teacher_model.eval()
    teacher_model.to(model.device)

    cc_dict = {
        'teacher_seed':0,
        'student_seed': [],
        'epoch': [],
        'd2_logit': [],
        'canonical_correlations':[],
        'moment_eig_vals':[]
    }

    for i, current_seed in tqdm(enumerate(student_seeds)):
        for current_epoch in student_epochs:
            checkpoint_name = make_checkpoint_name(current_epoch -1, current_seed, args, 0, args.model, args.loss_type)

            model_path = os.path.join(result_folder, checkpoint_name + '.pt')

            model.load_state_dict(torch.load(model_path))
            model.to(model.device)

            y_true, t_embs, y_pred, s_embs, teacher_pred, s_logits, t_logits = evaluate_reprs(model, test_loader, args,
                                                        teacher=teacher_model,
                                                        last=True)
            num_inputs = s_logits.shape[0]
            squared_diff = np.power(s_logits - t_logits, 2)
            d2_logit = np.sum(squared_diff) / num_inputs
            cc_dict['d2_logit'].append(d2_logit.item())


            cca = CCA(n_components=n_components, max_iter=1000)
            cca.fit(t_embs, s_embs)

            # CCA correlations in decending order
            X_c, Y_c = cca.transform(t_embs, s_embs)
            corrs = [np.corrcoef(X_c[:, k], Y_c[:, k])[0, 1].item() for k in range(n_components)]
            cc_dict['canonical_correlations'].append(corrs)
            cc_dict['epoch'].append(current_epoch)
            cc_dict['student_seed'].append(current_seed)

        if i == 0:
            moment_matrix = np.matmul(t_logits.T, t_logits)/t_logits.shape[0]
            eigenvalues, _ = np.linalg.eig(moment_matrix)
            cc_dict['moment_eig_vals'].extend([eig.item() for eig in eigenvalues])
    
    cc_vals_name = 'low_dim_distillation_canonical_corr.json'
    cc_vals_path = os.path.join(result_folder, cc_vals_name)

    save_load_json.save_as_json(cc_dict, cc_vals_path)


def plot_theorem_F1_values():
    """
    Make the plot
    """
    result_folder = 'data/runs'
    cc_vals_name = 'low_dim_distillation_canonical_corr.json'
    cc_vals_path = os.path.join(result_folder, cc_vals_name)

    cc_vals_dict = save_load_json.load_json(cc_vals_path)

    student_seeds = cc_vals_dict['student_seed']
    left_right_bound_epochs = np.zeros((len(student_seeds), 4))
    moment_eigs = np.array(cc_vals_dict['moment_eig_vals'])
    num_moment_eigs = moment_eigs.shape[0]


    for i, current_seed in enumerate(student_seeds):
        current_epoch = cc_vals_dict['epoch'][i]
        current_d2_logit = cc_vals_dict['d2_logit'][i]
        current_cc_vals = np.array(cc_vals_dict['canonical_correlations'][i])
        padding_zeros = np.zeros(num_moment_eigs-current_cc_vals.shape[0])
        current_cc_vals = np.concatenate([current_cc_vals, padding_zeros])

        left_side = ((1-current_cc_vals**2)*moment_eigs).sum()
        left_right_bound_epochs[i, 0] = left_side
        left_right_bound_epochs[i, 1] = current_d2_logit
        left_right_bound_epochs[i, 2] = current_epoch
        left_right_bound_epochs[i, 3] = current_seed


    unique_seeds = np.unique(np.array(student_seeds))

    fontsize = 14
    colours_to_use = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    for i, current_seed in enumerate(unique_seeds):
        current_bounds = left_right_bound_epochs[left_right_bound_epochs[:, 3]==current_seed]
        if i == 0:
            ax.plot(current_bounds[:, 2], current_bounds[:, 1], c= 'black', label = 'd^2_logit', linestyle='dashed')
            ax.plot(current_bounds[:, 2], current_bounds[:, 0], c= 'black', label = 'canonical correlations')
            ax.plot(current_bounds[:, 2], current_bounds[:, 1], c= colours_to_use[i], linestyle='dashed', marker='o')
            ax.plot(current_bounds[:, 2], current_bounds[:, 0], c= colours_to_use[i], marker='o')
        else:
            ax.plot(current_bounds[:, 2], current_bounds[:, 1], c= colours_to_use[i], linestyle='dashed', marker='o')
            ax.plot(current_bounds[:, 2], current_bounds[:, 0], c= colours_to_use[i], marker='o')

    ax.legend(fontsize = fontsize)
    unique_epochs = [str(e) for e in np.unique(cc_vals_dict['epoch'])]
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels = [l if l in unique_epochs else '' for l in labels]
    ax.set_xticklabels(labels)

    ax.set_xlabel('epoch', fontsize = fontsize)
    ax.set_ylabel('measurements', fontsize = fontsize)
    fig.show()
