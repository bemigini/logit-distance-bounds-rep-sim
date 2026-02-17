"""
Tables and plots showing the metrics from CIFAR-100 experiments
"""

import os

from argparse import Namespace

import torch 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from tqdm import tqdm


from figures.inverse_variance_weighted_average import inverse_var_weighted_avg_and_var
from utils import save_load_json
from utils.checkpoint import make_checkpoint_name




def load_model_and_save_metrics_to_dict(final_epoch, current_s_seed, args, current_t_seed, current_model_type, current_loss_type, result_folder, combined_d):
    checkpoint_name = make_checkpoint_name(final_epoch, current_s_seed, args, current_t_seed, current_model_type, current_loss_type)
    metric_name = f'metrics-{checkpoint_name}.json'
    metric_path = os.path.join(result_folder, metric_name)
    metric_d = save_load_json.load_json(metric_path)
    
    current_epochs = metric_d['epoch_num']
    t_seed_extend = [current_t_seed for _ in current_epochs]
    s_seed_extend = [current_s_seed for _ in current_epochs]
    if current_model_type == 'logitdistill':
        current_model_type = f'{current_model_type}_{current_loss_type}'
    model_type_extend = [current_model_type for _ in current_epochs]

    combined_d['teacher_seed'].extend(t_seed_extend)
    combined_d['student_seed'].extend(s_seed_extend)
    combined_d['model_type'].extend(model_type_extend)
    combined_d['epoch'].extend(current_epochs)
    combined_d['loss_train'].extend(metric_d['loss'])
    
    combined_d['acc_train'].extend(metric_d['train']['acc'])
    combined_d['f1_train'].extend(metric_d['train']['f1'])
    combined_d['acc_to_teacher_train'].extend(metric_d['train']['acc_to_teacher'])
    combined_d['f1_to_teacher_train'].extend(metric_d['train']['f1_to_teacher'])
    combined_d['cca_train'].extend(metric_d['train']['cca'])
    combined_d['m_cca_train'].extend(metric_d['train']['m_cca'])
    combined_d['d_rep_train'].extend(metric_d['train']['d_rep'])
    combined_d['d_LLD_train'].extend(metric_d['train']['d_LLD'])
    combined_d['kl_div_train'].extend(metric_d['train']['kl_to_teacher'])

    combined_d['acc_test'].extend(metric_d['test']['acc'])
    combined_d['f1_test'].extend(metric_d['test']['f1'])
    combined_d['acc_to_teacher_test'].extend(metric_d['test']['acc_to_teacher'])
    combined_d['f1_to_teacher_test'].extend(metric_d['test']['f1_to_teacher'])
    combined_d['cca_test'].extend(metric_d['test']['cca'])
    combined_d['m_cca_test'].extend(metric_d['test']['m_cca'])
    combined_d['d_rep_test'].extend(metric_d['test']['d_rep'])
    combined_d['d_LLD_test'].extend(metric_d['test']['d_LLD'])
    combined_d['kl_div_test'].extend(metric_d['test']['kl_to_teacher'])

    return combined_d


def combine_metrics():
    result_folder = 'data/runs'
    combined_metrics_name = 'all_distillation_metrics.json'
    combined_metrics_path = os.path.join(result_folder, combined_metrics_name)

    args = Namespace(seed=0, all_metrics=True, checkin=None, checkout=None, teacher_ckpt='data/runs/cifar100-standard-N0-classifier-pretrain_resnet50v2-50-epoch9-seed1.pt', 
                            dataset='cifar100', modality='standard',
                            model='logitdistill', teacher='classifier', entropy=False, 
                            backbone='resnet18', teacher_backbone='resnet50', n_epochs=50, final_dim = 50, distill_only = True)
    
    student_seeds=[0, 1, 2, 3, 4]
    teacher_seeds=[0, 1, 2, 3, 4] 
    model_types=['logitdistill', 'kldistill']
    loss_types=['l1', 'l2']
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


def combine_base_model_metrics():
    result_folder = 'data/runs'
    combined_metrics_name = 'all_base_model_metrics.json'
    combined_metrics_path = os.path.join(result_folder, combined_metrics_name)

    seeds = [0, 1, 2, 3, 4]
    final_epoch = 49
    args = Namespace(seed=0, all_metrics=True, checkin=None, checkout=None, teacher_ckpt=None, data_dir='data',
                            dataset='cifar100', modality='standard',
                            model='classifier', teacher='classifier', entropy=False,
                            backbone='resnet18', teacher_backbone='resnet50', n_epochs=50, final_dim = 50, 
                            validate=False, num_workers=4, batch_size=32)

    comb_metrics_d = {
        'seed': [],
        'model_type':[],
        'epoch':[],
        'loss_train':[],        
        'acc_train':[],
        'f1_train':[],
        'acc_test':[],
        'f1_test':[]
    }

    for current_seed in tqdm(seeds):
        checkpoint_name = make_checkpoint_name(final_epoch, current_seed, args, 0, args.model)
        metric_name = f'metrics-{checkpoint_name}.json'
        metric_path = os.path.join(result_folder, metric_name)
        metric_d = save_load_json.load_json(metric_path)

        current_epochs = metric_d['epoch_num']
        seed_extend = [current_seed for _ in current_epochs]
        model_type_extend = [args.model for _ in current_epochs]

        comb_metrics_d['seed'].extend(seed_extend)
        comb_metrics_d['model_type'].extend(model_type_extend)
        comb_metrics_d['epoch'].extend(current_epochs)
        comb_metrics_d['loss_train'].extend(metric_d['loss'])
        
        comb_metrics_d['acc_train'].extend(metric_d['train']['acc'])
        comb_metrics_d['f1_train'].extend(metric_d['train']['f1'])

        comb_metrics_d['acc_test'].extend(metric_d['test']['acc'])
        comb_metrics_d['f1_test'].extend(metric_d['test']['f1'])
    
    save_load_json.save_as_json(comb_metrics_d, combined_metrics_path)


def combine_teacher_metrics():
    result_folder = 'data/runs'
    combined_metrics_name = 'all_teacher_metrics.json'
    combined_metrics_path = os.path.join(result_folder, combined_metrics_name)

    seeds = [0, 1, 2, 3, 4]
    final_epoch = 9
    args = Namespace(seed=0, all_metrics=True, checkin=None, checkout=None, teacher_ckpt=None, 
                            dataset='cifar100', modality='standard',
                            model='classifier', teacher='', entropy=False, 
                            backbone='pretrain_resnet50v2', teacher_backbone='', n_epochs=10, final_dim = 50)

    comb_metrics_d = {
        'seed': [],
        'model_type':[],
        'epoch':[],
        'loss_train':[],        
        'acc_train':[],
        'f1_train':[],
        'acc_test':[],
        'f1_test':[]
    }

    for current_seed in tqdm(seeds):
        checkpoint_name = make_checkpoint_name(final_epoch, current_seed, args, 0, args.model)
        metric_name = f'metrics-{checkpoint_name}.json'
        metric_path = os.path.join(result_folder, metric_name)
        metric_d = save_load_json.load_json(metric_path)

        current_epochs = metric_d['epoch_num']
        seed_extend = [current_seed for _ in current_epochs]
        model_type_extend = [args.model for _ in current_epochs]

        comb_metrics_d['seed'].extend(seed_extend)
        comb_metrics_d['model_type'].extend(model_type_extend)
        comb_metrics_d['epoch'].extend(current_epochs)
        comb_metrics_d['loss_train'].extend(metric_d['loss'])
        
        comb_metrics_d['acc_train'].extend(metric_d['train']['acc'])
        comb_metrics_d['f1_train'].extend(metric_d['train']['f1'])

        comb_metrics_d['acc_test'].extend(metric_d['test']['acc'])
        comb_metrics_d['f1_test'].extend(metric_d['test']['f1'])
    
    save_load_json.save_as_json(comb_metrics_d, combined_metrics_path)


def get_all_metrics(metric_type: str):
    result_folder = 'data/runs'
    combined_metrics_name = f'all_{metric_type}_metrics.json'
    combined_metrics_path = os.path.join(result_folder, combined_metrics_name)

    all_metrics = save_load_json.load_json(combined_metrics_path)
    metrics_df = pd.DataFrame(all_metrics)

    return metrics_df


def get_all_base_model_metrics():
    return get_all_metrics('base_model')


def get_all_teacher_metrics():
    return get_all_metrics('teacher')


def get_all_distillation_metrics():
    return get_all_metrics('distillation')


def plot_rep_metrics_progress():
    plot_folder = 'data/plots'
    dpi = 300    

    metrics_df = get_all_distillation_metrics()
    base_metrics_df = get_all_base_model_metrics()
    teacher_metrics_df = get_all_teacher_metrics()

    final_epoch_acc = teacher_metrics_df[teacher_metrics_df['epoch']==10]['acc_test']
    mean_teacher_acc = final_epoch_acc.mean()
    std_teacher_acc = final_epoch_acc.std()
    print(f'Teacher acc: mean: {mean_teacher_acc}, std: {std_teacher_acc}')

    dummy_teacher_seed = [0 for _ in range(base_metrics_df.shape[0])]
    base_metrics_df.insert(1, "teacher_seed", dummy_teacher_seed)

    unique_epochs = metrics_df['epoch'].unique()
    unique_model_types = metrics_df['model_type'].unique()
    print(unique_model_types)

    distil_types = ['logitdistill_l1', 'kldistill']
    comb_file_name = 'metric_over_epochs_l1_kl_iv_avg.png'


    current_metric = 'cca'
    cca_avg_var_dict = inverse_var_weighted_avg_and_var(metrics_df, current_metric)

    fontsize = 14

    fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    colours_to_use = ['#002347', '#004a95', '#9d3f00', '#ff6600']

    train_test = ['train', 'test']    
    for j, current_distill in enumerate(distil_types):
        for i, current_train_test in enumerate(train_test):        
            current_idx = j*len(train_test)+i

            avg = np.array(cca_avg_var_dict[current_distill][f'iv_avg_{current_train_test}_pr_epoch'])
            std = np.sqrt(np.array(cca_avg_var_dict[current_distill][f'iv_avg_var_{current_train_test}_pr_epoch']))
            axes.plot(unique_epochs, avg, 
                        c=colours_to_use[current_idx], linewidth=2.5, label = f'{current_distill} {current_train_test}')
            axes.fill_between(
                unique_epochs, 
                avg-std, avg+std, 
                alpha = 0.2,
                color=colours_to_use[current_idx])
    
    axes.set_xlabel('epochs', fontsize = fontsize)
    axes.set_ylabel(current_metric, fontsize = fontsize)
    axes.legend(fontsize = fontsize)
    axes.tick_params(axis='both', which='major', labelsize=fontsize-2)
    axes.set_title('CCA score vs epochs', fontsize=fontsize+2)
    fig.tight_layout()

    #fig.show()
    figure_name = comb_file_name.replace('metric', current_metric)
    figure_path = os.path.join(plot_folder, figure_name)
    plt.savefig(figure_path, dpi = dpi)
    plt.close()



    current_metric = 'm_cca'
    cca_avg_var_dict = inverse_var_weighted_avg_and_var(metrics_df, current_metric)

    fontsize = 14

    fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    colours_to_use = ['#002347', '#004a95', '#9d3f00', '#ff6600']

    train_test = ['train', 'test']
    for j, current_distill in enumerate(distil_types):
        for i, current_train_test in enumerate(train_test):        
            current_idx = j*len(train_test)+i

            avg = np.array(cca_avg_var_dict[current_distill][f'iv_avg_{current_train_test}_pr_epoch'])
            std = np.sqrt(np.array(cca_avg_var_dict[current_distill][f'iv_avg_var_{current_train_test}_pr_epoch']))
            print(f'{current_distill} {current_train_test}: avg:{avg[-1]}, std:{std[-1]}')
            axes.plot(unique_epochs, avg, 
                        c=colours_to_use[current_idx], linewidth=2.5, label = f'{current_distill} {current_train_test}')
            axes.fill_between(
                unique_epochs, 
                avg-(2*std), avg+(2*std), 
                alpha = 0.2,
                color=colours_to_use[current_idx])
    
    axes.set_xlabel('epochs', fontsize = fontsize)
    axes.set_ylabel(current_metric, fontsize = fontsize)
    #axes.set_ylim(0, 1)
    axes.legend(fontsize = fontsize)
    axes.tick_params(axis='both', which='major', labelsize=fontsize-2)
    axes.set_title('m_CCA vs epochs', fontsize=fontsize+2)
    fig.tight_layout()

    #fig.show()
    figure_name = comb_file_name.replace('metric', current_metric)
    figure_path = os.path.join(plot_folder, figure_name)
    plt.savefig(figure_path, dpi = dpi)
    plt.close()


    current_metric = 'd_rep'
    d_rep_avg_var_dict = inverse_var_weighted_avg_and_var(metrics_df, current_metric)

    fontsize = 14

    fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    colours_to_use = ['#002347', '#004a95', '#9d3f00', '#ff6600']

    train_test = ['train', 'test']    
    for j, current_distill in enumerate(distil_types):
        for i, current_train_test in enumerate(train_test):        
            current_idx = j*len(train_test)+i

            avg = np.array(d_rep_avg_var_dict[current_distill][f'iv_avg_{current_train_test}_pr_epoch'])
            std = np.sqrt(np.array(d_rep_avg_var_dict[current_distill][f'iv_avg_var_{current_train_test}_pr_epoch']))
            print(f'{current_distill} {current_train_test}: avg:{avg[-1]}, std:{std[-1]}')
            axes.plot(unique_epochs, avg, 
                        c=colours_to_use[current_idx], linewidth=2.5, label = f'{current_distill} {current_train_test}')
            axes.fill_between(
                unique_epochs, 
                avg-(2*std), avg+(2*std), 
                alpha = 0.2,
                color=colours_to_use[current_idx])
    
    axes.set_xlabel('epochs', fontsize = fontsize)
    axes.set_ylabel(current_metric, fontsize = fontsize)
    axes.legend(fontsize = fontsize)
    axes.tick_params(axis='both', which='major', labelsize=fontsize-2)
    axes.set_title('d_rep^2 vs epochs', fontsize=fontsize+2)
    fig.tight_layout()

    #fig.show()
    figure_name = comb_file_name.replace('metric', f'{current_metric}_2')
    figure_path = os.path.join(plot_folder, figure_name)
    plt.savefig(figure_path, dpi = dpi)
    plt.close()


    current_metric = 'd_rep'
    d_rep_avg_var_dict = inverse_var_weighted_avg_and_var(metrics_df, current_metric, use_sqrt_val=True)

    fontsize = 14

    fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    colours_to_use = ['#002347', '#004a95', '#9d3f00', '#ff6600']

    train_test = ['train', 'test']    
    for j, current_distill in enumerate(distil_types):
        for i, current_train_test in enumerate(train_test):        
            current_idx = j*len(train_test)+i

            avg = np.array(d_rep_avg_var_dict[current_distill][f'iv_avg_{current_train_test}_pr_epoch'])
            std = np.sqrt(np.array(d_rep_avg_var_dict[current_distill][f'iv_avg_var_{current_train_test}_pr_epoch']))
            print(f'{current_distill} {current_train_test}: avg:{avg[-1]}, std:{std[-1]}')
            axes.plot(unique_epochs, avg, 
                        c=colours_to_use[current_idx], linewidth=2.5, label = f'{current_distill} {current_train_test}')
            axes.fill_between(
                unique_epochs, 
                avg-(2*std), avg+(2*std), 
                alpha = 0.2,
                color=colours_to_use[current_idx])
    
    axes.set_xlabel('epochs', fontsize = fontsize)
    axes.set_ylabel(current_metric, fontsize = fontsize)
    axes.legend(fontsize = fontsize)
    axes.tick_params(axis='both', which='major', labelsize=fontsize-2)
    axes.set_title('d_rep vs epochs', fontsize=fontsize+2)
    fig.tight_layout()

    #fig.show()
    figure_name = comb_file_name.replace('metric', current_metric)
    figure_path = os.path.join(plot_folder, figure_name)
    plt.savefig(figure_path, dpi = dpi)
    plt.close()


    current_metric = 'd_logit'
    d_rep_avg_var_dict = inverse_var_weighted_avg_and_var(metrics_df, 'd_LLD')

    fontsize = 14

    fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    colours_to_use = ['#002347', '#004a95', '#9d3f00', '#ff6600']

    train_test = ['train', 'test']
    for j, current_distill in enumerate(distil_types):
        for i, current_train_test in enumerate(train_test):        
            current_idx = j*len(train_test)+i

            avg = np.array(d_rep_avg_var_dict[current_distill][f'iv_avg_{current_train_test}_pr_epoch'])
            std = np.sqrt(np.array(d_rep_avg_var_dict[current_distill][f'iv_avg_var_{current_train_test}_pr_epoch']))
            print(f'{current_distill} {current_train_test}: avg:{avg[-1]}, std:{std[-1]}')
            axes.plot(unique_epochs, avg, 
                        c=colours_to_use[current_idx], linewidth=2.5, label = f'{current_distill} {current_train_test}')
            axes.fill_between(
                unique_epochs, 
                avg-(2*std), avg+(2*std), 
                alpha = 0.2,
                color=colours_to_use[current_idx])
    
    axes.set_xlabel('epochs', fontsize = fontsize)
    axes.set_ylabel(current_metric, fontsize = fontsize)
    axes.legend(fontsize = fontsize)
    axes.tick_params(axis='both', which='major', labelsize=fontsize-2)
    axes.set_title(f'{current_metric}^2 vs epochs', fontsize=fontsize+2)
    fig.tight_layout()

    #fig.show()
    figure_name = comb_file_name.replace('metric', f'{current_metric}_2')
    figure_path = os.path.join(plot_folder, figure_name)
    plt.savefig(figure_path, dpi = dpi)
    plt.close()
    


    current_metric = 'd_logit'
    d_rep_avg_var_dict = inverse_var_weighted_avg_and_var(metrics_df, 'd_LLD', use_sqrt_val=True)

    fontsize = 14

    fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    colours_to_use = ['#002347', '#004a95', '#9d3f00', '#ff6600']

    train_test = ['train', 'test']
    for j, current_distill in enumerate(distil_types):
        for i, current_train_test in enumerate(train_test):        
            current_idx = j*len(train_test)+i

            avg = np.array(d_rep_avg_var_dict[current_distill][f'iv_avg_{current_train_test}_pr_epoch'])
            std = np.sqrt(np.array(d_rep_avg_var_dict[current_distill][f'iv_avg_var_{current_train_test}_pr_epoch']))
            print(f'{current_distill} {current_train_test}: avg:{avg[-1]}, std:{std[-1]}')
            axes.plot(unique_epochs, avg, 
                        c=colours_to_use[current_idx], linewidth=2.5, label = f'{current_distill} {current_train_test}')
            axes.fill_between(
                unique_epochs, 
                avg-(2*std), avg+(2*std), 
                alpha = 0.2,
                color=colours_to_use[current_idx])
    
    axes.set_xlabel('epochs', fontsize = fontsize)
    axes.set_ylabel(current_metric, fontsize = fontsize)
    axes.legend(fontsize = fontsize)
    axes.tick_params(axis='both', which='major', labelsize=fontsize-2)
    axes.set_title(f'{current_metric} vs epochs', fontsize=fontsize+2)
    fig.tight_layout()

    #fig.show()
    figure_name = comb_file_name.replace('metric', current_metric)
    figure_path = os.path.join(plot_folder, figure_name)
    plt.savefig(figure_path, dpi = dpi)
    plt.close()



    current_metric = 'kl_div'
    d_rep_avg_var_dict = inverse_var_weighted_avg_and_var(metrics_df, current_metric)

    fontsize = 14

    fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    colours_to_use = ['#002347', '#004a95', '#9d3f00', '#ff6600']

    train_test = ['train', 'test']    
    for j, current_distill in enumerate(distil_types):
        for i, current_train_test in enumerate(train_test):        
            current_idx = j*len(train_test)+i

            avg = np.array(d_rep_avg_var_dict[current_distill][f'iv_avg_{current_train_test}_pr_epoch'])
            std = np.sqrt(np.array(d_rep_avg_var_dict[current_distill][f'iv_avg_var_{current_train_test}_pr_epoch']))
            print(f'{current_distill} {current_train_test}: avg:{avg[-1]}, std:{std[-1]}')
            axes.plot(unique_epochs, avg, 
                        c=colours_to_use[current_idx], linewidth=2.5, label = f'{current_distill} {current_train_test}')
            axes.fill_between(
                unique_epochs, 
                avg-(2*std), avg+(2*std), 
                alpha = 0.2,
                color=colours_to_use[current_idx])
    
    axes.set_xlabel('epochs', fontsize = fontsize)
    axes.set_ylabel(current_metric, fontsize = fontsize)
    axes.legend(fontsize = fontsize)
    axes.tick_params(axis='both', which='major', labelsize=fontsize-2)
    axes.set_title('kl_div vs epochs', fontsize=fontsize+2)
    fig.tight_layout()

    #fig.show()
    figure_name = comb_file_name.replace('metric', current_metric)
    figure_path = os.path.join(plot_folder, figure_name)
    plt.savefig(figure_path, dpi = dpi)
    plt.close()


    #figure_name = comb_file_name.replace('cls.json', f'cls{classes_str}.png')
    #figure_path = os.path.join(figure_folder, figure_name)
    #plt.savefig(figure_path, dpi = dpi)
    #plt.close()


    current_metric = 'acc'
    acc_distill_avg_var_d = inverse_var_weighted_avg_and_var(metrics_df, current_metric)
    acc_base_avg_var_d = inverse_var_weighted_avg_and_var(base_metrics_df, current_metric)

    fontsize = 14

    fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    colours_to_use = ['#002347', '#004a95', '#9d3f00', '#ff6600']
    base_model_colours = ['#006600', '#009900']

    train_test = ['train', 'test']
    for j, current_distill in enumerate(distil_types):
        for i, current_train_test in enumerate(train_test):        
            current_idx = j*len(train_test)+i

            avg = np.array(acc_distill_avg_var_d[current_distill][f'iv_avg_{current_train_test}_pr_epoch'])
            std = np.sqrt(np.array(acc_distill_avg_var_d[current_distill][f'iv_avg_var_{current_train_test}_pr_epoch']))
            print(f'{current_distill} {current_train_test}: avg:{avg[-1]}, std:{std[-1]}')
            axes.plot(unique_epochs, avg, 
                        c=colours_to_use[current_idx], linewidth=2.5, label = f'{current_distill} {current_train_test}')
            axes.fill_between(
                unique_epochs, 
                avg-(2*std), avg+(2*std), 
                alpha = 0.2,
                color=colours_to_use[current_idx])

    for i, current_train_test in enumerate(train_test):
        avg = np.array(acc_base_avg_var_d['classifier'][f'iv_avg_{current_train_test}_pr_epoch'])
        std = np.sqrt(np.array(acc_base_avg_var_d['classifier'][f'iv_avg_var_{current_train_test}_pr_epoch']))
        axes.plot(unique_epochs, avg, 
                    c=base_model_colours[i], linewidth=2.5, label = f'base {current_train_test}')
        axes.fill_between(
            unique_epochs, 
            avg-(2*std), avg+(2*std), 
            alpha = 0.2,
            color=base_model_colours[i])
    
    axes.set_xlabel('epochs', fontsize = fontsize)
    axes.set_ylabel(current_metric, fontsize = fontsize)
    axes.legend(fontsize = fontsize)
    axes.tick_params(axis='both', which='major', labelsize=fontsize-2)
    axes.set_title('acc vs epochs', fontsize=fontsize+2)
    fig.tight_layout()

    #fig.show()
    figure_name = comb_file_name.replace('metric', current_metric)
    figure_path = os.path.join(plot_folder, figure_name)
    plt.savefig(figure_path, dpi = dpi)
    plt.close()




    current_metric = 'acc_to_teacher'
    d_rep_avg_var_dict = inverse_var_weighted_avg_and_var(metrics_df, current_metric)

    fontsize = 14

    fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    colours_to_use = ['#002347', '#004a95', '#9d3f00', '#ff6600']

    train_test = ['train', 'test']
    for j, current_distill in enumerate(distil_types):
        for i, current_train_test in enumerate(train_test):        
            current_idx = j*len(train_test)+i

            avg = np.array(d_rep_avg_var_dict[current_distill][f'iv_avg_{current_train_test}_pr_epoch'])
            std = np.sqrt(np.array(d_rep_avg_var_dict[current_distill][f'iv_avg_var_{current_train_test}_pr_epoch']))
            axes.plot(unique_epochs, avg, 
                        c=colours_to_use[current_idx], linewidth=2.5, label = f'{current_distill} {current_train_test}')
            axes.fill_between(
                unique_epochs, 
                avg-(2*std), avg+(2*std), 
                alpha = 0.2,
                color=colours_to_use[current_idx])
            print(f'{current_distill} {current_train_test}: avg:{avg[-1]}, std:{std[-1]}')
    
    axes.set_xlabel('epochs', fontsize = fontsize)
    axes.set_ylabel('acc to teacher', fontsize = fontsize)
    axes.legend(fontsize = fontsize)
    axes.tick_params(axis='both', which='major', labelsize=fontsize-2)
    axes.set_title('acc to teacher vs epochs', fontsize=fontsize+2)
    fig.tight_layout()

    #fig.show()
    figure_name = comb_file_name.replace('metric', current_metric)
    figure_path = os.path.join(plot_folder, figure_name)
    plt.savefig(figure_path, dpi = dpi)
    plt.close()


    distil_types = ['logitdistill_l1', 'logitdistill_l2']
    comb_file_name = 'metric_over_epochs_l1_l2_iv_avg.png'

    current_metric = 'm_cca'
    cca_avg_var_dict = inverse_var_weighted_avg_and_var(metrics_df, current_metric)

    fontsize = 14

    fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    colours_to_use = ['#002347', '#004a95', '#9d3f00', '#ff6600']

    train_test = ['train', 'test']
    for j, current_distill in enumerate(distil_types):
        for i, current_train_test in enumerate(train_test):        
            current_idx = j*len(train_test)+i

            avg = np.array(cca_avg_var_dict[current_distill][f'iv_avg_{current_train_test}_pr_epoch'])
            std = np.sqrt(np.array(cca_avg_var_dict[current_distill][f'iv_avg_var_{current_train_test}_pr_epoch']))
            print(f'{current_distill} {current_train_test}: avg:{avg[-1]}, std:{std[-1]}')
            axes.plot(unique_epochs, avg, 
                        c=colours_to_use[current_idx], linewidth=2.5, label = f'{current_distill} {current_train_test}')
            axes.fill_between(
                unique_epochs, 
                avg-(2*std), avg+(2*std), 
                alpha = 0.2,
                color=colours_to_use[current_idx])
    
    axes.set_xlabel('epochs', fontsize = fontsize)
    axes.set_ylabel(current_metric, fontsize = fontsize)
    #axes.set_ylim(0, 1)
    axes.legend(fontsize = fontsize)
    axes.tick_params(axis='both', which='major', labelsize=fontsize-2)
    axes.set_title('m_CCA vs epochs', fontsize=fontsize+2)
    fig.tight_layout()

    #fig.show()
    figure_name = comb_file_name.replace('metric', current_metric)
    figure_path = os.path.join(plot_folder, figure_name)
    plt.savefig(figure_path, dpi = dpi)
    plt.close()


    current_metric = 'd_rep'
    d_rep_avg_var_dict = inverse_var_weighted_avg_and_var(metrics_df, current_metric)

    fontsize = 14

    fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    colours_to_use = ['#002347', '#004a95', '#9d3f00', '#ff6600']

    train_test = ['train', 'test']    
    for j, current_distill in enumerate(distil_types):
        for i, current_train_test in enumerate(train_test):        
            current_idx = j*len(train_test)+i

            avg = np.array(d_rep_avg_var_dict[current_distill][f'iv_avg_{current_train_test}_pr_epoch'])
            std = np.sqrt(np.array(d_rep_avg_var_dict[current_distill][f'iv_avg_var_{current_train_test}_pr_epoch']))
            print(f'{current_distill} {current_train_test}: avg:{avg[-1]}, std:{std[-1]}')
            axes.plot(unique_epochs, avg, 
                        c=colours_to_use[current_idx], linewidth=2.5, label = f'{current_distill} {current_train_test}')
            axes.fill_between(
                unique_epochs, 
                avg-(2*std), avg+(2*std), 
                alpha = 0.2,
                color=colours_to_use[current_idx])
    
    axes.set_xlabel('epochs', fontsize = fontsize)
    axes.set_ylabel(current_metric, fontsize = fontsize)
    axes.legend(fontsize = fontsize)
    axes.tick_params(axis='both', which='major', labelsize=fontsize-2)
    axes.set_title('d_rep^2 vs epochs', fontsize=fontsize+2)
    fig.tight_layout()

    #fig.show()
    figure_name = comb_file_name.replace('metric', f'{current_metric}_2')
    figure_path = os.path.join(plot_folder, figure_name)
    plt.savefig(figure_path, dpi = dpi)
    plt.close()


    current_metric = 'd_rep'
    d_rep_avg_var_dict = inverse_var_weighted_avg_and_var(metrics_df, current_metric, use_sqrt_val=True)

    fontsize = 14

    fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    colours_to_use = ['#002347', '#004a95', '#9d3f00', '#ff6600']

    train_test = ['train', 'test']    
    for j, current_distill in enumerate(distil_types):
        for i, current_train_test in enumerate(train_test):        
            current_idx = j*len(train_test)+i

            avg = np.array(d_rep_avg_var_dict[current_distill][f'iv_avg_{current_train_test}_pr_epoch'])
            std = np.sqrt(np.array(d_rep_avg_var_dict[current_distill][f'iv_avg_var_{current_train_test}_pr_epoch']))
            print(f'{current_distill} {current_train_test}: avg:{avg[-1]}, std:{std[-1]}')
            axes.plot(unique_epochs, avg, 
                        c=colours_to_use[current_idx], linewidth=2.5, label = f'{current_distill} {current_train_test}')
            axes.fill_between(
                unique_epochs, 
                avg-(2*std), avg+(2*std), 
                alpha = 0.2,
                color=colours_to_use[current_idx])
    
    axes.set_xlabel('epochs', fontsize = fontsize)
    axes.set_ylabel(current_metric, fontsize = fontsize)
    axes.legend(fontsize = fontsize)
    axes.tick_params(axis='both', which='major', labelsize=fontsize-2)
    axes.set_title('d_rep vs epochs', fontsize=fontsize+2)
    fig.tight_layout()

    #fig.show()
    figure_name = comb_file_name.replace('metric', current_metric)
    figure_path = os.path.join(plot_folder, figure_name)
    plt.savefig(figure_path, dpi = dpi)
    plt.close()


    current_metric = 'd_logit'
    d_rep_avg_var_dict = inverse_var_weighted_avg_and_var(metrics_df, 'd_LLD')

    fontsize = 14

    fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    colours_to_use = ['#002347', '#004a95', '#9d3f00', '#ff6600']

    train_test = ['train', 'test']
    for j, current_distill in enumerate(distil_types):
        for i, current_train_test in enumerate(train_test):        
            current_idx = j*len(train_test)+i

            avg = np.array(d_rep_avg_var_dict[current_distill][f'iv_avg_{current_train_test}_pr_epoch'])
            std = np.sqrt(np.array(d_rep_avg_var_dict[current_distill][f'iv_avg_var_{current_train_test}_pr_epoch']))
            print(f'{current_distill} {current_train_test}: avg:{avg[-1]}, std:{std[-1]}')
            axes.plot(unique_epochs, avg, 
                        c=colours_to_use[current_idx], linewidth=2.5, label = f'{current_distill} {current_train_test}')
            axes.fill_between(
                unique_epochs, 
                avg-(2*std), avg+(2*std), 
                alpha = 0.2,
                color=colours_to_use[current_idx])
    
    axes.set_xlabel('epochs', fontsize = fontsize)
    axes.set_ylabel(current_metric, fontsize = fontsize)
    axes.legend(fontsize = fontsize)
    axes.tick_params(axis='both', which='major', labelsize=fontsize-2)
    axes.set_title(f'{current_metric}^2 vs epochs', fontsize=fontsize+2)
    fig.tight_layout()

    #fig.show()
    figure_name = comb_file_name.replace('metric', f'{current_metric}_2')
    figure_path = os.path.join(plot_folder, figure_name)
    plt.savefig(figure_path, dpi = dpi)
    plt.close()
    


    current_metric = 'd_logit'
    d_rep_avg_var_dict = inverse_var_weighted_avg_and_var(metrics_df, 'd_LLD', use_sqrt_val=True)

    fontsize = 14

    fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    colours_to_use = ['#002347', '#004a95', '#9d3f00', '#ff6600']

    train_test = ['train', 'test']
    for j, current_distill in enumerate(distil_types):
        for i, current_train_test in enumerate(train_test):        
            current_idx = j*len(train_test)+i

            avg = np.array(d_rep_avg_var_dict[current_distill][f'iv_avg_{current_train_test}_pr_epoch'])
            std = np.sqrt(np.array(d_rep_avg_var_dict[current_distill][f'iv_avg_var_{current_train_test}_pr_epoch']))
            print(f'{current_distill} {current_train_test}: avg:{avg[-1]}, std:{std[-1]}')
            axes.plot(unique_epochs, avg, 
                        c=colours_to_use[current_idx], linewidth=2.5, label = f'{current_distill} {current_train_test}')
            axes.fill_between(
                unique_epochs, 
                avg-(2*std), avg+(2*std), 
                alpha = 0.2,
                color=colours_to_use[current_idx])
    
    axes.set_xlabel('epochs', fontsize = fontsize)
    axes.set_ylabel(current_metric, fontsize = fontsize)
    axes.legend(fontsize = fontsize)
    axes.tick_params(axis='both', which='major', labelsize=fontsize-2)
    axes.set_title(f'{current_metric} vs epochs', fontsize=fontsize+2)
    fig.tight_layout()

    #fig.show()
    figure_name = comb_file_name.replace('metric', current_metric)
    figure_path = os.path.join(plot_folder, figure_name)
    plt.savefig(figure_path, dpi = dpi)
    plt.close()



    current_metric = 'kl_div'
    d_rep_avg_var_dict = inverse_var_weighted_avg_and_var(metrics_df, current_metric)

    fontsize = 14

    fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    colours_to_use = ['#002347', '#004a95', '#9d3f00', '#ff6600']

    train_test = ['train', 'test']    
    for j, current_distill in enumerate(distil_types):
        for i, current_train_test in enumerate(train_test):        
            current_idx = j*len(train_test)+i

            avg = np.array(d_rep_avg_var_dict[current_distill][f'iv_avg_{current_train_test}_pr_epoch'])
            std = np.sqrt(np.array(d_rep_avg_var_dict[current_distill][f'iv_avg_var_{current_train_test}_pr_epoch']))
            print(f'{current_distill} {current_train_test}: avg:{avg[-1]}, std:{std[-1]}')
            axes.plot(unique_epochs, avg, 
                        c=colours_to_use[current_idx], linewidth=2.5, label = f'{current_distill} {current_train_test}')
            axes.fill_between(
                unique_epochs, 
                avg-(2*std), avg+(2*std), 
                alpha = 0.2,
                color=colours_to_use[current_idx])
    
    axes.set_xlabel('epochs', fontsize = fontsize)
    axes.set_ylabel(current_metric, fontsize = fontsize)
    axes.legend(fontsize = fontsize)
    axes.tick_params(axis='both', which='major', labelsize=fontsize-2)
    axes.set_title('kl_div vs epochs', fontsize=fontsize+2)
    fig.tight_layout()

    #fig.show()
    figure_name = comb_file_name.replace('metric', current_metric)
    figure_path = os.path.join(plot_folder, figure_name)
    plt.savefig(figure_path, dpi = dpi)
    plt.close()


    #figure_name = comb_file_name.replace('cls.json', f'cls{classes_str}.png')
    #figure_path = os.path.join(figure_folder, figure_name)
    #plt.savefig(figure_path, dpi = dpi)
    #plt.close()


    current_metric = 'acc'
    acc_distill_avg_var_d = inverse_var_weighted_avg_and_var(metrics_df, current_metric)
    acc_base_avg_var_d = inverse_var_weighted_avg_and_var(base_metrics_df, current_metric)

    fontsize = 14

    fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    colours_to_use = ['#002347', '#004a95', '#9d3f00', '#ff6600']
    base_model_colours = ['#006600', '#009900']

    train_test = ['train', 'test']
    for j, current_distill in enumerate(distil_types):
        for i, current_train_test in enumerate(train_test):        
            current_idx = j*len(train_test)+i

            avg = np.array(acc_distill_avg_var_d[current_distill][f'iv_avg_{current_train_test}_pr_epoch'])
            std = np.sqrt(np.array(acc_distill_avg_var_d[current_distill][f'iv_avg_var_{current_train_test}_pr_epoch']))
            print(f'{current_distill} {current_train_test}: avg:{avg[-1]}, std:{std[-1]}')
            axes.plot(unique_epochs, avg, 
                        c=colours_to_use[current_idx], linewidth=2.5, label = f'{current_distill} {current_train_test}')
            axes.fill_between(
                unique_epochs, 
                avg-(2*std), avg+(2*std), 
                alpha = 0.2,
                color=colours_to_use[current_idx])

    for i, current_train_test in enumerate(train_test):
        avg = np.array(acc_base_avg_var_d['classifier'][f'iv_avg_{current_train_test}_pr_epoch'])
        std = np.sqrt(np.array(acc_base_avg_var_d['classifier'][f'iv_avg_var_{current_train_test}_pr_epoch']))
        axes.plot(unique_epochs, avg, 
                    c=base_model_colours[i], linewidth=2.5, label = f'base {current_train_test}')
        axes.fill_between(
            unique_epochs, 
            avg-(2*std), avg+(2*std), 
            alpha = 0.2,
            color=base_model_colours[i])
    
    axes.set_xlabel('epochs', fontsize = fontsize)
    axes.set_ylabel(current_metric, fontsize = fontsize)
    axes.legend(fontsize = fontsize)
    axes.tick_params(axis='both', which='major', labelsize=fontsize-2)
    axes.set_title('acc vs epochs', fontsize=fontsize+2)
    fig.tight_layout()

    #fig.show()
    figure_name = comb_file_name.replace('metric', current_metric)
    figure_path = os.path.join(plot_folder, figure_name)
    plt.savefig(figure_path, dpi = dpi)
    plt.close()




    current_metric = 'acc_to_teacher'
    d_rep_avg_var_dict = inverse_var_weighted_avg_and_var(metrics_df, current_metric)

    fontsize = 14

    fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    colours_to_use = ['#002347', '#004a95', '#9d3f00', '#ff6600']

    train_test = ['train', 'test']
    for j, current_distill in enumerate(distil_types):
        for i, current_train_test in enumerate(train_test):        
            current_idx = j*len(train_test)+i

            avg = np.array(d_rep_avg_var_dict[current_distill][f'iv_avg_{current_train_test}_pr_epoch'])
            std = np.sqrt(np.array(d_rep_avg_var_dict[current_distill][f'iv_avg_var_{current_train_test}_pr_epoch']))
            axes.plot(unique_epochs, avg, 
                        c=colours_to_use[current_idx], linewidth=2.5, label = f'{current_distill} {current_train_test}')
            axes.fill_between(
                unique_epochs, 
                avg-(2*std), avg+(2*std), 
                alpha = 0.2,
                color=colours_to_use[current_idx])
            print(f'{current_distill} {current_train_test}: avg:{avg[-1]}, std:{std[-1]}')
    
    axes.set_xlabel('epochs', fontsize = fontsize)
    axes.set_ylabel('acc to teacher', fontsize = fontsize)
    axes.legend(fontsize = fontsize)
    axes.tick_params(axis='both', which='major', labelsize=fontsize-2)
    axes.set_title('acc to teacher vs epochs', fontsize=fontsize+2)
    fig.tight_layout()

    #fig.show()
    figure_name = comb_file_name.replace('metric', current_metric)
    figure_path = os.path.join(plot_folder, figure_name)
    plt.savefig(figure_path, dpi = dpi)
    plt.close()

    
    
    
    distil_types = ['logitdistill_l2', 'kldistill']
    comb_file_name = 'metric_over_epochs_l2_kl_iv_avg.png'

    current_metric = 'm_cca'
    cca_avg_var_dict = inverse_var_weighted_avg_and_var(metrics_df, current_metric)

    fontsize = 14

    fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    colours_to_use = ['#002347', '#004a95', '#9d3f00', '#ff6600']

    train_test = ['train', 'test']
    for j, current_distill in enumerate(distil_types):
        for i, current_train_test in enumerate(train_test):        
            current_idx = j*len(train_test)+i

            avg = np.array(cca_avg_var_dict[current_distill][f'iv_avg_{current_train_test}_pr_epoch'])
            std = np.sqrt(np.array(cca_avg_var_dict[current_distill][f'iv_avg_var_{current_train_test}_pr_epoch']))
            print(f'{current_distill} {current_train_test}: avg:{avg[-1]}, std:{std[-1]}')
            axes.plot(unique_epochs, avg, 
                        c=colours_to_use[current_idx], linewidth=2.5, label = f'{current_distill} {current_train_test}')
            axes.fill_between(
                unique_epochs, 
                avg-(2*std), avg+(2*std), 
                alpha = 0.2,
                color=colours_to_use[current_idx])
    
    axes.set_xlabel('epochs', fontsize = fontsize)
    axes.set_ylabel(current_metric, fontsize = fontsize)
    #axes.set_ylim(0, 1)
    axes.legend(fontsize = fontsize)
    axes.tick_params(axis='both', which='major', labelsize=fontsize-2)
    axes.set_title('m_CCA vs epochs', fontsize=fontsize+2)
    fig.tight_layout()

    #fig.show()
    figure_name = comb_file_name.replace('metric', current_metric)
    figure_path = os.path.join(plot_folder, figure_name)
    plt.savefig(figure_path, dpi = dpi)
    plt.close()


    current_metric = 'd_rep'
    d_rep_avg_var_dict = inverse_var_weighted_avg_and_var(metrics_df, current_metric)

    fontsize = 14

    fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    colours_to_use = ['#002347', '#004a95', '#9d3f00', '#ff6600']

    train_test = ['train', 'test']    
    for j, current_distill in enumerate(distil_types):
        for i, current_train_test in enumerate(train_test):        
            current_idx = j*len(train_test)+i

            avg = np.array(d_rep_avg_var_dict[current_distill][f'iv_avg_{current_train_test}_pr_epoch'])
            std = np.sqrt(np.array(d_rep_avg_var_dict[current_distill][f'iv_avg_var_{current_train_test}_pr_epoch']))
            print(f'{current_distill} {current_train_test}: avg:{avg[-1]}, std:{std[-1]}')
            axes.plot(unique_epochs, avg, 
                        c=colours_to_use[current_idx], linewidth=2.5, label = f'{current_distill} {current_train_test}')
            axes.fill_between(
                unique_epochs, 
                avg-(2*std), avg+(2*std), 
                alpha = 0.2,
                color=colours_to_use[current_idx])
    
    axes.set_xlabel('epochs', fontsize = fontsize)
    axes.set_ylabel(current_metric, fontsize = fontsize)
    axes.legend(fontsize = fontsize)
    axes.tick_params(axis='both', which='major', labelsize=fontsize-2)
    axes.set_title('d_rep^2 vs epochs', fontsize=fontsize+2)
    fig.tight_layout()

    #fig.show()
    figure_name = comb_file_name.replace('metric', f'{current_metric}_2')
    figure_path = os.path.join(plot_folder, figure_name)
    plt.savefig(figure_path, dpi = dpi)
    plt.close()


    current_metric = 'd_rep'
    d_rep_avg_var_dict = inverse_var_weighted_avg_and_var(metrics_df, current_metric, use_sqrt_val=True)

    fontsize = 14

    fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    colours_to_use = ['#002347', '#004a95', '#9d3f00', '#ff6600']

    train_test = ['train', 'test']    
    for j, current_distill in enumerate(distil_types):
        for i, current_train_test in enumerate(train_test):        
            current_idx = j*len(train_test)+i

            avg = np.array(d_rep_avg_var_dict[current_distill][f'iv_avg_{current_train_test}_pr_epoch'])
            std = np.sqrt(np.array(d_rep_avg_var_dict[current_distill][f'iv_avg_var_{current_train_test}_pr_epoch']))
            print(f'{current_distill} {current_train_test}: avg:{avg[-1]}, std:{std[-1]}')
            axes.plot(unique_epochs, avg, 
                        c=colours_to_use[current_idx], linewidth=2.5, label = f'{current_distill} {current_train_test}')
            axes.fill_between(
                unique_epochs, 
                avg-(2*std), avg+(2*std), 
                alpha = 0.2,
                color=colours_to_use[current_idx])
    
    axes.set_xlabel('epochs', fontsize = fontsize)
    axes.set_ylabel(current_metric, fontsize = fontsize)
    axes.legend(fontsize = fontsize)
    axes.tick_params(axis='both', which='major', labelsize=fontsize-2)
    axes.set_title('d_rep vs epochs', fontsize=fontsize+2)
    fig.tight_layout()

    #fig.show()
    figure_name = comb_file_name.replace('metric', current_metric)
    figure_path = os.path.join(plot_folder, figure_name)
    plt.savefig(figure_path, dpi = dpi)
    plt.close()


    current_metric = 'd_logit'
    d_rep_avg_var_dict = inverse_var_weighted_avg_and_var(metrics_df, 'd_LLD')

    fontsize = 14

    fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    colours_to_use = ['#002347', '#004a95', '#9d3f00', '#ff6600']

    train_test = ['train', 'test']
    for j, current_distill in enumerate(distil_types):
        for i, current_train_test in enumerate(train_test):        
            current_idx = j*len(train_test)+i

            avg = np.array(d_rep_avg_var_dict[current_distill][f'iv_avg_{current_train_test}_pr_epoch'])
            std = np.sqrt(np.array(d_rep_avg_var_dict[current_distill][f'iv_avg_var_{current_train_test}_pr_epoch']))
            print(f'{current_distill} {current_train_test}: avg:{avg[-1]}, std:{std[-1]}')
            axes.plot(unique_epochs, avg, 
                        c=colours_to_use[current_idx], linewidth=2.5, label = f'{current_distill} {current_train_test}')
            axes.fill_between(
                unique_epochs, 
                avg-(2*std), avg+(2*std), 
                alpha = 0.2,
                color=colours_to_use[current_idx])
    
    axes.set_xlabel('epochs', fontsize = fontsize)
    axes.set_ylabel(current_metric, fontsize = fontsize)
    axes.legend(fontsize = fontsize)
    axes.tick_params(axis='both', which='major', labelsize=fontsize-2)
    axes.set_title(f'{current_metric}^2 vs epochs', fontsize=fontsize+2)
    fig.tight_layout()

    #fig.show()
    figure_name = comb_file_name.replace('metric', f'{current_metric}_2')
    figure_path = os.path.join(plot_folder, figure_name)
    plt.savefig(figure_path, dpi = dpi)
    plt.close()
    


    current_metric = 'd_logit'
    d_rep_avg_var_dict = inverse_var_weighted_avg_and_var(metrics_df, 'd_LLD', use_sqrt_val=True)

    fontsize = 14

    fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    colours_to_use = ['#002347', '#004a95', '#9d3f00', '#ff6600']

    train_test = ['train', 'test']
    for j, current_distill in enumerate(distil_types):
        for i, current_train_test in enumerate(train_test):        
            current_idx = j*len(train_test)+i

            avg = np.array(d_rep_avg_var_dict[current_distill][f'iv_avg_{current_train_test}_pr_epoch'])
            std = np.sqrt(np.array(d_rep_avg_var_dict[current_distill][f'iv_avg_var_{current_train_test}_pr_epoch']))
            print(f'{current_distill} {current_train_test}: avg:{avg[-1]}, std:{std[-1]}')
            axes.plot(unique_epochs, avg, 
                        c=colours_to_use[current_idx], linewidth=2.5, label = f'{current_distill} {current_train_test}')
            axes.fill_between(
                unique_epochs, 
                avg-(2*std), avg+(2*std), 
                alpha = 0.2,
                color=colours_to_use[current_idx])
    
    axes.set_xlabel('epochs', fontsize = fontsize)
    axes.set_ylabel(current_metric, fontsize = fontsize)
    axes.legend(fontsize = fontsize)
    axes.tick_params(axis='both', which='major', labelsize=fontsize-2)
    axes.set_title(f'{current_metric} vs epochs', fontsize=fontsize+2)
    fig.tight_layout()

    #fig.show()
    figure_name = comb_file_name.replace('metric', current_metric)
    figure_path = os.path.join(plot_folder, figure_name)
    plt.savefig(figure_path, dpi = dpi)
    plt.close()



    current_metric = 'kl_div'
    d_rep_avg_var_dict = inverse_var_weighted_avg_and_var(metrics_df, current_metric)

    fontsize = 14

    fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    colours_to_use = ['#002347', '#004a95', '#9d3f00', '#ff6600']

    train_test = ['train', 'test']    
    for j, current_distill in enumerate(distil_types):
        for i, current_train_test in enumerate(train_test):        
            current_idx = j*len(train_test)+i

            avg = np.array(d_rep_avg_var_dict[current_distill][f'iv_avg_{current_train_test}_pr_epoch'])
            std = np.sqrt(np.array(d_rep_avg_var_dict[current_distill][f'iv_avg_var_{current_train_test}_pr_epoch']))
            print(f'{current_distill} {current_train_test}: avg:{avg[-1]}, std:{std[-1]}')
            axes.plot(unique_epochs, avg, 
                        c=colours_to_use[current_idx], linewidth=2.5, label = f'{current_distill} {current_train_test}')
            axes.fill_between(
                unique_epochs, 
                avg-(2*std), avg+(2*std), 
                alpha = 0.2,
                color=colours_to_use[current_idx])
    
    axes.set_xlabel('epochs', fontsize = fontsize)
    axes.set_ylabel(current_metric, fontsize = fontsize)
    axes.legend(fontsize = fontsize)
    axes.tick_params(axis='both', which='major', labelsize=fontsize-2)
    axes.set_title('kl_div vs epochs', fontsize=fontsize+2)
    fig.tight_layout()

    #fig.show()
    figure_name = comb_file_name.replace('metric', current_metric)
    figure_path = os.path.join(plot_folder, figure_name)
    plt.savefig(figure_path, dpi = dpi)
    plt.close()


    #figure_name = comb_file_name.replace('cls.json', f'cls{classes_str}.png')
    #figure_path = os.path.join(figure_folder, figure_name)
    #plt.savefig(figure_path, dpi = dpi)
    #plt.close()


    current_metric = 'acc'
    acc_distill_avg_var_d = inverse_var_weighted_avg_and_var(metrics_df, current_metric)
    acc_base_avg_var_d = inverse_var_weighted_avg_and_var(base_metrics_df, current_metric)

    fontsize = 14

    fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    colours_to_use = ['#002347', '#004a95', '#9d3f00', '#ff6600']
    base_model_colours = ['#006600', '#009900']

    train_test = ['train', 'test']
    for j, current_distill in enumerate(distil_types):
        for i, current_train_test in enumerate(train_test):        
            current_idx = j*len(train_test)+i

            avg = np.array(acc_distill_avg_var_d[current_distill][f'iv_avg_{current_train_test}_pr_epoch'])
            std = np.sqrt(np.array(acc_distill_avg_var_d[current_distill][f'iv_avg_var_{current_train_test}_pr_epoch']))
            print(f'{current_distill} {current_train_test}: avg:{avg[-1]}, std:{std[-1]}')
            axes.plot(unique_epochs, avg, 
                        c=colours_to_use[current_idx], linewidth=2.5, label = f'{current_distill} {current_train_test}')
            axes.fill_between(
                unique_epochs, 
                avg-(2*std), avg+(2*std), 
                alpha = 0.2,
                color=colours_to_use[current_idx])

    for i, current_train_test in enumerate(train_test):
        avg = np.array(acc_base_avg_var_d['classifier'][f'iv_avg_{current_train_test}_pr_epoch'])
        std = np.sqrt(np.array(acc_base_avg_var_d['classifier'][f'iv_avg_var_{current_train_test}_pr_epoch']))
        axes.plot(unique_epochs, avg, 
                    c=base_model_colours[i], linewidth=2.5, label = f'base {current_train_test}')
        axes.fill_between(
            unique_epochs, 
            avg-(2*std), avg+(2*std), 
            alpha = 0.2,
            color=base_model_colours[i])
    
    axes.set_xlabel('epochs', fontsize = fontsize)
    axes.set_ylabel(current_metric, fontsize = fontsize)
    axes.legend(fontsize = fontsize)
    axes.tick_params(axis='both', which='major', labelsize=fontsize-2)
    axes.set_title('acc vs epochs', fontsize=fontsize+2)
    fig.tight_layout()

    #fig.show()
    figure_name = comb_file_name.replace('metric', current_metric)
    figure_path = os.path.join(plot_folder, figure_name)
    plt.savefig(figure_path, dpi = dpi)
    plt.close()




    current_metric = 'acc_to_teacher'
    d_rep_avg_var_dict = inverse_var_weighted_avg_and_var(metrics_df, current_metric)

    fontsize = 14

    fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    colours_to_use = ['#002347', '#004a95', '#9d3f00', '#ff6600']

    train_test = ['train', 'test']
    for j, current_distill in enumerate(distil_types):
        for i, current_train_test in enumerate(train_test):        
            current_idx = j*len(train_test)+i

            avg = np.array(d_rep_avg_var_dict[current_distill][f'iv_avg_{current_train_test}_pr_epoch'])
            std = np.sqrt(np.array(d_rep_avg_var_dict[current_distill][f'iv_avg_var_{current_train_test}_pr_epoch']))
            axes.plot(unique_epochs, avg, 
                        c=colours_to_use[current_idx], linewidth=2.5, label = f'{current_distill} {current_train_test}')
            axes.fill_between(
                unique_epochs, 
                avg-(2*std), avg+(2*std), 
                alpha = 0.2,
                color=colours_to_use[current_idx])
            print(f'{current_distill} {current_train_test}: avg:{avg[-1]}, std:{std[-1]}')
    
    axes.set_xlabel('epochs', fontsize = fontsize)
    axes.set_ylabel('acc to teacher', fontsize = fontsize)
    axes.legend(fontsize = fontsize)
    axes.tick_params(axis='both', which='major', labelsize=fontsize-2)
    axes.set_title('acc to teacher vs epochs', fontsize=fontsize+2)
    fig.tight_layout()

    #fig.show()
    figure_name = comb_file_name.replace('metric', current_metric)
    figure_path = os.path.join(plot_folder, figure_name)
    plt.savefig(figure_path, dpi = dpi)
    plt.close()


