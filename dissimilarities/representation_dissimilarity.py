"""

Implementation of the linear identifiability dissimilarity, d_{rep}. 

"""




import torch 

import numpy as np

from tqdm import tqdm 

from utils.metrics import evaluate_reprs



def calculate_d_rep(f1, f2, g1, g2, 
                    num_pivots:int = 100, num_combinations:int = 200, seed:int = 0):
    """
    Calculating an approximation of the squared linear identifiability dissimilarity for two models
    
    :param f1: Embeddings of model 1
    :param f2: Embeddings of model 2
    :param g1: Unembeddings of model 1
    :param g2: Unembeddings of model 2
    :param num_pivots: Number of pivots to use in the approximation
    :param num_combinations: How many combinations of labels to use
    :param seed: Seed for the rng to pick the label combinations
    """
    rng = np.random.default_rng(seed)

    dimension = g1.shape[1]
    num_possible_pivots = g1.shape[0]
    num_embeddings = f1.shape[0]

    unemb_indexes = torch.arange(0,num_possible_pivots)
    pivot_indexes = rng.choice(unemb_indexes, num_pivots, replace=False)

    without_pivot_indexes = range(num_possible_pivots-1)
    centered_g1 = g1 - g1.mean(dim=0)
    centered_g2 = g2 - g2.mean(dim=0)

    mean_norms_2_f1_Af2 = []
    for p in tqdm(pivot_indexes):
        unemb_1_pivot = centered_g1[unemb_indexes == p]
        unemb_2_pivot = centered_g2[unemb_indexes == p]

        unemb_1_no_pivot = centered_g1[unemb_indexes != p]
        unemb_2_no_pivot = centered_g2[unemb_indexes != p]

        for _ in torch.arange(0,num_combinations):
            l_label_choices_idx = rng.choice(without_pivot_indexes, dimension, replace=False)

            l_unemb_1 = unemb_1_no_pivot[l_label_choices_idx]
            l_unemb_2 = unemb_2_no_pivot[l_label_choices_idx]

            L1_T = (l_unemb_1 - unemb_1_pivot)
            L2_T = (l_unemb_2 - unemb_2_pivot)

            current_A = torch.matmul(torch.linalg.inv(L1_T), L2_T)
            current_A_f2 = torch.matmul(current_A, f2.unsqueeze(2))

            current_norm_2_f1_Af2 = ((f1 - current_A_f2.squeeze())**2).sum(dim = 1)
            mean_norms_2_f1_Af2.append((current_norm_2_f1_Af2/(num_embeddings*num_pivots*num_combinations)).sum())

    d_rep = torch.Tensor(mean_norms_2_f1_Af2).mean()
    print(d_rep)

    return d_rep


def get_model_unembeddings(model):
    unembs = torch.Tensor(model.linear.state_dict()['weight'])
    return unembs


def calculate_d_rep_models(student_model, teacher_model, data_loader, args,
                           num_pivots:int = 100, num_combinations:int = 200, seed:int = 0):
    _, t_embs, _, s_embs = evaluate_reprs(student_model, data_loader, args, 
                                                    teacher=teacher_model,
                                                    last=True)
    t_embs = torch.Tensor(t_embs).to(student_model.device)
    s_embs = torch.Tensor(s_embs).to(student_model.device)
    t_unembs = get_model_unembeddings(teacher_model)
    s_unembs = get_model_unembeddings(student_model)

    return calculate_d_rep(f1=s_embs, f2=t_embs, g1=s_unembs, g2=t_unembs, 
                           num_pivots=num_pivots, num_combinations=num_combinations, seed=seed)
