"""

Evaluation on both train and test data


"""

from typing import Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from dissimilarities.distribution_metric import calculate_d_2_logit
from dissimilarities.representation_dissimilarity import get_model_unembeddings, calculate_d_rep
from dissimilarities.mean_cca import mCCA
from utils.metrics import evaluate_reprs, evaluate_mix
from sklearn.cross_decomposition import CCA


def evaluate_train_test(model, teacher_model, train_loader, test_loader, args) -> Tuple[Tensor, Tensor]:
    """
    Evaluate accuracies and representations 
    """
    model.to(model.device)
    model.eval()

    y_true, t_embs, y_pred, s_embs, teacher_pred, s_logits, t_logits = evaluate_reprs(model, test_loader, args, 
                                                    teacher=teacher_model,
                                                    last=True)

    yac, yf1 = evaluate_mix(y_true, y_pred)
    print(f'\nLabels:\n      ACC: {yac}, F1: {yf1}\n')
    s_to_t_acc, s_to_t_f1 = evaluate_mix(teacher_pred, y_pred)
    

    y_true_train, t_embs_train, y_pred_train, s_embs_train, teacher_pred_train, s_logits_train, t_logits_train = evaluate_reprs(model, train_loader, args, 
                                                    teacher=teacher_model,
                                                    last=True)

    yac_train, yf1_train = evaluate_mix(y_true_train, y_pred_train)
    print(f'Labels:\n      ACC train: {yac_train}, F1: {yf1_train}')
    s_to_t_acc_train, s_to_t_f1_train = evaluate_mix(teacher_pred_train, y_pred_train)


    if teacher_model is not None:
        print(f'Student to teacher:\n      ACC: {s_to_t_acc}, F1: {s_to_t_f1}\n')
        print(f'Student to teacher:\n      ACC train: {s_to_t_acc_train}, F1: {s_to_t_f1_train}\n')

        kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        log_p_target = F.log_softmax(torch.Tensor(t_logits).to(model.device), dim=1)
        log_p_student = F.log_softmax(torch.Tensor(s_logits).to(model.device), dim=1)
        kl_div_test = kl_loss(log_p_student, log_p_target)
        t_min_prob_test = torch.exp(log_p_target).min()
        s_min_prob_test = torch.exp(log_p_student).min()

        kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        log_p_target = F.log_softmax(torch.Tensor(t_logits_train).to(model.device), dim=1)
        log_p_student = F.log_softmax(torch.Tensor(s_logits_train).to(model.device), dim=1)
        kl_div_train = kl_loss(log_p_student, log_p_target)
        t_min_prob_train = torch.exp(log_p_target).min()
        s_min_prob_train = torch.exp(log_p_student).min()

        acc_metrics = (yac, yf1, yac_train, yf1_train, 
                       s_to_t_acc, s_to_t_f1, s_to_t_acc_train, s_to_t_f1_train, 
                       kl_div_test, kl_div_train, 
                       t_min_prob_test, s_min_prob_test, t_min_prob_train, s_min_prob_train)

        teacher_model.to(model.device)
        teacher_model.eval()
        cca = CCA(n_components=model.repr_dim, max_iter = 1000)
        cca.fit(t_embs, s_embs)
        cca_score_test = cca.score(t_embs, s_embs)
        print(f'CCA score between Teacher and Student Embeddings on ID test data: {cca_score_test}\n')

        cca_train = CCA(n_components=model.repr_dim, max_iter = 1000)
        cca_train.fit(t_embs_train, s_embs_train)
        cca_score_train = cca_train.score(t_embs_train, s_embs_train)
        print(f'CCA score between Teacher and Student Embeddings on train data: {cca_score_train}\n')


        if args.all_metrics:
            # m_CCA
            m_cca_test, _, _, _ = mCCA(t_embs, s_embs, model.repr_dim)
            m_cca_train, _, _, _ = mCCA(t_embs_train, s_embs_train, model.repr_dim)

            print(f'm_CCA between Teacher and Student Embeddings on ID test data: {m_cca_test}\n')
            print(f'm_CCA between Teacher and Student Embeddings on train data: {m_cca_train}\n')

            if args.dataset == 'synthetic':
                num_pivots = 7
                num_combinations = 15
            elif args.dataset == 'cifar100':
                num_pivots = 100
                num_combinations = 200
            elif args.dataset == 'sub':
                num_pivots = 33
                num_combinations = 250
            else:
                raise NotImplementedError(f'num pivots and combinations not implemented for dataset: {args.dataset}')

            seed = 0

            t_unembs = get_model_unembeddings(teacher_model)
            s_unembs = get_model_unembeddings(model)

            t_embs = torch.Tensor(t_embs).to(model.device)
            s_embs = torch.Tensor(s_embs).to(model.device)

            d_rep_test = calculate_d_rep(f1 =s_embs, f2=t_embs, g1=s_unembs, g2=t_unembs, 
                            num_pivots=num_pivots, num_combinations=num_combinations, seed=seed)
            d_LLD_test = calculate_d_2_logit(f1 =s_embs, f2=t_embs, g1=s_unembs, g2=t_unembs)
            
            t_embs_train = torch.Tensor(t_embs_train).to(model.device)
            s_embs_train = torch.Tensor(s_embs_train).to(model.device)

            d_rep_train = calculate_d_rep(f1 =s_embs_train, f2=t_embs_train, g1=s_unembs, g2=t_unembs, 
                            num_pivots=num_pivots, num_combinations=num_combinations, seed=seed)
            d_LLD_train = calculate_d_2_logit(f1 =s_embs_train, f2=t_embs_train, g1=s_unembs, g2=t_unembs)
            
            rep_metrics = (cca_score_test, cca_score_train, d_rep_test, d_rep_train, d_LLD_test, d_LLD_train, m_cca_test, m_cca_train)
        else:
            rep_metrics = (cca_score_test, cca_score_train)
    else:
        rep_metrics = None
        acc_metrics = (yac, yf1, yac_train, yf1_train)

    return y_true, t_embs, y_pred, s_embs, acc_metrics, rep_metrics


def posthoc_eval(model, dataset, teacher_model, args):
    """
    For posthoc evaluation
    """
    model.to(model.device)
    train_loader, _, test_loader = dataset.get_data_loaders()
    _ = evaluate_train_test(
        model, teacher_model=teacher_model, train_loader=train_loader, test_loader=test_loader, args=args)
    

