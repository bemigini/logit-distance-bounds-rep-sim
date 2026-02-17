"""
Implementation of the logit distance, d_{logit}
"""

import torch 



def calculate_d_2_logit(f1, f2, g1, g2):
    """
    Calculating the squared logit distance for two models
    
    :param f1: Embeddings of model 1
    :param f2: Embeddings of model 2
    :param g1: Unembeddings of model 1
    :param g2: Unembeddings of model 2
    """
    dimension = g1.shape[1]
    num_labels = g1.shape[0]
    num_embeddings = f1.shape[0]
    centered_g1 = g1 - g1.mean(dim=0)
    centered_g2 = g2 - g2.mean(dim=0)

    logits1 = torch.zeros(num_labels, num_embeddings)
    logits2 = torch.zeros(num_labels, num_embeddings)
    for i, _ in enumerate(g1):
        logits1[i] = torch.matmul(f1.unsqueeze(1), centered_g1[i].unsqueeze(1)).squeeze()
        logits2[i] = torch.matmul(f2.unsqueeze(1), centered_g2[i].unsqueeze(1)).squeeze()

    logit_diff_norm_2 = ((logits1-logits2)**2).sum(dim=0)

    d_2_logit = logit_diff_norm_2.mean()

    return d_2_logit
