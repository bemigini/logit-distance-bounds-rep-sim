import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def evaluate_mix(true, pred):
    ac = accuracy_score(true, pred)
    f1 = f1_score(true, pred, average='weighted')
    # pc = precision_score(true, pred)
    # rc = recall_score(true, pred)
    
    return ac, f1 #, pc, rc 

def evaluate_reprs(model, loader, args, teacher=None, last=False):
    L = len(loader)
    tloss, yacc = 0, 0

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            images, labels, concepts = data
            images, labels, concepts = images.to(model.device), labels.to(model.device), concepts.to(model.device)

            out_dict = model(images)
            out_dict.update({'INPUTS': images, 'LABELS': labels, 'CONCEPTS': concepts})
            
            if teacher is not None:
                with torch.no_grad():
                    t_out_dict = teacher(images)
                    out_dict.update({'TLOGITS': t_out_dict['LOGITS'], 'TEMBS': t_out_dict['EMBS']})

            if last and i == 0:
                x_true = images.detach().cpu().numpy()
                y_true = labels.detach().cpu().numpy()
                c_true = out_dict['TEMBS'].detach().cpu().numpy() if teacher is not None else \
                        concepts.detach().cpu().numpy()
                t_y_pred = out_dict['TLOGITS'].detach().cpu().numpy() if teacher is not None else \
                        labels.detach().cpu().numpy()
                y_pred = out_dict['LOGITS'].detach().cpu().numpy()
                c_pred = out_dict['EMBS'].detach().cpu().numpy()
            elif last and i > 0:  
                x_true = np.concatenate([x_true, images.detach().cpu().numpy()], axis=0)          
                y_true = np.concatenate([y_true, labels.detach().cpu().numpy()], axis=0)
                c_true = np.concatenate([c_true, out_dict['TEMBS'].detach().cpu().numpy()], axis=0) if teacher is not None else \
                        np.concatenate([c_true, concepts.detach().cpu().numpy()], axis=0)
                t_y_pred = np.concatenate([t_y_pred, out_dict['TLOGITS'].detach().cpu().numpy()], axis=0) if teacher is not None else \
                        np.concatenate([t_y_pred, labels.detach().cpu().numpy()], axis=0)
                y_pred = np.concatenate([y_pred, out_dict['LOGITS'].detach().cpu().numpy()], axis=0)
                c_pred = np.concatenate([c_pred, out_dict['EMBS'].detach().cpu().numpy()], axis=0)  
                    
            if args.dataset in ['synthetic'] and not last:
                loss, acc = eval_tloss_acc_reprs(out_dict, concepts)
            else:
                NotImplementedError()
            if not last:
                tloss += loss.item()
                yacc  += acc    
    
    
    if last:
        s_logits = y_pred
        t_logits = t_y_pred
        ys = np.argmax(s_logits, axis=1)
        teacher_pred = np.argmax(t_logits, axis=1) if teacher is not None else t_y_pred
        gs = c_true
        cs = c_pred
        
        # assert gs.shape == cs.shape, f'gs: {gs.shape}, cs: {cs.shape}'
                
        return y_true, gs, ys, cs, teacher_pred, s_logits, t_logits
    else:          
        return tloss / L, yacc /L
    

def eval_tloss_acc_reprs(out_dict, concepts):
    
    y = out_dict['LOGITS']
    y_true = out_dict['LABELS']

    loss = torch.nn.CrossEntropyLoss()(y, y_true).item()

    y_pred = torch.argmax(y, dim=-1)
    assert y_pred.size() == y_true.size(), f'size PREDS: {y_pred.size()}, size GT: {y_true.size()}' 

    acc = (y_pred == y_true).sum().item() / len(y_true)
    
    return loss, acc * 100

# def evaluate_metrics(model, loader, args, last=False):
#     L = len(loader)
#     tloss, cacc, yacc = 0, 0, 0
#     for i, data in enumerate(loader):
#         images, labels, concepts = data
#         images, labels, concepts = images.to(model.device), labels.to(model.device), concepts.to(model.device)

#         out_dict = model(images)
#         out_dict.update({'INPUTS': images, 'LABELS': labels, 'CONCEPTS': concepts})
        
#         if last and i == 0:
#             y_true = labels.detach().cpu().numpy()
#             c_true = concepts.detach().cpu().numpy()
#             y_pred = out_dict['YS'].detach().cpu().numpy()
#             c_pred = out_dict['CS'].detach().cpu().numpy()
#         elif last and i > 0:            
#             y_true = np.concatenate([y_true, labels.detach().cpu().numpy()], axis=0)
#             c_true = np.concatenate([c_true, concepts.detach().cpu().numpy()], axis=0)
#             y_pred = np.concatenate([y_pred, out_dict['YS'].detach().cpu().numpy()], axis=0)
#             c_pred = np.concatenate([c_pred, out_dict['CS'].detach().cpu().numpy()], axis=0)  
#         if args.dataset in ['addmnist', 'shortmnist', 'restrictedmnist'] and not last:
#             loss, ac, acc = ADDMNIST_eval_tloss_cacc_acc(out_dict, concepts)
#         else:
#             NotImplementedError()
#         if not last:
#             tloss += loss.item()
#             cacc  += ac
#             yacc  += acc
#     if last:            
#         ys = np.argmax(y_pred, axis=1)
#         gs = np.split(c_true, c_true.shape[1], axis=1)
#         cs = np.split(c_pred, c_pred.shape[1], axis=1)
#         assert len(gs) == len(cs), f'gs: {gs.shape}, cs: {cs.shape}'
#         gs = np.concatenate(gs, axis=0).squeeze(1)
#         cs = np.concatenate(cs, axis=0).squeeze(1).argmax(axis=1)
#         assert gs.shape == cs.shape, f'gs: {gs.shape}, cs: {cs.shape}'
#         return y_true, gs, ys, cs
#     else:          
#         return tloss / L, cacc / L, yacc /L

def calibration_score(y_true, logits):
    from sklearn.calibration import calibration_curve

    print(y_true.shape)
    print(logits.shape)

    logits = torch.from_numpy(logits)

    y_prob = torch.nn.functional.softmax(logits, dim=-1).detach().numpy()
    y_prob = np.max(y_prob, axis=1)

    y_true = np.ones_like(y_prob)

    ca_true, ca_pred = calibration_curve(y_true, y_prob, n_bins=10)

    score = np.sum(np.abs(ca_pred - ca_true))

    return score


def expected_calibration_error(y_true_binary, y_prob, n_bins=15):
    """
    Computes ECE for binary y_true_binary and probability vector y_prob.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_prob)

    for i in range(n_bins):
        mask = (y_prob > bins[i]) & (y_prob <= bins[i+1])
        if np.sum(mask) == 0:
            continue
        prob_avg = np.mean(y_prob[mask])
        true_avg = np.mean(y_true_binary[mask])
        ece += np.abs(prob_avg - true_avg) * np.sum(mask)

    return ece / n


def multiclass_calibration(y_true, y_preds, n_bins=15):
    """
    y_preds: (n, k) predicted probabilities for k classes
    y_true:  (n,) integer class labels
    returns calibration metrics per class and macro averages
    """
    n, k = y_preds.shape

    # One-vs-rest binarization
    Y = np.zeros((n, k))
    Y[np.arange(n), y_true] = 1

    results = {
        "brier_per_class": [],
        "ece_per_class": [],
        "calibration_curve_per_class": [],  # list of (bin_true, bin_pred)
    }

    for c in range(k):
        y_true_c = Y[:, c]
        y_prob_c = y_preds[:, c]

        # Brier score
        brier = np.mean((y_prob_c - y_true_c) ** 2)

        # ECE
        ece = expected_calibration_error(y_true_c, y_prob_c, n_bins)

        # Calibration curve (bin statistics)
        bins = np.linspace(0, 1, n_bins + 1)
        bin_true = []
        bin_pred = []

        for i in range(n_bins):
            mask = (y_prob_c > bins[i]) & (y_prob_c <= bins[i+1])
            if np.sum(mask) == 0:
                bin_true.append(np.nan)
                bin_pred.append(np.nan)
            else:
                bin_true.append(np.mean(y_true_c[mask]))
                bin_pred.append(np.mean(y_prob_c[mask]))

        results["brier_per_class"].append(brier)
        results["ece_per_class"].append(ece)
        # results["calibration_curve_per_class"].append((bin_true, bin_pred))

    # Macro averages
    results["brier_macro"] = np.nanmean(results["brier_per_class"])
    results["ece_macro"]   = np.nanmean(results["ece_per_class"])

    return results

def neg_log_likelihood(y_true, y_preds):
    y_preds = torch.from_numpy(y_preds)
    y_true  = torch.from_numpy(y_true).to(torch.long)

    y_preds = y_preds + 1e-5
    y_preds /= torch.sum(y_preds, dim=-1, keepdim=True)
    
    results = {}
    results['nll'] = torch.nn.functional.nll_loss(y_preds.log(), y_true, reduction='mean').item()

    return results

def prob_to_rankings(P):
    """Convert probability matrix (n, k) → rankings (n, k)."""
    return np.argsort(-P, axis=1)


def pairwise_accuracy_rank(r_t, r_s):
    n, k = r_t.shape
    accs = []

    for i in range(n):
        pos_t = np.zeros(k, dtype=int)
        pos_s = np.zeros(k, dtype=int)
        pos_t[r_t[i]] = np.arange(k)
        pos_s[r_s[i]] = np.arange(k)

        correct = 0
        total = 0
        for a in range(k):
            for b in range(a+1, k):
                total += 1
                if (pos_t[a] < pos_t[b]) == (pos_s[a] < pos_s[b]):
                    correct += 1

        accs.append(correct / total)

    return np.mean(accs)


def kendall_tau_rank(r_t, r_s):
    n, k = r_t.shape
    taus = []

    for i in range(n):
        pos_t = np.zeros(k, dtype=int)
        pos_s = np.zeros(k, dtype=int)
        pos_t[r_t[i]] = np.arange(k)
        pos_s[r_s[i]] = np.arange(k)

        conc = 0
        disc = 0

        for a in range(k):
            for b in range(a+1, k):
                if (pos_t[a] < pos_t[b]) == (pos_s[a] < pos_s[b]):
                    conc += 1
                else:
                    disc += 1

        taus.append((conc - disc) / (conc + disc))

    return np.mean(taus)


def spearman_rank_corr(r_t, r_s):
    n, k = r_t.shape
    rhos = []

    for i in range(n):
        pos_t = np.zeros(k, dtype=int)
        pos_s = np.zeros(k, dtype=int)
        pos_t[r_t[i]] = np.arange(k)
        pos_s[r_s[i]] = np.arange(k)

        d = pos_t - pos_s
        rho = 1 - (6 * np.sum(d * d)) / (k * (k * k - 1))
        rhos.append(rho)

    return np.mean(rhos)


def dcg(scores):
    return np.sum(scores / np.log2(np.arange(2, len(scores) + 1 + 1)))


def ndcg(p_t, r_s):
    n, k = p_t.shape
    scores = []

    for i in range(n):
        relevance_student = p_t[i][r_s[i]]
        relevance_teacher = np.sort(p_t[i])[::-1]
        scores.append(dcg(relevance_student) / dcg(relevance_teacher))

    return np.mean(scores)


# ------------------------------------------------------
# MAIN EVALUATION FUNCTION
# ------------------------------------------------------

def evaluate_ranking_metrics(p_t, p_s):
    """
    p_t: teacher probabilities (n, k)
    p_s: student probabilities (n, k)
    returns: dict of ranking similarity metrics
    """
    r_t = prob_to_rankings(p_t)
    r_s = prob_to_rankings(p_s)

    return {
        "pairwise_accuracy": pairwise_accuracy_rank(r_t, r_s),
        "kendall_tau": kendall_tau_rank(r_t, r_s),
        "spearman_rho": spearman_rank_corr(r_t, r_s),
        "ndcg": ndcg(p_t, r_s),
    }
    
from sklearn.cross_decomposition import CCA

def mCCA(rep1, rep2, n_components) -> float:
    """ Get mean canonical correlation of the two representations.
        Uses [n_components] components for the CCA.
    """
    cca = CCA(n_components=n_components, max_iter=1000)
    cca.fit(rep1, rep2)

    # Mean of the CCA correlations
    X_c, Y_c = cca.transform(rep1, rep2)
    corrs = [np.corrcoef(X_c[:, k], Y_c[:, k])[0, 1] for k in range(n_components)]
    mean_corr = np.mean(corrs)

    return mean_corr, cca.x_rotations_, cca.y_rotations_, np.max(corrs) 


def MSE(rep1, rep2) -> float:
    """ Mean squared error between representations 1 and 2 """
    return np.sum((rep1 - rep2) ** 2) / len(rep1)


from tqdm import tqdm 

def calculate_d_rep(f1, f2, g1, g2, 
                    num_pivots:int = 100, num_combinations:int = 200, seed:int = 0):
    rng = np.random.default_rng(seed)


    dimension = g1.shape[1]
    num_possible_pivots = g1.shape[0]
    num_embeddings = f1.shape[0]


    unemb_indexes = torch.arange(0,num_possible_pivots)
    pivot_indexes = rng.choice(unemb_indexes, num_pivots, replace=False)

    without_pivot_indexes = range(num_possible_pivots-1)

    mean_norms_f1_Af2 = []
    for p in tqdm(pivot_indexes):
        unemb_1_pivot = g1[unemb_indexes == p]
        unemb_2_pivot = g2[unemb_indexes == p]

        unemb_1_no_pivot = g1[unemb_indexes != p]
        unemb_2_no_pivot = g2[unemb_indexes != p]

        for _ in torch.arange(0,num_combinations):
            l_label_choices_idx = rng.choice(without_pivot_indexes, dimension, replace=False)

            l_unemb_1 = unemb_1_no_pivot[l_label_choices_idx]
            l_unemb_2 = unemb_2_no_pivot[l_label_choices_idx]

            L1_T = (l_unemb_1 - unemb_1_pivot)
            L2_T = (l_unemb_2 - unemb_2_pivot)

            current_A = torch.matmul(torch.linalg.inv(L1_T), L2_T)
            current_A_f2 = torch.matmul(current_A, f2.unsqueeze(2))

            current_norm_f1_Af2 = torch.linalg.norm(f1 - current_A_f2.squeeze(), dim = 1)
            mean_norms_f1_Af2.append((current_norm_f1_Af2/(num_embeddings*num_pivots*num_combinations)).sum())
            #norms_f1_Af2.extend(current_norm_f1_Af2)

    #mean_norms_f1_Af2 = torch.Tensor(norms_f1_Af2).mean()

    d_rep = torch.Tensor(mean_norms_f1_Af2).sum()
    # print(d_rep)

    return d_rep


def KL_from_logits(logits_p, logits_q, temperature=1.0):
    """ Compute KL Divergence D_KL(P || Q) from logits.
        logits_p: logits from distribution P
        logits_q: logits from distribution Q
        temperature: temperature scaling for softening distributions
    """
    p = torch.nn.functional.softmax(logits_p / temperature, dim=-1)
    q = torch.nn.functional.softmax(logits_q / temperature, dim=-1)

    #add constant to avoid log(0)
    p = p + 1e-10
    p = p / torch.sum(p, dim=-1, keepdim=True)
    q = q + 1e-10
    q = q / torch.sum(q, dim=-1, keepdim=True)


    kl = torch.nn.functional.kl_div(p.log(), q, reduction='batchmean') * (temperature ** 2)

    return kl