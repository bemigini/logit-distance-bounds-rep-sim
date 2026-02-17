import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import sys, os 

conf_path = os.getcwd() 
sys.path.append(conf_path)

print(conf_path)

np.set_printoptions(precision=5, suppress=True)

from dsets.synthetic import SYNTHETIC
from models.logitdistill import LogitDistill
from models.kldistill import KLDistill
from models.classifier import Classifier

import pandas as pd


class fake_args:
    def __init__(self, dataset='synthetic', model='classifier', batch_size=1024):
        self.dataset = dataset
        self.model = model
        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.lr = 0.001
        self.n_epochs = 250
        self.seed = 42
        self.exp_decay = 0.995
        self.lambda_distill = 0.5
        self.modality = 'standard'
        self.min_radius = 1
        self.max_radius = 10

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.cross_decomposition import CCA


def evaluate_models_synthetic(teacher_location, klstudent_location, logstudent_location, test_loader):
    teacher.load_state_dict(torch.load(teacher_location))
    teacher.eval()
    
    kl_student.load_state_dict(torch.load(klstudent_location))
    kl_student.eval()

    log_student.load_state_dict(torch.load(logstudent_location))
    log_student.eval()

    X_train, Y_train, Z_train = [], [], []
    tlogs, kllogs, llogs = [], [], []
    treprs, klreprs, lreprs = [], [], []
    for data in test_loader:
        X, Y, Z = data
        X_train.append(X.numpy())
        Y_train.append(Y.numpy())
        Z_train.append(Z.numpy())   

        with torch.no_grad():
            t_out = teacher(X)
            kl_out = kl_student(X)
            log_out = log_student(X)

            tlogs.append(t_out['LOGITS'].cpu().numpy())
            kllogs.append(kl_out['LOGITS'].cpu().numpy())
            llogs.append(log_out['LOGITS'].cpu().numpy())
            
            treprs.append(t_out['EMBS'].cpu().numpy())
            klreprs.append(kl_out['EMBS'].cpu().numpy())
            lreprs.append(log_out['EMBS'].cpu().numpy())

    X_test = np.concatenate(X_train, axis=0)
    Y_test = np.concatenate(Y_train, axis=0)
    Z_test = np.concatenate(Z_train, axis=0)

    tlogs = np.concatenate(tlogs, axis=0)
    kllogs = np.concatenate(kllogs, axis=0)
    llogs = np.concatenate(llogs, axis=0)

    treprs = np.concatenate(treprs, axis=0)
    klreprs = np.concatenate(klreprs, axis=0)
    lreprs = np.concatenate(lreprs, axis=0)

    acc_teacher    = accuracy_score(Y_test, np.argmax(tlogs, axis=1))
    acc_klstudent  = accuracy_score(Y_test, np.argmax(kllogs, axis=1))
    acc_logstudent = accuracy_score(Y_test, np.argmax(llogs, axis=1))

    f1_score_teacher   = f1_score(Y_test, np.argmax(tlogs, axis=1), average='macro')
    f1_score_klstudent = f1_score(Y_test, np.argmax(kllogs, axis=1), average='macro')
    f1_score_logstudent = f1_score(Y_test, np.argmax(llogs, axis=1), average='macro')

    from utils.metrics import multiclass_calibration, evaluate_ranking_metrics, neg_log_likelihood


    t_probs  = torch.softmax(torch.from_numpy(tlogs),  dim=1).detach().numpy()
    kl_probs = torch.softmax(torch.from_numpy(kllogs), dim=1).detach().numpy()
    l_probs  = torch.softmax(torch.from_numpy(llogs),  dim=1).detach().numpy()

    nll_teacher = neg_log_likelihood(Y_test, t_probs)
    nll_kl =  neg_log_likelihood(Y_test, kl_probs)['nll']
    nll_log = neg_log_likelihood(Y_test, l_probs)['nll']

    cal_teacher    = multiclass_calibration(Y_test, t_probs)
    cal_klstudent  = multiclass_calibration(Y_test, kl_probs)
    cal_logstudent = multiclass_calibration(Y_test, l_probs)

    kl_ranks = evaluate_ranking_metrics(t_probs, kl_probs)
    l_ranks  = evaluate_ranking_metrics(t_probs, l_probs)

    

    # print('Teacher Accuracy:', acc_teacher, f1_score_teacher)
    # print('        Calibration', cal_teacher['brier_macro'], cal_teacher['ece_macro'])
    # print('KL Student Accuracy:', acc_klstudent, f1_score_klstudent) 
    # print('         Calibration', cal_klstudent['brier_macro'], cal_klstudent['ece_macro'])
    # print('         Rankings', kl_ranks)
    # print('Logit Student Accuracy:', acc_logstudent, f1_score_logstudent)
    # print('          Calibration', cal_logstudent['brier_macro'], cal_logstudent['ece_macro'])
    # print('          Rankings', l_ranks)

    # print('\n')

    cca_t_kl = CCA(n_components=2)
    cca_t_log = CCA(n_components=2)
    cca_kl_log = CCA(n_components=2)

    cca_t_kl.fit(treprs, klreprs)
    s_t_kl = cca_t_kl.score(treprs, klreprs)

    cca_t_log.fit(treprs, lreprs)
    s_t_log = cca_t_log.score(treprs, lreprs)

    cca_kl_log.fit(klreprs, lreprs)
    s_kl_log = cca_kl_log.score(klreprs, lreprs)

    # print('CCA Teacher-KL Student:', s_t_kl)
    # print('CCA Teacher-Logit Student:', s_t_log)
    # print('CCA KL Student-Logit Student:', s_kl_log)

    kl_dict = {'acc': acc_klstudent, 'f1': f1_score_klstudent, 
               'nll': nll_kl,
               'mbrier': cal_klstudent['brier_macro'], 'mece': cal_klstudent['ece_macro'],
               'kendall': kl_ranks['kendall_tau'], 'spearman': kl_ranks['spearman_rho'], 'ndcg': kl_ranks['ndcg'], 
               'cca': s_t_kl 
               }

    l_dict  = {'acc': acc_logstudent, 'f1': f1_score_logstudent, 
               'nll': nll_log,
               'mbrier': cal_logstudent['brier_macro'], 'mece': cal_logstudent['ece_macro'],
               'kendall': l_ranks['kendall_tau'], 'spearman': l_ranks['spearman_rho'], 'ndcg': l_ranks['ndcg'], 
               'cca': s_t_log 
               }

    return kl_dict, l_dict # [acc_teacher, acc_klstudent, acc_logstudent], [f1_score_teacher, f1_score_klstudent, f1_score_logstudent], [s_t_kl, s_t_log, s_kl_log]

def extract_model_representations(model, loader):
    return 0

if __name__=='__main__':

    args_test = fake_args()

    dataset = SYNTHETIC(args_test)
    train_loader, val_loader, test_loader = dataset.get_data_loaders()
    encoder_teach = dataset.get_backbone()
    encoder_kl = dataset.get_backbone()
    encoder_log = dataset.get_backbone()

    teacher = Classifier(encoder_teach, args_test, n_classes=7)
    kl_student = KLDistill(encoder_kl, args_test, n_classes=7)
    log_student = LogitDistill(encoder_log, args_test, n_classes=7) 

    train_loader, val_loader, test_loader = dataset.get_data_loaders()

    teacher_location = 'data/runs/synthetic-classifier-epoch1499-seed5.pt'
    # klstudent_location = 'data/runs/synthetic-kldistill-epoch249-seed1.pt'
    # logstudent_location = 'data/runs/synthetic-logitdistill-epoch249-seed1.pt'
    rows_kl, rows_log = [], []
    for i in range(10):
        n = 70*(i+1)
        klstudent_location  = f'data/runs/synthetic-standard-N{n}-kldistill-epoch999-seed1.pt'
        logstudent_location = f'data/runs/synthetic-standard-N{n}-logitdistill-epoch999-seed1.pt'

        #accs, f1s, ccas 
        kl_metrics, l_metrics = evaluate_models_synthetic(teacher_location, klstudent_location, logstudent_location, test_loader)
        
        kl_metrics['run'] = i
        l_metrics['run']  = i

        kl_metrics['N'] = n
        l_metrics['N'] = n
        
        
        rows_kl.append(kl_metrics), rows_log.append(l_metrics)

    df_kl  = pd.DataFrame(rows_kl).set_index('run')
    df_log = pd.DataFrame(rows_log).set_index('run')

    
    df_kl.to_csv('data/N_runs_kl_student_standard.csv')
    df_log.to_csv('data/N_runs_log_student_standard.csv')
    

    quit()