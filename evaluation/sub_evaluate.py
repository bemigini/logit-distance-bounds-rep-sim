import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import sys, os 

conf_path = os.getcwd() 
sys.path.append(conf_path)

print(conf_path)

np.set_printoptions(precision=5, suppress=True)

from dsets.sub import SUB
from models.logitdistill import LogitDistill
from models.kldistill import KLDistill
from models.classifier import Classifier
from models.cbm import CBM

import pandas as pd


class fake_args:
    def __init__(self, dataset='sub', model='classifier', batch_size=1024):
        self.dataset = dataset
        self.model = model
        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.lr = 0.001
        self.n_epochs = 250
        self.seed = 42
        self.exp_decay = 0.995
        self.modality = 'standard'
        self.lambda_distill = 0.5
        self.min_radius = 1
        self.max_radius = 10

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from utils.metrics import KL_from_logits, MSE, mCCA, calculate_d_rep


def create_histogram_of_norms(Embs_model, model_name):
    norms = np.linalg.norm(Embs_model, axis=1)
    plt.figure()
    plt.hist(np.log(norms), bins=100)
    plt.title(f'Histogram of representation norms for {model_name}')
    plt.xlabel('Norm')
    plt.ylabel('Frequency')
    plt.savefig(f'data/debug_{model_name}_representation_norms_histogram.png')
    plt.close()

def fit_linear_probe_concepts(C_true, Embs_model, save_str=''):
    X_train, X_test, y_train, y_test = train_test_split(
        Embs_model, C_true, test_size=0.5, random_state=42, stratify=C_true
    )

    clf = LogisticRegression().fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    #evaluate negative log likelihood
    # extract probabilities from clf
    probs = clf.predict_proba(X_test)
    nll = negative_log_likelihood(probs, y_test)

    # save Embs_model, C_true, y_pred, linear probe coefficients for debugging
    os.makedirs(f'data/{save_str}', exist_ok=True)
    np.save(f'data/{save_str}/Embs_model_.npy', Embs_model)
    np.save(f'data/{save_str}/C_true.npy', C_true)
    np.save(f'data/{save_str}/y_pred.npy', y_pred)
    np.save(f'data/{save_str}/linear_probe_coefficients.npy', clf.coef_)
    np.save(f'data/{save_str}/linear_probe_intercept.npy', clf.intercept_)

    return acc, f1, nll

def negative_log_likelihood(embs, labels):
    logits = torch.from_numpy(embs)
    labels = torch.from_numpy(labels).to(torch.long)

    nll_loss = F.cross_entropy(logits, labels, reduction='mean').item()
    return nll_loss


def evaluate_models_sub(teacher_location, klstudent_location, logstudent_location, test_loader,
                        save_str=''):
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

    create_histogram_of_norms(treprs, 'Teacher')
    create_histogram_of_norms(klreprs, 'KL Student')
    create_histogram_of_norms(lreprs, 'Logit Student')

    t_c_acc, t_c_f1, t_c_nll = fit_linear_probe_concepts(Z_test, treprs, save_str='teacher_' + save_str)
    kl_c_acc, kl_c_f1, kl_c_nll = fit_linear_probe_concepts(Z_test, klreprs, save_str='kl_student_' + save_str)
    l_c_acc, l_c_f1, l_c_nll = fit_linear_probe_concepts(Z_test, lreprs, save_str='logit_student_' + save_str)

    acc_teacher    = accuracy_score(Y_test, np.argmax(tlogs, axis=1))
    acc_klstudent  = accuracy_score(Y_test, np.argmax(kllogs, axis=1))
    acc_logstudent = accuracy_score(Y_test, np.argmax(llogs, axis=1))

    f1_score_teacher   = f1_score(Y_test, np.argmax(tlogs, axis=1), average='macro')
    f1_score_klstudent = f1_score(Y_test, np.argmax(kllogs, axis=1), average='macro')
    f1_score_logstudent = f1_score(Y_test, np.argmax(llogs, axis=1), average='macro')

    kl_klstudent = KL_from_logits(torch.from_numpy(tlogs), torch.from_numpy(kllogs)).item()
    kl_logstudent = KL_from_logits(torch.from_numpy(tlogs), torch.from_numpy(llogs)).item()

    mse_kl  = MSE(tlogs, kllogs)
    mse_log = MSE(tlogs, llogs)

    g_teach = torch.from_numpy(teacher.return_unembs())
    g_kl    = torch.from_numpy(kl_student.return_unembs())
    g_log   = torch.from_numpy(log_student.return_unembs())

    drep_kl = calculate_d_rep(torch.from_numpy(treprs), 
                              torch.from_numpy(klreprs), 
                              g_teach,  
                              g_kl, num_pivots=33, num_combinations=250)

    drep_log = calculate_d_rep(torch.from_numpy(treprs), 
                               torch.from_numpy(lreprs), 
                               g_teach,  
                               g_log, num_pivots=33, num_combinations=250)

    mcca_kl = mCCA(treprs, klreprs, n_components=2)
    mcca_log = mCCA(treprs, lreprs, n_components=2)

    kl_dict = {'acc': acc_klstudent, 'f1': f1_score_klstudent,
               'acc_teacher': acc_teacher, 'f1_teacher': f1_score_teacher,
               'kl': kl_klstudent, 'logitmse': mse_kl.item(),
               'cacc_teacher': t_c_acc, 'cf1_teacher': t_c_f1,
               'cacc': kl_c_acc, 'cf1': kl_c_f1, 'cnll': kl_c_nll,
                'drep': drep_kl.item(),  'mcca': mcca_kl[0]
               }

    l_dict  = {'acc': acc_logstudent, 'f1': f1_score_logstudent, 
                'acc_teacher': acc_teacher, 'f1_teacher': f1_score_teacher,
               'kl': kl_logstudent, 'logitmse': mse_log.item(),
               'cacc_teacher': t_c_acc, 'cf1_teacher': t_c_f1,
                'cacc': l_c_acc, 'cf1': l_c_f1, 'cnll': l_c_nll,
                'drep': drep_log.item(),  'mcca': mcca_log[0]
               }

    return kl_dict, l_dict # [acc_teacher, acc_klstudent, acc_logstudent], [f1_score_teacher, f1_score_klstudent, f1_score_logstudent], [s_t_kl, s_t_log, s_kl_log]

def extract_model_representations(model, loader):
    return 0

if __name__=='__main__':

    args_test = fake_args()

    dataset = SUB(args_test)
    train_loader, val_loader, test_loader = dataset.get_data_loaders()
    encoder_teach = dataset.get_backbone()
    encoder_kl = dataset.get_backbone()
    encoder_log = dataset.get_backbone()

    teacher = CBM(encoder_teach, args_test, n_classes=33)
    kl_student = KLDistill(encoder_kl, args_test, n_classes=33)
    log_student = LogitDistill(encoder_log, args_test, n_classes=33) 

    train_loader, val_loader, test_loader = dataset.get_data_loaders()

    tseed = 0
    tseed = 100
    tseed = 200
    tseed = 300
    tseed = 400

    teacher_location = f'data/runs/sub-standard-N0-cbm-epoch499-seed{tseed}.pt'
    # klstudent_location = 'data/runs/synthetic-kldistill-epoch249-seed1.pt'
    # logstudent_location = 'data/runs/synthetic-logitdistill-epoch249-seed1.pt'
    run_start = int(tseed/10) 
    
    rows_kl, rows_log = [], []
    for i in range(run_start, run_start + 5):
        klstudent_location  = f'data/runs/sub-standard-N0-kldistill-epoch499-seed{i}.pt'
        logstudent_location = f'data/runs/sub-standard-N0-logitdistill-epoch499-seed{i}.pt'

        #accs, f1s, ccas 
        kl_metrics, l_metrics = evaluate_models_sub(teacher_location, 
                                                    klstudent_location, 
                                                    logstudent_location, 
                                                    test_loader,
                                                    save_str=f'sub-standard-N0-seed{i}')
        
        kl_metrics['seed'] = i
        l_metrics['seed']  = i
        
        rows_kl.append(kl_metrics), rows_log.append(l_metrics)

    df_kl  = pd.DataFrame(rows_kl).set_index('seed')
    df_log = pd.DataFrame(rows_log).set_index('seed')


    df_kl.to_csv(f'data/teacher-seed{tseed}-sub_seed_runs_kl_student.csv')
    df_log.to_csv(f'data/teacher-seed{tseed}-sub_seed_runs_log_student.csv')

    print('Saved at location:', \
          f'data/teacher-seed{tseed}-sub_seed_runs_kl_student.csv', \
          f'data/teacher-seed{tseed}-sub_seed_runs_log_student.csv')
    
    quit()

    # teacher_location = 'data/runs/synthetic-classifier-epoch1499-seed5.pt'
    # teacher.load_state_dict(torch.load(teacher_location))
    # teacher.eval()

    # klstudent_location = 'data/runs/synthetic-kldistill-epoch249-seed3.pt'
    # kl_student.load_state_dict(torch.load(klstudent_location))
    # kl_student.eval()

    # logstudent_location = 'data/runs/synthetic-logitdistill-epoch249-seed1.pt'
    # log_student.load_state_dict(torch.load(logstudent_location))
    # log_student.eval()

    # train_loader, val_loader, test_loader = dataset.get_data_loaders()

    # X_train, Y_train, Z_train = [], [], []
    # tlogs, kllogs, llogs = [], [], []
    # treprs, klreprs, lreprs = [], [], []
    # for data in test_loader:
    #     X, Y, Z = data
    #     X_train.append(X.numpy())
    #     Y_train.append(Y.numpy())
    #     Z_train.append(Z.numpy())   

    #     with torch.no_grad():
    #         t_out = teacher(X)
    #         kl_out = kl_student(X)
    #         log_out = log_student(X)

    #         tlogs.append(t_out['LOGITS'].cpu().numpy())
    #         kllogs.append(kl_out['LOGITS'].cpu().numpy())
    #         llogs.append(log_out['LOGITS'].cpu().numpy())
            
    #         treprs.append(t_out['EMBS'].cpu().numpy())
    #         klreprs.append(kl_out['EMBS'].cpu().numpy())
    #         lreprs.append(log_out['EMBS'].cpu().numpy())

    # X_test = np.concatenate(X_train, axis=0)
    # Y_test = np.concatenate(Y_train, axis=0)
    # Z_test = np.concatenate(Z_train, axis=0)

    # tlogs = np.concatenate(tlogs, axis=0)
    # kllogs = np.concatenate(kllogs, axis=0)
    # llogs = np.concatenate(llogs, axis=0)

    # treprs = np.concatenate(treprs, axis=0)
    # klreprs = np.concatenate(klreprs, axis=0)
    # lreprs = np.concatenate(lreprs, axis=0)


    # # from datasets.synthetic import plot_2d_data

    # # plot_2d_data(treprs, np.argmax(tlogs, axis=1), filename='data/debug-teacher_synthetic_plot.png', s=2)
    # # plot_2d_data(klreprs, np.argmax(kllogs, axis=1), filename='data/debug-klstudent_synthetic_plot.png', s=2)
    # # plot_2d_data(lreprs, np.argmax(llogs, axis=1), filename='data/debug-logitstudent_synthetic_plot.png', s=2)

    # acc_teacher    = accuracy_score(Y_test, np.argmax(tlogs, axis=1))
    # acc_klstudent  = accuracy_score(Y_test, np.argmax(kllogs, axis=1))
    # acc_logstudent = accuracy_score(Y_test, np.argmax(llogs, axis=1))

    # f1_score_teacher   = f1_score(Y_test, np.argmax(tlogs, axis=1), average='macro')
    # f1_score_klstudent = f1_score(Y_test, np.argmax(kllogs, axis=1), average='macro')
    # f1_score_logstudent = f1_score(Y_test, np.argmax(llogs, axis=1), average='macro')

    # print('Teacher Accuracy:', acc_teacher, f1_score_teacher)
    # print('KL Student Accuracy:', acc_klstudent, f1_score_klstudent)
    # print('Logit Student Accuracy:', acc_logstudent, f1_score_logstudent)

    # print('\n')

    # cca_t_kl = CCA(n_components=2)
    # cca_t_log = CCA(n_components=2)
    # cca_kl_log = CCA(n_components=2)

    # cca_t_kl.fit(treprs, klreprs)
    # cca_t_kl.score(treprs, klreprs)

    # cca_t_log.fit(treprs, lreprs)
    # cca_t_kl.score(treprs, klreprs)

    # cca_kl_log.fit(klreprs, lreprs)
    # cca_kl_log.score(klreprs, lreprs)

    # print('CCA Teacher-KL Student:', cca_t_kl.score(treprs, klreprs))
    # print('CCA Teacher-Logit Student:', cca_t_log.score(treprs, lreprs))
    # print('CCA KL Student-Logit Student:', cca_kl_log.score(klreprs, lreprs))




    

    




