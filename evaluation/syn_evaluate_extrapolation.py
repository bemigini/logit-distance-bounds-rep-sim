import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import sys, os 

conf_path = os.getcwd() 
sys.path.append(conf_path)

print(conf_path)

from dsets.synthetic import SYNTHETIC
from models.logitdistill import LogitDistill
from models.kldistill import KLDistill
from models.classifier import Classifier
from dsets.synthetic import plot_2d_data

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

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
        self.min_radius = 3
        self.max_radius = 7
        self.modality = 'extrapolation'

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.cross_decomposition import CCA


def evaluate_models_synthetic(teacher_location, klstudent_location, logstudent_location, loader):
    teacher.load_state_dict(torch.load(teacher_location))
    teacher.eval()
    
    kl_student.load_state_dict(torch.load(klstudent_location))
    kl_student.eval()

    log_student.load_state_dict(torch.load(logstudent_location))
    log_student.eval()

    X_train, Y_train, Z_train = [], [], []
    tlogs, kllogs, llogs = [], [], []
    treprs, klreprs, lreprs = [], [], []
    for data in loader:
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

    plot_2d_data(X_test, np.argmax(tlogs, axis=1), filename='data/extrapolation/debug-extrapolation-input-teacher_logits_synthetic_plot.png', s=2)
    plot_2d_data(X_test, np.argmax(kllogs, axis=1), filename='data/extrapolation/debug-extrapolation-input-klstudent_logits_synthetic_plot.png', s=2)
    plot_2d_data(X_test, np.argmax(llogs, axis=1), filename='data/extrapolation/debug-extrapolation-input-logitstudent_logits_synthetic_plot.png', s=2)

    treprs = np.concatenate(treprs, axis=0)
    klreprs = np.concatenate(klreprs, axis=0)
    lreprs = np.concatenate(lreprs, axis=0)


    plot_2d_data(treprs, np.argmax(tlogs, axis=1), filename='data/extrapolation/debug-extrapolation-teacher_synthetic_plot.png', s=2)
    plot_2d_data(klreprs, np.argmax(kllogs, axis=1), filename='data/extrapolation/debug-extrapolation-klstudent_synthetic_plot.png', s=2)   
    plot_2d_data(lreprs, np.argmax(llogs, axis=1), filename='data/extrapolation/debug-extrapolation-logitstudent_synthetic_plot.png', s=2)

    acc_teacher    = accuracy_score(Y_test, np.argmax(tlogs, axis=1))
    acc_klstudent  = accuracy_score(Y_test, np.argmax(kllogs, axis=1))
    acc_logstudent = accuracy_score(Y_test, np.argmax(llogs, axis=1))

    f1_score_teacher   = f1_score(Y_test, np.argmax(tlogs, axis=1), average='macro')
    f1_score_klstudent = f1_score(Y_test, np.argmax(kllogs, axis=1), average='macro')
    f1_score_logstudent = f1_score(Y_test, np.argmax(llogs, axis=1), average='macro')

    print('Teacher Accuracy:', acc_teacher, f1_score_teacher)
    print('KL Student Accuracy:', acc_klstudent, f1_score_klstudent)
    print('Logit Student Accuracy:', acc_logstudent, f1_score_logstudent)

    n_labels = tlogs.shape[1]

    Tmap, KLmap, Lmap = np.zeros((n_labels, n_labels)), np.zeros((n_labels, n_labels)), np.zeros((n_labels, n_labels))
    T = 10
    for i in np.unique(Y_test):
        idxs = np.where(Y_test==i)[0]
        tlogs_i = softmax(tlogs[idxs]/T)  
        kllogs_i = softmax(kllogs[idxs]/T)
        llogs_i = softmax(llogs[idxs]/T)

        Tmap[i] = tlogs_i.mean(axis=0) #/ np.sum(tlogs_i.mean(axis=0))
        KLmap[i] = kllogs_i.mean(axis=0) #/ np.sum(kllogs_i.mean(axis=0))
        Lmap[i] = llogs_i.mean(axis=0) #/ np.sum(llogs_i.mean(axis=0))

    cf_t = Tmap
    cf_kl = KLmap
    cf_log = Lmap

    print('Confusion Matrices:')
    print('Teacher:\n', cf_t.shape)
    print('KL Student:\n', cf_kl.shape)
    print('Logit Student:\n', cf_log.shape)

    # cf_t = confusion_matrix(Y_test, np.argmax(tlogs, axis=1))
    # cf_kl = confusion_matrix(Y_test, np.argmax(kllogs, axis=1))
    # cf_log = confusion_matrix(Y_test, np.argmax(llogs, axis=1))

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(cf_t, cmap='viridis')
    plt.title('Teacher Confusion Matrix')
    plt.subplot(1,3,2)
    plt.imshow(cf_kl, cmap='viridis')
    plt.title('KL Student Confusion Matrix')
    plt.subplot(1,3,3)
    plt.imshow(cf_log, cmap='viridis')
    plt.title('Logit Student Confusion Matrix')

    plt.colorbar()
    plt.savefig('data/extrapolation/debug-extrapolation-confusion_matrices.png')


    print('\n')

    from utils.metrics import multiclass_calibration, evaluate_ranking_metrics, neg_log_likelihood


    t_probs  = torch.softmax(torch.from_numpy(tlogs),  dim=1).detach().numpy()
    kl_probs = torch.softmax(torch.from_numpy(kllogs), dim=1).detach().numpy()
    l_probs  = torch.softmax(torch.from_numpy(llogs),  dim=1).detach().numpy()

    nll_teacher = neg_log_likelihood(Y_test, t_probs)['nll']
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
    ood_loader = dataset.ood_loader

    teacher_location = 'data/runs/synthetic-classifier-epoch1499-seed5.pt'
    # klstudent_location = 'data/runs/synthetic-kldistill-epoch249-seed1.pt'
    # logstudent_location = 'data/runs/synthetic-logitdistill-epoch249-seed1.pt'

    # klstudent_location  = f'data/runs/synthetic-kldistill-epoch249-seed10.pt'
    # logstudent_location = f'data/runs/synthetic-logitdistill-epoch249-seed10.pt'

    # klstudent_location  = f'data/runs/synthetic-extrapolation-kldistill-epoch249-seed1.pt'
    # logstudent_location = f'data/runs/synthetic-extrapolation-logitdistill-epoch249-seed1.pt'

    # klstudent_location  = f'data/runs/synthetic-standard-y0246-kldistill-epoch999-seed1.pt'
    # logstudent_location = f'data/runs/synthetic-standard-y0246-logitdistill-epoch999-seed1.pt'

    # klstudent_location  = f'data/runs/synthetic-standard-y012345-kldistill-epoch999-seed1.pt'
    # logstudent_location = f'data/runs/synthetic-standard-y012345-logitdistill-epoch999-seed1.pt'

    


    # kl_dict, log_dict = evaluate_models_synthetic(teacher_location, klstudent_location, logstudent_location, 
    #                                             loader=ood_loader)

    rows_kl, rows_log = [], []
    for i in range(1,4):
        klstudent_location  = f'data/runs/synthetic-extrapolation-N0-kldistill-epoch249-seed{i}.pt'
        logstudent_location = f'data/runs/synthetic-extrapolation-N0-logitdistill-epoch499-seed{i}.pt'

        #accs, f1s, ccas 
        kl_metrics, l_metrics = evaluate_models_synthetic(teacher_location, klstudent_location, logstudent_location, 
                                                          ood_loader)
        
        kl_metrics['seed'] = i
        l_metrics['seed']  = i
        
        rows_kl.append(kl_metrics), rows_log.append(l_metrics)

    import pandas as pd
    df_kl  = pd.DataFrame(rows_kl).set_index('seed')
    df_log = pd.DataFrame(rows_log).set_index('seed')

    
    # df_kl.to_csv('data/extra_runs_kl_student_standard.csv')
    # df_log.to_csv('data/extra_runs_log_student_standard.csv')

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




    

    




