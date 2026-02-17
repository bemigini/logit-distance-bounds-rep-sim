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

from utils.metrics import KL_from_logits, MSE, mCCA, calculate_d_rep


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

    X_prov, Y_prov, Z_prov = [], [], []
    tlogs, kllogs, llogs = [], [], []
    treprs, klreprs, lreprs = [], [], []
    for data in test_loader:
        X, Y, Z = data
        X_prov.append(X.numpy())
        Y_prov.append(Y.numpy())
        Z_prov.append(Z.numpy())   

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

    X_test = np.concatenate(X_prov, axis=0)
    Y_test = np.concatenate(Y_prov, axis=0)
    Z_test = np.concatenate(Z_prov, axis=0)

    tlogs = np.concatenate(tlogs, axis=0)
    kllogs = np.concatenate(kllogs, axis=0)
    llogs = np.concatenate(llogs, axis=0)

    treprs = np.concatenate(treprs, axis=0)
    klreprs = np.concatenate(klreprs, axis=0)
    lreprs = np.concatenate(lreprs, axis=0)

    if False:

        plot_2d_data(treprs, Y_test,  filename='logs/teacher_synthetic_plot_test.png', s=8, cmap='rainbow')
        plot_2d_data(klreprs, Y_test, filename='logs/klstudent_synthetic_plot_test.png', s=8, cmap='rainbow')
        plot_2d_data(lreprs, Y_test,  filename='logs/logitstudent_synthetic_plot_test.png', s=8, cmap='rainbow')

        print('Saved representation plots.')
    
        quit()

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
                              g_kl, num_pivots=7, num_combinations=15)

    drep_log = calculate_d_rep(torch.from_numpy(treprs), 
                               torch.from_numpy(lreprs), 
                               g_teach,  
                               g_log, num_pivots=7, num_combinations=15)


    mcca_kl = mCCA(treprs, klreprs, n_components=2)
    mcca_log = mCCA(treprs, lreprs, n_components=2)


    kl_dict = {'acc': acc_klstudent, 'f1': f1_score_klstudent, 
               'acc-teach': acc_teacher, 'f1-teach': f1_score_teacher,
               'kl': kl_klstudent, 'logitmse': mse_kl.item(),
               'drep': drep_kl.item(),  'mcca': mcca_kl[0]
               }

    l_dict  = {'acc': acc_logstudent, 'f1': f1_score_logstudent, 
               'acc-teach': acc_teacher, 'f1-teach': f1_score_teacher,
               'kl': kl_logstudent,'logitmse': mse_log.item(),
               'drep': drep_log.item(),  'mcca': mcca_log[0]
               }
    
    # print('KL Student Metrics:', kl_dict)
    # print('Logit Student Metrics:', l_dict)
    # quit()

    return kl_dict, l_dict 

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

    tseed = 5
    tseed = 42
    tseed = 1518
    tseed = 31
    tseed = 1611

    # teacher_location = f'data/runs/synthetic-classifier-epoch1499-seed{tseed}.pt'
    teacher_location = f'data/runs/synthetic-standard-N0-classifier-epoch1499-seed{tseed}.pt'
    # klstudent_location = 'data/runs/synthetic-kldistill-epoch249-seed1.pt'
    # logstudent_location = 'data/runs/synthetic-logitdistill-epoch249-seed1.pt'
    rows_kl, rows_log = [], []
    for i in range(40,45):
        # klstudent_location  = f'data/runs/synthetic-kldistill-epoch249-seed{i}.pt'
        # logstudent_location = f'data/runs/synthetic-logitdistill-epoch249-seed{i}.pt'
        klstudent_location  = f'data/runs/synthetic-standard-N0-kldistill-epoch249-seed{i}.pt'
        logstudent_location = f'data/runs/synthetic-standard-N0-logitdistill-epoch249-seed{i}.pt'

        #accs, f1s, ccas 
        kl_metrics, l_metrics = evaluate_models_synthetic(teacher_location, klstudent_location, logstudent_location, 
                                                           test_loader)
        
        kl_metrics['seed'] = i
        l_metrics['seed']  = i
        
        rows_kl.append(kl_metrics), rows_log.append(l_metrics)

    df_kl  = pd.DataFrame(rows_kl).set_index('seed')
    df_log = pd.DataFrame(rows_log).set_index('seed')

    
    df_kl.to_csv(f'data/syn-teach-{tseed}-seed_runs_kl_student_standard.csv')
    df_log.to_csv(f'data/syn-teach-{tseed}-seed_runs_log_student_standard.csv')

    print('Saved evaluation CSVs for KL and Logit students at:')
    print(f'data/syn-teach-{tseed}-seed_runs_kl_student_standard.csv')
    print(f'data/syn-teach-{tseed}-seed_runs_log_student_standard.csv') 

        # all_acs.append(np.array(accs))
        # all_f1s.append(np.array(f1s))
        # all_ccas.append(np.array(ccas))



    # all_acs = np.stack(all_acs, axis=0)
    # all_f1s = np.stack(all_f1s, axis=0)
    # all_ccas = np.stack(all_ccas, axis=0)

    # print('Average Accuracies over 10 seeds:', all_acs.mean(axis=0), ' +/- ', all_acs.std(axis=0))
    # # print('Average F1 Scores over 5 seeds:', all_f1s.mean(axis=0), ' +/- ', all_f1s.std(axis=0))
    # print('Average CCA Scores over 10 seeds:', all_ccas.mean(axis=0), ' +/- ', all_ccas.std(axis=0))


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




    

    




