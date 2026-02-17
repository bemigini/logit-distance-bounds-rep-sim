import datetime

import torch
import numpy as np

from torch.utils.data import TensorDataset
from dsets.utils.base_dataset import BaseDataset
from dsets.utils.base_dataset import get_loader
from dsets.synthetic import plot_2d_data


def distill_knowledge_to_dataset(model, dataset: BaseDataset, args):

    # Default Setting for Training
    model.to(model.device)
    train_loader, val_loader, test_loader = dataset.get_data_loaders()
    
    now = datetime.datetime.now()
    print(f'\n--- Start of Distillation ---\n{now.strftime("%Y-%m-%d %H:%M:%S")}\n')

    for i, data in enumerate(train_loader):
        images, labels, _ = data
        images, labels = images.to(model.device), labels.to(model.device)

        out_dict = model(images)
        # out_dict.update({'INPUTS': images, 'LABELS': labels, 'CONCEPTS': concepts})

        if  i == 0:
            x_true = images.detach().cpu().numpy()
            y_true = labels.detach().cpu().numpy()
            l_pred = out_dict['LOGITS'].detach().cpu().numpy()
        elif  i > 0: 
            x_true = np.concatenate([x_true, images.detach().cpu().numpy()], axis=0)           
            y_true = np.concatenate([y_true, labels.detach().cpu().numpy()], axis=0)
            l_pred = np.concatenate([l_pred, out_dict['LOGITS'].detach().cpu().numpy()], axis=0)

    x_train = torch.tensor(x_true.copy())
    y_train = torch.tensor(y_true.copy())
    l_train = torch.tensor(l_pred.copy())

    for i, data in enumerate(val_loader):
        images, labels, _ = data
        images, labels = images.to(model.device), labels.to(model.device)

        out_dict = model(images)
        # out_dict.update({'INPUTS': images, 'LABELS': labels, 'CONCEPTS': concepts})
        
        if  i == 0:
            x_true = images.detach().cpu().numpy()
            y_true = labels.detach().cpu().numpy()
            l_pred = out_dict['LOGITS'].detach().cpu().numpy()
        elif  i > 0: 
            x_true = np.concatenate([x_true, images.detach().cpu().numpy()], axis=0)           
            y_true = np.concatenate([y_true, labels.detach().cpu().numpy()], axis=0)
            l_pred = np.concatenate([l_pred, out_dict['LOGITS'].detach().cpu().numpy()], axis=0)

    x_val = torch.tensor(x_true.copy())
    y_val = torch.tensor(y_true.copy())
    l_val = torch.tensor(l_pred.copy())

    for i, data in enumerate(test_loader):
        images, labels, _ = data
        images, labels = images.to(model.device), labels.to(model.device)

        out_dict = model(images)
        # out_dict.update({'INPUTS': images, 'LABELS': labels, 'CONCEPTS': concepts})
        
        if  i == 0:
            x_true = images.detach().cpu().numpy()
            y_true = labels.detach().cpu().numpy()
            l_pred = out_dict['LOGITS'].detach().cpu().numpy()
        elif  i > 0: 
            x_true = np.concatenate([x_true, images.detach().cpu().numpy()], axis=0)           
            y_true = np.concatenate([y_true, labels.detach().cpu().numpy()], axis=0)
            l_pred = np.concatenate([l_pred, out_dict['LOGITS'].detach().cpu().numpy()], axis=0)
    
    x_test = torch.tensor(x_true.copy())
    y_test = torch.tensor(y_true.copy())
    l_test = torch.tensor(l_pred.copy())

    if dataset.args.dataset == 'synthetic':
        plot_2d_data(x_train.numpy(), l_train.argmax(dim=-1).numpy(), filename='data/distilled_synthetic_plot_train.png', s=2)
        plot_2d_data(x_val.numpy(),   l_val.argmax(dim=-1).numpy(),   filename='data/distilled_synthetic_plot_val.png', s=2)
        plot_2d_data(x_test.numpy(),  l_test.argmax(dim=-1).numpy(),  filename='data/distilled_synthetic_plot_test.png', s=2)


    train_set =  TensorDataset(x_train, y_train, l_train)
    val_set   =  TensorDataset(x_val,   y_val,   l_val)
    test_set  =  TensorDataset(x_test,  y_test,  l_test)

    train_loader = get_loader(train_set, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader   = get_loader(val_set,   batch_size=args.batch_size, num_workers=args.num_workers, val_test=False)
    test_loader  = get_loader(test_set,  batch_size=args.batch_size, num_workers=args.num_workers, val_test=True)

    if hasattr(dataset, 'override_loaders'):
        dataset.override_loaders(train_loader, val_loader, test_loader)
        
        print('--- Dataset Loaders Overridden ---')
        
        return dataset
    else:
        return NotImplementedError('Override Loaders method not implemented in dataset class.')


        
        