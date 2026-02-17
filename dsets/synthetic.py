
import numpy as np
from torch.utils.data import TensorDataset
import torch 
import os, sys

sys.path.append(os.getcwd())


from backbones.nn import MLP
from dsets.utils.base_dataset import BaseDataset, get_loader

def plot_2d_data(X, Y, filename='data_plot.png', cmap='tab10', s=1):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,6))
    
    # scatter with label classes

    for c in range(len(np.unique(Y))):
        plt.scatter(X[Y.squeeze()==c,0], X[Y.squeeze()==c,1], s=s,
                    c=plt.get_cmap(cmap)(c / len(np.unique(Y))),
                     label=f'Class {c}', alpha=0.7, cmap=cmap)

    plt.axis('equal')
    plt.tight_layout()
    plt.grid(True)
    # plt.xlim(-10,10)
    # plt.ylim(-10,10)
    # legend with increased point size appeareance
    # plt.legend( loc='lower right', ncol=1, markerscale=8, fontsize=16)
    plt.savefig(filename, dpi=300)
    plt.close()


def generate_2d_data(n_spirals, n_labels, N=100):
    l = np.linspace(1, 10, N) 
    #
    X = []
    
    for i in range(n_labels):
        for k in range(n_spirals):
            for j in l:
                s = np.random.uniform(-0.75,0.75)
                # X.append(
                #     m[i].reshape(-1,2)
                #     # np.random.normal(loc=m[i], scale=0.025*(1+0.125*(i//(n_labels*n_spirals))**0.25),
                #     #                 size=(1,2))
                # )
                loc = (j)*np.array(
                        [
                            np.cos(
                                0.2*np.pi/n_labels*j/2 +
                                (i*n_spirals+k)*2*np.pi/(n_labels*n_spirals)
                                # i*np.pi/(n_labels/2)
                                # + (2*k*np.pi/(n_labels*n_spirals))
                                + s*np.pi/(n_labels*n_spirals)
                                    ),
                            np.sin(
                                0.2*np.pi/n_labels*j/2 +
                                (i*n_spirals+k)*2*np.pi/(n_labels*n_spirals)
                                # i*np.pi/(n_labels/2)
                                # + (2*k*np.pi/(n_labels*n_spirals))
                                + s*np.pi/(n_labels*n_spirals)
                                    )
                        ]
                        ).reshape(-1,2)
                X.append(loc)

            

    X = np.concatenate(X)

    # define labels
    label_vec = np.arange(1,n_labels)
    Y = []
    for k in range(n_spirals):
        new_labels = np.random.permutation(label_vec)
        for i in range(n_labels):
            if i == 0:
                lab = i
            else:
                lab = new_labels[i-1]
            for j in range(N):
                    Y.append(
                        np.array([lab % n_labels])
                    )

    Y = np.concatenate(Y)

    Z = np.ones_like(X) * -1  # dummy concepts

    print(len(X), 'samples generated.')

    plot_2d_data(X, Y, filename='logs/synthetic_all_data.png', s=1, cmap='rainbow')

    #remove this for restarting
    quit()


    from sklearn.model_selection import train_test_split
    X_train, X_tmp, Z_train, Z_tmp, Y_train, Y_tmp = train_test_split(
        X, Z, Y, test_size=0.5, random_state=42, stratify=Y
    )
    X_val, X_test, Z_val, Z_test, Y_val, Y_test = train_test_split(
        X_tmp, Z_tmp, Y_tmp, test_size=0.5, random_state=42, stratify=Y_tmp
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    Z_train = torch.tensor(Z_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.long)
    X_val   = torch.tensor(X_val,   dtype=torch.float32)
    Z_val   = torch.tensor(Z_val,   dtype=torch.float32)
    Y_val   = torch.tensor(Y_val,   dtype=torch.long)
    X_test  = torch.tensor(X_test,  dtype=torch.float32)
    Z_test  = torch.tensor(Z_test,  dtype=torch.float32)
    Y_test  = torch.tensor(Y_test,  dtype=torch.long)


    np.savez('data/synthetic_data.npz',
             X_train=X_train.numpy(), Y_train=Y_train.numpy(), Z_train=Z_train.numpy(),
             X_val=X_val.numpy(),     Y_val=Y_val.numpy(),     Z_val=Z_val.numpy(),
             X_test=X_test.numpy(),   Y_test=Y_test.numpy(),   Z_test=Z_test.numpy(),
            )

    plot_2d_data(X_train, Y_train, filename='data/synthetic_plot_train.png', s=2)
    plot_2d_data(X_val, Y_val, filename='data/synthetic_plot_val.png', s=2)
    plot_2d_data(X_test, Y_test, filename='data/synthetic_plot_test.png', s=2)

    data_train =  TensorDataset(X_train, Y_train, Z_train)
    data_val   =  TensorDataset(X_val, Y_val, Z_val)
    data_test  =  TensorDataset(X_test, Y_test, Z_test)
    
    return data_train, data_val, data_test


def filtrate_dataset_by_modality(X, Y, Z, min_radius, max_radius, modality):
    if modality == 'interpolation':
        mask_radius = (X.norm(dim=1) <= min_radius) | (X.norm(dim=1) >= max_radius)
    elif modality == 'extrapolation':
        mask_radius = (X.norm(dim=1) >= min_radius) & (X.norm(dim=1) <= max_radius)
    else:
        mask_radius = torch.ones(X.shape[0], dtype=torch.bool)

    X_in = X[mask_radius]
    Y_in = Y[mask_radius]
    Z_in = Z[mask_radius]

    X_out = X[~mask_radius]
    Y_out = Y[~mask_radius]
    Z_out = Z[~mask_radius]

    return X_in, Y_in, Z_in, X_out, Y_out, Z_out

class SYNTHETIC(BaseDataset):
    NAME = 'synthetic'

    def get_data_loaders(self):
        PATH = 'data/synthetic_data.npz'
        if not self.distilled: 
            if not os.path.exists(PATH):
                dataset_train, dataset_val, dataset_test  = generate_2d_data(
                    n_spirals=4, n_labels=7, N=1000)
            
                # self.train_loader = get_loader(dataset_train, self.args.batch_size, val_test=False)
                # self.val_loader   = get_loader(dataset_val,   self.args.batch_size, val_test=True)
                # self.test_loader  = get_loader(dataset_test,  self.args.batch_size, val_test=True)
                print('Dataset generated and saved to', PATH)
                print('Please restart the script.')
                exit()
            else:
                data = np.load(PATH)
                X_train = torch.tensor(data['X_train'], dtype=torch.float32)
                Y_train = torch.tensor(data['Y_train'], dtype=torch.long)
                Z_train = torch.tensor(data['Z_train'], dtype=torch.float32)
                X_val   = torch.tensor(data['X_val'],   dtype=torch.float32)
                Y_val   = torch.tensor(data['Y_val'],   dtype=torch.long)
                Z_val   = torch.tensor(data['Z_val'],   dtype=torch.float32)
                X_test  = torch.tensor(data['X_test'],  dtype=torch.float32)
                Y_test  = torch.tensor(data['Y_test'],  dtype=torch.long)
                Z_test  = torch.tensor(data['Z_test'],  dtype=torch.float32)

                X_ood, Y_ood, Z_ood = None, None, None


                X_train, Y_train, Z_train, X_ood_r, Y_ood_r, Z_ood_r = filtrate_dataset_by_modality(X_train, Y_train, Z_train,
                                            self.args.min_radius, self.args.max_radius, self.args.modality)    
                X_val,   Y_val,   Z_val,   X_ood_v, Y_ood_v, Z_ood_v     = filtrate_dataset_by_modality(X_val,   Y_val,   Z_val,
                                            self.args.min_radius, self.args.max_radius, self.args.modality)
                X_test,  Y_test,  Z_test,  X_ood_t, Y_ood_t, Z_ood_t     = filtrate_dataset_by_modality(X_test,  Y_test,  Z_test,
                                            self.args.min_radius, self.args.max_radius, self.args.modality)


                dataset_train = TensorDataset(X_train, Y_train, Z_train)
                dataset_val   = TensorDataset(X_val,   Y_val,   Z_val)
                dataset_test  = TensorDataset(X_test,  Y_test,  Z_test)

                self.train_loader = get_loader(dataset_train, self.args.batch_size, val_test=False)
                self.val_loader   = get_loader(dataset_val,   self.args.batch_size, val_test=False)
                self.test_loader  = get_loader(dataset_test,  self.args.batch_size, val_test=True)

                if X_ood is not None:
                    X_ood = torch.cat([X_ood, X_ood_r, X_ood_v, X_ood_t], dim=0)
                    Y_ood = torch.cat([Y_ood, Y_ood_r, Y_ood_v, Y_ood_t], dim=0)
                    Z_ood = torch.cat([Z_ood, Z_ood_r, Z_ood_v, Z_ood_t], dim=0)
                else:
                    X_ood = torch.cat([X_ood_r, X_ood_v, X_ood_t], dim=0)
                    Y_ood = torch.cat([Y_ood_r, Y_ood_v, Y_ood_t], dim=0)
                    Z_ood = torch.cat([Z_ood_r, Z_ood_v, Z_ood_t], dim=0)

                if X_ood.shape[0] > 0:
                    dataset_ood = TensorDataset(X_ood, Y_ood, Z_ood)
                    self.ood_loader = get_loader(dataset_ood, batch_size=self.args.batch_size, val_test=True)
                else:
                    self.ood_loader = None
                

                # if self.args.modality == 'interpolation':
                #     train_mask_radius = (X_train.norm(dim=1) <= self.args.min_radius) | (X_train.norm(dim=1) >= self.args.max_radius)
                #     val_mask_radius   = (X_val.norm(dim=1)   <= self.args.min_radius) | (X_val.norm(dim=1)   >= self.args.max_radius)
                #     test_mask_radius  = (X_test.norm(dim=1)  <= self.args.min_radius) | (X_test.norm(dim=1)  >= self.args.max_radius)
                # elif self.args.modality == 'extrapolation':
                #     train_mask_radius = (X_train.norm(dim=1) >= self.args.min_radius) & (X_train.norm(dim=1) <= self.args.max_radius)
                #     val_mask_radius   = (X_val.norm(dim=1)   >= self.args.min_radius) & (X_val.norm(dim=1)   <= self.args.max_radius)
                #     test_mask_radius  = (X_test.norm(dim=1)  >= self.args.min_radius) & (X_test.norm(dim=1)  <= self.args.max_radius)
                # else:
                #     train_mask_radius = torch.ones(X_train.shape[0], dtype=torch.bool)
                #     val_mask_radius   = torch.ones(X_val.shape[0], dtype=torch.bool)
                #     test_mask_radius  = torch.ones(X_test.shape[0], dtype=torch.bool)

                # if train_mask_radius.sum() == 0 or val_mask_radius.sum() == 0 or test_mask_radius.sum() == 0:

                #     dataset_train = TensorDataset(X_train, Y_train, Z_train)
                #     dataset_val   = TensorDataset(X_val,   Y_val,   Z_val)
                #     dataset_test  = TensorDataset(X_test,  Y_test,  Z_test)
                #     dataset_ood = None
                #     self.ood_loader = None

                # else:
                #     dataset_train = TensorDataset(X_train[train_mask_radius], Y_train[train_mask_radius], Z_train[train_mask_radius])
                #     dataset_val   = TensorDataset(X_val[val_mask_radius],     Y_val[val_mask_radius],     Z_val[val_mask_radius])
                #     dataset_test  = TensorDataset(X_test[test_mask_radius],   Y_test[test_mask_radius],   Z_test[test_mask_radius])
                #     dataset_ood = TensorDataset(
                #         torch.cat([X_train[~train_mask_radius], X_val[~val_mask_radius], X_test[~test_mask_radius]], dim=0),
                #         torch.cat([Y_train[~train_mask_radius], Y_val[~val_mask_radius], Y_test[~test_mask_radius]], dim=0),
                #         torch.cat([Z_train[~train_mask_radius], Z_val[~val_mask_radius], Z_test[~test_mask_radius]], dim=0),
                #     )
                #     self.ood_loader = get_loader(dataset_ood, batch_size=self.args.batch_size, val_test=True)

        return self.train_loader, self.val_loader, self.test_loader

    def get_backbone(self):
        return MLP(input_dim=2)


    def override_loaders(self, train_loader, val_loader, test_loader):
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader

        self.distilled = True
    
    def get_n_classes(self):
        return 7

if __name__ == '__main__':
    generate_2d_data(n_spirals=4, n_labels=7, N=1500)
