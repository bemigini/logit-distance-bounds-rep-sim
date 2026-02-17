
import os, sys 

from torch.utils.data import TensorDataset
import torch 

from sklearn.model_selection import train_test_split

sys.path.append(os.getcwd())

from dsets.utils.base_dataset import BaseDataset, get_loader
from backbones.nn import Shallow

class SUB(BaseDataset):
    NAME = 'sub'

    def get_data_loaders(self):
        if not self.distilled: 
            PATH = 'data/SUB_dinov2_embeddings.pt'

            data = torch.load(PATH)
            embeddings = data['embeddings']
            attr_labels = data['attributes']
            bird_labels = data['labels']


            print(f'Loaded SUB dataset from {PATH}')

            X_temp, X_test, y_attr_temp, y_attr_test, y_bird_temp, y_bird_test = train_test_split(
                embeddings, attr_labels, bird_labels, test_size=0.3, random_state=42, stratify=bird_labels
            )
            X_train, X_val, y_attr_train, y_attr_val, y_bird_train, y_bird_val = train_test_split(
                X_temp, y_attr_temp, y_bird_temp, test_size=0.2, random_state=42, stratify=y_bird_temp            
            )

            data_train = TensorDataset(X_train, y_bird_train, y_attr_train)
            data_val   = TensorDataset(X_val,   y_bird_val,   y_attr_val)
            data_test  = TensorDataset(X_test,  y_bird_test,  y_attr_test)

            self.train_loader = get_loader(data_train, self.args.batch_size, num_workers=4, val_test=False)
            self.val_loader   = get_loader(data_val,   self.args.batch_size, num_workers=4, val_test=True)
            self.test_loader  = get_loader(data_test,  self.args.batch_size, num_workers=4, val_test=True)

            self.ood_loader = None

        return self.train_loader, self.val_loader, self.test_loader


    def get_backbone(self):
        return Shallow(input_dim=768, hidden_units=2048, repr_dim=10) 
    
    def override_loaders(self, train_loader, val_loader, test_loader):
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader

        self.distilled = True
    
    def get_n_classes(self):
        return 33

if __name__ == '__main__':
    
    from datasets import load_dataset

    dataset = load_dataset("Jessica-bader/SUB")
    test_data = dataset["test"]
    features = dataset["test"].features
    print(dataset)
    print(features)

    for i, sample in enumerate(range(33)):
        print(i, '->', dataset["test"].features["attr_label"].int2str(i)
        ) 
