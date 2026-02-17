"""

Getting the CIFAR-100 dataset


"""


from argparse import Namespace
from pathlib import Path
from typing import Any, Callable, Optional, Union
#from pprint import pprint


from PIL import Image

import torchvision
from torchvision import transforms
import numpy as np
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler
import torch

from dsets.utils.base_dataset import BaseDataset
from backbones import resnet


class background_CIFAR100(torch.utils.data.Dataset):
    """
    The background CIFAR100 dataset to use to overwrite getitem to include dummy teacher logits
    """

    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False        
    ) -> None:
        self.cifar100_data = torchvision.datasets.CIFAR100(
        root=root, train=train,
        transform=transform, target_transform=target_transform,
        download=download)

        print(type(self.cifar100_data))
        print(self.cifar100_data.__dict__.keys())

        # The below is to make the code work on an older torch version
        
        torch_version_numbers = [int(c) for c in torch.__version__.split('+')[0].split('.')]
        print(torch_version_numbers)

        if torch_version_numbers[0] == 2 and torch_version_numbers[1] <= 5:
            if train:
                # this is to fix behaviour on old torch version
                # pylint: disable=no-member
                self.data = self.cifar100_data.train_data
                self.targets = self.cifar100_data.train_labels
            else:
                # this is to fix behaviour on old torch version
                # pylint: disable=no-member
                self.data = self.cifar100_data.test_data
                self.targets = self.cifar100_data.test_labels
        else:
            self.data = self.cifar100_data.data
            self.targets = self.cifar100_data.targets

        self.transform = self.cifar100_data.transform
        self.target_transform = self.cifar100_data.target_transform


    def __len__(self):
        return self.cifar100_data.__len__()
    

    def __getitem__(self, index: int) -> tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        dummy_logits = torch.ones_like(torch.tensor(target)) * -1

        return img, target, dummy_logits



class CIFAR100_dataset(BaseDataset):
    NAME = 'cifar100'

    def __init__(self, args: Namespace) -> None:
        """
        Initializes chosen backbone and final dimension.
        :param args: the arguments which contain the hyperparameters
        """
        super().__init__(args)
        self.chosen_backbone = args.backbone
        self.final_dim = args.final_dim
        self.ood_loader = None


    def get_data_loaders(self):
        data_dir = self.args.data_dir
        if not self.distilled:
            use_transform = transforms.Compose([
                transforms.ToTensor()
            ])

            train_val_data = background_CIFAR100(
            root=data_dir, train=True,
            download=True, transform=use_transform)

            if self.args.validate:
                validation_size=0.1
                num_train = len(train_val_data)
                indices = list(range(num_train))
                split = int(np.floor(validation_size * num_train))

                np.random.seed(self.args.seed)
                np.random.shuffle(indices)

                train_idx, valid_idx = indices[split:], indices[:split]
                train_sampler = SubsetRandomSampler(train_idx)
                valid_sampler = SubsetRandomSampler(valid_idx)

                self.train_loader = torch.utils.data.DataLoader(
                train_val_data, batch_size=self.args.batch_size, sampler=train_sampler,
                num_workers=self.args.num_workers
                )
                self.val_loader = torch.utils.data.DataLoader(
                    train_val_data, batch_size=self.args.batch_size, sampler=valid_sampler,
                    num_workers=self.args.num_workers
                    )
            else:
                # validation same as train
                self.train_loader = torch.utils.data.DataLoader(
                    train_val_data, batch_size=self.args.batch_size, #sampler=train_sampler,
                    num_workers=self.args.num_workers
                    )
                self.val_loader = torch.utils.data.DataLoader(
                    train_val_data, batch_size=self.args.batch_size, #sampler=valid_sampler,
                    num_workers=self.args.num_workers
                    )
            
            test_data = background_CIFAR100(
                root=data_dir, train=False,
                download=True, transform=use_transform
            )

            self.test_loader = torch.utils.data.DataLoader(
                test_data, batch_size=self.args.batch_size, shuffle=False,
                num_workers=self.args.num_workers
            )

        return self.train_loader, self.val_loader, self.test_loader


    def get_backbone(self):
        """
        Returns the resnet backbone to use for this data
        """
        match self.chosen_backbone:
            case 'resnet18':
                backbone = resnet.ResNet18(final_dim=self.final_dim)
            case 'resnet34':
                backbone = resnet.ResNet34(final_dim=self.final_dim)
            case 'resnet50':
                backbone = resnet.ResNet50(final_dim=self.final_dim)
            case 'pretrain_resnet50v1':
                backbone = resnet.pretrain_ResNet50_V1(final_dim=self.final_dim)
            case 'pretrain_resnet50v2':
                backbone = resnet.pretrain_ResNet50_V2(final_dim=self.final_dim)
            case _:
                raise ValueError(f'Chosen backbone not recognized: {self.chosen_backbone}')
        return backbone


    def override_loaders(self, train_loader, val_loader, test_loader):
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader

        self.distilled = True

    def get_n_classes(self):
        return 100
