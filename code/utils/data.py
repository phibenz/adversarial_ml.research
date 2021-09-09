import os
import numpy as np

import torch
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as tfunc

from config.config import DATA_PATH, IMAGENET_PATH, FNB_DATASET_PATH


class DatasetFromTorchTensor(Dataset):
    def __init__(self, data, target, transform=None):
        # Data type handling must be done beforehand. It is too difficult at this point.
        self.data = data
        self.target = target
        if len(self.target.shape)==1:
            self.target = target.long()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        if self.transform:
            # x = tfunc.to_pil_image(x) # Included in transform
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)

def get_data_specs(dataset):
    mean = [0., 0., 0.]
    std = [1., 1., 1.]

    if dataset in ['mnist']:
        num_classes = 10
        img_size = 28
        num_channels = 1
        mean = [0.]
        std = [1.]
    elif dataset == 'svhn':
        num_classes = 10
        img_size = 32
        num_channels = 3
    elif dataset in ['cifar10', 'grad_imgs_cifar10'] \
            or dataset in ['d_non_robust_cifar', 'd_robust_cifar', 'ddet_cifar', 'drand_cifar'] \
            or dataset.startswith('cifar10c_'):
        num_classes = 10
        img_size = 32
        num_channels = 3
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif dataset in ['cifar100']:
        num_classes = 100
        img_size = 32
        num_channels = 3
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif dataset in ["imagenet", 'grad_imgs_imagenet'] \
            or dataset.starswith('imagenetc_'):
        num_classes = 1000
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img_size = 224
        num_channels = 3
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))

    return num_classes, mean, std, img_size, num_channels


def get_transforms(dataset, augmentation=True):
    _, mean, std, img_size, _ = get_data_specs(dataset)
    if dataset in ['mnist', 'fmnist']:
        if augmentation:
            train_transform = transforms.Compose(
                    [transforms.RandomCrop(img_size, padding=3),
                     transforms.ToTensor(),
                     transforms.Normalize(mean, std)
                     ])
        else:
            train_transform = transforms.Compose(
                    [transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                    ])
                 
        test_transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize(mean, std)
                ])
    
    elif dataset in ['grad_imgs_mnist']:
        if augmentation:
            train_transform = transforms.Compose(
                    [transforms.Lambda(lambda x: tfunc.to_pil_image(x)),
                     transforms.RandomCrop(img_size, padding=3),
                     transforms.ColorJitter(.25),
                     transforms.RandomRotation(2),
                     transforms.ToTensor(),
                     transforms.Normalize(mean, std)
                     ])
        else:
            train_transform = transforms.Compose(
                    [transforms.Lambda(lambda x: tfunc.to_pil_image(x)),
                     transforms.ToTensor(),
                     transforms.Normalize(mean, std)
                    ])
                 
        test_transform = transforms.Compose(
                [transforms.Lambda(lambda x: tfunc.to_pil_image(x)),
                 transforms.ToTensor(),
                 transforms.Normalize(mean, std)
                ])
    
    elif dataset == 'svhn':
        if augmentation:
            train_transform = transforms.Compose(
                    [transforms.RandomCrop(img_size, padding=4),
                     transforms.ToTensor(),
                     transforms.Normalize(mean, std)
                     ])
        else:
            train_transform = transforms.Compose(
                    [transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                    ])

        test_transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize(mean, std)
                ])
    
    elif dataset in ['cifar10', 'cifar100'] or dataset.startswith('cifar10c_'):
        if augmentation:
            train_transform = transforms.Compose(
                    [transforms.RandomHorizontalFlip(),
                     transforms.RandomCrop(img_size, padding=4),
                     transforms.ToTensor(),
                     transforms.Normalize(mean, std)
                     ])
        else:
            train_transform = transforms.Compose(
                    [transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                    ])

        test_transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize(mean, std)
                ])

    elif dataset in ['d_non_robust_cifar', 'd_robust_cifar', 'ddet_cifar', 'drand_cifar']:
        if augmentation:
            train_transform = transforms.Compose(
                    [transforms.Lambda(lambda x: tfunc.to_pil_image(x)),
                     transforms.RandomHorizontalFlip(),
                     transforms.RandomCrop(img_size, padding=4),
                     transforms.ColorJitter(.25,.25,.25),
                     transforms.RandomRotation(2),
                     transforms.ToTensor(),
                     transforms.Normalize(mean, std)
                     ])
        else:
            train_transform = transforms.Compose(
                    [transforms.ToTensor(),
                     transforms.Normalize(mean, std)
                    ])

        test_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize(mean, std)
                ])
    
    elif dataset in ['grad_imgs_cifar10', 'grad_imgs_cifar100']:
        if augmentation:
            train_transform = transforms.Compose(
                    [transforms.Lambda(lambda x: tfunc.to_pil_image(x)),
                     transforms.RandomHorizontalFlip(),
                     transforms.RandomCrop(img_size, padding=4),
                     transforms.ColorJitter(.25,.25,.25),
                     transforms.RandomRotation(2),
                     transforms.ToTensor(),
                     transforms.Normalize(mean, std)
                     ])
        else:
            train_transform = transforms.Compose(
                    [transforms.Normalize(mean, std)
                    ])

        test_transform = transforms.Compose(
                [transforms.Normalize(mean, std)
                ])
    
    elif dataset in ['imagenet']:
        if augmentation:
            train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                    ])
        else:
            train_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                    ])
        
        test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                ])
    
    elif dataset in ['grad_imgs_imagenet']:
        if augmentation:    
            train_transform = transforms.Compose([
                    transforms.Lambda(lambda x: tfunc.to_pil_image(x)),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                    ])
        else:
            train_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                    ])
        
        test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                ])
    
    elif dataset.startswith('imagenetc_'):
        if augmentation:    
            train_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                    ])
        else:
            train_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                    ])
        test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                ])
    
    return train_transform, test_transform
    

def get_data(dataset, train_transform, test_transform, severity=1):
    if dataset == 'mnist':
        train_data = torchvision.datasets.MNIST(DATA_PATH, train=True, transform=train_transform, download=True)
        test_data = torchvision.datasets.MNIST(DATA_PATH, train=False, transform=test_transform, download=True)
    elif dataset == 'fmnist':
        train_data = torchvision.datasets.FashionMNIST(DATA_PATH, train=True, transform=train_transform, download=True)
        test_data = torchvision.datasets.FashionMNIST(DATA_PATH, train=False, transform=test_transform, download=True)
    elif dataset == 'cifar10':
        train_data = torchvision.datasets.CIFAR10(DATA_PATH, train=True, transform=train_transform, download=True)
        test_data = torchvision.datasets.CIFAR10(DATA_PATH, train=False, transform=test_transform, download=True)
    elif dataset == 'cifar100':
        train_data = torchvision.datasets.CIFAR100(DATA_PATH, train=True, transform=train_transform, download=True)
        test_data = torchvision.datasets.CIFAR100(DATA_PATH, train=False, transform=test_transform, download=True)
    elif dataset == 'svhn':
        train_data = torchvision.datasets.SVHN(DATA_PATH, split='train', transform=train_transform, download=True)
        test_data = torchvision.datasets.SVHN(DATA_PATH, split='test', transform=test_transform, download=True)
    elif dataset == 'imagenet':
        # Data loading code
        traindir = os.path.join(IMAGENET_PATH, 'train')
        valdir = os.path.join(IMAGENET_PATH, 'val')
        train_data = torchvision.datasets.ImageFolder(root=traindir, transform=train_transform)
        test_data = torchvision.datasets.ImageFolder(root=valdir, transform=test_transform)
    elif dataset.startswith('imagenetc_'):
        corruption_type = dataset.split('imagenetc_')[-1]
        valdir = os.path.join(DATA_PATH, 'IMAGENET-C', 'brightness', str(severity))
        train_data = torchvision.datasets.ImageFolder(root=valdir, transform=train_transform)
        test_data = torchvision.datasets.ImageFolder(root=valdir, transform=test_transform)
    elif dataset in ['d_non_robust_cifar', 'd_robust_cifar', 'ddet_cifar', 'drand_cifar']:
        fnb_subfolder_name = dataset.split('_cifar')[0] + "_CIFAR"
        train_imgs = torch.cat(torch.load(os.path.join(FNB_DATASET_PATH, fnb_subfolder_name, f'CIFAR_ims')))
        train_labels = torch.cat(torch.load(os.path.join(FNB_DATASET_PATH, fnb_subfolder_name, f'CIFAR_lab')))
        train_data = DatasetFromTorchTensor(train_imgs, train_labels, transform=train_transform)
        test_data = torchvision.datasets.CIFAR10(DATA_PATH, train=False, transform=test_transform, download=True)
    elif dataset.startswith('cifar10c_'):
        corruption_type = dataset.split('cifar10c_')[-1]
        data_path = os.path.join(DATA_PATH, 'CIFAR-10-C', '{}.npy'.format(corruption_type))
        if not os.path.isfile(data_path):
            raise ValueError
        train_imgs = torch.tensor(np.transpose(np.load(data_path), (0,3,1,2)))
        train_labels = torch.tensor(np.load(os.path.join(DATA_PATH, 'CIFAR-10-C', 'labels.npy')))
        train_data = DatasetFromTorchTensor(train_imgs, train_labels, transform=train_transform)
        test_data = DatasetFromTorchTensor(train_imgs, train_labels, transform=test_transform)
    else:
        raise ValueError('Unknwon dataset: {}'.format(dataset))
    return train_data, test_data
