import numpy as np
import torch
import torch.nn as nn
import torchvision.models as torchmodels

from network.cifar.resnet_fnb import ResNet18, ResNet50


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]

class UnNormalize(nn.Module):
    def __init__(self, mean, std):
        super(UnNormalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def forward(self, x):
        return (x * self.std.type_as(x)[None,:,None,None]) + self.mean.type_as(x)[None,:,None,None]

def get_network(arch, num_classes):
    #### MNIST models ####
    if arch == "test":
        pass
    #### CIFAR-10 & CIFAR-100 models ####
    elif arch == 'resnet18_bn_cifar':
        net = ResNet18(normalization="bn", num_classes=num_classes)
    elif arch == 'resnet50_bn_cifar':
        net = ResNet50(normalization="bn", num_classes=num_classes)
    elif arch == 'resnet18_id_cifar':
        net = ResNet18(normalization="id", num_classes=num_classes)
    elif arch == 'resnet50_id_cifar':
        net = ResNet50(normalization="id", num_classes=num_classes)
    ### ImageNet Models ###
    elif arch in torchmodels.__dict__:
        net = torchmodels.__dict__[arch](pretrained=True)
    else:
        raise ValueError("Network {} not supported".format(arch))
    return net

def get_num_parameters(model):
    return sum(p.numel() for p in model.parameters())

def get_num_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad==True, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])

def get_num_non_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad==False, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])

def get_input_grad(model, img, lbl, eps, norm, unnorm, delta_init='none', backprop=False, cuda=False):
    criterion = torch.nn.CrossEntropyLoss()

    if delta_init == 'none':
        delta = torch.zeros_like(img)
    elif delta_init == 'gaussian':
        delta = torch.randn_like(img) * eps
    elif delta_init == 'uniform':
        delta = (torch.rand_like(img) -0.5) * 2 * eps
    else:
        raise ValueError('Invalid delta init')
    
    if cuda:
        delta = delta.cuda()

    x = unnorm(img) # ((img * std_tensor) + mean_tensor)
    x = x + delta
    x = x.clone().detach().requires_grad_(True)
    x_n = norm(x) # (x - mean_tensor)/std_tensor

    out = model(x_n)
    loss = criterion(out, lbl)
    grad = torch.autograd.grad(loss, [x], create_graph=True if backprop else False)[0]
    if not backprop:
        grad, delta = grad.detach(), delta.detach()
    return out, grad