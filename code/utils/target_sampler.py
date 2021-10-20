import torch
import numpy as np 


def one_hot(class_labels, num_classes):
    return torch.zeros(len(class_labels), num_classes).scatter_(1, class_labels.unsqueeze(1), 1.)

class RandomOneHotSampler():
    def __init__(self, num_classes, **kwargs):
        self.num_classes = num_classes
    
    def sample(self, x_ref, y_ref, y_bg):
        random_y = torch.randint_like(y_bg, low=0, high=self.num_classes)
        target_one_hot = one_hot(random_y.cpu(), num_classes=self.num_classes)        
        return target_one_hot

class OffsetOneHotSampler():
    def __init__(self, num_classes, **kwargs):
        self.num_classes = num_classes
        self.offset = kwargs['offset']
    
    def sample(self, x_ref, y_ref, y_bg):
        offset_y = (y_bg + self.offset) % self.num_classes
        target_one_hot = one_hot(offset_y.cpu(), num_classes=self.num_classes)
        return target_one_hot

class GtOneHotSampler():
    def __init__(self, num_classes, **kwargs):
        self.num_classes = num_classes
    
    def sample(self, x_ref, y_ref, y_bg):
        target_one_hot = one_hot(y_bg.cpu(), num_classes=self.num_classes)
        return target_one_hot

class LatentSampler():
    def __init__(self, num_classes, **kwargs):
        self.num_classes = num_classes
        self.model = kwargs['model']
        self.model.eval()
        
    def sample(self, x_ref, y_ref, y_bg):
        with torch.no_grad():
            latent = self.model(x_ref, with_latent=True)
        return latent
