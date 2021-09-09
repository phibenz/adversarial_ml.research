import torch
import numpy as np
from utils.network import Normalize, UnNormalize

class PGDL2():
    def __init__(self, model, attack_criterion, mean, std,
                step_size=0.1, epsilon=1., iterations=100, random_start=False, 
                latent_loss=False, img_min=0, img_max=1):
        """
        PGD L2 attack
        
        Resources for reference:
            https://github.com/MadryLab/robustness/blob/master/robustness/attacker.py
            https://github.com/MadryLab/robustness/blob/master/robustness/attack_steps.py
        """
        self.model = model
        # Model should be in evaluation mode
        self.model.eval()

        self.attack_criterion = attack_criterion
        
        self.norm = Normalize(mean, std)
        self.unnorm = UnNormalize(mean, std)
        
        self.step_size = step_size 
        self.epsilon = epsilon
        self.iterations = iterations
        self.random_start = random_start
        self.latent_loss = latent_loss
        self.img_min = img_min 
        self.img_max = img_max
        
    def run(self, orig_img, lbl):
        # Unnormalize
        img = self.unnorm(orig_img)
        
        x = img.clone().detach().requires_grad_(True)
        
        if self.random_start:
            x = x + (torch.rand_like(x) - 0.5).renorm(p=2, dim=0, maxnorm=self.epsilon)
            x = torch.clamp(x, self.img_min, self.img_max)
        
        for i in range(self.iterations):
            x = x.clone().detach().requires_grad_(True)
            
            # Before feeding into the model: normalize
            x_n = self.norm(x)
            
            if self.latent_loss:
                x_out = self.model(x_n, with_latent=True, fake_relu=True)
            else:
                x_out = self.model(x_n)
            loss = self.attack_criterion(x_out, lbl)

            with torch.no_grad():
                grad, = torch.autograd.grad(loss, [x])
                
                # Take one step
                l = len(x.shape) - 1
                grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, *([1]*l))
                scaled_grad = grad / (grad_norm + 1e-10)
                x = x - scaled_grad * self.step_size

                # Project
                pert = x - img
                pert = pert.renorm(p=2, dim=0, maxnorm=self.epsilon)
                x = torch.clamp(img + pert, self.img_min, self.img_max)
        
        # Return in normalized form 
        return self.norm(x)


class PGDLinf():
    def __init__(self, model, attack_criterion, 
                mean, std,
                step_size=0.1, epsilon=1., iterations=100, random_start=False, 
                latent_loss=False, img_min=0, img_max=1):
        """
        PGD L infinity attack
        
        Resources for reference:
            https://github.com/MadryLab/robustness/blob/master/robustness/attacker.py
            https://github.com/MadryLab/robustness/blob/master/robustness/attack_steps.py
        """
        self.model = model
        # Model should be in evaluation mode
        self.model = self.model.eval()

        self.attack_criterion = attack_criterion

        self.norm = Normalize(mean, std)
        self.unnorm = UnNormalize(mean, std)
        
        self.step_size = step_size 
        self.epsilon = epsilon
        self.iterations = iterations
        self.random_start = random_start
        self.latent_loss = latent_loss
        self.img_min = img_min 
        self.img_max = img_max
        
    def run(self, orig_img, lbl):
        # Unnormalize
        img = self.unnorm(orig_img)
        
        x = img.clone().detach().requires_grad_(True)
        
        if self.random_start:
            x = x + 2 * (torch.rand_like(x) - 0.5) * self.epsilon
            x = torch.clamp(x, self.img_min, self.img_max)
        
        for i in range(self.iterations):
            x = x.clone().detach().requires_grad_(True)
            
            # Before feeding into the model: normalize
            x_n = self.norm(x)
            
            if self.latent_loss:
                x_out = self.model(x_n, with_latent=True, fake_relu=True)
            else:
                x_out = self.model(x_n)
            loss = self.attack_criterion(x_out, lbl)
            
            with torch.no_grad():
                grad, = torch.autograd.grad(loss, [x])
                
                # Take one step
                step = torch.sign(grad) * self.step_size
                x = x - step

                # Project
                pert = x - img
                pert = torch.clamp(pert, -self.epsilon, self.epsilon)
                x = torch.clamp(img + pert, self.img_min, self.img_max)
        
        # Return in normalized form 
        return self.norm(x)
        
