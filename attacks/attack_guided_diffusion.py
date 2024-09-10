import torch
import numpy
from tools.utils import *
from tools.load_dm import get_imagenet_dm_conf, GUIDED_DIFFUSION_MODEL_NAME
from tools.dataset import get_dataset
from tqdm import tqdm
import argparse

class GuidedDiffusion_Attacker():
    def __init__(self, diffusion, model):
        self.diffusion = diffusion
        self.model = model
        # self.model.eval()
        
    
    @torch.no_grad()
    def sdedit(self, x, t, to_01=True):
        """
        x: input image, B C H W
        t: editing strength from 0  to 1
        to_01: the original output of diffusion model is [-1, 1], set to True will transfer to [0, 1]
        """
        
        assert t < 1 and t > 0
        t = int(t * len(self.diffusion.use_timesteps))
        sample_indices = list(range(t+1))[::-1]
        
        B, _, _, _ = x.shape
        
        # make sure the input image is scaled to [-1, 1]
        if x.min() >= 0:
            x = x * 2 - 1
        
        t = torch.full((B, ), t).long().to(x.device)
        x_t = self.diffusion.q_sample(x, t) 
        sample = x_t
        
        print('Run Diffusion Sampling ...')
        for i in tqdm(sample_indices):
            t_tensor = torch.full((B, ), i).long().to(x.device)
            out = self.diffusion.p_sample(self.model, sample, t_tensor)
            sample = out["sample"]

        # the output of diffusion model is [-1, 1], should be transformed to [0, 1]
        if to_01:
            sample = (sample + 1) / 2
        
        return sample


    def gen_gpd_conf(self, eps=16, steps=100, step_size=1, clip_min=-1, clip_max=1):
        self.eps = eps/255 * (clip_max - clip_min)
        self.step_size = step_size / 255 * (clip_max - clip_min)
        self.steps = steps
        self.clip_min = clip_min
        self.clip_max = clip_max
    
    
    def gen_edit_results(self, x, strength_list = [0.5, 0.7]):
        return torch.cat([x]+[self.sdedit(x, strength) for strength in strength_list], -1)
    
    
    def attack_advdm(self, X, random_start=False, grad_mode='+'):
        
        # attack pdm by maximize / minimize the diffusion training loss: 
        #    max_{\delta} E_{noise}(\eps(x + \delta + noise) - noise)
        
        edit_function = self.sdedit
        
        x_clean = X
        
        # attack 
        if random_start:
            print('using random start')
            X_adv = X.clone().detach() + (torch.rand(*X.shape) * 2 * self.eps - self.eps).cuda() # add random noise to start
        else:
            X_adv = X.clone().detach() * 2 - 1
            X_raw = X * 2 - 1


        pbar = tqdm(range(self.steps))
        
        with torch.enable_grad():
            for i in pbar:
                X_adv = X_adv.clone().detach()
                X_adv.requires_grad = True
                # actual_step_size = self.step_size - (self.step_size - self.step_size / 100) / self.steps * i 
                actual_step_size = self.step_size
                t = torch.randint(0, len(self.diffusion.use_timesteps), (X.shape[0],), device=X.device).long()
                
                # noise, x_t = self.diffusion.q_sample(X_adv, t, return_noise=True, noise=NOISE) 
                # eps_pred = self.diffusion.p_mean_variance(self.model, x_t, t)['model_output']
                
                # crit = torch.nn.MSELoss()
                # loss = crit(eps_pred, noise) if not use_ita else crit(eps_pred, fixed_noise)

                loss = self.diffusion.training_losses(self.model, X_adv, t, noise=None)["mse"]
                
                loss.backward()
                
                g = X_adv.grad.detach()
                
                if grad_mode == '+':
                    # using gradient ascent 
                    X_adv = X_adv + g.sign() * actual_step_size
                elif grad_mode == '-':
                    # using gradient descent
                    X_adv = X_adv - g.sign() * actual_step_size
                else:
                    raise KeyboardInterrupt('Mode not defined')       
                
                X_adv = torch.minimum(torch.maximum(X_adv, X_raw - self.eps), X_raw + self.eps)   
                X_adv.data = torch.clamp(X_adv, min=self.clip_min, max=self.clip_max)
                X_adv.grad=None
        
        
        X_adv = (X_adv + 1)/2
        # breakpoint()
        clean_sdedit, adv_sdedit = self.gen_edit_results(torch.cat([X, X_adv], 0)) 
        return X_adv, adv_sdedit, clean_sdedit



