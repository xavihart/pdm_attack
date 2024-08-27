import torch
import numpy 
import os


from tools.utils import si, load_png
from tools.load_dm import get_imagenet_dm_conf, GUIDED_DIFFUSION_MODEL_NAME
from tools.dataset import get_dataset
from attacks.attack_guided_diffusion import GuidedDiffusion_Attacker
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='imagenet-256')
parser.add_argument("--image_path", type=str, default='parrot.png')
parser.add_argument("--save_path", type=str, default='examples/parrot_attack.png')
parser.add_argument("--image_size", type=int, default=224)
parser.add_argument("--grad_mode", type=str, default='+')
parser.add_argument("--eps", type=int, default=16)
parser.add_argument("--steps", type=int, default=100)
args = parser.parse_args()

if __name__ == '__main__':
    
    device = 0
    model, dm = get_imagenet_dm_conf(model_name=args.model)
    x = load_png(p=args.image_path, size=args.image_size)[None, ...].to(device)
    
    attacker = GuidedDiffusion_Attacker(diffusion=dm, model=model)

    with torch.no_grad():
        attacker.gen_gpd_conf(eps=args.eps, steps=args.steps, step_size=1, clip_min=-1, clip_max=1)
        X_adv, adv_sdedit, clean_sdedit = attacker.attack_advdm(x, grad_mode=args.grad_mode)
        si(torch.cat([clean_sdedit, adv_sdedit], -2), args.save_path)


    

