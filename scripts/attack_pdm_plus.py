import argparse

from attacks.attack_atk_pdm import Atk_PDM_Attacker
from tools.utils import si, load_png
from tools.load_dm import get_imagenet_dm_conf
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='imagenet-256')
parser.add_argument("--image_path", type=str, default='parrot.png')
parser.add_argument("--save_path", type=str, default='examples/parrot_attack.png')
parser.add_argument("--attack_mode", type=str, default='base')
parser.add_argument("--image_size", type=int, default=224)
parser.add_argument("--optimization_steps", type=int, default=300)
parser.add_argument("--fidelity_delta", type=float, default=0.85)
parser.add_argument("--gamma_1", type=float, default=0.5)
parser.add_argument("--gamma_2", type=float, default=0.5)
args = parser.parse_args()

if __name__ == '__main__':
    device = 0
    model, dm = get_imagenet_dm_conf(model_name=args.model)
    x = load_png(p=args.image_path, size=args.image_size)[None, ...].to(device)

    attacker = Atk_PDM_Attacker(diffusion=dm, model=model, mode=args.attack_mode)

    with torch.no_grad():
        attacker.gen_pdm_atkp_config(delta=args.fidelity_delta,
                                     gamma1=args.gamma_1,
                                     gamma2=args.gamma_2,
                                     optimization_steps=args.optimization_steps,
                                     device=device)
        X_adv, adv_sdedit, clean_sdedit = attacker.attack_pdm_atk(x)
        si(torch.cat([clean_sdedit, adv_sdedit], -2), args.save_path)