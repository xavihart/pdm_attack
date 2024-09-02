import argparse

from attacks.attack_atk_pdmplus import Atk_PDMPlus_Attacker
from tools.utils import si, load_png
from tools.load_dm import get_imagenet_dm_conf
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='imagenet-256')
parser.add_argument("--image_path", type=str, default='parrot.png')
parser.add_argument("--save_path", type=str, default='examples/parrot_attack_pdm_plus.png')
parser.add_argument("--image_size", type=int, default=224)
parser.add_argument("--steps", type=int, default=100)
args = parser.parse_args()

if __name__ == '__main__':
    device = 0
    model, dm = get_imagenet_dm_conf(model_name=args.model)
    x = load_png(p=args.image_path, size=args.image_size)[None, ...].to(device)

    attacker = Atk_PDMPlus_Attacker(diffusion=dm, model=model)

    with torch.no_grad():
        attacker.gen_pdm_atkp_config()
        X_adv, adv_sdedit, clean_sdedit = attacker.attack_pdm_atk_plus(x)
        si(torch.cat([clean_sdedit, adv_sdedit], -2), args.save_path)