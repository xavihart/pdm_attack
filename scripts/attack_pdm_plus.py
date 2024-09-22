import argparse

from attacks.attack_atk_pdm import Atk_PDM_Attacker
from tools.utils import si, load_png
from tools.load_dm import get_imagenet_dm_conf
import torch
from diffusers import AutoencoderKL

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='imagenet-256')
parser.add_argument("--vae_url", type=str, default="CompVis/stable-diffusion-v1-4")
parser.add_argument("--image_path", type=str, default='parrot.png')
parser.add_argument("--save_path", type=str, default='examples/parrot_attack_latent.png')
parser.add_argument("--attack_mode", type=str, default='base')
parser.add_argument("--image_size", type=int, default=224)
parser.add_argument("--optimization_steps", type=int, default=100)
parser.add_argument("--fidelity_delta", type=float, default=0.4)
parser.add_argument("--fidelity_step_size", type=float, default=40)
parser.add_argument("--step_size", type=float, default=1)
parser.add_argument("--eps", type=float, default=32)
parser.add_argument("--respace", type=str, default="ddim100")
args = parser.parse_args()

if __name__ == '__main__':
    device = 0
    model, dm = get_imagenet_dm_conf(model_name=args.model, respace=args.respace)

    vae = None

    if args.attack_mode == 'latent':
        print('Loading VAE   ---------------')
        vae = AutoencoderKL.from_pretrained(args.vae_url, subfolder="vae").to(device)


    x = load_png(p=args.image_path, size=args.image_size)[None, ...].to(device)

    attacker = Atk_PDM_Attacker(diffusion=dm, model=model, mode=args.attack_mode, vae=vae)

    with torch.no_grad():
        attacker.gen_pdm_atkp_config(delta=args.fidelity_delta,
                                     fidelity_step_size=args.fidelity_step_size,
                                     optimization_steps=args.optimization_steps,
                                     device=device,
                                     eps=args.eps,
                                     step_size=args.step_size,
                                     clip_min=-1,
                                     clip_max=1,
                                     )
        results, x_adv = attacker.attack_pdm_atk(x)
        # results B * C* H * W, split it into two halves
        results_clean, results_adv = torch.split(results, results.shape[0] // 2, dim=0)
        si(results_adv, args.save_path)