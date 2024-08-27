## Installation:

1. Follow https://github.com/xavihart/Diff-PGD to install the env
2. download checkpoint into `ckpt/` from [[here]](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt)


## Run:

(1) Basic AdvDM for PDM ([ADM](https://github.com/openai/guided-diffusion?tab=readme-ov-file)):

```
python -m scripts.attack_pdm --image_path parrot.png
```
