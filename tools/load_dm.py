# import os
# import sys

# # Specify the path to the directory containing the module you want to import
# module_path = '../'

# Insert the directory into the system path
# if module_path not in sys.path:
#     sys.path.insert(0, module_path)

from guided_diffusion.script_util import (
                NUM_CLASSES,
                model_and_diffusion_defaults,
                classifier_defaults,
                create_model_and_diffusion,
                create_classifier,
                add_dict_to_argparser,
                args_to_dict,
            )

import torch as th
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
import torch.nn as nn
from torchvision import transforms as T, utils
import numpy as np
import argparse
import os
from argparse import Namespace
import time
import datetime
from .utils import *
ROOT = 'ckpt/'

GUIDED_DIFFUSION_MODEL_NAME = {
    'imagenet-256':{
        'path': '256x256_diffusion_uncond.pt',
        'cond': False,
        'resolution': 256
        },
    
    # 'imagenet-512':{
    #     'path': '256x256_diffusion_uncond.pt',
    #     'cond': False,
    #     'resolution': 512
    #     },
    
    'imagenet-512-cond':{
        'path': '512x512_diffusion.pt',
        'cond': True,
        'resolution': 512
        },
    
    'lsun-bed':{
        'path': 'lsun_bedroom.pt',
        'cond': False,
        'resolution': 256
        },
    
    'lsun-cat':{
        'path': 'lsun_cat.pt',
        'cond': False,
        'resolution': 256
        },
    
    'lsun-horse':{
        'path': 'lsun_horse.pt',
        'cond': False,
        'resolution': 256
        }
}

def get_imagenet_dm_conf(model_name, respace="", device='cuda',
                         ):
    
    
    model_info = GUIDED_DIFFUSION_MODEL_NAME[model_name]
    
    model_path = ROOT + model_info['path']
    class_cond = model_info['cond']
    resolution = model_info['resolution']
    
    #     MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True
    # --diffusion_steps 1000 --image_size 512 --learn_sigma True 
    # --noise_schedule linear --num_channels 256 --num_head_channels 64 
    # --num_res_blocks 2 --resblock_updown True --use_fp16 False
    # --use_scale_shift_norm True"
    # python classifier_sample.py $MODEL_FLAGS --classifier_scale 4.0 
    # --classifier_path models/512x512_classifier.pt 
    # --model_path models/512x512_diffusion.pt $SAMPLE_FLAGS
    
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
    )

    model_config = dict(
            use_fp16=False,
            attention_resolutions="32, 16, 8",
            class_cond=class_cond,
            diffusion_steps=1000,
            image_size=resolution,
            learn_sigma=True,
            noise_schedule='linear',
            num_channels=256,
            num_head_channels=64,
            num_res_blocks=2,
            resblock_updown=True,
            use_scale_shift_norm=True,
            timestep_respacing=respace,
        )

    defaults.update(model_and_diffusion_defaults())
    defaults.update(model_config)
    args = Namespace(**defaults)
    

    model, diffusion = create_model_and_diffusion(
    **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    cprint('Create DM   ---------------', 'y')
    
    
    # load ckpt
    
    ckpt = th.load(model_path)
    model.load_state_dict(ckpt)
    model = model.to(device)
    
    cprint('Load DM Ckpt      ---------------', 'y')
    
    return model, diffusion





