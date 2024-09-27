import torch
from tqdm import tqdm, trange
from torch.cuda import device


class Atk_PDM_Attacker():
    def __init__(self, diffusion, model, mode='base', vae=None):
        self.diffusion = diffusion
        self.model = model
        self.mode = mode
        self.vae = vae

    def gen_pdm_atkp_config(self, delta, fidelity_step_size, optimization_steps, device, clip_min, clip_max, eps, step_size):
        self.delta = delta
        self.T = len(self.diffusion.use_timesteps)
        self.optimization_steps = optimization_steps
        self.device = device
        self.clip_min = clip_min
        self.clip_max = clip_max
        
        self.eps = eps / 255 * (clip_max - clip_min)
        self.step_size = step_size / 255 * (clip_max - clip_min)
        self.fidelity_step_size = fidelity_step_size / 255 * (clip_max - clip_min)

    # Helper attack function
    def attack_pdm_atk(self, x):
        if self.mode == 'base':
            return self.attack_pdm_atk_base(x)
        elif self.mode == 'latent':
            return self.attack_pdm_atk_latent(x)
        else:
            print('Unknown attack mode (use base or latent')
            raise RuntimeError()

    def attack_pdm_atk_latent(self, x):
        x_adv = x.clone().detach() * 2 - 1
        x_raw = x.clone().detach() * 2 - 1

        z_adv = self.encode(x_adv)  # Encode by VAE to latent space
        with torch.enable_grad():
            for i in trange(self.optimization_steps):
                z_adv = z_adv.clone().detach()
                z_adv.requires_grad = True

                x_adv = self.decode(z_adv) # Decode by VAE

                timestep = self.sample_timestep().long()  # Sample random t \in [0, T]
                e1, e2 = self.sample_noise(x_raw.shape)  # Standard Normal
                sample_clean = self.compute_sample(x_raw, timestep, e1)  # Computing samples with noise
                sample_adv = self.compute_sample(x_adv, timestep, e2)

                intermediate_clean = self.get_unet_intermediate(sample_clean, timestep)  # Running denoising UNETs
                intermediate_adv = self.get_unet_intermediate(sample_adv, timestep)

                attack_loss = self.compute_attack_loss(intermediate_clean, intermediate_adv)  # Compute loss
                attack_loss.backward()  # Populate gradients
                g_att = z_adv.grad.detach()
                # x_adv = x_adv + self.gamma1 * torch.sign(g_att) # Gradient Descent for x_adv
                z_adv = z_adv + torch.sign(g_att) * self.step_size  # Step size
                z_adv = z_adv.clone().detach()
                z_adv.requires_grad = True

                fidelity_loss = self.compute_fidelity_loss(x_raw, self.decode(z_adv))
                while fidelity_loss > self.delta:
                    print(fidelity_loss, self.fidelity_step_size)
                    fidelity_loss.backward()
                    g_fdl = z_adv.grad.detach()
                    z_adv = z_adv - self.fidelity_step_size * g_fdl
                    # Reset gradients before next run
                    z_adv = z_adv.clone().detach()
                    z_adv.requires_grad = True
                    fidelity_loss = self.compute_fidelity_loss(x_raw, self.decode(z_adv)) # Recalculate

        x_adv = self.decode(z_adv)
        x_adv.data = torch.clamp(x_adv, min=self.clip_min, max=self.clip_max)  # Data must be [-1; 1]
        x_adv = (x_adv + 1) / 2
        self.model.eval()  # Eval mode to compute SDEdit
        # Get SDEdit outputs
        # generate more samples
        ori = torch.cat([x] * 5 + [x_adv] * 5, 0)
        results = self.gen_edit_results(ori)
        return results, x_adv

    # Base attack without latent space
    def attack_pdm_atk_base(self, x):
        x_adv = x.clone().detach() * 2 - 1
        x_raw = x.clone().detach() * 2 - 1
        with torch.enable_grad():
            for i in trange(self.optimization_steps):
                x_adv = x_adv.clone().detach()
                x_adv.requires_grad = True
                timestep = self.sample_timestep().long() # Sample random t \in [0, T]
                e1, e2 = self.sample_noise(x_raw.shape) # Standard Normal
                sample_clean = self.compute_sample(x_raw, timestep, e1) # Computing samples with noise
                sample_adv = self.compute_sample(x_adv, timestep, e2)

                intermediate_clean = self.get_unet_intermediate(sample_clean, timestep) # Running denoising UNETs
                intermediate_adv = self.get_unet_intermediate(sample_adv, timestep)

                attack_loss = self.compute_attack_loss(intermediate_clean, intermediate_adv) # Compute loss
                attack_loss.backward() # Populate gradients
                g_att = x_adv.grad.detach()
                # x_adv = x_adv + self.gamma1 * torch.sign(g_att) # Gradient Descent for x_adv
                x_adv = x_adv + torch.sign(g_att) * self.step_size # Step size
                
                x_adv = torch.clamp(x_adv, min=x_raw - self.eps, max=x_raw + self.eps) # Clip using epsilon ball
                x_adv.data = torch.clamp(x_adv, min=self.clip_min, max=self.clip_max) # Data must be [-1; 1]
        
        x_adv = (x_adv + 1) / 2
        self.model.eval() # Eval mode to compute SDEdit
        # Get SDEdit outputs 
        # generate more samples
        ori = torch.cat([x] * 5 + [x_adv] * 5, 0)
        results = self.gen_edit_results(ori)
        return results, x_adv

    # Encode image into latent vector using VAE
    def encode(self, x, sample_posterior=False, generator=None):
        posterior = self.vae.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()

        return z

    # Decode latent vector into image using VAE
    def decode(self, z):
        return self.vae.decode(z).sample

    # Sample timestep, from 0 to t-1
    def sample_timestep(self):
        return torch.randint(0, self.T, (1,), device=self.device)


    # Sample Gaussian noise for diffusion
    def sample_noise(self, shape):
        means = torch.zeros(shape, device=self.device)
        var = torch.ones(shape, device=self.device)
        e1 = torch.normal(means, var)
        e2 = torch.normal(means, var)
        return e1, e2

    # Get intermediate output from denoising UNET middle block
    def get_unet_intermediate(self, x, timestep):
        external_otp = []

        def get_intermediate_output(_1, _2, output):
            external_otp.append(output)# Save output

        hook = self.model.middle_block.register_forward_hook(get_intermediate_output) # Save intermediate output
        _ = self.diffusion.p_mean_variance(self.model, x, timestep)['model_output'] # Run denoising step
        hook.remove()

        return external_otp[0]

    # Compute samples
    def compute_sample(self, x, t, e):
        return self.diffusion.q_sample(x, t, return_noise=False, noise=e)

    # Adversarial attack loss
    def compute_attack_loss(self, unet_intermediate_clean, unet_intermediate_adv):
        return torch.nn.functional.mse_loss(unet_intermediate_clean, unet_intermediate_adv)

    # Attack fidelity loss
    def compute_fidelity_loss(self, x, x_adv):
        return torch.nn.functional.mse_loss(x.detach(), x_adv)

    # SDEdit
    @torch.no_grad()
    def sdedit(self, x, t, to_01=True):
        """
        x: input image, B C H W
        t: editing strength from 0  to 1
        to_01: the original output of diffusion model is [-1, 1], set to True will transfer to [0, 1]
        """

        assert t < 1 and t > 0
        t = int(t * len(self.diffusion.use_timesteps))
        sample_indices = list(range(t + 1))[::-1]

        B, _, _, _ = x.shape

        # make sure the input image is scaled to [-1, 1]
        if x.min() >= 0:
            x = x * 2 - 1

        t = torch.full((B,), t).long().to(x.device)
        x_t = self.diffusion.q_sample(x, t)
        sample = x_t

        print('Run Diffusion Sampling ...')
        for i in tqdm(sample_indices):
            t_tensor = torch.full((B,), i).long().to(x.device)
            out = self.diffusion.ddim_sample(self.model, sample, t_tensor)
            sample = out["sample"]

        # the output of diffusion model is [-1, 1], should be transformed to [0, 1]
        if to_01:
            sample = (sample + 1) / 2
        
        return sample
    
    # Helper function
    def gen_edit_results(self, x, strength_list = [0.05, 0.3]): # [0.5, 0.7]):
        return torch.cat([x]+[self.sdedit(x, strength) for strength in strength_list], -1)