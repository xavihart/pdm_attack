import torch
import tqdm

class Atk_PDMPlus_Attacker():
    def __init__(self, diffusion, model, encoder, decoder):
        self.diffusion = diffusion
        self.model = model
        self.encoder = encoder
        self.decoder = decoder

    def gen_pdm_atkp_config(self, delta, gamma1, gamma2, T):
        self.delta = delta
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.T = T

    def attack_pdm_atk_plus(self, x):
        x_adv = x.clone()
        attack_loss = 1e9
        z_adv = self.encode(x_adv)
        while not self.is_convergent(attack_loss):
            x_adv = self.decode(z_adv) # Decode by VAE
            timestep = self.ssample_timestep() # Sample random t \in [0, T]
            e1, e2 = self.sample_noise() # Standard Normal
            sample_clean = self.compute_sample(x, timestep, e1)
            sample_adv = self.compute_sample(x_adv, timestep, e2)

            attack_loss = self.compute_attack_loss(sample_clean, sample_adv) # Compute loss
            attack_loss.backward() # Populate gradients
            # Gradient Descent for z_adv
            z_adv -= self.gamma1 * torch.sign(z_adv.grad)
            z_adv.zero_grad() # Reset Gradient
            # Optimize for Fidelity Loss
            fidelity_loss = self.compute_fidelity_loss(x, self.decode(z_adv))
            while fidelity_loss > self.delta:
                fidelity_loss.backward()
                z_adv -= self.gamma2 * z_adv.grad
                z_adv.zero_grad()  # Reset Gradient
                fidelity_loss = self.compute_fidelity_loss(x, self.decode(z_adv)) # Recalculate loss
        # Get final result
        x_adv = self.decode(z_adv)
        # Get SDEdit outputs
        clean_sdedit, adv_sdedit = self.gen_edit_results(torch.cat([x, x_adv], 0))

        return x_adv, clean_sdedit, adv_sdedit

    # Check if attack loss has converged
    def is_convergent(self, attack_loss):
        pass

    # Encode image into latent vector using VAE
    def encode(self, x):
        pass

    # Decode latent vector into image using VAE
    def decode(self, z):
        pass

    # Sample timestep
    def sample_timestep(self):
        pass

    # Sample Gaussian noise for diffusion
    def sample_noise(self):
        pass

    # Compute samples
    def compute_sample(self, x, t, e):
        pass

    # Adversarial attack loss
    def compute_attack_loss(self, x, x_adv):
        pass

    # Attack fidelity loss
    def compute_fidelity_loss(self, x, x_adv):
        pass

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
            out = self.diffusion.p_sample(self.model, sample, t_tensor)
            sample = out["sample"]

        # the output of diffusion model is [-1, 1], should be transformed to [0, 1]
        if to_01:
            sample = (sample + 1) / 2

        return sample

    # Helper function
    def gen_edit_results(self, x, strength_list = [0.5, 0.7]):
        return torch.cat([x]+[self.sdedit(x, strength) for strength in strength_list], -1)