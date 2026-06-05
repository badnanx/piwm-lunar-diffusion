# PIWM Lunar Diffusion

Goal: build a minimal Lunar Lander latent diffusion pipeline for PIWM-style experiments.

Current plan:
1. Load Lunar Lander .npz files.
2. Train a small VAE/autoencoder on Lunar Lander images.
3. Freeze the decoder.
4. Extract latents.
5. Train latent DDPM on those latents.
6. Decode generated/corrected latents and evaluate image/physical consistency.

Data format:
- imgs: (T, 100, 150, 3), uint8
- states: (T, 8), float32
- acts: (T,), int32
- noisy_states_2/5/10: noisy physical labels

Initial debug run:
- Dataset loader works.
- VAE forward pass works.
- Tiny VAE training run works on 10 train files and 5 test files.
