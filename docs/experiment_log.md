# Experiment Log

This log records important runs, observations, and decisions for the Lunar PIWM diffusion project.

## Format

Each experiment should include:

- Date:
- Git commit:
- Command/config:
- Goal:
- Dataset:
- Key settings:
- Results:
- Visual observations:
- Decision / next step:

---

## 2026-06-06 — Initial VAE and PIWM setup

### Goal

Build a standalone Lunar Lander repo, inspect the dataset, train initial VAE baselines, and move toward a PIWM-style latent baseline before adding diffusion.

### Dataset observations

- Train split: 345 `.npz` files, 30,776 frames.
- Test split: 55 `.npz` files, 4,928 frames.
- Each file contains:
  - `imgs`
  - `acts`
  - `states`
  - `noisy_states_2`
  - `noisy_states_5`
  - `noisy_states_10`

### Visibility analysis

Using renderer geometry:

- Fully visible 2D lander: about 69.06%.
- Any clipped: about 30.94%.
- Partial top clipping: about 25.31%.
- Fully above frame: about 5.01%.
- Horizontal clipping was negligible.

This explains why early visualization grids often showed the lander near or above the top edge.

### VAE observations

Plain VAE and P1-style VAE reconstructed background/terrain/flag poles but failed to reconstruct the tiny moving lander.

Likely reason:

- Pixel MSE is dominated by sky/ground/background.
- The lander is small, moving, rotating, and sometimes clipped.

### Decision

Stop iterating on single-frame VAE-only tweaks. Move toward a fuller PIWM baseline:

- Pair dataset with consecutive frames.
- P1 structured latent.
- P2 transition consistency.
- Dynamics model.
- Later P3/P4 and diffusion correction.
