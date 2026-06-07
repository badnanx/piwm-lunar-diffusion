# Experiment Log

This file records what we tried, what we learned, and what to investigate next.

## 1. Single-frame VAE experiments

We first trained a basic ConvVAE on Lunar Lander frames.

Observation:
  reconstructions learned black sky, terrain, and some flag poles
  lander was missing or extremely blurry

Interpretation:
  full-image pixel MSE is dominated by background and terrain
  the lander is small and easy for the model to ignore
  single-frame VAE reconstruction is not enough for physically meaningful prediction

## 2. State-supervised latent VAE

We added state supervision to the first latent dimensions.

Tried:
  all state variables
  selected visible-ish variables such as x, y, theta

Observation:
  reconstruction improved but still struggled with lander
  velocity from a single frame is questionable

Interpretation:
  physical supervision helps, but a single-frame setup is incomplete for dynamics.

## 3. Paired PIWM baseline

We built LunarPairDataset:

  image_t
  image_next
  state_t
  state_next
  action_t

We trained:

  encoder(image_t) -> mu_t
  dynamics(mu_t, action_t) -> pred_mu_next
  decoder(pred_mu_next) -> pred image_next

Loss terms:
  recon_loss
  kl_loss
  p1_loss
  p2_loss
  dynamics_loss
  pred_recon_loss

The serious run was:

  outputs/piwm_pair_physstrong_v1

Best epoch:
  9

Evaluation:
  p1_t_rmse: 0.234198
  p1_next_rmse: 0.243476
  dynamics_state_rmse: 0.238846
  dynamics_latent_rmse: 0.080708
  p2_rmse: 0.063492

Observation:
  lander appears in recon/pred images, but is blurry
  terrain shape is captured, but edges are soft
  test losses are much worse than train losses

Interpretation:
  paired PIWM is much better than single-frame VAE
  model overfits
  physical latent generalization is imperfect
  pixel reconstruction is still not object-aware enough

## 4. P3/noisy supervision smoke test

We added state_key support.

Tested:
  states
  noisy_states_2

Observation:
  noisy supervision did not catastrophically break the model
  evaluation against clean states was still reasonable in debug
  p2 against noisy labels looked worse because noisy deltas are noisy

Interpretation:
  P3 support exists, but systematic P3 experiments are future work.

## 5. State extractor / P4 checker

We trained a CNN:

  image -> x,y,theta

Observation:
  x,y are learnable from real images
  theta remains difficult
  visible-only filtering did not fix theta

Interpretation:
  image-to-position is feasible
  theta is hard because the lander is small/blurry
  a checker should start with x,y before theta

## 6. Latent correction baseline

We exported latent transitions:

  z_t
  z_next
  z_pred_next
  correction = z_next - z_pred_next

Debug correction on same latent file:
  baseline_rmse: 0.034124
  corrected_rmse: 0.005798
  improvement_ratio: 5.886x

Proper train/test correction:
  test baseline_rmse: 0.060517
  test corrected_rmse: 0.055764
  improvement_ratio: 1.085x

Interpretation:
  correction is learnable on same distribution
  generalization to test is modest
  diffusion will need to beat this simple MLP correction baseline

## 7. Current visual issue

Current best paired PIWM images:
  lander present but blurry
  terrain shape plausible but soft
  sky/background very good

Likely causes:
  global MSE underweights small lander
  MSE encourages averaging
  decoder smooths high-frequency edges
  physical latent is useful but image generation is not object-aware

## 8. Next planned experiment

P4-lite crop loss.

Idea:
  keep global image reconstruction
  add extra crop reconstruction around lander x,y

Why:
  lander is physically important but small
  crop loss is a simple approximation of P4 output partitioning
  easier than SAM masks and multiple decoders

Experiment:
  baseline: outputs/piwm_pair_physstrong_v1
  new: outputs/piwm_pair_crop_*

Compare:
  visual grids
  crop_loss
  pred_crop_loss
  p1/dynamics RMSE
  per-dimension RMSE

## 9. P4-lite crop experiments

We implemented a state-guided crop loss around the Lunar Lander x,y position.

P4-lite v1:
  crop_size = 32
  crop_weight = 1.0
  pred_crop_weight = 0.5
  crop_loss_type = mse

Observation:
  lander appears inside the crop region
  reconstructions/predictions show the lander as a blurry blob
  flags are visible
  terrain remains soft

Interpretation:
  crop location is correct
  crop loss helps focus attention, but does not solve decoder blur or tiny-object detail

P4-lite v2:
  crop_size = 24
  crop_weight = 3.0
  pred_crop_weight = 1.0
  crop_loss_type = mse_l1

Observation:
  visual quality got worse
  lander and flags disappeared in non-real rows
  crop box still correctly contained the real lander

Interpretation:
  crop location is not the bottleneck
  heavier crop + MSE/L1 dominated the objective without producing sharper object detail
  the model still prefers smoothing/averaging inside the crop
  this suggests the bottleneck is decoder/object representation, not crop placement

Conclusion:
  Keep P4-lite v1 as the better crop baseline.
  Do not continue increasing crop weight or adding L1 blindly.
  Next promising direction is extrinsic/VQ-VAE or true object/mask-based partitioning.
