# PIWM Lunar Diffusion: Concepts

This repo studies Lunar Lander world models with physically interpretable latent states and later latent diffusion/correction.

The core project idea is:

image_t
  -> encoder
  -> latent_t
  -> dynamics(latent_t, action_t)
  -> predicted latent_{t+1}
  -> correction / diffusion model
  -> corrected latent_{t+1}
  -> decoder
  -> predicted image_{t+1}

The goal is not only to generate visually plausible images. The goal is to predict physically meaningful future states and images.

## 1. World model pieces

A world model usually has three pieces:

Encoder:
  image -> latent

Dynamics:
  latent_t + action_t -> predicted latent_{t+1}

Decoder:
  latent -> image

In code, the current paired baseline uses:

ConvVAE:
  image -> mu, logvar
  latent -> reconstructed image

LatentDynamicsMLP:
  mu_t + action_t -> predicted_mu_next

## 2. What is a latent?

A latent vector is a compressed numerical representation of an image.

Example:
  image shape: 3 x 100 x 150
  latent shape: 64

So instead of predicting directly in image space, the model predicts in latent space.

## 3. What is mu?

In a VAE, the encoder predicts a distribution over latents, not just one latent vector.

It outputs:
  mu     = mean of the latent distribution
  logvar = log variance of the latent distribution

For physics and dynamics, we usually use mu because it is deterministic and stable.

In code:
  mu_t = encoder(image_t)

Shape example:
  mu_t.shape = [batch_size, latent_dim]
  for example [32, 64]

## 4. PyTorch indexing: mu[:, 2]

If mu has shape [32, 64], then:

  mu[:, 0] means latent dimension 0 for all 32 samples
  mu[:, 1] means latent dimension 1 for all 32 samples
  mu[:, 2] means latent dimension 2 for all 32 samples

The colon means "all rows in the batch."

Our current physical latent mapping is:

  mu[:, 0] -> x
  mu[:, 1] -> y
  mu[:, 2] -> vx
  mu[:, 3] -> vy
  mu[:, 4] -> theta
  mu[:, 5] -> omega

The rest of the latent dimensions are residual visual latents:

  mu[:, 6:] -> visual / residual information

## 5. Lunar Lander state variables

The first six state dimensions are:

  x      = horizontal position
  y      = vertical position
  vx     = horizontal velocity
  vy     = vertical velocity
  theta  = lander angle / rotation
  omega  = angular velocity

The last two are leg-contact indicators:

  left_leg
  right_leg

## 6. What is theta?

Theta is the lander's angle/orientation.

Roughly:
  theta = 0 means upright
  theta > 0 or theta < 0 means tilted

Theta is visually hard because the lander is small and often blurry. If the decoded lander is just a purple blob, it may not contain enough information to tell whether the feet or head are contacting the ground.

## 7. Loss functions

A loss function is a number that measures how wrong the model is.

Training tries to make the total loss smaller.

Our current paired PIWM-style baseline has this total loss:

  total_loss =
      recon_loss
    + kl_weight * kl_loss
    + p1_weight * p1_loss
    + p2_weight * p2_loss
    + dynamics_weight * dynamics_loss
    + pred_recon_weight * pred_recon_loss

The serious baseline used:

  kl_weight         = 0.0001
  p1_weight         = 1.0
  p2_weight         = 0.5
  dynamics_weight   = 1.0
  pred_recon_weight = 0.1

## 8. recon_loss

recon_loss trains the autoencoder.

Current frame:
  image_t -> encoder -> mu_t -> decoder -> recon_t

We want:
  recon_t close to image_t

Next frame:
  image_next -> encoder -> mu_next -> decoder -> recon_next

We want:
  recon_next close to image_next

This is reconstruction, not prediction. The model sees the image it is reconstructing.

## 9. pred_recon_loss

pred_recon_loss trains the predicted next image.

Pipeline:

  image_t -> encoder -> mu_t
  mu_t + action_t -> dynamics -> predicted_mu_next
  predicted_mu_next -> decoder -> pred_image_next

We want:
  pred_image_next close to image_next

This is the world-model prediction row in the visual grids.

## 10. kl_loss

KL regularizes the VAE latent distribution.

In our current PIWM-style model, KL is applied only to the residual visual dimensions, not the physical dimensions.

Why:
  physical dims should be free to match x, y, vx, vy, theta, omega
  residual dims should stay well-behaved

## 11. P1 loss

P1 is physical latent supervision.

It says:
  the first latent dimensions should directly match physical state variables

For us:
  mu_t[:6] should match state_t[:6]
  mu_next[:6] should match state_next[:6]

Plain English:
  latent dimension 0 should mean x
  latent dimension 1 should mean y
  etc.

## 12. P2 loss

P2 is transition consistency.

It says:
  physical changes in latent space should match physical changes in real state

If the true state changes by:

  state_delta = state_next[:6] - state_t[:6]

then the physical latent should change similarly:

  mu_next[:6] should be close to mu_t[:6] + state_delta

Plain English:
  if the lander moved upward in real state, the y latent should also move upward.

## 13. dynamics_loss

The dynamics model predicts the next latent:

  pred_mu_next = dynamics(mu_t, action_t)

The dynamics loss has two ideas:

  pred_mu_next should be close to mu_next
  pred_mu_next[:6] should be close to state_next[:6]

So it asks the predicted next latent to be both:
  close to the encoder's next latent
  physically meaningful

## 14. Why weights are fixed

The weights are not learned by the model because they express the researcher's priorities.

If the model learned the weights freely, it might set hard losses like P1 or P2 close to zero and ignore physical interpretability.

Fixed weights let us say:
  physical interpretability matters
  dynamics matters
  image prediction matters, but should not dominate everything

Changing weights changes model behavior:

  higher p1_weight:
    stronger physical latent alignment
    possible worse image reconstruction

  higher p2_weight:
    stronger transition consistency
    possible overemphasis on local deltas

  higher pred_recon_weight:
    stronger predicted images
    possible return to pixel-MSE shortcuts

  higher recon emphasis:
    better image fitting
    possible overfitting to background/terrain

## 15. MSE, MAE, RMSE

MSE:
  mean squared error
  squares each error and averages
  penalizes large errors strongly
  often causes blurry image predictions because averages can minimize squared error

MAE:
  mean absolute error
  uses absolute value instead of square
  often less dominated by large errors
  can sometimes produce sharper images

RMSE:
  root mean squared error
  sqrt(MSE)
  useful for reporting because it is in the same units as the original quantity

For images:
  low MSE does not guarantee physically correct lander state.
  black sky and terrain can dominate pixel count.

For physical states:
  RMSE is useful because x RMSE, y RMSE, theta RMSE can be inspected separately.

## 16. Why image MSE can be misleading

Most pixels are background or terrain.

The lander is small.

A model can reduce image MSE by reconstructing:
  black sky
  white/gray terrain
  general flag/terrain shape

while still making the lander blurry or wrong.

This is why we inspect:
  visual grids
  physical RMSE
  p1_loss
  dynamics_state_loss
  per-dimension errors

not only recon_loss.
