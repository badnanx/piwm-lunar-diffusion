# Paper Alignment Notes

This document explains how this repo relates to two PIWM papers/repos:

1. Four Principles for Physically Interpretable World Models
2. Physically Interpretable World Models via Weakly Supervised Representation Learning

## 1. Four Principles paper

The 4P paper defines four principles:

P1:
  functionally organized latent space

P2:
  aligned invariant/equivariant representations

P3:
  multiple forms and strengths of supervision

P4:
  partitioned generative outputs

## 2. Our current model vs P1

4P P1 is broader than our current implementation.

4P idea:
  split the latent space into functional branches, such as physical state, style, interactions, and residual features.

Our current model:
  one latent vector of length 64
  first six dimensions are physical
  remaining dimensions are residual visual dimensions

So our model is P1-inspired, but not a full modular multi-branch implementation.

## 3. Our current model vs P2

4P P2 uses explicit observation transformations and corresponding latent transformations.

Example from the paper:
  shift the Lunar Lander position in image space
  enforce a corresponding shift in latent state

Our current model:
  uses consecutive-frame state deltas
  enforces that latent physical changes match physical state changes

This is related to equivariance, but it is not the exact 4P transformation experiment.

## 4. Our current model vs P3

4P P3 covers multiple supervision strengths:
  exact labels
  partial labels
  weak labels
  noisy labels
  temporal smoothness
  estimated velocities

Our repo currently supports:
  clean state labels
  noisy state keys such as noisy_states_2
  selected state indices

We have done only limited P3 experiments. We have not yet done a full systematic partial/noisy supervision study.

## 5. Our current model vs P4

4P P4 partitions generated outputs into multiple physically meaningful pieces.

Paper-style P4:
  separate decoder for segment 1
  separate decoder for segment 2
  separate decoder for segment 3
  combine generated segments into full image
  train with global reconstruction plus segment reconstruction

Our current model:
  one decoder generates the whole image

So we have not implemented full P4 yet.

## 6. P4-lite crop idea

The proposed crop loss is a lightweight P4-inspired step.

Instead of using SAM masks and multiple decoders, we use the known physical x,y state to crop around the lander and add extra reconstruction loss on that crop.

Baseline image loss:
  whole image MSE

P4-lite loss:
  whole image MSE
  plus lander crop MSE

Why this is aligned with P4:
  it gives extra importance to a physically meaningful output region
  it addresses the small-object problem
  it avoids needing hand-labeled masks
  it is easier than full SAM/multi-decoder partitioning

Limitations:
  not full output partitioning
  still one decoder
  depends on state x,y
  crop may need care for offscreen lander frames

## 7. WSRL PIWM paper

The WSRL paper studies intrinsic vs extrinsic and continuous vs discrete PIWM variants.

Intrinsic:
  one encoder maps image directly to physical/interpretable latent

Extrinsic:
  first train a vision autoencoder
  freeze it
  then train a physical encoder/extractor on top of the vision latent

Continuous:
  latent values are real numbers

Discrete:
  VQ-VAE/codebook quantizes latents into discrete learned vectors

The WSRL paper reports that extrinsic-discrete is strongest overall because it decouples perception from physical interpretation and quantization regularizes visual noise.

## 8. Our current model vs WSRL

Our current model is closest to:
  intrinsic continuous PIWM-style baseline

It is not:
  extrinsic
  discrete
  VQ-VAE
  staged/frozen in the WSRL sense

Why we started here:
  faster to build
  easier to debug
  establishes data loading, paired dynamics, evaluation, export, and correction pipeline

Why we should not stop here:
  WSRL suggests extrinsic-discrete is a stronger final architecture
  our current model overfits and produces blurry images
  physical calibration on test is still imperfect

## 9. Roadmap

Current status:
  intrinsic-continuous PIWM-style baseline works
  lander appears but is blurry
  dynamics predicts latents moderately
  MLP correction improves test latents only slightly

Near-term:
  add per-dimension diagnostics
  add P4-lite crop loss
  compare against baseline

Then:
  build extrinsic and/or VQ-VAE branch
  train/freeze staged components
  export better latent transitions
  train diffusion/correction on PIWM latents
