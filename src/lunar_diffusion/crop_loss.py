import torch
import torch.nn.functional as F


# LunarLander coordinate constants, matching our visibility geometry.
SCALE = 30.0
VIEWPORT_W = 600
VIEWPORT_H = 400

WORLD_W = VIEWPORT_W / SCALE
WORLD_H = VIEWPORT_H / SCALE

HALF_WORLD_W = WORLD_W / 2.0
HALF_WORLD_H = WORLD_H / 2.0

LEG_DOWN = 18
HELIPAD_Y = WORLD_H / 4.0
Y_OFFSET = HELIPAD_Y + LEG_DOWN / SCALE


def state_xy_to_pixel(states: torch.Tensor, img_h: int, img_w: int):
    """
    Convert LunarLander state x,y into approximate image pixel coordinates.

    states[:, 0] = x
    states[:, 1] = y

    Returns:
        px, py tensors of shape (B,)
    """
    x_state = states[:, 0]
    y_state = states[:, 1]

    world_x = x_state * HALF_WORLD_W + HALF_WORLD_W
    world_y = y_state * HALF_WORLD_H + Y_OFFSET

    px = world_x / WORLD_W * img_w
    py = img_h - (world_y / WORLD_H * img_h)

    return px, py


def crop_around_state(
    images: torch.Tensor,
    states: torch.Tensor,
    crop_size: int = 32,
):
    """
    Differentiably crop a square patch around the lander's x,y state.

    images: (B, C, H, W)
    states: (B, state_dim), with states[:,0]=x and states[:,1]=y

    Returns:
        crops: (B, C, crop_size, crop_size)
    """
    if images.dim() != 4:
        raise ValueError(f"Expected images with shape (B,C,H,W), got {images.shape}")

    batch_size, channels, img_h, img_w = images.shape
    device = images.device
    dtype = images.dtype

    px, py = state_xy_to_pixel(states.to(device=device, dtype=dtype), img_h, img_w)

    half = (crop_size - 1) / 2.0
    offsets = torch.linspace(-half, half, crop_size, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")

    sample_x = px[:, None, None] + xx[None, :, :]
    sample_y = py[:, None, None] + yy[None, :, :]

    # grid_sample expects normalized coordinates in [-1, 1].
    norm_x = 2.0 * sample_x / max(img_w - 1, 1) - 1.0
    norm_y = 2.0 * sample_y / max(img_h - 1, 1) - 1.0

    grid = torch.stack([norm_x, norm_y], dim=-1)

    crops = F.grid_sample(
        images,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    return crops


def crop_reconstruction_loss(
    pred_crop: torch.Tensor,
    target_crop: torch.Tensor,
    loss_type: str = "mse",
):
    """
    Reconstruction loss for a cropped image region.

    loss_type options:
        mse:
            Standard mean squared error. Stable, but can encourage blur.

        l1:
            Mean absolute error. Often less blurry than MSE.

        mse_l1:
            Combination of MSE and L1. Keeps MSE stability while adding
            sharper L1 pressure.
    """
    if loss_type == "mse":
        return F.mse_loss(pred_crop, target_crop, reduction="mean")

    if loss_type == "l1":
        return F.l1_loss(pred_crop, target_crop, reduction="mean")

    if loss_type == "mse_l1":
        mse = F.mse_loss(pred_crop, target_crop, reduction="mean")
        l1 = F.l1_loss(pred_crop, target_crop, reduction="mean")
        return mse + l1

    raise ValueError(f"Unknown crop loss type: {loss_type}")


def state_guided_crop_loss(
    pred_images: torch.Tensor,
    target_images: torch.Tensor,
    states: torch.Tensor,
    crop_size: int = 32,
    loss_type: str = "mse",
):
    """
    Reconstruction loss on a crop centered around the true lander x,y position.
    """
    pred_crop = crop_around_state(pred_images, states, crop_size=crop_size)
    target_crop = crop_around_state(target_images, states, crop_size=crop_size)
    return crop_reconstruction_loss(
        pred_crop=pred_crop,
        target_crop=target_crop,
        loss_type=loss_type,
    )


# Backward-compatible name used by older scripts.
def state_guided_crop_mse(
    pred_images: torch.Tensor,
    target_images: torch.Tensor,
    states: torch.Tensor,
    crop_size: int = 32,
):
    return state_guided_crop_loss(
        pred_images=pred_images,
        target_images=target_images,
        states=states,
        crop_size=crop_size,
        loss_type="mse",
    )
