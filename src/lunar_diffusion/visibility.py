import math

import numpy as np


SCALE = 30.0
VIEWPORT_W = 600
VIEWPORT_H = 400

W = VIEWPORT_W / SCALE
H = VIEWPORT_H / SCALE
HALF_W = W / 2.0
HALF_H = H / 2.0

LEG_DOWN = 18
HELIPAD_Y = H / 4.0
Y_OFFSET = HELIPAD_Y + LEG_DOWN / SCALE

LANDER_POLY = (
    np.array(
        [
            (-14, +17),
            (-17, 0),
            (-17, -10),
            (+17, -10),
            (+17, 0),
            (+14, +17),
        ],
        dtype=np.float32,
    )
    / SCALE
)

LEG_AWAY = 20 / SCALE
LEG_LEN = 18 / SCALE


def state_to_world(state):
    x = state[0] * HALF_W + HALF_W
    y = state[1] * HALF_H + Y_OFFSET
    angle = state[4]
    return x, y, angle


def world_to_pixel(x, y, img_w, img_h):
    px = x / W * img_w
    py = img_h - (y / H * img_h)
    return px, py


def rotate_points(points, angle):
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    rotation = np.array([[c, -s], [s, c]], dtype=np.float32)
    return points @ rotation.T


def lander_pixel_bounds(state, img_h=100, img_w=150):
    x, y, angle = state_to_world(state)

    points_local = [LANDER_POLY]

    for sign in (-1.0, 1.0):
        hip_local = np.array([[sign * LEG_AWAY, -0.1]], dtype=np.float32)
        foot_local = np.array([[sign * (LEG_AWAY + 0.15), -LEG_LEN]], dtype=np.float32)
        points_local.extend([hip_local, foot_local])

    all_world_points = []

    for local in points_local:
        rotated = rotate_points(local, angle)
        world = rotated + np.array([x, y], dtype=np.float32)
        all_world_points.append(world)

    all_world_points = np.concatenate(all_world_points, axis=0)

    pixel_points = np.array(
        [world_to_pixel(px, py, img_w, img_h) for px, py in all_world_points],
        dtype=np.float32,
    )

    left = float(pixel_points[:, 0].min())
    right = float(pixel_points[:, 0].max())
    top = float(pixel_points[:, 1].min())
    bottom = float(pixel_points[:, 1].max())

    return left, right, top, bottom


def is_fully_visible(state, img_h=100, img_w=150, margin=0):
    left, right, top, bottom = lander_pixel_bounds(state, img_h=img_h, img_w=img_w)

    return (
        left >= margin
        and right < img_w - margin
        and top >= margin
        and bottom < img_h - margin
    )
