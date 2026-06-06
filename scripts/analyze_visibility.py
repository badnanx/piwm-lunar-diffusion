import argparse
import glob
import math
import os

import numpy as np


# Constants copied from the Slack renderer
FPS = 50
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
    c = math.cos(angle)
    s = math.sin(angle)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float32)
    return points @ rotation.T


def lander_pixel_bounds(state, img_h=100, img_w=150, include_legs=True):
    x, y, angle = state_to_world(state)

    points_local = [LANDER_POLY]

    if include_legs:
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

    x_pixels = pixel_points[:, 0]
    y_pixels = pixel_points[:, 1]

    left = float(x_pixels.min())
    right = float(x_pixels.max())
    top = float(y_pixels.min())
    bottom = float(y_pixels.max())

    return left, right, top, bottom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--img_h", type=int, default=100)
    parser.add_argument("--img_w", type=int, default=150)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))

    total = 0

    full_visible = 0

    partial_top = 0
    fully_above = 0
    partial_bottom = 0
    fully_below = 0

    partial_left = 0
    fully_left = 0
    partial_right = 0
    fully_right = 0

    any_clipped = 0

    state_x_values = []
    state_y_values = []
    left_pixels = []
    right_pixels = []
    top_pixels = []
    bottom_pixels = []

    for path in files:
        with np.load(path) as data:
            states = data["states"]

        for state in states:
            total += 1
            left, right, top, bottom = lander_pixel_bounds(
                state, img_h=args.img_h, img_w=args.img_w
            )

            state_x_values.append(state[0])
            state_y_values.append(state[1])
            left_pixels.append(left)
            right_pixels.append(right)
            top_pixels.append(top)
            bottom_pixels.append(bottom)

            visible_x = (left >= 0) and (right < args.img_w)
            visible_y = (top >= 0) and (bottom < args.img_h)

            if visible_x and visible_y:
                full_visible += 1
            else:
                any_clipped += 1

            if right < 0:
                fully_left += 1
            elif left < 0:
                partial_left += 1

            if left >= args.img_w:
                fully_right += 1
            elif right >= args.img_w:
                partial_right += 1

            if bottom < 0:
                fully_above += 1
            elif top < 0:
                partial_top += 1

            if top >= args.img_h:
                fully_below += 1
            elif bottom >= args.img_h:
                partial_bottom += 1

    def pct(n):
        return 100.0 * n / total

    print("num_frames:", total)
    print()
    print(f"fully visible 2D: {full_visible:6d} ({pct(full_visible):6.2f}%)")
    print(f"any clipped:      {any_clipped:6d} ({pct(any_clipped):6.2f}%)")
    print()
    print("vertical clipping:")
    print(f"  partial_top:    {partial_top:6d} ({pct(partial_top):6.2f}%)")
    print(f"  fully_above:    {fully_above:6d} ({pct(fully_above):6.2f}%)")
    print(f"  partial_bottom: {partial_bottom:6d} ({pct(partial_bottom):6.2f}%)")
    print(f"  fully_below:    {fully_below:6d} ({pct(fully_below):6.2f}%)")
    print()
    print("horizontal clipping:")
    print(f"  partial_left:   {partial_left:6d} ({pct(partial_left):6.2f}%)")
    print(f"  fully_left:     {fully_left:6d} ({pct(fully_left):6.2f}%)")
    print(f"  partial_right:  {partial_right:6d} ({pct(partial_right):6.2f}%)")
    print(f"  fully_right:    {fully_right:6d} ({pct(fully_right):6.2f}%)")

    state_x_values = np.array(state_x_values)
    state_y_values = np.array(state_y_values)
    left_pixels = np.array(left_pixels)
    right_pixels = np.array(right_pixels)
    top_pixels = np.array(top_pixels)
    bottom_pixels = np.array(bottom_pixels)

    print()
    clipped_top_y = state_y_values[top_pixels < 0]
    if len(clipped_top_y) > 0:
        print("state y where top starts clipping:")
        print("  min y with top clipping:", clipped_top_y.min())
        print("  5th percentile:", np.percentile(clipped_top_y, 5))
        print("  median:", np.percentile(clipped_top_y, 50))

    clipped_left_x = state_x_values[left_pixels < 0]
    clipped_right_x = state_x_values[right_pixels >= args.img_w]

    print()
    if len(clipped_left_x) > 0:
        print("state x where left starts clipping:")
        print("  max x with left clipping:", clipped_left_x.max())
        print("  median:", np.percentile(clipped_left_x, 50))
    else:
        print("no left clipping detected")

    if len(clipped_right_x) > 0:
        print("state x where right starts clipping:")
        print("  min x with right clipping:", clipped_right_x.min())
        print("  median:", np.percentile(clipped_right_x, 50))
    else:
        print("no right clipping detected")


if __name__ == "__main__":
    main()
