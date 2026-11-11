"""
coords.py -- pixel grid  <->  complex plane.

A Region says WHERE in the complex plane a picture (or the output canvas) lives:
a centre c, a plane-width, and a rotation.  The plane-height follows from the
pixel aspect ratio, so pixels stay square (the map is a similarity z = c + s*e^{i*rot}*(u+iv),
which is conformal -- angles are preserved).

Conventions: pixel (px,py) has its centre at integer coords, py grows DOWNWARD,
while Im(z) grows UPWARD, hence the sign flip on the y axis.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Region:
    center: complex = 0j     # complex-plane point at the middle of the image
    width: float = 4.0       # horizontal extent in the plane (x from c-width/2 to c+width/2)
    rotation: float = 0.0    # radians, counter-clockwise rotation about `center`


def pixel_to_complex(px, py, W, H, region):
    """Pixel coordinates (float arrays ok) -> z = x + iy.  W, H = image size in pixels."""
    # +0.5 moves from "pixel centre at integer" to "pixel spans [i, i+1)", so the image
    # edges sit exactly at u = -1/2 and u = +1/2 (the middle of the image is u = 0)
    u = (px + 0.5) / W - 0.5                # horizontal offset in [-1/2, 1/2]
    v = 0.5 - (py + 0.5) / H                # vertical offset, flipped so +v is up
    # scale to plane units; the same factor (width/W per pixel) is used in x and y -> square pixels
    local = region.width * (u + 1j * v * H / W)   # plane units; height = width*H/W
    # rotate about the centre, then translate to the region's centre
    return region.center + local * np.exp(1j * region.rotation)


def complex_to_pixel(z, W, H, region):
    """Inverse of pixel_to_complex: z -> fractional pixel coords (px, py)."""
    local = (z - region.center) * np.exp(-1j * region.rotation)   # undo the translation and the rotation
    u = local.real / region.width + 0.5                           # back to [0, 1] across the width
    v = local.imag / (region.width * H / W)                       # vertical offset in units of the height
    return u * W - 0.5, (0.5 - v) * H - 0.5                       # to pixel coords, flipping y again
