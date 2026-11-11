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
    """Pixel coordinates (float arrays ok) -> z = x + iy."""
    u = (px + 0.5) / W - 0.5                # horizontal offset in [-1/2, 1/2]
    v = 0.5 - (py + 0.5) / H                # vertical offset, flipped so +v is up
    local = region.width * (u + 1j * v * H / W)   # plane units; height = width*H/W
    return region.center + local * np.exp(1j * region.rotation)


def complex_to_pixel(z, W, H, region):
    """Inverse of pixel_to_complex: z -> fractional pixel coords (px, py)."""
    local = (z - region.center) * np.exp(-1j * region.rotation)   # undo the rotation
    u = local.real / region.width + 0.5
    v = local.imag / (region.width * H / W)
    return u * W - 0.5, (0.5 - v) * H - 0.5
