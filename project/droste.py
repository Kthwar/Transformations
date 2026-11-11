"""
droste.py -- the Droste engine, written as a PULL-BACK: given a point z of the picture plane
it returns the point z' whose colour should be shown.  Coordinates are "picture-local":
the picture spans x in [-1, 1] (radius 1 = half the picture's width), y in [-h, h], h = H/W.
Every model assumes the picture contains a smaller copy of itself (rotated by psi) and
replaces whatever is inside by an enlarged copy of the picture, again and again.

MODEL A -- ANNULUS (conformal log map).  Inner copy = scaled by 1/R (R = r2/r1), rotated by psi.
  1. w = ln(z - c) = ln r + i*theta.   Scaling by R is the translation w -> w + L, L = ln R;
     rotating by psi is w -> w + i*psi.  The whole self-similarity is ONE translation
     tau = L - i*psi of the log plane.
  2. Level n = floor((Re w - ln r1)/L) says how many copies deep we are.  Undo them:
     w' = w - n*tau   (only for -depth <= n < 0; deeper points keep the original content).
  3. z' = c + exp(w').   exp is conformal, so each copy is an undistorted similar copy.
  (No branch-cut seam here: ln's arg jumps by 2*pi*i across the cut, and exp(w') ignores that.)

MODEL B -- DISC.  T(z) = c + s*e^{i*psi}*z shrinks the picture's inscribed disc (radius
  frame_radius) into the disc (centre c, radius rho), s = rho/frame_radius.  A pixel inside the
  disc is pulled back with T^{-1}; repeat up to `depth` times.

MODEL C -- DRAWN REGION (polygon).  A pixel inside the polygon P is pulled back to picture
  coordinates, repeated up to `depth` times:
    * 4 points = the corners of the inner copy in the order TL, TR, BR, BL.  The map picture
      rectangle -> quad is a homography (projective map, handles tilted / perspective frames);
      we apply its inverse.   (Not conformal, but exact for rectangles & perspective.)
    * other point counts = free shape: the copy is the picture scaled to fit the polygon's
      bounding box, shown only inside the polygon.
  psi rotates the inner copy in B and C as well (after the pull-back).
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class DrosteParams:
    enabled: bool = False
    model: str = "annulus"        # "annulus" | "disc" | "polygon"
    center: complex = 0j
    r1: float = 0.2
    r2: float = 0.8
    rho: float = 0.4
    psi: float = 0.0              # rotation of the inner copy (radians)
    depth: int = 6
    frame_radius: float = 1.0     # inscribed-disc radius of the picture, local units
    aspect: float = 1.0           # h = H/W
    poly: np.ndarray = field(default_factory=lambda: np.zeros(0, complex))


def droste_annulus(z, dp):
    r2 = max(dp.r2, dp.r1 * 1.05)
    L = np.log(r2 / dp.r1)
    tau = L - 1j * dp.psi                                    # the single self-similarity translation
    w = np.log(z - dp.center)                                # step 1
    n = np.floor((w.real - np.log(dp.r1)) / L)               # step 2: copy level
    w2 = np.where((n < 0) & (n >= -dp.depth), w - n * tau, w)
    return dp.center + np.exp(w2)                            # step 3


def droste_disc(z, dp):
    s, rot = dp.rho / dp.frame_radius, np.exp(-1j * dp.psi)
    z = z.copy()
    for _ in range(dp.depth):
        inside = np.abs(z - dp.center) < dp.rho              # NaN compares False -> untouched
        if not inside.any():
            break
        z[inside] = (z[inside] - dp.center) / s * rot        # T^{-1}, then rotate the copy
    return z


def in_polygon(z, P):
    """Even-odd rule: a ray towards +x crosses an odd number of edges iff z is inside."""
    x, y, inside, j = z.real, z.imag, np.zeros(z.shape, bool), len(P) - 1
    for i in range(len(P)):
        cross = ((P[i].imag > y) != (P[j].imag > y)) & \
                (x < (P[j].real - P[i].real) * (y - P[i].imag) / (P[j].imag - P[i].imag + 1e-300) + P[i].real)
        inside ^= cross
        j = i
    return inside


def homography(src, dst):
    """3x3 projective map sending 4 points src -> dst (complex numbers as x+iy); direct linear solve."""
    A, b = [], []
    for s, d in zip(src, dst):
        x, y, X, Y = s.real, s.imag, d.real, d.imag
        A += [[x, y, 1, 0, 0, 0, -X * x, -X * y], [0, 0, 0, x, y, 1, -Y * x, -Y * y]]
        b += [X, Y]
    return np.append(np.linalg.solve(A, b), 1).reshape(3, 3)


def apply_homography(Hm, z):
    x, y = z.real, z.imag
    d = Hm[2, 0] * x + Hm[2, 1] * y + 1
    return ((Hm[0, 0] * x + Hm[0, 1] * y + Hm[0, 2]) + 1j * (Hm[1, 0] * x + Hm[1, 1] * y + Hm[1, 2])) / d


def droste_polygon(z, dp):
    P, h, rot = dp.poly, dp.aspect, np.exp(-1j * dp.psi)
    if len(P) < 3:
        return z
    if len(P) == 4:
        Hm = homography(P, np.array([-1 + 1j * h, 1 + 1j * h, 1 - 1j * h, -1 - 1j * h]))
        back = lambda q: apply_homography(Hm, q) * rot
    else:
        c0 = complex((P.real.min() + P.real.max()) / 2, (P.imag.min() + P.imag.max()) / 2)
        s = max(min((P.real.max() - P.real.min()) / 2, (P.imag.max() - P.imag.min()) / 2 / h), 1e-6)
        back = lambda q: (q - c0) / s * rot
    z = z.copy()
    for _ in range(dp.depth):
        inside = in_polygon(z, P)
        if not inside.any():
            break
        z[inside] = back(z[inside])
    return z


def droste_pullback(z, dp):
    return {"annulus": droste_annulus, "disc": droste_disc, "polygon": droste_polygon}[dp.model](z, dp)
