"""
droste.py -- the Droste engine, written as a PULL-BACK: given a point z of the picture
plane, return the point z' whose colour we should show.  Coordinates are "picture-local":
the picture spans x in [-1, 1] (so a radius of 1 = half the picture's width).

MODEL A: ANNULUS / ESCHER (conformal log-spiral tiling)
  Assume the picture contains a smaller copy of itself between radii r1 < r2 around the
  centre c (ratio R = r2/r1, L = ln R).
  1. w = ln(z - c) = ln r + i*theta.  Scaling z by R is now a TRANSLATION of w by L.
  2. w' = (alpha + i*beta) * w.  Multiplying by a complex number = scale + rotate in the
     log plane; beta != 0 makes the copies spiral.   [u + iv = w']
  3. Self-similarity means u is only meaningful modulo L, so wrap u into [ln r1, ln r2):
        u' = u - n*L,  n = floor((u - ln r1)/L)
  4. z' = c + exp(u' + i*v).  exp is conformal, so every tile is an undistorted
     (angle-preserving) copy.
  SEAMLESS condition: going once around (w -> w + 2*pi*i) shifts w' by 2*pi*i*(alpha+i*beta)
  = 2*pi*i*alpha - 2*pi*beta.  exp ignores 2*pi*i*alpha only if alpha is an integer, and the
  wrap ignores -2*pi*beta only if it's a multiple of L, i.e. beta = p*L/(2*pi), p = number
  of spiral arms.  Otherwise a visible seam appears along the branch cut at theta0.
  `depth` limits how many nested levels (n = -1 ... -depth) get the recursion; deeper
  points show the picture unchanged.

MODEL B: NESTED COPY (iterated similarity)
  The picture's inscribed disc is shrunk into a disc (centre c, radius rho): that is the
  similarity T(z) = c + s*e^{i*psi}*z with s = rho/frame_radius.  For a pixel inside the
  disc we apply T^{-1} (blow it back up to full picture size) and repeat up to `depth`
  times, until the point lands outside the disc (= ordinary picture content).
"""
from dataclasses import dataclass
import numpy as np
from transforms import arg_cut


@dataclass
class DrosteParams:
    enabled: bool = False
    model: str = "annulus"        # "annulus" or "nested"
    center: complex = 0j
    r1: float = 0.2
    r2: float = 0.8
    alpha: float = 1.0
    beta: float = 0.0
    seamless: bool = True         # snap alpha to an integer, beta = arms*L/(2*pi)
    arms: int = 1
    depth: int = 6
    rho: float = 0.4              # nested model: radius of the inner disc
    psi: float = 0.0              # nested model: rotation of the inner copy (radians)
    frame_radius: float = 1.0     # radius of the picture's inscribed disc, in local units
    theta0: float = -np.pi        # branch cut of the ln(z - c) used by the annulus model


def droste_annulus(z, dp):
    r2 = max(dp.r2, dp.r1 * 1.05)
    L = np.log(r2 / dp.r1)
    if dp.seamless:
        a, b = max(1.0, round(dp.alpha)), dp.arms * L / (2 * np.pi)
    else:
        a, b = dp.alpha, dp.beta
    d = z - dp.center
    w = np.log(np.abs(d)) + 1j * arg_cut(d, dp.theta0)      # step 1: log coordinates
    wp = (a + 1j * b) * w                                    # step 2: scale + rotate
    u, v = wp.real, wp.imag
    n = np.floor((u - np.log(dp.r1)) / L)                    # which level of the tiling?
    wrap = (n < 0) & (n >= -dp.depth)                        # recurse only `depth` levels
    u2 = np.where(wrap, u - n * L, u)                        # step 3: u modulo L
    return dp.center + np.exp(u2 + 1j * v)                   # step 4: back with exp


def droste_nested(z, dp):
    s = dp.rho / dp.frame_radius
    rot = np.exp(-1j * dp.psi)
    z = z.copy()
    for _ in range(dp.depth):
        inside = np.abs(z - dp.center) < dp.rho              # NaN compares False -> untouched
        if not inside.any():
            break
        z[inside] = (z[inside] - dp.center) * rot / s        # T^{-1}: enlarge back to picture size
    return z


def droste_pullback(z, dp):
    return droste_annulus(z, dp) if dp.model == "annulus" else droste_nested(z, dp)
