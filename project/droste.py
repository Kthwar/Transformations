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

MODEL C -- DRAWN REGION (polygons).  Two polygons can be drawn on the picture:

    OUTPUT FRAME  P  ("the screen")   where the inner copy appears.
    INPUT  FRAME  Q  ("the cut-out")  which PART of the picture is shown inside P.
                                      Left empty (< 3 points) = the whole picture.

  A pixel inside P is pulled back to a point of Q (i.e. to picture coordinates), repeated up
  to `depth` times, so a Q that contains P gives the endless recursion:
    * 4 points = the corners of a quadrilateral in the order TL, TR, BR, BL.  If both P and Q
      have 4 points, a homography (projective map, handles tilted / perspective frames) sends
      P exactly onto Q.  (Not conformal, but exact for rectangles & perspective.)
    * any other point count (>= 3) = free shape.  A free P is replaced by the largest rectangle
      with Q's aspect ratio that fits in P's bounding box (so the copy is not stretched) and the
      copy is shown only inside the polygon.  A free Q is replaced by its bounding box, and
      afterwards everything that falls outside the polygon Q is cut away (transparent).
  psi rotates the inner copy in B and C as well (after the pull-back; in C about Q's centre).
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class DrosteParams:
    enabled: bool = False
    model: str = "annulus"        # "annulus" | "disc" | "polygon"
    center: complex = 0j          # centre of the Droste spiral / disc (models A, B)
    r1: float = 0.2               # inner radius (A)
    r2: float = 0.8               # outer radius (A)
    rho: float = 0.4              # radius of the inner disc (B)
    psi: float = 0.0              # rotation of the inner copy (radians)
    depth: int = 6                # maximum number of recursion levels
    frame_radius: float = 1.0     # inscribed-disc radius of the picture, local units
    aspect: float = 1.0           # h = H/W
    # Output frame P: polygon (picture-local complex points) where the copy appears (model C)
    poly: np.ndarray = field(default_factory=lambda: np.zeros(0, complex))
    # Input frame Q: polygon selecting which part of the picture is shown inside P (model C).
    # Fewer than 3 points = use the whole picture.
    in_poly: np.ndarray = field(default_factory=lambda: np.zeros(0, complex))


# ---------------------------------------------------------------------------------------
# MODEL A: annulus / logarithmic map
# ---------------------------------------------------------------------------------------
def droste_annulus(z, dp):
    r2 = max(dp.r2, dp.r1 * 1.05)                            # keep r2 > r1, otherwise L would be <= 0
    L = np.log(r2 / dp.r1)                                   # width of one copy in the log plane
    tau = L - 1j * dp.psi                                    # the single self-similarity translation
    w = np.log(z - dp.center)                                # step 1: go to the log plane
    n = np.floor((w.real - np.log(dp.r1)) / L)               # step 2: copy level (negative = inside)
    w2 = np.where((n < 0) & (n >= -dp.depth), w - n * tau, w)  # undo n translations (limited depth)
    return dp.center + np.exp(w2)                            # step 3: back to the picture plane


# ---------------------------------------------------------------------------------------
# MODEL B: disc
# ---------------------------------------------------------------------------------------
def droste_disc(z, dp):
    s, rot = dp.rho / dp.frame_radius, np.exp(-1j * dp.psi)  # shrink factor and (inverse) rotation
    z = z.copy()
    for _ in range(dp.depth):
        inside = np.abs(z - dp.center) < dp.rho              # NaN compares False -> untouched
        if not inside.any():                                 # nothing left to pull back
            break
        z[inside] = (z[inside] - dp.center) / s * rot        # T^{-1}, then rotate the copy
    return z


# ---------------------------------------------------------------------------------------
# helpers for MODEL C
# ---------------------------------------------------------------------------------------
def in_polygon(z, P):
    """Even-odd rule: a ray towards +x crosses an odd number of edges iff z is inside.
    Works on a whole array of points at once; P is a complex array of vertices."""
    x, y, inside, j = z.real, z.imag, np.zeros(z.shape, bool), len(P) - 1
    for i in range(len(P)):                                  # edge from vertex j to vertex i
        # the edge straddles the horizontal line through the point AND the point lies to the
        # left of the crossing -> the ray towards +x hits this edge
        cross = ((P[i].imag > y) != (P[j].imag > y)) & \
                (x < (P[j].real - P[i].real) * (y - P[i].imag) / (P[j].imag - P[i].imag + 1e-300) + P[i].real)
        inside ^= cross                                      # each crossing flips inside/outside
        j = i
    return inside


def homography(src, dst):
    """3x3 projective map sending 4 points src -> dst (complex numbers as x+iy); direct linear solve.
    A homography is X = (h00 x + h01 y + h02)/(h20 x + h21 y + 1), Y = (h10 x + h11 y + h12)/(...);
    each point pair gives two linear equations in the 8 unknowns h00..h21 -> 8x8 system."""
    A, b = [], []
    for s, d in zip(src, dst):
        x, y, X, Y = s.real, s.imag, d.real, d.imag
        A += [[x, y, 1, 0, 0, 0, -X * x, -X * y], [0, 0, 0, x, y, 1, -Y * x, -Y * y]]
        b += [X, Y]
    return np.append(np.linalg.solve(A, b), 1).reshape(3, 3)   # append h22 = 1


def apply_homography(Hm, z):
    """Apply the 3x3 matrix to complex points (x+iy), including the perspective division."""
    x, y = z.real, z.imag
    d = Hm[2, 0] * x + Hm[2, 1] * y + 1
    return ((Hm[0, 0] * x + Hm[0, 1] * y + Hm[0, 2]) + 1j * (Hm[1, 0] * x + Hm[1, 1] * y + Hm[1, 2])) / d


def _bbox_rect(P):
    """Axis-aligned bounding box of the points P as 4 corners in the order TL, TR, BR, BL."""
    x0, x1, y0, y1 = P.real.min(), P.real.max(), P.imag.min(), P.imag.max()
    return np.array([complex(x0, y1), complex(x1, y1), complex(x1, y0), complex(x0, y0)])


def _quad_aspect(q):
    """width / height of a quadrilateral given as TL, TR, BR, BL (average of opposite sides)."""
    width = (abs(q[1] - q[0]) + abs(q[2] - q[3])) / 2        # top and bottom edge
    height = (abs(q[3] - q[0]) + abs(q[2] - q[1])) / 2       # left and right edge
    return width / max(height, 1e-12)


def _fit_rect(P, aspect):
    """Largest rectangle with width/height = `aspect` that fits, centred, in the bounding box of P.
    Corners returned as TL, TR, BR, BL."""
    x0, x1, y0, y1 = P.real.min(), P.real.max(), P.imag.min(), P.imag.max()
    bw, bh = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    aspect = max(aspect, 1e-12)
    if bw / max(bh, 1e-12) > aspect:      # box is "wider" than the rectangle -> height is the limit
        hh = bh / 2; hw = hh * aspect
    else:                                 # box is "taller" -> width is the limit
        hw = bw / 2; hh = hw / aspect
    return np.array([complex(cx - hw, cy + hh), complex(cx + hw, cy + hh),
                     complex(cx + hw, cy - hh), complex(cx - hw, cy - hh)])


def _input_frame(dp):
    """Resolve the INPUT frame Q into (quad, clip):
         quad = 4 corners TL,TR,BR,BL of the region of the picture that gets shown,
         clip = polygon the shown content must stay inside (None = no cropping needed)."""
    Q, h = dp.in_poly, dp.aspect
    if len(Q) < 3:      # nothing (usable) drawn -> the whole picture rectangle, as before
        return np.array([-1 + 1j * h, 1 + 1j * h, 1 - 1j * h, -1 - 1j * h]), None
    if len(Q) == 4:     # a quadrilateral: used exactly (perspective allowed)
        return Q, Q
    return _bbox_rect(Q), Q    # free shape: use its bounding box, then crop to the polygon


def droste_polygon(z, dp):
    P = dp.poly                                              # OUTPUT frame: where the copy goes
    if len(P) < 3:                                           # not drawn yet -> no effect
        return z
    quad_in, clip = _input_frame(dp)                         # INPUT frame: what the copy shows
    # Quadrilateral of the output side.  4 clicks: used as is.  Free shape: a rectangle with the
    # input frame's aspect ratio, fitted into the shape's bounding box (copy is not stretched).
    quad_out = P if len(P) == 4 else _fit_rect(P, _quad_aspect(quad_in))
    try:
        Hm = homography(quad_out, quad_in)                   # maps output frame -> input frame
    except np.linalg.LinAlgError:
        raise ValueError("A drawn frame is degenerate (collinear or repeated points).")
    centre = quad_in.mean()                                  # rotation pivot = centre of the input frame
    rot = np.exp(-1j * dp.psi)
    # Crop to Q when it is a free shape, or when a rotation could pull in content from outside
    # Q.  (A plain 4-point Q with psi = 0 maps exactly onto itself, so no test is needed there.)
    use_clip = clip is not None and (len(clip) != 4 or dp.psi != 0)
    void = complex(np.nan, np.nan)                           # NaN point = nothing there (transparent)

    def back(q):
        """One pull-back step: point in the output frame -> point of the picture."""
        q = apply_homography(Hm, q)                          # output frame -> input frame
        q = centre + (q - centre) * rot                      # rotate the copy about the input centre
        if use_clip:
            q = np.where(in_polygon(q, clip), q, void)       # outside the cut-out -> transparent
        return q

    z = z.copy()
    for _ in range(dp.depth):                                # each pass = one more recursion level
        inside = in_polygon(z, P)                            # NaN points count as "outside"
        if not inside.any():                                 # no pixel is inside the frame any more
            break
        z[inside] = back(z[inside])
    return z
  
def droste_polygon_fixed_points(dp):
    P = dp.poly
    if len(P) < 3:
        return []
    quad_in, _ = _input_frame(dp)
    quad_out = P if len(P) == 4 else _fit_rect(P, _quad_aspect(quad_in))
    Hm = homography(quad_out, quad_in)            # same matrix the renderer uses

    cx, cy = quad_in.mean().real, quad_in.mean().imag
    co, si = np.cos(dp.psi), np.sin(dp.psi)
    # multiplication by e^{-i psi} about the input centre, as a 3x3 matrix
    R = np.array([[ co, si, cx - ( co * cx + si * cy)],
                  [-si, co, cy - (-si * cx + co * cy)],
                  [0.0, 0.0, 1.0]])
    M = R @ Hm                                    # = back(): homography, then rotation

    vals, vecs = np.linalg.eig(M)
    pts = []
    for k in range(3):
        if abs(vals[k].imag) > 1e-9:              # complex pair -> no real fixed point
            continue
        v = vecs[:, k]
        if abs(v[2]) < 1e-12 * np.linalg.norm(v): # w = 0 -> point at infinity
            continue
        v = v / v[2]
        pts.append(complex(v[0].real, v[1].real))
    return pts

def droste_pullback(z, dp):
    """Dispatch to the chosen model."""
    centres = droste_polygon_fixed_points(dp)
    for sexo in centres:
        print(f"Fixed point: x={2*sexo.real}, y={2*sexo.imag}")
        print("Inside output frame:", in_polygon(np.array([sexo]), dp.poly)[0])
    return {"annulus": droste_annulus, "disc": droste_disc, "polygon": droste_polygon}[dp.model](z, dp)