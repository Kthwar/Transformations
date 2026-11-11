"""
transforms.py -- the maps w = f(z) and their (multi-valued) inverses.

WHY INVERSES?  To paint the output w-plane without holes we go backwards: for each
output pixel w, find z = f^{-1}(w) and read the source picture at z.  Every output
pixel gets exactly one lookup, so no void gaps / pixelation (forward mapping can't
promise that -- see render.py).

BRANCH CUTS.  Several inverses are multi-valued:
  * ln w  = ln|w| + i(arg w + 2*pi*k)    -> inverse of exp, infinitely many sheets k
  * w^(1/n)                              -> n sheets for integer n
  * arcsin / arccos / arctan             -> sheets shifted by pi*k or 2*pi*k
A *branch cut* is the ray where we choose to jump between sheets so the function
is single-valued.  We cut along the ray at angle theta0 (default -pi, i.e. the
negative real axis) by forcing arg into [theta0, theta0 + 2*pi).

  cuts ON : use only the principal sheet k=0 -> the picture appears ONCE (a strip /
            wedge, "frieze").
  cuts OFF: all sheets k=0,+-1,+-2,... are composited -> the strip repeats and
            tiles the plane ("pattern").
"""
from dataclasses import dataclass
from typing import Callable
import numpy as np

TWO_PI = 2 * np.pi


@dataclass
class TParams:
    theta0: float = -np.pi   # branch cut angle (radians)
    cuts_on: bool = True     # True: principal sheet only
    n: float = 2.0           # exponent for z^n
    n_sheets: int = 5        # how many sheets to composite when cuts are OFF
    seed_center: complex = 0j  # Newton start points (custom f(z) only): centred on the picture
    seed_radius: float = 2.0


def arg_cut(z, theta0):
    """arg z, shifted into [theta0, theta0+2*pi): the cut is the ray at angle theta0."""
    return theta0 + np.mod(np.angle(z) - theta0, TWO_PI)


def sheet_indices(p):
    """[0] with cuts on; otherwise 0, 1, -1, 2, -2, ... (nearest sheets first)."""
    if p.cuts_on:
        return [0]
    ks, m = [0], 1
    while len(ks) < p.n_sheets:
        ks += [m, -m]
        m += 1
    return ks[:p.n_sheets]


@dataclass
class Transform:
    name: str
    forward: Callable          # (z, p) -> list of w arrays (one per sheet)
    inverse: Callable          # (w, p) -> list of (z, ok_mask), nearest sheet first
    default_half_width: float  # sensible output window for this map


def _id_f(z, p): return [z]
def _id_i(w, p): return [(w, np.ones(w.shape, bool))]


# --- w = e^z.  e^{x+iy} = e^x (cos y + i sin y): vertical lines -> circles, horizontal
#     lines -> rays.  Periodic in y with period 2*pi, so the inverse is ln w + 2*pi*i*k.
def _exp_f(z, p): return [np.exp(z)]
def _exp_i(w, p):
    lg, th = np.log(np.abs(w)), arg_cut(w, p.theta0)
    return [(lg + 1j * (th + TWO_PI * k), np.isfinite(lg)) for k in sheet_indices(p)]


# --- w = ln z = ln|z| + i*arg z.  Polar -> Cartesian: circles -> vertical lines, rays ->
#     horizontal lines.  Forward is multivalued (arg z + 2*pi*k).  The inverse z = e^w is
#     single-valued and 2*pi*i-periodic, so "cuts on" means: only show the strip
#     theta0 <= Im w < theta0 + 2*pi (one trip around the origin).
def _log_f(z, p):
    lg, th = np.log(np.abs(z)), arg_cut(z, p.theta0)
    return [lg + 1j * (th + TWO_PI * k) for k in sheet_indices(p)]
def _log_i(w, p):
    ok = np.ones(w.shape, bool) if not p.cuts_on else (w.imag >= p.theta0) & (w.imag < p.theta0 + TWO_PI)
    return [(np.exp(w), ok)]


# --- w = z^n = exp(n * ln z): angles are multiplied by n, so the plane wraps n times.
#     Inverse: z = |w|^(1/n) * exp(i(arg w + 2*pi*k)/n), k = 0..n-1 (integer n).
def _pow_f(z, p): return [np.exp(p.n * (np.log(np.abs(z)) + 1j * arg_cut(z, p.theta0)))]
def _pow_i(w, p):
    n = p.n if p.n != 0 else 1.0
    ks = sheet_indices(p)
    if float(n).is_integer():
        ks = ks[:int(abs(n))]          # integer n has only |n| distinct roots
    mag, th = np.abs(w) ** (1.0 / n), arg_cut(w, p.theta0)
    return [(mag * np.exp(1j * (th + TWO_PI * k) / n), np.isfinite(mag)) for k in ks]


# --- Trig maps.  sin/cos/tan use numpy's principal arcsin/arccos/arctan (their own cuts lie on
#     the real axis outside [-1,1] or the imaginary axis), so theta0 doesn't move them; the
#     cuts toggle still selects principal sheet vs all sheets.
#     sin z = sin(z + 2*pi k) = sin(pi - z):  z = (-1)^k arcsin w + pi*k
#     cos z = cos(-z):                        z = +-arccos w + 2*pi*k
#     tan z has period pi:                    z = arctan w + pi*k
def _sin_i(w, p):
    a = np.arcsin(w + 0j)
    return [(((-1) ** k) * a + np.pi * k, np.isfinite(a)) for k in sheet_indices(p)]
def _cos_i(w, p):
    a, out = np.arccos(w + 0j), []
    ok = np.isfinite(a)
    for k in sheet_indices(p):
        out += [(a + TWO_PI * k, ok), (-a + TWO_PI * k, ok)]
    return out[:1] if p.cuts_on else out
def _tan_i(w, p):
    a = np.arctan(w + 0j)
    return [(a + np.pi * k, np.isfinite(a)) for k in sheet_indices(p)]


TRANSFORMS = {t.name: t for t in [
    Transform("identity", _id_f, _id_i, 2.0),
    Transform("exp(z)", _exp_f, _exp_i, 8.0),
    Transform("ln(z)", _log_f, _log_i, 4.0),
    Transform("z^n", _pow_f, _pow_i, 5.0),
    Transform("sin(z)", lambda z, p: [np.sin(z)], _sin_i, 3.0),
    Transform("cos(z)", lambda z, p: [np.cos(z)], _cos_i, 3.0),
    Transform("tan(z)", lambda z, p: [np.tan(z)], _tan_i, 3.0),
]}


# =====================================================================================
# CUSTOM FUNCTIONS  (typed by the user; variable is z, or w for an inverse; k = sheet index)
# =====================================================================================
_NS = {n: getattr(np, n) for n in ["sin", "cos", "tan", "exp", "log", "sqrt", "sinh", "cosh", "tanh",
                                   "arcsin", "arccos", "arctan", "arcsinh", "arccosh", "arctanh",
                                   "conj", "abs", "real", "imag", "angle", "power"]}
_NS.update(pi=np.pi, e=np.e, i=1j, j=1j, ln=np.log, asin=np.arcsin, acos=np.arccos, atan=np.arctan)


def _compile(expr, var):
    """Turn text like 'z^2 + 1' into a vectorised function of `var` (and the sheet index k).
    eval() with no builtins -- fine for a local tool where you type your own formulas."""
    code = compile(expr.replace("^", "**"), "<custom>", "eval")
    def fn(x, k=0):
        ns = dict(_NS); ns[var] = x; ns["k"] = k
        return eval(code, {"__builtins__": {}}, ns) + 0 * x      # "+ 0*x" broadcasts constants
    return fn


_SEEDS = [0, 1, -1, 1j, -1j, 1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]   # 3x3 grid of Newton start points


def _newton(f, w, z0, iters=40):
    """Solve f(z) = w for every pixel at once.  Newton: z <- z - (f(z)-w)/f'(z), where for an
    analytic f the derivative is the complex difference quotient f'(z) ~ (f(z+h)-f(z))/h.
    Steps are capped at length 1 for stability.  Pixels whose residual stays large are
    flagged invalid (they become transparent) instead of showing garbage."""
    z, h = z0.copy(), 1e-6
    for _ in range(iters):
        fz = f(z)
        step = (fz - w) / ((f(z + h) - fz) / h)
        step = np.where(np.abs(step) > 1, step / np.abs(step), step)
        z = z - np.where(np.isfinite(step), step, 0)
    ok = np.isfinite(z) & (np.abs(f(z) - w) < 1e-6 * (1 + np.abs(w)))
    return z, ok


def make_custom(expr, mode):
    """mode 'inverse': expr is f^-1(w) (may use k for multi-valued sheets) -> used directly.
       mode 'forward': expr is f(z) -> inverted numerically by Newton from several start points;
       each start point can land on a different preimage (= a different 'sheet').  Cuts ON uses
       one start point, cuts OFF uses `n_sheets` of them.  Root-finding can fail or pick odd
       roots near critical points of f -- supplying the true inverse is more reliable."""
    f = _compile(expr, "w" if mode == "inverse" else "z")
    f(np.array([0.5 + 0.5j]))                     # fail early on typos / unknown names
    if mode == "inverse":
        def inv(w, p):
            for k in sheet_indices(p):
                z = f(w, k)
                yield z, np.isfinite(z)
        def fwd(z, p):
            raise NotImplementedError("Forward demo needs f(z); you supplied only the inverse.")
    else:
        def fwd(z, p):
            return [f(z)]
        def inv(w, p):
            for s in _SEEDS[:1 if p.cuts_on else p.n_sheets]:
                yield _newton(f, w, np.full(w.shape, p.seed_center + 0.6 * p.seed_radius * s, complex))
    return Transform("custom", fwd, inv, 4.0)
