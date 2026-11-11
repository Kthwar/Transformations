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

CUSTOM FUNCTIONS.  At the bottom of this file the user can type their own formula.
Besides the variable (z or w) and the sheet index k, the formula may use four
free CONSTANTS  a, b, c, d.  The UI gives each one a slider, so a formula such as
    a*z^2 + b*z + c
can be explored interactively (the slider values travel inside TParams.consts).
"""
from dataclasses import dataclass, field
from typing import Callable
import numpy as np

TWO_PI = 2 * np.pi

# Names of the constants a user can use inside a custom formula, and the value each
# one has before its slider is touched.  (app.py builds one slider per name.)
CONST_NAMES = ("a", "b", "c", "d")
DEFAULT_CONSTS = {"a": 1.0, "b": 0.0, "c": 0.0, "d": 0.0}


@dataclass
class TParams:
    """Bundle of every UI setting a transform may need (passed to every f / f^-1)."""
    theta0: float = -np.pi   # branch cut angle (radians)
    cuts_on: bool = True     # True: principal sheet only
    n: float = 2.0           # exponent for z^n
    n_sheets: int = 5        # how many sheets to composite when cuts are OFF
    seed_center: complex = 0j  # Newton start points (custom f(z) only): centred on the picture
    seed_radius: float = 2.0
    # Values of the custom-formula constants a, b, c, d (read from the sliders each render)
    consts: dict = field(default_factory=lambda: dict(DEFAULT_CONSTS))


def arg_cut(z, theta0):
    """arg z, shifted into [theta0, theta0+2*pi): the cut is the ray at angle theta0.
    np.angle gives an angle in (-pi, pi]; subtracting theta0, taking mod 2*pi and adding
    theta0 back slides that value into the window that starts at theta0."""
    return theta0 + np.mod(np.angle(z) - theta0, TWO_PI)


def sheet_indices(p):
    """[0] with cuts on; otherwise 0, 1, -1, 2, -2, ... (nearest sheets first)."""
    if p.cuts_on:
        return [0]
    ks, m = [0], 1
    while len(ks) < p.n_sheets:       # add the pair +m, -m until we have enough sheets
        ks += [m, -m]
        m += 1
    return ks[:p.n_sheets]            # the pair may overshoot by one -> trim


@dataclass
class Transform:
    """One complex map, described by both directions."""
    name: str
    forward: Callable          # (z, p) -> list of w arrays (one per sheet)
    inverse: Callable          # (w, p) -> list of (z, ok_mask), nearest sheet first
    default_half_width: float  # sensible output window for this map


# Identity: nothing happens (useful with Droste alone).
def _id_f(z, p): return [z]
def _id_i(w, p): return [(w, np.ones(w.shape, bool))]


# --- w = e^z.  e^{x+iy} = e^x (cos y + i sin y): vertical lines -> circles, horizontal
#     lines -> rays.  Periodic in y with period 2*pi, so the inverse is ln w + 2*pi*i*k.
def _exp_f(z, p): return [np.exp(z)]
def _exp_i(w, p):
    lg, th = np.log(np.abs(w)), arg_cut(w, p.theta0)     # ln|w| and the (cut) argument
    # one (z, ok) pair per sheet k; ok masks out w = 0 where ln|w| = -inf
    return [(lg + 1j * (th + TWO_PI * k), np.isfinite(lg)) for k in sheet_indices(p)]


# --- w = ln z = ln|z| + i*arg z.  Polar -> Cartesian: circles -> vertical lines, rays ->
#     horizontal lines.  Forward is multivalued (arg z + 2*pi*k).  The inverse z = e^w is
#     single-valued and 2*pi*i-periodic, so "cuts on" means: only show the strip
#     theta0 <= Im w < theta0 + 2*pi (one trip around the origin).
def _log_f(z, p):
    lg, th = np.log(np.abs(z)), arg_cut(z, p.theta0)
    return [lg + 1j * (th + TWO_PI * k) for k in sheet_indices(p)]
def _log_i(w, p):
    # cuts on: only the strip of the w-plane that corresponds to one turn around the origin
    ok = np.ones(w.shape, bool) if not p.cuts_on else (w.imag >= p.theta0) & (w.imag < p.theta0 + TWO_PI)
    return [(np.exp(w), ok)]


# --- w = z^n = exp(n * ln z): angles are multiplied by n, so the plane wraps n times.
#     Inverse: z = |w|^(1/n) * exp(i(arg w + 2*pi*k)/n), k = 0..n-1 (integer n).
def _pow_f(z, p): return [np.exp(p.n * (np.log(np.abs(z)) + 1j * arg_cut(z, p.theta0)))]
def _pow_i(w, p):
    n = p.n if p.n != 0 else 1.0          # n = 0 would divide by zero below
    ks = sheet_indices(p)
    if float(n).is_integer():
        ks = ks[:int(abs(n))]          # integer n has only |n| distinct roots
    mag, th = np.abs(w) ** (1.0 / n), arg_cut(w, p.theta0)   # modulus and argument of the root
    return [(mag * np.exp(1j * (th + TWO_PI * k) / n), np.isfinite(mag)) for k in ks]


# --- Trig maps.  sin/cos/tan use numpy's principal arcsin/arccos/arctan (their own cuts lie on
#     the real axis outside [-1,1] or the imaginary axis), so theta0 doesn't move them; the
#     cuts toggle still selects principal sheet vs all sheets.
#     sin z = sin(z + 2*pi k) = sin(pi - z):  z = (-1)^k arcsin w + pi*k
#     cos z = cos(-z):                        z = +-arccos w + 2*pi*k
#     tan z has period pi:                    z = arctan w + pi*k
def _sin_i(w, p):
    a = np.arcsin(w + 0j)             # "+ 0j" forces the complex branch (no NaN outside [-1,1])
    return [(((-1) ** k) * a + np.pi * k, np.isfinite(a)) for k in sheet_indices(p)]
def _cos_i(w, p):
    a, out = np.arccos(w + 0j), []
    ok = np.isfinite(a)
    for k in sheet_indices(p):
        out += [(a + TWO_PI * k, ok), (-a + TWO_PI * k, ok)]   # both signs: cos is even
    return out[:1] if p.cuts_on else out
def _tan_i(w, p):
    a = np.arctan(w + 0j)
    return [(a + np.pi * k, np.isfinite(a)) for k in sheet_indices(p)]

# Registry shown in the UI's drop-down list (name -> Transform).
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
# CUSTOM FUNCTIONS  (typed by the user; variable is z, or w for an inverse; k = sheet index;
#                    a, b, c, d = constants controlled by sliders)
# =====================================================================================
# Names the typed formula is allowed to use.  Everything is a numpy function so it works on
# whole arrays of complex numbers at once (vectorised -> fast).
_NS = {n: getattr(np, n) for n in ["sin", "cos", "tan", "exp", "log", "sqrt", "sinh", "cosh", "tanh",
                                   "arcsin", "arccos", "arctan", "arcsinh", "arccosh", "arctanh",
                                   "conj", "abs", "real", "imag", "angle", "power"]}
# Constants and short aliases: pi, e, the imaginary unit (i or j), ln = log, asin/acos/atan.
_NS.update(pi=np.pi, e=np.e, i=1j, j=1j, ln=np.log, asin=np.arcsin, acos=np.arccos, atan=np.arctan)


def _compile(expr, var):
    """Turn text like 'a*z^2 + 1' into a vectorised function fn(x, k, consts) of `var`.
    `k` is the sheet index and `consts` is the {a, b, c, d} dictionary from the sliders.
    eval() with no builtins -- fine for a local tool where you type your own formulas."""
    code = compile(expr.replace("^", "**"), "<custom>", "eval")   # allow ^ for powers; parse once
    def fn(x, k=0, consts=None):
        ns = dict(_NS)                      # fresh namespace for every call
        ns.update(DEFAULT_CONSTS)           # a, b, c, d start at their defaults ...
        if consts:
            ns.update(consts)               # ... and are overridden by the current slider values
        ns[var] = x                         # the variable (z or w)
        ns["k"] = k                         # sheet index (only meaningful for inverses)
        return eval(code, {"__builtins__": __builtins__}, ns) + 0 * x
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
        step = (fz - w) / ((f(z + h) - fz) / h)                       # Newton step
        step = np.where(np.abs(step) > 1, step / np.abs(step), step)  # cap the length at 1
        z = z - np.where(np.isfinite(step), step, 0)                  # ignore NaN/inf steps
    # converged only if f(z) really equals w (relative tolerance)
    ok = np.isfinite(z) & (np.abs(f(z) - w) < 1e-6 * (1 + np.abs(w)))
    return z, ok


def make_custom(expr, mode, consts=None):
    """mode 'inverse': expr is f^-1(w) (may use k for multi-valued sheets) -> used directly.
       mode 'forward': expr is f(z) -> inverted numerically by Newton from several start points;
       each start point can land on a different preimage (= a different 'sheet').  Cuts ON uses
       one start point, cuts OFF uses `n_sheets` of them.  Root-finding can fail or pick odd
       roots near critical points of f -- supplying the true inverse is more reliable.

       In both modes the formula may contain the constants a, b, c, d.  They are NOT frozen
       here: they are read from `p.consts` every time the map is evaluated, so moving a
       slider changes the picture without re-applying the formula."""
    f = _compile(expr, "w" if mode == "inverse" else "z")
    f(
        np.array([0.5 + 0.5j]),
        consts=consts or DEFAULT_CONSTS
    )
    if mode == "inverse":
        def inv(w, p):
            for k in sheet_indices(p):            # one sheet per k, nearest first
                z = f(w, k, p.consts)
                yield z, np.isfinite(z)
        def fwd(z, p):
            raise NotImplementedError("Forward demo needs f(z); you supplied only the inverse.")
    else:
        def fwd(z, p):
            return [f(z, 0, p.consts)]
        def inv(w, p):
            g = lambda x: f(x, 0, p.consts)       # f as a one-argument function, constants baked in
            # cuts ON: only the first seed; cuts OFF: n_sheets seeds -> different preimages
            for s in _SEEDS[:1 if p.cuts_on else p.n_sheets]:
                yield _newton(g, w, np.full(w.shape, p.seed_center + 0.6 * p.seed_radius * s, complex))
    return Transform("custom", fwd, inv, 4.0)
