"""
render.py -- the rendering pipeline.
  INVERSE (default):  output pixel -> w -> z = f^{-1}(w) [each sheet] -> (Droste pull-back)
                      -> source pixel -> bilinear sample.  Every output pixel is visited once,
                      so there are no holes; bilinear interpolation (+ optional supersampling)
                      removes blockiness.
  FORWARD (demo):     source pixel -> z -> w = f(z) -> output pixel.  Source is oversampled to
                      reduce -- but cannot eliminate -- holes where f stretches the plane.
Output is RGBA: alpha = fraction of (sub)samples that hit the picture (0 = void).

PERIOD STACKING (optional, `PeriodParams`): the finished output is repeated as copies shifted by
n*T (T = any complex number, n = -n_neg .. +n_pos) and the copies are LAYERED over each other.
A copy is transparent where it is void, so the next copy shows through.  Where several copies
have picture at the same pixel, the one whose point lies FARTHEST from the centre of the
original picture is on top.  This is plain translation of the final result, not a Droste effect.
"""
from dataclasses import dataclass
import numpy as np
from PIL import Image
from coords import Region, pixel_to_complex, complex_to_pixel
from droste import droste_pullback

NORM = Region(0j, 2.0, 0.0)   # picture-local frame: picture spans x in [-1, 1]


@dataclass
class PeriodParams:
    """Repeat the finished OUTPUT as stacked copies shifted by n*T, n = -n_neg .. +n_pos.
    T is ANY complex number: a real T is a plain horizontal frieze, T = 2-0.3i repeats the
    picture a little bit downward, ... (a pure translation: no rotation, no scaling).
    `centre` = the point of the ORIGINAL picture (plane coordinates; where f(z) blows up,
    i.e. z = 0 for ln) that decides who is on top: the farthest point from it wins."""
    enabled: bool = False
    T: complex = 2 + 0j
    n_neg: int = 2            # copies at -T, -2T, ...
    n_pos: int = 2            # copies at +T, +2T, ...
    centre: complex = 0j


def load_image(path, max_side=1500):
    """Read an image file as a float32 RGB array (H, W, 3); large images are shrunk for speed."""
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side))                     # keeps the aspect ratio
    return np.asarray(im, dtype=np.float32)


def to_local(z, region):
    """Plane point -> picture-local coords (undo shift / rotation / scale of the picture)."""
    return (z - region.center) * np.exp(-1j * region.rotation) / (region.width / 2)


def bilinear_sample(img, px, py):
    """Blend the 4 neighbouring pixels. Returns (rgb, valid); valid=False if outside the picture
    (or if px/py is NaN/inf, e.g. a Droste 'void' point)."""
    H, W = img.shape[:2]
    # a position is usable only if it is finite and within the picture (half a pixel of margin)
    valid = np.isfinite(px) & np.isfinite(py) & (px >= -0.5) & (px <= W - 0.5) & (py >= -0.5) & (py <= H - 0.5)
    x = np.clip(np.where(valid, px, 0), 0, W - 1)          # invalid points are parked at 0 so indexing is safe
    y = np.clip(np.where(valid, py, 0), 0, H - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)          # top-left neighbour
    x1, y1 = np.minimum(x0 + 1, W - 1), np.minimum(y0 + 1, H - 1)      # bottom-right neighbour (clamped at the edge)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]                  # fractional parts = blend weights
    # weighted average of the four corner colours
    rgb = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy)
           + img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)
    return rgb, valid


def _rgba(rgb, alpha):
    """Stack colour and alpha (0..1 -> 0..255) into a uint8 (H, W, 4) image."""
    return np.dstack([rgb, alpha[..., None] * 255]).clip(0, 255).astype(np.uint8)


def _shade(w, src, tf, tp, dp, src_region, centre):
    """Colour of the picture seen at the output-plane points w (the existing pipeline: f^-1 on each
    sheet, nearest sheet first -> Droste pull-back -> bilinear sample).
    Returns (color, filled, dist).  dist = |z - centre| of the plane point that supplied the colour
    (z = f^-1(w), BEFORE the Droste pull-back, so it measures the distance from the centre in the
    picture as it is shown); only computed when `centre` is given."""
    sh, sw = src.shape[:2]
    filled = np.zeros(w.shape, bool)                       # pixels that already got a colour
    color = np.zeros(w.shape + (3,), np.float32)
    dist = np.full(w.shape, -np.inf)
    for z, ok in tf.inverse(w, tp):                        # nearest sheet first; first hit wins
        if filled.all():                                   # everything painted -> farther sheets are not needed
            break
        zl = to_local(z, src_region)                       # plane point -> picture-local coordinates
        if dp.enabled:
            zl = droste_pullback(zl, dp)                   # recursive "picture in picture"
        px, py = complex_to_pixel(zl, sw, sh, NORM)        # picture-local -> source pixel
        c, valid = bilinear_sample(src, px, py)
        take = ok & valid & ~filled                        # sheet is valid here, hits the picture, pixel still empty
        color[take] = c[take]
        if centre is not None:
            dist[take] = np.abs(z - centre)[take]
        filled |= take
    return color, filled, dist


def _stack_copies(w, shade, period):
    """Layer the copies n*T (n = -n_neg .. +n_pos).  Copy n shows, at the output point w, whatever the
    single picture shows at w - n*T.  Void parts of a copy are transparent, so a neighbouring copy shows
    through there.  Where several copies have picture, the one whose source point is FARTHEST from the
    centre of the original picture is on top (the outer part of a copy covers the inner part of its neighbour)."""
    color = np.zeros(w.shape + (3,), np.float32)
    filled = np.zeros(w.shape, bool)
    best = np.full(w.shape, -np.inf)                       # distance of the current winner at every pixel
    for n in range(-period.n_neg, period.n_pos + 1):
        c, f, d = shade(w - n * period.T)
        d = np.where(np.isfinite(d), d, -np.inf)
        win = f & (~filled | (d > best))                   # first picture seen, or farther from the centre
        color[win] = c[win]
        best[win] = d[win]
        filled |= win
    return color, filled


def render_inverse(src, tf, tp, dp, src_region, out_region, out_size, ss=1, period=None):
    """Paint the output by pulling colours back from the source picture.
    src = picture, tf = Transform, tp = TParams, dp = DrosteParams,
    src_region / out_region = where the picture / the output window lie in the plane,
    out_size = (width, height) px, ss = supersampling factor (ss x ss samples per pixel),
    period = optional PeriodParams: stack translated copies of the finished output (see module docstring)."""
    ow, oh = out_size
    stack = period is not None and period.enabled and period.T != 0
    centre = period.centre if stack else None
    shade = lambda x: _shade(x, src, tf, tp, dp, src_region, centre)
    # acc = sum of the colours of all hitting samples, cnt = how many samples hit the picture
    acc, cnt = np.zeros((oh, ow, 3), np.float32), np.zeros((oh, ow), np.float32)
    with np.errstate(all="ignore"):                # log(0), division by zero ... are expected; NaN just means "void"
        for sy in range(ss):                       # ss x ss sub-pixel samples per output pixel
            for sx in range(ss):
                # pixel grid shifted by the sub-pixel offset of this sample
                jj, ii = np.meshgrid(np.arange(ow) + (sx + .5) / ss - .5, np.arange(oh) + (sy + .5) / ss - .5)
                w = pixel_to_complex(jj, ii, ow, oh, out_region)      # output pixel -> point w of the output plane
                if stack:
                    color, filled = _stack_copies(w, shade, period)
                else:
                    color, filled, _ = shade(w)
                acc += color
                cnt += filled
    # average colour over the samples that hit; alpha = fraction of samples that hit
    return _rgba(acc / np.maximum(cnt, 1)[..., None], cnt / (ss * ss))


def render_forward(src, tf, tp, src_region, out_region, out_size, over=2):
    """Demo: push every source pixel through f and drop it on the nearest output pixel.
    Pixels that nobody lands on stay transparent, which is why this mode shows holes."""
    sh, sw = src.shape[:2]
    ow, oh = out_size
    out, alpha = np.zeros((oh, ow, 3), np.float32), np.zeros((oh, ow), np.float32)
    # oversampled grid of source positions (over x over points per source pixel)
    jj, ii = np.meshgrid((np.arange(sw * over) + .5) / over - .5, (np.arange(sh * over) + .5) / over - .5)
    rgb, _ = bilinear_sample(src, jj, ii)                          # colour at every grid point
    z = pixel_to_complex(jj, ii, sw, sh, src_region)               # source position -> complex plane
    with np.errstate(all="ignore"):
        for w in tf.forward(z, tp):                                # one array of images per sheet
            ox, oy = complex_to_pixel(w, ow, oh, out_region)       # w -> output pixel coordinates
            ok = np.isfinite(ox) & np.isfinite(oy)
            ix, iy = np.rint(np.where(ok, ox, -1)).astype(int), np.rint(np.where(ok, oy, -1)).astype(int)
            ok &= (ix >= 0) & (ix < ow) & (iy >= 0) & (iy < oh)    # keep only points inside the output window
            out[iy[ok], ix[ok]] = rgb[ok]          # nearest-pixel splat; unhit pixels stay void
            alpha[iy[ok], ix[ok]] = 1
    return _rgba(out, alpha)


def composite_preview(rgba, bg=(70, 70, 70)):
    """Show transparent (void) pixels over a grey background."""
    im = Image.fromarray(rgba, "RGBA")
    return Image.alpha_composite(Image.new("RGBA", im.size, bg + (255,)), im).convert("RGB")
