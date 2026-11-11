"""
render.py -- the rendering pipeline.
  INVERSE (default):  output pixel -> w -> z = f^{-1}(w) [each sheet] -> (Droste pull-back)
                      -> source pixel -> bilinear sample.  Every output pixel is visited once,
                      so there are no holes; bilinear interpolation (+ optional supersampling)
                      removes blockiness.
  FORWARD (demo):     source pixel -> z -> w = f(z) -> output pixel.  Source is oversampled to
                      reduce -- but cannot eliminate -- holes where f stretches the plane.
Output is RGBA: alpha = fraction of (sub)samples that hit the picture (0 = void).
"""
import numpy as np
from PIL import Image
from coords import Region, pixel_to_complex, complex_to_pixel
from droste import droste_pullback

NORM = Region(0j, 2.0, 0.0)   # picture-local frame: picture spans x in [-1, 1]


def load_image(path, max_side=1500):
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side))
    return np.asarray(im, dtype=np.float32)


def to_local(z, region):
    """Plane point -> picture-local coords (undo shift / rotation / scale of the picture)."""
    return (z - region.center) * np.exp(-1j * region.rotation) / (region.width / 2)


def bilinear_sample(img, px, py):
    """Blend the 4 neighbouring pixels. Returns (rgb, valid); valid=False if outside the picture."""
    H, W = img.shape[:2]
    valid = np.isfinite(px) & np.isfinite(py) & (px >= -0.5) & (px <= W - 0.5) & (py >= -0.5) & (py <= H - 0.5)
    x = np.clip(np.where(valid, px, 0), 0, W - 1)
    y = np.clip(np.where(valid, py, 0), 0, H - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, W - 1), np.minimum(y0 + 1, H - 1)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    rgb = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy)
           + img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)
    return rgb, valid


def _rgba(rgb, alpha):
    return np.dstack([rgb, alpha[..., None] * 255]).clip(0, 255).astype(np.uint8)


def render_inverse(src, tf, tp, dp, src_region, out_region, out_size, ss=1):
    sh, sw = src.shape[:2]
    ow, oh = out_size
    acc, cnt = np.zeros((oh, ow, 3), np.float32), np.zeros((oh, ow), np.float32)
    with np.errstate(all="ignore"):
        for sy in range(ss):                       # ss x ss sub-pixel samples per output pixel
            for sx in range(ss):
                jj, ii = np.meshgrid(np.arange(ow) + (sx + .5) / ss - .5, np.arange(oh) + (sy + .5) / ss - .5)
                w = pixel_to_complex(jj, ii, ow, oh, out_region)
                filled = np.zeros((oh, ow), bool)
                color = np.zeros((oh, ow, 3), np.float32)
                for z, ok in tf.inverse(w, tp):    # nearest sheet first; first hit wins
                    zl = to_local(z, src_region)
                    if dp.enabled:
                        zl = droste_pullback(zl, dp)
                    px, py = complex_to_pixel(zl, sw, sh, NORM)
                    c, valid = bilinear_sample(src, px, py)
                    take = ok & valid & ~filled
                    color[take] = c[take]
                    filled |= take
                acc += color
                cnt += filled
    return _rgba(acc / np.maximum(cnt, 1)[..., None], cnt / (ss * ss))


def render_forward(src, tf, tp, src_region, out_region, out_size, over=2):
    sh, sw = src.shape[:2]
    ow, oh = out_size
    out, alpha = np.zeros((oh, ow, 3), np.float32), np.zeros((oh, ow), np.float32)
    jj, ii = np.meshgrid((np.arange(sw * over) + .5) / over - .5, (np.arange(sh * over) + .5) / over - .5)
    rgb, _ = bilinear_sample(src, jj, ii)
    z = pixel_to_complex(jj, ii, sw, sh, src_region)
    with np.errstate(all="ignore"):
        for w in tf.forward(z, tp):
            ox, oy = complex_to_pixel(w, ow, oh, out_region)
            ok = np.isfinite(ox) & np.isfinite(oy)
            ix, iy = np.rint(np.where(ok, ox, -1)).astype(int), np.rint(np.where(ok, oy, -1)).astype(int)
            ok &= (ix >= 0) & (ix < ow) & (iy >= 0) & (iy < oh)
            out[iy[ok], ix[ok]] = rgb[ok]          # nearest-pixel splat; unhit pixels stay void
            alpha[iy[ok], ix[ok]] = 1
    return _rgba(out, alpha)


def composite_preview(rgba, bg=(70, 70, 70)):
    """Show transparent (void) pixels over a grey background."""
    im = Image.fromarray(rgba, "RGBA")
    return Image.alpha_composite(Image.new("RGBA", im.size, bg + (255,)), im).convert("RGB")
