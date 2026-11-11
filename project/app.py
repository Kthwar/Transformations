"""
app.py -- Tkinter UI.  Run:  pip install numpy pillow   then   python app.py
Left = original (click it to draw the Droste polygon), right = result.
Every slider has a number box next to it: type a value + Enter (values may exceed the slider range).
"""
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
from coords import Region
from transforms import TRANSFORMS, TParams, make_custom
from droste import DrosteParams
from render import load_image, render_inverse, render_forward, composite_preview

BASE_W, PREVIEW = 4.0, 420    # picture is BASE_W plane-units wide at zoom 1


class App:
    def __init__(self, root):
        self.root, self.src, self.pending = root, None, None
        self.val, self.chk, self.custom_tf, self.poly = {}, {}, None, []
        self.dw = self.dh = PREVIEW; self.ah = 1.0           # displayed size / aspect of the original
        root.title("TMA: complex maps and Droste effect")
        top = ttk.Frame(root); top.pack()
        mk = lambda: tk.Canvas(top, width=PREVIEW, height=PREVIEW, bg="#333", highlightthickness=0)
        self.cv_in, self.cv_out = mk(), mk()
        self.cv_in.pack(side="left", padx=4); self.cv_out.pack(side="left", padx=4)
        self.cv_in.bind("<Button-1>", self.add_point)
        for b in ("<Button-2>", "<Button-3>"): self.cv_in.bind(b, self.pop_point)
        self.status = tk.StringVar()
        ttk.Label(root, textvariable=self.status, foreground="#a00").pack()
        panel = ttk.Frame(root); panel.pack(fill="x")
        cols = [ttk.Frame(panel) for _ in range(4)]
        for c in cols: c.pack(side="left", anchor="n", padx=6)

        for s in [("zoom", "Picture zoom", .1, 5, 1, .05), ("sx", "Picture shift x", -3, 3, 0, .05),
                  ("sy", "Picture shift y", -3, 3, 0, .05), ("rot", "Picture rotation (deg)", 0, 360, 0, 1),
                  ("ohalf", "Output half-width", .2, 20, 2, .05), ("ocx", "Output centre x", -10, 10, 0, .1),
                  ("ocy", "Output centre y", -10, 10, 0, .1)]:
            self.slider(cols[0], *s)

        c1 = cols[1]
        self.combo = ttk.Combobox(c1, values=list(TRANSFORMS) + ["custom"], state="readonly")
        self.combo.set("ln(z)"); self.combo.pack(fill="x")
        self.combo.bind("<<ComboboxSelected>>", self.on_transform)
        self.expr, self.cmode = tk.StringVar(value="log(w) + 2*pi*i*k"), tk.StringVar(value="inverse")
        ttk.Label(c1, text="Custom: variable z (or w), k = sheet, i = imaginary unit").pack(anchor="w")
        e = ttk.Entry(c1, textvariable=self.expr); e.pack(fill="x"); e.bind("<Return>", lambda _: self.apply_custom())
        for txt, val in [("expression is the INVERSE f^-1(w)", "inverse"), ("expression is f(z) (numeric inverse)", "forward")]:
            ttk.Radiobutton(c1, text=txt, variable=self.cmode, value=val).pack(anchor="w")
        ttk.Button(c1, text="Apply custom function", command=self.apply_custom).pack(fill="x")
        self.check(c1, "cuts", "Branch cuts ON (principal sheet)", True)
        self.check(c1, "fwd", "Forward mapping (demo, no Droste)", False)
        self.check(c1, "cross", "Crosshair at centre", False)
        for s in [("theta0", "Branch cut angle (deg)", -180, 180, -180, 1), ("nsheets", "Sheets when cuts off", 1, 9, 5, 1),
                  ("n", "Exponent n (z^n)", -4, 6, 2, .5), ("ss", "Supersampling", 1, 3, 1, 1)]:
            self.slider(c1, *s)

        self.check(cols[2], "dr", "Droste ON", False)
        self.model = tk.StringVar(value="annulus")
        for txt, val in [("A: annulus (log map)", "annulus"), ("B: disc", "disc"), ("C: drawn polygon (click original)", "polygon")]:
            ttk.Radiobutton(cols[2], text=txt, variable=self.model, value=val, command=self.schedule).pack(anchor="w")
        ttk.Label(cols[2], text="C: 4 clicks = TL,TR,BR,BL of inner copy;\nother counts = free shape. Right-click undoes.").pack(anchor="w")
        for s in [("dcx", "Droste centre x (A,B)", -1, 1, 0, .01), ("dcy", "Droste centre y (A,B)", -1, 1, 0, .01),
                  ("r1", "Inner radius r1 (A)", .02, 1.5, .2, .01), ("r2", "Outer radius r2 (A)", .05, 2, .8, .01)]:
            self.slider(cols[2], *s)
        for s in [("rho", "Inner disc radius (B)", .05, 1, .4, .01), ("psi", "Droste rotation (deg)", 0, 360, 0, 1),
                  ("depth", "Max recursion depth", 1, 12, 6, 1), ("export", "Export size (px)", 400, 4000, 1600, 100)]:
            self.slider(cols[3], *s)
        for txt, cmd in [("Open image...", self.open_image), ("Save output (PNG)...", self.save), ("Clear drawn region", self.clear_poly)]:
            ttk.Button(cols[3], text=txt, command=cmd).pack(fill="x")
        self.on_transform()

    # ---- widgets ---------------------------------------------------------------------
    def slider(self, parent, key, label, lo, hi, default, res):
        """Slider + number box.  self.val[key] is the single source of truth."""
        self.val[key] = default
        row = ttk.Frame(parent); row.pack(fill="x")
        ttk.Label(row, text=label, width=24).pack(side="left")
        sv = tk.StringVar(value=f"{default:g}")
        ent = ttk.Entry(row, textvariable=sv, width=8); ent.pack(side="left")
        shown = lambda x: round(min(max(x, lo), hi) / res) * res

        def on_scale(x):
            if abs(float(x) - shown(self.val[key])) < res * .01: return   # echo of a typed out-of-range value
            self.val[key] = float(x); sv.set(f"{float(x):g}"); self.schedule()

        def on_entry(_=None):
            try: x = float(sv.get())
            except ValueError: sv.set(f"{self.val[key]:g}"); return
            self.val[key] = x; sc.set(x); self.schedule()

        sc = tk.Scale(parent, from_=lo, to=hi, resolution=res, orient="horizontal", showvalue=False,
                      length=250, command=on_scale)
        sc.set(default); sc.pack()
        ent.bind("<Return>", on_entry); ent.bind("<FocusOut>", on_entry)

    def check(self, parent, key, label, default):
        self.chk[key] = tk.BooleanVar(value=default)
        ttk.Checkbutton(parent, text=label, variable=self.chk[key], command=self.schedule).pack(anchor="w")

    def current_tf(self):
        if self.combo.get() == "custom":
            if self.custom_tf is None: raise ValueError("Press 'Apply custom function' first.")
            return self.custom_tf
        return TRANSFORMS[self.combo.get()]

    def on_transform(self, *_):
        try: hw = self.current_tf().default_half_width
        except ValueError: hw = 4.0
        self.val["ohalf"] = hw; self.val["ocx"] = self.val["ocy"] = 0
        self.root.after(0, self.schedule)

    def apply_custom(self):
        try: self.custom_tf = make_custom(self.expr.get(), self.cmode.get())
        except Exception as e:
            self.custom_tf = None; self.status.set(f"Custom function error: {type(e).__name__}: {e}"); return
        self.combo.set("custom"); self.schedule()

    # ---- drawn Droste region (picture-local coords: x in [-1,1], y in [-h,h]) ----------
    def to_canvas(self, z): return (z.real + 1) / 2 * self.dw, (1 - z.imag / self.ah) / 2 * self.dh
    def from_canvas(self, x, y): return complex(2 * x / self.dw - 1, (1 - 2 * y / self.dh) * self.ah)

    def add_point(self, ev):
        if self.src is None or ev.x > self.dw or ev.y > self.dh: return
        self.poly.append(self.from_canvas(ev.x, ev.y)); self.model.set("polygon"); self.schedule()

    def pop_point(self, _):
        if self.poly: self.poly.pop(); self.schedule()

    def clear_poly(self): self.poly = []; self.schedule()

    # ---- rendering -------------------------------------------------------------------
    def render(self, size, ss, fwd_over=2):
        g = lambda k: self.val[k]
        sh, sw = self.src.shape[:2]; asp = sh / sw
        tf = self.current_tf()
        src_region = Region(complex(g("sx"), g("sy")), BASE_W * g("zoom"), math.radians(g("rot")))
        tp = TParams(math.radians(g("theta0")), self.chk["cuts"].get(), g("n"), int(g("nsheets")),
                     src_region.center, src_region.width / 2)
        fwd = self.chk["fwd"].get()
        dp = DrosteParams(self.chk["dr"].get() and not fwd, self.model.get(), complex(g("dcx"), g("dcy")),
                          g("r1"), g("r2"), g("rho"), math.radians(g("psi")), int(g("depth")),
                          min(1.0, asp), asp, np.array(self.poly, dtype=complex))
        out_region = Region(complex(g("ocx"), g("ocy")), 2 * g("ohalf"), 0.0)
        if fwd:
            return render_forward(self.src, tf, tp, src_region, out_region, size, fwd_over)
        return render_inverse(self.src, tf, tp, dp, src_region, out_region, size, ss)

    def schedule(self, *_):
        if self.pending: self.root.after_cancel(self.pending)
        self.pending = self.root.after(150, self.redraw)

    def redraw(self):
        if self.src is None: return
        try:
            rgba = self.render((PREVIEW, PREVIEW), int(self.val["ss"])); self.status.set("")
            self.tk_out = ImageTk.PhotoImage(composite_preview(rgba))
            self.cv_out.delete("img"); self.cv_out.create_image(0, 0, anchor="nw", image=self.tk_out, tags="img")
        except Exception as e:
            self.status.set(f"{type(e).__name__}: {e}")
        self.draw_overlays()

    def draw_overlays(self):
        self.cv_in.delete("ov"); self.cv_out.delete("ov")
        if self.chk["cross"].get():
            for cv, (w, h) in ((self.cv_in, (self.dw, self.dh)), (self.cv_out, (PREVIEW, PREVIEW))):
                cv.create_line(w / 2, 0, w / 2, h, fill="red", tags="ov"); cv.create_line(0, h / 2, w, h / 2, fill="red", tags="ov")
        if self.src is None: return
        m, v = self.model.get(), self.val
        if self.chk["dr"].get() and m in ("annulus", "disc"):
            cx, cy = self.to_canvas(complex(v["dcx"], v["dcy"])); k = self.dw / 2
            for r, dash in ([(v["r1"], None), (v["r2"], (4, 3))] if m == "annulus" else [(v["rho"], None)]):
                self.cv_in.create_oval(cx - r * k, cy - r * k, cx + r * k, cy + r * k, outline="yellow", dash=dash, tags="ov")
        if m == "polygon":
            pts = [self.to_canvas(p) for p in self.poly]
            for i, (x, y) in enumerate(pts):
                self.cv_in.create_oval(x - 3, y - 3, x + 3, y + 3, fill="cyan", tags="ov")
                self.cv_in.create_text(x + 8, y - 8, text=str(i + 1), fill="cyan", tags="ov")
            if len(pts) > 1:
                flat = [c for p in pts for c in p] + (list(pts[0]) if len(pts) > 2 else [])
                self.cv_in.create_line(*flat, fill="cyan", tags="ov")

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.webp")])
        if not path: return
        self.src = load_image(path); self.poly = []
        sh, sw = self.src.shape[:2]; self.ah = sh / sw
        im = Image.fromarray(self.src.astype("uint8")); im.thumbnail((PREVIEW, PREVIEW))
        self.dw, self.dh = im.size
        self.tk_in = ImageTk.PhotoImage(im)
        self.cv_in.delete("all"); self.cv_in.create_image(0, 0, anchor="nw", image=self.tk_in)
        self.redraw()

    def save(self):
        if self.src is None: return messagebox.showinfo("No image", "Open an image first.")
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if not path: return
        n = int(self.val["export"])
        try: rgba = self.render((n, n), max(2, int(self.val["ss"])), fwd_over=3)
        except Exception as e: return messagebox.showerror("Render failed", str(e))
        Image.fromarray(rgba, "RGBA").save(path)


if __name__ == "__main__":
    root = tk.Tk(); App(root); root.mainloop()
