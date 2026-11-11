"""
app.py -- Tkinter UI.  Run:  pip install numpy pillow   then   python app.py
Left = original picture, right = result.  Sliders re-render a small preview; "Save output"
re-renders at the chosen export size (with supersampling) and writes an RGBA PNG.
"""
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from coords import Region
from transforms import TRANSFORMS, TParams
from droste import DrosteParams
from render import load_image, render_inverse, render_forward, composite_preview

BASE_W, PREVIEW = 4.0, 420    # picture is BASE_W plane-units wide at zoom 1


class App:
    def __init__(self, root):
        self.root, self.src, self.pending, self.v = root, None, None, {}
        root.title("Math & Art: complex maps and Droste")
        top = ttk.Frame(root); top.pack()
        self.lbl_in, self.lbl_out = ttk.Label(top), ttk.Label(top)
        self.lbl_in.pack(side="left", padx=4); self.lbl_out.pack(side="left", padx=4)
        panel = ttk.Frame(root); panel.pack(fill="x")
        cols = [ttk.Frame(panel) for _ in range(4)]
        for c in cols: c.pack(side="left", anchor="n", padx=6)

        # column 0: picture placement + output window
        for s in [("zoom", "Picture zoom", .1, 5, 1, .05), ("sx", "Picture shift x", -3, 3, 0, .05),
                  ("sy", "Picture shift y", -3, 3, 0, .05), ("rot", "Picture rotation (deg)", 0, 360, 0, 1),
                  ("ohalf", "Output half-width", .2, 20, 2, .05), ("ocx", "Output centre x", -10, 10, 0, .1),
                  ("ocy", "Output centre y", -10, 10, 0, .1)]:
            self.slider(cols[0], *s)

        # column 1: transform + branch cuts
        self.combo = ttk.Combobox(cols[1], values=list(TRANSFORMS), state="readonly")
        self.combo.set("ln(z)"); self.combo.pack(fill="x")
        self.combo.bind("<<ComboboxSelected>>", self.on_transform)
        self.check(cols[1], "cuts", "Branch cuts ON (principal sheet)", True)
        self.check(cols[1], "fwd", "Forward mapping (demo, no Droste)", False)
        for s in [("theta0", "Branch cut angle (deg)", -180, 180, -180, 1), ("nsheets", "Sheets when cuts off", 1, 9, 5, 1),
                  ("n", "Exponent n (z^n)", -4, 6, 2, .5), ("ss", "Supersampling", 1, 3, 1, 1)]:
            self.slider(cols[1], *s)

        # columns 2-3: Droste
        self.check(cols[2], "dr", "Droste ON", False)
        self.model = tk.StringVar(value="annulus")
        for txt, val in [("A: annulus / spiral (log map)", "annulus"), ("B: nested copy in disc", "nested")]:
            ttk.Radiobutton(cols[2], text=txt, variable=self.model, value=val, command=self.schedule).pack(anchor="w")
        self.check(cols[2], "seam", "Seamless (integer alpha, beta=arms*L/2pi)", True)
        for s in [("dcx", "Droste centre x", -1, 1, 0, .01), ("dcy", "Droste centre y", -1, 1, 0, .01),
                  ("r1", "Inner radius r1 (A)", .02, 1.5, .2, .01), ("r2", "Outer radius r2 (A)", .05, 2, .8, .01)]:
            self.slider(cols[2], *s)
        for s in [("alpha", "alpha (scale)", .2, 3, 1, .05), ("beta", "beta (twist, if not seamless)", -3, 3, 0, .05),
                  ("arms", "Spiral arms p", -4, 4, 1, 1), ("depth", "Max recursion depth", 1, 12, 6, 1),
                  ("rho", "Inner disc radius (B)", .05, 1, .4, .01), ("psi", "Inner rotation deg (B)", 0, 360, 0, 1)]:
            self.slider(cols[3], *s)
        self.slider(cols[3], "export", "Export size (px)", 400, 4000, 1600, 100)
        ttk.Button(cols[3], text="Open image...", command=self.open_image).pack(fill="x")
        ttk.Button(cols[3], text="Save output (PNG)...", command=self.save).pack(fill="x")
        self.on_transform()

    # ---- small widget helpers -------------------------------------------------
    def slider(self, parent, key, label, lo, hi, default, res):
        self.v[key] = tk.DoubleVar(value=default)
        tk.Scale(parent, label=label, from_=lo, to=hi, resolution=res, orient="horizontal", length=210,
                 variable=self.v[key], command=lambda _: self.schedule()).pack()

    def check(self, parent, key, label, default):
        self.v[key] = tk.BooleanVar(value=default)
        ttk.Checkbutton(parent, text=label, variable=self.v[key], command=self.schedule).pack(anchor="w")

    def on_transform(self, *_):
        tf = TRANSFORMS[self.combo.get()]
        self.v["ohalf"].set(tf.default_half_width); self.v["ocx"].set(0); self.v["ocy"].set(0)
        self.schedule()

    # ---- parameter collection + rendering -----------------------------------------
    def render(self, size, ss, fwd_over=2):
        g = lambda k: self.v[k].get()
        sh, sw = self.src.shape[:2]
        tf = TRANSFORMS[self.combo.get()]
        tp = TParams(math.radians(g("theta0")), g("cuts"), g("n"), int(g("nsheets")))
        dp = DrosteParams(g("dr") and not g("fwd"), self.model.get(), complex(g("dcx"), g("dcy")), g("r1"), g("r2"),
                          g("alpha"), g("beta"), g("seam"), int(g("arms")), int(g("depth")), g("rho"),
                          math.radians(g("psi")), min(1.0, sh / sw), math.radians(g("theta0")))
        src_region = Region(complex(g("sx"), g("sy")), BASE_W * g("zoom"), math.radians(g("rot")))
        out_region = Region(complex(g("ocx"), g("ocy")), 2 * g("ohalf"), 0.0)
        if g("fwd"):
            return render_forward(self.src, tf, tp, src_region, out_region, size, fwd_over)
        return render_inverse(self.src, tf, tp, dp, src_region, out_region, size, ss)

    def schedule(self, *_):
        if self.pending: self.root.after_cancel(self.pending)
        self.pending = self.root.after(150, self.redraw)

    def redraw(self):
        if self.src is None: return
        rgba = self.render((PREVIEW, PREVIEW), int(self.v["ss"].get()))
        self.tk_out = ImageTk.PhotoImage(composite_preview(rgba))
        self.lbl_out.configure(image=self.tk_out)

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.webp")])
        if not path: return
        self.src = load_image(path)
        im = Image.fromarray(self.src.astype("uint8")); im.thumbnail((PREVIEW, PREVIEW))
        self.tk_in = ImageTk.PhotoImage(im); self.lbl_in.configure(image=self.tk_in)
        self.redraw()

    def save(self):
        if self.src is None:
            return messagebox.showinfo("No image", "Open an image first.")
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if not path: return
        n = int(self.v["export"].get())
        rgba = self.render((n, n), max(2, int(self.v["ss"].get())), fwd_over=3)
        Image.fromarray(rgba, "RGBA").save(path)


if __name__ == "__main__":
    root = tk.Tk(); App(root); root.mainloop()
