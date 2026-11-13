"""
app.py -- Tkinter UI.  Run:  pip install numpy pillow   then   python app.py
Left = original (click it to draw the Droste polygons), right = result.
Every slider has a number box next to it: type a value + Enter (values may exceed the slider range).

Droste model C has TWO polygons you can draw on the original (choose which one with the
"Clicks draw" radio buttons):
    output frame (cyan)   = where the inner copy is placed
    input frame  (orange) = which part of the picture the copy shows (empty = whole picture)
The custom formula can use the constants a, b, c, d, each controlled by its own slider.

PERIOD REPEAT (left column, "Repeat the output period"): draw a line on the RESULT canvas from a point
to the same feature one repeat further (or type the complex vector T: it need not be horizontal, e.g.
2-0.3i repeats the picture a bit downward).  The finished output is then copied at T, 2T, ... and -T,
-2T, ... and the copies are layered: void parts are transparent, and where copies overlap the point
FARTHEST from the "Priority centre" (the centre of the original picture, z = 0 for ln) is on top.
Right-click on the result canvas cancels a half-drawn line.
MOVE THE COPIES BY HAND: with "Repeat period ON", drag on the result canvas (the first copy toward +T follows
the mouse; T updates live), or click the canvas and use the arrow keys (1 preview pixel, Shift = 5).
"""
import cmath
import math
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
from coords import Region, pixel_to_complex, complex_to_pixel
from transforms import TRANSFORMS, TParams, make_custom, CONST_NAMES, DEFAULT_CONSTS
from droste import DrosteParams
from render import load_image, render_inverse, render_forward, composite_preview, PeriodParams

BASE_W, PREVIEW = 4.0, 420    # picture is BASE_W plane-units wide at zoom 1; PREVIEW = canvas size in px
PAD, PAD_RANGE, PAD_COLS = 160, 5.0, 4   # 2D constant pads: size in px, covers -PAD_RANGE..PAD_RANGE on both axes,
                                         # pads per row (4 = one compact row; 2 with PAD=100 = bigger 2x2 grid, taller window)

# ---- expressions typed into the number boxes ---------------------------------------------
# Names usable there (cmath = complex-aware, so sqrt(-4) = 2i instead of an error).
_BOX_NS = {n: getattr(cmath, n) for n in ["sin", "cos", "tan", "exp", "log", "sqrt", "sinh", "cosh", "tanh",
                                          "asin", "acos", "atan", "asinh", "acosh", "atanh"]}
_BOX_NS.update(pi=math.pi, e=math.e, tau=math.tau, i=1j, j=1j, ln=cmath.log, arg=cmath.phase,
               abs=abs, conj=lambda z: complex(z).conjugate(), real=lambda z: complex(z).real,
               imag=lambda z: complex(z).imag, deg=math.radians)
_IMPLICIT_I = re.compile(r"(?<![A-Za-z_0-9.])(\d+\.?\d*|\.\d+)i(?![A-Za-z_0-9])")   # "2i" -> "2*i"


def eval_box(text):
    """Evaluate a typed expression (e.g. '2*pi/3', '1+2i', 'exp(i*pi/4)', 'sqrt(2)^3') to a complex number.
    eval() with no builtins -- fine for a local tool where you type your own input."""
    text = _IMPLICIT_I.sub(r"\1*i", text.strip().replace("^", "**").replace(",", "."))
    z = complex(eval(compile(text, "<box>", "eval"), {"__builtins__": {}}, _BOX_NS))
    if not (math.isfinite(z.real) and math.isfinite(z.imag)): raise ValueError("not finite")
    return z


def eval_real(text):
    """Like eval_box, but the result must be a real number (tiny imaginary rounding noise is dropped)."""
    z = eval_box(text)
    if abs(z.imag) > 1e-9 * max(1.0, abs(z.real)): raise ValueError("result is complex, a real number is needed")
    return z.real


def fmt_complex(z):
    """Complex number -> short text that eval_box reads back, e.g. 1.5-2i, 3, -0.5i."""
    re_, im = z.real + 0.0, z.imag + 0.0                   # "+ 0.0" turns -0.0 into 0.0
    if im == 0: return f"{re_:g}"
    if re_ == 0: return f"{im:g}i"
    return f"{re_:g}{'+' if im > 0 else '-'}{abs(im):g}i"


class App:
    def __init__(self, root):
        self.root, self.src, self.pending = root, None, None   # src = loaded picture (float array), pending = scheduled redraw
        # val: current value of every slider; chk: checkbox variables;
        # custom_tf: the compiled custom function; poly / poly_in: output / input Droste frames
        self.val, self.chk, self.custom_tf, self.poly, self.poly_in = {}, {}, None, [], []
        self.dw = self.dh = PREVIEW; self.ah = 1.0           # displayed size / aspect of the original
        self.ox = self.oy = 0                                  # image centering offsets on canvas
        self.drag, self.drag_job = None, None                  # hand-moving the copies: (mouse start point, T at start); pending live-render job
        self.picking, self.pick_p1, self.per_anchor = 0, None, None   # period line: 0 idle / 1 wait start / 2 wait end; its start; arrow start
        root.title("TMA: complex maps and Droste effect")

        # ---- the two image canvases (left: original, right: result) ----
        top = ttk.Frame(root); top.pack()
        mk = lambda: tk.Canvas(top, width=PREVIEW, height=PREVIEW, bg="#333", highlightthickness=0)
        self.cv_in, self.cv_out = mk(), mk()
        self.cv_in.pack(side="left", padx=4); self.cv_out.pack(side="left", padx=4)
        self.cv_in.bind("<Button-1>", self.add_point)                         # left click adds a polygon point
        for b in ("<Button-2>", "<Button-3>"): self.cv_in.bind(b, self.pop_point)   # right/middle click undoes
        self.cv_out.bind("<Button-1>", self.on_out_click)                     # period line: 1st click = start, 2nd = end
        self.cv_out.bind("<Motion>", self.on_out_motion)                      # rubber band while the line is half drawn
        self.cv_out.bind("<B1-Motion>", self.on_out_drag)                     # drag = move the copies (T follows the mouse)
        self.cv_out.bind("<ButtonRelease-1>", self.on_out_release)
        self.cv_out.bind("<Key>", self.on_out_key)                            # arrow keys nudge T (canvas must have focus: click it)
        for b in ("<Button-2>", "<Button-3>"): self.cv_out.bind(b, self.cancel_pick)
        self.status = tk.StringVar()                                          # red error line under the canvases
        ttk.Label(root, textvariable=self.status, foreground="#a00").pack()

        # ---- control panel: four columns of widgets ----
        panel = ttk.Frame(root); panel.pack(fill="x")
        cols = [ttk.Frame(panel) for _ in range(4)]
        for c in cols: c.pack(side="left", anchor="n", padx=6)

        # Column 0: where the picture sits in the plane and what part of the plane is shown.
        # Each tuple = (key, label, min, max, default, step)
        for s in [("zoom", "Picture zoom", .1, 5, 1, .05), ("sx", "Picture shift x", -3, 3, 0, .05),
                  ("sy", "Picture shift y", -3, 3, 0, .05), ("rot", "Picture rotation (deg)", 0, 360, 0, 1),
                  ("ocx", "Output centre x", -10, 10, 0, .1),
                  ("ocy", "Output centre y", -10, 10, 0, .1),
                  ("ozoom", "Output zoom", .1, 20, 1, .05)]:      # <-- new
            self.slider(cols[0], *s)
        # Repeat the finished output as stacked copies shifted by the complex vector T.
        pf = ttk.LabelFrame(cols[0], text="Repeat the output period"); pf.pack(fill="x", pady=4)
        self.check(pf, "per", "Repeat period ON", False)
        ttk.Button(pf, text="Draw period line on result (2 clicks)", command=self.start_pick).pack(fill="x")
        self.set_T = self.cbox(pf, "T", "Period T (complex)", 2.0)
        self.cbox(pf, "pc", "Priority centre", 0j)
        for s in [("nneg", "Copies toward -T", 0, 8, 2, 1), ("npos", "Copies toward +T", 0, 8, 2, 1)]:
            self.slider(pf, *s)
        ttk.Label(pf, text="Drag on the result (or click it + arrow keys, Shift = x5)\n"
                           "to move the copies.  Where copies overlap, the point\n"
                           "farthest from the priority centre is on top.").pack(anchor="w")

        # Column 1: choice of the complex function + custom formula + branch-cut settings.
        c1 = cols[1]
        self.combo = ttk.Combobox(c1, values=list(TRANSFORMS) + ["custom"], state="readonly")
        self.combo.set("ln(z)"); self.combo.pack(fill="x")
        self.combo.bind("<<ComboboxSelected>>", self.on_transform)
        self.expr, self.cmode = tk.StringVar(value="log(w) + 2*pi*i*k"), tk.StringVar(value="inverse")
        ttk.Label(c1, text="Custom: variable z (or w), k = sheet, i = imaginary unit,\n"
                           "a b c d = complex constants, e.g.  a*log(w) + b + 2*pi*i*k").pack(anchor="w")
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

        # Column 2: Droste effect.
        self.check(cols[2], "dr", "Droste ON", False)
        self.model = tk.StringVar(value="annulus")
        for txt, val in [("A: annulus (log map)", "annulus"), ("B: disc", "disc"), ("C: drawn polygon (click original)", "polygon")]:
            ttk.Radiobutton(cols[2], text=txt, variable=self.model, value=val, command=self.schedule).pack(anchor="w")
        ttk.Label(cols[2], text="C: 4 clicks = TL,TR,BR,BL of a quad; other counts =\n"
                                "free shape. Right-click undoes the last point.\n"
                                "Output frame = where the copy goes; input frame =\n"
                                "which part of the picture it shows (none = all).").pack(anchor="w")
        # Which polygon the next clicks on the original belong to.
        self.target = tk.StringVar(value="out")
        for txt, val in [("Clicks draw: OUTPUT frame (cyan)", "out"), ("Clicks draw: INPUT frame (orange)", "in")]:
            ttk.Radiobutton(cols[2], text=txt, variable=self.target, value=val).pack(anchor="w")
        for s in [("dcx", "Droste centre x (A,B)", -1, 1, 0, .01), ("dcy", "Droste centre y (A,B)", -1, 1, 0, .01),
                  ("r1", "Inner radius r1 (A)", .02, 1.5, .2, .01), ("r2", "Outer radius r2 (A)", .05, 2, .8, .01)]:
            self.slider(cols[2], *s)

        # Column 3: remaining Droste settings, constants of the custom formula, file buttons.
        for s in [("rho", "Inner disc radius (B)", .05, 1, .4, .01), ("psi", "Droste rotation (deg)", 0, 360, 0, 1),
                  ("depth", "Max recursion depth", 1, 12, 6, 1), ("export", "Export size (px)", 400, 4000, 1600, 100)]:
            self.slider(cols[3], *s)
        # One slider per constant a, b, c, d usable inside the custom formula.
        cf = ttk.LabelFrame(cols[3], text="Custom-function constants (complex)")
        cf.pack(fill="x", pady=4)
        for idx, name in enumerate(CONST_NAMES):              # grid of 2D pads (x = real part, y = imaginary part)
            cell = ttk.Frame(cf); cell.grid(row=idx // PAD_COLS, column=idx % PAD_COLS, padx=1, pady=1)
            self.cpad(cell, f"const_{name}", name, DEFAULT_CONSTS[name])
        for txt, cmd in [("Open image...", self.open_image), ("Save output (PNG)...", self.save),
                         ("Clear output frame", self.clear_poly), ("Clear input frame", self.clear_poly_in)]:
            ttk.Button(cols[3], text=txt, command=cmd).pack(fill="x")
        self.on_transform()                                   # set defaults for the initially selected map

    # ---- widgets ---------------------------------------------------------------------
    def slider(self, parent, key, label, lo, hi, default, res):
        """Slider + number box.  self.val[key] is the single source of truth.
        The Scale can only show values in [lo, hi] on a grid of step `res`, whereas the number
        box accepts anything, so the typed value is kept in self.val and the Scale merely shows
        the nearest value it can."""
        self.val[key] = default
        row = ttk.Frame(parent); row.pack(fill="x")
        ttk.Label(row, text=label, width=24).pack(side="left")
        sv = tk.StringVar(value=f"{default:g}")               # text shown in the number box
        ent = ttk.Entry(row, textvariable=sv, width=8); ent.pack(side="left")
        # what the Scale shows for a given value (clamped to its range, snapped to its step)
        shown = lambda x: round(min(max(x, lo), hi) / res) * res

        def on_scale(x):
            """Scale moved (by the mouse, or by our own sc.set below)."""
            if abs(float(x) - shown(self.val[key])) < res * .01: return   # echo of a typed out-of-range value
            self.val[key] = float(x); sv.set(f"{float(x):g}"); self.schedule()

        def on_entry(_=None):
            """Enter / focus-out in the number box: evaluate it, move the Scale, redraw."""
            try: x = eval_real(sv.get())                      # the box accepts expressions: pi/2, 3*2, sqrt(2) ...
            except Exception as err:                          # bad text -> restore the old value
                sv.set(f"{self.val[key]:g}"); self.status.set(f"'{label}': {err}"); return
            if x == self.val[key]: return                     # nothing changed (e.g. plain focus-out)
            self.val[key] = x; sc.set(x); self.schedule()

        sc = tk.Scale(parent, from_=lo, to=hi, resolution=res, orient="horizontal", showvalue=False,
                      length=250, command=on_scale)
        sc.set(default); sc.pack()
        ent.bind("<Return>", on_entry); ent.bind("<FocusOut>", on_entry)

    def check(self, parent, key, label, default):
        """Checkbox whose BooleanVar is stored in self.chk[key]; toggling it redraws."""
        self.chk[key] = tk.BooleanVar(value=default)
        ttk.Checkbutton(parent, text=label, variable=self.chk[key], command=self.schedule).pack(anchor="w")

    def cpad(self, parent, key, label, default, rng=PAD_RANGE):
        """2D slider for a COMPLEX number: x axis = real part, y axis = imaginary part.
        Drag (or click) on the pad (centre = 0), or type an expression in the box below it ('1+2i', '2*exp(i*pi/4)').
        Typed values may lie outside the pad; the marker then sticks to the border and turns red.
        self.val[key] (a complex number) is the single source of truth."""
        self.val[key] = complex(default)
        cv = tk.Canvas(parent, width=PAD, height=PAD, bg="#222", highlightthickness=1, highlightbackground="#888")
        cv.pack()
        sv = tk.StringVar(value=fmt_complex(self.val[key]))
        ent = ttk.Entry(parent, textvariable=sv, width=9, justify="center"); ent.pack()
        to_px = lambda z: ((z.real / rng + 1) / 2 * PAD, (1 - z.imag / rng) / 2 * PAD)   # complex -> canvas pixel
        from_px = lambda x, y: complex((2 * x / PAD - 1) * rng, (1 - 2 * y / PAD) * rng)  # canvas pixel -> complex

        def draw():
            cv.delete("all")
            for t in (-rng / 2, 0, rng / 2):                     # grid lines at -rng/2, 0, +rng/2
                p = (t / rng + 1) / 2 * PAD
                col = "#aaa" if t == 0 else "#444"               # the two axes are brighter
                cv.create_line(p, 0, p, PAD, fill=col); cv.create_line(0, p, PAD, p, fill=col)
            cv.create_text(4, 2, text=label, anchor="nw", fill="#ddd", font=("TkDefaultFont", 9, "bold"))   # which constant
            z = self.val[key]
            out = abs(z.real) > rng or abs(z.imag) > rng         # typed value outside the pad?
            x, y = to_px(z)
            x, y = min(max(x, 3), PAD - 3), min(max(y, 3), PAD - 3)
            cv.create_oval(x - 4, y - 4, x + 4, y + 4, fill="red" if out else "yellow", outline="black")

        def set_value(z, update_box=True):
            if z == self.val[key]: return
            self.val[key] = z
            if update_box: sv.set(fmt_complex(z))
            draw(); self.schedule()

        def on_drag(ev):                                          # click / drag on the pad
            x, y = min(max(ev.x, 0), PAD), min(max(ev.y, 0), PAD)  # keep the mouse inside while dragging outside
            z = from_px(x, y)
            set_value(complex(round(z.real, 2), round(z.imag, 2)))  # snap to 0.01 so the box stays tidy

        def on_entry(_=None):
            try: z = eval_box(sv.get())
            except Exception as err:
                sv.set(fmt_complex(self.val[key])); self.status.set(f"Constant {label}: {err}"); return
            set_value(z, update_box=False)                        # keep the typed text as it is

        cv.bind("<Button-1>", on_drag); cv.bind("<B1-Motion>", on_drag)
        ent.bind("<Return>", on_entry); ent.bind("<FocusOut>", on_entry)
        draw()

    def cbox(self, parent, key, label, default):
        """Number box for a COMPLEX value ('2-0.3i', '2*exp(-i*0.1)').  self.val[key] is the source of truth.
        Returns a function set(z) to change the value from code (updates the box and redraws)."""
        self.val[key] = complex(default)
        row = ttk.Frame(parent); row.pack(fill="x")
        ttk.Label(row, text=label, width=20).pack(side="left")
        sv = tk.StringVar(value=fmt_complex(self.val[key]))
        ent = ttk.Entry(row, textvariable=sv, width=14); ent.pack(side="left")

        def set_value(z, update_box=True, redraw=True):
            if z == self.val[key]: return
            self.val[key] = z
            if update_box: sv.set(fmt_complex(z))
            if redraw: self.schedule()

        def on_entry(_=None):
            if sv.get() == fmt_complex(self.val[key]): return     # text untouched (keeps full precision of a drawn value)
            try: z = eval_box(sv.get())
            except Exception as err:
                sv.set(fmt_complex(self.val[key])); self.status.set(f"'{label}': {err}"); return
            set_value(z, update_box=False)

        ent.bind("<Return>", on_entry); ent.bind("<FocusOut>", on_entry)
        return set_value

    def current_tf(self):
        """The Transform selected in the drop-down (the compiled formula for 'custom')."""
        if self.combo.get() == "custom":
            if self.custom_tf is None: raise ValueError("Press 'Apply custom function' first.")
            return self.custom_tf
        return TRANSFORMS[self.combo.get()]

    def on_transform(self, *_):
        """A different map was chosen: reset the output window to a sensible size and redraw."""
        self.val["ocx"] = self.val["ocy"] = 0
        self.root.after(0, self.schedule)

    def apply_custom(self):
        """Compile the typed formula; on a typo show the error and keep no custom function."""
        consts = {name: self.val[f"const_{name}"] for name in CONST_NAMES}
        try: self.custom_tf = make_custom(self.expr.get(), self.cmode.get(), consts=consts)
        except Exception as e:
            self.custom_tf = None; self.status.set(f"Custom function error: {type(e).__name__}: {e}"); return
        self.combo.set("custom"); self.schedule()

    # ---- drawn Droste frames (picture-local coords: x in [-1,1], y in [-h,h]) ----------
    # Canvas pixels <-> picture-local complex numbers.  Canvas y grows downward, Im grows upward.
    def to_canvas(self, z): return (z.real + 1) / 2 * self.dw + self.ox, (1 - z.imag / self.ah) / 2 * self.dh + self.oy
    def from_canvas(self, x, y): return complex(2 * (x - self.ox) / self.dw - 1, (1 - 2 * (y - self.oy) / self.dh) * self.ah)

    def active_poly(self):
        """The polygon (list of points) the clicks currently edit."""
        return self.poly if self.target.get() == "out" else self.poly_in

    def add_point(self, ev):
        """Left click on the original: append a vertex to the active polygon."""
        if self.src is None or not (self.ox <= ev.x <= self.ox + self.dw and self.oy <= ev.y <= self.oy + self.dh): return   # ignore clicks outside the picture
        self.active_poly().append(self.from_canvas(ev.x, ev.y)); self.model.set("polygon"); self.schedule()

    def pop_point(self, _):
        """Right click: remove the last vertex of the active polygon."""
        if self.active_poly(): self.active_poly().pop(); self.schedule()

    def clear_poly(self): self.poly.clear(); self.schedule()        # forget the output frame
    def clear_poly_in(self): self.poly_in.clear(); self.schedule()  # forget the input frame

    # ---- period line on the RESULT canvas (output-plane coordinates) -----------------
    def out_region(self):
        """The window of the output plane that the result canvas shows (never rotated)."""
        g = self.val
        return Region(complex(g["ocx"], g["ocy"]), 4 / g["ozoom"], 0.0)

    def out_to_plane(self, x, y):
        """Result-canvas pixel -> point of the output plane."""
        return pixel_to_complex(x - .5, y - .5, PREVIEW, PREVIEW, self.out_region())

    def plane_to_out(self, z):
        """Point of the output plane -> result-canvas pixel."""
        px, py = complex_to_pixel(z, PREVIEW, PREVIEW, self.out_region())
        return px + .5, py + .5

    def start_pick(self):
        """Button: the next two clicks on the result canvas define the period vector T."""
        if self.src is None: self.status.set("Open an image first."); return
        self.picking, self.pick_p1 = 1, None
        self.status.set("Period line: click a feature on the result (START).  Right-click cancels.")

    def cancel_pick(self, _=None):
        self.picking, self.pick_p1 = 0, None
        self.cv_out.delete("rub"); self.status.set(""); self.draw_overlays()

    def on_out_click(self, ev):
        """1st click = start of the period line, 2nd click = the same feature one repeat further; T = end - start."""
        if not self.picking:                                  # no line being drawn: start moving the copies by hand
            self.cv_out.focus_set()                           # so that the arrow keys reach the canvas
            if self.src is not None and self.chk["per"].get() and not self.chk["fwd"].get():
                self.drag = (self.out_to_plane(ev.x, ev.y), self.val["T"])
            return
        p = self.out_to_plane(ev.x, ev.y)
        if self.picking == 1:
            self.pick_p1, self.picking = p, 2
            self.status.set("Period line: now click the SAME feature one repeat further (END).")
            self.draw_overlays(); return
        T = p - self.pick_p1
        self.picking = 0; self.cv_out.delete("rub")
        if abs(T) < 1e-9:
            self.status.set("The period line has zero length."); self.draw_overlays(); return
        self.per_anchor = self.pick_p1
        self.set_T(T); self.chk["per"].set(True); self.status.set(""); self.schedule()

    def on_out_drag(self, ev):
        """Mouse moved with the button down: T = T at the start + mouse movement (in plane units), so the first
        copy toward +T (and every copy, proportionally) follows the mouse.  Live preview at half resolution."""
        if self.drag is None: return
        p0, T0 = self.drag
        self.set_T(T0 + (self.out_to_plane(ev.x, ev.y) - p0), redraw=False)      # updates the T box, no full redraw yet
        if self.drag_job is None:                                                # coalesce motion events
            self.drag_job = self.root.after(30, self.drag_render)

    def drag_render(self):
        self.drag_job = None
        self.redraw(fast=True)

    def on_out_release(self, _):
        """Button released: forget the drag and draw the full-quality picture."""
        if self.drag is None: return
        self.drag = None
        if self.drag_job is not None: self.root.after_cancel(self.drag_job); self.drag_job = None
        self.schedule()

    def on_out_key(self, ev):
        """Arrow keys nudge T by one preview pixel (Shift = 5 pixels): Up/Down move the copies up/down."""
        step = {"Left": -1, "Right": 1, "Up": 1j, "Down": -1j}.get(ev.keysym)
        if step is None or not self.chk["per"].get(): return
        px = 4 / self.val["ozoom"] / PREVIEW * (5 if ev.state & 1 else 1)      # one preview pixel in plane units
        self.set_T(self.val["T"] + step * px)

    def on_out_motion(self, ev):
        """Rubber band from the start point to the mouse while the line is half drawn."""
        if self.picking != 2: return
        self.cv_out.delete("rub")
        x0, y0 = self.plane_to_out(self.pick_p1)
        self.cv_out.create_line(x0, y0, ev.x, ev.y, fill="yellow", width=2, arrow="last", tags="rub")

    # ---- rendering -------------------------------------------------------------------
    def render(self, size, ss, fwd_over=2):
        """Collect all UI settings into parameter objects and run the renderer.
        `size` = (width, height) of the output in px, `ss` = supersampling factor."""
        g = lambda k: self.val[k]
        sh, sw = self.src.shape[:2]; asp = sh / sw          # picture aspect ratio h = H/W
        tf = self.current_tf()
        # where the picture lives in the complex plane (centre, width, rotation)
        src_region = Region(complex(g("sx"), g("sy")), BASE_W * g("zoom"), math.radians(g("rot")))
        # settings for the map; consts = current values of the sliders for a, b, c, d
        tp = TParams(theta0=math.radians(g("theta0")), cuts_on=self.chk["cuts"].get(), n=g("n"),
                     n_sheets=int(g("nsheets")), seed_center=src_region.center,
                     seed_radius=src_region.width / 2,
                     consts={name: g(f"const_{name}") for name in CONST_NAMES})
        fwd = self.chk["fwd"].get()
        # Droste settings, including both drawn frames (output frame, input frame)
        dp = DrosteParams(self.chk["dr"].get() and not fwd, self.model.get(), complex(g("dcx"), g("dcy")),
                          g("r1"), g("r2"), g("rho"), math.radians(g("psi")), int(g("depth")),
                          min(1.0, asp), asp, np.array(self.poly, dtype=complex),
                          np.array(self.poly_in, dtype=complex))
        # the window of the output plane that is shown (never rotated)
        out_region = self.out_region()
        if fwd:
            return render_forward(self.src, tf, tp, src_region, out_region, size, fwd_over)
        # stacked copies of the finished output (T = any complex vector; priority = farthest from the centre)
        per = PeriodParams(self.chk["per"].get() and g("T") != 0, g("T"), int(g("nneg")), int(g("npos")), g("pc"))
        return render_inverse(self.src, tf, tp, dp, src_region, out_region, size, ss, period=per)

    def schedule(self, *_):
        """Debounce: redraw 150 ms after the LAST change, so dragging a slider stays smooth."""
        if self.pending: self.root.after_cancel(self.pending)
        self.pending = self.root.after(150, self.redraw)

    def redraw(self, fast=False):
        """Render a preview into the right canvas, then refresh the overlays on both canvases.
        fast=True (used while dragging the copies): half-resolution, no supersampling, scaled up."""
        if self.src is None: return
        try:
            n = PREVIEW // 2 if fast else PREVIEW
            rgba = self.render((n, n), 1 if fast else int(self.val["ss"])); self.status.set("")
            im = composite_preview(rgba)
            if fast: im = im.resize((PREVIEW, PREVIEW))
            self.tk_out = ImageTk.PhotoImage(im)                        # keep a reference or Tk drops the image
            self.cv_out.delete("img"); self.cv_out.create_image(0, 0, anchor="nw", image=self.tk_out, tags="img")
        except Exception as e:
            self.status.set(f"{type(e).__name__}: {e}")
        self.draw_overlays()

    def draw_overlays(self):
        """Crosshair, Droste circles and the drawn polygons (all tagged 'ov' so they can be wiped)."""
        self.cv_in.delete("ov"); self.cv_out.delete("ov")
        if self.chk["cross"].get():
            self.cv_in.create_line(self.ox + self.dw / 2, self.oy, self.ox + self.dw / 2, self.oy + self.dh, fill="red", tags="ov")
            self.cv_in.create_line(self.ox, self.oy + self.dh / 2, self.ox + self.dw, self.oy + self.dh / 2, fill="red", tags="ov")
            self.cv_out.create_line(PREVIEW / 2, 0, PREVIEW / 2, PREVIEW, fill="red", tags="ov")
            self.cv_out.create_line(0, PREVIEW / 2, PREVIEW, PREVIEW / 2, fill="red", tags="ov")
        if self.src is None: return
        if self.picking == 2 and self.pick_p1 is not None:           # start point of a half-drawn period line
            x, y = self.plane_to_out(self.pick_p1)
            self.cv_out.create_oval(x - 4, y - 4, x + 4, y + 4, fill="yellow", outline="black", tags="ov")
        elif self.chk["per"].get() and not self.chk["fwd"].get():    # the period vector T as an arrow
            T = self.val["T"]
            a = self.per_anchor if self.per_anchor is not None else self.out_region().center - T / 2
            (x0, y0), (x1, y1) = self.plane_to_out(a), self.plane_to_out(a + T)
            self.cv_out.create_line(x0, y0, x1, y1, fill="yellow", width=2, arrow="last", tags="ov")
            self.cv_out.create_oval(x0 - 3, y0 - 3, x0 + 3, y0 + 3, fill="yellow", outline="black", tags="ov")
        m, v = self.model.get(), self.val
        # models A and B: show the circles (solid = r1 or rho, dashed = r2) on the original
        if self.chk["dr"].get() and m in ("annulus", "disc"):
            cx, cy = self.to_canvas(complex(v["dcx"], v["dcy"])); k = self.dw / 2
            for r, dash in ([(v["r1"], None), (v["r2"], (4, 3))] if m == "annulus" else [(v["rho"], None)]):
                self.cv_in.create_oval(cx - r * k, cy - r * k, cx + r * k, cy + r * k, outline="yellow", dash=dash, tags="ov")
        # model C: draw both polygons -- output frame in cyan (labels 1,2,..), input frame in orange (i1,i2,..)
        if m == "polygon":
            for poly, colour, prefix in ((self.poly, "cyan", ""), (self.poly_in, "orange", "i")):
                pts = [self.to_canvas(p) for p in poly]
                for i, (x, y) in enumerate(pts):                  # numbered vertex markers
                    self.cv_in.create_oval(x - 3, y - 3, x + 3, y + 3, fill=colour, tags="ov")
                    self.cv_in.create_text(x + 8, y - 8, text=f"{prefix}{i + 1}", fill=colour, tags="ov")
                if len(pts) > 1:                                  # outline; closed once there are 3+ points
                    flat = [c for p in pts for c in p] + (list(pts[0]) if len(pts) > 2 else [])
                    self.cv_in.create_line(*flat, fill=colour, tags="ov")

    def open_image(self):
        """Load a picture, show it (shrunk) on the left canvas and reset the drawn frames."""
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.webp")])
        if not path: return
        self.src = load_image(path); self.poly.clear(); self.poly_in.clear()
        sh, sw = self.src.shape[:2]; self.ah = sh / sw
        im = Image.fromarray(self.src.astype("uint8")); im.thumbnail((PREVIEW, PREVIEW))
        self.dw, self.dh = im.size                            # displayed size (keeps the aspect ratio)
        self.ox = (PREVIEW - self.dw) // 2                    # horizontal centering offset
        self.oy = (PREVIEW - self.dh) // 2                    # vertical centering offset
        self.tk_in = ImageTk.PhotoImage(im)
        self.cv_in.delete("all"); self.cv_in.create_image(self.ox, self.oy, anchor="nw", image=self.tk_in)
        self.redraw()

    def save(self):
        """Render at the export size (with at least 2x supersampling) and write a PNG with alpha."""
        if self.src is None: return messagebox.showinfo("No image", "Open an image first.")
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if not path: return
        n = int(self.val["export"])
        try: rgba = self.render((n, n), max(2, int(self.val["ss"])), fwd_over=3)
        except Exception as e: return messagebox.showerror("Render failed", str(e))
        Image.fromarray(rgba, "RGBA").save(path)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()