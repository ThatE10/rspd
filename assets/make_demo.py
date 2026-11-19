"""The README animation, drawn in manim from a real ``rspd`` solve.

    pip install rspd manim          # manim also needs LaTeX and ffmpeg
    python assets/make_demo.py      # writes assets/demo.gif

The picture: an input activation space holds a target set ``X`` and a retained
set ``Y`` that overlap. A point moves through ``X``, the overlap and ``Y``; the
column beside it is its activation. The layer ``W`` carries the whole space
across. The component ``M`` carries only what ``X`` alone uses.

What comes from the code (:func:`demo_data`), not from the drawing:

* the toy is :func:`rspd.toy.two_sets`: ``X`` is active on its own directions
  and a shared one, ``Y`` on the shared one and its own -- so the two ellipses
  are the two sets of directions and their overlap is the shared direction;
* the numbers in the moving column are :func:`rspd.toy.activation` at the
  point;
* ``M`` is ``rspd.core.erased_component`` on the toy's samples, and how strongly
  each region is drawn on the far side of ``M`` is the share of that region's
  directions that ``M`` carries (:func:`rspd.toy.carried`): about 1 for ``X``
  alone, small for the overlap, about 0 for ``Y`` alone.

The data half runs without manim (``tests/test_demo_data.py``).
"""
from __future__ import annotations

import pathlib

import numpy as np
import torch

import rspd
import rspd.toy

SEED = 0
MU_PER_TOKEN = 0.1
HERE = pathlib.Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# the data: a toy and one solve                                                #
# --------------------------------------------------------------------------- #
def demo_data(seed=SEED, mu_per_token=MU_PER_TOKEN) -> dict:
    """Everything the animation shows that is computed rather than drawn."""
    toy = rspd.toy.two_sets(seed=seed)
    M = rspd.core.erased_component(toy.W, toy.X, toy.Y, mu=toy.mu(mu_per_token))
    per_direction = rspd.toy.carried(toy.W, M)
    share = {g: float(per_direction[toy.index(g)].mean()) for g in ("x", "shared", "y")}
    leaks = rspd.core.leaks(toy.W, M, toy.X, toy.Y)

    def activation(in_x, in_y, phase=0.0):
        return [float(v) for v in rspd.toy.activation(toy, float(in_x), float(in_y), float(phase))]

    return {"toy": toy, "component": M, "carried": [float(v) for v in per_direction],
            "share": share, "leaks": leaks, "activation": activation, "groups": list(toy.groups)}


# --------------------------------------------------------------------------- #
# the drawing                                                                  #
# --------------------------------------------------------------------------- #
def build_scene(data=None):
    from manim import (DEGREES, DOWN, LEFT, TAU, UP, Arrow, Create, CurvedArrow, DecimalNumber,
                       Difference, Dot, Ellipse, GrowArrow, Intersection, ManimColor, MathTex,
                       Rectangle, Scene, Text, Transform, ValueTracker, VGroup, VMobject,
                       always_redraw, interpolate_color)

    data = data or demo_data()
    share, groups = data["share"], data["groups"]

    INK = "#000000"
    QF, QS = "#dbe7fb", "#3f6fd1"     # input activation space
    XF, XS = "#f8dc85", "#b98a00"     # X
    YF, YS = "#f6aec6", "#c43d6f"     # Y
    OF = "#f3b27a"                    # overlap of X and Y
    WHITE = ManimColor("#ffffff")
    ROW = {"x": XF, "shared": OF, "y": YF}

    def pale(c, a=0.72):
        return interpolate_color(ManimColor(c), WHITE, a)

    def rot(a):
        return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])

    class IsolateGif(Scene):
        def construct(self):
            cL = np.array([-3.3, -0.55, 0.0])
            cR = np.array([3.95, -0.5, 0.0])
            Wm = rot(75 * DEGREES) @ np.diag([1.05, 0.68, 1.0])

            QP = (np.array([0, 0, 0.0]), 2.2, 1.6, 0.0)
            XP = (np.array([0.25, 0.72, 0.0]), 1.2, 0.45, -10 * DEGREES)
            YP = (np.array([-0.2, -0.15, 0.0]), 1.3, 0.42, 45 * DEGREES)

            def make(par, fill, stroke):
                off, a, b, ang = par
                e = Ellipse(width=2 * a, height=2 * b, stroke_color=stroke, stroke_width=3)
                e.set_fill(fill, opacity=1)
                e.rotate(ang).move_to(cL + off)
                return e

            def r2(p, par):
                off, a, b, ang = par
                q = rot(-ang) @ (np.asarray(p) - cL - off)
                return (q[0] / a) ** 2 + (q[1] / b) ** 2

            def member(p, par, k=7.0):
                return 1.0 / (1.0 + np.exp(-k * (1.0 - r2(p, par))))

            def image(mob):
                return mob.copy().apply_matrix(Wm, about_point=cL).shift(cR - cL)

            def Wmap(p):
                return cR + Wm @ (np.asarray(p) - cL)

            Q = make(QP, QF, QS)
            X = make(XP, XF, XS)
            Y = make(YP, YF, YS)
            O = Intersection(X, Y, stroke_width=0).set_fill(OF, opacity=1)
            Xo = X.copy().set_fill(opacity=0)   # outlines redrawn above the overlap
            Yo = Y.copy().set_fill(opacity=0)

            lblX = MathTex("X", color=INK, font_size=40).move_to(
                cL + XP[0] + np.array([-0.55, 0.05, 0]))
            lblY = MathTex("Y", color=INK, font_size=40).move_to(
                cL + YP[0] + np.array([-0.5, -0.5, 0]))
            lblQ = Text("Input Activation Space", font_size=24, color=INK).next_to(
                Q, DOWN, buff=0.18)
            left = VGroup(Q, X, Y, O, Xo, Yo)
            self.add(left, lblX, lblY, lblQ)

            # ---------- a moving point and its activation (from rspd.toy) ----------
            path = VMobject()
            path.set_points_smoothly([cL + np.array(v + [0]) for v in
                                      [[-0.75, 0.9], [-0.1, 0.78], [0.6, 0.66], [0.45, 0.5],
                                       [0.1, 0.15], [-0.4, -0.35], [-0.9, -0.85]]])
            t = ValueTracker(0.0)

            def P():
                return path.point_from_proportion(t.get_value())

            rh, cw = 0.38, 1.1
            n = len(groups)

            def vector():
                p = P()
                v = data["activation"](member(p, XP), member(p, YP), 5 * t.get_value())
                g = VGroup()
                h = (n + 1) * rh + 0.16
                x0, y0 = p[0] + 0.3, p[1] + 0.25
                card = Rectangle(width=cw + 0.3, height=h, stroke_width=0)
                card.set_fill("#ffffff", opacity=1)
                card.move_to([x0 + (cw + 0.3) / 2, y0 + h / 2, 0])
                g.add(card)
                cx = card.get_center()[0]
                top = y0 + h - 0.08 - rh / 2
                for i in range(n):
                    act = min(1.0, abs(v[i]) / 0.25)
                    y = top - i * rh
                    r = Rectangle(width=cw, height=rh - 0.04, stroke_width=0)
                    r.set_fill(interpolate_color(WHITE, ManimColor(ROW[groups[i]]), act),
                               opacity=1).move_to([cx, y, 0])
                    num = DecimalNumber(v[i], num_decimal_places=2, include_sign=True,
                                        font_size=28,
                                        color=interpolate_color(ManimColor("#9aa0aa"),
                                                                ManimColor(INK), act))
                    num.move_to([cx, y, 0])
                    g.add(r, num)
                g.add(MathTex(r"\vdots", font_size=26, color=INK).move_to(
                    [cx, top - n * rh + 0.02, 0]))
                xl, xr = card.get_left()[0], card.get_right()[0]
                yb, yt = card.get_bottom()[1], card.get_top()[1]
                for xa, d in ((xl, 0.12), (xr, -0.12)):
                    g.add(VMobject(stroke_width=2.5, stroke_color=INK).set_points_as_corners(
                        [[xa + d, yt, 0], [xa, yt, 0], [xa, yb, 0], [xa + d, yb, 0]]))
                return g.set_z_index(8)

            vec = always_redraw(vector)
            mover = always_redraw(lambda: Dot(P(), radius=0.085, color=INK).set_z_index(9))
            self.add(vec, mover)

            self.wait(0.3)
            self.play(t.animate.set_value(0.22), run_time=0.7)
            self.wait(0.25)
            self.play(t.animate.set_value(0.43), run_time=0.7)
            self.wait(0.4)
            self.play(t.animate.set_value(1.0), run_time=1.0)
            self.wait(0.3)
            self.remove(vec, mover)

            # ---------- W carries the whole space across ----------
            arrowW = CurvedArrow(cL + np.array([1.5, 1.45, 0]), cR + np.array([-0.85, 2.1, 0]),
                                 angle=-TAU / 6, stroke_width=4, tip_length=0.25, color=INK)
            lblW = MathTex("W", font_size=44, color=INK).move_to([0.75, 1.15, 0])
            self.play(Create(arrowW), run_time=0.35)
            self.add(lblW)

            parts = [Q, X, Y, O, Xo, Yo]
            copies = [m.copy() for m in parts]
            self.add(*copies)
            self.play(*[Transform(c, image(m)) for c, m in zip(copies, parts)],
                      run_time=1.2, path_arc=-45 * DEGREES)
            Qc, Xc, Yc, Oc, Xoc, Yoc = copies
            lblWQ = MathTex("WQ", font_size=36, color=INK).move_to(cR + np.array([1.2, -2.05, 0]))
            lblWX = MathTex("WX", font_size=34, color=INK).move_to(
                Wmap(cL + XP[0]) + np.array([-0.95, 0.35, 0]))
            lblWY = MathTex("WY", font_size=34, color=INK).move_to(
                Wmap(cL + YP[0]) + np.array([1.05, -0.35, 0]))
            self.add(lblWQ, lblWX, lblWY)
            self.wait(0.6)

            # ---------- M carries only what X alone uses; the image under W recedes ----------
            grey = ManimColor("#b5bac4")
            self.remove(lblWX)
            self.play(Qc.animate.set_fill(pale(QF)).set_stroke(pale(QS, 0.6)),
                      Xc.animate.set_fill(pale(XF)).set_stroke(pale(XS, 0.6)),
                      Yc.animate.set_fill(pale(YF)).set_stroke(pale(YS, 0.6)),
                      Oc.animate.set_fill(pale(OF)),
                      Xoc.animate.set_stroke(pale(XS, 0.6)), Yoc.animate.set_stroke(pale(YS, 0.6)),
                      arrowW.animate.set_color(grey), lblW.animate.set_color(grey),
                      lblWQ.animate.set_color(grey), lblWY.animate.set_color(grey),
                      run_time=0.45)

            arrowM = Arrow(cL + np.array([1.75, 0.55, 0]),
                           Wmap(cL + XP[0]) + np.array([-0.75, -0.25, 0]),
                           stroke_width=5, tip_length=0.28, color=INK, buff=0)
            lblM = MathTex("M", font_size=44, color=INK).next_to(arrowM.get_center(), UP, buff=0.15)
            self.play(GrowArrow(arrowM), run_time=0.4)
            self.add(lblM)

            # The three regions travel under M, each as strongly as the solve says M
            # carries its directions: X alone ~1, the overlap a little, Y alone ~0.
            pieces = [
                (Difference(X, Y), XF, share["x"]),
                (Intersection(X, Y), OF, share["shared"]),
                (Difference(Y, X), YF, share["y"]),
            ]
            movers, targets = [], []
            for shape, fill, s in pieces:
                s = float(np.clip(s, 0.0, 1.0))
                m = shape.set_stroke(XS, width=3, opacity=s).set_fill(fill, opacity=s)
                m.set_z_index(5)
                movers.append(m)
                targets.append(image(m).set_stroke(INK, width=6, opacity=s if s > 0.5 else 0.0))
            self.add(*movers)
            self.play(*[Transform(m, tg) for m, tg in zip(movers, targets)],
                      run_time=1.3, path_arc=-20 * DEGREES)
            lblMX = MathTex("MX", font_size=38, color=INK).move_to(lblWX).shift(0.05 * LEFT)
            self.add(lblMX)
            self.wait(1.6)

    return IsolateGif


def render(out=HERE / "demo.gif", *, width=560, height=420, fps=20, colors=96, seed=SEED,
           mu_per_token=MU_PER_TOKEN) -> pathlib.Path:
    """Render the scene to ``out`` (a GIF). Needs manim, LaTeX and ffmpeg.

    The scene is rendered to video and converted with one fixed palette and no
    dithering: the picture is flat colour, so this is both sharper and several
    times smaller than a dithered GIF. ``gifsicle``, when installed, squeezes
    it further without changing a pixel."""
    import shutil
    import subprocess
    import tempfile

    from manim import tempconfig

    torch.manual_seed(seed)
    data = demo_data(seed, mu_per_token)
    out = pathlib.Path(out)
    with tempfile.TemporaryDirectory() as tmp:
        with tempconfig({"frame_rate": fps, "pixel_width": width, "pixel_height": height,
                         "frame_width": 11.6, "frame_height": 8.7, "background_color": "#ffffff",
                         "format": "mp4", "media_dir": tmp, "output_file": "demo",
                         "verbosity": "WARNING", "disable_caching": True,
                         "progress_bar": "none"}):
            build_scene(data)().render()
        video = next(pathlib.Path(tmp).rglob("demo.mp4"))
        gif = pathlib.Path(tmp) / "demo.gif"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(video), "-filter_complex",
             f"[0:v]split[a][b];[a]palettegen=max_colors={colors}:stats_mode=full[p];"
             "[b][p]paletteuse=dither=none", "-loop", "0", str(gif)], check=True)
        if shutil.which("gifsicle"):
            subprocess.run(["gifsicle", "-O3", str(gif), "-o", str(gif)], check=True)
        shutil.copyfile(gif, out)
    return out


if __name__ == "__main__":
    d = demo_data()
    print("share of each region's directions that M carries:",
          {k: round(v, 3) for k, v in d["share"].items()})
    print("wrote", render())
