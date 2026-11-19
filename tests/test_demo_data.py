"""The README animation (``assets/make_demo.py``): its data half runs without
manim and says what the picture claims, and the committed GIF is small and
short enough to sit at the top of a README."""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("numpy")

ASSETS = pathlib.Path(__file__).resolve().parents[1] / "assets"


@pytest.fixture(scope="module")
def demo():
    spec = importlib.util.spec_from_file_location("_rspd_make_demo", ASSETS / "make_demo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)            # must not need manim to import
    return mod


def test_the_solve_carries_x_alone(demo):
    d = demo.demo_data()
    assert d["share"]["x"] > 0.9            # X's own directions travel under M
    assert d["share"]["shared"] < 0.3       # the overlap is mostly left behind
    assert d["share"]["y"] < 0.05           # Y's own directions do not travel
    assert d["groups"] == ["x", "x", "shared", "y", "y"] and len(d["carried"]) == 5
    assert d["leaks"]["L_Y_rel"] < 0.1


def test_a_larger_dose_takes_more_of_the_overlap(demo):
    small, large = demo.demo_data(mu_per_token=0.03), demo.demo_data(mu_per_token=3.0)
    assert large["share"]["shared"] > small["share"]["shared"] + 0.5


def test_the_column_shows_the_point_s_activation(demo):
    d = demo.demo_data()
    in_x, in_y, both = d["activation"](1, 0), d["activation"](0, 1), d["activation"](1, 1)
    assert in_x[3] == in_x[4] == 0 and in_y[0] == in_y[1] == 0
    assert all(v != 0 for v in both)
    assert d["activation"](1, 1, 0.0) != d["activation"](1, 1, 2.0)      # it moves


def test_the_committed_gif_is_small_and_short(demo):
    gif = ASSETS / "demo.gif"
    assert gif.is_file(), "render it with: python assets/make_demo.py"
    assert gif.stat().st_size < 2 * 2 ** 20
    Image = pytest.importorskip("PIL.Image")
    with Image.open(gif) as im:
        ms = 0
        for i in range(im.n_frames):
            im.seek(i)
            ms += im.info.get("duration", 0)
        assert im.size == (560, 420) and 3_000 < ms < 10_000, ms
        assert im.info.get("loop", 1) == 0                              # loops forever
