from typing import List, Tuple

import bezier
import numpy as np

from osu_vqvae.osu.hit_objects import NDIntArray, Slider


def approx_eq(a: int, b: int) -> bool:
    return abs(a - b) < 1e-8


def binom_coeffs(n: int) -> List[float]:
    out = []
    c = 1.0
    for k in range(n + 1):
        out.append(c)
        c = c * (n + 1) / (k + 1) - c
    return out


class Line(Slider):
    def __init__(
        self: "Line",
        t: int,
        beat_length: float,
        sldier_mult: float,
        new_combo: bool,
        slides: int,
        length: float,
        start: NDIntArray,
        end: NDIntArray,
    ) -> None:
        super().__init__(t, beat_length, sldier_mult, new_combo, slides, length)
        self.start = start

        vec = end - self.start
        self.end = self.start + vec / np.linalg.norm(vec) * length

    def lerp(self: "Line", t: float) -> NDIntArray:
        out = (1 - t) * self.start + t * self.end
        return out.round(0).astype(np.int32)

    def vel(self: "Line", t: float) -> NDIntArray:
        out = (self.end - self.start) / (self.slide_duration / self.slides)
        return out.round(0).astype(np.int32)


class Perfect(Slider):
    def __init__(
        self: "Perfect",
        t: int,
        beat_length: float,
        slider_mult: float,
        new_combo: bool,
        slides: int,
        length: float,
        center: NDIntArray,
        radius: float,
        start: float,
        end: float,
    ) -> None:
        super().__init__(t, beat_length, slider_mult, new_combo, slides, length)
        self.center = center
        self.radius = radius
        self.start = start

        self.end = start + length / radius * np.sign(end - start)

    def lerp(self: "Perfect", t: float) -> NDIntArray:
        theta = (1 - t) * self.start + t * self.end
        out = self.center + self.radius * np.array([np.cos(theta), np.sin(theta)])
        return out.round(0).astype(np.int32)

    def vel(self: "Perfect", t: float) -> NDIntArray:
        theta = (1 - t) * self.start + t * self.end
        out = (
            self.radius
            * np.array([-np.sin(theta), np.cos(theta)])
            / (self.slide_duration / self.slides)
        )
        return out.round(0).astype(np.int32)


class Bezier(Slider):
    SEG_LEN = 10

    def __init__(
        self: "Bezier",
        t: int,
        beat_length: float,
        slider_mult: float,
        new_combo: bool,
        slides: int,
        length: float,
        control_points: List[NDIntArray],
    ) -> None:
        super().__init__(t, beat_length, slider_mult, new_combo, slides, length)
        self.control_points = control_points

        control_curves: List[List[NDIntArray]] = []
        last_idx = 0
        for i, point in enumerate(control_points[1:]):
            if (control_points[i] == point).all():
                control_curves.append(control_points[last_idx : i + 1])
                last_idx = i + 1
        control_curves.append(control_points[last_idx:])

        total_len = 0
        curves = []
        for curve in control_curves:
            if len(curve) < 2:
                continue  # invalid bezier

            nodes = np.array(curve).T
            bezier_curve = bezier.Curve.from_nodes(nodes)
            total_len += bezier_curve.length
            curves.append(bezier_curve)

        tail_len = self.length - total_len
        if tail_len > 0:
            last_curve_nodes = curves[-1].nodes
            point = last_curve_nodes[:, -1]
            vec = point - last_curve_nodes[:, -2]

            nodes = np.array([point, point + vec / np.linalg.norm(vec) * tail_len]).T
            bezier_curve = bezier.Curve.from_nodes(nodes)

            assert approx_eq(
                bezier_curve.length,
                tail_len,
            ), f"{bezier_curve.length} != {tail_len}"
            curves.append(bezier_curve)

        self.path_segments = curves
        self.cum_t = np.cumsum([c.length for c in curves]) / self.length
        self.cum_t[-1] = 1.0

    def curve_reparameterize(self: "Bezier", t: float) -> Tuple[int, float]:
        idx = np.searchsorted(self.cum_t, min(1, max(0, t)))
        assert idx < len(self.cum_t), f"{idx} >= {len(self.cum_t)}"

        range_start = np.insert(self.cum_t, 0, 0)[idx]
        range_end = self.cum_t[idx]

        t = (t - range_start) / (range_end - range_start)
        return int(idx), t

    def lerp(self: "Bezier", t: float) -> NDIntArray:
        idx, t = self.curve_reparameterize(t)
        return self.path_segments[idx].evaluate(t)[:, 0].round(0).astype(np.int32)

    def vel(self: "Bezier", t: float) -> NDIntArray:
        idx, t = self.curve_reparameterize(t)
        out = self.path_segments[idx].evaluate_hodograph(t)[:, 0] / (
            self.slide_duration / self.slides
        )
        return out.round(0).astype(np.int32)


def from_control_points(
    t: int,
    beat_length: float,
    slider_mult: float,
    new_combo: bool,
    slides: int,
    length: float,
    control_points: List[NDIntArray],
) -> Slider:
    assert len(control_points) >= 2, "control points must have at least 2 points"

    if len(control_points) == 2:  # L type
        _a, _b = control_points
        return Line(t, beat_length, slider_mult, new_combo, slides, length, _a, _b)
    if len(control_points) == 3:  # P type
        _a, _b, _c = control_points

        if (_b == _c).all():
            return Line(t, beat_length, slider_mult, new_combo, slides, length, _a, _c)

        _abc = np.cross(_b - _a, _c - _b)

        if _abc == 0:  # collinear
            if np.dot(_b - _a, _c - _b) > 0:
                return Line(
                    t,
                    beat_length,
                    slider_mult,
                    new_combo,
                    slides,
                    length,
                    _a,
                    _c,
                )
            else:
                control_points.insert(1, control_points[1])
                return Bezier(
                    t,
                    beat_length,
                    slider_mult,
                    new_combo,
                    slides,
                    length,
                    control_points,
                )

        a = np.linalg.norm(_c - _b)
        b = np.linalg.norm(_c - _a)
        c = np.linalg.norm(_b - _a)
        s = (a + b + c) / 2.0
        _r = a * b * c / 4.0 / np.sqrt(s * (s - a) * (s - b) * (s - c))

        if _r > 320 and np.dot(_c - _b, _b - _a) < 0:  # circle too large
            return Bezier(
                t,
                beat_length,
                slider_mult,
                new_combo,
                slides,
                length,
                control_points,
            )

        b1 = a * a * (b * b + c * c - a * a)
        b2 = b * b * (a * a + c * c - b * b)
        b3 = c * c * (a * a + b * b - c * c)
        _p = np.column_stack((_a, _b, _c)).dot(np.hstack((b1, b2, b3)))
        _p /= b1 + b2 + b3

        start_angle = np.arctan2(*(_a - _p)[[1, 0]])
        end_angle = np.arctan2(*(_c - _p)[[1, 0]])

        if _abc < 0:  # clockwise
            while end_angle > start_angle:
                end_angle -= 2 * np.pi
        else:  # counter-clockwise
            while start_angle > end_angle:
                start_angle -= 2 * np.pi

        return Perfect(
            t,
            beat_length,
            slider_mult,
            new_combo,
            slides,
            length,
            _p,
            _r,
            start_angle,
            end_angle,
        )
    else:
        return Bezier(
            t,
            beat_length,
            slider_mult,
            new_combo,
            slides,
            length,
            control_points,
        )
