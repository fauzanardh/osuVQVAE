import numpy as np
import numpy.typing as npt


class Timed:
    def __init__(self: "Timed", t: int) -> None:
        self.t = t

    def __lt__(self: "Timed", other: "Timed") -> bool:
        return self.t < other.t


class TimingPoint(Timed):
    def __init__(
        self: "TimingPoint",
        t: int,
        beat_length: float,
        slider_mult: float,
        meter: int,
        kiai: bool,
    ) -> None:
        super().__init__(t)
        self.beat_length = beat_length
        self.slider_mult = slider_mult
        self.meter = meter
        self.kiai = kiai

    def __eq__(self: "TimingPoint", other: "TimingPoint") -> bool:
        return all(
            [
                self.beat_length == other.beat_length,
                self.slider_mult == other.slider_mult,
                self.meter == other.meter,
                self.kiai == other.kiai,
            ],
        )


class HitObject(Timed):
    def __init__(self: "HitObject", t: int, new_combo: bool) -> None:
        super().__init__(t)
        self.new_combo = new_combo

    def end_time(self: "HitObject") -> int:
        raise NotImplementedError

    def start_pos(self: "HitObject") -> npt.ArrayLike:
        raise NotImplementedError

    def end_pos(self: "HitObject") -> npt.ArrayLike:
        return self.start_pos()


class Circle(HitObject):
    def __init__(self: "Circle", t: int, new_combo: bool, x: int, y: int) -> None:
        super().__init__(t, new_combo)
        self.x = x
        self.y = y

    def end_time(self: "Circle") -> int:
        return self.t

    def start_pos(self: "Circle") -> npt.ArrayLike:
        return np.array([self.x, self.y])


class Spinner(HitObject):
    def __init__(self: "Spinner", t: int, new_combo: bool, u: int) -> None:
        super().__init__(t, new_combo)
        self.u = u

    def end_time(self: "Spinner") -> int:
        return self.u

    def start_pos(self: "Spinner") -> npt.ArrayLike:
        return np.array([256, 192])


class Slider(HitObject):
    def __init__(
        self: "Slider",
        t: int,
        beat_length: float,
        slider_mult: float,
        new_combo: bool,
        slides: int,
        length: float,
    ) -> None:
        super().__init__(t, new_combo)
        self.slides = slides
        self.length = length
        self.slider_mult = slider_mult
        self.slide_duration = length / (slider_mult * 100) * beat_length * slides

    def end_time(self: "Slider") -> int:
        return int(self.t + self.slide_duration)

    def lerp(self: "Slider", _: float) -> npt.ArrayLike:
        raise NotImplementedError

    def start_pos(self: "Slider") -> npt.ArrayLike:
        return self.lerp(0.0)

    def end_pos(self: "Slider") -> npt.ArrayLike:
        return self.lerp(self.slides % 2)
