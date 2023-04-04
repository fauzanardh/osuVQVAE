import numpy as np
import numpy.typing as npt


NDIntArray = npt.NDArray[np.int32]


class Timed:
    def __init__(self, t: int):
        self.t = t

    def __lt__(self, other):
        return self.t < other.t


class TimingPoint(Timed):
    def __init__(
        self, t: int, beat_length: float, slider_mult: float, meter: int, kiai: bool
    ):
        super().__init__(t)
        self.beat_length = beat_length
        self.slider_mult = slider_mult
        self.meter = meter
        self.kiai = kiai

    def __eq__(self, other):
        return all(
            [
                self.beat_length == other.beat_length,
                self.slider_mult == other.slider_mult,
                self.meter == other.meter,
                self.kiai == other.kiai,
            ]
        )


class HitObject(Timed):
    def __init__(self, t: int, new_combo: bool):
        super().__init__(t)
        self.new_combo = new_combo

    def end_time(self) -> int:
        raise NotImplementedError

    def start_pos(self) -> NDIntArray:
        raise NotImplementedError

    def end_pos(self) -> NDIntArray:
        return self.start_pos()


class Circle(HitObject):
    def __init__(self, t: int, new_combo: bool, x: int, y: int):
        super().__init__(t, new_combo)
        self.x = x
        self.y = y

    def end_time(self) -> int:
        return self.t

    def start_pos(self) -> NDIntArray:
        return np.array([self.x, self.y])


class Spinner(HitObject):
    def __init__(self, t: int, new_combo: bool, u: int):
        super().__init__(t, new_combo)
        self.u = u

    def end_time(self) -> int:
        return self.u

    def start_pos(self) -> NDIntArray:
        return np.array([256, 192])


class Slider(HitObject):
    def __init__(
        self,
        t: int,
        beat_length: float,
        slider_mult: float,
        new_combo: bool,
        slides: int,
        length: float,
    ):
        super().__init__(t, new_combo)
        self.slides = slides
        self.length = length
        self.slider_mult = slider_mult
        self.slide_duration = length / (slider_mult * 100) * beat_length * slides

    def end_time(self) -> int:
        return int(self.t + self.slide_duration)

    def lerp(self, _: float) -> NDIntArray:
        raise NotImplementedError

    def start_pos(self) -> NDIntArray:
        return self.lerp(0.0)

    def end_pos(self) -> NDIntArray:
        return self.lerp(self.slides % 2)
