import bisect
import re
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple

import numpy as np

from osu_vqvae.osu.hit_objects import Circle, Slider, Spinner, Timed, TimingPoint
from osu_vqvae.osu.sliders import from_control_points


class Beatmap:
    def __init__(self: "Beatmap", filename: Path, meta_only: bool = False) -> None:
        self.filename = Path(filename)
        with open(filename, encoding="utf-8") as f:
            cfg = self.parse_map_file(f)

        self.audio_filename = self.filename.parent / cfg["General"]["AudioFilename"]

        self.mode = int(cfg["General"]["Mode"])

        self.title = cfg["Metadata"]["Title"]
        self.artist = cfg["Metadata"]["Artist"]
        self.creator = cfg["Metadata"]["Creator"]
        self.version = cfg["Metadata"]["Version"]
        try:
            self.mapset_id = int(cfg["Metadata"]["BeatmapSetID"])
        except KeyError:
            self.mapset_id = None

        self.hp = float(cfg["Difficulty"]["HPDrainRate"])
        self.cs = float(cfg["Difficulty"]["CircleSize"])
        self.od = float(cfg["Difficulty"]["OverallDifficulty"])
        try:
            self.ar = float(cfg["Difficulty"]["ApproachRate"])
        except KeyError:
            self.ar = 7

        self.slider_mult = float(cfg["Difficulty"]["SliderMultiplier"])
        self.slider_tick = float(cfg["Difficulty"]["SliderTickRate"])

        try:
            self.beat_divisor = int(cfg["Editor"]["BeatDivisor"])
        except KeyError:
            self.beat_divisor = 4

        self.unparsed_hitobjects = cfg["HitObjects"]
        self.unparsed_timingpoints = cfg["TimingPoints"]
        self.unparsed_events = cfg["Events"]

        if not meta_only:
            self.parse_map_data()

    def parse_timing_points(self: "Beatmap", lines: List[str]) -> None:
        self.timing_points = []
        self.uninherited_timing_points = []

        cur_beat_length = None
        cur_slider_mult = 1.0
        cur_meter = None

        for line in lines:
            vals = [float(x) for x in line.strip().split(",")]
            t, x, meter = vals[:3]
            kiai = int(vals[7] if len(vals) >= 8 else 0) % 2 == 1

            if vals[6] == 0:
                if len(self.timing_points) == 0:
                    continue

                if self.timing_points[-1].t == t:
                    self.timing_points.pop()

                cur_slider_mult = min(10.0, max(0.1, round(-100 / float(x), 3)))
            else:
                cur_beat_length = x
                cur_slider_mult = 1.0
                cur_meter = meter

            tp = TimingPoint(int(t), cur_beat_length, cur_slider_mult, cur_meter, kiai)
            if len(self.timing_points) == 0 or tp != self.timing_points[-1]:
                self.timing_points.append(tp)

            utp = TimingPoint(int(t), cur_beat_length, None, cur_meter, None)
            if (
                len(self.uninherited_timing_points) == 0
                or utp != self.uninherited_timing_points[-1]
            ):
                self.uninherited_timing_points.append(utp)

        if len(self.timing_points) == 0:
            msg = "no uninherited timing points"
            raise ValueError(msg)

    def get_active_timing_point(self: "Beatmap", t: int) -> TimingPoint:
        idx = bisect.bisect(self.timing_points, Timed(t)) - 1
        if idx < 0:
            idx = 0

        return self.timing_points[idx]

    def parse_hit_objects(self: "Beatmap", lines: List[str]) -> None:
        self.hit_objects = []
        for line in lines:
            spl = line.strip().split(",")
            x, y, t, k = [int(x) for x in spl[:4]]
            new_combo = (k & (1 << 2)) > 0
            if k & (1 << 0):
                ho = Circle(t, new_combo, x, y)
            elif k & (1 << 1):
                curve, slides, length = spl[5:8]
                _, *control_points = curve.split("|")
                control_points = [np.array([x, y])] + [
                    np.array(list(map(int, p.split(":")))) for p in control_points
                ]

                tp = self.get_active_timing_point(t)
                ho = from_control_points(
                    t,
                    tp.beat_length,
                    self.slider_mult * tp.slider_mult,
                    new_combo,
                    int(slides),
                    float(length),
                    control_points,
                )
            elif k & (1 << 3):
                ho = Spinner(t, new_combo, int(spl[5]))

            if len(self.hit_objects) and ho.t < self.hit_objects[-1].end_time():
                msg = f"hit object starts before previous hit object ends: {t}"
                raise ValueError(
                    msg,
                )

            self.hit_objects.append(ho)

        if len(self.hit_objects) == 0:
            msg = "no hit objects"
            raise ValueError(msg)

    def parse_events(self: "Beatmap", lines: List[str]) -> None:
        self.events = []
        for line in lines:
            ev = line.strip().split(",")
            if ev[0] == 2:
                self.events.append(ev)

    def parse_map_data(self: "Beatmap") -> None:
        self.parse_timing_points(self.unparsed_timingpoints)
        del self.unparsed_timingpoints

        self.parse_hit_objects(self.unparsed_hitobjects)
        del self.unparsed_hitobjects

        self.parse_events(self.unparsed_events)
        del self.unparsed_events

    def cursor(self: "Beatmap", t: int) -> Tuple[Tuple[int, int], float]:  # noqa: C901
        """
        return cursor position + time since last click at time t (ms)
        """

        cx, cy = 256, 192

        # before first hit object
        if t < self.hit_objects[0].t:
            ho = self.hit_objects[0]
            if isinstance(ho, Circle):
                return (ho.x, ho.y), np.inf
            elif isinstance(ho, Slider):
                return (cx, cy), np.inf
            elif isinstance(ho, Slider):
                return ho.lerp(0), np.inf

        for ho, nho in zip(self.hit_objects, self.hit_objects[1:]):
            if ho.t <= t < nho.t:
                break
        else:
            ho = self.hit_objects[-1]
            nho = None

        t -= ho.t

        # next hit object
        if isinstance(nho, Circle):
            nx, ny = nho.x, nho.y
        elif isinstance(nho, Spinner):
            nx, ny = (cx, cy)  # spin starting point
        elif isinstance(nho, Slider):
            nx, ny = nho.lerp(0)

        if isinstance(ho, Spinner):
            spin_duration = ho.u - ho.t
            if t < spin_duration:  # spinning
                return (cx, cy), 0
            else:  # moving
                t -= spin_duration
                if nho:  # to next hit object
                    f = t / (nho.t - ho.t - spin_duration)  # interpolation factor
                    return ((1 - f) * cx + f * nx, (1 - f) * cy + f * ny), t
                else:  # last object
                    return (cx, cy), t

        elif isinstance(ho, Circle):
            if nho:  # moving to next hit object
                f = t / (nho.t - ho.t)  # interpolation factor

                return ((1 - f) * ho.x + f * nx, (1 - f) * ho.y + f * ny), t
            else:
                return (ho.x, ho.y), t
        elif isinstance(ho, Slider):
            slide_duration = ho.slide_duration

            if t < slide_duration:  # sliding
                single_slide = slide_duration / ho.slides

                ts = t % (single_slide * 2)
                if ts < single_slide:  # start -> end
                    return ho.lerp(ts / single_slide), 0
                else:  # end -> start
                    return ho.lerp(2 - ts / single_slide), 0
            else:  # moving
                t -= slide_duration
                end = ho.lerp(ho.slides % 2)

                if nho:  # to next hit object
                    f = t / (nho.t - ho.t - slide_duration)  # interpolation factor
                    return ((1 - f) * end[0] + f * nx, (1 - f) * end[1] + f * ny), t
                else:
                    return (end[0], end[1]), t

    @staticmethod
    def parse_map_file(bmlines: List[str]) -> Dict[str, Any]:
        list_sections = ["Events", "TimingPoints", "HitObjects"]
        cfg = {}
        section = None
        for line in bmlines:
            if line.startswith("//"):
                continue

            if line.strip() == "":
                section = None
                continue

            m = re.search(r"^\[(.*)\]$", line)
            if m is not None:
                section = m.group(1)
                if section in list_sections:
                    cfg[section] = []
                else:
                    cfg[section] = {}
                continue

            if section is None:
                continue

            if section in list_sections:
                cfg[section].append(line.strip())
            else:
                m = re.search(r"^(\w*)\s?:\s?(.*)$", line)
                if m is not None:
                    cfg[section][m.group(1)] = m.group(2).strip()
        return cfg

    @classmethod
    def all_maps(cls: "Beatmap", src_path: str) -> Generator["Beatmap", None, None]:
        for path in Path(src_path).glob("*/*.osu"):
            try:
                bm = Beatmap(path)
            except Exception as e:
                print(f"{path}: {e}")
                continue

            if bm.mode != 0:
                continue

            yield bm

    @classmethod
    def all_mapsets(  # noqa: C901
        cls: "Beatmap",
        src_path: str,
    ) -> Generator[Tuple[int, str, List["Beatmap"]], None, None]:
        for mapset_dir in Path(src_path).iterdir():
            if not mapset_dir.is_dir():
                continue

            maps = []
            mapset_id = None
            audio_file = None
            for map_file in mapset_dir.glob("*.osu"):
                try:
                    bm = Beatmap(map_file)
                except Exception as e:
                    print(f"{map_file}: {e}")
                    continue

                if bm.mode != 0:
                    continue

                maps.append(bm)

                if audio_file is None:
                    audio_file = bm.audio_filename
                elif audio_file != bm.audio_filename:
                    print(f"{map_file}: audio file mismatch")
                    break

                if mapset_id is None:
                    mapset_id = bm.mapset_id
                elif mapset_id != bm.mapset_id:
                    print(f"{map_file}: mapset id mismatch")
                    break
            else:
                if audio_file is None or mapset_id is None or len(maps) == 0:
                    continue
                yield (mapset_id, audio_file, maps)

        return
