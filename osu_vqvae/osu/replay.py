import lzma
import struct
import datetime
from pathlib import Path
from enum import Enum, IntFlag
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np


class GameMode(Enum):
    STD = 0
    TAIKO = 1
    CTB = 2
    MANIA = 3


class Key(IntFlag):
    M1 = 1 << 0
    M2 = 1 << 1
    K1 = 1 << 2
    K2 = 1 << 3
    SMOKE = 1 << 4


@dataclass
class ReplayEventOsu:
    time_delta: int
    x: float
    y: float
    keys: Key


@dataclass
class LifeBarState:
    time: int
    life: float


class _unpacker:
    def __init__(self, replay_data: bytes):
        self.replay_data = replay_data
        self.offset = 0

    def string_length(self) -> int:
        result = 0
        shift = 0
        while True:
            b = self.replay_data[self.offset]
            self.offset += 1
            result |= (b & 0x7F) << shift
            if not b & 0x80:
                break
            shift += 7
        return result

    def unpack_string(self) -> Optional[str]:
        if self.replay_data[self.offset] == 0x0:
            self.offset += 1
            return None
        elif self.replay_data[self.offset] == 0xB:
            self.offset += 1
            length = self.string_length()
            result = self.replay_data[self.offset : self.offset + length].decode(
                "utf-8"
            )
            self.offset += length
            return result
        else:
            raise ValueError("Invalid string")

    def unpack_once(self, fmt: str):
        specifier = f"<{fmt}"
        unpacked = struct.unpack_from(specifier, self.replay_data, self.offset)
        self.offset += struct.calcsize(specifier)
        return unpacked[0]

    def unpack_timestamp(self) -> datetime.datetime:
        ticks = self.unpack_once("q")
        timestamp = datetime.datetime.min + datetime.timedelta(microseconds=ticks / 10)
        timestamp = timestamp.replace(tzinfo=datetime.timezone.utc)
        return timestamp

    @staticmethod
    def parse_replay_data(
        replay_data_str: str,
    ) -> Tuple[Optional[int], List[ReplayEventOsu]]:
        replay_data_str = replay_data_str.rstrip(",")
        events = [event.split("|") for event in replay_data_str.split(",")]

        rng_seed = None
        play_data = []
        for event in events:
            time_delta = int(event[0])
            x = float(event[1])
            y = float(event[2])
            keys = int(event[3])

            if time_delta == -12345 and event == events[-1]:
                rng_seed = keys
                continue

            play_data.append(ReplayEventOsu(time_delta, x, y, Key(keys)))
        return rng_seed, play_data

    def unpack_replay_data(self) -> Tuple[Optional[int], List[ReplayEventOsu]]:
        length = self.unpack_once("i")
        data = self.replay_data[self.offset : self.offset + length]
        data = lzma.decompress(data, format=lzma.FORMAT_AUTO)
        data_str = data.decode("ascii")
        self.offset += length
        return self.parse_replay_data(data_str)

    def unpack_replay_id(self) -> int:
        try:
            replay_id = self.unpack_once("q")
        except struct.error:
            replay_id = self.unpack_once("l")
        return replay_id

    def unpack_life_bar(self) -> Optional[List[LifeBarState]]:
        lifebar = self.unpack_string()
        if not lifebar:
            return None

        lifebar = lifebar.rstrip(",")
        states = [state.split("|") for state in lifebar.split(",")]

        return [LifeBarState(int(state[0]), float(state[1])) for state in states]


class Replay:
    def __init__(self, replay_path: str, to_np: bool = True):
        self._unpacker = _unpacker(Path(replay_path).read_bytes())

        # Only store the replay data we need
        if GameMode(self._unpacker.unpack_once("b")) != GameMode.STD:
            raise ValueError("Only std replays are supported")
        self._unpacker.unpack_once("i")  # game_version
        self._unpacker.unpack_string()  # beatmap_hash
        self._unpacker.unpack_string()  # username
        self._unpacker.unpack_string()  # replay_hash
        self._unpacker.unpack_once("h")  # count_300
        self._unpacker.unpack_once("h")  # count_100
        self._unpacker.unpack_once("h")  # count_50
        self._unpacker.unpack_once("h")  # count_geki
        self._unpacker.unpack_once("h")  # count_katu
        self._unpacker.unpack_once("h")  # count_miss
        self._unpacker.unpack_once("i")  # score
        self._unpacker.unpack_once("h")  # max_combo
        self._unpacker.unpack_once("?")  # perfect
        self._unpacker.unpack_once("i")  # mods
        self._unpacker.unpack_life_bar()  # life_bar
        self._unpacker.unpack_timestamp()  # timestamp
        (
            _,
            self._replay_data,
        ) = self._unpacker.unpack_replay_data()  # rng_seed, replay_data
        self._unpacker.unpack_replay_id()  # replay_id

        # delete unpacker to free memory
        del self._unpacker

        # convert replay data to numpy array
        self.to_np = to_np
        if self.to_np:
            self.replay_data_to_np()

    def replay_data_to_np(self):
        t = 0
        # ignoring keys for now
        arr = np.zeros((len(self._replay_data), 3), dtype=np.float32)
        for i, event in enumerate(self._replay_data):
            t += event.time_delta
            arr[i, 0] = float(t)
            arr[i, 1] = event.x
            arr[i, 2] = event.y
        # sort by time
        self._replay_data = arr[arr[:, 0].argsort()]

    def cursor(self, t):
        """
        interpolates linearly between events
        return cursor position + time since last click at time t (ms)
        """

        assert self.to_np, "Replay data must be converted to numpy array"

        # find closest event before t
        idx = np.searchsorted(self._replay_data[:, 0], t, side="right") - 1
        if idx < 0:
            raise ValueError("t is before first event")

        # if t is after last event, return last event
        if idx == len(self._replay_data) - 1:
            return (self._replay_data[idx, 1], self._replay_data[idx, 2]), 0

        # interpolate between events
        t0, x0, y0 = self._replay_data[idx]
        t1, x1, y1 = self._replay_data[idx + 1]
        alpha = (t - t0) / (t1 - t0)
        return (x0 + alpha * (x1 - x0), y0 + alpha * (y1 - y0)), t1 - t
