from typing import List, Union

import numpy as np
import numpy.typing as npt
import scipy

# std dev of impulse indicating a hit
HIT_SD = 3


def sigmoid(x: npt.ArrayLike) -> npt.ArrayLike:
    """1/(e^-x + 1)"""
    return np.exp(-np.logaddexp(-x, 0))


def encode_hit(sig: npt.ArrayLike, frame_times: npt.ArrayLike, i: float) -> None:
    z = (frame_times - i) / HIT_SD

    # hits are impulses
    # sig += 2 * np.exp(-.5 * z**2)

    # hits are flips
    sig *= 1 - 2 * sigmoid(z)


def encode_hold(
    sig: npt.ArrayLike,
    frame_times: npt.ArrayLike,
    i: Union[float, int],
    j: Union[float, int],
) -> None:
    m = 2 * sigmoid((j - i) / 2 / HIT_SD) - 1  # maximum value at (j-i)/2
    sig += (
        2
        * (sigmoid((frame_times - i) / HIT_SD) - sigmoid((frame_times - j) / HIT_SD))
        / m
    )


def flips(sig: npt.ArrayLike) -> List[npt.ArrayLike]:
    sig_grad = np.gradient(sig)
    return (
        scipy.signal.find_peaks(sig_grad, height=0.5)[0].astype(np.int32),
        scipy.signal.find_peaks(-sig_grad, height=0.5)[0].astype(np.int32),
    )


def decode_hit(sig: npt.ArrayLike) -> List[npt.ArrayLike]:
    rising, falling = flips(sig)
    return sorted([*rising, *falling])


def decode_hold(sig: npt.ArrayLike) -> List[npt.ArrayLike]:
    rising, falling = flips(sig)
    start_idxs, end_idxs = list(rising), list(falling)

    # ensure that first start is before first end
    while len(start_idxs) and len(end_idxs) and start_idxs[0] >= end_idxs[0]:
        end_idxs.pop(0)

    # ensure that there is one end for every start
    if len(start_idxs) > len(end_idxs):
        start_idxs = start_idxs[: len(end_idxs) - len(start_idxs)]
    elif len(end_idxs) > len(start_idxs):
        end_idxs = end_idxs[: len(start_idxs) - len(end_idxs)]

    return start_idxs, end_idxs
