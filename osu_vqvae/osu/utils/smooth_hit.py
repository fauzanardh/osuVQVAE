from typing import List, Tuple, Union

import numpy as np
import numpy.typing as npt
import scipy

# std dev of impulse indicating a hit
HIT_SD = 5


def smooth_hit(
    x: npt.ArrayLike,
    mu: Union[int, float, Tuple[float, float]],
    sigma: float = HIT_SD,
) -> npt.ArrayLike:
    if isinstance(mu, (float, int)):
        z = (x - mu) / sigma
    elif isinstance(mu, tuple):
        start, end = mu
        z = np.where(x < start, x - start, np.where(x < end, 0, x - end)) / sigma
    else:
        msg = f"mu must be float or tuple, not {type(mu)}"
        raise TypeError(msg)

    return np.exp(-0.5 * z**2)


_feature_bound = max(2, HIT_SD * 6)
_feature = smooth_hit(np.arange(-_feature_bound, _feature_bound + 1), 0)


def _decode(
    sig: npt.ArrayLike,
    peak_h: float,
    hit_offset: Union[int, float],
) -> npt.ArrayLike:
    correlation = scipy.signal.correlate(sig, _feature, mode="same")
    hit_peaks = scipy.signal.find_peaks(correlation, height=peak_h)[0] + hit_offset
    return hit_peaks.astype(np.int32).tolist()


def decode_hit(sig: npt.ArrayLike) -> List[int]:
    return _decode(sig, peak_h=0.5, hit_offset=0)


def decode_hold(sig: npt.ArrayLike) -> List[int]:
    sig_grad = np.gradient(sig)
    start_sig = np.maximum(0, sig_grad)
    end_sig = -np.minimum(0, sig_grad)

    start_idxs = _decode(start_sig, peak_h=0.25, hit_offset=1)
    end_idxs = _decode(end_sig, peak_h=0.25, hit_offset=-1)

    # ensure that first start is before first end
    while len(start_idxs) and len(end_idxs) and start_idxs[0] >= end_idxs[0]:
        end_idxs.pop(0)

    # ensure that there is one end for each start
    if len(start_idxs) > len(end_idxs):
        start_idxs = start_idxs[: len(end_idxs) - len(start_idxs)]
    elif len(end_idxs) > len(start_idxs):
        end_idxs = end_idxs[: len(start_idxs) - len(end_idxs)]

    return start_idxs, end_idxs
