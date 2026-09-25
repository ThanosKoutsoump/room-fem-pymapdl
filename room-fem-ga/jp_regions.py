from dataclasses import dataclass
import numpy as np
from scipy.signal import find_peaks


@dataclass
class Region:
    kind: str      # "peak" or "valley"
    f_lo: float     # region lower bound, Hz
    f_hi: float     # region upper bound, Hz
    f_rep: float     # representative frequency (peak or valley), Hz
    idx_rep: int      # index into the freqs array for f_rep
    idx_lo: int
    idx_hi: int


def _drop_boundary(curve_db, i_peak, drop_db, direction):
    target = curve_db[i_peak] - drop_db
    i = i_peak
    n = len(curve_db)
    while 0 <= i + direction < n and curve_db[i] > target:
        i += direction
    return i


def find_regions(freqs, jp_n_db, drop_db=6.0, min_peak_prominence_db=1.0):
    freqs = np.asarray(freqs)
    jp_n_db = np.asarray(jp_n_db)
    n = len(freqs)

    peak_idx, _ = find_peaks(jp_n_db, prominence=min_peak_prominence_db)
    if peak_idx.size == 0:
        i_rep = int(np.argmax(jp_n_db))
        return [Region("peak", float(freqs[0]), float(freqs[-1]),
                        float(freqs[i_rep]), i_rep, 0, n - 1)]

    windows = []
    for ip in peak_idx:
        lo = _drop_boundary(jp_n_db, ip, drop_db, -1)
        hi = _drop_boundary(jp_n_db, ip, drop_db, +1)
        windows.append([lo, ip, hi])
    windows.sort(key=lambda w: w[1])

    merged = [windows[0]]
    for lo, ip, hi in windows[1:]:
        if lo <= merged[-1][2]:
            merged[-1][2] = max(merged[-1][2], hi)
            if jp_n_db[ip] > jp_n_db[merged[-1][1]]:
                merged[-1][1] = ip
        else:
            merged.append([lo, ip, hi])

    regions = []
    cursor = 0
    for lo, ip, hi in merged:
        if lo > cursor:
            valley_idx = cursor + int(np.argmin(jp_n_db[cursor:lo + 1]))
            regions.append(Region("valley", float(freqs[cursor]), float(freqs[lo]),
                                   float(freqs[valley_idx]), valley_idx, cursor, lo))
        regions.append(Region("peak", float(freqs[lo]), float(freqs[hi]),
                               float(freqs[ip]), ip, lo, hi))
        cursor = hi
    if cursor < n - 1:
        valley_idx = cursor + int(np.argmin(jp_n_db[cursor:n]))
        regions.append(Region("valley", float(freqs[cursor]), float(freqs[n - 1]),
                               float(freqs[valley_idx]), valley_idx, cursor, n - 1))
    return regions


def region_for_index(regions, idx):
    for r in regions:
        if r.idx_lo <= idx <= r.idx_hi:
            return r
    return regions[-1] if idx >= regions[-1].idx_hi else regions[0]