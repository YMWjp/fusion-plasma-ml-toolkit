from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy import signal
import matplotlib.pyplot as plt


def label_by_derivative(isat: np.ndarray, *, sigma: float, threshold_percentile: float) -> int | None:
    smoothed = gaussian_filter1d(isat, sigma=sigma)
    deriv = np.gradient(smoothed)
    neg = -deriv
    thr = np.percentile(neg, threshold_percentile)
    cand = np.where(neg > thr)[0]
    return int(cand[0]) if cand.size else None


def label_by_threshold(isat: np.ndarray, *, threshold_percentile: float) -> int | None:
    mx = float(np.max(isat)) if isat.size else 0.0
    thr = mx * (threshold_percentile / 100.0)
    idx = np.where(isat < thr)[0]
    return int(idx[0]) if idx.size else None


def label_by_peak(isat: np.ndarray, *, min_prominence: float) -> int | None:
    peaks, _ = signal.find_peaks(-isat, prominence=min_prominence)
    return int(peaks[0]) if peaks.size else None


def apply_window_labels(length: int, start_index: int, *,
                        pre_range: int, transition_range: int, post_range: int,
                        pre_label: int, transition_label: int, post_label: int) -> np.ndarray:
    labels = np.zeros(length, dtype=int)
    max_idx = length - 1
    pre_s = max(0, start_index - pre_range)
    pre_e = max(0, start_index - transition_range)
    tr_s = max(0, start_index - transition_range)
    tr_e = min(max_idx + 1, start_index + transition_range)
    po_s = min(max_idx + 1, start_index + transition_range)
    po_e = min(max_idx + 1, start_index + post_range)
    labels[pre_s:pre_e] = pre_label
    labels[tr_s:tr_e] = transition_label
    labels[po_s:po_e] = post_label
    return labels


def choose_index_by_click(*, shot_no: int, time_wp: np.ndarray, wp: np.ndarray,
                          time_isat: np.ndarray, isat: np.ndarray) -> int | None:
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    ax1.plot(time_wp, wp, label='Wp', color='orange')
    ax1.set_ylabel('Wp')
    ax1.legend()
    ax2.plot(time_isat, isat, label='Isat_7L')
    ax2.set_ylabel('Isat_7L')
    ax2.legend()
    fig.suptitle(f'Shot Number: {shot_no} (Manual Mode - Click to select detachment point)')

    clicked: dict[str, int | None] = { 'idx': None }

    def onclick(event):
        if event.inaxes is ax2 and event.xdata is not None:
            click_time = float(event.xdata)
            idx = int((np.abs(time_isat - click_time)).argmin())
            clicked['idx'] = idx
            plt.close()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    return int(clicked['idx']) if clicked['idx'] is not None else None


