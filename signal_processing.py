import sys
from typing import overload
import numpy as np
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import spectrogram
from scipy.fft import fft, ifft
from scipy.io import wavfile
import matplotlib.pyplot as plt

def topk_local_maxima_per_col(S, k, index_offset=0):
    """
    For 2D array S (F x T), return per-column top-k peak indices/values.

    Returns:
      top_idx:  (k x T) int indices into freq axis (filled with -1 where <k peaks)
      top_vals: (k x T) values (filled with NaN where <k peaks)
    """
    S = np.asarray(S)
    F, T = S.shape
    if F < 3 or k <= 0:
        return (np.full((k, T), -1, dtype=int),
                np.full((k, T), np.nan, dtype=S.dtype))

    # Peak occurs when y[i - 1] < y[i] and y[i + 1] < y[i]
    peak_mask = (S[1:-1, :] > S[:-2, :]) & (S[1:-1, :] > S[2:, :])

    top_idx  = np.full((k, T), -1, dtype=int)
    top_vals = np.full((k, T), np.nan, dtype=S.dtype)

    for j in range(T):
        idx = np.nonzero(peak_mask[:, j])[0] + 1
        if idx.size == 0:
            continue
        vals = S[idx, j]

        if idx.size > k:
            sel = np.argpartition(vals, -k)[-k:]
            idx, vals = idx[sel], vals[sel]

        order = np.argsort(-vals)
        kk = min(k, idx.size)
        top_idx[:kk, j]  = idx[order][:kk] + index_offset
        top_vals[:kk, j] = vals[order][:kk]

    return top_idx, top_vals


def fundamental_per_col(top_idx, W=5):
    """
    Pick one index per time slice closest to a rolling average (fixed window W).
    Seed rolling_avg from the average of the per-column peak maxima (row 0).
    
    top_idx : (K, T) int (use -1 where no peak)
    W       : int, window length for the rolling average
    """
    top_idx = np.asarray(top_idx)
    if top_idx.ndim != 2:
        raise ValueError("top_idx must be 2D (K, T).")
    K, T = top_idx.shape

    fund_idx = np.full(T, -1, dtype=int)

    # ---- Seed from the average of peak maxima (row 0) ----
    rolling = None
    if K > 0:
        seed = top_idx[0, :]
        seed = seed[seed >= 0]
        if seed.size > 0:
            rolling = float(seed.mean())
        else:
            # fallback: average all valid entries if row0 has none
            all_valid = top_idx[top_idx >= 0]
            if all_valid.size > 0:
                rolling = float(all_valid.mean())

    # ---- Fixed-window rolling mean via ring buffer ----
    if W <= 0:
        W = 1
    buf = np.zeros(W, dtype=float)
    buf_count = 0   # how many items currently in buffer (<= W)
    buf_pos = 0     # next position to overwrite
    run_sum = 0.0

    def update_mean(x):
        nonlocal buf, buf_count, buf_pos, run_sum
        if buf_count < W:
            buf[buf_count] = x
            run_sum += x
            buf_count += 1
        else:
            run_sum += x - buf[buf_pos]
            buf[buf_pos] = x
            buf_pos = (buf_pos + 1) % W
        return run_sum / buf_count

    # ---- Sweep columns ----
    for j in range(T):
        cand = top_idx[:, j]
        cand = cand[cand >= 0]
        if cand.size == 0:
            continue

        if rolling is None:
            rolling = float(cand.mean())

        chosen = int(cand[np.argmin(np.abs(cand - rolling))])
        fund_idx[j] = chosen
        rolling = update_mean(chosen)

    return fund_idx


import numpy as np

def transform_spectrum_1d(amplitudes, ref_index, num_orders, bins_per_interval):
    """
    Map a single spectrum (length F) onto a new 1-D axis measured in
    multiples of the reference bin `ref_index` (i.e., 0x .. num_orders x)
    with `bins_per_interval` subdivision between consecutive integers.

    Output length is: L = num_orders * bins_per_interval + 1
    (e.g., bins_per_interval=4 -> 0.00x, 0.25x, 0.50x, 0.75x, 1.00x, 1.25x, ...)

    Linear interpolation is used at fractional indices.
    """
    F = amplitudes.shape[0]
    if ref_index is None or ref_index < 1:
        # Can't define multiples if f0 is missing or at DC – return NaNs
        return np.full(num_orders * bins_per_interval + 1, np.nan, dtype=float)

    L = num_orders * bins_per_interval + 1
    # multiples grid u = 0, 1/B, 2/B, ..., num_orders
    u = np.arange(L, dtype=float) / float(bins_per_interval)

    # Fractional indices into the ORIGINAL spectrum (scale by ref_index!)
    p = ref_index * u  # float indices in [0, F-1] ideally

    # Linear interpolation
    i0 = np.floor(p).astype(int)
    i1 = i0 + 1
    alpha = p - i0

    # Valid where i0 in [0, F-1] and i1 in [0, F-1] (for strict interp).
    # Allow exact p==F-1 (alpha==0) by clamping i1 to F-1 and zeroing alpha.
    valid = (i0 >= 0) & (i0 < F)
    # For i1==F, set i1=F-1 and alpha=0 (equivalent to y[i0])
    hit_end = valid & (i1 >= F)
    i1 = np.where(hit_end, F - 1, i1)
    alpha = np.where(hit_end, 0.0, alpha)
    valid &= (i1 >= 0) & (i1 < F)

    out = np.full(L, np.nan, dtype=float)
    y0 = amplitudes[i0.clip(0, F - 1)]
    y1 = amplitudes[i1.clip(0, F - 1)]
    out[valid] = (1.0 - alpha[valid]) * y0[valid] + alpha[valid] * y1[valid]
    return out


def transform_spectrogram_to_orders(Sxx, fund_idx, num_orders=8, bins_per_interval=30):
    """
    Apply transform_spectrum_1d to each time slice (column) of Sxx using
    the per-column fundamental index fund_idx[j].

    Returns:
        S_mult : (L x T) array, where L = num_orders*bins_per_interval + 1
                 and the vertical axis is multiples of f0 (0..num_orders).
    """
    Sxx = np.asarray(Sxx)
    F, T = Sxx.shape
    L = num_orders * bins_per_interval + 1
    S_mult = np.full((L, T), np.nan, dtype=float)

    for j in range(T):
        r = int(fund_idx[j]) if j < len(fund_idx) else -1
        if r >= 1:
            S_mult[:, j] = transform_spectrum_1d(Sxx[:, j], r, num_orders, bins_per_interval)
        # else leave column as NaN (no reliable f0 for that frame)

    return S_mult


def time_avg_harmonic_profile(
    Sxx, fund_idx, num_orders=8, bins_per_interval=30, normalize_at_1x=False
):
    """
    Time-independent harmonic profile on a multiples-of-f0 axis.

    Args:
        Sxx : (F, T) ndarray
            Power spectrogram (e.g., from scipy.signal.spectrogram, default mode).
        fund_idx : (T,) int
            Fundamental bin index per frame (use -1 where unknown).
        num_orders : int
            Highest multiple to compute (0..M).
        bins_per_interval : int
            Subdivisions between integer multiples.
        normalize_at_1x : bool
            If True, also return a per-frame-1× normalized mean profile (still power).

    Returns:
        x_mult   : (L,) multiples grid in units of ×f0
        P_mean   : (L,) mean POWER over time
        P_rel_mean (optional) : (L,) mean POWER after per-frame normalization at 1×
                                (returned only if normalize_at_1x=True and valid data)
    """
    Sxx = np.asarray(Sxx, dtype=float)

    # Transform each time slice to multiples-of-f0 (uses your helper)
    S_mult = transform_spectrogram_to_orders(
        Sxx, fund_idx, num_orders=num_orders, bins_per_interval=bins_per_interval
    )  # (L, T)

    S_mult = np.nan_to_num(S_mult, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    # Average POWER over time (ignore NaN frames)
    P_mean = S_mult.mean(axis=1)
    x_mult = np.arange(P_mean.size, dtype=float) / float(bins_per_interval)

    if not normalize_at_1x:
        return x_mult, P_mean, None

    # Per-frame normalization at 1× (still power)
    i1 = bins_per_interval  # index corresponding to 1×
    valid = np.isfinite(S_mult[i1, :])
    if not np.any(valid):
        return x_mult, P_mean, None

    P_rel = S_mult[:, valid] / (S_mult[i1, valid][None, :] + 1e-12)
    P_rel_mean = np.nanmean(P_rel, axis=1)
    return x_mult, P_mean, P_rel_mean


def process_audio_wav(
    wav_filepath: str, 
    nperseg=4096, noverlap=256,                     # spectrogram
    num_peaks=3,                                    # peak finding
    num_orders=8, bins_per_interval = 30,           # order domain
    normalize_at_1x=False,                          # time averaging
    to_decibel=True                                 # spectrum return
):
    """
    Process .wav file into order-domain, time-independent, spectrum and cepstrum

    Returns:
        spectrum_power  : (num_orders * bins_per_interval + 1) power (amplitude**2), in dB if `to_decibel=True`
        cepstrum_mag    : (len(spectrum_power)//2) magnitude of cepstrum (not in dB), the first 
                                cell is replaced with zero magnitude
        spectrum_order  : (num_orders * bins_per_interval + 1) order of fundamental frequency
        cepstrum_qref   : (len(spectrum_power)//2n) quefrency
    """

    # ---- Reading the .wav file  ----
    sr, x = wavfile.read(wav_filepath)

    # Mono + float normalize to [-1, 1]
    if x.ndim == 2:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        x = x / np.iinfo(x.dtype).max
    x = x.astype(np.float32, copy=False)

    return process_audio_array(x, sampling_rate=sr, 
                                nperseg=nperseg, noverlap=noverlap,
                                num_peaks=num_peaks,
                                num_orders=num_orders, bins_per_interval=bins_per_interval,
                                normalize_at_1x=normalize_at_1x,
                                to_decibel=to_decibel)


def process_audio_array(
    x: np.ndarray,
    sampling_rate= 44100,
    nperseg=4096, noverlap=256,                     # spectrogram
    num_peaks=3,                                    # peak finding
    num_orders=8, bins_per_interval = 30,           # order domain
    normalize_at_1x=False,                          # time averaging
    to_decibel=True                                 # spectrum return
):
    """
    Process sample array into order-domain, time-independent, spectrum and cepstrum

    Arguments:
        x   : (len) float array normalized to -1 to 1

    Returns:
        spectrum_power  : (len) power (amplitude**2), in dB if `to_decibel=True`
        cepstrum_mag    : (len) magnitude of cepstrum (not in dB), the first 
                                cell is replaced with zero magnitude
        spectrum_order  : (len) order of fundamental frequency
        cepstrum_qref   : (len) quefrency
    """


    # ---- STFT/Spectrogram ----
    f, t, Sxx = spectrogram(x, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap)

    # ---- Finding top K peaks ----
    top_idx, top_vals = topk_local_maxima_per_col(Sxx, num_peaks)

    # ---- Fundamental from rolling-average of peaks ----
    fund_idx = fundamental_per_col(top_idx)

    # ---- Transform to order domain + time averaging ----
    x_mult, P_mean, P_rel_mean = time_avg_harmonic_profile(
        Sxx, fund_idx,
        num_orders=num_orders,
        bins_per_interval=bins_per_interval,
        normalize_at_1x=normalize_at_1x
    )

    # ---- convert to dB ----
    if to_decibel:
        eps = 1e-12
        P_mean_db = 10.0 * np.log10(P_mean + eps)
        P_rel_mean_db = None if P_rel_mean is None else 10.0 * np.log10(P_rel_mean + eps)

    # ---- Cepstrum ----
    len_cepstrum = len(P_mean) // 2
    cep_mag = np.abs(ifft(np.log(np.abs(P_mean) + eps))[0 : len_cepstrum])
    quef = np.arange(0, len_cepstrum) / (len(P_mean) / bins_per_interval)
    
    # discarding the first magnitude
    cep_mag[0] = 0.0

    # ---- returns ----
    if to_decibel:
        spectrum_power = P_mean_db
    else:
        spectrum_power = P_mean
    
    return spectrum_power, cep_mag, x_mult, quef


def plot_audio_file(
    wav_filepath: str, 
    plot_name=None,
    show=True,
    save=False,
    nperseg=4096, noverlap=256,                     # spectrogram
    num_peaks=3,                                    # peak finding
    num_orders=8, bins_per_interval = 30,           # order domain
    normalize_at_1x=False,                          # time averaging
    to_decibel=True                                 # spectrum return
):
    spec_power_db, ceps_mag, spec_order, ceps_qref = process_audio_wav(wav_filepath)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))  # 2 rows, 1 column
    if plot_name != None:
        plt.title(plot_name)

    # First subplot: harmonic profile
    ax1.plot(spec_order, spec_power_db)
    ax1.set_xlabel("Multiple of fundamental (× f0)")
    ax1.set_ylabel("Power [dB]")
    ax1.set_title("Time-independent harmonic profile (dB)")

    # Second subplot: cepstrum
    ax2.plot(ceps_qref, ceps_mag)
    ax2.set_xlabel("Quefrency")
    ax2.set_ylabel("Magnitude")
    ax2.set_title("Cepstrum")

    # Adjust layout
    plt.tight_layout()

    # Save
    if save & (plot_name != None):
        print("saving plot")
        plt.savefig(plot_name + ".png")  

    plt.show()


