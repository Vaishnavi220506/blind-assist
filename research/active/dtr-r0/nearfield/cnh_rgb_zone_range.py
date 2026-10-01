"""Label-blind coarse-cell radial geometry distribution summaries.

Every eligible pixel receives equal weight. Fixed 5 cm distance bins and their
Gaussian-smoothed counts are a uniform-pixel geometry proxy, not a physical ToF
histogram, photon count, reflectance model or recorded sensor measurement. No
target, instance, semantic or evaluator label is accepted.
"""

import json

import numpy as np


BIN_WIDTH_M = 0.05
N_BINS = 256
MAX_RANGE_M = BIN_WIDTH_M * N_BINS
MIN_SUPPORT_FRACTION = 0.05
MAX_MODES = 4
NMS_DISTANCE_BINS = 3


def _plateau_peaks(smoothed):
    """Strict local maxima; equal-value plateau uses its leftmost index.

    The histogram boundaries compare against negative infinity. A peak must
    have positive strength; a zero plateau never yields a mode.
    """
    peaks = []
    start = 0
    while start < len(smoothed):
        end = start
        while end + 1 < len(smoothed) and smoothed[end + 1] == smoothed[start]:
            end += 1
        left = smoothed[start - 1] if start else -np.inf
        right = smoothed[end + 1] if end + 1 < len(smoothed) else -np.inf
        if smoothed[start] > 0 and smoothed[start] > left and smoothed[start] > right:
            peaks.append(start)
        start = end + 1
    return peaks


def summarize_distribution(values, radial_prior=None):
    """Summarize positive, finite radial ranges in [0, 12.8) meters.

    Smoothing is fixed Gaussian sigma=1 bin, zero extension, truncate=4 sigma.
    Peaks sort by smoothed count descending, ties by smaller bin. Raw support
    is the peak bin and its immediate neighbors, with at least 5% of eligible
    pixels required before nonmaximum suppression (centers <=3 bins apart).
    Each retained range is the median of its raw support, at most four modes.
    An optional positive finite RGB radial prior selects the closest retained
    mode; ties preserve strength order. There is no adaptive thresholding.

    Nonfinite/nonpositive input and positive finite out-of-range input are
    counted separately. All undefined numeric fields are JSON null.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("values must be a one-dimensional radial distance array")
    positive_finite = np.isfinite(values) & (values > 0)
    in_range = positive_finite & (values < MAX_RANGE_M)
    eligible = values[in_range]
    result = dict(status="NO_VALID_RANGE", n_total=int(values.size),
                  n_valid=int(eligible.size), n_dropped=int(values.size - eligible.size),
                  n_invalid_nonpositive=int((~positive_finite).sum()),
                  n_out_of_range=int((positive_finite & ~in_range).sum()),
                  median_radial_m=None, q10_radial_m=None, modes=[],
                  strongest_radial_m=None, guided_radial_m=None,
                  guided_status="MISSING_PRIOR", radial_prior_m=None,
                  bin_width_m=BIN_WIDTH_M, n_bins=N_BINS,
                  max_range_exclusive_m=MAX_RANGE_M,
                  proxy="uniform-pixel radial geometry; not physical ToF counts")
    if radial_prior is not None:
        try:
            prior = float(radial_prior)
        except (TypeError, ValueError, OverflowError):
            prior = float("nan")
        if not np.isfinite(prior) or prior <= 0:
            result["guided_status"] = "INVALID_PRIOR"
        else:
            result.update(radial_prior_m=prior, guided_status="NO_MODES")
    if not eligible.size:
        return result
    result.update(status="OK", median_radial_m=float(np.median(eligible)),
                  q10_radial_m=float(np.quantile(eligible, .10, method="linear")))
    bin_indices = np.floor(eligible / BIN_WIDTH_M).astype(np.int64)
    counts = np.bincount(bin_indices, minlength=N_BINS)
    gaussian_offsets = np.arange(-4, 5, dtype=np.float64)
    kernel = np.exp(-.5 * gaussian_offsets ** 2)
    kernel /= kernel.sum()
    # Exactly the fixed sigma=1, truncate=4, zero-extension discrete Gaussian.
    smoothed = np.convolve(counts.astype(np.float64), kernel, mode="same")
    candidates = sorted(_plateau_peaks(smoothed), key=lambda b: (-smoothed[b], b))
    modes = result["modes"]
    for peak in candidates:
        support_mask = np.abs(bin_indices - peak) <= 1
        support_samples = int(support_mask.sum())
        fraction = support_samples / eligible.size
        if fraction < MIN_SUPPORT_FRACTION:
            continue
        if any(abs(peak - mode["peak_bin"]) <= NMS_DISTANCE_BINS for mode in modes):
            continue
        modes.append(dict(radial_m=float(np.median(eligible[support_mask])),
                          support_fraction=float(fraction), support_samples=support_samples,
                          peak_bin=int(peak), peak_bin_center_m=float((peak + .5) * BIN_WIDTH_M),
                          support_lower_radial_m=float(max(0, peak - 1) * BIN_WIDTH_M),
                          support_upper_radial_exclusive_m=float(min(N_BINS, peak + 2) * BIN_WIDTH_M),
                          smoothed_strength=float(smoothed[peak])))
        if len(modes) == MAX_MODES:
            break
    if modes:
        result["strongest_radial_m"] = modes[0]["radial_m"]
        if result["radial_prior_m"] is not None:
            selected = min(modes, key=lambda mode: abs(mode["radial_m"] - result["radial_prior_m"]))
            result.update(guided_radial_m=selected["radial_m"], guided_status="OK")
    return result


def self_check():
    """Pure synthetic mathematical checks, with no model or dataset access."""
    plane = summarize_distribution(np.full(100, 2.025))
    assert plane["status"] == "OK" and len(plane["modes"]) == 1
    assert plane["median_radial_m"] == plane["q10_radial_m"] == 2.025
    assert plane["strongest_radial_m"] == 2.025
    assert plane["guided_status"] == "MISSING_PRIOR" and plane["guided_radial_m"] is None
    two = np.concatenate((np.full(80, 5.025), np.full(20, 2.025)))
    mixture = summarize_distribution(two, radial_prior=2.0)
    assert len(mixture["modes"]) == 2
    assert mixture["strongest_radial_m"] == 5.025
    assert mixture["guided_radial_m"] == 2.025
    assert mixture["modes"][0]["support_fraction"] == .8
    assert mixture["modes"][1]["support_fraction"] == .2
    clipped = summarize_distribution([np.nan, np.inf, -np.inf, -1., 0., 12.8, 20., .025])
    assert clipped["n_total"] == 8 and clipped["n_valid"] == 1 and clipped["n_dropped"] == 7
    assert clipped["n_invalid_nonpositive"] == 5 and clipped["n_out_of_range"] == 2
    assert clipped["strongest_radial_m"] == .025
    upper = summarize_distribution([12.775])
    assert upper["modes"][0]["peak_bin"] == 255
    empty = summarize_distribution([], radial_prior=2.)
    assert empty["status"] == "NO_VALID_RANGE" and empty["guided_status"] == "NO_MODES"
    invalid_prior = summarize_distribution([2.025], radial_prior=float("nan"))
    assert invalid_prior["guided_status"] == "INVALID_PRIOR" and invalid_prior["radial_prior_m"] is None
    plateau = _plateau_peaks(np.array([0., 2., 2., 0., 1., 0.]))
    assert plateau == [1, 4]
    weak = summarize_distribution(np.concatenate((np.full(97, 5.025), np.full(3, 2.025))), 2.)
    assert len(weak["modes"]) == 1 and weak["guided_radial_m"] == 5.025
    five = summarize_distribution(np.repeat([1.025, 2.025, 3.025, 4.025, 5.025], 20))
    assert len(five["modes"]) == 4
    assert [mode["peak_bin"] for mode in five["modes"]] == [20, 40, 60, 80]
    for summary in (plane, mixture, clipped, upper, empty, invalid_prior, weak, five):
        json.dumps(summary, allow_nan=False)
    return dict(status="PASS", checks=["single_plane", "80_20_two_surfaces", "near_weak_guided",
                "invalid_and_range_truncation", "upper_boundary_bin", "no_prior", "empty", "invalid_prior",
                "deterministic_plateau", "minimum_support", "four_mode_cap_strength_ties", "strict_json_finite"])


if __name__ == "__main__":
    print(json.dumps(self_check(), allow_nan=False))
