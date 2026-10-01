"""Equal-angle Hypersim adapter for an uncalibrated CNH response proxy.

This keeps the RGB clearance probe's 45-degree / 8-by-8 equal-angle cells.
It differs from cnh_route_sensor.angular_rays' equal-tangent cells and is not
a measured L8CH projection.  Inputs are native first-visible radial geometry
and public calibration only: no labels, target edge or target truth is read.
"""
from dataclasses import asdict, replace
import json

import numpy as np

from cnh_route_sensor import H3, RAW_BIN_M, RAW_BINS, SensorParameters, derive_readout, synthesize_response


NATIVE_SHAPE = (768, 1024)
SAMPLES_PER_AXIS = 16
FOV_DEG = 45.
NIR_REFLECTANCE = .5
INCIDENCE_COS = 1.
GRID_ROLE = "45deg equal-angle 8x8 proxy; not original equal-tangent grid or measured ToF"


def _angular_samples():
    n = SAMPLES_PER_AXIS
    half = np.deg2rad(FOV_DEG / 2)
    step = 2 * half / (8 * n)
    angles = -half + (np.arange(8 * n) + .5) * step
    vertical, horizontal = np.meshgrid(angles, angles, indexing="ij")
    a, b = np.tan(horizontal), np.tan(vertical)
    norm = np.sqrt(1 + a*a + b*b)
    rays = np.stack((a, -b, -np.ones_like(a)), axis=-1) / norm[..., None]
    # dOmega = sec(a)^2 sec(b)^2 / (1+tan(a)^2+tan(b)^2)^1.5 da db.
    solid_angle = (1 + a*a) * (1 + b*b) / norm**3 * step**2

    def group(array):
        tail = array.shape[2:]
        return array.reshape(8, n, 8, n, *tail).transpose(
            0, 2, 1, 3, *range(4, 4 + len(tail))).reshape(8, 8, n*n, *tail)

    return group(rays), group(solid_angle)


def quadrature(radial, camera_matrix):
    """16x16 angular midpoints per cell, nearest native radial samples.

    Graphic camera axes are +x right, +y up, -z forward.  The inverse full
    M_cam_from_uv projects rays to normalized UV, including tilt-shift terms.
    Native array indices refer to pixel centers.  Out-of-image or invalid
    depth samples become NaN; their solid-angle weight remains in the zone's
    denominator.  No edge-dependent sampling or clamping is performed.
    """
    radial = np.asarray(radial)
    matrix = np.asarray(camera_matrix, dtype=np.float64)
    if radial.shape != NATIVE_SHAPE:
        raise ValueError("Need native Hypersim radial shape (768,1024)")
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Need finite complete M_cam_from_uv matrix")
    rays, weights = _angular_samples()
    uv_homogeneous = rays @ np.linalg.inv(matrix).T
    valid_projection = np.isfinite(uv_homogeneous).all(-1) & (uv_homogeneous[..., 2] > 1e-12)
    uv = np.zeros_like(uv_homogeneous[..., :2])
    np.divide(uv_homogeneous[..., :2], uv_homogeneous[..., 2:],
              out=uv, where=valid_projection[..., None])
    height, width = NATIVE_SHAPE
    pixel_x = (uv[..., 0] + 1) * width / 2 - .5
    pixel_y = (1 - uv[..., 1]) * height / 2 - .5
    # Positive half-integer tie goes to the next index; no banker rounding.
    xx = np.floor(pixel_x + .5).astype(np.int64)
    yy = np.floor(pixel_y + .5).astype(np.int64)
    inside = valid_projection & (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
    sampled = np.full(weights.shape, np.nan)
    sampled[inside] = radial[yy[inside], xx[inside]]
    known = inside & np.isfinite(sampled) & (sampled > 0)
    sampled[~known] = np.nan
    normalized_weights = weights / weights.sum(-1, keepdims=True)
    return dict(radial_m=sampled, solid_angle_weights=weights,
                unit_rays_graphics=rays,
                sampled_yx=np.stack((yy, xx), -1),
                projected_yx=np.stack((pixel_y, pixel_x), -1),
                in_image=inside, valid_radial=known,
                in_image_fraction=inside.mean(-1),
                valid_radial_fraction=known.mean(-1),
                weighted_in_image_fraction=(normalized_weights * inside).sum(-1),
                weighted_valid_radial_fraction=(normalized_weights * known).sum(-1),
                grid_role=GRID_ROLE)


def _serialize_readout(response, parameters):
    readout = derive_readout(response, H3)
    valid = np.asarray(readout["valid"]).reshape(64)
    distance = np.asarray(readout["distance_m"]).reshape(64)
    histogram = np.asarray(readout["histogram"]).reshape(64, H3.bins)
    indices = histogram.argmax(-1)
    return dict(distance_m=[float(d) if v else None for d, v in zip(distance, valid)],
                valid=valid.tolist(),
                bin_index=[int(i) if v else None for i, v in zip(indices, valid)],
                status=np.asarray(readout["status"]).reshape(64).astype(int).tolist(),
                histogram=histogram.tolist(),
                ambient=np.asarray(readout["ambient"]).reshape(64).tolist(),
                bin_centers_m=np.asarray(readout["bin_centers_m"]).tolist(),
                params=asdict(parameters), config=asdict(H3),
                detector=readout["detector"],
                scalar_role="strongest SNR-passing aggregated-bin center; not ST scalar firmware")


def sensor_readouts(radial, camera_matrix, seed):
    """Paired clean/stress response readouts; caller supplies a fixed seed.

    Clean removes stochastic noise, tail, neighbour leak and residual xtalk,
    while retaining sigma=2 raw-bin pulse width.  Both arms assume uniform
    rho=.5 and cos=1, inverse-square return, native3.75348cm bins and H3's
    8-bin aggregation.  No seed or parameter is chosen using label/edge truth.
    """
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("Need a nonnegative caller-declared integer seed")
    samples = quadrature(radial, camera_matrix)
    base = SensorParameters()
    clean = replace(base, noise_scale=0., tail_mass=0., crosstalk_fraction=0., neighbour_leak=0.)
    result = dict(seed=int(seed), grid_role=GRID_ROLE,
                  nir_reflectance=NIR_REFLECTANCE, incidence_cos=INCIDENCE_COS,
                  samples_per_axis=SAMPLES_PER_AXIS,
                  radial_window_m=[0., RAW_BIN_M * RAW_BINS],
                  aggregated_bin_width_m=RAW_BIN_M * H3.sub_sample,
                  hardware_role="uncalibrated first-visible diffuse geometry proxy; no measured ToF gate")
    for name, parameters in (("clean", clean), ("stress", base)):
        response = synthesize_response(samples["radial_m"], NIR_REFLECTANCE,
            INCIDENCE_COS, samples["solid_angle_weights"], params=parameters, seed=int(seed))
        result[name] = _serialize_readout(response, parameters)
    result["coverage"] = {key: samples[key].reshape(64).tolist() for key in
        ("in_image_fraction", "valid_radial_fraction", "weighted_in_image_fraction", "weighted_valid_radial_fraction")}
    result["coverage"].update(total_samples=int(samples["radial_m"].size),
        in_image_samples=int(samples["in_image"].sum()),
        valid_radial_samples=int(samples["valid_radial"].sum()))
    return result


def self_check():
    """Synthetic math checks only; no scientific frame or prediction is read."""
    matrix = np.diag([.6, .45, -1.])
    yy, xx = np.indices(NATIVE_SHAPE)
    uv = np.stack(((xx + .5) * 2 / NATIVE_SHAPE[1] - 1,
                   1 - (yy + .5) * 2 / NATIVE_SHAPE[0], np.ones(NATIVE_SHAPE)), -1)
    pixel_rays = uv @ matrix.T
    plane_radial = 2 * np.linalg.norm(pixel_rays, axis=-1) / -pixel_rays[..., 2]
    sample = quadrature(plane_radial, matrix)
    assert sample["radial_m"].shape == (8, 8, 256)
    assert sample["in_image"].all() and sample["valid_radial"].all()
    py, px = sample["sampled_yx"][..., 0], sample["sampled_yx"][..., 1]
    np.testing.assert_allclose(sample["radial_m"], plane_radial[py, px], atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(sample["unit_rays_graphics"], axis=-1), 1., atol=1e-12)
    # Midpoint angular quadrature and nearest pixels stay within half a pixel.
    np.testing.assert_array_less(abs(sample["projected_yx"] - sample["sampled_yx"]), .50000001)
    sampled_pixel_rays = pixel_rays[py, px]
    optical = sample["radial_m"] * -sampled_pixel_rays[..., 2] / np.linalg.norm(sampled_pixel_rays, axis=-1)
    np.testing.assert_allclose(optical, 2., atol=1e-12)
    assert np.all(sample["solid_angle_weights"] > 0)
    edge = np.tan(np.deg2rad(FOV_DEG / 2))
    expected_omega = 4 * np.arctan(edge**2 / np.sqrt(1 + 2 * edge**2))
    np.testing.assert_allclose(sample["solid_angle_weights"].sum(), expected_omega, rtol=1e-4)
    # Off-diagonal and tilt terms exercise the complete inverse projection.
    oblique = matrix.copy(); oblique[0, 1] = .01; oblique[2, 1] = .02
    other = quadrature(plane_radial, oblique)
    ryx = other["projected_yx"]
    restored_uv = np.stack(((ryx[..., 1] + .5) * 2 / 1024 - 1,
                           1 - (ryx[..., 0] + .5) * 2 / 768, np.ones(ryx.shape[:-1])), -1)
    restored = restored_uv @ oblique.T
    restored /= np.linalg.norm(restored, axis=-1, keepdims=True)
    np.testing.assert_allclose(restored, other["unit_rays_graphics"], atol=1e-12)
    narrow = quadrature(plane_radial, np.diag([.1, .1, -1.]))
    assert 0 < narrow["in_image"].sum() < narrow["in_image"].size
    assert np.isnan(narrow["radial_m"][~narrow["in_image"]]).all()
    np.testing.assert_array_equal(narrow["solid_angle_weights"], sample["solid_angle_weights"])
    result = sensor_readouts(plane_radial, matrix, 2026100201)
    assert len(result["clean"]["distance_m"]) == len(result["stress"]["distance_m"]) == 64
    assert all(result["clean"]["valid"])
    blank = sensor_readouts(np.full(NATIVE_SHAPE, np.nan), matrix, 2026100201)
    assert not any(blank["clean"]["valid"])
    assert all(value is None for value in blank["clean"]["distance_m"])
    assert blank["coverage"]["weighted_valid_radial_fraction"] == [0.] * 64
    json.dumps(result, allow_nan=False); json.dumps(blank, allow_nan=False)
    return dict(status="PASS", checks=["single_plane_native_projection", "native_radial_to_optical_Z",
        "unit_ray_and_solid_angle", "complete_oblique_inverse_projection", "nearest_pixel_centers",
        "out_of_image_nan_keeps_weights", "paired_64_zone_readouts", "unknown_clean_readout", "strict_json_finite"])


if __name__ == "__main__":
    print(json.dumps(self_check(), allow_nan=False, indent=2))
