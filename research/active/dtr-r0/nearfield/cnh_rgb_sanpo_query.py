"""SANPO native-depth adapter with unresolved optical-Z/radial hypotheses.

This inherits the local SANPO preflight's integer-index pinhole convention;
it does not establish that convention or the depth axis from official evidence.
No poses, floor estimates, split enumeration or source traversal are used.
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

import cnh_rgb_visible_query as V
import cnh_rgb_range_anchor as A


def camera_geometry(params):
    """Native public rays and whole-range queries under integer pixel centres.

    Returns the two-query geometry accepted by range_anchor. Matrix offsets
    compensate visible_query's normalized (u+.5,v+.5) coordinates so the rays
    exactly equal (u-cx)/fx and (v-cy)/fy at integer indices u,v.
    """
    required = ('fx', 'fy', 'cx', 'cy', 'image_width', 'image_height')
    p = {k: float(params[k]) for k in required}
    if not all(np.isfinite(x) for x in p.values()):
        raise ValueError('camera parameters must be finite')
    if p['fx'] <= 0 or p['fy'] <= 0:
        raise ValueError('positive pinhole focal lengths required')
    w, h = int(p['image_width']), int(p['image_height'])
    if w <= 0 or h <= 0 or w != p['image_width'] or h != p['image_height']:
        raise ValueError('positive integer native image dimensions required')
    M = np.array([[w/(2*p['fx']), 0., (w/2-.5-p['cx'])/p['fx']],
                  [0., h/(2*p['fy']), (p['cy']-h/2+.5)/p['fy']],
                  [0., 0., -1.]])
    geometry = A.whole_geometry(V.ray_geometry((h, w), M))
    geometry.update(native_shape=(h, w), source_intrinsics={**p, 'image_width': w, 'image_height': h},
                    source='SANPO public pinhole parameters; native pixels, no resize/subsample',
                    pixel_centre_convention='UNCONFIRMED: inherited local integer-index convention (u-cx)/fx,(v-cy)/fy',
                    depth_axis='UNRESOLVED: optical_Z and radial hypotheses both retained',
                    fov_contract='45 degrees horizontal by 45 degrees vertical; not old diagonal-45 preflight',
                    physical_query_frame='camera-relative convention only; camera/body/world alignment unconfirmed')
    return geometry


def _decode_payload(payload):
    """Decode decompressed bytes without altering missing or nonpositive values."""
    if len(payload) < 4 or len(payload) % 2:
        raise ValueError('SANPO depth requires two float16 dimension values and float16 pixels')
    values = np.frombuffer(payload, dtype='<f2')
    dims = values[:2].astype(np.float64)
    if not np.isfinite(dims).all() or np.any(dims <= 0) or np.any(dims != np.floor(dims)):
        raise ValueError('invalid SANPO float16 height/width header')
    h, w = map(int, dims)
    if values.size != 2+h*w:
        raise ValueError(f'SANPO depth payload length differs from header {h}x{w}')
    return values[2:].reshape(h, w).astype(np.float32)


def decode_depth(path):
    """Read one explicitly supplied gzip little-endian-float16 native depth.

    The first two float16 numbers encode height,width. Return float32 raw metre
    values, with no axis conversion, clipping, filling or validity selection.
    """
    return _decode_payload(gzip.decompress(Path(path).read_bytes()))


def truth_hypotheses(raw, geometry):
    """Return both visible-truth interpretations; neither is authoritative truth."""
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape != geometry['shape']:
        raise ValueError('raw depth dimensions must match native camera geometry')
    hypotheses = {'optical_Z': raw, 'radial': raw/geometry['radial_factor']}
    result = {}
    for name, optical_z in hypotheses.items():
        truth = A.visible_truth(optical_z, geometry)
        truth.update(depth_hypothesis=name, depth_axis_status='UNRESOLVED; paired interpretation, no chosen winner',
                     pixel_centre_convention=geometry['pixel_centre_convention'],
                     native_shape=geometry['native_shape'], native_support_pixels=V.MIN_SUPPORT,
                     ground_truth_scope='hypothesized camera-relative visible points only; source depth axis and body alignment unresolved')
        result[name] = truth
    return result


def fixtures():
    # These dimensions are the complete native synthetic image, never subsampled.
    params = dict(fx=60., fy=50., cx=31.2, cy=23.7, image_width=64, image_height=48)
    g = camera_geometry(params)
    y, x = np.indices((48, 64))
    np.testing.assert_allclose(g['fx'], (x-params['cx'])/params['fx'], atol=1e-15, rtol=0)
    np.testing.assert_allclose(g['fy'], (y-params['cy'])/params['fy'], atol=1e-15, rtol=0)
    assert g['shape'] == g['native_shape'] == (48, 64)
    assert g['possible_query_rays'].shape == (2, 48, 64)
    assert g['zone_ray_count'].sum() == g['fov_mask'].sum()
    assert g['queries'][0]['z_near'] == .6 and g['queries'][0]['z_far'] == 2.1
    assert g['native_covers_nominal_fov']
    narrow = camera_geometry(dict(params, fx=200., fy=200.))
    assert not narrow['native_covers_nominal_fov']
    # Raw constant1.4 is a plane under Z and a constant-radius shell under radial.
    raw = np.full((48, 64), 1.4, dtype=np.float32)
    result = truth_hypotheses(raw, g)
    assert set(result) == {'optical_Z', 'radial'}
    for name, z in [('optical_Z', raw), ('radial', raw/g['radial_factor'])]:
        expected = A.visible_truth(z, g)
        for field in ('category', 'contact_bin', 'return_count', 'contact_count', 'score'):
            np.testing.assert_array_equal(result[name][field], expected[field])
        assert result[name]['native_support_pixels'] == result[name]['minimum_pixel_support'] == 16
    np.testing.assert_allclose((raw/g['radial_factor'])*g['radial_factor'], raw, atol=1e-15, rtol=0)
    assert np.any(raw/g['radial_factor'] < raw)
    pixels = np.array([[1., 0.], [np.nan, np.inf]], dtype='<f2')
    payload = np.concatenate((np.asarray([2, 2], dtype='<f2'), pixels.ravel())).astype('<f2').tobytes()
    decoded = _decode_payload(gzip.decompress(gzip.compress(payload)))
    assert decoded.dtype == np.float32
    np.testing.assert_array_equal(decoded, pixels.astype(np.float32))
    return dict(integer_pixel_rays=True, halfpixel_compensation=True, both_depth_hypotheses=True,
                native_pixel_support16=True, footprint_preserved=True, gzip_float16_decode=True,
                missing_values_preserved=True, scientific_data_accessed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=['fixtures'])
    parser.parse_args()
    print(json.dumps(fixtures()))
