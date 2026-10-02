"""RGB-only features on caller-supplied public pixel links.

No depth, instance, semantic truth, query mask, or file path is accepted by the
feature functions. Coordinates are native pixel-centre indices (row, column),
with (0, 0) the top-left pixel centre. Endpoints must lie inside the image.
RGB values are observed sRGB / 255, not linear radiance or Lab.
"""
import argparse

import numpy as np

RGB_COLUMNS = tuple(
    f'{stat}_{channel}'
    for stat in ('endpoint_absdiff', 'endpoint_mean', 'path_std',
                 'path_total_variation', 'path_max_step')
    for channel in ('r', 'g', 'b')
) + ('endpoint_l2', 'path_max_step_l2', 'path_linear_deviation_l2_mean')
DEEP_COLUMNS = tuple(f'segformer_stage{k}_cosine_distance' for k in range(1, 5))
PATH_SAMPLES = 9
STAGE_STRIDES = (4, 8, 16, 32)


def _inputs(rgb, nodes_yx, pairs):
    image = np.asarray(rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('RGB must be uint8 HxWx3; PIL input must already be RGB')
    h, w = image.shape[:2]
    if min(h, w) < 1:
        raise ValueError('Empty image')
    nodes = np.asarray(nodes_yx, dtype=np.float64)
    links = np.asarray(pairs)
    if nodes.ndim != 2 or nodes.shape[1] != 2 or not np.isfinite(nodes).all():
        raise ValueError('nodes_yx must be finite Nx2 pixel-centre coordinates')
    if np.any(nodes < 0) or np.any(nodes[:, 0] > h-1) or np.any(nodes[:, 1] > w-1):
        raise ValueError('Nodes outside native pixel-centre bounds')
    if links.ndim != 2 or links.shape[1] != 2 or not np.issubdtype(links.dtype, np.integer):
        raise ValueError('pairs must be integer Mx2 node indices')
    if np.any(links < 0) or np.any(links >= len(nodes)):
        raise ValueError('Invalid node index')
    return image, nodes, links.astype(np.int64, copy=False)


def _bilinear(image, yx):
    y, x = yx[..., 0], yx[..., 1]
    y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
    y1, x1 = np.minimum(y0+1, image.shape[0]-1), np.minimum(x0+1, image.shape[1]-1)
    wy, wx = (y-y0)[..., None], (x-x0)[..., None]
    return ((1-wy)*((1-wx)*image[y0, x0]+wx*image[y0, x1])
            + wy*((1-wx)*image[y1, x0]+wx*image[y1, x1]))


def rgb_features(rgb, nodes_yx, pairs):
    """Return Mx18 symmetric features from nine evenly spaced RGB samples."""
    image, nodes, links = _inputs(rgb, nodes_yx, pairs)
    if not len(links):
        return np.empty((0, len(RGB_COLUMNS)), np.float32)
    t = np.linspace(0., 1., PATH_SAMPLES)[None, :, None]
    ends = nodes[links]
    path = (1-t)*ends[:, :1]+t*ends[:, 1:]
    samples = _bilinear(image.astype(np.float64)/255., path)
    first, last = samples[:, 0], samples[:, -1]
    step = np.abs(np.diff(samples, axis=1))
    linear = (1-t)*first[:, None]+t*last[:, None]
    features = np.column_stack((np.abs(first-last), (first+last)/2,
        samples.std(axis=1), step.sum(axis=1), step.max(axis=1),
        np.linalg.norm(first-last, axis=1),
        np.linalg.norm(step, axis=2).max(axis=1),
        np.linalg.norm(samples-linear, axis=2).mean(axis=1)))
    assert features.shape == (len(links), len(RGB_COLUMNS)) and np.isfinite(features).all()
    return features.astype(np.float32)


def load():
    """Explicit model acquisition hook; reuse the frozen local-only loader."""
    from cnh_rgb_dense_semantics_infer import load as frozen_load
    return frozen_load()


def _sample_stages(torch, states, nodes, native_shape):
    """Respect SegFormer's odd-kernel symmetric-padding receptive centres.

    Resize maps native centre x to (x+.5)*512/W-.5. Four patch embeddings
    have cumulative strides 4/8/16/32 and zero centre offset. This is different
    from treating hidden tensors as ordinary resized images of equal extent.
    Border padding extends the outermost observed feature when resize-centres
    fall just outside the feature-centre span. No extrapolated truth is used.
    """
    if len(states) != 4:
        raise ValueError('Exactly four hidden stages required')
    h, w = native_shape
    resized = (nodes[:, ::-1]+.5)*np.array([512/w, 512/h])-.5
    sampled = []
    for state, stride in zip(states, STAGE_STRIDES):
        if state.ndim != 4 or state.shape[0] != 1 or tuple(state.shape[-2:]) != (512//stride,)*2:
            raise ValueError('Expected full-image BCHW SegFormer stage at fixed512 input')
        if not bool(torch.isfinite(state).all()):
            raise ValueError('Nonfinite hidden features')
        xy = resized/stride
        extent = np.array([state.shape[-1], state.shape[-2]])
        grid = torch.as_tensor(2*(xy+.5)/extent-1, device=state.device, dtype=state.dtype)
        values = torch.nn.functional.grid_sample(state, grid[None, :, None, :],
            mode='bilinear', padding_mode='border', align_corners=False)[0, :, :, 0].T
        sampled.append(values)
    return sampled


def deep_features(image, nodes_yx, pairs, loaded):
    """Return Mx4 frozen endpoint cosine distances; only RGB enters the model.

    loaded is the tuple returned by load(). No labels/argmax are computed for
    this feature. Zero-norm endpoints produce distance1 by the fixed normalized
    dot-product convention; values are clipped only for floating-point roundoff.
    """
    rgb, nodes, links = _inputs(image, nodes_yx, pairs)
    if not len(links):
        return np.empty((0, len(DEEP_COLUMNS)), np.float32)
    torch, processor, model, _ = loaded
    if model.training:
        raise ValueError('Frozen eval-mode model required')
    tensor = processor(images=rgb, return_tensors='pt')['pixel_values']
    if tuple(tensor.shape) != (1, 3, 512, 512):
        raise ValueError('Frozen full512 processor required')
    tensor = tensor.to(device=next(model.parameters()).device, dtype=torch.float32)
    with torch.inference_mode():
        output = model(pixel_values=tensor, output_hidden_states=True, return_dict=True)
        sampled = _sample_stages(torch, output.hidden_states, nodes, rgb.shape[:2])
        link_tensor = torch.as_tensor(links, device=tensor.device)
        columns = []
        for values in sampled:
            normalized = torch.nn.functional.normalize(values, p=2, dim=1, eps=1e-12)
            cosine = (normalized[link_tensor[:, 0]]*normalized[link_tensor[:, 1]]).sum(dim=1)
            columns.append((1-cosine.clamp(-1, 1)).cpu().numpy())
    result = np.column_stack(columns).astype(np.float32)
    assert result.shape == (len(links), 4) and np.isfinite(result).all()
    return result


def selftest():
    """Synthetic CPU math/interface checks; no weights, files or data loaded."""
    import torch
    nodes = np.array([[0., 0.], [16., 16.], [7.5, 8.5]])
    pairs = np.array([[0, 1], [1, 0], [2, 2]])
    constant = np.full((17, 17, 3), 127, np.uint8)
    features = rgb_features(constant, nodes, pairs)
    expected = np.zeros_like(features); expected[:, 3:6] = 127/255
    np.testing.assert_allclose(features, expected, atol=1e-7)
    colors = constant.copy(); colors[:, 8:] = [255, 0, 0]
    features = rgb_features(colors, nodes, pairs)
    np.testing.assert_allclose(features[0], features[1], atol=1e-7)
    assert features[0, 0] > 0 and features[0, -1] > 0
    assert rgb_features(colors, nodes, np.empty((0, 2), int)).shape == (0, 18)
    # Feature ramp encodes its input512 receptive centre; verify half-pixels,
    # cumulative stride and axis order without invoking any learned network.
    states = []
    for stride in STAGE_STRIDES:
        yy, xx = torch.meshgrid(torch.arange(512//stride), torch.arange(512//stride), indexing='ij')
        states.append(torch.stack((xx*stride, yy*stride)).float()[None])
    p = np.array([[100., 200.], [200.5, 300.5]])
    sampled = _sample_stages(torch, states, p, (768, 1024))
    expected_xy = (p[:, ::-1]+.5)*np.array([.5, 2/3])-.5
    for values in sampled:
        np.testing.assert_allclose(values.numpy(), expected_xy, atol=4e-5)
    class Processor:
        def __call__(self, images, return_tensors):
            return {'pixel_values': torch.zeros((1, 3, 512, 512))}
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.dummy = torch.nn.Parameter(torch.zeros(()))
        def forward(self, **kwargs):
            from types import SimpleNamespace
            return SimpleNamespace(hidden_states=tuple(torch.ones((1, 3, 512//s, 512//s)) for s in STAGE_STRIDES))
    deep = deep_features(colors, nodes, pairs, (torch, Processor(), Model().eval(), {}))
    np.testing.assert_allclose(deep, 0, atol=2e-7)
    bad = False
    try:
        rgb_features(colors, np.array([[-1., 0.]]), np.array([[0, 0]]))
    except ValueError:
        bad = True
    assert bad
    print('PASS_RGB18_DEEP4_SYNTHETIC_ONLY_NO_MODEL_OR_DATA')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['selftest'])
    selftest() if parser.parse_args().action == 'selftest' else None
