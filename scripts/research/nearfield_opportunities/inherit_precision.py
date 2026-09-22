"""Private-namespace quantization adapter; frozen modules are never mutated."""
from __future__ import annotations

import math
from types import FunctionType, ModuleType, SimpleNamespace

import initial_relative_inference as original


def _clone_functions(module, replacements):
    private = ModuleType(module.__name__ + '_inherit_precision_private')
    private.__dict__.update(vars(module))
    private.__dict__.update(replacements)
    for name, value in vars(module).items():
        if isinstance(value, FunctionType) and value.__globals__ is module.__dict__:
            private.__dict__[name] = FunctionType(value.__code__, private.__dict__,
                name=value.__name__, argdefs=value.__defaults__, closure=value.__closure__)
            private.__dict__[name].__kwdefaults__ = value.__kwdefaults__
    return private


def model_for(step):
    if step not in (.1, .001):
        raise ValueError('Only the frozen coarse and fine quantizers are allowed')
    base = SimpleNamespace(**vars(original.shared.base))
    base.RANGE_STEP = step
    shared = _clone_functions(original.shared, {'base': base})
    model = _clone_functions(original, {'shared': shared})
    model.quantization_step_m = step
    return model


def public_views(views, step):
    if step not in (.1, .001):
        raise ValueError('Unsupported quantizer')
    public = []
    for view in views:
        ranges = view['biased_ranges']
        if len(ranges) != 8 or any(not math.isfinite(v) or v <= 0 for v in ranges):
            raise ValueError('Eight positive finite analog simulated ranges required')
        public.append({'camera': list(view['camera']),
                       'bins': [math.floor(v / step + .5) for v in ranges]})
    original._inputs(public)
    return public
