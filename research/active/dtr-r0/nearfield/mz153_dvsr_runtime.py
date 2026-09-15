"""Pinned official DVSR inference through narrow legacy dependency adapters.

Third-party source remains byte-identical in ignored runtime/upstream. Both
SPyNet directions, both hourglasses, and all four propagation passes are kept.
No source simulator, dense target or sequence-dependent normalization is used.
"""
from contextlib import contextmanager
import hashlib
import importlib
import json
import logging
from pathlib import Path
import sys
import threading
import time
import types

import numpy as np
import torch
from torch import nn
from torchvision.ops import deform_conv2d

TREE='bc9cd22153e1183310342b0a2a14e11f93b1be62'
CHECKPOINT_SHA256='bea5ce12c314472884214864d00bc77c89da6900e6fe36ebc329282a6a7dcf4e'
_LOCK=threading.RLock()

def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class _Registry:
    def register_module(self,*args,**kwargs):
        if 'module' in kwargs:return kwargs['module']
        return lambda cls:cls

class _ConvModule(nn.Module):
    """Only official SPyNet's Conv2d -> optional ReLU configuration."""
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,norm_cfg=None,act_cfg=None,**kwargs):
        super().__init__()
        if norm_cfg is not None or kwargs:raise ValueError('Unsupported legacy ConvModule configuration')
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=True)
        if act_cfg is not None and act_cfg!={'type':'ReLU'}:raise ValueError('Unsupported activation')
        self.activate=nn.ReLU(inplace=True) if act_cfg else nn.Identity()
    def forward(self,x):return self.activate(self.conv(x))

def _constant_init(module,val,bias=0):
    if getattr(module,'weight',None) is not None:nn.init.constant_(module.weight,val)
    if getattr(module,'bias',None) is not None:nn.init.constant_(module.bias,bias)

def _kaiming_init(module,a=0,mode='fan_out',nonlinearity='relu',bias=0,distribution='normal'):
    init=nn.init.kaiming_normal_ if distribution=='normal' else nn.init.kaiming_uniform_
    init(module.weight,a=a,mode=mode,nonlinearity=nonlinearity)
    if getattr(module,'bias',None) is not None:nn.init.constant_(module.bias,bias)

def modulated_deform_adapter(x,offset,mask,weight,bias=None,stride=1,padding=0,dilation=1,groups=1,deform_groups=1):
    """MMCV's interleaved (dy,dx) offsets map directly to torchvision.

Convolution groups and offset groups are independent. torchvision infers the
former from weight shape and latter from offset channels; assert both here.
"""
    kh,kw=weight.shape[-2:]
    if x.shape[1]//weight.shape[1]!=groups:raise ValueError('Convolution groups mismatch')
    if offset.shape[1]!=2*deform_groups*kh*kw or mask.shape[1]!=deform_groups*kh*kw:
        raise ValueError('Offset/mask groups mismatch')
    return deform_conv2d(x,offset,weight,bias,stride,padding,dilation,mask)

class _ModulatedDeformConv2d(nn.Conv2d):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1,deform_groups=1,bias=True):
        super().__init__(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias)
        self.deform_groups=deform_groups

def _unneeded_checkpoint(*args,**kwargs):
    raise RuntimeError('Separate legacy checkpoint loading is forbidden; use one strict full state_dict')

@contextmanager
def _legacy_imports():
    """Temporary import aliases, restored immediately after source loading."""
    modules={}
    for name in ('mmcv','mmcv.cnn','mmcv.runner','mmcv.ops','mmcv.utils','mmcv.utils.parrots_wrapper','mmseg','mmseg.utils'):
        modules[name]=types.ModuleType(name);modules[name].__path__=[]
    cnn=modules['mmcv.cnn'];cnn.ConvModule=_ConvModule;cnn.constant_init=_constant_init
    cnn.kaiming_init=_kaiming_init;cnn.CONV_LAYERS=_Registry()
    cnn.build_activation_layer=lambda cfg:nn.ReLU(inplace=cfg.get('inplace',True))
    modules['mmcv.runner'].load_checkpoint=_unneeded_checkpoint
    modules['mmcv.ops'].ModulatedDeformConv2d=_ModulatedDeformConv2d
    modules['mmcv.ops'].modulated_deform_conv2d=modulated_deform_adapter
    modules['mmcv.utils.parrots_wrapper']._BatchNorm=nn.modules.batchnorm._BatchNorm
    modules['mmseg.utils'].get_root_logger=lambda:logging.getLogger('mz153-dvsr')
    previous={name:sys.modules.get(name) for name in modules}
    try:
        sys.modules.update(modules)
        yield
    finally:
        for name,old in previous.items():
            if old is None:sys.modules.pop(name,None)
            else:sys.modules[name]=old

def _load_class(runtime):
    name='_mz153_official_dvsr_'+TREE[:12]
    with _LOCK:
        if name+'.dvsr' in sys.modules:return sys.modules[name+'.dvsr'].DVSR
        package=types.ModuleType(name);package.__path__=[str(runtime/'upstream/model')]
        sys.modules[name]=package
        registry=types.ModuleType(name+'.registry');registry.BACKBONES=_Registry()
        sys.modules[name+'.registry']=registry
        with _legacy_imports():return importlib.import_module(name+'.dvsr').DVSR

class _FrozenModel(nn.Module):
    def __init__(self,generator):super().__init__();self.generator=generator
    def forward(self,lqs,guides):return self.generator(lqs,guides)

def _full_bidirectional_flow(self,lqs):
    # Official mirror optimization leaves a sticky True and returns None while
    # hg_forward unconditionally interpolates that value. Disable the shortcut
    # only: unchanged compute_flow executes both original SPyNet directions.
    self.is_mirror_extended=False

def load_model(workdir,device='cuda'):
    """Return (model, info), loading all 592 original generator.* keys.

workdir is the runtime directory containing source-receipt.json and upstream/.
No network request, package installation or state-dict key rewriting occurs.
"""
    runtime=Path(workdir).resolve();receipt=json.loads((runtime/'source-receipt.json').read_text())
    if receipt['tree_sha']!=TREE:raise ValueError('Unexpected upstream revision')
    for entry in receipt['files']:
        path=runtime/'upstream'/entry['path']
        if _sha(path)!=entry['sha256']:raise ValueError('Upstream hash mismatch: '+entry['path'])
    checkpoint=runtime/'upstream/chkpts/dvsr_tartan.pth'
    if _sha(checkpoint)!=CHECKPOINT_SHA256:raise ValueError('Unexpected checkpoint')
    cls=_load_class(runtime)
    generator=cls(mid_channels=64,num_blocks=7,scale=16,is_low_res_input=True,spynet_pretrained=None)
    generator.check_if_mirror_extended=types.MethodType(_full_bidirectional_flow,generator)
    model=_FrozenModel(generator)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if len(state)!=592 or not all(key.startswith('generator.') for key in state):raise ValueError('Unexpected official state dictionary')
    incompat=model.load_state_dict(state,strict=True)
    model.eval().requires_grad_(False).to(device)
    model.runtime_metadata=dict(upstream_tree=TREE,checkpoint_sha256=CHECKPOINT_SHA256,state_dict_keys=len(state),
        strict_load=True,missing_keys=list(incompat.missing_keys),unexpected_keys=list(incompat.unexpected_keys),
        deform_backend='torchvision.ops.deform_conv2d',compute='OFFICIAL_TWO_HOURGLASS_FOUR_PASS_BIDIRECTIONAL',
        compatibility='LEGACY_IMPORT_APIS_AND_DISABLE_BROKEN_OPTIONAL_MIRROR_FLOW_REUSE',
        normalization='CALLER_SUPPLIES_RGB01_AND_NORMALIZED_DEPTH01_NO_INTERNAL_RESCALE_OR_CLAMP',
        parameters=sum(p.numel() for p in model.parameters()),source_receipt_sha256=_sha(runtime/'source-receipt.json'))
    return model,dict(model.runtime_metadata)

def infer_latest(model,rgb,depth):
    """Return latest float32 numpy[128,128] for the supplied causal prefix.

The caller owns which observed prefix is provided. All official bidirectional
passes operate inside that prefix; only its latest dense frame is returned.
Outputs are not clamped and do not certify metric geometry or free space.
"""
    start=time.perf_counter();device=next(model.parameters()).device
    rgb=torch.as_tensor(rgb,dtype=torch.float32,device=device).contiguous()
    depth=torch.as_tensor(depth,dtype=torch.float32,device=device).contiguous()
    if rgb.ndim!=4 or tuple(rgb.shape[1:])!=(3,128,128):raise ValueError('rgb must be T,3,128,128')
    if tuple(depth.shape)!=(rgb.shape[0],1,8,8) or not 2<=rgb.shape[0]<=6:raise ValueError('depth must be T,1,8,8 and T in2..6')
    for name,value in [('rgb',rgb),('depth',depth)]:
        if not torch.isfinite(value).all() or value.min()<0 or value.max()>1:raise ValueError(name+' must be finite in0..1')
    if device.type=='cuda':torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
    compute_start=time.perf_counter()
    with torch.inference_mode():
        dense,_=model(depth.unsqueeze(0),rgb.unsqueeze(0))
    if device.type=='cuda':torch.cuda.synchronize(device)
    compute_s=time.perf_counter()-compute_start
    output=dense[0,-1,0].detach().float().cpu().numpy().copy()
    if output.shape!=(128,128) or not np.isfinite(output).all():raise RuntimeError('Nonfinite or wrong-shaped dense depth')
    runtime=dict(device=str(device),device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
        torch=torch.__version__,cuda=torch.version.cuda,T=rgb.shape[0],compute_seconds=compute_s,
        total_seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
        output_min=float(output.min()),output_max=float(output.max()),finite=True,clamped=False)
    model.last_inference_runtime=runtime
    return output
