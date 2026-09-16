"""Direct corridor-risk decoder over frozen visual context and native detail."""
import torch
from torch import nn
from torch.nn import functional as F
from mz161_dense_task import DenseTaskNet

METHOD = dict(schema='MZ165_FROZEN_PRETRAINING_DIRECT_TASK_CONTEXT',
    arms=['random','pretrained'], frozen_encoder='DINOv2_ViT-S14_no_registers',
    embedding=[384,37,66], context_projection=[384,24],
    native_detail=[22,360,640], independent_return_tokens=[132,21],
    context_upsample='BILINEAR_ALIGN_CORNERS_FALSE_TO_NATIVE_FULL_IMAGE',
    reducer='MAX_PIXEL_AND_INDEPENDENT_TOKEN', threshold_logit=0.,
    loss='FRAME_BCE_PLUS_CLASS_BALANCED_KNOWN_PIXEL_BCE_IN_BOTH_ARMS',
    unknown='UNSAVED_REFERENCE_PIXELS_MASKED; NONALERT_NOT_CLEAR',
    claim='PRETRAINED_VS_RANDOM_CONTEXT_EFFECT; NOT_METRIC_DEPTH_OR_EQUIVARIANCE')


class ContextTaskNet(DenseTaskNet):
    def __init__(self):
        super().__init__()
        self.context = nn.Conv2d(384,24,1)

    def forward(self,batch):
        context = F.interpolate(self.context(batch['context']),size=batch['image'].shape[-2:],
                                mode='bilinear',align_corners=False)
        x=F.silu(self.stem(batch['image'])+context)
        for block in self.blocks: x=F.silu(block(x))
        pixel=self.pixel(x)[:,0]
        token=self.returns(batch['tokens']).squeeze(-1).masked_fill(~batch['valid'],-30.)
        local=pixel.masked_fill(batch['image'][:,15]<=0,-30.).flatten(1).amax(1)
        independent=token.amax(1)
        return dict(pixel=pixel,token=token,local=local,independent=independent,
                    frame=torch.maximum(local,independent))


def evaluate_context(rows,es,labels,maps,predictions,baseline,spec,metadata):
    """Reuse sealed-map MZ161 arithmetic through explicit legacy arm aliases."""
    from evaluate_mz161_dense_task import evaluate
    aliases={'frame':'random','dense':'pretrained'}
    value=evaluate(rows,es,labels,{k:maps[v] for k,v in aliases.items()},
        {k:predictions[v] for k,v in aliases.items()},baseline,spec,metadata)
    def rename(x):
        if isinstance(x,dict):return {aliases.get(k,k):rename(v) for k,v in x.items()}
        if isinstance(x,list):return [rename(v) for v in x]
        return x
    value=rename(value);s=value['summary'];g=s['gates'];checks=g['heldout_alert_checks']
    checks['fewer_fp_than_random_control']=checks.pop('fewer_fp_than_frame_control')
    g['fit_native_supported_nonalerts_zero']=not s['partitions']['fit']['pretrained']['native']['nonalert_with_native_corridor_contributors']
    g['overall_pass']=bool(g['fit_accuracy_at_least_95pct'] and g['heldout_alert_pass'] and g['fit_native_supported_nonalerts_zero'])
    g['spatial_gate_role']='DIAGNOSTIC_NOT_A_REQUIRED_METRIC_SURFACE_CLAIM'
    s['legacy_evaluator_arm_aliases']=aliases
    s['decision']='MZ165_PRETRAINED_CONTEXT_ALERT_COMPONENT_PASS' if g['overall_pass'] else 'MZ165_PRETRAINED_CONTEXT_ALERT_GAIN_NOT_MET'
    s['method']=METHOD
    return value
