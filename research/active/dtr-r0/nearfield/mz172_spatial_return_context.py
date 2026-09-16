"""Public full-zone visual messages with an independent raw-return fallback."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from mz165_pretrained_task import ContextTaskNet
from mz171_return_supervision import loss as witness_loss,evaluate_witness

ARMS=('pooled','registered')
METHOD=dict(schema='MZ172_PUBLIC_ZONE_RETURN_CONTEXT',arms=list(ARMS),samples_per_zone=[8,8],
    sample_authority='FULL_PUBLIC_ANGULAR_ZONE_CELL_CENTERS_NOT_NATIVE_HIT_LOCATIONS',
    message='RETURN_QUERY_DOT_SPATIAL_KEY_VS_UNIFORM_VISIBLE_ZONE_MEAN',
    raw_return_path='UNCHANGED_INDEPENDENT_132_SLOT_MLP;NO_RGB_VALIDITY_VETO',
    tof_readout='MAX_RAW_AND_CONTEXTUAL_LOGIT',radar_readout='RAW_ONLY',
    threshold_logit=0.,loss='SAME_FRAME_PIXEL_AND_KNOWN_SLOT_BCE_IN_BOTH_ARMS',
    geometry='ATTENTION_IS_NOT_RETURN_OWNERSHIP_OR_SUPPORT_PRUNING')


class SpatialReturnTaskNet(ContextTaskNet):
    def __init__(self,arm):
        if arm not in ARMS:raise ValueError('Unknown context arm')
        super().__init__();self.arm=arm
        self.return_query=nn.Linear(21,24)
        self.spatial_key=nn.Linear(48,24)
        self.visual_value=nn.Linear(46,24)
        self.contextual_return=nn.Sequential(nn.Linear(46,32),nn.SiLU(),nn.Linear(32,16),nn.SiLU(),nn.Linear(16,1))
        axis=(torch.arange(8,dtype=torch.float32)+.5)/4-1
        yy,xx=torch.meshgrid(axis,axis,indexing='ij')
        self.register_buffer('zone_position',torch.stack((xx,yy),-1).reshape(64,2))

    def pool_samples(self,tokens,samples,visible):
        # Samples B x 64 zones x 64 points x (22 public detail + 24 visual context).
        b=tokens.shape[0]
        if samples.shape!=(b,64,64,46) or visible.shape!=(b,64,64):
            raise ValueError('Invalid full-zone context tensors')
        q=self.return_query(tokens[:,:128].reshape(b,64,2,21))
        pos=self.zone_position[None,None].expand(b,64,-1,-1)
        key=self.spatial_key(torch.cat((samples,pos),-1))
        score=torch.einsum('bzsd,bzpd->bzsp',q,key)/math.sqrt(24)
        if self.arm=='pooled':score=torch.zeros_like(score)
        mask=visible[:,:,None,:]
        attention=torch.softmax(score.masked_fill(~mask,-10000.),-1)*mask
        attention=attention/attention.sum(-1,keepdim=True).clamp_min(1e-12)
        value=self.visual_value(samples)
        message=torch.einsum('bzsp,bzpd->bzsd',attention,value).reshape(b,128,24)
        return message,attention.reshape(b,128,64)

    def forward(self,batch):
        image=batch['image'];tokens=batch['tokens'];valid=batch['valid']
        context=F.interpolate(self.context(batch['context']),size=image.shape[-2:],mode='bilinear',align_corners=False)
        x=F.silu(self.stem(image)+context)
        for block in self.blocks:x=F.silu(block(x))
        pixel=self.pixel(x)[:,0]
        raw=self.returns(tokens).squeeze(-1).masked_fill(~valid,-30.)
        samples=F.grid_sample(torch.cat((image,context),1),batch['zone_grid'],
            mode='bilinear',padding_mode='zeros',align_corners=True).permute(0,2,3,1)
        visible=batch['zone_visible'];message,attention=self.pool_samples(tokens,samples,visible)
        coverage=visible.float().mean(-1).repeat_interleave(2,-1)
        has_context=visible.any(-1).repeat_interleave(2,-1)&valid[:,:128]
        inputs=torch.cat((tokens[:,:128],message,coverage[:,:,None]),-1)
        contextual=self.contextual_return(inputs).squeeze(-1).masked_fill(~has_context,-30.)
        contextual=torch.cat((contextual,torch.full_like(raw[:,128:],-30.)),1)
        tof=torch.where(has_context,torch.maximum(raw[:,:128],contextual[:,:128]),raw[:,:128])
        token=torch.cat((tof,raw[:,128:]),1).masked_fill(~valid,-30.)
        attention=attention*valid[:,:128,None]
        local=pixel.masked_fill(image[:,15]<=0,-30.).flatten(1).amax(1)
        independent=token.amax(1)
        return dict(pixel=pixel,token=token,local=local,independent=independent,
            frame=torch.maximum(local,independent),token_raw=raw,token_context=contextual,
            attention=attention,context_valid=has_context)


def loss(output,frame_target,pixel_target,pixel_weight,slot_target,slot_known,arm):
    if arm not in ARMS:raise ValueError('Unknown context arm')
    return witness_loss(output,frame_target,pixel_target,pixel_weight,slot_target,slot_known,'witness')


def evaluate_context(rows,es,labels,maps,predictions,baseline,spec,metadata,
                     token_logits,slot_target,slot_known):
    aliases={'control':'pooled','witness':'registered'}
    v=evaluate_witness(rows,es,labels,{k:maps[a] for k,a in aliases.items()},
        {k:predictions[a] for k,a in aliases.items()},baseline,spec,metadata,
        {k:token_logits[a] for k,a in aliases.items()},slot_target,slot_known)
    def rename(x):
        if isinstance(x,dict):return {('vs_pooled' if k=='vs_control' else aliases.get(k,k)):rename(z) for k,z in x.items()}
        if isinstance(x,list):return [rename(z) for z in x]
        return x
    v=rename(v);s=v['summary'];g=s['gates'];checks=g['heldout_alert_checks']
    checks['fewer_fp_than_pooled']=checks.pop('fewer_fp_than_matched_control')
    checks['all_pooled_true_frames_retained']=checks.pop('all_control_true_frames_retained')
    checks['all_pooled_events_without_extra_delay']=checks.pop('all_control_events_without_extra_delay')
    g['warning_requirements_pass']=g['overall_pass']
    g['registered_slot_discrimination_improves']=s['branch_learning_effect']
    g['overall_pass']=bool(g['overall_pass'] and s['branch_learning_effect'])
    s['method']=METHOD;s['legacy_evaluator_arm_aliases']={'frame':'pooled','dense':'registered'}
    s['decision']='MZ172_REGISTERED_RETURN_CONTEXT_COMPONENT_PASS' if g['overall_pass'] else 'MZ172_REGISTERED_RETURN_CONTEXT_GAIN_NOT_MET'
    return v
