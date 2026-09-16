"""Public return-to-return messages; no RGB, labels, IDs or native geometry.

Centers are inherited MZ161 hypotheses, not measured hit points. In particular,
Radar's zero center height is the original unknown-height placeholder. The
neighbor graph never changes packet validity, envelopes or independent support.
"""
import random

import torch
from torch import nn


METHOD = dict(schema='MZ174_PUBLIC_RETURN_GRAPH_V1', nodes=132, input_channels=21,
    hidden_channels=32, neighbors=8, message_layers=1,
    center_columns=[5,6,7], center_to_meters=[4.,2.,3.], range_column=2,
    range_to_meters=4., radar_modality_column=17,
    neighbor_rule='EUCLIDEAN_NOMINAL_BODY_CENTER_EXCLUDE_SELF_VALID_ONLY',
    tie_break='STABLE_CANONICAL_SLOT_ORDER',
    relative_descriptors='ACTUAL_SENDER_MINUS_RECEIVER_XYZ_M_RANGE_M_SAME_MODALITY',
    control='FIXED_SATTOLO_PERMUTATION_OF_VALID_SENDER_CONTENT_ONLY',
    permutation_seed=174016, permutation_by_count='LOCAL_RANDOM_SEED_PLUS_VALID_COUNT',
    message_mlp=[69,32,32], output_mlp=[64,32,1], raw_receiver_skip=[21,1],
    aggregation='MASKED_MEAN_ZERO_FOR_EMPTY_NEIGHBORHOOD',
    activation='RELU', threshold_logit=0., invalid_logit=-30.,
    sensor_scope='128_TOF_PLUS4_RADAR_PUBLIC_TOKENS_NO_RGB_FOV_GATE',
    labels='CALLER_ONLY;RADAR_UNKNOWN;NO_LABEL_INPUT_TO_GRAPH',
    authority='NOMINAL_CENTER_PROXY_NOT_TRUE_HIT_OR_SURFACE_CORRESPONDENCE')


def _clean_inputs(tokens, valid):
    if tokens.ndim != 3 or tokens.shape[1:] != (132,21) or tokens.shape[0] < 1:
        raise ValueError('Expected nonempty Bx132x21 public tokens')
    if valid.shape != tokens.shape[:2] or valid.dtype != torch.bool:
        raise ValueError('Expected boolean Bx132 validity mask')
    if tokens.device != valid.device or not tokens.is_floating_point():
        raise ValueError('Floating tokens and validity must share a device')
    # Mask before any arithmetic: unused NaN/Inf sentinels cannot poison a graph.
    clean = torch.where(valid[...,None],tokens,torch.zeros_like(tokens))
    if not bool(torch.isfinite(clean).all()):
        raise ValueError('Usable public token values must be finite')
    return clean


def build_graph(tokens, valid):
    """Return shared natural geometry tensors; indices -1 denote absent edges.

    Directed k=8 nearest valid senders per valid receiver, excluding the receiver.
    All calculations remain on the input device. Distances and relative features
    use restored meters, not differently normalized token coordinate axes.
    """
    clean = _clean_inputs(tokens,valid)
    b,n,_ = clean.shape
    centers = clean[...,5:8]*clean.new_tensor([4.,2.,3.])
    displacement = centers[:,None,:,:]-centers[:,:,None,:]
    squared_distance = displacement.square().sum(-1)
    available = valid[:,:,None] & valid[:,None,:]
    available = available & ~torch.eye(n,dtype=torch.bool,device=valid.device)[None]
    order = squared_distance.masked_fill(~available,float('inf')).argsort(dim=-1,stable=True)[...,:8]
    mask = available.gather(2,order)
    indices = order.masked_fill(~mask,-1)
    batch = torch.arange(b,device=clean.device)[:,None,None]
    sender_centers = centers[batch,order]
    relative_center = sender_centers-centers[:,:,None,:]
    ranges = clean[...,2]*4.
    relative_range = (ranges[batch,order]-ranges[:,:,None])[...,None]
    modality = clean[...,17] > .5
    same_modality = (modality[batch,order] == modality[:,:,None])[...,None].to(clean.dtype)
    descriptors = torch.cat((relative_center,relative_range,same_modality),dim=-1)
    descriptors = torch.where(mask[...,None],descriptors,torch.zeros_like(descriptors))
    return dict(neighbor_indices=indices,neighbor_mask=mask,
                relative_descriptors=descriptors,neighbor_count=mask.sum(-1))


def _permutation_table():
    """Fixed, label-independent single-cycle permutations for counts2..132.

    Sattolo's algorithm guarantees every valid sender changes content when there
    are at least two valid nodes. Counts0/1 have no valid directed neighborhoods.
    Uses a local Python RNG at construction only, never alters Torch/global RNG.
    """
    table = torch.full((133,132),-1,dtype=torch.long)
    for count in range(1,133):
        permutation = list(range(count))
        rng = random.Random(METHOD['permutation_seed']+count)
        for i in range(count-1,0,-1):
            j = rng.randrange(i)
            permutation[i],permutation[j] = permutation[j],permutation[i]
        table[count,:count] = torch.tensor(permutation,dtype=torch.long)
    return table


class ReturnGraph(nn.Module):
    """One message layer plus an explicit unpermuted raw receiver skip.

    API: ``model(tokens, valid) -> {'logits': Bx132, 'audit': tensor dict}``.
    Both arms have identical parameter/buffer shapes and initialization behavior.
    Load the same state_dict to obtain a strictly matched pair. The arm affects
    only sender-content lookup; natural edges and their descriptors are shared.
    Caller supplies the loss mask; no Radar target or frame label is accepted.
    """
    def __init__(self,arm='natural'):
        super().__init__()
        if arm not in ('natural','shuffled'):
            raise ValueError('Arm must be natural or shuffled')
        self.arm = arm
        self.stem = nn.Sequential(nn.Linear(21,32),nn.ReLU())
        self.message = nn.Sequential(nn.Linear(69,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU())
        self.output = nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
        self.raw_skip = nn.Linear(21,1)
        self.register_buffer('permutation_table',_permutation_table())

    def _sender_permutation(self,valid):
        b,n = valid.shape
        identity = torch.arange(n,device=valid.device).expand(b,n)
        if self.arm == 'natural':
            return identity.masked_fill(~valid,-1)
        # Pack valid nodes in canonical slot order, then permute their ranks.
        # No per-example host copies or loops are used in the forward path.
        packed = valid.to(torch.int32).argsort(dim=1,descending=True,stable=True)
        ranks = (valid.long().cumsum(1)-1).clamp_min(0)
        counts = valid.sum(1)
        content_ranks = self.permutation_table[counts[:,None],ranks].clamp_min(0)
        permutation = packed.gather(1,content_ranks)
        return permutation.masked_fill(~valid,-1)

    def forward(self,tokens,valid):
        clean = _clean_inputs(tokens,valid)
        graph = build_graph(clean,valid)
        indices,mask = graph['neighbor_indices'],graph['neighbor_mask']
        b,n,k = indices.shape
        permutation = self._sender_permutation(valid)
        content = permutation.gather(1,indices.clamp_min(0).reshape(b,n*k)).reshape(b,n,k)
        content = content.masked_fill(~mask,-1)
        receiver = self.stem(clean)
        batch = torch.arange(b,device=tokens.device)[:,None,None]
        sender = receiver[batch,content.clamp_min(0)]
        expanded_receiver = receiver[:,:,None,:].expand(b,n,k,32)
        message_input = torch.cat((expanded_receiver,sender,graph['relative_descriptors']),dim=-1)
        messages = self.message(message_input)
        messages = torch.where(mask[...,None],messages,torch.zeros_like(messages))
        aggregate = messages.sum(2)/graph['neighbor_count'].clamp_min(1)[...,None]
        raw = self.raw_skip(clean).squeeze(-1)
        logits = raw+self.output(torch.cat((receiver,aggregate),dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~valid,METHOD['invalid_logit'])
        audit = dict(graph,content_indices=content,sender_permutation=permutation,
                     receiver_features=receiver,aggregated_message=aggregate,
                     raw_receiver_logit=raw.masked_fill(~valid,METHOD['invalid_logit']))
        return dict(logits=logits,audit=audit)
