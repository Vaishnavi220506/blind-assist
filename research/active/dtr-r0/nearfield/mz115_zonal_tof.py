"""Hypothetical finite-footprint ToF forward model, not calibrated ST firmware.

Private input is nine first-hit range/reflectance samples. Output is at most two
zone-level returns; sample coordinates, geometry identity and lineage stay private.
"""
import math


GRID = 8
FOV_DEG = 45.
SUBRAYS_PER_AXIS = 3
MAX_RANGE_M = 4.
BIN_M = .05
PEAK_SEPARATION_M = .60
SIGNAL_FLOOR = .01
MAX_TARGETS = 2
RANGE_NOISE_SIGMA_M = .04
RANGE_QUANTUM_M = .02
MERGED_SPREAD_M = .10
MODEL = 'HYPOTHETICAL_FINITE_FOOTPRINT_V1'


def zone_geometry(zone_id):
    """Ascending angular bounds; row zero is the upper image-facing zone row."""
    if not isinstance(zone_id,int) or not 0<=zone_id<GRID*GRID:raise ValueError('Invalid zone ID')
    iy,ix=divmod(zone_id,GRID);size=FOV_DEG/GRID
    theta_low=-FOV_DEG/2+ix*size;theta_high=theta_low+size
    phi_high=FOV_DEG/2-iy*size;phi_low=phi_high-size
    rays=[]
    for sy in range(SUBRAYS_PER_AXIS):
        for sx in range(SUBRAYS_PER_AXIS):
            rays.append(dict(subray=sy*SUBRAYS_PER_AXIS+sx,
                             theta_deg=theta_low+(sx+.5)*size/SUBRAYS_PER_AXIS,
                             phi_deg=phi_high-(sy+.5)*size/SUBRAYS_PER_AXIS))
    return dict(zone_id=zone_id,theta_bounds_deg=[theta_low,theta_high],
                phi_bounds_deg=[phi_low,phi_high]),rays


def peak_bins(values):
    """A positive local-max plateau is one peak, using its lower middle bin."""
    peaks=[];i=0
    while i<len(values):
        j=i
        while j+1<len(values) and values[j+1]==values[i]:j+=1
        before=values[i-1] if i else 0.
        after=values[j+1] if j+1<len(values) else 0.
        if values[i]>0 and values[i]>before and values[j]>after:peaks.append((i+j)//2)
        i=j+1
    return peaks


def measure_zone(private_hits,rng,packet_received=True):
    """Deterministic reduction plus chosen Gaussian range noise.

    Histogram bins round half up; smoothing has zero padding and no edge
    renormalization. Hits attach to the nearest positive local maximum (ties:
    lower bin). Adjacent maxima less than .60m apart merge transitively. Report
    SIM_MERGED if multiple maxima merge OR component hit span is at least .10m.
    Strength is summed hit contribution; range is its weighted mean. These are
    frozen simulation rules, not a firmware emulator or measured photon model.
    """
    if len(private_hits)!=9:raise ValueError('Exactly nine private subray samples required')
    histogram=[0.]*(int(round(MAX_RANGE_M/BIN_M))+1)
    contributions=[]
    for index,hit in enumerate(private_hits):
        distance=hit.get('range_m')
        if distance is None:continue
        reflectance=hit.get('reflectance_proxy',1.)
        if not math.isfinite(distance) or distance<=0 or distance>MAX_RANGE_M+1e-9:
            raise ValueError('Private hit outside finite range')
        if not math.isfinite(reflectance) or not 0<=reflectance<=1:
            raise ValueError('Reflectance proxy must lie in [0,1]')
        weight=reflectance/(9*max(distance,.2)**2)
        if weight<=0:continue
        b=min(len(histogram)-1,int(math.floor(distance/BIN_M+.5)))
        histogram[b]+=weight
        contributions.append(dict(hit_index=index,range_m=distance,weight=weight,bin=b))
    smooth=[.25*(histogram[i-1] if i else 0.)+.5*histogram[i]+
            .25*(histogram[i+1] if i+1<len(histogram) else 0.) for i in range(len(histogram))]
    peaks=peak_bins(smooth);clusters=[]
    for peak in peaks:
        if clusters and (peak-clusters[-1][-1])*BIN_M<PEAK_SEPARATION_M-1e-12:
            clusters[-1].append(peak)
        else:clusters.append([peak])
    memberships=[[] for _ in clusters]
    for contribution in contributions:
        if not peaks:continue
        nearest=min(peaks,key=lambda p:(abs(contribution['range_m']-p*BIN_M),p))
        owner=next(i for i,cluster in enumerate(clusters) if nearest in cluster)
        memberships[owner].append(contribution)
    components=[]
    for cluster,hits in zip(clusters,memberships):
        if not hits:continue
        strength=sum(h['weight'] for h in hits)
        distance=sum(h['range_m']*h['weight'] for h in hits)/strength
        spread=max(h['range_m'] for h in hits)-min(h['range_m'] for h in hits)
        merged=len(cluster)>1 or spread>=MERGED_SPREAD_M-1e-12
        components.append(dict(peak_bins=cluster,hit_indices=[h['hit_index'] for h in hits],
                               pre_noise_range_m=distance,signal_strength_proxy=strength,
                               private_hit_span_m=spread,status='SIM_MERGED' if merged else 'SIM_VALID',
                               detected=strength>=SIGNAL_FLOOR))
    detected=sorted([c for c in components if c['detected']],
                    key=lambda c:(-c['signal_strength_proxy'],c['pre_noise_range_m']))
    selected=detected[:MAX_TARGETS] if packet_received else []
    targets=[];lineage=[]
    for component in selected:
        noisy=component['pre_noise_range_m']+rng.gauss(0.,RANGE_NOISE_SIGMA_M)
        measured=max(RANGE_QUANTUM_M,round(noisy/RANGE_QUANTUM_M)*RANGE_QUANTUM_M)
        targets.append(dict(distance_m=measured,range_noise_sigma_m=RANGE_NOISE_SIGMA_M,
                            signal_strength_proxy=component['signal_strength_proxy'],status=component['status']))
        lineage.append(dict(target_index=len(targets)-1,peak_bins=component['peak_bins'],
                            hit_indices=component['hit_indices'],pre_noise_range_m=component['pre_noise_range_m']))
    private=dict(histogram=histogram,smoothed_histogram=smooth,positive_peak_bins=peaks,components=components,
                 returned_lineage=lineage,packet_received=bool(packet_received),
                 output_state='SIM_PACKET_MISSING' if not packet_received else 'SIM_RETURNS' if targets else 'SIM_NO_DETECTION')
    return targets,private
