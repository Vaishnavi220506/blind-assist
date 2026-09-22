"""One public opposing-witness path heuristic; no actual future observation."""
import active_view as av
import active_view_positive as pos


def poses(action):
    x,z=av.ACTIONS[action]
    return tuple((round(x*i/12,8),round(z*i/12,8)) for i in range(13))


def choose(initial_bins, result):
    pair=result.get("witnesses",{})
    if pair.get("IN") is None or pair.get("OUT") is None:
        return dict(action=0,reason="NO_OPPOSING_WITNESSES_FIXED_FALLBACK",forecasts=[])
    scenes=[pos.source_scene(dict(scene=pair[label])) for label in ("IN","OUT")]
    if any(av.observe(scene,av.ORIGIN)!=tuple(initial_bins) for scene in scenes):
        raise ValueError("Witness does not explain public initial bins")
    costs=[]
    forecasts=[]
    for i in range(4):
        views=[[av.observe(scene,p) for p in poses(i)] for scene in scenes]
        differing=[j for j in range(1,13) if views[0][j]!=views[1][j]]
        costs.append((min(differing,default=13),-len(differing),i))
        forecasts.append(dict(action=i,IN_bins=views[0],OUT_bins=views[1],differing_indices=differing))
    best=min(costs)
    return dict(action=best[2] if best[0]<=12 else 0,costs=costs,forecasts=forecasts,
        reason="EARLIEST_PUBLIC_WITNESS_SPLIT" if best[0]<=12 else "WITNESS_PAIR_PATH_ALIAS_FIXED_FALLBACK")
