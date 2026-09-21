"""Frozen initial-observation witness reuse; never new exclusion evidence."""
from copy import deepcopy
import json
from pathlib import Path
import re
import time

import active_view as sensor
import path_constraint_inference as model


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _order(entry):
    return (_canonical(entry["geometry"]), entry["label"], entry["provenancequery"])


def build_initial_pool(oldfolder):
    """Read only initial-query witness files; IDs establish provenance, not routing."""
    oldfolder = Path(oldfolder)
    sensor.verify_seal(oldfolder / "completion-seal.json")
    # The mapping is used only to identify the old initial-query files. Case IDs
    # and endpoint-query pointers never enter the returned library or matching.
    query_ids = sorted({r["queries"]["initial"] for r in _read(oldfolder / "witness-query-mapping.json")})
    unique = {}
    for query in query_ids:
        if not isinstance(query, str) or not re.fullmatch(r"query_\d+", query):
            raise ValueError("Invalid initial-query provenance identifier")
        row = _read(oldfolder / f"witness-{query}.json")
        if row["query_id"] != query:
            raise ValueError("Witness query provenance mismatch")
        observations = model._inputs(row["observations"])
        if len(observations) != 1 or observations[0]["camera"] != (0., 0.):
            raise ValueError("Initial library cannot include future observations")
        for label in ("IN", "OUT"):
            geometry = row["result"]["witnesses"][label]
            if geometry is None:
                continue
            if not model.validate_witness(geometry, observations, label):
                raise ValueError("Stored initial witness failed original model validation")
            key = (label, _canonical(geometry))
            entry = dict(label=label, geometry=deepcopy(geometry), provenancequery=query)
            # Sorted provenance makes duplicates deterministic without a target ID.
            unique.setdefault(key, entry)
    return sorted(unique.values(), key=_order)


def match(observations, pool):
    """First compatible witness per label, using only the supplied public history."""
    public = model._inputs(observations)
    started = time.perf_counter()
    checks = {"IN": 0, "OUT": 0}
    matches = {"IN": None, "OUT": None}
    for entry in sorted(pool, key=_order):
        label = entry["label"]
        if label not in matches:
            raise ValueError("Unknown cached witness label")
        if matches[label] is not None:
            continue
        checks[label] += 1
        if model.validate_witness(entry["geometry"], public, label):
            matches[label] = deepcopy(entry)
    return dict(witnesses={label: deepcopy(entry["geometry"]) if entry is not None else None
                           for label, entry in matches.items()},
                matches=matches, checks={**checks, "total": sum(checks.values())},
                elapsed_seconds=time.perf_counter()-started,
                authority="CONSTRUCTIVE_PUBLIC_HISTORY_WITNESSES_ONLY", solver_calls=0)


def reconcile(oldresult, cached):
    """Merge already validated cache matches, without upgrading old UNKNOWN.

    `cached` must be the output of match for this same public history. Keeping
    validation in match avoids inferring witness validity from labels or IDs.
    Numerical solver receipts remain immutable historical evidence.
    """
    result = deepcopy(oldresult)
    olddecision = oldresult["decision"]
    if olddecision not in ("UNKNOWN", "IN_MODEL_CONDITIONAL", "OUT_MODEL_CONDITIONAL"):
        raise ValueError("Unknown original decision authority")
    added = []
    for label in ("IN", "OUT"):
        if result["witnesses"][label] is None and cached["witnesses"][label] is not None:
            result["witnesses"][label] = deepcopy(cached["witnesses"][label])
            added.append(label)
    excluded = {m["requested_label"] for m in oldresult.get("solver_metadata", [])
                if m.get("solver_status") == 2 and m.get("exclusion_supported")}
    excluded.update(label for label, meta in oldresult.get("exclusion_metadata", {}).items()
                    if meta.get("solver_status") == 2 and meta.get("reported_infeasible"))
    if olddecision != "UNKNOWN":
        excluded.add("OUT" if olddecision == "IN_MODEL_CONDITIONAL" else "IN")
    conflicts = sorted(label for label in excluded if result["witnesses"][label] is not None)
    result["valid_witness_count"] = sum(v is not None for v in result["witnesses"].values())
    if conflicts:
        result.update(decision="UNKNOWN", status="UNKNOWN", reason="VALID_WITNESS_CONTRADICTS_NUMERICAL_EXCLUSION")
    elif olddecision == "UNKNOWN":
        result.update(decision="UNKNOWN", status="UNKNOWN")
        if added and result["valid_witness_count"] == 2:
            result["reason"] = "OPPOSING_WITNESSES_VALIDATED_WITH_INITIAL_PUBLIC_CACHE"
    if added:
        # Prior pair forecasts do not describe a newly completed pair. No extra
        # forecast or optimization is performed to manufacture such metadata.
        result["opposing_witness_analysis"] = None
    result["cache_recovery"] = dict(added_labels=added, exclusion_conflict_labels=conflicts,
        original_decision=olddecision, cached_alone_may_create_conditional_decision=False,
        original_solver_receipts_preserved=True, solver_calls=0,
        added_provenance={label: cached["matches"][label]["provenancequery"] for label in added},
        checks=deepcopy(cached["checks"]), elapsed_seconds=cached["elapsed_seconds"])
    return result


def recover_with_cache(observations, pool, oldresult):
    matched = match(observations, pool)
    return dict(result=reconcile(oldresult, matched), matches=matched,
                solver_calls=0, cache_entries=len(pool))
