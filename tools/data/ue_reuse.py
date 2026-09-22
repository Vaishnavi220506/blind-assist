"""UE reuse admission for declared experiment inputs (not an OS sandbox)."""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import asset_catalog as catalog

DEFAULT_POLICY = Path(__file__).resolve().parents[2] / "data/ue-reuse-policy.json"
MODES = {"regression", "diagnostic", "development", "training"}


def within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def input_target(item, *, artifact_root, repo_root, connection):
    """Resolve identity without consumption, hashing or opening payloads."""
    root = artifact_root.resolve()
    asset = component = None
    if bool(item.get("asset")) == bool(item.get("path")):
        raise ValueError("input requires exactly one of asset or path")
    if item.get("asset"):
        asset, component = catalog.resolve_asset_target(connection, item["asset"])
        base = catalog.path_for_locator(asset["locator"], root)
        if component is not None:
            base = (base / component["relative_path"]).resolve()
        path = (base / item.get("relative_path", ".")).resolve()
        if not within(path, base):
            raise ValueError("relative_path escapes selected asset")
    else:
        if item.get("relative_path"):
            raise ValueError("relative_path requires asset selector")
        path = Path(item["path"])
        path = (path if path.is_absolute() else repo_root / path).resolve()
        matches = [row for row in connection.execute("SELECT * FROM assets")
                   if within(path, catalog.path_for_locator(row["locator"], root))]
        asset = max(matches, key=lambda row: len(row["locator"]), default=None)
    if not within(path, root):
        raise ValueError("input escapes artifacts.local")
    if not path.exists():
        raise ValueError(f"missing input: {path}")
    return path, asset, component


def preflight(spec, *, artifact_root, repo_root, database, policy_path=None, required=False):
    root = artifact_root.resolve()
    inventory_path = root / "evidence/ue-reuse/current/inventory.json"
    inventory = catalog.read_json(inventory_path) if inventory_path.exists() else {"assets": []}
    inventory_units = {row["locator"].casefold() for row in inventory["assets"]}
    selected, errors = [], []
    if not database.is_file():
        return dict(status="BLOCKED", errors=["Master asset catalog is missing"], inputs=[], candidates=[], source_families=[])
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        # Newly produced runs are already in the master catalog even before the
        # human browsing snapshot is refreshed. Include them in future searches.
        for row in connection.execute("SELECT locator,metadata_json FROM assets"):
            derived = json.loads(row["metadata_json"]).get("ue_reuse")
            if derived and row["locator"].casefold() not in inventory_units:
                inventory["assets"].append(dict(locator=row["locator"], category="派生结果（evaluator）",
                                                source_families=derived["source_families"]))
        for item in spec.get("inputs", []):
            try:
                path, asset, component = input_target(item, artifact_root=root, repo_root=repo_root, connection=connection)
                locator = path.relative_to(root).as_posix()
                metadata = json.loads(asset["metadata_json"]) if asset is not None else {}
                selected.append((item, path, locator, asset, component, metadata.get("ue_reuse")))
            except (ValueError, catalog.CatalogError) as exc:
                errors.append(f"{item.get('alias')}: {exc}")
    finally:
        connection.close()
    ue_command = any("/unreal/" in arg.replace("\\", "/") or "/nearfield/" in arg.replace("\\", "/") for arg in spec.get("command", []))
    applies = required or "reuse" in spec or spec.get("route") == "ue-reuse" or ue_command or any(
        locator.startswith(("unreal/", "nearfield/")) or "/".join(locator.split("/")[:2]).casefold() in inventory_units or inherited
        for _, _, locator, _, _, inherited in selected)
    if not applies:
        return None
    config_path = policy_path or DEFAULT_POLICY
    config = catalog.read_json(config_path)
    if config.get("schema") != "blindassist-ue-reuse-policy-v1":
        raise ValueError("Unsupported UE reuse policy")
    reuse = spec.get("reuse") or {}
    if not isinstance(reuse, dict):
        reuse = {}
    if reuse.get("mode") not in MODES:
        errors.append("UE reuse requires mode regression/diagnostic/development/training; no fresh/final authority")
    query = reuse.get("query", "")
    if not isinstance(query, str) or not query.strip():
        errors.append("UE reuse requires a capability query before execution")
        query = ""
    if not selected:
        errors.append("UE run requires explicit admitted inputs")
    if spec.get("cache_inputs"):
        errors.append("UE cache inputs need an admitted data-role contract; declare the cataloged payload explicitly")
    admitted, families = [], set()
    for item, path, locator, asset, component, inherited in selected:
        alias = item["alias"]
        if asset is None:
            errors.append(f"{alias}: input is not registered in the master catalog")
        matches = [rule for rule in config["contracts"] if within(path, (root / rule["locator"]).resolve())]
        contract = max(matches, key=lambda rule: len(rule["locator"]), default=None)
        status = component["evidence_status"] if component is not None else asset["evidence_status"] if asset is not None else "unknown"
        if status in {"reserved", "sealed_final", "fresh"}:
            errors.append(f"{alias}: protected catalog state {status}")
        if contract is None:
            if inherited:
                families.update(inherited["source_families"])
                if item.get("role") not in inherited["roles"]:
                    errors.append(f"{alias}: derived data role mismatch")
                if reuse.get("mode") == "training" and item.get("role") != "observation":
                    errors.append(f"{alias}: evaluator output cannot enter training")
                admitted.append(dict(alias=alias, locator=locator, asset_id=asset["asset_id"], asset_locator=asset["locator"], source_families=inherited["source_families"], role=item.get("role"), evidence_status="development_consumed"))
                continue
            errors.append(f"{alias}: no reviewed UE input contract for {locator}; consult reuse candidates")
            continue
        family = contract["source_family"]
        families.add(family)
        evidence = (repo_root / contract["evidence"]).resolve()
        if not within(evidence, repo_root.resolve()) or not evidence.is_file():
            errors.append(f"{alias}: missing policy evidence")
        if contract.get("blocked_reason"):
            errors.append(f"{alias}: {contract['blocked_reason']}")
        allowed = []
        for scope in contract.get("allowed_inputs", []):
            candidate = (root / contract["locator"] / scope["relative_path"]).resolve()
            if within(path, candidate) and scope["role"] == item.get("role"):
                allowed.append(scope)
        if not allowed:
            errors.append(f"{alias}: mixed bundle or role mismatch; select an admitted observation/configuration/evaluator path")
        if reuse.get("mode") == "training" and item.get("role") == "evaluator":
            errors.append(f"{alias}: evaluator input cannot enter training")
        if contract["evidence_status"] not in {"development_consumed", "diagnostic", "source_material", "not_applicable"}:
            errors.append(f"{alias}: input contract authority is {contract['evidence_status']}")
        admitted.append(dict(alias=alias, locator=locator, source_families=[family], role=item.get("role"),
                             asset_id=asset["asset_id"] if asset is not None else None,
                             asset_locator=asset["locator"] if asset is not None else None,
                             evidence_status=contract["evidence_status"], evidence=contract["evidence"],
                             evidence_sha256=catalog.sha256_file(evidence) if evidence.is_file() else None))
    # Rank the metadata-only local inventory; never read candidate payloads.
    terms = re.findall(r"[\w]+", query.casefold())
    ranked = []
    for row in inventory["assets"]:
        haystack = (row["locator"] + " " + row.get("category", "") + " " + " ".join(row.get("source_families", []))).casefold()
        score = sum(term in haystack for term in terms)
        if score:
            ranked.append(dict(locator=row["locator"], category=row.get("category"), score=score))
    ranked.sort(key=lambda row: (-row["score"], row["locator"]))
    return dict(status="BLOCKED" if errors else "PASS", errors=errors, mode=reuse.get("mode"), query=query,
                inputs=admitted, candidates=ranked[:12], source_families=sorted(families),
                independent_source_count="NOT_ESTABLISHED", policy_sha256=catalog.sha256_file(config_path),
                boundary="Consumed Development reuse only; families are lineage labels, not proof of independence")


def record_output_lineage(report, *, database, artifact_root, outputs, run_id, evaluator):
    """Add declared output edges and carry source-role restrictions forward."""
    connection = catalog.open_catalog(database)
    try:
        inputs = []
        for entry in report["inputs"]:
            path = artifact_root / entry["locator"]
            rows = [r for r in connection.execute("SELECT * FROM assets")
                    if within(path, artifact_root / r["locator"])]
            if rows:
                inputs.append(max(rows, key=lambda r: len(r["locator"]))["asset_id"])
            else:
                raise ValueError(f"Missing source identity during lineage registration: {path}")
        edges = []
        with connection:
            for output in outputs:
                path = artifact_root / output["path"]
                if not path.exists():
                    continue
                rows = [r for r in connection.execute("SELECT * FROM assets") if within(path, artifact_root / r["locator"])]
                if not rows:
                    raise ValueError(f"Missing output identity during lineage registration: {path}")
                asset = max(rows, key=lambda r: len(r["locator"]))
                if asset["asset_id"] in inputs:
                    raise ValueError("UE outputs must not overwrite an input asset unit")
                metadata = json.loads(asset["metadata_json"])
                # Results/predictions remain evaluator artifacts on later reuse.
                inherited = metadata.get("ue_reuse", {})
                roles = set(inherited.get("roles", [])) | {"evaluator"}
                source_families = set(inherited.get("source_families", [])) | set(report["source_families"])
                metadata["ue_reuse"] = dict(source_families=sorted(source_families), roles=sorted(roles), run_id=run_id)
                connection.execute("UPDATE assets SET metadata_json=? WHERE asset_id=?", (json.dumps(metadata), asset["asset_id"]))
                edge = "ue-run:" + catalog.sha256_bytes(f"{run_id}:{asset['asset_id']}".encode())
                connection.execute("INSERT OR IGNORE INTO derivations(derivation_id,output_asset_id,transform,transform_version,producer,parameters_json,recorded_at) VALUES(?,?,?,?,?,?,?)",
                                   (edge, asset["asset_id"], evaluator, "ue-reuse-v1", run_id, json.dumps(report), catalog.utc_now()))
                for source in set(inputs):
                    connection.execute("INSERT OR IGNORE INTO derivation_inputs VALUES(?,?,?)", (edge, source, "declared-input"))
                edges.append(edge)
        return sorted(set(edges))
    finally:
        connection.close()
