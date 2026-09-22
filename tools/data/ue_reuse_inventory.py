#!/usr/bin/env python3
"""Metadata-only UE reuse view backed by the existing asset catalog; no payload reads."""
from __future__ import annotations

import argparse
import collections
import csv
import html
import json
import re
import subprocess
from pathlib import Path

import asset_catalog as catalog

REPO = Path(__file__).resolve().parents[2]
CATEGORIES = {
    "scenes": "01 场景工程与素材",
    "body": "02 BODY／HEAD 静态查询",
    "corridor": "03 相机走廊与事件数据",
    "city": "04 城市与近场采集",
    "replay": "05 闭环回放与传感器诊断",
    "presentation": "06 预览与展示",
    "support": "07 工具验证与混合来源待核验",
}


def category(locator: str) -> str:
    name = locator.split("/")[-1].lower()
    if name in {"citysample", "blindassiststreetlab", "blindassistobstaclelab"} or any(x in name for x in ("asset-download", "realism-assets", "sample-material")):
        return "scenes"
    if any(x in name for x in ("showcase", "preview", "visual", "finish", "delivery", "human-pose")):
        return "presentation"
    if name.startswith(("body-", "head-")):
        return "body"
    if name.startswith(("ba-core", "ba-full-event", "ba-spatial", "ba-tof", "ba-camera-corridor", "corridor-")):
        return "corridor"
    if locator.startswith("nearfield/") or name.startswith("city-"):
        return "city"
    if locator.startswith("unreal/") or name.startswith("mz"):
        return "replay"
    return "support"


def discover(root: Path) -> tuple[dict[str, set[str]], list[str]]:
    candidates: dict[str, set[str]] = collections.defaultdict(set)
    for top in ("unreal", "nearfield"):
        if (root / top).exists():
            for path in (root / top).iterdir():
                if path.is_dir() and not catalog.is_reparse_point(path):
                    candidates[f"{top}/{path.name}"].add(f"directory:{top}")
    tracked = subprocess.check_output(
        ["git", "-c", "core.quotepath=false", "ls-files", "research/active/dtr-r0/nearfield/*.md", "research/active/dtr-r0/unreal/*.md"],
        cwd=REPO, text=True, encoding="utf-8",
    ).splitlines()
    for source in tracked:
        text = (REPO / source).read_text(encoding="utf-8").replace("\\", "/")
        for locator in re.findall(r"(?:artifacts\.local/|`)((?:work|unreal|nearfield)/[A-Za-z0-9_.-]+)", text):
            locator = locator.rstrip(".")  # prose punctuation aliases real paths on Windows
            candidates[locator].add(source)
    missing = sorted(key for key in candidates if not (root / key).exists())
    return {key: refs for key, refs in candidates.items() if (root / key).is_dir() and not catalog.is_reparse_point(root / key)}, missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=REPO / "artifacts.local")
    parser.add_argument("--reuse-metadata", action="store_true", help="Rebuild navigation from the previous metadata snapshot without rescanning known assets")
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    output = root / "evidence/ue-reuse/current"
    output.mkdir(parents=True, exist_ok=True)
    candidates, missing = discover(root)
    connection = catalog.open_catalog(root / catalog.DEFAULT_DATABASE_RELATIVE)
    previous = catalog.read_json(output / "inventory.json") if args.reuse_metadata else None
    known = {row["locator"]: row for row in previous["assets"]} if previous else {}
    rows = []
    try:
        for index, (locator, refs) in enumerate(sorted(candidates.items()), 1):
            if locator in known:
                old = known[locator]
                entries = [dict(row) for row in connection.execute(
                    "SELECT relative_path, bytes, mtime_ns, sha256 FROM asset_files WHERE asset_id=?", (old["asset_id"],))]
                if len(entries) != old["files"]:
                    raise RuntimeError(f"Snapshot/catalog mismatch: {locator}; rerun without --reuse-metadata")
                scan = dict(entry_type="directory", bytes=old["bytes"], file_count=old["files"],
                            metadata_sha256=old["metadata_sha256"], entries=entries,
                            vanished_entries=old["vanished"], reparse_entries=old["skipped_reparse"])
            else:
                scan = catalog.scan_asset(root / locator)
            rule = dict(asset_kind="ue_related_bundle", asset_class="data", evidence_status="unknown",
                        storage_status="unknown", owner="dtr-r0",
                        retention_reason="UE reuse navigation; keep original payload and source authority",
                        claim_ceiling="INDEX_ONLY_NO_NEW_EVIDENCE_AUTHORITY")
            record = catalog.discovered_record(locator, locator.split("/")[0], rule, scan, "ue-reuse-index", catalog.utc_now())
            with connection:
                catalog.upsert_asset(connection, record, scan["entries"])
            actual = catalog.resolve_asset(connection, locator)
            extensions = collections.Counter(Path(entry["relative_path"]).suffix.lower() or "(none)" for entry in scan["entries"])
            # Names only: protected manifests, truth and labels are never opened.
            entrypoints = [entry["relative_path"] for entry in scan["entries"] if Path(entry["relative_path"]).name.lower() in
                           {"summary.json", "manifest.json", "spec.json", "protocol.json", "results.json", "source-admission.json"}]
            row = dict(locator=locator, category=CATEGORIES[category(locator)], asset_id=actual["asset_id"],
                       bytes=scan["bytes"], files=scan["file_count"], extensions=dict(extensions),
                       evidence_status=actual["evidence_status"], metadata_sha256=scan["metadata_sha256"],
                       vanished=scan["vanished_entries"], skipped_reparse=scan["reparse_entries"],
                       sources=sorted(refs), entrypoints=entrypoints,
                       classification_basis="navigation heuristic; mixed sources possible; consult linked protocol")
            rows.append(row)
            if index % 25 == 0:
                print(f"Indexed {index}/{len(candidates)} bundles", flush=True)
    finally:
        connection.close()
    result = dict(generated_at=catalog.utc_now(), logical_root="artifacts.local", physical_root=str(root),
                  metadata_snapshot_at=(previous.get("metadata_snapshot_at", previous["generated_at"]) if previous else catalog.utc_now()),
                  scope="Local unreal/nearfield directories plus work bundles referenced by tracked UE/nearfield documentation; remote-only payload excluded",
                  identity="metadata only; logical bytes are not physical allocation or deduplicated size",
                  missing_references=missing, assets=rows)
    catalog.atomic_write_json(output / "inventory.json", result)
    with (output / "inventory.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["category", "locator", "asset_id", "bytes", "files", "evidence_status"])
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in rows)
    e = html.escape
    body = []
    for row in rows:
        location = root / row["locator"]
        links = " ".join(f'<a href="{e((REPO / src).as_uri())}">{e(Path(src).name)}</a>' for src in row["sources"] if not src.startswith("directory:"))
        details = "<br>".join(e(item) for item in row["entrypoints"][:30])
        body.append(f'<tr><td>{e(row["category"])}</td><td><a href="{e(location.as_uri())}">{e(row["locator"])}</a><details><summary>来源与入口（{len(row["entrypoints"])}）</summary>{links}<br>{details}</details></td><td>{row["bytes"]/1024**3:.3f}</td><td>{row["files"]}</td><td>{e(str(row["extensions"]))}</td><td>{e(row["evidence_status"])}</td></tr>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>UE 数据复用目录</title>
<style>body{font:15px system-ui;margin:32px;background:#f6f8fb;color:#172238}input{padding:12px;width:70%;margin:16px 0}table{border-collapse:collapse;width:100%;background:white}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:left;vertical-align:top}a{color:#1557a0}details{max-width:650px;overflow-wrap:anywhere}th{position:sticky;top:0;background:#e3ebf5}</style>
<h1>UE 数据复用目录</h1><p>分类入口保留原始路径。目录按用途初分；混合来源须查协议。文件数不等于帧数，GiB 为逻辑大小，不代表去重后占用。</p>
<p>未知权限需核验；已消费数据可做诊断和明确披露的 Development 复用。保留集不能自动加入训练，真值和世界位姿不能作为可观测输入。此目录未读取标签内容，未校验完整采集成功。</p>
<p><a href="inventory.csv">CSV 清单</a> · <a href="inventory.json">完整 JSON 与缺失引用</a> · <a href="''' + e((REPO / "docs/asset-management/UE_REUSE.md").as_uri()) + '''">复用指南</a></p><input id="q" placeholder="搜索类别、路径、格式或来源文档"><p id="count"></p>
<table><thead><tr><th>分类</th><th>原路径／来源</th><th>GiB</th><th>文件数</th><th>文件格式</th><th>证据状态</th></tr></thead><tbody>''' + "".join(body) + '''</tbody></table>
<script>const rows=[...document.querySelectorAll('tbody tr')];function filter(){let n=0;for(const r of rows){r.hidden=!r.textContent.toLowerCase().includes(q.value.toLowerCase());if(!r.hidden)n++}document.querySelector('#count').textContent=n+' / '+rows.length+' 个目录'}const q=document.querySelector('#q');q.addEventListener('input',filter);filter();</script></html>'''
    catalog.atomic_write_text(output / "index.html", page)
    print(json.dumps(dict(output=str(output), bundles=len(rows), bytes=sum(row["bytes"] for row in rows), missing=len(missing), categories=dict(collections.Counter(row["category"] for row in rows))), ensure_ascii=False))


if __name__ == "__main__":
    main()
