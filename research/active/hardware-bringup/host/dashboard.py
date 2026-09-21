"""Local diagnostic viewer; bounded recording through existing evidence collectors."""
from __future__ import annotations

import argparse
import bisect
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, urlencode, urlsplit
import uuid

from capture import ports
from orientation import plan as orientation_plan, cue as orientation_cue
import manual_orientation

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE.parents[3] / "artifacts.local" / "hardware-bringup"
FRESH_MS = 1500  # UI freshness only, not a validated alert threshold.
LIMITS = "主机接收时间对应，非曝光同步；网格为原始区域顺序，未做相机投影。近零提示不是已验证的过滤器，原始值保留。"


def cell_view(record):
    sensor = record["sensor"]
    derived = record.get("derived", {})
    diagnostic = sensor.get("diagnostic", {})
    matches = derived.get("diagnostic", {}).get("distance_conversion_match", [])
    histograms = derived.get("hist_normalized", [])
    cells = []
    for zone, (distance, status, targets) in enumerate(zip(
            sensor["distance_mm"], sensor["target_status"], sensor["nb_target"])):
        reasons = []
        valid = distance > 0 and status == 5 and targets > 0
        if not valid:
            quality = "UNKNOWN"
            if distance <= 0:
                reasons.append("非正距离")
            if status != 5:
                reasons.append(f"status={status} 未通过当前规则")
            if targets <= 0:
                reasons.append("无目标返回")
        elif distance <= 3:
            quality = "SUSPECT"
            reasons.append("1–3 mm 近零读数：诊断提示，非已验证阈值")
        else:
            quality = "KNOWN"
        if zone < len(matches) and matches[zone] is False:
            reasons.append("Q2 距离转换不一致")
            if valid:
                quality = "SUSPECT"
        q2 = diagnostic.get("distance_q2", [])
        q2 = q2[zone] if zone < len(q2) else None
        if q2 is not None and q2 < 0:
            reasons.append("换算前距离为负，ULD 输出截为零")

        def value(name):
            values = diagnostic.get(name, [])
            return values[zone] if zone < len(values) else None

        cells.append({"zone": zone, "raw_mm": distance, "status": status,
                      "targets": targets, "quality": quality, "reasons": reasons,
                      "q2": q2, "sigma_mm": value("range_sigma_mm"),
                      "signal_kcps_spad": value("signal_kcps_spad"),
                      "ambient_kcps_spad": value("ambient_kcps_spad"),
                      "hist": histograms[zone] if zone < len(histograms) else []})
    return cells


class RunIndex:
    """Incremental JSONL reader; only complete lines enter the live view."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.rows = {"camera": [], "tof": []}
        self.times = {"camera": [], "tof": []}
        self.offsets, self.pending, self.line_numbers = {}, {}, {}
        self.images = set()
        self.issues = []

    def issue(self, message):
        if message not in self.issues:
            self.issues.append(message)

    def read_lines(self, relative):
        path = self.root / relative
        if not path.exists():
            return []
        position = self.offsets.get(relative, 0)
        if path.stat().st_size < position:
            self.issue(f"{relative}: 文件被截断，停止增量读取")
            return []
        with path.open("rb") as stream:
            stream.seek(position)
            data = stream.read()
            self.offsets[relative] = stream.tell()
        combined = self.pending.get(relative, b"") + data
        lines = combined.split(b"\n")
        self.pending[relative] = lines.pop()
        result = []
        for line in lines:
            number = self.line_numbers.get(relative, 0) + 1
            self.line_numbers[relative] = number
            try:
                value = json.loads(line, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")))
                if not isinstance(value, dict):
                    raise ValueError("not an object")
                result.append(value)
            except (ValueError, UnicodeError):
                self.issue(f"{relative}:{number} 无效 JSON，原文件保留")
        return result

    def refresh(self):
        for stream in self.rows:
            for record in self.read_lines(f"{stream}/frames.jsonl"):
                stamp = record.get("host_received_monotonic_ns")
                if type(stamp) is not int or stamp < 0:
                    self.issue(f"{stream}: 缺少有效主机接收时间")
                    continue
                if self.times[stream] and stamp < self.times[stream][-1]:
                    self.issue(f"{stream}: 接收时间倒退，该记录不进入时间轴")
                    continue
                if stream == "camera":
                    filename = record.get("filename", "")
                    image = (self.root / "camera" / filename).resolve()
                    if (record.get("jpeg_validated") is not True or
                            not image.is_relative_to((self.root / "camera").resolve()) or
                            not image.is_file()):
                        self.issue("camera: 缺失或未通过验证的图片不显示")
                        continue
                    self.images.add(filename)
                else:
                    try:
                        cells = cell_view(record)
                        if len(cells) not in (16, 64):
                            raise ValueError("shape")
                    except (TypeError, KeyError, ValueError, IndexError):
                        self.issue("tof: 区域字段不完整，该记录不显示")
                        continue
                self.rows[stream].append(record)
                self.times[stream].append(stamp)
        for issue in self.read_lines("tof/issues.jsonl"):
            self.issue(f"ToF 第 {issue.get('line', '?')} 行：{issue.get('reason', 'issue')}")
        for event in self.read_lines("tof/events.jsonl"):
            sensor = event.get("sensor", {})
            if sensor.get("type") == "error" or sensor.get("status", 0) != 0:
                self.issue(f"ToF 设备记录：{sensor.get('message', sensor.get('op', 'error'))}")
        for event in self.read_lines("camera/events.jsonl"):
            if event.get("kind") in ("capture_failure", "sequence_discontinuity", "interrupted"):
                self.issue(f"相机：{event.get('kind')} {event.get('error', '')}")

    def extent(self):
        nonempty = [v for v in self.times.values() if v]
        if not nonempty:
            return 0, 0
        return min(v[0] for v in nonempty), max(v[-1] for v in nonempty)

    def select(self, stream, at_ns):
        index = bisect.bisect_right(self.times[stream], at_ns) - 1
        return self.rows[stream][index] if index >= 0 else None

    def image_path(self, filename):
        if filename not in self.images:
            raise ValueError("图片未列入已验证帧记录")
        path = (self.root / "camera" / filename).resolve()
        if not path.is_relative_to((self.root / "camera").resolve()):
            raise ValueError("无效图片路径")
        return path


class Dashboard:
    def __init__(self, data_root, camera_port=None, tof_port=None):
        self.data_root = Path(data_root).resolve()
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.controls = self.data_root / ".dashboard-control"
        self.controls.mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.mode, self.current = "idle", None
        self.process = self.log = self.stop_file = None
        self.started, self.seconds = 0, 0
        self.camera_port, self.tof_port = camera_port, tof_port
        self.run_id = None
        self.closing = False
        self.guide = None
        self.manual = None
        self.manual_timer = None

    def available_ports(self):
        return [{**p, "role_hint": "camera" if p["port"] == self.camera_port else
                 "tof" if p["port"] == self.tof_port else None} for p in ports()]

    def runs(self):
        result = []
        for path in self.data_root.iterdir():
            if (path.is_dir() and path.resolve().is_relative_to(self.data_root) and
                    (path / "camera/frames.jsonl").exists() and (path / "tof/frames.jsonl").exists()):
                result.append({"id": path.name, "label": path.name})
        return sorted(result, key=lambda r: r["id"], reverse=True)

    def active(self):
        return self.process is not None and self.process.poll() is None

    def start(self, payload):
        with self.lock:
            if self.closing or self.active():
                raise ValueError("已有录制正在进行，或服务正在关闭")
            camera, tof = payload.get("camera_port"), payload.get("tof_port")
            available = {p["port"]: p for p in ports()}
            if (not camera or not tof or camera == tof or any(
                    p not in available or available[p]["vid"] != 0x303A for p in (camera, tof))):
                raise ValueError("请选择两个不同且已枚举的 ESP32 端口")
            seconds = float(payload.get("seconds", 20))
            if not math.isfinite(seconds) or not 3 <= seconds <= 300:
                raise ValueError("录制时长须为 3–300 秒")
            guide = payload.get("guide")
            if guide not in (None, "orientation-v1", "vertical-manual-v1", manual_orientation.CNH_GUIDE):
                raise ValueError("未知引导类型")
            if guide == "orientation-v1" and seconds != 50:
                raise ValueError("方向检查固定为 50 秒")
            if guide == "vertical-manual-v1" and seconds != 180:
                raise ValueError("手动上下检查的会话上限固定为 180 秒")
            if guide == manual_orientation.CNH_GUIDE and seconds != 240:
                raise ValueError("CNH 近远对照的会话上限固定为 240 秒")
            name = "dashboard-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:6]
            root = self.data_root / name
            stop = self.controls / (name + ".stop")
            command = [sys.executable, "-B", str(HERE / "pair_capture.py"), "--camera-port", camera,
                       "--tof-port", tof, "--seconds", str(seconds), "--output", str(root),
                       "--tof-query-config", "--stop-file", str(stop), "--label",
                       "Dashboard user-started bounded recording; scene unmeasured; no exposure synchronization"]
            if self.log:
                self.log.close()
            log = (self.controls / (name + ".log")).open("xb")
            try:
                schedule = orientation_plan(time.monotonic_ns()) if guide == "orientation-v1" else None
                manual = manual_orientation.plan(time.monotonic_ns(), guide) if guide in ("vertical-manual-v1", manual_orientation.CNH_GUIDE) else None
                if guide == manual_orientation.CNH_GUIDE:
                    protocol_path = HERE.parent / "cnh-components-protocol.json"
                    protocol_bytes = protocol_path.read_bytes()
                    json.loads(protocol_bytes)
                    manual.update(protocol_file=protocol_path.name,
                                  protocol_sha256=hashlib.sha256(protocol_bytes).hexdigest())
                if schedule:
                    schedule.update(run_id=name, camera_port=camera, tof_port=tof)
                    with (self.controls / (name + ".orientation.json")).open("x", encoding="utf-8") as handle:
                        json.dump(schedule, handle, ensure_ascii=False, indent=2)
                if manual:
                    manual.update(run_id=name, camera_port=camera, tof_port=tof)
                    with (self.controls / (name + ".manual.json")).open("x", encoding="utf-8") as handle:
                        json.dump(manual, handle, ensure_ascii=False, indent=2)
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except Exception:
                log.close()
                raise
            self.process, self.log, self.stop_file = process, log, stop
            self.mode, self.current, self.run_id = "live", RunIndex(root), name
            self.started, self.seconds = time.monotonic(), seconds
            self.guide = schedule
            self.manual = manual
            return {"run_id": name, "status": "录制启动中"}

    def mark_manual(self):
        with self.lock:
            if not self.manual or not self.active():
                raise ValueError("没有正在运行的手动分段检查")
            state = self.state()
            if any(not state[k] or state[k]["age_ms"] > 500 for k in ("camera", "tof")):
                raise ValueError("请等两路实时数据恢复后再标记；当前缺数据或数据过期")
            candidate = {**self.manual, "segments": list(self.manual["segments"])}
            segment = manual_orientation.mark(candidate, time.monotonic_ns())
            path = self.controls / (self.run_id + ".manual.json")
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(path)
            self.manual = candidate
            if len(self.manual["segments"]) == len(manual_orientation.phases(self.manual)):
                run_id = self.run_id
                self.manual_timer = threading.Timer(manual_orientation.SEGMENT_SECONDS, self.finish_manual, args=(run_id,))
                self.manual_timer.daemon = True
                self.manual_timer.start()
            return {"status": "开始记录 6 秒，请保持不动", "segment": segment}

    def finish_manual(self, run_id):
        with self.lock:
            if self.run_id == run_id and self.manual:
                self.stop()

    def stop(self):
        with self.lock:
            if self.manual_timer:
                self.manual_timer.cancel()
                self.manual_timer = None
            if self.active():
                self.stop_file.write_text("User requested graceful stop\n", encoding="utf-8")
            return {"status": "正在结束当前帧并保存" if self.active() else "录制已结束"}

    def replay(self, run_id):
        with self.lock:
            if self.active():
                raise ValueError("请先结束当前录制，再打开回放")
            if run_id not in {r["id"] for r in self.runs()}:
                raise ValueError("未知录制目录")
            self.current, self.mode, self.run_id = RunIndex(self.data_root / run_id), "replay", run_id
            self.guide = None
            self.manual = None
            self.current.refresh()
            return {"run_id": run_id, "status": "回放已载入"}

    def state(self, at_ms=None):
        with self.lock:
            active = self.active()
            state = {"mode": self.mode, "status": "等待选择录制或回放", "run_id": self.run_id,
                     "duration_ms": 0, "position_ms": 0, "camera": None, "tof": None,
                     "counts": {"known": 0, "unknown": 0, "suspect": 0}, "recording": None,
                     "issues": [], "limits": LIMITS, "guide": None, "manual": None}
            if self.current is None:
                return state
            self.current.refresh()
            start, end = self.current.extent()
            duration = max(0, (end - start) / 1e6)
            if self.mode == "live":
                now = time.monotonic_ns()
                if self.guide:
                    state["guide"] = orientation_cue(self.guide, now, active)
                if self.manual:
                    state["manual"] = manual_orientation.status(self.manual, now, active)
                position = max(0, (now - start) / 1e6) if start else 0
                state["status"] = "录制中" if active else "录制已结束，可选择回放"
                state["recording"] = {"active": active,
                                      "elapsed_s": max(0, time.monotonic() - self.started) if active else duration / 1000,
                                      "requested_seconds": self.seconds, "output": str(self.current.root)}
                if not active and self.process and self.process.returncode != 0:
                    self.current.issue("采集以非零状态结束，原始日志与错误已保留")
                if not active and self.log:
                    self.log.close()
                    self.log = None
            else:
                position = float(at_ms or 0)
                if not math.isfinite(position):
                    raise ValueError("回放位置必须为有限数值")
                position = min(duration, max(0, position))
                now = start + round(position * 1e6)
                state["status"] = "回放：按主机接收时间，无未来帧填充"
            state.update(duration_ms=duration, position_ms=position, issues=list(self.current.issues))
            camera = self.current.select("camera", now)
            if camera:
                age = max(0, (now - camera["host_received_monotonic_ns"]) / 1e6)
                state["camera"] = {"seq": camera["header"]["seq"], "age_ms": age,
                                   "stale": age > FRESH_MS,
                                   "url": "/api/frame?" + urlencode({"run": self.run_id, "file": camera["filename"]})}
            tof = self.current.select("tof", now)
            if tof:
                age = max(0, (now - tof["host_received_monotonic_ns"]) / 1e6)
                cells = cell_view(tof)
                sensor = tof["sensor"]
                side = math.isqrt(len(cells))
                state["tof"] = {"seq": sensor["seq"], "age_ms": age, "stale": age > FRESH_MS,
                                "rows": side, "cols": side, "cells": cells}
                state["counts"] = {k.lower(): sum(c["quality"] == k for c in cells)
                                   for k in ("KNOWN", "UNKNOWN", "SUSPECT")}
            return state

    def frame(self, run_id, filename):
        with self.lock:
            if run_id != self.run_id or not self.current:
                raise ValueError("图片不属于当前会话")
            return self.current.image_path(filename).read_bytes()

    def close(self):
        self.closing = True
        self.stop()
        # pair_capture owns bounded child timeouts; give it time to finalize first.
        if self.process and self.active():
            self.process.wait(timeout=self.seconds + 20)
        if self.log:
            self.log.close()
            self.log = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def send(self, status, payload, content_type="application/json; charset=utf-8"):
        if not isinstance(payload, bytes):
            payload = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; frame-ancestors 'none'")
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def local_request(self):
        allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        if self.headers.get("Host") not in allowed:
            raise ValueError("只允许本机界面访问")
        origin = self.headers.get("Origin")
        if origin and origin not in {f"http://{host}" for host in allowed}:
            raise ValueError("拒绝跨站请求")

    def do_GET(self):
        try:
            self.local_request()
            url = urlsplit(self.path)
            query = parse_qs(url.query)
            dashboard = self.server.dashboard
            if url.path == "/":
                self.send(200, (HERE / "dashboard.html").read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/ports":
                self.send(200, dashboard.available_ports())
            elif url.path == "/api/runs":
                self.send(200, dashboard.runs())
            elif url.path == "/api/state":
                self.send(200, dashboard.state(query.get("at_ms", [None])[0]))
            elif url.path == "/api/frame":
                self.send(200, dashboard.frame(query.get("run", [None])[0], query.get("file", [None])[0]), "image/jpeg")
            else:
                self.send(404, {"error": "未找到"})
        except (ValueError, OSError, TypeError) as exc:
            self.send(400, {"error": str(exc)})

    def do_POST(self):
        try:
            self.local_request()
            if self.headers.get_content_type() != "application/json":
                raise ValueError("仅接受 JSON 请求")
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 8192:
                raise ValueError("无效请求长度")
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError("请求必须为对象")
            dashboard = self.server.dashboard
            if self.path == "/api/start":
                result = dashboard.start(payload)
            elif self.path == "/api/stop":
                result = dashboard.stop()
            elif self.path == "/api/manual-mark":
                result = dashboard.mark_manual()
            elif self.path == "/api/replay":
                result = dashboard.replay(payload.get("id"))
            elif self.path == "/api/shutdown":
                dashboard.closing = True
                result = dashboard.stop()
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else:
                self.send(404, {"error": "未找到"})
                return
            self.send(200, result)
        except (ValueError, OSError, TypeError) as exc:
            self.send(400, {"error": str(exc)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--data-root", type=Path, default=ARTIFACTS / "paired")
    parser.add_argument("--camera-port", help="explicit initial UI selection; does not open a port")
    parser.add_argument("--tof-port", help="explicit initial UI selection; does not open a port")
    parser.add_argument("--ready-file", type=Path)
    args = parser.parse_args()
    if not args.data_root.resolve().is_relative_to(ARTIFACTS.resolve()):
        parser.error("data root must stay under canonical hardware-bringup artifacts")
    dashboard = Dashboard(args.data_root, args.camera_port, args.tof_port)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.dashboard = dashboard
    ready = {"url": f"http://127.0.0.1:{server.server_port}/", "scope": "localhost only; no recording until Start"}
    if args.ready_file:
        args.ready_file.write_text(json.dumps(ready, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(ready), flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        dashboard.close()


if __name__ == "__main__":
    main()
