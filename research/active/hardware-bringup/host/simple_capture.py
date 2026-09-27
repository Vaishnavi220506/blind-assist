"""Four short, operator-started camera/ToF captures with a localhost guide and live view."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

from capture import ports, write_json
from dashboard import Handler as LocalHandler, RunIndex

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE.parents[3] / "artifacts.local" / "hardware-bringup"
SEGMENTS = [
    {"label": "原有背景", "id": "background", "seconds": 30},
    {"label": "放入、拿走物体两次", "id": "object", "seconds": 60},
    {"label": "缓慢移动物体", "id": "movement", "seconds": 30},
    {"label": "恢复原有背景", "id": "restored", "seconds": 30},
]


class Guide:
    def __init__(self, data_root, process_factory=subprocess.Popen, port_source=ports, camera_url=None):
        self.data_root = Path(data_root)
        self.process_factory, self.port_source = process_factory, port_source
        self.camera_latest = self.camera_summary = self.camera_error = None
        self.camera_frames = 0
        self.current = self.stop_file = None
        self.camera_port = None
        self.lock = threading.RLock()
        self.phase, self.step = "idle", 0
        self.output = self.latest = self.summary = self.error = None
        self.frames = self.total_frames = self.attempt = 0
        self.deadline = 0
        self.stop_state = {}
        self.worker = None
        self.stopping = self.closing = False
        self.results = []
        self.selected_port = None
        self.port_cache, self.port_cache_at = [], 0
        self.preview_process = self.preview_log = self.preview_index = self.preview_stop = None
        self.preview_id = None
        self.preview_mode = False
        self.camera_url = camera_url

    def preview(self, payload):
        with self.lock:
            if self.phase == "recording" or self.closing or (self.worker and self.worker.is_alive()):
                raise ValueError("正式录制正在进行")
            if self.preview_process and self.preview_process.poll() is None:
                return self.state()
            port = payload.get("camera_port")
            if not self.camera_url and port not in {p["port"] for p in self.available_ports()}:
                raise ValueError("请选择相机端口")
            tof_port = payload.get("port")
            if (not self.camera_url and tof_port == port) or tof_port not in {p["port"] for p in self.available_ports()}:
                raise ValueError("请选择不同的 ToF 端口")
            if self.preview_log:
                self.preview_log.close()
            self.preview_id = uuid.uuid4().hex[:8]
            root = self.data_root.parent / "previews" / ("preview-" + self.preview_id)
            root.mkdir(parents=True, exist_ok=False)
            self.preview_stop = root / "stop"
            self.preview_index = RunIndex(root)
            self.preview_log = (root / "console.log").open("xb")
            command = [sys.executable, "-B", str(HERE / "pair_capture.py"), "--camera-port", port,
                       "--tof-port", tof_port, "--seconds", "300", "--output", str(root / "data"),
                       "--stop-file", str(self.preview_stop), "--label", "PREVIEW_ONLY_NOT_GUIDED_COLLECTION"]
            if self.camera_url:
                command += ["--camera-url", self.camera_url]
            self.preview_index = RunIndex(root / "data")
            try:
                self.preview_process = self.process_factory(command, stdout=self.preview_log,
                    stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except Exception:
                self.preview_log.close()
                raise
            self.preview_mode = True
            self.camera_latest, self.camera_error, self.camera_frames = None, None, 0
        return self.state()

    def stop_preview(self):
        if self.preview_process and self.preview_process.poll() is None:
            self.preview_stop.touch(exist_ok=True)
            self.preview_process.wait(timeout=8)
        if self.preview_log:
            self.preview_log.close()
            self.preview_log = None
        self.preview_mode = False

    def available_ports(self):
        # Enumerate only; do not open/reset a device during page load.
        if time.monotonic() - self.port_cache_at > 3:
            self.port_cache = self.port_source()
            self.port_cache_at = time.monotonic()
        return self.port_cache

    def state(self):
        available = self.available_ports()
        with self.lock:
            if self.preview_mode:
                self.preview_index.refresh()
                rows = self.preview_index.rows["camera"]
                self.camera_latest = rows[-1] if rows else None
                self.camera_frames = len(rows)
                tof_rows = self.preview_index.rows["tof"]
                self.latest = tof_rows[-1] if tof_rows else None
                self.frames = len(tof_rows)
                summary_path = self.preview_index.root / "camera/summary.json"
                if summary_path.exists():
                    try:
                        self.camera_summary = json.loads(summary_path.read_text())
                    except ValueError:
                        pass  # Collector may still be writing its final receipt.
                    if self.camera_summary and not self.camera_summary.get("frames"):
                        self.camera_error = "USB 未收到相机画面；若仍是 Wi-Fi 固件，请开启原手机热点并让电脑连接同一网络"
                if self.preview_process.poll() is not None:
                    if not rows and not self.camera_error:
                        self.camera_error = "预览未收到画面，请检查相机连接；日志已保留"
                    if self.preview_log:
                        self.preview_log.close()
                        self.preview_log = None
            stamp = self.latest.get("host_received_monotonic_ns") if self.latest else None
            return {"phase": self.phase, "step": self.step, "segments": SEGMENTS,
                    "remaining_s": max(0, self.deadline-time.monotonic()) if self.phase == "recording" else 0,
                    "frames": self.frames, "total_frames": self.total_frames,
                    "latest": self.latest, "age_ms": (time.monotonic_ns()-stamp)/1e6 if stamp else None,
                    "summary": self.summary, "error": self.error,
                    "output": str(self.output) if self.output else None,
                    "ports": available, "issues": self.summary.get("issues", {}) if self.summary else {},
                    "results": list(self.results), "stopping": self.stopping,
                    "selected_port": self.selected_port, "camera_port": self.camera_port,
                    "camera_transport_url": self.camera_url,
                    "suggested_tof": next((p["port"] for p in available if str(p.get("serial_number", "")).replace(":", "").upper() == "98A316F7881C"), None),
                    "suggested_camera": next((p["port"] for p in available if str(p.get("serial_number", "")).replace(":", "").upper() == "B43A45BD12D8"), None),
                    "camera_latest": self.camera_latest, "camera_frames": self.camera_frames,
                    "camera_error": self.camera_error, "camera_summary": self.camera_summary,
                    "camera_age_ms": ((time.monotonic_ns()-self.camera_latest["host_received_monotonic_ns"])/1e6
                                      if self.camera_latest else None),
                    "preview_active": bool(self.preview_mode and self.preview_process.poll() is None),
                    "preview_mode": self.preview_mode,
                    "preview_output": str(self.preview_index.root) if self.preview_index else None,
                    "camera_url": (f'/api/image/{"preview-"+self.preview_id if self.preview_mode else self.attempt}/{self.camera_latest["filename"]}'
                                   if self.camera_latest else None)}

    def start(self, payload):
        self.stop_preview()
        with self.lock:
            if self.closing or self.stopping or self.phase in ("recording", "complete") or (self.worker and self.worker.is_alive()):
                raise ValueError("当前不能开始；录制时请等待本段结束")
            port = payload.get("port")
            camera_port = payload.get("camera_port")
            if not isinstance(port, str) or port not in {p["port"] for p in self.available_ports()}:
                raise ValueError("请选择当前列表中的 XIAO 串口")
            if not self.camera_url and (camera_port == port or camera_port not in {p["port"] for p in self.available_ports()}):
                raise ValueError("请选择与 ToF 不同的相机串口")
            if self.selected_port and (self.selected_port != port or self.camera_port != camera_port):
                raise ValueError("同一轮请使用同一设备；设备重连后请重启本页服务另建一轮")
            if self.output is None:
                object_name = str(payload.get("object", "未填写"))[:200]
                moved = str(payload.get("moved", "未知"))[:200]
                self.output = self.data_root / ("simple-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:6])
                self.output.mkdir(parents=True, exist_ok=False)
                write_json(self.output / "session.json", {
                    "schema": "cnh.simple-guide.v1", "segments": SEGMENTS,
                    "port": port, "camera_port": camera_port, "camera_url": self.camera_url,
                    "object": object_name, "sensor_movement_operator_note": moved,
                    "scope": "operator-guided exploratory capture; no measured distance or detection claims"})
                (self.output / "notes.md").write_text(
                    f"# 简单采集备注\n\n物体：{object_name}\n\n传感器是否移动：{moved}\n\n距离：未知\n\n"
                    "请在结束后补记实际动作、漏做步骤和传感器是否被碰动。分段提示不证明动作已完成。\n", encoding="utf-8")
            self.selected_port, self.camera_port = port, camera_port
            self.attempt += 1
            segment = SEGMENTS[self.step]
            target = self.output / f'{self.step+1:02d}-{segment["id"]}-attempt{self.attempt}'
            self.phase, self.error, self.summary = "recording", None, None
            self.latest, self.frames = None, 0
            self.stop_state = {}
            self.camera_latest = self.camera_summary = self.camera_error = None
            self.camera_frames = 0
            self.current = RunIndex(target)
            self.stop_file = self.output / f"attempt{self.attempt}.stop"
            self.deadline = time.monotonic() + segment["seconds"]
            self.worker = threading.Thread(target=self.record, args=(target, segment), daemon=False)
            self.worker.start()
        return self.state()

    def record(self, target, segment):
        command = [sys.executable, "-B", str(HERE / "pair_capture.py"),
                   "--camera-port", self.camera_port, "--tof-port", self.selected_port,
                   "--seconds", str(segment["seconds"]), "--output", str(target),
                   "--stop-file", str(self.stop_file), "--label", segment["id"]]
        if self.camera_url:
            command += ["--camera-url", self.camera_url]
        process = None
        failure = None
        try:
            with (self.output / f"attempt{self.attempt}-console.log").open("xb") as log:
                process = self.process_factory(command, stdout=log, stderr=subprocess.STDOUT,
                                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                started = time.monotonic()
                while True:
                    finished = process.poll() is not None
                    with self.lock:
                        self.current.refresh()
                        tof, camera = self.current.rows["tof"], self.current.rows["camera"]
                        self.frames, self.camera_frames = len(tof), len(camera)
                        self.latest = tof[-1] if tof else None
                        self.camera_latest = camera[-1] if camera else None
                        if self.latest and (self.latest["sensor"].get("type") != "cnh_frame" or
                                            self.latest["derived"].get("rows") != 8 or
                                            self.latest["sensor"].get("bins") != 16):
                            failure = "ToF 不是 8×8×16 CNH；请核对设备，已有记录保留"
                        if time.monotonic()-started > 8 and (not tof or not camera):
                            failure = failure or "8 秒内未收到两路有效数据；请核对端口与设备"
                        if failure:
                            self.stop_file.touch(exist_ok=True)
                            self.error = failure
                    if finished:
                        break
                    # Collector has its own bounded cleanup deadline; let it finalize evidence.
                    time.sleep(0.1)
                pair_path = target / "summary.json"
                pair = json.loads(pair_path.read_text()) if pair_path.exists() else {}
                streams = pair.get("streams", {})
                summary, camera_summary = streams.get("tof") or {}, streams.get("camera") or {}
                with self.lock:
                    self.summary, self.camera_summary = summary, camera_summary
                    self.total_frames += summary.get("frames", 0)
                    self.results.append({"step": self.step, "label": segment["label"],
                                         "directory": target.name, "summary": pair})
                    if failure or not summary.get("frames") or not camera_summary.get("frames") or summary.get("acquisition_error"):
                        self.phase = "error"
                        self.error = failure or summary.get("acquisition_error") or "一路未成功录制；日志已保存，请核对连接后重试本段"
                    elif self.stopping:
                        self.phase = "complete"
                    else:
                        self.step += 1
                        self.phase = "complete" if self.step == len(SEGMENTS) else "ready"
                    if camera_summary.get("failure_count"):
                        self.camera_error = "相机存在采集问题，原始日志已保留"
                    self.save_progress()
        except Exception as exc:
            with self.lock:
                self.phase, self.error = "error", str(exc)
        finally:
            if process and process.poll() is None:
                self.stop_file.touch(exist_ok=True)
                process.wait(timeout=segment["seconds"]+20)

    def image(self, attempt, filename):
        with self.lock:
            if self.preview_index and attempt == "preview-"+self.preview_id:
                return self.preview_index.image_path(filename).read_bytes()
            if attempt != str(self.attempt) or self.current is None:
                raise ValueError("图片不属于当前分段")
            return self.current.image_path(filename).read_bytes()

    def save_progress(self):
        write_json(self.output / "guide-summary.json", {
            "completed_steps": self.step, "planned_steps": len(SEGMENTS),
            "ended_early": self.stopping, "phase": self.phase, "error": self.error,
            "total_parsed_tof_frames": self.total_frames, "attempts": self.results})

    def stop(self):
        with self.lock:
            self.stopping = True
            self.stop_state["stopped_by_request"] = True
            if self.stop_file:
                self.stop_file.touch(exist_ok=True)
            if self.phase != "recording":
                self.phase = "complete"
                if self.output:
                    self.save_progress()
        return self.state()

    def close(self):
        self.closing = True
        self.stop_preview()
        if self.phase != "complete":
            self.stop()
        if self.worker:
            self.worker.join(timeout=80)
            if self.worker.is_alive():
                raise RuntimeError("采集仍在收尾，请保留进程和原始文件")


class Handler(LocalHandler):
    def do_GET(self):
        try:
            self.local_request()
            if self.path == "/":
                self.send(200, (HERE / "simple_capture.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path.startswith("/api/image/"):
                attempt, filename = self.path[len("/api/image/"):].split("/", 1)
                self.send(200, self.server.guide.image(attempt, filename), "image/jpeg")
            elif self.path == "/api/state":
                self.send(200, self.server.guide.state())
            else:
                self.send(404, {"error": "未找到"})
        except (ValueError, OSError, ImportError) as exc:
            self.send(400, {"error": str(exc)})

    def do_POST(self):
        try:
            self.local_request()
            if self.headers.get_content_type() != "application/json":
                raise ValueError("仅接受 JSON")
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 4096:
                raise ValueError("请求过大或为空")
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError("请求必须为对象")
            guide = self.server.guide
            if self.path == "/api/start":
                result = guide.start(payload)
            elif self.path == "/api/preview":
                result = guide.preview(payload)
            elif self.path == "/api/stop":
                result = guide.stop()
            elif self.path == "/api/shutdown":
                guide.closing = True
                result = guide.stop()
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else:
                self.send(404, {"error": "未找到"})
                return
            self.send(200, result)
        except (ValueError, OSError, ImportError) as exc:
            self.send(400, {"error": str(exc)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--camera-url", help="local Wi-Fi camera MJPEG URL")
    args = parser.parse_args()
    if args.camera_url:
        from wifi_camera_capture import validate_url
        args.camera_url = validate_url(args.camera_url)
    guide = Guide(ARTIFACTS / "captures", camera_url=args.camera_url)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.guide = guide
    print(f"http://127.0.0.1:{server.server_port}/ — click Start to open the sensor", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        guide.close()
        server.server_close()


if __name__ == "__main__":
    main()
