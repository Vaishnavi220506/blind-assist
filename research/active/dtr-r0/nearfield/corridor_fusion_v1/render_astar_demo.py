"""Render all sealed A / independently replayed A* frames; never run a model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import cv2
from PIL import Image, ImageDraw, ImageFont

FAMILIES = {"suspended_head": "HEAD · 悬空障碍", "substantial_body": "BODY · 实体障碍",
            "near_rod_farwall": "ROD · 近细杆 / 远墙", "shallow_boundary_stress": "BOUNDARY · 浅边界"}
W, H, FPS = 1440, 900, 4


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def outcome(alert, truth):
    return ("TP" if truth else "FP") if alert else ("FN" if truth else "TN")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    p.add_argument("--font", type=Path, required=True)
    args = p.parse_args()
    if not args.ffmpeg:
        p.error("--ffmpeg is required when ffmpeg is not on PATH")
    capture = args.source / "source/returned-v1/capture-v1"
    old_path = args.source / "confirmation/predictions.json"
    cases_path = args.source / "confirmation/cases.json"
    old = load(old_path)
    cases = {r["id"]: r for r in load(cases_path)}
    new = [json.loads(s) for s in args.predictions.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    raw = [json.loads(s) for s in (capture / "raw.jsonl").read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    assert len(new) == len(old) == len(raw) == 288
    assert [r["id"] for r in new] == [r["id"] for r in old] == [r["id"] for r in raw]
    assert len({r["id"] for r in new}) == 288
    assert len({r["threshold"] for r in new}) == len({r["model_id"] for r in new}) == 1
    args.out.mkdir(parents=True, exist_ok=True)
    fonts = {s: ImageFont.truetype(str(args.font), s) for s in [18, 20, 22, 26, 30, 38]}
    episodes, manifest = [], []
    for i, (a, b, r) in enumerate(zip(old, new, raw)):
        assert isinstance(a["A"], bool) and isinstance(b["alert"], bool)
        assert b["episode_id"] == a["episode_id"] == r["episode_id"]
        assert b["time_s"] == a["time_s"] == r["time_s"]
        c = cases[r["id"]]
        if not episodes or episodes[-1]["id"] != b["episode_id"]:
            episodes.append({"id": b["episode_id"], "family": c["family"], "start": i / FPS,
                             "label": b["episode_id"].replace("singleconfirm_", ""), "frame_count": 0})
        episodes[-1]["frame_count"] += 1
        manifest.append({"video_frame": i, "video_time_s": i / FPS, "id": b["id"],
                         "episode_id": b["episode_id"], "episode_index": len(episodes)-1,
                         "source_time_s": b["time_s"], "family": c["family"],
                         "A_alert": a["A"], "A_score": a["A_score"], "Astar_alert": b["alert"],
                         "Astar_score": b["score"], "Astar_threshold": b["threshold"],
                         "Astar_model_id": b["model_id"], "truth_annotation_only": c["truth"],
                         "Astar_reminder_onset": b["alert"] and (i == 0 or new[i-1]["episode_id"] != b["episode_id"] or not new[i-1]["alert"]),
                         "stratum": "boundary" if c["stratum"] == "boundary" else "clear",
                         "A_outcome": outcome(a["A"], c["truth"]),
                         "Astar_outcome": outcome(b["alert"], c["truth"]),
                         "rgb_path": r["rgb_path"], "tof_packet_received": b["tof_packet_received"]})
    assert len(episodes) == 48 and all(e["frame_count"] == 6 for e in episodes)
    assert set(e["family"] for e in episodes) == set(FAMILIES)
    reminders_path = args.predictions.with_name("reminders.jsonl")
    if reminders_path.exists():
        reminders = [json.loads(s) for s in reminders_path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
        assert len(reminders) == len(manifest)
        assert all(r["id"] == m["id"] and r["alert"] == m["Astar_alert"] and r["reminder_onset"] == m["Astar_reminder_onset"] for r, m in zip(reminders, manifest))
    (args.out / "frame-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "episodes.json").write_text(json.dumps(episodes, ensure_ascii=False, indent=2), encoding="utf-8")

    def render(m):
        canvas = Image.new("RGB", (W, H), "#101824")
        d = ImageDraw.Draw(canvas)
        def t(x, y, text, size=22, color="#dce8f3"):
            d.text((x, y), text, font=fonts[size], fill=color)
        t(40, 25, "BlindAssist  /  A 与 A* 同步回放", 38)
        t(40, 82, FAMILIES[m["family"]] + f"    片段 {m['episode_index']+1:02d}/48    帧 {m['video_frame']+1:03d}/288", 26)
        t(40, 127, m["episode_id"].replace("singleconfirm_", ""), 20, "#91a7bc")
        t(760, 123, m["Astar_model_id"], 18, "#91a7bc")
        t(760, 149, f"固定阈值 {m['Astar_threshold']}", 18, "#91a7bc")
        rgb = Image.open(capture / m["rgb_path"]).convert("RGB")
        assert rgb.size == (640, 360)
        for x, name, active, score, result in [(40, "A · 原冻结基线", m["A_alert"], m["A_score"], m["A_outcome"]),
                                               (760, "A* · 独立重训模型", m["Astar_alert"], m["Astar_score"], m["Astar_outcome"])]:
            d.rounded_rectangle((x-12, 177, x+652, 746), radius=16, fill="#1a2838")
            t(x, 189, name, 30)
            col = "#ef8354" if active else "#529d9b"
            d.rounded_rectangle((x+390, 186, x+640, 239), radius=10, fill=col)
            t(x+410, 195, "提醒：有障碍" if active else "当前未提醒", 26, "#101824")
            canvas.paste(rgb, (x, 259))
            t(x, 637, f"保存分数 {score:.6f}", 22)
            t(x, 676, f"离线评分  {result}    ·    {m['stratum']}", 22, "#a8b8c8")
            t(x, 712, "预测状态来自保存输出；GT 仅用于评分", 18, "#91a7bc")
        t(40, 768, f"片段内 {m['source_time_s']:.2f} s    |    4 Hz / 总长 72 s    |    GT：" + ("通道内障碍" if m["truth_annotation_only"] else "通道外") + "（旁注）", 22)
        for j in range(6):
            d.ellipse((1150+j*35, 775, 1166+j*35, 791), fill="#ef8354" if j == m["video_frame"] % 6 else "#43566c")
        t(40, 820, "完整 48 片段，含漏报 / 误报；核心事件片段左截断，首次提醒仅相对片段起点。", 20, "#91a7bc")
        t(40, 852, "同模拟器既有数据回放 · 非连续行走或实机安全证据 · 视频帧率不代表模型运行速度", 18, "#91a7bc")
        return canvas

    video = args.out / "A-vs-Astar-all48.mp4"
    cmd = [args.ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "19",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)]
    with (args.out / "encode.log").open("wb") as log:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=log)
        try:
            representatives = set()
            for m in manifest:
                im = render(m)
                proc.stdin.write(im.tobytes())
                if m["video_frame"] == 0:
                    im.save(args.out / "poster.png")
                if m["family"] not in representatives:
                    im.save(args.out / (m["family"] + ".png"))
                    representatives.add(m["family"])
        finally:
            proc.stdin.close()
            code = proc.wait()
        assert code == 0, f"ffmpeg failed: {code}"
    cap = cv2.VideoCapture(str(video))
    decoded = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame.shape[:2] == (H, W)
        decoded += 1
    rate = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    assert decoded == 288 and rate == FPS
    ffprobe = str(Path(args.ffmpeg).with_name("ffprobe" + Path(args.ffmpeg).suffix))
    probe = json.loads(subprocess.check_output([ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video)]))
    assert abs(float(probe["format"]["duration"])-72) < .001
    write_html(args.out, manifest, episodes)
    receipt = {"status": "PASS", "decoded_frames": decoded, "width": W, "height": H, "fps": rate,
               "duration_s": 72, "episodes": len(episodes), "all_source_ids_in_original_order": True,
               "A_source_field": "A", "Astar_source_field": "alert", "audio_source_field": "Astar_reminder_onset, derived only from new alert; episode reset",
               "new_prediction_sha256": digest(args.predictions), "old_prediction_sha256": digest(old_path),
               "cases_sha256": digest(cases_path), "video_sha256": digest(video), "ffprobe": probe,
               "compute": "TASK_NOT_GPU_SUITABLE: CPU Pillow layout and software H264 encoding; no inference"}
    (args.out / "verification.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "README.md").write_text("# A / A* 全片段同步演示\n\n打开 index.html 或 A-vs-Astar-all48.mp4。48 完整片段，288 帧，4 Hz，72 秒。\n\n"
        "左侧只读原 sealed `A`；右侧徽标、逐帧记录与可选声音只读新独立推理 `alert`。不重算阈值、不推理、不训练。"
        "GT 仅在预测输出读取后用于旁注与 TP/FP/FN/TN 评分。包含全部 clear 与 boundary 成败案例。\n\n"
        "播放器默认静音；开启提示音后，仅当前 A* alert 上升沿（每 episode 重置）且播放时发声；暂停或跳转停止当前声音。MP4 本身无音轨。"
        "浏览器调度可能限制高倍速声音，视觉状态始终以视频为准。\n\n"
        "verification.json 保存 288 帧完整解码、72 秒、分辨率及文件哈希检查；frame-manifest.json 是每一视频帧的来源。"
        "视频制作速度不代表模型推理速度；核心事件片段有左截断，同模拟器回放不等于实机安全证据。\n", encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ["status", "decoded_frames", "episodes", "duration_s"]}))


def write_html(out, frames, episodes):
    data = json.dumps({"frames": frames, "episodes": episodes, "families": FAMILIES}, ensure_ascii=False).replace("</", "<\\/")
    page = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BlindAssist · A / A* 完整同步回放</title><style>
*{box-sizing:border-box}body{margin:0;background:#101824;color:#dce8f3;font:16px system-ui,"Microsoft YaHei",sans-serif}main{max-width:1480px;margin:auto;padding:24px}h1{font-size:28px;margin:0 0 10px}p{color:#a8b8c8;line-height:1.6}video{width:100%;max-height:72vh;background:#000;border-radius:12px}button,select,a{font:inherit}button,select{background:#23364b;color:#dce8f3;border:1px solid #526a82;border-radius:8px;padding:9px 13px;cursor:pointer}.bar{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:14px 0}.status{background:#1a2838;padding:14px;border-radius:8px}#badge{font-weight:bold;color:#ef8354}#segments{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:8px;margin:16px 0}#segments button{text-align:left;font-size:13px}#segments button.active{border-color:#ef8354;background:#423129}a{color:#8dcbff}.small{font-size:13px}</style>
<main><h1>BlindAssist · A / A* 完整同步回放</h1><p>既有 288 帧 · 48 完整片段 · 4 类场景 · 72 秒。包含误报与漏报；右侧只展示新独立 A* 保存输出。</p>
<video id="video" controls playsinline preload="metadata" poster="poster.png"><source src="A-vs-Astar-all48.mp4" type="video/mp4"></video>
<div class="bar"><button id="play">播放 / 暂停</button><label>速度 <select id="speed"><option value=".5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label><label><input id="sound" type="checkbox"> 开启 A* 提示音（默认关闭）</label><label>类别 <select id="family"><option value="all">全部 48 片段</option></select></label></div>
<div class="status"><span id="position"></span>　<span id="badge"></span><div class="small" id="detail"></div></div><div id="segments"></div>
<p class="small">GT / TP / FP / FN 为离线旁注，不驱动预测。核心事件片段左截断，首次提醒仅相对片段起点；同模拟器回放，不是实机或安全效果证据。播放帧率不是模型运行速度。提示音仅反映当前 A* alert，高倍速可能受浏览器调度限制。</p>
<p class="small"><a href="A-vs-Astar-all48.mp4" download>下载视频</a> · <a href="frame-manifest.json">逐帧记录</a> · <a href="verification.json">验证记录</a></p></main><script>
const DATA=__DATA__;const video=document.querySelector('#video'),sound=document.querySelector('#sound');let audio=null,osc=null,last=-1;
function stopTone(){if(osc){try{osc.stop()}catch(e){}osc=null}}
function beep(){stopTone();if(!sound.checked||video.paused||video.seeking)return;if(!audio)audio=new(window.AudioContext||window.webkitAudioContext)();audio.resume();osc=audio.createOscillator();const gain=audio.createGain();osc.frequency.value=660;gain.gain.setValueAtTime(.035,audio.currentTime);gain.gain.exponentialRampToValueAtTime(.001,audio.currentTime+.10);osc.connect(gain).connect(audio.destination);osc.start();osc.stop(audio.currentTime+.11)}
function show(){const i=Math.min(DATA.frames.length-1,Math.max(0,Math.floor(video.currentTime*4+1e-5))),f=DATA.frames[i];document.querySelector('#position').textContent=`片段 ${f.episode_index+1}/48 · 帧 ${i+1}/288 · ${f.source_time_s.toFixed(2)} s`;document.querySelector('#badge').textContent=f.Astar_alert?'A* 提醒：有障碍':'A* 当前未提醒';document.querySelector('#badge').style.color=f.Astar_alert?'#ef8354':'#7bc7c0';document.querySelector('#detail').textContent=`${f.id} · 保存分数 ${f.Astar_score.toFixed(6)} · 固定阈值 ${f.Astar_threshold} · ${f.stratum} · 离线评分 ${f.Astar_outcome}`;document.querySelectorAll('#segments button').forEach(b=>b.classList.toggle('active',+b.dataset.episode===f.episode_index));if(i!==last){if(f.Astar_reminder_onset)beep();else stopTone();last=i}if(!video.paused)requestAnimationFrame(show)}
function list(){const fam=document.querySelector('#family').value;document.querySelector('#segments').replaceChildren();DATA.episodes.forEach((e,i)=>{if(fam!=='all'&&e.family!==fam)return;const b=document.createElement('button');b.textContent=`${String(i+1).padStart(2,'0')} · ${e.label}`;b.dataset.episode=i;b.onclick=()=>{stopTone();video.currentTime=e.start;last=-1;show()};document.querySelector('#segments').append(b)});show()}
Object.entries(DATA.families).forEach(([v,n])=>{const o=document.createElement('option');o.value=v;o.textContent=n;document.querySelector('#family').append(o)});
document.querySelector('#family').onchange=list;document.querySelector('#play').onclick=()=>video.paused?video.play():video.pause();document.querySelector('#speed').onchange=e=>video.playbackRate=+e.target.value;sound.onchange=()=>{stopTone();if(sound.checked){if(!audio)audio=new(window.AudioContext||window.webkitAudioContext)();audio.resume();last=-1;show()}};video.onplay=()=>{last=-1;show()};video.onpause=()=>{stopTone();show()};video.onseeking=stopTone;video.onseeked=()=>{last=-1;show()};video.ontimeupdate=()=>{if(video.paused)show()};video.onended=stopTone;list();
</script></html>'''
    (out / "index.html").write_text(page.replace("__DATA__", data), encoding="utf-8")


if __name__ == "__main__":
    main()
