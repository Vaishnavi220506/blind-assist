"""Operator cue schedule; cues are not evidence that an action was performed."""
import math

PHASES = [
    (0, 6, "prepare", "准备：移开书，设备保持固定"),
    (6, 10, "baseline", "背景：保持不动，书在画面外"),
    (10, 16, "left", "← 左：书放在画面左侧靠近中央，停住"),
    (16, 19, "clear", "移开书，恢复背景"),
    (19, 25, "right", "右 →：书放在画面右侧靠近中央，停住"),
    (25, 28, "clear", "移开书，恢复背景"),
    (28, 34, "up", "↑ 上：书放在画面上侧靠近中央，停住"),
    (34, 37, "clear", "移开书，恢复背景"),
    (37, 43, "down", "↓ 下：书放在画面下侧靠近中央，停住"),
    (43, 50, "finish", "动作完成：移开书，等待保存"),
]


def plan(start_ns):
    return {"schema": "hardware-bringup.orientation-cues.v1", "start_host_monotonic_ns": start_ns,
            "seconds": 50, "phases": [{"start_s": a, "end_s": b, "phase": name, "instruction": text}
                                      for a, b, name, text in PHASES],
            "scope": "Scheduled operator cues only, not action timestamps or calibration truth; confirm actual book position from recorded RGB"}


def cue(schedule, now_ns, recording_active):
    elapsed = max(0, (now_ns-schedule["start_host_monotonic_ns"])/1e9)
    if not recording_active:
        return {"phase": "stopped", "instruction": "录制已结束；请移开书，等待检查", "remaining_s": 0,
                "active": False, "elapsed_s": elapsed}
    for phase in schedule["phases"]:
        if phase["start_s"] <= elapsed < phase["end_s"]:
            return {**phase, "remaining_s": math.ceil(phase["end_s"]-elapsed), "active": True, "elapsed_s": elapsed}
    return {"phase": "saving", "instruction": "动作完成，正在保存；无需再移动", "remaining_s": 0,
            "active": False, "elapsed_s": elapsed}
